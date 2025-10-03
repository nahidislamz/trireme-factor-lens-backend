from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import os

# Import your existing modules
from Factor_Lens.config import LensConfig
from Factor_Lens.crypto_factor_lens import CryptoFactorLens
from Factor_Lens.crypto_factor_lens.regime_gmm import RegimeGMM

app = FastAPI(title="Trireme Factor Lens API", version="1.0.0")

# --------------------------
# CORS Configuration
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trireme-factor-lens.web.app",
        "https://factor-lens.web.app",
        "https://www.trireme.com",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Global state
# --------------------------
class AppState:
    def __init__(self):
        self.lens: Optional[CryptoFactorLens] = None
        self.panel: Optional[pd.DataFrame] = None
        self.gmm_cache: Dict[int, dict] = {}
        self.cfg: Optional[LensConfig] = None

state = AppState()

# --------------------------
# Response Models
# --------------------------
class FactorDataResponse(BaseModel):
    dates: List[str]
    columns: List[str]
    data: List[List[float]]
    btc_column: str

class CorrelationResponse(BaseModel):
    factors: List[str]
    correlation_matrix: List[List[float]]

class RegimeDataResponse(BaseModel):
    dates: List[str]
    btc_price: List[float]
    regime_labels: List[int]
    regime_probabilities: Dict[str, List[float]]
    relaxed_labels: Optional[List[int]] = None
    n_regimes: int

class StatusResponse(BaseModel):
    status: str
    message: str
    lens_initialized: bool
    available_gmm_k: List[int]

# --------------------------
# Helper Functions
# --------------------------
def serialize_dataframe(df: pd.DataFrame) -> dict:
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        dates = df_copy.index.strftime('%Y-%m-%d').tolist()
    else:
        dates = df_copy.index.astype(str).tolist()
    df_copy = df_copy.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {
        "dates": dates,
        "columns": df_copy.columns.tolist(),
        "data": df_copy.values.tolist(),
    }

def calculate_relaxed_labels(labels: np.ndarray, window: int = 7) -> np.ndarray:
    relaxed = pd.Series(labels).rolling(window=window, center=True, min_periods=1).apply(
        lambda x: pd.Series(x).mode()[0] if len(x) > 0 else x.iloc[0],
        raw=False,
    )
    return relaxed.astype(int).values

# --------------------------
# Startup & Initialization
# --------------------------
@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing Trireme Factor Lens...")
        state.cfg = LensConfig()
        panel_path = Path(state.cfg.merged_csv)

        if not panel_path.exists():
            print(f"Panel file not found at {panel_path}")
            return

        df = pd.read_csv(panel_path)
        if "date" not in df.columns or "symbol" not in df.columns:
            print("CSV must have 'date' and 'symbol' columns")
            return

        df["date"] = pd.to_datetime(df["date"])

        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=["date", "symbol"], keep="last")

        # Set MultiIndex
        df.set_index(["date", "symbol"], inplace=True)
        state.panel = df

        # Initialize lens
        state.lens = CryptoFactorLens(state.cfg)
        state.lens.load_from_panel(state.panel)
        state.lens.build_factors()
        print("✅ Lens initialized successfully!")

    except Exception as e:
        print(f"Error during startup: {e}")
        traceback.print_exc()

# --------------------------
# API Endpoints
# --------------------------
@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="online",
        message="Trireme Factor Lens API",
        lens_initialized=state.lens is not None,
        available_gmm_k=list(state.gmm_cache.keys()),
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        status="healthy" if state.lens else "not_initialized",
        message="Lens ready" if state.lens else "Please run data downloader first",
        lens_initialized=state.lens is not None,
        available_gmm_k=list(state.gmm_cache.keys()),
    )

@app.get("/api/factors")
async def get_factors():
    if state.lens is None:
        raise HTTPException(status_code=503, detail="Lens not initialized.")

    factors_df = state.lens.factors
    if factors_df is None:
        if state.panel is not None:
            state.lens.build_factors()
            factors_df = state.lens.factors
        else:
            raise HTTPException(status_code=503, detail="No panel data available.")

    if "BTC" in state.lens.close_wide.columns:
        factors_df["BTC_price"] = state.lens.close_wide["BTC"]
    else:
        raise HTTPException(status_code=404, detail="BTC price not found")

    serialized = serialize_dataframe(factors_df)
    return FactorDataResponse(
        dates=serialized["dates"],
        columns=serialized["columns"],
        data=serialized["data"],
        btc_column="BTC_price",
    )

@app.get("/api/correlation")
async def get_correlation():
    if state.lens is None:
        raise HTTPException(status_code=503, detail="Lens not initialized")

    factors_df = state.lens.factors
    corr_matrix = factors_df.corr().fillna(0)
    return CorrelationResponse(
        factors=corr_matrix.columns.tolist(),
        correlation_matrix=corr_matrix.values.tolist(),
    )

@app.get("/api/regimes/{k}")
async def get_regimes(k: int):
    if state.lens is None:
        raise HTTPException(status_code=503, detail="Lens not initialized")
    if k < 2 or k > 10:
        raise HTTPException(status_code=400, detail="K must be between 2 and 10")

    if k in state.gmm_cache:
        return state.gmm_cache[k]

    try:
        btc = state.lens.close_wide["BTC"]
        factors_df = state.lens.factors

        # Fit GMM
        gmm = RegimeGMM(n_regimes=k, random_state=42, scale=True).fit(factors_df)
        labels = gmm.labels_

        # --- probabilities ---
        if hasattr(gmm, "probs_"):
            probs = gmm.probs_
            probabilities = probs.to_numpy() if isinstance(probs, pd.DataFrame) else np.array(probs)
        elif hasattr(gmm, "predict_proba"):
            probabilities = gmm.predict_proba(factors_df)
        elif hasattr(gmm, "probabilities_"):
            probabilities = gmm.probabilities_
        else:
            probabilities = np.zeros((len(labels), k))
            for i, label in enumerate(labels):
                probabilities[i, label] = 1

        # --- outputs ---
        relaxed_labels = calculate_relaxed_labels(labels)
        btc_aligned = btc.reindex(factors_df.index)

        regime_probs = {
            f"regime_{i}": probabilities[:, i].tolist()
            for i in range(probabilities.shape[1])
        }

        response = RegimeDataResponse(
            dates=factors_df.index.strftime("%Y-%m-%d").tolist(),
            btc_price=btc_aligned.replace({np.nan: None}).tolist(),
            regime_labels=labels.tolist(),
            regime_probabilities=regime_probs,
            relaxed_labels=relaxed_labels.tolist(),
            n_regimes=k,
        )

        state.gmm_cache[k] = response
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GMM error: {str(e)}")


@app.get("/api/regimes/rebuild/{k}")
async def rebuild_regimes(k: int, background_tasks: BackgroundTasks):
    if state.lens is None:
        raise HTTPException(status_code=503, detail="Lens not initialized")
    if k < 2 or k > 10:
        raise HTTPException(status_code=400, detail="K must be between 2 and 10")

    if k in state.gmm_cache:
        del state.gmm_cache[k]

    result = await get_regimes(k)
    return {"status": "success", "message": f"GMM rebuilt for K={k}", "data": result}

@app.delete("/api/cache/clear")
async def clear_cache():
    state.gmm_cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/api/available-factors")
async def get_available_factors():
    if state.lens is None:
        raise HTTPException(status_code=503, detail="Lens not initialized")

    return {
        "factors": state.lens.factors.columns.tolist(),
        "date_range": {
            "start": state.lens.factors.index.min().strftime("%Y-%m-%d"),
            "end": state.lens.factors.index.max().strftime("%Y-%m-%d"),
        },
        "n_observations": len(state.lens.factors),
    }

# --------------------------
# Entry point for Cloud Run / Local
# --------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("Factor_Lens.api:app", host="0.0.0.0", port=port, reload=False)
