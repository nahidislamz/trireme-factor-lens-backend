# crypto_factor_lens/dataloader.py
from __future__ import annotations
import os
import pandas as pd
from typing import Optional
from Factor_Lens.config import LensConfig

class DataLoader:
    """
    Loads the merged_sources.csv into a single MultiIndex panel (date, symbol).
    """

    def __init__(self, cfg: LensConfig, data_path: Optional[str] = None):
        self.cfg = cfg
        self.data_path = data_path or cfg.merged_csv
        self._panel: Optional[pd.DataFrame] = None

    def load_panel(self) -> pd.DataFrame:
        """Read merged_sources.csv and return MultiIndex (date, symbol) panel."""
        if self._panel is not None:
            return self._panel

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"merged CSV not found at: {self.data_path}")

        df = pd.read_csv(self.data_path)
        if "date" not in df.columns or "symbol" not in df.columns:
            raise ValueError("merged CSV must contain 'date' and 'symbol' columns")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "symbol"])
        df["symbol"] = df["symbol"].astype(str)

        # Standardize types where possible
        # (Leave non-numeric as-is; factors/filters will coerce safely later)
        for c in df.columns:
            if c in ("date", "symbol"): 
                continue
            # coerce to numeric when sensible; leave strings (e.g., cg_id_cg, pair_binance)
            if df[c].dtype == "object":
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except Exception:
                    pass

        panel = df.set_index(["date", "symbol"]).sort_index()

        # Ensure no duplicate (date, symbol) pairs
        panel = panel.groupby(level=["date", "symbol"]).first()

        self._panel = panel
        return panel

    def load_into(self, lens: "CryptoFactorLens"):
        """Load the panel into the analysis pipeline."""
        from .pipeline import CryptoFactorLens  # avoid circular import
        if not isinstance(lens, CryptoFactorLens):
            raise TypeError("lens must be an instance of CryptoFactorLens")
        panel = self.load_panel()
        lens.load_from_panel(panel)
        return lens
