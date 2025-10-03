from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import os
@dataclass
class LensConfig:
    # Downloader / API
    # -----------------------------
    cg_api_key = os.getenv("COINGECKO_API_KEY", "default-value-if-not-found")               # CoinGecko API key 
    cg_top_n: int = 500   # how many top coins to fetch
    cg_sleep: float = 0.2 # seconds between requests (avoid rate limit)
    # -----------------------------
    # Data
    # -----------------------------
    data_dir: str = "data/data_Last_4_Years"
    merged_csv: str = "data/data_Last_4_Years/merged_sources.csv"

    # Prefer which price column for returns
    preferred_price_col: str = "price_cg"    # fallback to close_binance
    fallback_price_col: str = "close_binance" 

    # Volume columns for liquidity ($ volume = price * volume)
    preferred_volume_col: str = "volume_cg"    # fallback to volume_binance
    fallback_volume_col: str = "volume_binance"

    # # -----------------------------
    # # Universe construction (dynamic, daily)
    # # -----------------------------
    # universe_selector: str = "rank_quantiles"  # registry key in universe.py
    # lower_quantile: float = 0.30               # for rank/vol quantile splits
    # upper_quantile: float = 0.70
    # min_symbols_per_day: int = 10              # minimum required after masking

    # History / availability
    use_history_filter: bool = True            # require history mask
    min_hist_window: int = 30                  # require >= this many non-NA returns

    # Liquidity
    use_liquidity_filter: bool = False
    liquidity_window: int = 30                 # rolling median window (days)
    min_dollar_volume: float = 100_000.0       # threshold for $ volume

    # -----------------------------
    # Stablecoin handling
    # -----------------------------
    stable_known_symbols: List[str] = field(default_factory=lambda: [
        "USDT", "USDC", "DAI", "BUSD", "FDUSD", "TUSD", "USDe",
        "EURS", "EURT", "sUSD", "EUR", "EUROe", "EURC", "DOLA",
        "USDD", "CRVUSD", "XAUT", "PAXG", "FDUSD", "USDP", "TUSD",
        "BUSD"
    ])
    stable_sigma_thresh: float = 0.02          # rolling std threshold
    stable_corr_thresh: float = 0.97           # rolling corr to reference stable
    stable_window: int = 30                    # rolling window for σ and corr
    stable_reference_symbol: List[str] = field(default_factory=lambda: ["USDT", "EURS"])  # symbols to correlate with

    # -----------------------------
    # Factor params
    # -----------------------------
    vol_window: int = 45                       # rolling σ window for Vol factor

    # --- Market (BTC:ETH = 60:40) ---
    market_weights: dict[str, float] = field(default_factory=lambda: {"BTC": 0.6, "ETH": 0.4})

    # --- Universe monthly reconstitution (Top 200 by 7D mean mcap at month start) ---
    universe_recalc: str = "MS"            # month start
    universe_top_n: int = 200
    universe_mcap_roll_window: int = 7     # 7-day rolling mean mcap

    # --- Factor rebalancing ---
    factor_rebalance_freq: str = "2W"      # daily; pandas offset string

    # --- Weighting ---
    weighting: str = "sqrt_mcap"           # new scheme (add below)
    dollar_neutral: bool = True            # +0.5 long / -0.5 short

    # --- Market beta-neutralization ---
    market_beta_neutral: bool = True
    hedge_window: int = 90                 # rolling window for beta
    hedge_halflife: int = 30               # EW halflife for beta


