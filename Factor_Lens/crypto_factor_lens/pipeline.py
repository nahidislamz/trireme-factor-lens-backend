from __future__ import annotations
import pandas as pd
from typing import Optional, Dict

from Factor_Lens.config import LensConfig

from .factors import (
    MarketCapFactor,
    VolatilityFactor,
    MomentumFactor,
    DFTValueFactor,
    BetaPremiumFactor,
)
from .universe_builder import UniverseMaskBuilder
from .stablecoins import StablecoinDetector


class CryptoFactorLens:
    def __init__(self, cfg: LensConfig):
        self.cfg = cfg
        self.panel: Optional[pd.DataFrame] = None

        # core wide frames
        self.close_wide: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.volatility: Optional[pd.DataFrame] = None
        self.market_cap_wide: Optional[pd.DataFrame] = None
        self.ranks_by_date: Optional[pd.DataFrame] = None  # used only by diagnostics

        # masks
        self.stable_mask: Optional[pd.DataFrame] = None
        self.universe_mask: Optional[pd.DataFrame] = None

        # market & outputs
        self.market: Optional[pd.Series] = None
        self.factors: Optional[pd.DataFrame] = None
        self.portfolios: Optional[pd.DataFrame] = None

    # -----------------------------
    # Data loading
    # -----------------------------
    def load_from_panel(self, panel: pd.DataFrame) -> "CryptoFactorLens":
        if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != ["date", "symbol"]:
            raise ValueError("panel must have MultiIndex (date, symbol)")
        self.panel = panel.copy()
        return self

    # -----------------------------
    # Prices, returns, volatility
    # -----------------------------
    def _select_price_column(self) -> str:
        assert self.panel is not None
        cols = self.panel.columns
        if self.cfg.preferred_price_col in cols:
            return self.cfg.preferred_price_col
        if self.cfg.fallback_price_col in cols:
            return self.cfg.fallback_price_col
        raise ValueError("No suitable price column found (tried preferred and fallback).")

    def _build_close_and_returns(self):
        if self.panel is None:
            raise ValueError("No panel loaded. Call load_from_panel first.")

        price_col = self._select_price_column()
        df = self.panel.reset_index()

        # pivot is safe because DataLoader deduped (date, symbol)
        self.close_wide = (
            df.pivot(index="date", columns="symbol", values=price_col)
              .sort_index()
        )
        self.returns = (
            self.close_wide.pct_change(fill_method=None)
            .replace([float("inf"), float("-inf")], pd.NA)
        )

        # rolling volatility (ending at t-1) for any diagnostic use
        self.volatility = self.returns.rolling(window=self.cfg.vol_window).std().shift(1)

    # -----------------------------
    # Market (60/40) construction
    # -----------------------------
    def _build_market_series(self):
        if self.returns is None:
            raise ValueError("Build returns first.")
        w = pd.Series(self.cfg.market_weights, dtype=float)
        w = w[w.index.isin(self.returns.columns)]
        if w.empty:
            raise ValueError("None of market weight symbols exist in returns.")
        w = w / w.sum()
        self.market = (self.returns[w.index] * w).sum(axis=1).rename("Market_60_40")

    # -----------------------------
    # Market cap wide + ranks (for diagnostics only)
    # -----------------------------
    def _map_ranks_by_date(self):
        if self.panel is None:
            raise ValueError("No panel loaded.")
        df = self.panel.reset_index()
        if "market_cap_cg" not in df.columns:
            raise ValueError("market_cap_cg column required")
        mc_wide = (
            df.pivot(index="date", columns="symbol", values="market_cap_cg")
              .sort_index()
        )
        self.market_cap_wide = mc_wide
        # still compute ranks for diagnostics
        self.ranks_by_date = mc_wide.rank(axis=1, ascending=False, method="first")

    # -----------------------------
    # Universe & stables masks
    # -----------------------------
    def _build_masks(self):
        if self.panel is None or self.returns is None or self.market_cap_wide is None:
            raise ValueError("Build returns and market_cap_wide first.")

        umb = UniverseMaskBuilder(self.cfg)

        # stablecoins (used for ex-stable selection)
        sdet = StablecoinDetector(self.cfg)
        self.stable_mask = sdet.detect(self.returns)

        # monthly Top-200 selections
        #  - EX-stables (spec universe)
        monthly_ex = umb.monthly_topN_by_rolling_mcap_exstables(self.market_cap_wide, self.stable_mask)

        # optional liquidity filter
        if self.cfg.use_liquidity_filter:
            price_col = self.cfg.preferred_price_col if self.cfg.preferred_price_col in self.panel.columns else self.cfg.fallback_price_col
            vol_col   = self.cfg.preferred_volume_col if self.cfg.preferred_volume_col in self.panel.columns else self.cfg.fallback_volume_col
            liq = umb.liquidity_mask(self.panel, price_col, vol_col)
        else:
            liq = pd.DataFrame(True, index=self.returns.index, columns=self.returns.columns)

        # optional history filter
        if getattr(self.cfg, "use_history_filter", False):
            hist = umb.history_mask(self.returns)
        else:
            hist = pd.DataFrame(True, index=self.returns.index, columns=self.returns.columns)

        # combine
        self.universe_mask = umb.combine(monthly_ex, liq, hist)

    # -----------------------------
    # Factor construction
    # -----------------------------
    def build_factors(self) -> "CryptoFactorLens":
        """
        Build all factor series on the ex-stable universe.
        """
        # --- prerequisites ---
        self._build_close_and_returns()
        self._build_market_series()
        self._map_ranks_by_date()
        self._build_masks()

        # --- instantiate factor objects ---
        common_kwargs = dict(
            cfg=self.cfg,
            returns=self.returns,
            market_cap_wide=self.market_cap_wide,
            market=self.market,
        )

        f_mcap = MarketCapFactor(**common_kwargs)
        f_vol  = VolatilityFactor(**common_kwargs)
        f_mom  = MomentumFactor(**common_kwargs)
        f_dft  = DFTValueFactor(**common_kwargs, close_wide=self.close_wide)
        f_beta = BetaPremiumFactor(**common_kwargs)

        F_MarketCap = f_mcap.build(self.universe_mask, exclude_mask=None)
        F_Volatility = f_vol.build(self.universe_mask, exclude_mask=None)
        F_Momentum = f_mom.build(self.universe_mask, exclude_mask=None)
        F_DFTValue = f_dft.build(self.universe_mask, exclude_mask=None)
        F_BetaPremium = f_beta.build(self.universe_mask, exclude_mask=None)

        # --- build factor series (only ex-stable universe) ---
        cols = {
            "MarketCap": F_MarketCap,
            "Volatility": F_Volatility,
            "Momentum": F_Momentum,
            "DFTValue": F_DFTValue,
            "BetaPremium": F_BetaPremium,
        }

        self.factors = pd.DataFrame(cols).sort_index()

        # keep references for debugging/explainability
        self.factor_objects: Dict[str, object] = {
            "MarketCap": f_mcap,
            "Volatility": f_vol,
            "Momentum": f_mom,
            "DFTValue": f_dft,
            "BetaPremium": f_beta,
        }

        return self

    
    # Extra parts no need for now

    # # -----------------------------
    # # Panel portfolios (EW) for diagnostics
    # # -----------------------------
    # def build_panel_portfolios(self) -> "CryptoFactorLens":
    #     # Diagnostics only; unchanged
    #     if self.returns is None or self.ranks_by_date is None or self.universe_mask is None:
    #         raise ValueError("Build returns/ranks/masks first.")

    #     selector = UNIVERSE_SELECTORS[self.cfg.universe_selector](self.cfg)
    #     rows, idxs = [], []
    #     for d in self.returns.index:
    #         r = self.returns.loc[d].copy()
    #         mask = self.universe_mask.loc[d].reindex(r.index).fillna(False)
    #         r = r[mask & (~r.isna())]
    #         if len(r) < self.cfg.min_symbols_per_day:
    #             rows.append([pd.NA, pd.NA, pd.NA]); idxs.append(d); continue

    #         ranks = self.ranks_by_date.loc[d].reindex(r.index)
    #         long_idx, short_idx = selector.select(ranks.dropna())

    #         allcap = r.mean()
    #         large  = r.loc[short_idx].mean() if len(short_idx) >= 3 else pd.NA
    #         small  = r.loc[long_idx].mean()  if len(long_idx)  >= 3 else pd.NA
    #         rows.append([allcap, large, small]); idxs.append(d)

    #     self.portfolios = pd.DataFrame(rows, index=idxs, columns=["AllCap_EW", "LargeCap_EW", "SmallCap_EW"]).astype(float)
    #     return self

    # def factor_breakdown_by_asset(self, factor_key: str, top_n: int = 5) -> pd.DataFrame:
    #     # unchanged example — adjust keys to your chosen column names
    #     if self.returns is None or self.ranks_by_date is None or self.universe_mask is None:
    #         raise ValueError("Build returns/ranks/masks first.")
    #     selector = UNIVERSE_SELECTORS[self.cfg.universe_selector](self.cfg)
    #     contribs = {}
    #     for d in self.returns.index:
    #         r = self.returns.loc[d].copy()
    #         mask = self.universe_mask.loc[d].reindex(r.index).fillna(False)
    #         r = r[mask & (~r.isna())]
    #         if len(r) < self.cfg.min_symbols_per_day:
    #             continue
    #         ranks = self.ranks_by_date.loc[d].reindex(r.index)
    #         long_idx, short_idx = selector.select(ranks.dropna())
    #         if factor_key.startswith("MarketCap_LS"):
    #             long_contrib  = r.loc[long_idx]   / max(len(long_idx), 1)
    #             short_contrib = -r.loc[short_idx] / max(len(short_idx), 1)
    #             day_contrib = pd.concat([long_contrib, short_contrib])
    #             contribs[d] = day_contrib
    #     df = pd.DataFrame.from_dict(contribs, orient="index").fillna(0.0)
    #     top_assets = df.abs().mean().nlargest(top_n).index
    #     return df[top_assets]
