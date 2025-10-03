from __future__ import annotations
import pandas as pd
from typing import Optional
from Factor_Lens.config import LensConfig

class UniverseMaskBuilder:
    """
    Builds date×symbol boolean masks for eligibility.
    """

    def __init__(self, cfg: LensConfig):
        self.cfg = cfg

    def history_mask(self, returns: pd.DataFrame) -> pd.DataFrame:
        """True where we have at least min_hist_window non-NA returns up to t-1."""
        # count of non-NA over trailing window, shifted to avoid look-ahead
        valid_count = returns.notna().rolling(self.cfg.min_hist_window).sum().shift(1)
        return (valid_count >= self.cfg.min_hist_window)

    def liquidity_mask(
        self,
        panel: pd.DataFrame,
        price_col: str,
        volume_col: str
    ) -> pd.DataFrame:
        """
        True where rolling median $volume >= threshold (shifted one day).
        panel is MultiIndex (date, symbol) with price/volume columns.
        """
        if not self.cfg.use_liquidity_filter:
            # If disabled, everything is True where returns exist (will be ANDed later)
            wide = panel.reset_index().pivot(index="date", columns="symbol", values=price_col)
            return wide.notna()

        df = panel.reset_index()
        if price_col not in df.columns or volume_col not in df.columns:
            # If missing, don't exclude anything here; history_mask covers availability.
            wide = df.pivot(index="date", columns="symbol", values=price_col)
            return wide.notna()

        dv = (df[price_col] * df[volume_col]).rename("dollar_vol")
        dv_wide = df.assign(dollar_vol=dv).pivot(index="date", columns="symbol", values="dollar_vol").sort_index()
        med = dv_wide.rolling(self.cfg.liquidity_window).median().shift(1)
        return (med >= self.cfg.min_dollar_volume)

    def combine(self, *masks: Optional[pd.DataFrame]) -> pd.DataFrame:
        """AND-combine a list of aligned boolean masks, ignoring None."""
        ms = [m for m in masks if m is not None]
        if not ms:
            raise ValueError("No masks provided to combine()")
        out = ms[0].copy()
        for m in ms[1:]:
            out = out & m.reindex_like(out)
        return out.fillna(False)

    def monthly_topN_by_rolling_mcap_exstables(
        self,
        mc_wide: pd.DataFrame,
        stable_mask: pd.DataFrame
        ) -> pd.DataFrame:
        # 7D mean mcap, using only data up to t-1 (no look-ahead)
        mc_roll = mc_wide.rolling(self.cfg.universe_mcap_roll_window).mean().shift(1)

        # month-start anchors (align to first date present)
        month_starts = mc_roll.resample(self.cfg.universe_recalc).first().index
        month_starts = month_starts[month_starts.isin(mc_roll.index)]

        mask = pd.DataFrame(False, index=mc_roll.index, columns=mc_roll.columns)

        for i, d in enumerate(month_starts):
            # exclude stablecoins at selection time
            ex_stable = ~stable_mask.loc[d].reindex(mc_roll.columns).fillna(True)
            scores = mc_roll.loc[d].where(ex_stable)

            # Top N by size (largest)
            top = scores.nlargest(self.cfg.universe_top_n).dropna().index

            # hold until next month-start (or end of index)
            end = month_starts[i+1] if i+1 < len(month_starts) else mc_roll.index[-1]
            mask.loc[d:end, top] = True

        return mask
