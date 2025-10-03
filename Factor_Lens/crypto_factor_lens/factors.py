from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from Factor_Lens.config import LensConfig
from .weighting import WEIGHTING
# Note: selectors aren’t used for these fixed-N factors; keeping import if you add quantile-based ones later.


# ========== Base: shared logic (template + helpers) ==========

class BaseFactor:
    """
    Template base for cross-sectional long–short factors with:
      - time-varying universe masks
      - rebalancing schedule (cfg.factor_rebalance_freq)
      - √mcap weighting (or other via cfg.weighting registry)
      - dollar neutrality (+0.5 long, -0.5 short)
      - forward-filled holdings between rebalances
      - optional beta-hedge vs market basket
    Subclasses must implement:
      * _precompute_signal() -> pd.DataFrame (date × symbol)
      * _long_short_config() -> (n_long:int, n_short:int, long_ascending:bool, factor_name:str)
    """

    def __init__(
        self,
        cfg: LensConfig,
        returns: pd.DataFrame,
        market_cap_wide: Optional[pd.DataFrame] = None,
        market: Optional[pd.Series] = None,
        close_wide: Optional[pd.DataFrame] = None,
    ):
        self.cfg = cfg
        self.returns = returns
        self.market_cap_wide = market_cap_wide
        self.market = market
        self.close_wide = close_wide

        # Weighting registry (sqrt_mcap expected by your spec)
        self.weighting = WEIGHTING[cfg.weighting]()
        self._debug_baskets: dict = {}   

        # Cached signal matrix (computed once per factor)
        self._signal_df: Optional[pd.DataFrame] = None

    # ---------- Template hooks (override in subclasses) ----------

    def _precompute_signal(self) -> pd.DataFrame:
        """Return date×symbol DataFrame of the selection metric (no masking)."""
        raise NotImplementedError

    def _long_short_config(self) -> tuple[int, int, bool, str]:
        """
        Returns (n_long, n_short, long_ascending, factor_name)
          - long_ascending=True  -> long the smallest values
          - long_ascending=False -> long the largest values
        """
        raise NotImplementedError

    # ---------- Shared helpers ----------

    def _rebal_dates(self) -> pd.DatetimeIndex:
        """Bi-weekly schedule by default (cfg.factor_rebalance_freq)."""
        return self.returns.resample(self.cfg.factor_rebalance_freq).first().index

    def _apply_masks(
        self,
        d: pd.Timestamp,
        sig: pd.Series,
        universe_mask: pd.DataFrame,
        exclude_mask: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Filter signal by eligibility masks on date d."""
        elig = universe_mask.loc[d].reindex(sig.index).fillna(False)
        if exclude_mask is not None:
            elig &= ~exclude_mask.loc[d].reindex(sig.index).fillna(False)
        return sig[elig].dropna()

    def _pick_top_bottom(
        self, sig: pd.Series, n_long: int, n_short: int, long_ascending: bool
    ) -> tuple[pd.Index, pd.Index]:
        """Return (long_idx, short_idx) by ranking the cross-section."""
        if long_ascending:
            # long the smallest values
            long_idx = sig.nsmallest(n_long).index
            short_idx = sig.nlargest(n_short).index
        else:
            # long the largest values
            long_idx = sig.nlargest(n_long).index
            short_idx = sig.nsmallest(n_short).index
        return long_idx, short_idx

    def _weights_for_baskets(
        self, d: pd.Timestamp, sig: pd.Series, long_idx: pd.Index, short_idx: pd.Index
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute √mcap (or configured) weights for long/short baskets on date d,
        then scale to dollar-neutral (+0.5, -0.5).
        """
        if self.market_cap_wide is None:
            raise ValueError("market_cap_wide required for weighting.")
        mcap_row = self.market_cap_wide.loc[d]
        wL = self.weighting.weights(sig, long_idx, mcap_row=mcap_row)
        wS = self.weighting.weights(sig, short_idx, mcap_row=mcap_row)

        if self.cfg.dollar_neutral:
            wL = (0.5 * wL) / max(wL.sum(), 1e-12) if len(wL) else wL
            wS = (-0.5 * wS) / max(wS.sum(), 1e-12) if len(wS) else wS
        return wL, wS

    def _beta_hedge(self, raw: pd.Series, market: pd.Series) -> pd.Series:
        """Hedge factor vs market using EW beta (expects market already aligned to raw.index)."""
        if not self.cfg.market_beta_neutral or market is None:
            return raw
        hl = self.cfg.hedge_halflife
        cov  = raw.ewm(halflife=hl, adjust=False).cov(market)
        var  = market.ewm(halflife=hl, adjust=False).var()
        beta = (cov / var).fillna(0.0)
        return (raw - beta * market).rename((raw.name or "factor") + "_betaHedged")
    


    # ---------- Main template method ----------

    # def build(
    #     self,
    #     universe_mask: pd.DataFrame,
    #     exclude_mask: Optional[pd.DataFrame] = None,
    # ) -> pd.Series:
    #     """
    #     Full lifecycle:
    #       1) precompute signal matrix once
    #       2) iterate rebal dates → pick baskets → compute weights
    #       3) forward-fill weights
    #       4) daily factor return = sum(weights * returns)
    #       5) optional beta-hedge
    #     """
    #     # 1) signal matrix
    #     if self._signal_df is None:
    #         self._signal_df = self._precompute_signal().reindex(self.returns.index)

    #     nL, nS, long_asc, name = self._long_short_config()
    #     rebal_dates = self._rebal_dates()

    #     # 2) weights only on rebal dates
    #     W = pd.DataFrame(0.0, index=self.returns.index, columns=self.returns.columns)

    #     for d in rebal_dates:
    #         if d not in self._signal_df.index:
    #             continue
    #         sig_row = self._signal_df.loc[d]

    #         # apply masks
    #         sig = self._apply_masks(d, sig_row, universe_mask, exclude_mask)
    #         if len(sig) < nL + nS:
    #             continue

    #         # select baskets
    #         long_idx, short_idx = self._pick_top_bottom(sig, nL, nS, long_asc)
    #         if len(long_idx) == 0 or len(short_idx) == 0:
    #             continue

    #         # weights
    #         wL, wS = self._weights_for_baskets(d, sig, long_idx, short_idx)

    #         nL, nS, long_asc, name = self._long_short_config()
    #         self._debug_baskets[d] = {
    #             "sig": sig.copy(),
    #             "long_idx": long_idx.copy(),
    #             "short_idx": short_idx.copy(),
    #             "wL": wL.copy(),
    #             "wS": wS.copy(),
    #             "name": name,
    #             "market": self.market.copy() if self.market is not None else None,
    #             "returns": self.returns.copy(),
    #         }

    #         if len(wL) == 0 or len(wS) == 0:
    #             continue

    #         W.loc[d, wL.index] = wL
    #         W.loc[d, wS.index] = wS
  
        # # 3) hold between rebalances
        # W = W.replace(0.0, np.nan).ffill().fillna(0.0)

        # # # Insteead of fillna(0.0), could also drop days with no holdings
        # # W = W.replace(0.0, np.nan).ffill().dropna(how='all')

        # # store full weights for later debugging
        # self._weights = W.copy()

        # # 4) daily factor returns
        # raw = (W * self.returns).sum(axis=1).rename(name)

        # # in BaseFactor.build(), after you set self._weights and build raw_series
        # self._market_used = self.market.reindex(raw.index).copy()


    #     # 5) hedge
    #     if self.cfg.market_beta_neutral:
    #         final_series = self._beta_hedge(raw, self._market_used)
    #     else:
    #         final_series = raw
    #     # Return everything for debugging purposes
    #     return final_series 
    def build(
    self,
    universe_mask: pd.DataFrame,
    exclude_mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
        """
        Debug-first build:
        - For each rebalance date d: select baskets, compute weights, and compute ALL daily
            series for the holding window [d, next_rebal-1].
        - Store everything in self._debug_baskets[d].
        - Concatenate windows to form full daily raw series and weights matrix.
        - Apply beta hedge once on the full series; also stash EW cov/var/beta per day
            back into each basket's window.
        """
        # 0) canonical daily index & precompute signal
        idx = self.returns.index
        if self._signal_df is None:
            self._signal_df = self._precompute_signal().reindex(idx)

        nL, nS, long_asc, name = self._long_short_config()
        rebal_dates = self._rebal_dates().intersection(idx)
        if len(rebal_dates) == 0:
            # nothing to do
            return pd.Series(index=idx, dtype=float, name=name)

        # containers for concatenation
        raw_segments: list[pd.Series] = []
        W_blocks: list[pd.DataFrame] = []

        # clear debug baskets for a fresh run
        self._debug_baskets = {}

        # 1) iterate over rebalance dates with their holding windows
        for i, d in enumerate(rebal_dates):
            # holding window: [d, next_d) on the daily index
            next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
            if next_d is None:
                win_mask = (idx >= d)
            else:
                win_mask = (idx >= d) & (idx < next_d)
            win_dates = idx[win_mask]
            if len(win_dates) == 0:
                continue

            # signal row & masks at d
            if d not in self._signal_df.index:
                continue
            sig_row = self._signal_df.loc[d]
            sig = self._apply_masks(d, sig_row, universe_mask, exclude_mask)
            if len(sig) < (nL + nS):
                continue

            # select baskets at d
            long_idx, short_idx = self._pick_top_bottom(sig, nL, nS, long_asc)
            if len(long_idx) == 0 or len(short_idx) == 0:
                continue

            # weights at d (dollar-neutral +0.5 / -0.5)
            wL, wS = self._weights_for_baskets(d, sig, long_idx, short_idx)
            if len(wL) == 0 or len(wS) == 0:
                continue
            W_row = wL.add(wS, fill_value=0.0)

            # slice returns only for assets in the basket for the whole window
            cols = W_row.index
            r_win = self.returns.loc[win_dates, cols]

            # per-side daily return series across the window
            long_series  = r_win.reindex(columns=wL.index).mul(wL, axis=1).sum(axis=1) if len(wL) else pd.Series(0.0, index=win_dates)
            short_series = r_win.reindex(columns=wS.index).mul(wS, axis=1).sum(axis=1) if len(wS) else pd.Series(0.0, index=win_dates)
            raw_series_win = (long_series + short_series).rename(name)

            # per-asset daily contributions (weight * return) for debug
            contrib_win = r_win.mul(W_row, axis=1)

            # market returns aligned to window (store; hedge is computed later on full series)
            mkt_win = self.market.reindex(win_dates) if self.market is not None else None

            # build a weight block for this window (same row repeated over win_dates)
            W_block = pd.DataFrame(
                data=np.tile(W_row.values, (len(win_dates), 1)),
                index=win_dates,
                columns=W_row.index,
            ).reindex(columns=self.returns.columns).fillna(0.0)

            # stash everything for this rebalance into debug_baskets[d]
            self._debug_baskets[d] = {
                "name": name,
                "anchor_rebalance": d,
                "window_start": win_dates[0],
                "window_end": win_dates[-1],
                "dates": win_dates,
                "signal": sig.copy(),
                "long_idx": list(long_idx),
                "short_idx": list(short_idx),
                "weights_long": wL.copy(),
                "weights_short": wS.copy(),
                "weights_row": W_row.copy(),
                "returns_long": long_series.copy(),
                "returns_short": short_series.copy(),
                "raw_returns": raw_series_win.copy(),
                "market_returns": mkt_win.copy() if mkt_win is not None else None,
                "contrib": contrib_win.copy(),  # daily per-asset contributions within window
                # placeholders filled after global hedge is computed:
                "beta": None,
                "ew_cov": None,
                "ew_var": None,
                "hedged_returns": None,
            }

            # collect for concatenation
            raw_segments.append(raw_series_win)
            W_blocks.append(W_block)

        # 2) concatenate windows to full daily raw series and full weights matrix
        if len(raw_segments) == 0:
            return pd.Series(index=idx, dtype=float, name=name)

        raw_full = pd.concat(raw_segments, axis=0).reindex(idx).fillna(0.0).rename(name)
        W_full = (
            pd.concat(W_blocks, axis=0)
            .groupby(level=0)  # if windows overlap on the boundary, sum => same row
            .sum()
            .reindex(idx)
            .fillna(0.0)
        )

        # store for external debuggers
        self._weights = W_full.copy()
        self._raw_series = raw_full.copy()

        # 3) hedge once on the full raw series; also compute EW cov/var/beta once
        mkt_aligned = self.market.reindex(idx) if self.market is not None else None
        self._market_used = mkt_aligned.copy() if mkt_aligned is not None else None

        if self.cfg.market_beta_neutral and mkt_aligned is not None:
            # compute EW components (to also store per window)
            hl = self.cfg.hedge_halflife
            cov_series = raw_full.ewm(halflife=hl, adjust=False).cov(mkt_aligned)
            var_series = mkt_aligned.ewm(halflife=hl, adjust=False).var()
            beta_series = (cov_series / var_series).fillna(0.0)

            hedged_full = (raw_full - beta_series * mkt_aligned).rename((name or "factor") + "_betaHedged")

            # write EW components and hedged window back into each basket
            for d, rec in self._debug_baskets.items():
                win_idx = rec["dates"]
                rec["beta"] = beta_series.reindex(win_idx).copy()
                rec["ew_cov"] = cov_series.reindex(win_idx).copy()
                rec["ew_var"] = var_series.reindex(win_idx).copy()
                rec["hedged_returns"] = hedged_full.reindex(win_idx).copy()

            return hedged_full
        else:
            # if no hedge, still fill placeholders with None
            for d, rec in self._debug_baskets.items():
                rec["beta"] = None
                rec["ew_cov"] = None
                rec["ew_var"] = None
                rec["hedged_returns"] = None

            return raw_full




# ========== Concrete factors ==========

class MarketCapFactor(BaseFactor):
    """
    Metric: 7-day rolling mean of market cap (shifted by 1 day).
    N: 30×30. Long = larger caps (ascending=True).
    """
    def _precompute_signal(self) -> pd.DataFrame:
        if self.market_cap_wide is None:
            raise ValueError("market_cap_wide is required for MarketCapFactor.")
        
        # Previous simple mean version:
        return self.market_cap_wide.rolling(7).mean().shift(1)

        # log transform to reduce extreme spread between mega-cap and micro-cap
        # return np.log(self.market_cap_wide).rolling(7).mean().shift(1)
    
    def _long_short_config(self):
        return 30, 30, False, "MarketCap"


class VolatilityFactor(BaseFactor):
    """
    Metric: 45-day rolling std of close-to-close returns (shifted by 1 day).
    N: 30×30. Long = high vol (ascending=True).
    """
    def _precompute_signal(self) -> pd.DataFrame:
        return self.returns.rolling(self.cfg.vol_window).std().shift(1)

    def _long_short_config(self):
        return 30, 30, False, "Volatility"


class MomentumFactor(BaseFactor):
    """
    Metric (x-mom): cumulative log return over 180d excluding most recent 7d.
    Tie-breaker: 30–7d momentum (small epsilon).
    N: 40×40. Long = high momentum (ascending=False).
    """
    def _precompute_signal(self) -> pd.DataFrame:
        log_ret = np.log1p(self.returns.fillna(0.0))
        xmom = log_ret.rolling(180).sum().shift(7)  # exclude last 7 days
        tiebreak = log_ret.rolling(30).sum().shift(7) - log_ret.rolling(7).sum().shift(7)
        return (xmom + 1e-6 * tiebreak)

    def _long_short_config(self):
        return 40, 40, False, "Momentum"


class DFTValueFactor(BaseFactor):
    """
    Metric: z-score of (log price - 180d EWMA(log price)) evaluated at t-7,
            scaled by 180d rolling std of the residual.
    Signal: long highest z (undervalued), short highest z.
    N: 30×30. Long = high z (ascending=True).
    """
    def _precompute_signal(self) -> pd.DataFrame:
        if self.close_wide is None:
            raise ValueError("close_wide is required for DFTValueFactor.")
        logP = np.log(self.close_wide)
        trend = logP.ewm(span=180, adjust=False).mean()
        resid = logP - trend
        sigma = resid.rolling(180).std()
        # evaluate with a 7-day gap
        return (logP.shift(7) - trend.shift(7)) / sigma

    def _long_short_config(self):
        return 30, 30, False, "DFTValue"
    
class BetaPremiumFactor(BaseFactor):
    """
    Metric: 90d exponentially-weighted OLS beta (half-life ≈ 21d) of r_i vs r_m (BTC60:ETH40),
            then shrinkage toward 1 with λ = k/(k+T), k=20, T=90.
    Signal: long highest beta, short lowest beta.
    N: 30×30. Long = high beta (ascending=False).
    """
    def _precompute_signal(self) -> pd.DataFrame:
        if self.market is None:
            raise ValueError("market series is required for BetaPremiumFactor.")

        # Covariance: returns vs market
        cov = self.returns.ewm(halflife=21, adjust=False).cov(self.market)
        var = self.market.ewm(halflife=21, adjust=False).var()

        # cov has MultiIndex (date, symbol); reshape into wide form
        if isinstance(cov, pd.Series):
            cov = cov.unstack()  # -> DataFrame with index=date, columns=symbols
        beta = cov.div(var, axis=0)

        # shrinkage toward 1
        T, k = 90, 20
        lam = k / (k + T)
        beta = (1 - lam) * beta + lam * 1.0
        return beta

    def _long_short_config(self):
        return 30, 30, False, "BetaPremium"