from __future__ import annotations
import pandas as pd
from typing import List, Optional
from Factor_Lens.config import LensConfig

class StablecoinDetector:
    """
    Produces a date×symbol boolean mask where True means the asset is a stablecoin today.
    Uses: known list OR (low rolling σ AND high rolling corr to a reference stable).
    """

    def __init__(self, cfg: LensConfig):
        self.cfg = cfg

    def known_mask(self, returns: pd.DataFrame) -> pd.DataFrame:
        cols = returns.columns
        known = pd.Series([c in set(self.cfg.stable_known_symbols) for c in cols], index=cols)
        # Broadcast down dates
        return pd.DataFrame({c: known[c] for c in cols}, index=returns.index)

    def sigma_mask(self, returns: pd.DataFrame) -> pd.DataFrame:
        sig = returns.rolling(self.cfg.stable_window).std()
        return (sig <= self.cfg.stable_sigma_thresh)

    def corr_mask(self, returns: pd.DataFrame) -> pd.DataFrame:
        references = self.cfg.stable_reference_symbol
        if not references:
            return pd.DataFrame(False, index=returns.index, columns=returns.columns)
        # Ensure references is a list
        if isinstance(references, str):
            references = [references]
        # Filter only references present in columns
        valid_refs = [ref for ref in references if ref in returns.columns]
        if not valid_refs:
            # If no valid reference, return all-False
            return pd.DataFrame(False, index=returns.index, columns=returns.columns)
        # Compute rolling correlation for each reference and take the max across references
        corr_masks = []
        for ref in valid_refs:
            ref_s = returns[ref]
            corr = returns.rolling(self.cfg.stable_window).corr(ref_s)
            corr_masks.append(corr >= self.cfg.stable_corr_thresh)
        # Combine masks: True if any reference passes the threshold
        combined_mask = corr_masks[0]
        for mask in corr_masks[1:]:
            combined_mask = combined_mask | mask
        return combined_mask

    def detect(self, returns: pd.DataFrame) -> pd.DataFrame:
        # shift by 1 to avoid look-ahead when used at day t
        known = self.known_mask(returns)
        sig   = self.sigma_mask(returns)
        cor   = self.corr_mask(returns)
        mask = known | (sig & cor)
        return mask.shift(1).fillna(False)
