from __future__ import annotations
import abc
import pandas as pd
from typing import Dict, Callable

class WeightingStrategy(abc.ABC):
    @abc.abstractmethod
    def weights(self, returns_row: pd.Series, idx, **kwargs) -> pd.Series:
        raise NotImplementedError

class EqualWeight(WeightingStrategy):
    def weights(self, returns_row: pd.Series, idx, **kwargs) -> pd.Series:
        n = len(idx)
        if n == 0:
            return pd.Series(dtype="float64")
        return pd.Series(1.0 / n, index=idx)

class SqrtMcapWeight(WeightingStrategy):
    def weights(self, returns_row, idx, **kwargs) -> pd.Series:
        mcap_row: pd.Series | None = kwargs.get("mcap_row")
        if mcap_row is None:
            raise ValueError("mcap_row required for sqrt_mcap weighting")
        w = mcap_row.reindex(idx).astype(float).clip(lower=0)
        if w.sum() == 0 or len(w) == 0:
            return pd.Series(dtype="float64")
        w = w.pow(0.5)
        return w / w.sum()

# registry (add inv_vol etc. later)
WEIGHTING: Dict[str, Callable[[], WeightingStrategy]] = {
    "equal": EqualWeight,
    "sqrt_mcap": SqrtMcapWeight,
}
