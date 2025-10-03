import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class RegimeGMM:
    def __init__(self, n_regimes: int = 3, random_state: int = 42, scale: bool = True):
        """
        n_regimes: number of GMM components (k)
        random_state: for reproducibility
        scale: whether to standardize factor values before fitting
        """
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scale = scale
        self.model: GaussianMixture | None = None
        self.scaler: StandardScaler | None = None
        self.labels_: pd.Series | None = None
        self.probs_: pd.DataFrame | None = None

    def fit(self, factors: pd.DataFrame):
        """Fit a Gaussian Mixture Model on daily factor values."""
        X = factors.values.astype(float)

        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            random_state=self.random_state
        )
        self.model.fit(X)

        self.labels_ = pd.Series(
            self.model.predict(X),
            index=factors.index,
            name="regime"
        )
        self.probs_ = pd.DataFrame(
            self.model.predict_proba(X),
            index=factors.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)]
        )
        return self

    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Predict regimes for new factors data."""
        X = factors.values.astype(float)
        if self.scale and self.scaler is not None:
            X = self.scaler.transform(X)
        labels = self.model.predict(X)
        return pd.Series(labels, index=factors.index, name="regime")
    
    def smooth_labels(self, window: int = 5) -> pd.Series:
        """Smooth regime labels using rolling mode."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before smoothing labels.")

        def rolling_mode(s):
            return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]

        smoothed = self.labels_.rolling(window=window, center=True, min_periods=1).apply(rolling_mode)
        return smoothed.astype(int)
