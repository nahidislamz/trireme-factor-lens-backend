# crypto_factor_lens/viz_plotly.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class PlotlyVisualizer:
    def __init__(self, factors: pd.DataFrame, btc_series: pd.Series | None = None):
        if factors is None or factors.empty:
            raise ValueError("No factors provided for visualization.")
        self.btc_series = btc_series  # optional
        self.factors = factors

    def cumulative(self, title: str = "Factor Cumulative Returns"):
        """
        Plot cumulative returns for all factor series with interactive date range selector.
        Optionally overlays BTC price (normalized).
        """
        cumrets = (1 + self.factors.fillna(0)).cumprod()

        fig = go.Figure()

        # Add factors
        for col in cumrets.columns:
            fig.add_trace(go.Scatter(
                x=cumrets.index,
                y=cumrets[col],
                mode="lines",
                name=col
            ))

        # Add BTC overlay if available
        if self.btc_series is not None and not self.btc_series.empty:
            btc_norm = (self.btc_series / self.btc_series.iloc[0]).reindex(cumrets.index)
            fig.add_trace(go.Scatter(
                x=btc_norm.index,
                y=btc_norm.values,
                mode="lines",
                name="BTC Price",
                line=dict(color="black", dash="dot", width=2)
            ))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=20)
            ),
            xaxis_title="Date",
            yaxis_title="Cumulative Growth",
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=1200,
            height=700,
            margin=dict(t=80, r=200, b=80, l=80)
        )

        # Add interactive range selector
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=12, label="1y", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )

        return fig

    def factor_correlation(self, title: str = "Factor Correlation Matrix"):
        """
        Show correlation matrix between factors with larger size.
        """
        corr = self.factors.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=title,
            width=900,
            height=800
        )

        return fig

    def plot_regimes(self, price: pd.Series, regimes: pd.Series, title="BTC with Regimes"):
        """Plot BTC price with colored regime background (darker colors)."""
        # align indices
        price = price.reindex(regimes.index).astype(float)
        regimes = regimes.astype(int)

        fig = go.Figure()

        # plot BTC
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name="BTC", line=dict(color="black")))

        # stronger colors for regimes
        colors = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#9467bd",  # purple
            "#ff7f0e",  # orange
        ]

        prev = regimes.iloc[0]
        start = regimes.index[0]

        for t, r in regimes.iloc[1:].items():
            if r != prev:
                fig.add_vrect(
                    x0=start, x1=t,
                    fillcolor=colors[prev % len(colors)],
                    opacity=0.3,       # darker highlight
                    line_width=0
                )
                start, prev = t, r
        # add last block
        fig.add_vrect(
            x0=start, x1=regimes.index[-1],
            fillcolor=colors[prev % len(colors)],
            opacity=0.3,
            line_width=0
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="BTC Price",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def plot_regime_probabilities(self, price: pd.Series, probs: pd.DataFrame, title="GMM Regime Probabilities"):
        """
        Plot BTC price with regime probabilities from GMM.
        
        Parameters
        ----------
        price : pd.Series
            BTC price series (indexed by date).
        probs : pd.DataFrame
            GMM probabilities (columns like 'regime_0', 'regime_1', ...).
        """
        # align
        common_idx = price.index.intersection(probs.index)
        price = price.loc[common_idx]
        probs = probs.loc[common_idx]

        # Create subplot: 2 rows (price, probs)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            subplot_titles=("BTC Price", "Regime Probabilities")
        )

        # --- Top panel: BTC price ---
        fig.add_trace(
            go.Scatter(x=price.index, y=price.values, name="BTC", line=dict(color="black")),
            row=1, col=1
        )

        # --- Bottom panel: stacked probabilities ---
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

        cum = np.zeros(len(probs))
        for i, col in enumerate(probs.columns):
            fig.add_trace(
                go.Scatter(
                    x=probs.index,
                    y=cum + probs[col].values,
                    mode="lines",
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                    stackgroup="one",
                    name=col
                ),
                row=2, col=1
            )
            cum += probs[col].values

        # Layout
        fig.update_layout(
            title=title,
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
        )

        return fig



