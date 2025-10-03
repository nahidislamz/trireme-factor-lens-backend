# Construct GMM for regime detection From the data saved in factors_data.csv
# ========================= construct_GMM.py =========================
import pandas as pd
from crypto_factor_lens.regime_gmm import RegimeGMM
from Factor_Lens.config import LensConfig

# Pass n_regimes from command line to main
def main():
    # Load factors data
    df = pd.read_csv("factors_data.csv", index_col=0, parse_dates=True)
    print(f"Loaded factors_data.csv with shape {df.shape}")

    # Separate factors and BTC price
    factors = df.drop(columns=["BTC"])
    btc = df["BTC"]

    # Compute number of regimes from config or set default
    cfg = LensConfig()

    # Fit GMM
    gmm = RegimeGMM(n_regimes=cfg.n_regimes, random_state=42, scale=True)
    gmm.fit(factors)

    # Get regime labels and probabilities
    regimes = gmm.labels_
    probs = gmm.probs_

    # Smooth regimes
    smoothed_regimes = gmm.smooth_labels(window=5)

    # Save results
    regimes_df = pd.DataFrame({
        "regime": regimes,
        "smoothed_regime": smoothed_regimes
    }, index=factors.index)
    regimes_df = pd.concat([regimes_df, probs, btc], axis=1)
    regimes_df.to_csv("gmm_regimes.csv", float_format="%.10g")
    print(f"Saved gmm_regimes.csv with shape {regimes_df.shape}")



if __name__ == "__main__":
    main()