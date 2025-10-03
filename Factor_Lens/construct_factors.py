from pathlib import Path
from Factor_Lens.config import LensConfig
from crypto_factor_lens.dataloader import DataLoader
from crypto_factor_lens.pipeline import CryptoFactorLens

def main():
    cfg = LensConfig()
    # load data
    loader = DataLoader(cfg)
    panel = loader.load_panel() 
    panel.head()
    # build factors (exactly as in main.ipynb)
    lens = CryptoFactorLens(cfg)
    lens.load_from_panel(panel)
    lens.build_factors()

    # Load data
    factors_df = lens.factors

    # combine
    df = factors_df.copy()
    df["BTC"] = lens.close_wide["BTC"].reindex(df.index)

    # Add prefix of factor_ to factor columns
    factor_cols = factors_df.columns.tolist()
    df.rename(columns={col: f"factor_{col}" for col in factor_cols},
                inplace=True)

    # save
    out_path = Path("factors_data.csv")
    df.to_csv(out_path, float_format="%.10g")
    print(f"Saved factors_data.csv with shape {df.shape}")

if __name__ == "__main__":
    main()