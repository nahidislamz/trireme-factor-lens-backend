import os
import pandas as pd
from typing import List


class DataManager:
    def __init__(self, outdir: str = "data_merged"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def merge_files(self, files: List[str], suffix: str) -> pd.DataFrame:
        """
        Load CSVs that contain 'date' and 'symbol'.
        Add suffix (_cg or _binance) to avoid column clashes, 
        but keep 'date' and 'symbol' unchanged.
        """
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, parse_dates=["date"])
                if "symbol" not in df.columns:
                    raise ValueError(f"{f} missing 'symbol' column.")

                # Only rename columns other than 'date' and 'symbol'
                rename_map = {
                    col: f"{col}_{suffix}"
                    for col in df.columns
                    if col not in ["date", "symbol"]
                }
                df = df.rename(columns=rename_map)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load {f}: {e}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def merge_sources(
        self,
        cg_files: List[str],
        binance_files: List[str],
        save: bool = True,
        filename: str = "merged_sources.csv",
    ) -> pd.DataFrame:
        """
        Merge CoinGecko and Binance data on (date, symbol).
        Always keeps all CoinGecko rows (left join).
        Adds Binance OHLCV if available, otherwise NaN.
        """
        cg_df = self.merge_files(cg_files, suffix="cg")
        bn_df = self.merge_files(binance_files, suffix="binance")

        if cg_df.empty and bn_df.empty:
            raise ValueError("No data to merge.")

        if not cg_df.empty:
            merged = pd.merge(
                cg_df,
                bn_df,
                on=["date", "symbol"],
                how="left",   # keep all CG rows
                validate="m:m"
            )
        else:
            merged = bn_df

        merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)

        if save:
            outpath = os.path.join(self.outdir, filename)
            merged.to_csv(outpath, index=False)
            print(f"✅ Saved merged dataset to {outpath} with shape {merged.shape}")

        # Diagnostics
        if not cg_df.empty and not bn_df.empty:
            cg_syms = set(cg_df["symbol"].unique())
            bn_syms = set(bn_df["symbol"].unique())
            inter = cg_syms & bn_syms
            print(f"Overlap: {len(inter)} symbols on Binance / {len(cg_syms)} from CoinGecko")

        return merged
