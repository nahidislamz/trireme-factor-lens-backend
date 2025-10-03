# crypto_factor_lens/data/binance_ccxt.py
import os, time
import pandas as pd
import ccxt
from typing import List
from config import LensConfig


class BinanceDownloader:
    """
    Downloader for Binance OHLCV data using CCXT.

    Fetches historical candlestick data for a list of trading pairs (symbols).
    Saves each symbol to CSV and can build a combined OHLCV panel.
    """

    def __init__(self, cfg: LensConfig):
        self.cfg = cfg
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.outdir = os.path.join(cfg.data_dir, "binance")
        os.makedirs(self.outdir, exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since_days: int = 1460,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a single symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTC/USDT".
        timeframe : str
            Candle timeframe (default "1d").
        since_days : int
            How many days back to fetch.
        limit : int
            Max candles per request (Binance max = 1000).

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume],
        indexed by date.
        """
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).timestamp() * 1000)

        all_candles = []
        while True:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(self.cfg.binance_sleep)

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        return df.drop(columns=["timestamp"]).set_index("date")

    def fetch_and_save(self) -> List[str]:
        """
        Download OHLCV data for all symbols in config and save as CSVs.
        Returns list of saved file paths.
        """
        if not self.cfg.binance_symbols:
            raise ValueError("binance_symbols must be set in cfg before downloading.")

        paths = []
        for sym in self.cfg.binance_symbols:
            try:
                print(f"Fetching OHLCV for {sym}...")
                df = self.fetch_ohlcv(
                    symbol=sym,
                    timeframe=self.cfg.binance_timeframe,
                    since_days=self.cfg.binance_days_back,
                    limit=self.cfg.binance_limit,
                )
                outpath = os.path.join(self.outdir, f"{sym.replace('/', '')}.csv")
                df.to_csv(outpath)
                print(f"  saved {len(df)} rows to {outpath}")
                paths.append(outpath)
            except Exception as e:
                print(f"Failed {sym}: {e}")
        return paths

    def build_panel(self, files: List[str]) -> dict[str, pd.DataFrame]:
        """
        Build OHLCV panels (dict of DataFrames).

        Returns
        -------
        dict with keys: "open", "high", "low", "close", "volume".
        Each value is a DataFrame indexed by date, columns = symbols.
        """
        panels = {field: [] for field in ["open", "high", "low", "close", "volume"]}
        symbols = []

        for f in files:
            sym = os.path.basename(f).replace(".csv", "")
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            symbols.append(sym)
            for field in panels:
                panels[field].append(df[[field]].rename(columns={field: sym}))

        # Concatenate per-field
        for field in panels:
            panels[field] = pd.concat(panels[field], axis=1).sort_index()

        return panels
