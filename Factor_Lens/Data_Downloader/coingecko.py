# crypto_factor_lens/data/coingecko.py
import ccxt
import os, time, requests
import pandas as pd
from typing import List, Tuple
from .downloader import DataDownloader
from config import LensConfig

class CoinGeckoDownloader(DataDownloader):
    def __init__(self, cfg: LensConfig):
        self.cfg = cfg
        self.api_key = cfg.cg_api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.headers = {
            "accept": "application/json",
            "x-cg-pro-api-key": self.api_key
        }
        self.outdir = os.path.join(cfg.data_dir, "coingecko")
        os.makedirs(self.outdir, exist_ok=True)

    # ✅ FIX: pagination added
    def fetch_top_universe(self, n: int = 500) -> List[Tuple[str, str, str]]:
        """
        Fetch top N coins by market cap from CoinGecko.
        Supports pagination (max per_page=250).
        Returns list of (id, symbol, name).
        """
        url = f"{self.base_url}/coins/markets"
        per_page = 250
        pages = (n + per_page - 1) // per_page  # ceiling division

        universe = []
        for p in range(1, pages + 1):
            size = min(per_page, n - len(universe))
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": size,
                "page": p
            }
            data = self.safe_request(url, headers=self.headers, params=params)
            universe.extend([(c["id"], c["symbol"], c["name"]) for c in data])
            if len(universe) >= n:
                break
        return universe[:n]

    def fetch_history(self, asset_id: str, vs: str = "usd", days_back: int = 1460) -> pd.DataFrame:
        url = f"{self.base_url}/coins/{asset_id}/market_chart"
        params = {"vs_currency": vs, "days": days_back, "interval": "daily"}
        data = self.safe_request(url, headers=self.headers, params=params)

        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        caps   = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        vols   = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

        df = prices.merge(caps, on="timestamp").merge(vols, on="timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.drop(columns=["timestamp"]).set_index("date")

        # Extra: supply & rank snapshot (retry-safe)
        try:
            url_coin = f"{self.base_url}/coins/{asset_id}"
            snap = self.safe_request(url_coin, headers=self.headers, params={"localization": "false"})
            md = snap.get("market_data", {})
            df["circulating_supply"] = md.get("circulating_supply")
            df["total_supply"] = md.get("total_supply")
            df["max_supply"] = md.get("max_supply")
            df["market_cap_rank"] = snap.get("market_cap_rank")
        except Exception as e:
            print(f"  no supply snapshot for {asset_id}: {e}")

        return df

    def fetch_and_save_all(self, n: int = 500, days_back: int = 1460) -> List[str]:
        coins = self.fetch_top_universe(n)
        paths = []
        for cid, sym, name in coins:
            outpath = os.path.join(self.outdir, f"{sym.lower()}_{cid}.csv")
            if os.path.exists(outpath):
                print(f"⏭️ Skipping {name} ({sym}) – already exists.")
                paths.append(outpath)
                continue
            try:
                print(f"⬇️ Fetching {name} ({sym})...")
                df = self.fetch_history(cid, days_back=days_back)
                df["symbol"] = sym.upper()        # add symbol column
                df.to_csv(outpath, index=True)    # keep date as index or reset as needed
                print(f"  ✅ saved {len(df)} rows to {outpath}")
                paths.append(outpath)
                time.sleep(self.cfg.cg_sleep)
            except Exception as e:
                print(f"❌ Failed {name}: {e}")
        return paths

    def build_panel(self, files: List[str]) -> pd.DataFrame:
        dfs = []
        for f in files:
            sym = os.path.basename(f).split("_")[0]
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            df = df.rename(
                columns={
                    "price": f"{sym}_price",
                    "market_cap": f"{sym}_mcap",
                    "volume": f"{sym}_vol",
                    "open": f"{sym}_open",
                    "high": f"{sym}_high",
                    "low": f"{sym}_low",
                    "close": f"{sym}_close",
                }
            )
            dfs.append(df)
        return pd.concat(dfs, axis=1).sort_index()

    def universe_to_binance_symbols(self, universe):
        """
        Convert CoinGecko universe into Binance USDT pairs,
        keeping only those actually tradable on Binance.
        """
        exchange = ccxt.binance()
        all_markets = exchange.load_markets()
        valid = []
        for _, sym, _ in universe:
            s = f"{sym.upper()}/USDT"
            if s in all_markets:
                valid.append(s)
            else:
                print(f"⚠️ Skipping {s}, not on Binance")
        return valid

    def safe_request(self, url, headers, params=None, max_retries=5, backoff=2):
        """
        Resilient request wrapper with retries + exponential backoff.
        """
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=20)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                wait = backoff * (2 ** attempt)
                print(f"⚠️ Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"❌ Failed after {max_retries} retries: {url}")
