# crypto_factor_lens/data/downloader.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class DataDownloader(ABC):
    """
    Abstract base class for all data downloaders.
    """

    @abstractmethod
    def fetch_top_universe(self, n: int = 500) -> list:
        """
        Return a list of assets (ids, symbols, names) in the top-N universe.
        """
        pass

    @abstractmethod
    def fetch_history(self, asset_id: str, **kwargs) -> pd.DataFrame:
        """
        Fetch historical daily data for a single asset.
        Must return a DataFrame indexed by date.
        """
        pass

    @abstractmethod
    def fetch_and_save_all(self, n: int = 500, **kwargs) -> List[str]:
        """
        Fetch history for the top-N universe and save to disk.
        Return list of file paths.
        """
        pass

    @abstractmethod
    def build_panel(self, files: List[str]) -> pd.DataFrame:
        """
        Combine multiple assets into a panel DataFrame.
        """
        pass
