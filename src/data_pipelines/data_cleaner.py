import pandas as pd
import numpy as np
import pathlib as pa
from typing import List, Optional, Union

class DataCleaner:
    def __init__(self, data_path: Union[str, pa.Path], tickers: List[str]):
        """Inizializzazione del DataFrame"""
        self.data_path = pa.Path(data_path)
        self.tickers = tickers
        self.prices = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load stock data from CSV files and combine into a single DataFrame."""
        dfs = []
        for ticker in self.tickers:
            try:
                df = pd.read_csv(
                    self.data_path / f"{ticker}.csv",
                    parse_dates=['Date'],
                    usecols=['Date', 'Close'],
                    index_col='Date'
                )
                df.columns = [ticker]
                dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: File for {ticker} not found in {self.data_path}.")
        
        if not dfs:
            raise ValueError("No valid stock data found.")
        
        combined = pd.concat(dfs, axis=1)
        return combined.ffill().dropna()
    
    def handle_missing_values(self, method: str = 'ffill') -> pd.DataFrame:
        """Gestisce i valori mancanti usando il ffill met"""
        if method == 'ffill':
            self.prices = self.prices.ffill().dropna()
        elif method == 'bfill':
            self.prices = self.prices.bfill().dropna()
        elif method == 'interpolate':
            self.prices = self.prices.interpolate().dropna()
        else:
            raise ValueError("Method must be 'ffill', 'bfill', or 'interpolate'.")
        
        return self.prices
    
    def remove_outliers(self, threshold: float = 3.0) -> pd.DataFrame:
        """Rimuove i valori outliers usando il Z-score method"""
        z_scores = (self.prices - self.prices.mean()) / self.prices.std()
        self.prices = self.prices[(z_scores.abs() < threshold).all(axis=1)]
        return self.prices
    
    #le 2 funzioni normalize e compute, deepseek consiglia di metterle in quanto molto utili per dati azionari
    def normalize_data(self, method: str = 'minmax') -> pd.DataFrame:
        """Normalizza i prezzi usando min-max oppure utilizza il Z-score method."""
        if method == 'minmax':
            self.prices = (self.prices - self.prices.min()) / (self.prices.max() - self.prices.min())
        elif method == 'zscore':
            self.prices = (self.prices - self.prices.mean()) / self.prices.std()
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'.")
        
        return self.prices
    
    def compute_returns(self, log_returns: bool = False) -> pd.DataFrame:
        """Trasforma i dati, in valori logaritmici"""
        if log_returns:
            returns = np.log(self.prices / self.prices.shift(1))
        else:
            returns = self.prices.pct_change()
        
        return returns.dropna()
    
    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned and processed DataFrame."""
        return self.prices.copy()   


cleaner = DataCleaner(data_path='../data/raw', tickers=['AAPL', 'GOOGL', 'MSFT'])
prices = cleaner.get_clean_data()  # Ottieni i dati puliti
prices.head(10)



