import pandas as pd
import pathlib as pa
from typing import List, Dict, Optional
from logging import logger
from data_cleaner import DataCleaner
from data_validation import DataValidator
from utils.logger import setup_logger

setup_logger(name=__name__)

# 1. Data Loader (già visto)
class DataLoader:
    def __init__(self, data_path: str, tickers: List[str]):
        self.data_path = pa.Path(data_path)
        self.tickers = tickers

    def load(self) -> pd.DataFrame:
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
                logger.info(f"Caricato {ticker}")
            except FileNotFoundError:
                logger.warning(f"File {ticker}.csv non trovato!")
        return pd.concat(dfs, axis=1).ffill().dropna()


# 4. Data Exporter
class DataExporter:
    @staticmethod
    def to_csv(df: pd.DataFrame, path: str):
        df.to_csv(path)
        logger.info(f"Dati esportati in {path}")

    @staticmethod
    def to_parquet(df: pd.DataFrame, path: str):
        df.to_parquet(path)
        logger.info(f"Dati esportati in {path}")

# 5. Pipeline Coordinata
def run_pipeline(tickers: List[str], input_dir: str, output_dir: str):
    """Esegue: Caricamento → Pulizia → Validazione → Export."""
    # Step 1: Load
    loader = DataCleaner._load_data(input_dir, tickers)
    data = loader.load()

    # Step 2: Clean
    cleaned_data = (
        DataCleaner(data)
        .remove_outliers(threshold=3.0)
        .normalize(method='minmax') #oppure Z-score
        .get_data()
    )

    # Step 3: Validate
    validator = DataValidator(cleaned_data)
    if not validator.run_checks():
        raise ValueError("Validazione fallita. Interrompo la pipeline.")

    # Step 4: Export
    output_path = pa.Path(output_dir) / "cleaned_stocks.parquet"
    DataExporter.to_parquet(cleaned_data, output_path)

# Esecuzione
if __name__ == "__main__":
    run_pipeline(
        tickers=['AAPL', 'GOOGL', 'MSFT'],
        input_dir="../data/raw",
        output_dir="../data/processed"
    )