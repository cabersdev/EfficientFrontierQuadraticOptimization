from typing import Any, Dict, List, Optional
import retry
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from pydantic import BaseModel, HttpUrl, field_validator
from datetime import datetime
from src.utils.helpers import load_config
from src.utils.logger import setup_logger

# Configurazione logger
logger = setup_logger(name=__name__)

class DataConfig(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    interval: str = '1d'
    auto_adjust: bool = True
    threads: int = 5
    max_retries: int = 3
    backoff: int = 2
    path_raw: Path = Path('data/raw')
    fields: list[str] = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, value: str) -> str:
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return value
        except ValueError:
            raise ValueError(f"Formato data non valido: {value}. Usare 'YYYY-MM-DD'")

# Caricamento e validazione configurazione
config = load_config('parameters/data_parameters.yaml')
params = DataConfig(**config)

@retry.retry(tries=params.max_retries, delay=params.backoff, logger=logger)
def fetch_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch dati storici con gestione errori avanzata"""
    try:
        logger.info(f"Downloading data for {ticker}...")
        
        data = yf.download(
            tickers=ticker,
            start=datetime.strptime(params.start_date, '%Y-%m-%d').strftime('%Y-%m-%d'),
            end=datetime.strptime(params.end_date, '%Y-%m-%d').strftime('%Y-%m-%d'),
            interval=params.interval,
            progress=False,
            auto_adjust=params.auto_adjust,
            threads=True
        )

        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None

        # Pulizia e formattazione dati
        data = data.reset_index()[params.fields]
        data['Ticker'] = ticker  # Aggiunge colonna ticker
        
        # Validazione dati
        if data.isnull().values.any():
            logger.error(f"Missing values detected in {ticker}")
            return None
            
        return data

    except yf.YFinanceError as e:
        logger.error(f"YFinance error for {ticker}: {str(e)}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching {ticker}")
        return None

def save_data_parquet(data: pd.DataFrame, ticker: str) -> bool:
    """Salva i dati in formato parquet con compressione"""
    try:
        params.path_raw.mkdir(parents=True, exist_ok=True)
        output_file = params.path_raw / f"{ticker}.parquet"
        
        data.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        logger.info(f"Data saved successfully for {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error saving {ticker}: {str(e)}")
        return False
    
def save_data_csv(data: pd.DataFrame, ticker: str) -> bool:
    """Salva i dati in formato CSV"""
    try:
        params.path_raw.mkdir(parents=True, exist_ok=True)
        output_file = params.path_raw / f"{ticker}.csv"
        
        data.to_csv(output_file, index=False)
        
        logger.info(f"Data saved successfully for {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error saving {ticker}: {str(e)}")
        return False

def process_ticker(ticker: str) -> bool:
    """Pipeline completa per un singolo ticker"""
    data = fetch_data(ticker)
    if data is not None:
        return save_data_parquet(data, ticker) and save_data_csv(data, ticker)
    return False

def main():
    """Esecuzione parallela con ThreadPool"""
    logger.info("Starting data pipeline...")
    
    with ThreadPoolExecutor(max_workers=params.threads) as executor:
        results = executor.map(process_ticker, params.tickers)
    
    success_rate = sum(results) / len(params.tickers)
    logger.info(f"Pipeline completed. Success rate: {success_rate:.2%}")

if __name__ == '__main__':
    main()