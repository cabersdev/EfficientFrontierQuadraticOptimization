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
from requests.exceptions import RequestException
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
    format: str = 'csv'
    strict_validation: bool = True

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
            start=params.start_date,
            end=params.end_date,
            interval=params.interval,
            progress=False,
            auto_adjust=params.auto_adjust,
            group_by='ticker'  # Modifica cruciale
        )

        # Gestione MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)  # Prendi il nome del ticker
        else:
            data.columns = [f"{col}_{ticker}" for col in data.columns]

        # Reset e pulizia
        data = data.reset_index()
        data.columns = data.columns.str.title()
        
        # Aggiungi colonna ticker
        data['Ticker'] = ticker
        
        # Rinomina colonne chiave
        data = data.rename(columns={
            f'Open_{ticker}': 'Open',
            f'High_{ticker}': 'High',
            f'Low_{ticker}': 'Low',
            f'Close_{ticker}': 'Close',
            f'Volume_{ticker}': 'Volume'
        })[params.fields + ['Ticker']]

        # Validazione tipi dati
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return data.dropna()

    except Exception as e:
        logger.error(f"Errore su {ticker}: {str(e)}")
        return None
    
def validate_data(data: pd.DataFrame) -> bool:
    """Validazione dati"""
    required_columns = {
        'Date': 'datetime64[ns]',
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Volume': 'int64',
        'Ticker': 'object'
    }

    if not all(col in data.columns for col in required_columns.keys()):
        logger.error("Missing columns in data")
        return False
    
    for col, dtype in required_columns.items():
        if not np.issubdtype(data[col].dtype, np.dtype(dtype)):
            logger.error(f"Invalid dtype for column {col}")
            return False
    
    return True

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
        
        # Formattazione per le date
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        data.to_csv(
            output_file,
            index=False,
            encoding='utf-8',
            date_format='%Y-%m-%d'
        )
        
        logger.info(f"Data saved successfully for {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error saving {ticker}: {str(e)}")
        return False

def process_ticker(ticker: str) -> bool:
    """Pipeline completa per un singolo ticker"""
    try:
        data = fetch_data(ticker)

        if data is None:
            return False
        
        if params.strict_validation and not validate_data(data):
            logger.error(f"Data validation failed for {ticker}")
            return False
        
        # Conversione 
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)

        if params.format == 'csv':
            return save_data_csv(data, ticker)
        elif params.format == 'parquet':
            return save_data_parquet(data, ticker)
        else:
            logger.error(f"Invalid format: {params.format}")
            return False
    
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
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