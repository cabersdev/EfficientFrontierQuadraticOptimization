from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from src.utils.logger import setup_logger

logger = setup_logger(name=__name__)

def load_parquet_data(tickers: List[str], raw_path: Path) -> pd.DataFrame:
    """Carica dati raw da file Parquet"""
    dfs = []
    for ticker in tickers:
        file_path = raw_path / f"{ticker}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['Ticker'] = ticker  # Mantieni traccia del ticker
            dfs.append(df)
        else:
            logger.error(f"File {file_path} non trovato")
            raise FileNotFoundError(f"Dati mancanti per {ticker}")
    
    combined_df = pd.concat(dfs, ignore_index=False)
    logger.info(f"Caricati {len(combined_df)} records da {len(tickers)} tickers")
    return combined_df

def clean_financial_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Pulizia dati finanziari"""
    try:
        # 1. Filtra colonne rilevanti
        cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols].copy()
        
        # 2. Converti date
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # 3. Rimuovi duplicati temporali
        df = df.sort_values(['Ticker', 'Date'])
        df = df.groupby('Ticker').apply(
            lambda x: x.drop_duplicates('Date', keep='first')
        )
        
        # 4. Gestione valori mancanti
        df = df.groupby('Ticker').apply(
            lambda x: x.ffill().bfill()
        )
        
        # 5. Filtra per data
        start_date = pd.to_datetime(config['start_date'])
        end_date = pd.to_datetime(config['end_date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # 6. Converti tipi dati
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        logger.info("Dati puliti correttamente")
        return df.reset_index(drop=True)
    
    except Exception as e:
        logger.error(f"Errore pulizia dati: {str(e)}")
        raise

def save_clean_data(df: pd.DataFrame, processed_path: Path) -> None:
    """Salva dati puliti in formato Parquet"""
    try:
        processed_path.mkdir(parents=True, exist_ok=True)
        path = processed_path / "cleaned_data.parquet"
        df.to_parquet(path, compression='snappy')
        logger.info(f"Dati puliti salvati in {path}")
    except Exception as e:
        logger.error(f"Errore salvataggio dati: {str(e)}")
        raise