from pathlib import Path
import pandas as pd
from typing import Dict
import numpy as np
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.required_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.date_range = {
            'start': pd.to_datetime(config['start_date']),
            'end': pd.to_datetime(config['end_date'])
        }

        
    def validate_raw_data(self, df: pd.DataFrame) -> bool:
        """Validazione dati grezzi"""
        try:
            # 1. Presenza colonne obbligatorie
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colonne mancanti: {missing_cols}")
            
            # 2. Controllo valori nulli
            if df[self.required_columns].isnull().any().any():
                null_counts = df.isnull().sum()
                raise ValueError(f"Valori nulli presenti:\n{null_counts}")
            
            # 3. Validazione intervallo date
            date_check = df['Date'].between(self.date_range['start'], self.date_range['end'])
            if not date_check.all():
                invalid_dates = df[~date_check]['Date']
                raise ValueError(f"Date non valide presenti: {invalid_dates.unique()}")
            
            # 4. Controllo valori positivi
            price_cols = ['Open', 'High', 'Low', 'Close']
            if (df[price_cols] <= 0).any().any():
                raise ValueError("Prezzi devono essere > 0")
            
            # 5. Validazione volume
            if (df['Volume'] < 0).any():
                raise ValueError("Volume non può essere negativo")
            
            logger.info("Validazione raw data completata con successo")
            return True
            
        except Exception as e:
            logger.error(f"Validazione fallita: {str(e)}")
            raise

    def validate_clean_data(self, df: pd.DataFrame) -> bool:
        """Validazione dati puliti"""
        try:
            # 1. Controllo consistenza tickers
            expected_tickers = set(self.config['tickers'])
            actual_tickers = set(df['Ticker'].unique())
            
            if expected_tickers != actual_tickers:
                missing = expected_tickers - actual_tickers
                extra = actual_tickers - expected_tickers
                raise ValueError(f"Tickers mismatch. Mancanti: {missing}, Extra: {extra}")
            
            # 2. Controllo completezza dati
            days_in_range = (self.date_range['end'] - self.date_range['start']).days + 1
            for ticker in expected_tickers:
                ticker_data = df[df['Ticker'] == ticker]
                if len(ticker_data) < days_in_range * 0.9:  # Almeno 90% dei dati
                    raise ValueError(f"{ticker} ha solo {len(ticker_data)} records")
            
            logger.info("Validazione clean data completata con successo")
            return True
            
        except Exception as e:
            logger.error(f"Validazione clean data fallita: {str(e)}")
            raise