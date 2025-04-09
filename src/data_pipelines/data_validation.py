import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from data_cleaner import DataCleaner

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.errors = []

    def check_missing_values(self) -> bool:
        """Controlla se ci sono valori nulli dopo la pulizia."""
        if self.df.isnull().any().any():
            self.errors.append("ERRORE: Sono presenti valori nulli nel DataFrame.")
            return False
        return True

    def check_duplicate_dates(self) -> bool:
        """Verifica date duplicate nell'indice."""
        if self.df.index.duplicated().any():
            self.errors.append("ERRORE: Date duplicate nell'indice.")
            return False
        return True

    def check_negative_prices(self) -> bool:
        """Controlla prezzi negativi (non validi per azioni)."""
        if (self.df < 0).any().any():
            self.errors.append("ERRORE: Prezzi negativi rilevati.")
            return False
        return True

    def check_date_range(self, start_date: str, end_date: str) -> bool:
        """Verifica se il DataFrame copre l'intervallo di date richiesto."""
        if self.df.index.min() > pd.to_datetime(start_date) or self.df.index.max() < pd.to_datetime(end_date):
            self.errors.append(f"ERRORE: Dati mancanti per l'intervallo {start_date} - {end_date}.")
            return False
        return True

    def validate(self, checks: Optional[List[str]] = None) -> Dict[str, bool]:
        """Esegue tutti i controlli e restituisce un report."""
        if checks is None:
            checks = ['missing', 'duplicates', 'negative_prices']

        results = {}
        if 'missing' in checks:
            results['missing_values'] = self.check_missing_values()
        if 'duplicates' in checks:
            results['duplicate_dates'] = self.check_duplicate_dates()
        if 'negative_prices' in checks:
            results['negative_prices'] = self.check_negative_prices()
        if 'date_range' in checks:
            results['date_range'] = self.check_date_range()
        
        if self.errors:
            print("\n".join(self.errors))
        else:
            print("Tutti i controlli superati.")
        
        return results


cleaner = DataCleaner(data_path='../data/raw', tickers=['AAPL', 'GOOGL', 'MSFT'])
cleaned_data = cleaner.get_clean_data()

validator = DataValidator(cleaned_data)
validation_report = validator.validate(checks=['missing', 'duplicates', 'negative_prices'])

print("\nReport di validazione:")
print(validation_report)