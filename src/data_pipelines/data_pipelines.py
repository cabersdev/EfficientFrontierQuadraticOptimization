from pathlib import Path
from typing import Dict, Any
import pandas as pd
from .data_cleaner import load_parquet_data, clean_financial_data, save_clean_data
from .data_validation import DataValidator
from src.utils.logger import setup_logger
from src.utils.helpers import load_config

logger = setup_logger(__name__)

class DataPipeline:
    def __init__(self, config_path: Path = Path('parameters/data_parameters.yaml')):
        self.config = load_config(config_path)['data']
        self.raw_path = Path(self.config['raw_path'])
        self.processed_path = Path(self.config['processed_path'])
        self.validator = DataValidator(self.config)

    def run_pipeline(self) -> pd.DataFrame:
        """Esegue l'intera pipeline dati"""
        try:
            logger.info("Avvio pipeline dati...")
            
            # 1. Caricamento dati grezzi
            raw_data = self._load_raw_data()
            
            # 2. Validazione dati grezzi
            self._validate_raw_stage(raw_data)
            
            # 3. Pulizia dati
            clean_data = self._clean_data(raw_data)
            
            # 4. Validazione dati puliti
            self._validate_clean_stage(clean_data)
            
            # 5. Salvataggio risultati
            self._save_results(clean_data)
            
            logger.info("Pipeline dati completata con successo")
            return clean_data

        except Exception as e:
            logger.error(f"Pipeline dati fallita: {str(e)}")
            raise

    def _load_raw_data(self) -> pd.DataFrame:
        """Carica i dati grezzi da Parquet"""
        logger.debug("Caricamento dati raw...")
        return load_parquet_data(
            self.config['tickers'],
            self.raw_path
        )

    def _validate_raw_stage(self, data: pd.DataFrame):
        """Validazione dati grezzi"""
        logger.debug("Validazione dati raw...")
        self.validator.validate_raw_data(data)

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pulizia dati"""
        logger.debug("Pulizia dati...")
        return clean_financial_data(data, self.config)

    def _validate_clean_stage(self, data: pd.DataFrame):
        """Validazione dati puliti"""
        logger.debug("Validazione dati puliti...")
        self.validator.validate_clean_data(data)

    def _save_results(self, data: pd.DataFrame):
        """Salvataggio risultati"""
        logger.debug("Salvataggio dati processati...")
        save_clean_data(data, self.processed_path)

def execute_data_pipeline():
    """Funzione principale per esecuzione esterna"""
    pipeline = DataPipeline()
    return pipeline.run_pipeline()

if __name__ == '__main__':
    execute_data_pipeline()