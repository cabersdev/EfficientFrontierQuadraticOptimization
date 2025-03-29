import logging
import logging.config
import yaml
from pathlib import Path

def setup_logger(name: str = __name__, 
                config_path: Path = Path('config/logging.yaml'),
                default_level=logging.INFO):
    """Configura il logger da file YAML"""
    try:
        config_path = Path(config_path) 
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
            logging.warning(f"File di configurazione {config_path} non trovato")
        
        logger = logging.getLogger(name)
        logger.info(f"Logger configurato per il modulo {name}")
        return logger

    except Exception as e:
        logging.basicConfig(level=default_level)
        logger = logging.getLogger(name)
        logger.error(f"Errore configurazione logger: {str(e)}")
        return logger