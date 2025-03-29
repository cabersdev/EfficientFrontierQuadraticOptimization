import logging
import logging.config
import yaml
from pathlib import Path

def setup_logger(config_path: Path = Path('config/logging.yaml'), 
                default_level=logging.INFO):
    """Logger configutrator"""
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Logger correctly configured")
    return logger