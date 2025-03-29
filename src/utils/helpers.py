import pathlib
import yaml

def load_config(config_path: pathlib.Path) -> dict:
    """Carica configurazione da file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)