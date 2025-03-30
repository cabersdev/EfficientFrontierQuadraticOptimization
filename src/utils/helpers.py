from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    """Carica configurazione con gestione avanzata dei percorsi"""
    config_path = Path(config_path)
    
    # Se il percorso è relativo, risali fino alla root del progetto
    if not config_path.is_absolute():
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)