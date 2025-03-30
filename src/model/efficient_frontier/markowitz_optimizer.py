from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd 
import scipy 
import yaml
import cvxpy
import pydantic
from sklearn.covariance import LedoitWolf
from src.utils.helpers import load_config
from src.utils.logger import setup_logger
from src.data_pipelines.data_pipelines import execute_data_pipeline

# Configurazione logger
logger = setup_logger(name=__name__)


config_data = load_config('parameters/data_parameters.yaml')
config_model = load_config('parameters/model_parameters.yaml')

PARAMETERS: Dict[str, Any] = {
    'tickers': config_data['tickers'],
    'covariance': {
        'method': config_model['covariance']['method'],
        'shrinkage_target': config_model['covariance']['shrinkage_target']
    },
    'optimization': {
        'min_weight': config_model['optimization']['min_weight'],
        'max_weight': config_model['optimization']['max_weight'],
        'target_returns': np.linspace(
            config_model['optimization']['target_returns']['min'],
            config_model['optimization']['target_returns']['max'],
            config_model['optimization']['target_returns']['steps']
        ),
        'risk_free_rate': config_model['optimization']['risk_free_rate']
    }
}

processed_data = execute_data_pipeline()


