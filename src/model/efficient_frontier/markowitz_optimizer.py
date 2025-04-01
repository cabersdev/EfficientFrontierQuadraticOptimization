from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd 
import scipy 
import yaml
import cvxpy
from pydantic import BaseModel, HttpUrl, field_validator
from sklearn.covariance import LedoitWolf
from src.utils.helpers import load_config
from src.utils.logger import setup_logger
from src.data_pipelines.data_pipelines import execute_data_pipeline

# Configurazione logger
logger = setup_logger(name=__name__)

# Caricamento e validazione configurazione
config = load_config('parameters/data_parameters.yaml')
params = MarkowitzOptimizer(**config)

class CovarianceConfig(BaseModel):
    method: str = 'ledoit-wolf'
    shrinkage_target: str = 'constant_variance'

    @field_validator('method')
    @classmethod
    def validate_method(cls, value: str) -> str:
        if value not in ['ledoit-wolf', 'empirical']:
            raise ValueError("Metodo di stima della matrice di covarianza non valido")
        return value

    @field_validator('shrinkage_target')
    @classmethod
    def validate_target(cls, value: str) -> str:
        if value not in ['constant_variance', 'single_factor', 'constant_correlation']:
            raise ValueError("Target di shrinkage non valido")
        return value
    
