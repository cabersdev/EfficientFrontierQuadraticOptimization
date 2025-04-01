from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd 
from pydantic import BaseModel, ConfigDict, Extra, HttpUrl, field_validator
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from src.utils.helpers import load_config
from src.utils.logger import setup_logger
from src.data_pipelines.data_pipelines import execute_data_pipeline

logger = setup_logger(name=__name__)

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
    
class OptimizationConfig(BaseModel):
    min_weight: float = 0.05
    max_weight: float = 0.3
    target_return: Dict[str, float] = {'min': 0.005, 'max': 0.015, 'step': 20}
    risk_free_rate: float = 0.02

    @field_validator('min_weight', 'max_weight')
    @classmethod
    def validate_weights(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("Pesi non validi")
        return value
    
    @field_validator('target_return')
    @classmethod
    def validate_return(cls, value: Dict[str, float]) -> Dict[str, float]:
        if value['min'] > value['max']:
            raise ValueError("Target di ritorno non validi")
        return value
    
    @field_validator('risk_free_rate')
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Tasso di rendimento privo di rischio non valido")
        return value
    
class ModelConfig(BaseModel):
    covariance: CovarianceConfig
    optimization: OptimizationConfig
    model_config = ConfigDict(extra=Extra.forbid)
    
class MarkowitzOptimizer:
    def __init__(self, returns: pd.DataFrame, config_path: Path = Path('parameters/optimizer_parameters.yaml')):
        self.returns = returns 
        self.config = self._load_config(config_path)
        self.expected_returns = self.returns.mean()
        self.cov_matrix = self._calculate_covariance()
        self._validate_inputs()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        raw_config = load_config(config_path)
        return ModelConfig(**raw_config)

    def _calculate_covariance(self) -> pd.DataFrame:
        if self.config.covariance.method == 'ledoit-wolf':
            lw = LedoitWolf(
                assume_centered=True,
                block_size=100,
                shrinkage=self.config.covariance.shrinkage_target
            )
            lw.fit(self.returns)
            return pd.DataFrame(lw.covariance_, index=self.returns.columns, columns=self.returns.columns)
        return self.returns.cov()
    
    def _validate_inputs(self):
        if self.cov_matrix.shape != (len(self.returns.columns), len(self.returns.columns)):
            raise ValueError("Matrice di covarianza non valida")
        if not np.allclose(self.cov_matrix, self.cov_matrix.T):
            raise ValueError("Matrice di covarianza non simmetrica")
        
    def _portfolio_return(self, weights: np.array) -> float:
        return np.dot(weights, self.expected_returns)
    
    def _portfolio_volatility(self, weights: np.array) -> float:
        return np.sqrt(weights.T @ self.cov_matrix @ weights)
    
    def efficient_frontier(self) -> List[Dict[str, Any]]:
        targets = np.linspace(
            self.config.optimization.target_return['min'],
            self.config.optimization.target_return['max'],
            self.config.optimization.target_return['step']
        )

        frontier = []
        for target in targets:
            results = self._optimize(target)
            if results['success']:
                frontier.append({
                    'weights': results['w'],
                    'return': target,
                    'volatility': results['fun']
                })
        return frontier
    
    def _optimize(self, target_return: float) -> Dict[str, Any]:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_return}
        ]

        bounds = [(self.config.optimization.min_weight, self.config.optimization.max_weight)] * len(self.returns.columns)

        result = minimize(
            self._portfolio_volatility,
            x0 = np.array([1/len(self.returns.columns)] * len(self.returns.columns)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return {
            'success': result.success,
            'w': result.w,
            'fun': result.fun
        }

        def max_sharpe_ratio(self) -> Dict[str, Any]:
            def negative_sharpe(w):
                ret = self._portfolio_return(w)
                vol = self._portfolio_volatility(w)
                return - (ret - self.config.optimization.risk_free_rate) / vol
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(self.config.optimization.min_weight, self.config.optimization.max_weight)] * len(self.returns.columns)

            result = minimize(
                negative_sharpe,
                x0 = np.array([1/len(self.returns.columns)] * len(self.returns.columns)),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            return {
                'weights': result.x,
                'return': self._portfolio_return(result.x),
                'volatility': self._portfolio_volatility(result.x),
                'sharpe_ratio': -result.fun
            }