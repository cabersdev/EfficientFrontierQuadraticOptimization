from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd 
from pydantic import BaseModel, ConfigDict, Extra, HttpUrl, field_validator
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize, Bounds
from utils.helpers import load_config
from utils.logger import setup_logger
from pathlib import Path #aggiunta per pipeline
from pipeline import run_pipeline #aggiunta per pipeline

logger = setup_logger(name=__name__)

class CovarianceConfig(BaseModel):
    method: str = 'ledoit-wolf'
    shrinkage: float | None = None
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
    def __init__(self, returns: pd.DataFrame, config_path: Path = Path('parameters/model_parameters.yaml')):
        self.returns = returns 
        self.config = self._load_config(config_path)
        self.expected_returns = self.returns.mean()
        self.cov_matrix = self._calculate_covariance()
        self._validate_inputs()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        raw_config = load_config(config_path)
        logger.info(f"Configurazione caricata da: {config_path}")
        return ModelConfig(**raw_config)

    def _calculate_covariance(self) -> pd.DataFrame:
        logger.info("Stima della matrice di covarianza")
        if self.config.covariance.method == 'ledoit-wolf':
            lw = LedoitWolf(
                assume_centered=False,
                block_size=1000,
                #shrinkage_=self.config.covariance.shrinkage
                #shrinkage=self.config.covariance.shrinkage_target
            )

            lw.fit(self.returns)
            logger.info("Matrice di covarianza stimata con Ledoit-Wolf")
            return pd.DataFrame(lw.covariance_, index=self.returns.columns, columns=self.returns.columns)
        return self.returns.cov()
    
    def _validate_inputs(self):
        logger.info("Validazione dei dati di input")
        if self.cov_matrix.shape != (len(self.returns.columns), len(self.returns.columns)):
            raise ValueError("Matrice di covarianza non valida")
        if not np.allclose(self.cov_matrix, self.cov_matrix.T):
            raise ValueError("Matrice di covarianza non simmetrica")
        
    def _portfolio_return(self, weights: np.array) -> float:
        logger.info("Calcolo del rendimento atteso del portafoglio")
        return np.dot(weights, self.expected_returns)
    
    def _portfolio_volatility(self, weights: np.array) -> float:
        logger.info("Portafoglio di volatilità calcolata")
        return np.sqrt(weights.T @ self.cov_matrix @ weights)
    
    def efficient_frontier(self) -> List[Dict[str, Any]]:
        logger.info("Calcolo della frontiera efficiente")
        targets = np.linspace(
            self.config.optimization.target_return['min'],
            self.config.optimization.target_return['max'],
            self.config.optimization.target_return['step']
        )
        logger.info(f"Target di ritorno: {targets}")

        frontier = []
        for target in targets:
            results = self._optimize(target)
            logger.info(f"Risultato dell'ottimizzazione {results}")
            if results:
                logger.info(f"Ottimizzazione riuscita per target di ritorno: {target}")
                frontier.append({
                    'weights': results['w'],
                    'return': target,
                    'volatility': results['fun']
                })
                logger.info(f"Ottimizzazione riuscita per target di ritorno: {target}")
            else:
                logger.warning(f"Ottimizzazione fallita per target di ritorno: {target}")
        return frontier
    
    def _optimize(self, target_return: float) -> Dict[str, Any]:
        logger.info(f"Ottimizzazione per target di ritorno: {target_return}")
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_return},
        ]

        initial_guess = self._min_varance_portfolio()

        bounds = Bounds(
            self.config.optimization.min_weight,
            self.config.optimization.max_weight,
            keep_feasible=True,
        )

        result = minimize(
            self._portfolio_volatility,
            x0 = np.random.dirichlet(np.ones(len(self.returns.columns)), size=1).flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'verbose': 0}
        )
        logger.info(f"Ottimizzazione completata: {result.success}")


        return {
            'success': result.success,
            'w': result.x,
            'fun': result.fun
        }
    def _min_varance_portfolio(self) -> np.array:
        logger.info("Calcolo del portafoglio a minima varianza")
        n = len(self.returns.columns)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(
            self._portfolio_volatility,
            x0 = np.ones(n) / n,
            method='SLSQP',
            bounds=[(0, 1)] * n,
            constraints=constraints
        )
        return result.x

    def max_sharpe_ratio(self) -> Dict[str, Any]:
        logger.info("Calcolo del massimo Sharpe Ratio")
        def negative_sharpe(w):
            ret = self._portfolio_return(w)
            vol = self._portfolio_volatility(w)
            logger.info(f"Rendimento: {ret}, Volatilità: {vol}")
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
        logger.info(f"Massimo Sharpe Ratio calcolato: {-result.fun}")
        return {
            'weights': result.x,
            'return': self._portfolio_return(result.x),
            'volatility': self._portfolio_volatility(result.x),
            'sharpe_ratio': -result.fun
        }
    
    if __name__ == "__main__":
    # Parametri configurabili per la pipeline e il modello
    PIPELINE_PARAMS = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'input_dir': "../data/raw",
        'output_dir': "../data/processed"
    }
    
    # 1. Esegui la pipeline di dati
    logger.info("Avvio della pipeline di dati...")
    try:
        run_pipeline(**PIPELINE_PARAMS)
        logger.info("Pipeline di dati completata con successo")
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione della pipeline: {str(e)}")
        raise

    # 2. Carica i dati puliti
    output_path = Path(PIPELINE_PARAMS['output_dir']) / "cleaned_stocks.parquet"
    try:
        cleaned_data = pd.read_parquet(output_path)
        logger.info(f"Dati puliti caricati da {output_path}")
    except Exception as e:
        logger.error(f"Errore nel caricamento dei dati puliti: {str(e)}")
        raise

    # 3. Calcola i rendimenti
    returns = cleaned_data.pct_change().dropna()
    if returns.empty:
        logger.error("Nessun dato disponibile dopo il calcolo dei rendimenti")
        raise ValueError("Dataset dei rendimenti vuoto")

    # 4. Esegui l'ottimizzazione
    try:
        optimizer = MarkowitzOptimizer(returns)
        logger.info("Modello Markowitz inizializzato correttamente")
        
        # Esempio: Calcola frontiera efficiente e Sharpe Ratio
        frontier = optimizer.efficient_frontier()
        max_sharpe = optimizer.max_sharpe_ratio()
        
        # Esempio: Salva risultati
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        pd.DataFrame([max_sharpe]).to_csv(results_dir / "optimal_portfolio.csv")
        logger.info(f"Risultati salvati in {results_dir}")
        
    except Exception as e:
        logger.error(f"Errore durante l'ottimizzazione: {str(e)}")
        raise
    