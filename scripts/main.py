from pathlib import Path
import pandas as pd
import numpy as np
from src.model.efficient_frontier.markowitz_optimizer import MarkowitzOptimizer
from src.model.postprocessing.visualizer import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger(name=__name__)

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def main():
    data_path = Path('data/raw')
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        prices = ""
        returns = calculate_returns(prices)

        optimizer = MarkowitzOptimizer(returns)

        visualizer = Visualizer(optimizer)

        visualizer.plot_efficient_frontier(output_path=output_dir / 'efficient_frontier.png')

        sharpes = optimizer.max_sharpe_ratio()
        visualizer.plot_weights_distribution(
            weights=dict(zip(returns.columns, sharpes['weights'])),
            output_path=output_dir / 'allocazione_pesi.png'
        )
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione dello script: {e}")

if __name__ == '__main__':
    main()
