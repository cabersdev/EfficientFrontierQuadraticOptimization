import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(name=__name__)
class Visualizer:
    
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def plot_efficient_frontier(
        self, 
        output_path: Optional[Path] = None,
        figsize: tuple = (10, 6),
        dpi: int = 100
    ) -> None:
        frontier_data = self.optimizer.efficient_frontier()
        if not frontier_data:
            raise ValueError("Nessun dato disponibile per la frontiera efficiente")

        sharpe_data = self.optimizer.max_sharpe_ratio()

        returns = [p['return'] for p in frontier_data]
        volatilities = [p['volatility'] for p in frontier_data]

        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.scatter(
            volatilities, 
            returns, 
            c='blue', 
            alpha=0.7,
            label='Frontiera Efficiente'
        )

        plt.scatter(
            sharpe_data['volatility'],
            sharpe_data['return'],
            c='red',
            marker='*',
            s=400,
            label='Max Sharpe Ratio'
        )

        rf = self.optimizer.config.optimization.risk_free_rate
        plt.plot(
            [0, sharpe_data['volatility']],
            [rf, sharpe_data['return']],
            'k--',
            label='Capital Market Line'
        )

        plt.title('Frontiera Efficiente e Capital Market Line')
        plt.xlabel('Volatilità (Deviazione Standard)')
        plt.ylabel('Ritorno Atteso')
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Grafico salvato in: {output_path}")
        else:
            plt.show()

    def plot_weights_distribution(
        self, 
        weights: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> None:
        
        plt.figure(figsize=(10, 4))
        
        assets = list(weights.keys())
        values = list(weights.values())

        plt.bar(assets, values)
        plt.title('Distribuzione Pesi Portafoglio')
        plt.xlabel('Asset')
        plt.ylabel('Peso')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()