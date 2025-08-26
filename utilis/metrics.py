
import numpy as np
import pandas as pd

def calculate_performance_metrics(returns, annualization_factor=252):
    """
    Calculates key performance metrics for a series of strategy returns.

    Args:
        returns (pd.Series): A pandas Series of periodic returns.
        annualization_factor (int): The number of periods in a year 
                                    (e.g., 252 for daily, 252*8 for hourly).

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    
    # --- Total Return ---
    total_return = (1 + returns).prod() - 1
    
    # --- Sharpe Ratio ---
    # Calculates the annualized Sharpe ratio
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Handle the case where standard deviation is zero
    if std_return == 0:
        sharpe_ratio = np.inf if mean_return > 0 else -np.inf
    else:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(annualization_factor)
        
    # --- Maximum Drawdown ---
    # Calculates the largest peak-to-trough decline
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Maximum Drawdown": f"{max_drawdown:.2%}"
    }
    
    return metrics
