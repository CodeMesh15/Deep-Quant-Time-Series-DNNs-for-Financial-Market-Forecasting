
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .metrics import calculate_performance_metrics

def run_backtest(predictions_df, transaction_cost=0.001, annualization_factor=252):
    """
    Runs a vectorized backtest for a given set of predictions.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing 'Close' prices and model 'predictions'.
        transaction_cost (float): The cost per trade as a percentage.
        annualization_factor (int): The number of trading periods in a year.
    """
    print("\n--- Running Backtest ---")
    
    # --- 1. Calculate Market Returns (Buy and Hold Strategy) ---
    predictions_df['market_returns'] = predictions_df['Close'].pct_change()

    # --- 2. Create Trading Positions ---
    # A simple strategy: if prediction is 1 (up), go long (position=1). 
    # If prediction is 0 (down), go flat (position=0).
    # We shift positions by 1 to ensure we use the signal from period t to trade in period t+1.
    predictions_df['position'] = predictions_df['predictions'].shift(1).fillna(0)

    # --- 3. Calculate Strategy Returns ---
    # The return is the market return multiplied by our position.
    predictions_df['strategy_returns'] = predictions_df['market_returns'] * predictions_df['position']

    # --- 4. Account for Transaction Costs ---
    # A cost is incurred whenever our position changes.
    position_changes = predictions_df['position'].diff().abs()
    predictions_df['transaction_costs'] = position_changes * transaction_cost
    predictions_df['strategy_returns'] -= predictions_df['transaction_costs']
    
    # Drop NaNs created by pct_change and shifts
    predictions_df.dropna(inplace=True)

    # --- 5. Performance Analysis ---
    print("\nPerformance Metrics:")
    metrics = calculate_performance_metrics(predictions_df['strategy_returns'], annualization_factor)
    for key, value in metrics.items():
        print(f"{key}: {value}")
        
    # --- 6. Plotting the Equity Curve ---
    print("\nPlotting equity curve...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate cumulative returns for both strategies
    cumulative_strategy_returns = (1 + predictions_df['strategy_returns']).cumprod()
    cumulative_market_returns = (1 + predictions_df['market_returns']).cumprod()
    
    cumulative_strategy_returns.plot(ax=ax, label='Model Strategy')
    cumulative_market_returns.plot(ax=ax, label='Buy and Hold')
    
    ax.set_title('Strategy Performance vs. Buy and Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    plt.show()

    return predictions_df
