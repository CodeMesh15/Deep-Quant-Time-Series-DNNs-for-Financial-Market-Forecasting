# /data/feature_engineering.py

import pandas as pd
import numpy as np
import argparse
import os

def create_features(df):
    """
    Engineers features from the raw financial data.

    Args:
        df (pd.DataFrame): DataFrame with raw price data (Open, High, Low, Close, Volume).

    Returns:
        pd.DataFrame: DataFrame with engineered features and a target variable.
    """
    print("Starting feature engineering...")
    
    df['returns'] = df['Close'].pct_change()
    
    # --- Feature Engineering ---
    
    # 1. Lagged Returns (e.g., returns from 1, 2, 3 periods ago)
    for lag in range(1, 4):
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # 2. Moving Averages
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # 3. Volatility (rolling standard deviation of returns)
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_50'] = df['returns'].rolling(window=50).std()
    
    # 4. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- Target Variable ---
    # The target is to predict if the next period's price will go up (1) or down (0).
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows with NaN values created by rolling windows and shifts
    df.dropna(inplace=True)
    
    print("Feature engineering complete.")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Engineer features for financial time series.")
    
    parser.add_argument('--input_file', type=str, required=True, help="Path to the raw data CSV file.")
    parser.add_argument('--output_dir', type=str, default='processed_data', help="Directory to save the processed data.")
    
    args = parser.parse_args()

    # Load the raw data
    raw_df = pd.read_csv(args.input_file, index_col=0, parse_dates=True)
    
    # Create features
    processed_df = create_features(raw_df)
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the processed data
    base_name = os.path.basename(args.input_file)
    output_path = os.path.join(args.output_dir, f"processed_{base_name}")
    processed_df.to_csv(output_path)
    
    print(f"Processed data with features saved to {output_path}")

    # Example Usage from command line:
    # python feature_engineering.py --input_file "raw_data/BTC-USD_1h.csv"
