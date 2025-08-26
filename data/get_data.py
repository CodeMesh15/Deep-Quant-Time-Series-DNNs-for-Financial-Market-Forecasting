
import yfinance as yf
import pandas as pd
import argparse
import os

def fetch_data(ticker, start_date, end_date, interval, output_dir):
    """
    Fetches historical financial data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock or cryptocurrency ticker symbol (e.g., 'AAPL', 'BTC-USD').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        interval (str): The data interval (e.g., '1d', '1h', '15m').
        output_dir (str): The directory to save the output CSV file.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}...")
    
    # Download data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"No data found for {ticker}. Please check the ticker symbol and date range.")
        return

    # Basic data cleaning - forward-fill missing values
    data.ffill(inplace=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data to a CSV file
    output_path = os.path.join(output_dir, f"{ticker}_{interval}.csv")
    data.to_csv(output_path)
    
    print(f"Data successfully saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download historical financial data.")
    
    parser.add_argument('--ticker', type=str, required=True, help="Ticker symbol (e.g., 'AAPL', 'BTC-USD')")
    parser.add_argument('--start', type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument('--end', type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument('--interval', type=str, default='1d', help="Data interval (e.g., '1d', '1h', '30m')")
    parser.add_argument('--output_dir', type=str, default='raw_data', help="Directory to save the raw data")

    args = parser.parse_args()
    
    fetch_data(args.ticker, args.start, args.end, args.interval, args.output_dir)

    # Example Usage from command line:
    # python get_data.py --ticker "BTC-USD" --start "2023-01-01" --end "2024-01-01" --interval "1h"
