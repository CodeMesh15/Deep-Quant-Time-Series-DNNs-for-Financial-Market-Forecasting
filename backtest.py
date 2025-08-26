
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

# Import project components
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.backtester import run_backtest
from train import create_sequences  # Reuse the sequence creation logic

def generate_predictions(model, data_loader, device):
    """Generates model predictions for a given dataset."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted_probs = torch.sigmoid(outputs).cpu().numpy()
            predicted_labels = (predicted_probs > 0.5).astype(int)
            predictions.extend(predicted_labels.flatten())
    return np.array(predictions)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and Prepare Data (same as in train.py to ensure consistency)
    df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']

    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X
