
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

# Import model classes from our models directory
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel

# --- Data Preparation ---

def create_sequences(features, target, sequence_length):
    """Creates sequences from time series data."""
    X_seq, y_seq = [], []
    for i in range(len(features) - sequence_length):
        X_seq.append(features[i:(i + sequence_length)])
        y_seq.append(target[i + sequence_length - 1])
    return np.array(X_seq), np.array(y_seq)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Main Training Function ---

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Prepare Data
    df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 2. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Create Sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, args.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, args.sequence_length)

    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model, Loss, and Optimizer
    input_size = len(features)
    if args.model_type == 'lstm':
        model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=2).to(device)
    elif args.model_type == 'transformer':
        model = TransformerModel(input_size=input_size, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=64).to(device)
    else:
        raise ValueError("Invalid model type specified. Choose 'lstm' or 'transformer'.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    print(f"--- Starting training for {args.model_type} model ---")
    for epoch in range(args.epochs):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                predicted = torch.round(torch.sigmoid(outputs))
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {100*correct/total:.2f}%")

    # 6. Save Model
    torch.save(model.state_dict(), f"{args.model_type}_model.pth")
    print(f"Model saved to {args.model_type}_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a time series model.")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the processed data CSV file.')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'], help='Model to train.')
    parser.add_argument('--sequence_length', type=int, default=60, help='Length of the input sequences.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    train_model(args)
