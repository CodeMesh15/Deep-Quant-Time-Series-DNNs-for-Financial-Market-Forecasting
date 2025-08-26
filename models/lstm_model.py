
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    An LSTM model for time series classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        """
        Args:
            input_size (int): The number of features in the input data.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output (1 for binary classification).
            dropout (float): Dropout probability.
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            dropout=dropout
        )
        
        # Define the fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We pass the input through the LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output of the last time step for classification
        last_time_step_out = out[:, -1, :]
        
        # Pass the last time step's output through the fully connected layer
        out = self.fc(last_time_step_out)
        
        return out
