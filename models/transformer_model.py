# /models/transformer_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A Transformer model for time series classification.
    """
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        """
        Args:
            input_size (int): The number of input features.
            d_model (int): The number of expected features in the encoder/decoder inputs (embedding dim).
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1) # 1 for binary classification
        
        self.d_model = d_model

    def forward(self, src):
        """
        Forward pass for the Transformer model.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        """
        # Embed the input
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through the transformer encoder
        output = self.transformer_encoder(src)
        
        # We use the mean of the sequence output for classification
        output = output.mean(dim=1)
        
        # Pass through the final output layer
        output = self.output_layer(output)
        
        return output
