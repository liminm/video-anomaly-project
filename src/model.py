import torch
import torch.nn as nn

class VideoPredictor(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        
        # The LSTM layer processes the sequence
        # batch_first=True means input shape is (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # Add some regularization
        )
        
        # The Head maps the LSTM's memory (hidden state) back to a feature vector
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, 512)
        
        # Run LSTM
        # out shape: (Batch, Seq_Len, Hidden_Size)
        # _ contains the hidden states (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We only care about the LAST time step's output for prediction
        # Take the last element in the sequence
        last_step_out = lstm_out[:, -1, :] 
        
        # Predict the next vector
        prediction = self.head(last_step_out)
        return prediction
    