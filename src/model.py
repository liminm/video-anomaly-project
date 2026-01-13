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


class VideoDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # 1. Expand the vector: 512 -> 2048 (which is 128 * 4 * 4)
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        # 2. Upscaling layers (The "Artist")
        self.net = nn.Sequential(
            # Input shape: (128, 4, 4)

            # Layer 1: Upscale to 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 2: Upscale to 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Layer 3: Upscale to 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # Layer 4: Upscale to 64x64 (Original Moving MNIST size)
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Squish pixel values to be between 0 and 1
        )

    def forward(self, x):
        # x shape: (Batch, 512)
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4) # Reshape into a tiny 4x4 image
        img = self.net(x)
        return img
