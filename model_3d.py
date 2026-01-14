import torch
import torch.nn as nn

class C3DAutoencoder(nn.Module):
    def __init__(self):
        super(C3DAutoencoder, self).__init__()
        
        # --- ENCODER ---
        # Input: (1, 16, 32, 32) -> (Channel, Depth, Height, Width)
        
        # Layer 1: Time Stride 1, Space Stride 2
        # Depth: (16 + 2 - 3)/1 + 1 = 16
        # Size:  (32 + 2 - 3)/2 + 1 = 16
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            # Layer 2: Time Stride 2, Space Stride 2
            # Depth: (16 + 2 - 3)/2 + 1 = 8
            # Size:  (16 + 2 - 3)/2 + 1 = 8
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Layer 3: Time Stride 2, Space Stride 2
            # Depth: (8 + 2 - 3)/2 + 1 = 4
            # Size:  (8 + 2 - 3)/2 + 1 = 4
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # --- DECODER ---
        
        self.decoder = nn.Sequential(
            # Layer 1 Transpose: Expands 4 -> 8
            # output_padding=(1,1,1) ensures we recover the even number size
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Layer 2 Transpose: Expands 8 -> 16
            # output_padding=(1,1,1) ensures we recover the even number size
            nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            # Layer 3 Transpose: Depth Stride 1, Space Stride 2
            # Depth: 16 -> 16 (padding=0 because stride is 1)
            # Space: 16 -> 32 (padding=1 because stride is 2)
            nn.ConvTranspose3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, output_padding=(0,1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out