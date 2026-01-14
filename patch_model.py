import torch
import torch.nn as nn

class PatchAutoencoder(nn.Module):
    def __init__(self):
        super(PatchAutoencoder, self).__init__()
        
        # --- ENCODER ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Flatten: 64 channels * 4 * 4 = 1024
        # CRUSH IT DOWN TO 32 VALUES (32x Compression)
        self.bottleneck_enc = nn.Linear(64 * 4 * 4, 32)

        # --- DECODER ---
        # Expand back to 1024
        self.bottleneck_dec = nn.Linear(32, 64 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 16x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) # Flatten
        z = self.bottleneck_enc(x) # Squeeze to 32 numbers
        
        # Decode
        x = self.bottleneck_dec(z) # Expand
        x = x.view(x.size(0), 64, 4, 4) # Reshape
        out = self.decoder_conv(x)
        
        return out