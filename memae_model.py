import torch
import torch.nn as nn
from memory_module import MemoryModule

class MemAE(nn.Module):
    def __init__(self):
        super(MemAE, self).__init__()
        
        # Encoder (Standard CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Memory Module (Bottleneck)
        # We store 2000 normal patterns of size 256
        self.mem_rep = MemoryModule(mem_dim=2000, fea_dim=256, shrink_thres=0.0025)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        f = self.encoder(x)
        
        # Query Memory
        # Instead of passing 'f' directly, we pass the retrieved memory 'res'
        res, att = self.mem_rep(f)
        
        # Decode the MEMORY, not the input
        out = self.decoder(res)
        
        return out, att