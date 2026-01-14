import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # --- ENCODER ---
        # Input: 4 Frames -> 32 filters
        self.enc1 = self.conv_block(4, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        # 32 -> 64
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # 64 -> 128
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK ---
        # 128 -> 256
        self.bottleneck = self.conv_block(128, 256)
        
        # --- DECODER ---
        
        # Up 3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Input to dec3 will be: 128 (from up3) + 128 (from enc3) = 256 channels
        self.dec3 = self.conv_block(256, 128) 
        
        # Up 2: 128 -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Input to dec2 will be: 64 (from up2) + 64 (from enc2) = 128 channels
        self.dec2 = self.conv_block(128, 64)
        
        # Up 1: 64 -> 32
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Input to dec1 will be: 32 (from up1) + 32 (from enc1) = 64 channels
        self.dec1 = self.conv_block(64, 32)
        
        # Output: 32 -> 1 Predicted Frame
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        d3 = self.up3(b)
        # Resize to match encoder feature map (fix for odd dimensions)
        if d3.shape != e3.shape:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat((d3, e3), dim=1) # 128 + 128 = 256
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1) # 64 + 64 = 128
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((d1, e1), dim=1) # 32 + 32 = 64
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))