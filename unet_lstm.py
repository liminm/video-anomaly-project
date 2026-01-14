import torch
import torch.nn as nn
from conv_lstm import ConvLSTMCell

class RecurrentUNet(nn.Module):
    def __init__(self):
        super(RecurrentUNet, self).__init__()
        
        # --- ENCODER ---
        self.enc1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK (ConvLSTM) ---
        self.bottleneck_lstm = ConvLSTMCell(128, 256, kernel_size=3, bias=True)
        
        # --- DECODER ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
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

    def forward(self, x, hidden_state=None):
        # Used for TRAINING (Sequence processing)
        # x shape: (Batch, Sequence, 1, H, W)
        batch_size, seq_len, _, h, w = x.size()
        predictions = []
        
        if hidden_state is None:
            hidden_state = self.bottleneck_lstm.init_hidden(batch_size, (h//8, w//8))
            
        for t in range(seq_len):
            input_frame = x[:, t, :, :, :]
            
            # Use the single-step logic defined below
            pred, hidden_state = self.step(input_frame, hidden_state)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)

    def step(self, x, hidden_state=None):
        # Used for DETECTION (Single frame processing)
        # x shape: (Batch, 1, H, W) - Note: No sequence dimension here
        batch_size, _, h, w = x.size()
        
        if hidden_state is None:
            hidden_state = self.bottleneck_lstm.init_hidden(batch_size, (h//8, w//8))
            
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # LSTM Update
        h_next, c_next = self.bottleneck_lstm(p3, hidden_state)
        new_hidden_state = (h_next, c_next)
        
        # Decode
        d3 = self.up3(h_next)
        if d3.shape != e3.shape:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        pred = torch.sigmoid(self.final(d1))
        
        # Return Prediction AND New State
        return pred, new_hidden_state