import torch
import torch.nn as nn
from conv_lstm import ConvLSTMCell

class RecurrentUNet(nn.Module):
    def __init__(self, hidden_channels: int = 256, lstm_layers: int = 1, dropout: float = 0.0):
        super(RecurrentUNet, self).__init__()
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_channels = hidden_channels
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # --- ENCODER ---
        self.enc1 = self.conv_block(1, 32, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(32, 64, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(64, 128, dropout)
        self.pool3 = nn.MaxPool2d(2)

        # --- BOTTLENECK (ConvLSTM) ---
        lstm_cells = []
        for layer_idx in range(lstm_layers):
            input_dim = 128 if layer_idx == 0 else hidden_channels
            lstm_cells.append(ConvLSTMCell(input_dim, hidden_channels, kernel_size=3, bias=True))
        self.bottleneck_lstm = nn.ModuleList(lstm_cells)

        # --- DECODER ---
        self.up3 = nn.ConvTranspose2d(hidden_channels, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128, dropout)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64, dropout)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32, dropout)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_c, out_c, dropout: float):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.bottleneck_lstm]

    def _normalize_hidden(self, hidden_state, batch_size, image_size):
        if hidden_state is None:
            return self._init_hidden(batch_size, image_size)
        if isinstance(hidden_state, tuple):
            hidden_state = [hidden_state]
        if len(hidden_state) != self.lstm_layers:
            return self._init_hidden(batch_size, image_size)
        return hidden_state

    def forward(self, x, hidden_state=None):
        # Used for TRAINING (Sequence processing)
        # x shape: (Batch, Sequence, 1, H, W)
        batch_size, seq_len, _, h, w = x.size()
        predictions = []
        
        hidden_state = self._normalize_hidden(hidden_state, batch_size, (h // 8, w // 8))
            
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
        
        hidden_state = self._normalize_hidden(hidden_state, batch_size, (h // 8, w // 8))
            
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # LSTM Update
        x_lstm = p3
        new_hidden_state = []
        for layer_idx, cell in enumerate(self.bottleneck_lstm):
            h_next, c_next = cell(x_lstm, hidden_state[layer_idx])
            new_hidden_state.append((h_next, c_next))
            x_lstm = h_next
        
        # Decode
        d3 = self.up3(x_lstm)
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
