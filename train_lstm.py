import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
from unet_lstm import RecurrentUNet

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/unet_lstm.pth"
ONNX_PATH = "models/unet_lstm.onnx"
BATCH_SIZE = 4   # LSTMs use more VRAM, so we lower the batch size
SEQ_LEN = 8      # Train on clips of 8 frames
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDLSTMDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        for folder in folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            # Sliding window of SEQ_LEN + 1 (Input + Target)
            for i in range(len(files) - SEQ_LEN - 1):
                self.samples.append(files[i : i + SEQ_LEN + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        frames = [self.transform(Image.open(p)) for p in paths]
        
        # Stack into (Seq_Len+1, 1, H, W)
        video_tensor = torch.stack(frames, dim=0)
        
        # Input: Frames 0 to N-1
        # Target: Frames 1 to N
        x = video_tensor[:-1] 
        y = video_tensor[1:]
        
        return x, y

def train():
    print(f"Training Recurrent U-Net (LSTM) on {DEVICE}...")
    dataset = UCSDLSTMDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = RecurrentUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR for LSTM stability
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (x, y) in enumerate(loader):
            # Shape: (Batch, Seq, 1, H, W)
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Reset hidden state for each new batch (handled inside forward if None)
            preds = model(x)
            
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.5f}")
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

    # Export final model to ONNX
    model.eval()
    model_cpu = model.to("cpu")
    dummy_input = torch.zeros(1, SEQ_LEN, 1, 256, 256, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )
    print(f"ONNX model saved to {ONNX_PATH}")

if __name__ == "__main__":
    train()