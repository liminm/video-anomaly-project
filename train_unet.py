import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
from unet_model import UNet

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/unet_video.pth"
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Standard size
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        for folder in folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            # Need 4 inputs + 1 target = 5 frames min
            for i in range(len(files) - 5):
                self.samples.append(files[i:i+5])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        frames = [self.transform(Image.open(p)) for p in paths]
        
        # Input: First 4 frames stacked (4, H, W)
        x = torch.cat(frames[:4], dim=0)
        # Target: 5th frame (1, H, W)
        y = frames[4]
        
        return x, y

def train():
    print(f"Training U-Net Video Predictor on {DEVICE}...")
    dataset = UCSDSequenceDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()