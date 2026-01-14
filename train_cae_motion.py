import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob
import os
import numpy as np
from autoencoder_model import ConvAutoencoder

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/cae_motion.pth" # New model name
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDMotionDataset(Dataset):
    def __init__(self, root_dir):
        # Find all videos (folders)
        self.video_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.samples = []
        
        # We need pairs of (Frame_t, Frame_t-1)
        for folder in self.video_folders:
            frames = sorted(glob.glob(os.path.join(folder, "*.tif")))
            # Start from index 1 so we always have a previous frame
            for i in range(1, len(frames)):
                self.samples.append((frames[i], frames[i-1]))

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        curr_path, prev_path = self.samples[idx]
        
        curr_img = Image.open(curr_path)
        prev_img = Image.open(prev_path)
        
        t_curr = self.transform(curr_img)
        t_prev = self.transform(prev_img)
        
        # Calculate Motion: Absolute Difference
        # High Motion = High Values (White). Static = 0 (Black).
        motion = torch.abs(t_curr - t_prev)
        
        return motion

def train():
    print(f"Training Motion-Aware Autoencoder on {DEVICE}...")
    
    dataset = UCSDMotionDataset(DATA_DIR)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    model = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for imgs in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training Complete. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()