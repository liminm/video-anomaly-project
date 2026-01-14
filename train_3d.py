import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import random
from model_3d import C3DAutoencoder

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/c3d_ae.pth"
CLIP_LEN = 16    # Depth (Time)
PATCH_SIZE = 32  # Spatial Size
BATCH_SIZE = 16  # Smaller batch due to 3D memory usage
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSD3DDataset(Dataset):
    def __init__(self, root_dir):
        self.video_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        # Virtual length: lots of random clips
        return 2000

    def __getitem__(self, idx):
        # Pick random video
        folder = random.choice(self.video_folders)
        files = sorted(glob.glob(os.path.join(folder, "*.tif")))
        
        # Pick random start time
        if len(files) < CLIP_LEN: return torch.zeros((1, CLIP_LEN, PATCH_SIZE, PATCH_SIZE))
        start_t = random.randint(0, len(files) - CLIP_LEN)
        
        # Load clip frames
        clip_files = files[start_t : start_t + CLIP_LEN]
        frames = [self.transform(Image.open(f)) for f in clip_files]
        
        # Stack into (Depth, 1, H, W) -> (1, Depth, H, W)
        # Note: Image size is full frame here
        clip_tensor = torch.stack(frames, dim=1) 
        
        # Random Spatial Crop (Patch)
        _, _, h, w = clip_tensor.shape
        top = random.randint(0, h - PATCH_SIZE)
        left = random.randint(0, w - PATCH_SIZE)
        
        patch_clip = clip_tensor[:, :, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
        return patch_clip

def train():
    print(f"Training C3D Autoencoder on {DEVICE}...")
    dataset = UCSD3DDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = C3DAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # L1 Loss is better for video sharpness
    criterion = nn.L1Loss() 
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, clips in enumerate(loader):
            clips = clips.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, clips)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()