import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import random
from patch_model import PatchAutoencoder

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/patch_ae.pth"
PATCH_SIZE = 32
NUM_PATCHES_PER_IMAGE = 50 # Extract 50 random squares from every frame
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDPatchDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*/*.tif")))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        # Virtual length: Num Images * Patches per Image
        return len(self.files) * NUM_PATCHES_PER_IMAGE

    def __getitem__(self, idx):
        # Map virtual index to real image index
        img_idx = idx // NUM_PATCHES_PER_IMAGE
        img_path = self.files[img_idx]
        
        img = Image.open(img_path)
        img_tensor = self.transform(img) # (1, H, W)
        
        # Random Crop
        _, h, w = img_tensor.shape
        top = random.randint(0, h - PATCH_SIZE)
        left = random.randint(0, w - PATCH_SIZE)
        
        patch = img_tensor[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
        return patch

def train():
    print(f"Training Patch Autoencoder on {DEVICE}...")
    dataset = UCSDPatchDataset(DATA_DIR)
    # We use a large batch size because patches are small
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = PatchAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, patches in enumerate(loader):
            patches = patches.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, patches)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.5f}")
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()