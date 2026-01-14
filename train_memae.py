import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
from memae_model import MemAE

DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/memae_ucsd.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*/*.tif")))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.transform(Image.open(self.files[i]))

def train():
    print(f"Training MemAE on {DEVICE}...")
    dataset = UCSDDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MemAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    
    for epoch in range(25): # 25 epochs is usually enough
        model.train()
        total_loss = 0
        for x in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, att = model(x)
            
            # Reconstruction Loss + Entropy Loss (force sparse memory usage)
            l_recon = loss_func(recon, x)
            l_entropy = (-att * torch.log(att + 1e-12)).sum() / att.size(0)
            
            loss = l_recon + (0.0002 * l_entropy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.5f}")
        torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()