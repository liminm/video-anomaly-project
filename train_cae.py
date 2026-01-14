import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob
import os
from autoencoder_model import ConvAutoencoder

# Config
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/cae_ucsd.pth"
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Custom Dataset for Raw Frames
class UCSDFrameDataset(Dataset):
    def __init__(self, root_dir):
        # We find ALL .tif files in all subfolders
        self.files = sorted(glob.glob(os.path.join(root_dir, "*/*.tif")))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), # Standard size for CAEs
            transforms.Grayscale(),        # 1 Channel
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.transform(img)

def train():
    print(f"Training Autoencoder on {DEVICE}...")
    
    # 1. Prepare Data
    dataset = UCSDFrameDataset(DATA_DIR)
    # Split Train/Val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # 2. Setup Model
    model = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for imgs in train_loader:
            imgs = imgs.to(DEVICE)
            
            # Input = Target (Reconstruction Task)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training Complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    train()