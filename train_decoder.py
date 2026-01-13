import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.model import VideoDecoder

# Configuration
FEATURES_PATH = "data/features.pt"       # The vectors (Inputs)
IMAGES_PATH = "data/mnist_test_seq.npy"  # The raw images (Targets)
SAVE_PATH = "models/decoder.pth"
BATCH_SIZE = 32
EPOCHS = 1000
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def load_data():
    # 1. Load Features (Inputs) -> (200000, 512)
    print("Loading features...")
    features = torch.load(FEATURES_PATH)
    if features.dim() == 3:
        features = features.reshape(-1, features.shape[-1])


    # 2. Load Raw Images (Targets) -> (20, 10000, 64, 64)
    print("Loading raw images...")
    raw_imgs = np.load(IMAGES_PATH)

    # Reshape to match features: (200000, 1, 64, 64)
    # Note: Raw data is (Seq, Batch, H, W).
    # We transpose to (Batch, Seq, H, W) then flatten.
    raw_imgs = raw_imgs.transpose(1, 0, 2, 3).reshape(-1, 1, 64, 64)

    # Convert to Tensor and duplicate channel to match RGB (3, 64, 64)
    # We divide by 255.0 to normalize pixel values to 0-1 range
    imgs_tensor = torch.tensor(raw_imgs, dtype=torch.float32) / 255.0
    imgs_tensor = imgs_tensor.repeat(1, 3, 1, 1) # 1 channel -> 3 channels

    return features, imgs_tensor

def train_decoder():
    print(f"Training Decoder on {DEVICE}...")

    X, y = load_data()
    print(f"Features shape: {X.shape}, Images shape: {y.shape}")
    

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VideoDecoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Compare generated pixels vs real pixels

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_features, batch_imgs in loader:
            batch_features = batch_features.to(DEVICE)
            batch_imgs = batch_imgs.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass: Vector -> Image
            reconstructed_imgs = model(batch_features)

            # Calculate loss
            loss = criterion(reconstructed_imgs, batch_imgs)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Reconstruction Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Decoder saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_decoder()
