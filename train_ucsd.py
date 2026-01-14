import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.model import VideoPredictor
import os

# Configuration
DATA_PATH = "data/ucsd_features.pt"
MODEL_SAVE_PATH = "models/lstm_ucsd_cosine.pth" # Renamed to avoid overwriting
BATCH_SIZE = 32
EPOCHS = 100 # You likely need fewer epochs with Cosine Loss
LEARNING_RATE = 0.001
SEQ_LEN = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

# --- NEW: Custom Cosine Loss Function ---
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        # Cosine Similarity is 1.0 for perfect match, -1.0 for opposite
        # We want to minimize Loss, so: Loss = 1 - Similarity
        similarity = self.cos(pred, target)
        return torch.mean(1 - similarity)

def train_ucsd():
    print(f"Training UCSD Model (Cosine Loss) on {DEVICE}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    video_list = torch.load(DATA_PATH)
    
    X_list = []
    y_list = []
    
    for video in video_list:
        num_frames = video.shape[0]
        if num_frames <= SEQ_LEN:
            continue
            
        for i in range(num_frames - SEQ_LEN):
            X_list.append(video[i : i+SEQ_LEN]) 
            y_list.append(video[i+SEQ_LEN])
            
    X = torch.stack(X_list)
    y = torch.stack(y_list)
    print(f"Total sequences: {len(X)}")
    
    dataset = TensorDataset(X, y)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    model = VideoPredictor().to(DEVICE)
    
    # --- CHANGED: Use Cosine Loss ---
    criterion = CosineLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"Training complete! Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_ucsd()