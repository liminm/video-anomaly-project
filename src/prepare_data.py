import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.dataset import MovingMNISTDataset

# Configuration
BATCH_SIZE = 16  # Reduce this if you run out of memory
DATA_PATH = "data/mnist_test_seq.npy"
OUTPUT_PATH = "data/features.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On Mac M1/M2 chips, use 'mps' for acceleration if available
if torch.backends.mps.is_available():
    DEVICE = "mps"


def extract_features():
    print(f"Using device: {DEVICE}")

    # 1. Load the Dataset
    dataset = MovingMNISTDataset(DATA_PATH)
    # We use a DataLoader to handle batching.
    # num_workers=0 is safer for simple scripts to avoid multiprocessing bugs.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Prepare the Feature Extractor (ResNet18)
    print("Loading ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace the last fully connected layer with Identity to get the 512 feature vector
    model.fc = nn.Identity()

    model.to(DEVICE)
    model.eval()  # Set to evaluation mode (no training)

    all_features = []

    print("Starting feature extraction (this may take a while)...")

    # 3. Processing Loop
    with torch.no_grad():  # Disable gradient calculation to save memory
        for video_batch in tqdm(loader):
            # video_batch shape: [Batch_Size, 20, 3, 224, 224]
            # We need to process each frame. ResNet expects [N, 3, 224, 224]

            batch_size, seq_len, c, h, w = video_batch.shape

            # Flatten the batch and sequence dimensions to process all frames at once
            # New shape: [Batch_Size * 20, 3, 224, 224]
            flat_input = video_batch.view(-1, c, h, w).to(DEVICE)

            # Pass through ResNet
            # Output shape: [Batch_Size * 20, 512]
            flat_features = model(flat_input)

            # Reshape back to [Batch_Size, 20, 512]
            features = flat_features.view(batch_size, seq_len, -1)

            # Move to CPU to save memory and add to list
            all_features.append(features.cpu())

    # 4. Save to disk
    # Concatenate all batches into one big tensor
    full_dataset = torch.cat(all_features, dim=0)
    print(f"Extraction complete." 
          f"Saving tensor of shape {full_dataset.shape} to {OUTPUT_PATH}...")
    torch.save(full_dataset, OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    extract_features()
