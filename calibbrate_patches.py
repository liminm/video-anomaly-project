import torch
import torch.nn.functional as F
import glob
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from autoencoder_model import ConvAutoencoder

# Config
TRAIN_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/cae_motion.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calibrate_patches():
    print(f"Calibrating Local Patch Metric on {DEVICE}...")
    
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # We'll check every folder in Train to find the absolute worst normal case
    folders = sorted(glob.glob(os.path.join(TRAIN_DIR, "*")))
    all_max_scores = []

    with torch.no_grad():
        for folder in folders:
            files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            for i in range(1, len(files)):
                # Load Pair
                curr = transform(Image.open(files[i])).to(DEVICE)
                prev = transform(Image.open(files[i-1])).to(DEVICE)
                motion = torch.abs(curr - prev).unsqueeze(0)
                
                # Reconstruct
                recon = model(motion)
                diff = (motion - recon).pow(2)
                
                # --- NEW METRIC: Local Patch Scoring ---
                # We slide a 16x16 window over the image and average the error inside it
                # Then we take the MAX of those windows.
                # This finds the "Worst Single Person" in the frame.
                patches = F.avg_pool2d(diff, kernel_size=16, stride=8)
                max_patch_error = patches.max().item()
                
                all_max_scores.append(max_patch_error)

    # Statistics
    max_score = np.max(all_max_scores)
    mean_score = np.mean(all_max_scores)
    
    # Set threshold slightly above the worst thing we ever saw in training
    # 1.1x safety buffer
    threshold = max_score * 1.1
    
    print("-" * 30)
    print(f"CALIBRATION RESULTS (Patch Metric):")
    print(f"  Avg Normal Peak: {mean_score:.5f}")
    print(f"  MAX Normal Peak: {max_score:.5f}")
    print(f"  >> USE THIS THRESHOLD: {threshold:.5f} <<")
    print("-" * 30)

if __name__ == "__main__":
    calibrate_patches()