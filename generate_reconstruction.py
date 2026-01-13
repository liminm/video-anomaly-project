import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.dataset import MovingMNISTDataset
from src.model import VideoDecoder

# Paths
DATA_PATH = "data/mnist_test_seq.npy"
DECODER_PATH = "models/decoder.pth"
OUTPUT_DIR = "generated_results"

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_reconstruction():
    print(f"Running reconstruction on {DEVICE}...")
    ensure_dir(OUTPUT_DIR)

    print("Loading models...")
    encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder.to(DEVICE)
    encoder.eval()

    decoder = VideoDecoder().to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    decoder.eval()

    dataset = MovingMNISTDataset(DATA_PATH)
    video_idx = np.random.randint(0, len(dataset))
    frame_idx = np.random.randint(0, dataset.data.shape[1])

    # Raw frame for display: (64, 64)
    raw_frame = dataset.data[video_idx][frame_idx]
    real_frame = np.stack([raw_frame] * 3, axis=-1)

    # Encoded frame: (3, 224, 224), normalized as in training
    processed_frame = dataset[video_idx][frame_idx]

    with torch.no_grad():
        features = encoder(processed_frame.unsqueeze(0).to(DEVICE))
        reconstructed = decoder(features).squeeze(0).cpu().permute(1, 2, 0).numpy()

    recon_frame = (reconstructed * 255).clip(0, 255).astype(np.uint8)
    real_frame = real_frame.astype(np.uint8)

    # Side-by-side comparison: left = real, right = reconstructed
    combined = np.hstack((real_frame, recon_frame))
    out_path = os.path.join(OUTPUT_DIR, f"reconstruction_{video_idx}_{frame_idx}.png")
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Reconstruction saved to {out_path}")
    print("Left = REAL, Right = RECONSTRUCTED")


if __name__ == "__main__":
    generate_reconstruction()
