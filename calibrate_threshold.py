import torch
import glob
import os
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from src.model import VideoPredictor

# Use a TRAIN video (guaranteed no anomalies) to find the baseline
NORMAL_VIDEO_PATH = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001"
LSTM_PATH = "models/lstm_ucsd.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def get_max_normal_error():
    print(f"Calibrating baseline on {NORMAL_VIDEO_PATH}...")
    
    # Load Models
    encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.to(DEVICE).eval()

    lstm = VideoPredictor().to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
    lstm.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_paths = sorted(glob.glob(os.path.join(NORMAL_VIDEO_PATH, "*.tif")))
    prev_frames = []
    errors = []

    with torch.no_grad():
        for path in frame_paths:
            img_pil = Image.open(path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            feature = encoder(img_tensor)
            prev_frames.append(feature)
            
            if len(prev_frames) > 10:
                prev_frames.pop(0)
                input_seq = torch.stack(prev_frames).view(1, 10, 512)
                pred_vector = lstm(input_seq)
                
                # Calculate Error
                loss = torch.mean((pred_vector - feature) ** 2).item()
                errors.append(loss)

    if not errors:
        print("Error: Could not calculate errors.")
        return 0.0

    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    # Rule of Thumb: Threshold = Max_Normal_Error + Safety_Margin
    recommended_threshold = max_error * 1.2  # 20% buffer
    
    print("-" * 30)
    print(f"Calibration Results:")
    print(f"  Average Normal Error: {mean_error:.5f}")
    print(f"  Max Normal Error:     {max_error:.5f}")
    print(f"  RECOMMENDED THRESHOLD: {recommended_threshold:.5f}")
    print("-" * 30)
    
    return recommended_threshold

if __name__ == "__main__":
    get_max_normal_error()