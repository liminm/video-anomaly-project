import cv2
import torch
import torch.nn as nn
import os
import glob
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# Configuration
# NOTE: Ensure this path matches where download_ucsd.py extracted the files
DATA_ROOT = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
OUTPUT_PATH = "data/ucsd_features.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def extract_features():
    print(f"Extracting features from {DATA_ROOT} on {DEVICE}...")
    
    # 1. Setup Model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.to(DEVICE).eval()
    
    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    video_folders = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
    if not video_folders:
        print(f"Error: No folders found in {DATA_ROOT}. Check your path.")
        return

    all_video_features = [] 

    with torch.no_grad():
        for folder in tqdm(video_folders):
            frame_paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
            video_frames = []
            for p in frame_paths:
                # Read image using OpenCV (to apply mask)
                img_cv = cv2.imread(p)
                
                # Apply Mask
                fgmask = fgbg.apply(img_cv)
                # Optional: Clean noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                
                # Create "Motion Image" (Black background, color person)
                motion_img = cv2.bitwise_and(img_cv, img_cv, mask=fgmask)
                
                # Convert to PIL for Transforms
                img_rgb = cv2.cvtColor(motion_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                video_frames.append(transform(img_pil))
            
            # Stack into batch: (Num_Frames, 3, 224, 224)
            # If video is too long for GPU RAM, you might need to chunk this loop
            batch = torch.stack(video_frames).to(DEVICE)
            
            features = model(batch) # (Num_Frames, 512)
            all_video_features.append(features.cpu())

    torch.save(all_video_features, OUTPUT_PATH)
    print(f"Saved features for {len(all_video_features)} videos to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_features()