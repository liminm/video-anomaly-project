import torch
import torch.nn as nn
import glob
import os
import cv2  # Needed for background subtraction
import numpy as np
from PIL import Image, ImageDraw
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from src.model import VideoPredictor

# Configuration
TEST_VIDEO_PATH = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
OUTPUT_PATH = "ucsd_cosine.gif"
LSTM_PATH = "models/lstm_ucsd_cosine.pth"

THRESHOLD_FACTOR = 2.0 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def generate_gif():
    print(f"Generating result using COSINE DISTANCE on {DEVICE}...")

    # 1. Load Models
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

    frame_paths = sorted(glob.glob(os.path.join(TEST_VIDEO_PATH, "*.tif")))
    
    prev_frames = []
    gif_frames = []
    raw_scores = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # --- NEW: Initialize Background Subtractor (Same as training) ---
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    print("Pass 1: Calculating scores with Background Subtraction...")
    with torch.no_grad():
        for i, path in enumerate(frame_paths):
            # --- CRITICAL FIX: APPLY MASKING ---
            # 1. Read with OpenCV
            img_cv = cv2.imread(path)
            
            # 2. Apply Background Subtraction
            fgmask = fgbg.apply(img_cv)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            motion_img = cv2.bitwise_and(img_cv, img_cv, mask=fgmask)
            
            # 3. Convert to PIL for Model
            img_rgb = cv2.cvtColor(motion_img, cv2.COLOR_BGR2RGB)
            img_pil_motion = Image.fromarray(img_rgb)
            
            # 4. Extract Feature from MOTION IMAGE (Black background)
            img_tensor = transform(img_pil_motion).unsqueeze(0).to(DEVICE)
            feature = encoder(img_tensor)
            
            prev_frames.append(feature)
            
            score = 0.0
            if len(prev_frames) > 10:
                prev_frames.pop(0)
                input_seq = torch.stack(prev_frames).view(1, 10, 512)
                pred_vector = lstm(input_seq)
                
                similarity = cos(pred_vector, feature)
                score = (1 - similarity).item()
                
            raw_scores.append(score)

    # 2. Smooth the Scores
    window_size = 5
    smoothed_scores = np.convolve(raw_scores, np.ones(window_size)/window_size, mode='same')
    
    # 3. Dynamic Threshold
    baseline_noise = np.percentile(smoothed_scores, 25)
    final_threshold = baseline_noise * THRESHOLD_FACTOR
    
    print(f"Baseline Noise: {baseline_noise:.5f}")
    print(f"Threshold:      {final_threshold:.5f}")

    # 4. Generate GIF
    print("Pass 2: Rendering GIF...")
    
    # Note: We reload raw images here because we WANT to see the background 
    # in the final GIF, even though the model ignored it.
    for i, path in enumerate(frame_paths):
        img_pil = Image.open(path).convert("RGB").resize((224, 224))
        img_np = np.array(img_pil)
        img_pil_draw = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil_draw)
        
        current_score = smoothed_scores[i]
        is_anomaly = current_score > final_threshold

        status_text = "NORMAL"
        status_color = (0, 255, 0)
        
        if is_anomaly:
            status_text = "ANOMALY"
            status_color = (255, 0, 0)
            draw.rectangle([(0,0), (223,223)], outline="red", width=5)

        draw.text((10, 10), f"Score: {current_score:.4f}", fill="white")
        draw.text((10, 25), f"Thresh: {final_threshold:.4f}", fill="yellow")
        draw.text((10, 200), status_text, fill=status_color)
        
        gif_frames.append(img_pil_draw)

    print(f"Saving to {OUTPUT_PATH}...")
    gif_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=gif_frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print("Done!")

if __name__ == "__main__":
    generate_gif()