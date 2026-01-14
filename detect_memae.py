import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from memae_model import MemAE

TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
MODEL_PATH = "models/memae_ucsd.pth"
OUTPUT_FILE = "memae_result.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect():
    print("Running MemAE Detection...")
    model = MemAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Must match training size
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    gif_frames = []
    scores = []

    with torch.no_grad():
        for i, path in enumerate(files):
            img_pil = Image.open(path)
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            
            # Reconstruct
            recon, _ = model(img_tensor)
            
            # Calculate Error (Pixel-wise Squared Error)
            diff = (img_tensor - recon).pow(2)
            diff_np = diff.cpu().numpy().squeeze()
            
            # Score = Sum of error (MemAE creates dense error blocks)
            score = np.sum(diff_np)
            scores.append(score)
            
            # Visualization
            # 1. Prepare original
            original_cv = cv2.imread(path)
            original_cv = cv2.resize(original_cv, (256, 256)) # Resize to match model output
            original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            
            # 2. Prepare Heatmap
            heatmap_norm = np.clip(diff_np * 1000, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            combined = np.hstack((original_cv, heatmap))
            pil_img = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_img)
            
            # Thresholding (Calibrate on first 30 frames)
            if i < 30:
                threshold = 9999.0
            else:
                threshold = np.mean(scores[:30]) * 2.5
            
            status = "NORMAL"
            color = (0, 255, 0)
            if i > 30 and score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (255,255)], outline="red", width=5)

            draw.text((10, 10), f"Error: {score:.2f}", fill="white")
            draw.text((10, 30), f"Limit: {threshold:.2f}", fill="yellow")
            draw.text((10, 230), status, fill=color)
            
            gif_frames.append(pil_img)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()