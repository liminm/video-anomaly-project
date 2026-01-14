import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from autoencoder_model import ConvAutoencoder

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
MODEL_PATH = "models/cae_ucsd.pth"
OUTPUT_FILE = "cae_result_max_score.gif"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_detection():
    print("Running Reconstruction Anomaly Detection (Max Score Strategy)...")
    
    # 1. Load Model
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    gif_frames = []
    
    # We will use the first 30 frames to calibrate what "Normal Max Error" looks like
    max_errors = []

    with torch.no_grad():
        for i, path in enumerate(files):
            # Load & Reconstruct
            img_pil = Image.open(path)
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            reconstruction = model(img_tensor)
            
            # Calculate Absolute Difference per pixel
            # Shape: (1, 1, 128, 128)
            diff = torch.abs(img_tensor - reconstruction)
            
            # --- NEW SCORING LOGIC ---
            # Instead of mean(), we take the max() of the error map.
            # This ignores the background completely and asks "What is the biggest failure?"
            
            # Optional: Apply a central mask to ignore edge noise
            # (Sometimes padding causes high error at the very borders)
            diff_np = diff.cpu().numpy().squeeze()
            height, width = diff_np.shape
            mask = np.zeros_like(diff_np)
            mask[10:height-10, 10:width-10] = 1 # Ignore 10px border
            masked_diff = diff_np * mask
            
            # SCORE = The average of the top 50 worst pixels (more robust than single max)
            # Flatten array, sort descending, take top 50, average them
            top_k_pixels = np.sort(masked_diff.flatten())[::-1][:50]
            frame_score = np.mean(top_k_pixels)
            
            max_errors.append(frame_score)
            
            # --- Visualization ---
            original_cv = cv2.imread(path)
            original_cv = cv2.resize(original_cv, (128, 128))
            original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            
            # Heatmap: Normalize based on a fixed "High Error" value (e.g. 0.5)
            # This ensures consistent coloring across frames
            diff_norm = np.clip(diff_np / 0.5 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            
            combined = np.hstack((original_cv, heatmap))
            pil_combined = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_combined)
            
            # Dynamic Thresholding
            # Calibrate on the first 30 frames (assuming they are mostly normal)
            if i < 30:
                threshold = 1.0 # Placeholder high value
            else:
                # Set threshold to 1.5x the average normal score seen so far
                normal_baseline = np.mean(max_errors[:30])
                threshold = normal_baseline * 1.5
            
            status = "NORMAL"
            color = (0, 255, 0)
            
            if i > 30 and frame_score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                # Draw border
                draw.rectangle([(0,0), (255,127)], outline="red", width=3)

            draw.text((5, 5), f"MaxError: {frame_score:.4f}", fill="white")
            draw.text((5, 20), f"Thresh: {threshold:.4f}", fill="yellow")
            draw.text((5, 110), status, fill=color)
            
            gif_frames.append(pil_combined)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(
        OUTPUT_FILE, 
        save_all=True, 
        append_images=gif_frames[1:], 
        optimize=False,
        duration=100, 
        loop=0
    )
    print("Done!")

if __name__ == "__main__":
    run_detection()