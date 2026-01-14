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
MODEL_PATH = "models/cae_motion.pth"
OUTPUT_FILE = "motion_anomaly.gif"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_detection():
    print("Running Motion-Aware Detection...")
    
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
    scores = []

    with torch.no_grad():
        # Start from 1 because we need a previous frame
        for i in range(1, len(files)):
            # 1. Load Pair
            curr_pil = Image.open(files[i])
            prev_pil = Image.open(files[i-1])
            
            t_curr = transform(curr_pil).to(DEVICE)
            t_prev = transform(prev_pil).to(DEVICE)
            
            # 2. Compute Motion Input
            motion_input = torch.abs(t_curr - t_prev).unsqueeze(0) # Add batch dim
            
            # 3. Reconstruct
            reconstruction = model(motion_input)
            
            # 4. Error Calculation (Reconstruction Error)
            # Normal Walker (Faint Ghost) -> Good Reconstruction -> Low Error
            # Fast Biker (Bright Ghost) -> Bad Reconstruction -> High Error
            diff = (motion_input - reconstruction).pow(2)
            
            # Filter noise: Only count errors above a tiny floor
            diff_np = diff.cpu().numpy().squeeze()
            score = np.sum(diff_np[diff_np > 0.01]) # Sum of significant errors
            scores.append(score)
            
            # --- Visualization ---
            original_cv = cv2.imread(files[i])
            original_cv = cv2.resize(original_cv, (128, 128))
            original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
            
            # Visualize the ERROR heatmap
            error_map = np.clip(diff_np * 5000, 0, 255).astype(np.uint8) # Amplify for visibility
            heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
            
            combined = np.hstack((original_cv, heatmap))
            pil_combined = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_combined)
            
            # Dynamic Threshold (Calibrate on first 20 frames)
            if i < 25:
                threshold = 1000.0 # High value during warmup
            else:
                # Set threshold slightly above the noise floor seen so far
                threshold = np.mean(scores[:25]) * 3.0
            
            status = "NORMAL"
            color = (0, 255, 0)
            if i > 25 and score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (127,127)], outline="red", width=3)

            draw.text((5, 5), f"MotionError: {score:.2f}", fill="white")
            draw.text((5, 20), f"Limit: {threshold:.2f}", fill="yellow")
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