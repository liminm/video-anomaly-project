import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from unet_model import UNet

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test002"
MODEL_PATH = "models/unet_video.pth"
OUTPUT_FILE = "unet_result_smooth.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect():
    print("Running U-Net Prediction with Temporal Smoothing...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    gif_frames = []
    
    # Store raw scores to calculate the smoothed average
    raw_scores = []
    smoothed_scores = []

    with torch.no_grad():
        for i in range(4, len(files)):
            # 1. Inputs
            input_imgs = [transform(Image.open(files[i-4+j])) for j in range(4)]
            x = torch.cat(input_imgs, dim=0).unsqueeze(0).to(DEVICE)
            y_real = transform(Image.open(files[i])).unsqueeze(0).to(DEVICE)
            
            # 2. Predict
            y_pred = model(x)
            
            # 3. Motion Masking
            diff = torch.abs(y_real - y_pred).squeeze().cpu().numpy()
            curr_np = y_real.squeeze().cpu().numpy()
            prev_np = input_imgs[-1].squeeze().cpu().numpy()
            motion_mask = (np.abs(curr_np - prev_np) > 0.05).astype(float)
            masked_diff = diff * motion_mask
            
            # 4. TOP-K SCORING
            # Average of the 200 worst pixels
            top_k_pixels = np.sort(masked_diff.flatten())[::-1][:200]
            raw_score = np.mean(top_k_pixels)
            raw_scores.append(raw_score)
            
            # 5. TEMPORAL SMOOTHING (The Fix)
            # Take the average of the last 5 frames
            # This fills in the gaps where the score briefly dips
            if len(raw_scores) < 5:
                smooth_score = raw_score
            else:
                smooth_score = np.mean(raw_scores[-5:])
            
            smoothed_scores.append(smooth_score)
            
            # --- Visualization ---
            img_cv = cv2.imread(files[i])
            img_cv = cv2.resize(img_cv, (256, 256))
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Heatmap
            heatmap_norm = np.clip(masked_diff * 5.0 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            combined = np.hstack((img_rgb, heatmap))
            pil_img = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_img)
            
            # 6. Calibration
            if i < 34:
                threshold = 1.0
            elif i == 34:
                # Calibrate on the SMOOTHED scores of normal walkers
                # Using 1.2x margin (tighter than before because smoothing reduces noise)
                normal_max = np.max(smoothed_scores[:30])
                threshold = normal_max * 1.2
                print(f"Calibration Complete. Threshold: {threshold:.4f}")
            
            status = "NORMAL"
            color = (0, 255, 0)
            
            # Check against the SMOOTH score
            if i > 34 and smooth_score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (255,255)], outline="red", width=5)
                
            draw.text((10, 10), f"SmoothScore: {smooth_score:.4f}", fill="white")
            draw.text((10, 30), f"Limit: {threshold:.4f}", fill="yellow")
            draw.text((10, 230), status, fill=color)
            
            gif_frames.append(pil_img)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()