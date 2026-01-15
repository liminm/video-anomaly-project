import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from unet_lstm import RecurrentUNet

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004" 
MODEL_PATH = "models/unet_lstm.pth"
OUTPUT_FILE = "lstm_result.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect():
    print("Running LSTM Detection (Stateful)...")
    model = RecurrentUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    gif_frames = []
    
    # LSTM Memory (Starts Empty)
    hidden_state = None 
    
    # Scoring Buffers
    raw_scores = []
    smoothed_scores = []

    with torch.no_grad():
        # Loop through all frames
        for i in range(1, len(files)):
            # 1. Input: Current Frame (t)
            # Remove the second unsqueeze!
            # Shape becomes: (1, 1, 256, 256) -> (Batch, Channel, H, W)
            img_curr = transform(Image.open(files[i-1])).unsqueeze(0).to(DEVICE)
            
            # Target for comparison
            img_target = transform(Image.open(files[i])).unsqueeze(0).to(DEVICE)
            
            # 2. Forward with State (Step-by-Step)
            # This returns the prediction AND the updated memory (hidden_state)
            # We feed 'hidden_state' back into the model in the next loop iteration
            pred, hidden_state = model.step(img_curr, hidden_state)
            
            # 3. Calculate Error
            # Shape match: (1, 1, 256, 256)
            diff = torch.abs(img_target - pred).cpu().numpy().squeeze()
            
            # 4. Motion Masking (Same logic as U-Net)
            curr_np = img_target.squeeze().cpu().numpy()
            prev_np = img_curr.squeeze().cpu().numpy()
            mask = (np.abs(curr_np - prev_np) > 0.05).astype(float)
            masked_diff = diff * mask
            
            # 5. Top-50 Scoring
            score = np.mean(np.sort(masked_diff.flatten())[::-1][:50])
            raw_scores.append(score)
            
            # 6. Smoothing
            if len(raw_scores) < 5:
                s_score = score
            else:
                s_score = np.mean(raw_scores[-5:])
            
            smoothed_scores.append(s_score)
            
            # --- Visualization ---
            img_cv = cv2.imread(files[i])
            img_cv = cv2.resize(img_cv, (256, 256))
            
            heatmap = cv2.applyColorMap(np.clip(masked_diff*5*255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            combined = np.hstack((img_cv, heatmap))
            pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Calibration (Auto-tune on first 30 frames)
            if i < 30:
                threshold = 1.0
            elif i == 30:
                threshold = np.max(smoothed_scores[:30]) * 1.2
                print(f"Calibration Complete. Threshold: {threshold:.4f}")
            
            status = "NORMAL"
            color = (0, 255, 0)
            
            if i > 30 and s_score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (255,255)], outline="red", width=5)
            
            draw.text((10,10), f"LSTM Score: {s_score:.4f}", fill="white")
            draw.text((10,30), f"Limit: {threshold:.4f}", fill="yellow")
            draw.text((10,230), status, fill=color)
            
            gif_frames.append(pil_img)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()