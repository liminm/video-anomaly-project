import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from patch_model import PatchAutoencoder

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
MODEL_PATH = "models/patch_ae.pth"
OUTPUT_FILE = "patch_result.gif"
PATCH_SIZE = 32
STRIDE = 16 # Overlap patches for better coverage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect():
    print("Running Patch-Based Detection...")
    model = PatchAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    
    gif_frames = []
    max_scores = [] # Track frame-level max scores

    with torch.no_grad():
        for i, path in enumerate(files):
            img_pil = Image.open(path)
            # Use Original Resolution (Do not resize full image!)
            img_tensor = transform(img_pil).to(DEVICE) # (1, H, W)
            _, h, w = img_tensor.shape
            
            # Create an empty error map
            error_map = torch.zeros((h, w), device=DEVICE)
            counts = torch.zeros((h, w), device=DEVICE)
            
            # 1. Slide Window
            patches = []
            coords = []
            for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                    patch = img_tensor[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patches.append(patch)
                    coords.append((y, x))
            
            # 2. Batch Process Patches (Much faster)
            if not patches: continue
            batch = torch.stack(patches) # (N, 1, 32, 32)
            recon = model(batch)
            
            # 3. Calculate Error per Patch
            diff = (batch - recon).pow(2).mean(dim=1).squeeze() # (N, 32, 32)
            
            # 4. Reassemble Error Map
            for idx, (y, x) in enumerate(coords):
                error_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += diff[idx]
                counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
                
            # Average the overlapping areas
            error_map /= (counts + 1e-12)
            
            # 5. Scoring
            # Only consider the worst 1% of pixels (The Anomaly)
            # This ignores the 99% of normal background
            np_error = error_map.cpu().numpy()
            frame_score = np.percentile(np_error, 99.5) 
            max_scores.append(frame_score)
            
            # --- Visualization ---
            img_cv = cv2.imread(path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Create Heatmap
            # Normalize error for display (0.05 is an empirical high error for this data)
            norm_error = np.clip(np_error / 0.05 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_error, cv2.COLORMAP_JET)
            
            combined = np.hstack((img_cv, heatmap))
            pil_img = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_img)
            
            # Dynamic Threshold (Calibrate on first 30 frames)
            if i < 30:
                threshold = 1.0
            else:
                threshold = np.mean(max_scores[:30]) * 2.0 # 2x the normal baseline
            
            status = "NORMAL"
            color = (0, 255, 0)
            if i > 30 and frame_score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (img_cv.shape[1]-1, img_cv.shape[0]-1)], outline="red", width=4)

            draw.text((10, 10), f"PatchErr: {frame_score:.4f}", fill="white")
            draw.text((10, 30), f"Limit: {threshold:.4f}", fill="yellow")
            draw.text((10, 200), status, fill=color)
            
            gif_frames.append(pil_img)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()