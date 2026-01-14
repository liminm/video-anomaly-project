import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from model_3d import C3DAutoencoder

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
MODEL_PATH = "models/c3d_ae.pth"
OUTPUT_FILE = "c3d_result.gif"
CLIP_LEN = 16
PATCH_SIZE = 32
STRIDE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def detect():
    print("Running 3D Spatiotemporal Detection...")
    model = C3DAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.tif")))
    
    gif_frames = []
    max_scores = [] 

    with torch.no_grad():
        # We process the video in chunks of CLIP_LEN
        # We start at frame 16 and look back
        for i in range(CLIP_LEN, len(files)):
            # 1. Load CLIP history
            clip_imgs = []
            for j in range(CLIP_LEN):
                img = Image.open(files[i - CLIP_LEN + j + 1])
                clip_imgs.append(transform(img))
            
            # Full Frame Clip: (1, 16, H, W)
            full_clip = torch.stack(clip_imgs, dim=1).to(DEVICE)
            _, _, h, w = full_clip.shape
            
            error_map = torch.zeros((h, w), device=DEVICE)
            counts = torch.zeros((h, w), device=DEVICE)
            
            # 2. Extract 3D Patches
            patches = []
            coords = []
            
            for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                    patch = full_clip[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patches.append(patch)
                    coords.append((y, x))
            
            # 3. Batch Process
            if not patches: continue
            batch = torch.stack(patches) # (N, 1, 16, 32, 32)
            recon = model(batch)
            
            # 4. Calculate Error (MSE over the whole cube volume)
            # We average over Time (dim 2) to get a 2D error map for this specific block
            diff_cube = (batch - recon).abs() # L1 error
            diff_score = diff_cube.mean(dim=(1, 2)).squeeze() # (N, 32, 32) Average over Time/Channels
            
            # 5. Reassemble
            for idx, (y, x) in enumerate(coords):
                # Ignore edges of the patch (padding artifacts)
                border = 4
                patch_err = diff_score[idx]
                error_map[y+border:y+PATCH_SIZE-border, x+border:x+PATCH_SIZE-border] += patch_err[border:-border, border:-border]
                counts[y+border:y+PATCH_SIZE-border, x+border:x+PATCH_SIZE-border] += 1
                
            error_map /= (counts + 1e-12)
            
            # 6. Scoring (Max Error Strategy)
            np_error = error_map.cpu().numpy()
            # Focus on the worst region
            frame_score = np.percentile(np_error, 99.9) 
            max_scores.append(frame_score)
            
            # --- Visualization ---
            current_frame_path = files[i]
            img_cv = cv2.imread(current_frame_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Heatmap
            norm_error = np.clip(np_error / 0.02 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_error, cv2.COLORMAP_JET)
            
            combined = np.hstack((img_cv, heatmap))
            pil_img = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_img)
            
            # Threshold
            if i < 40:
                threshold = 1.0
            else:
                threshold = np.mean(max_scores[:30]) * 1.5
            
            status = "NORMAL"
            color = (0, 255, 0)
            if i > 40 and frame_score > threshold:
                status = "ANOMALY"
                color = (255, 0, 0)
                draw.rectangle([(0,0), (w-1, h-1)], outline="red", width=4)

            draw.text((10, 10), f"3D-Err: {frame_score:.4f}", fill="white")
            draw.text((10, 30), f"Limit: {threshold:.4f}", fill="yellow")
            draw.text((10, 200), status, fill=color)
            
            gif_frames.append(pil_img)

    print(f"Saving GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()