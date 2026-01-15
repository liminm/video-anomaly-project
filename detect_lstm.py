import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from unet_lstm import RecurrentUNet

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001" 
MODEL_PATH = "models/unet_lstm.pth"
OUTPUT_FILE = "lstm_pro_viz.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIG ---
BOX_SIZE = 40 
THRESH_TRIGGER_COUNT = 150
THRESH_RESET_COUNT   = 80

def detect():
    print(f"Running Detection with Professional HUD Visualization...")
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
    hidden_state = None 
    alarm_active = False
    
    kernel_close = np.ones((5,5), np.uint8)

    with torch.no_grad():
        for i in range(1, len(files)):
            # 1. Inputs
            img_curr = transform(Image.open(files[i-1])).unsqueeze(0).to(DEVICE)
            img_target = transform(Image.open(files[i])).unsqueeze(0).to(DEVICE)
            
            # 2. Prediction
            pred, hidden_state = model.step(img_curr, hidden_state)
            
            # 3. Raw Error & Masking
            diff = torch.abs(img_target - pred).cpu().numpy().squeeze()
            curr_np = img_target.squeeze().cpu().numpy()
            prev_np = img_curr.squeeze().cpu().numpy()
            mask = (np.abs(curr_np - prev_np) > 0.04).astype(float)
            masked_diff = diff * mask
            
            # 4. Processing
            _, binary_map = cv2.threshold(masked_diff, 0.10, 1.0, cv2.THRESH_BINARY)
            binary_map_uint8 = (binary_map * 255).astype(np.uint8)
            closed_map = cv2.morphologyEx(binary_map_uint8, cv2.MORPH_CLOSE, kernel_close)
            density_map = cv2.boxFilter(closed_map / 255.0, -1, (BOX_SIZE, BOX_SIZE), normalize=False)
            score = np.max(density_map)
            
            # --- VISUALIZATION (THE UPGRADE) ---
            img_cv = cv2.imread(files[i])
            img_cv = cv2.resize(img_cv, (256, 256))
            
            # PANEL 2: Heatmap
            heatmap_norm = np.clip(density_map / float(THRESH_TRIGGER_COUNT) * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # PANEL 3: Professional HUD
            # Start with a clean copy of the original image
            vis_hud = img_cv.copy()
            
            # 1. Define the Anomaly Mask
            region_of_interest = (density_map > THRESH_RESET_COUNT).astype(np.uint8)
            precise_mask = cv2.bitwise_and(closed_map, closed_map, mask=region_of_interest)
            
            # 2. Alpha Blending (Semi-Transparent Overlay)
            # Create a red overlay layer
            overlay = vis_hud.copy()
            overlay[precise_mask > 0] = [0, 0, 255] # Paint red
            
            # Blend it: 0.7 * Original + 0.3 * Red Overlay
            cv2.addWeighted(overlay, 0.4, vis_hud, 0.6, 0, vis_hud)
            
            # 3. Contours & Bounding Boxes
            # Find shapes in the precise mask
            contours, _ = cv2.findContours(precise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Filter noise
                if cv2.contourArea(cnt) > 20:
                    # Draw solid outline (Thickness 1)
                    cv2.drawContours(vis_hud, [cnt], -1, (0, 0, 255), 1)
                    
                    # Draw Bounding Box (The "Target Lock" look)
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Add some padding
                    pad = 5
                    cv2.rectangle(vis_hud, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 2)
                    
                    # Add Label Tag
                    cv2.putText(vis_hud, "ANOMALY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 4. State Update
            if not alarm_active:
                if score > THRESH_TRIGGER_COUNT:
                    alarm_active = True
            else:
                if score < THRESH_RESET_COUNT:
                    alarm_active = False
            
            # 5. Borders
            status = "ANOMALY" if alarm_active else "NORMAL"
            border_color = (0, 0, 255) if alarm_active else (0, 255, 0)
            
            if alarm_active:
                for p in [img_cv, heatmap, vis_hud]:
                    cv2.rectangle(p, (0,0), (255,255), border_color, 5)

            # Combine
            combined = np.hstack((img_cv, heatmap, vis_hud))
            pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Header Text
            draw.text((10, 10), "Input Stream", fill="white")
            draw.text((266, 10), f"Logic Heatmap ({score:.0f})", fill="white")
            draw.text((522, 10), "Augmented Output", fill="white")
            draw.text((10, 230), status, fill=border_color[::-1])
            
            gif_frames.append(pil_img)

    print(f"Saving Professional GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()