import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from unet_lstm import RecurrentUNet

# Config
TEST_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004" 
MODEL_PATH = "models/unet_lstm.pth"
OUTPUT_FILE = "lstm_convex_hull.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIG ---
BOX_SIZE = 40 
THRESH_TRIGGER_COUNT = 200
THRESH_RESET_COUNT   = 100
THRESH_VIZ_BOX = 180

def detect():
    print(f"Running Detection with Convex Hull Visualization...")
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
    smooth_density_map = np.zeros((256, 256), dtype=np.float32)

    with torch.no_grad():
        for i in range(1, len(files)):
            # 1. Prediction Loop
            img_curr = transform(Image.open(files[i-1])).unsqueeze(0).to(DEVICE)
            img_target = transform(Image.open(files[i])).unsqueeze(0).to(DEVICE)
            pred, hidden_state = model.step(img_curr, hidden_state)
            
            # 2. Error Calculation
            diff = torch.abs(img_target - pred).cpu().numpy().squeeze()
            curr_np = img_target.squeeze().cpu().numpy()
            prev_np = img_curr.squeeze().cpu().numpy()
            mask = (np.abs(curr_np - prev_np) > 0.04).astype(float)
            masked_diff = diff * mask
            
            _, binary_map = cv2.threshold(masked_diff, 0.10, 1.0, cv2.THRESH_BINARY)
            binary_map_uint8 = (binary_map * 255).astype(np.uint8)
            closed_map = cv2.morphologyEx(binary_map_uint8, cv2.MORPH_CLOSE, kernel_close)
            
            density_map_raw = cv2.boxFilter(closed_map / 255.0, -1, (BOX_SIZE, BOX_SIZE), normalize=False)
            smooth_density_map = cv2.addWeighted(smooth_density_map, 0.6, density_map_raw.astype(np.float32), 0.4, 0)
            score = np.max(smooth_density_map)
            
            # --- VISUALIZATION ---
            img_cv = cv2.imread(files[i])
            img_cv = cv2.resize(img_cv, (256, 256))
            
            # Panel 1: Logic Heatmap
            heatmap_norm = np.clip(smooth_density_map / float(THRESH_TRIGGER_COUNT) * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # Panel 2: Inference Output
            vis_hud = img_cv.copy()
            overlay = vis_hud.copy()
            
            # A. CONVEX HULL FILL (The Fix for "Spotty" Paint)
            # Find contours in the high-detail map
            mask_paint = (density_map_raw > 100).astype(np.uint8)
            precise_mask = cv2.bitwise_and(closed_map, closed_map, mask=mask_paint)
            contours_paint, _ = cv2.findContours(precise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours_paint:
                if cv2.contourArea(cnt) > 20: # Filter tiny noise
                    # Calculate the "Shrink Wrap" shape
                    hull = cv2.convexHull(cnt)
                    # Draw the solid shape in RED on the overlay
                    cv2.drawContours(overlay, [hull], -1, (0, 0, 255), -1) 
            
            # Blend the solid shapes smoothly (Alpha = 0.4)
            cv2.addWeighted(overlay, 0.4, vis_hud, 0.6, 0, vis_hud)

            # B. Bounding Boxes
            mask_box = (smooth_density_map > THRESH_VIZ_BOX).astype(np.uint8)
            contours_box, _ = cv2.findContours(mask_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours_box:
                if cv2.contourArea(cnt) > 50:
                    x, y, w, h = cv2.boundingRect(cnt)
                    color = (0, 0, 255) 
                    
                    # Brackets
                    d = 10; t = 2
                    cv2.line(vis_hud, (x, y), (x+d, y), color, t)
                    cv2.line(vis_hud, (x, y), (x, y+d), color, t)
                    cv2.line(vis_hud, (x+w, y), (x+w-d, y), color, t)
                    cv2.line(vis_hud, (x+w, y), (x+w, y+d), color, t)
                    cv2.line(vis_hud, (x, y+h), (x+d, y+h), color, t)
                    cv2.line(vis_hud, (x, y+h), (x, y+h-d), color, t)
                    cv2.line(vis_hud, (x+w, y+h), (x+w-d, y+h), color, t)
                    cv2.line(vis_hud, (x+w, y+h), (x+w, y+h-d), color, t)

                    cv2.putText(vis_hud, "ANOMALY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # State Logic & Borders
            if not alarm_active:
                if score > THRESH_TRIGGER_COUNT: alarm_active = True
            else:
                if score < THRESH_RESET_COUNT: alarm_active = False
            
            status = "ALARM ACTIVE" if alarm_active else "MONITORING"
            border_color = (0, 0, 255) if alarm_active else (0, 255, 0)
            
            if alarm_active:
                for p in [heatmap, vis_hud]: cv2.rectangle(p, (0,0), (255,255), border_color, 5)

            # Combine
            combined = np.hstack((heatmap, vis_hud))
            pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f"Density Heatmap ({score:.0f})", fill="white")
            draw.text((266, 10), "Inference Output", fill="white")
            draw.text((266, 230), status, fill=border_color[::-1])
            
            gif_frames.append(pil_img)

    print(f"Saving Convex Hull GIF to {OUTPUT_FILE}...")
    gif_frames[0].save(OUTPUT_FILE, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == "__main__":
    detect()