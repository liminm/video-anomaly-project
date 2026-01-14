import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet_model import UNet

# Config
TEST_ROOT = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
MODEL_PATH = "models/unet_video.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    print(f"Evaluating Model on UCSD Ped2 Test Set using {DEVICE}...")
    
    # 1. Load Model
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # 2. Get all Test Folders (Test001, Test002...)
    test_folders = sorted(glob.glob(os.path.join(TEST_ROOT, "Test*")))
    # Exclude the "gt" folders if they exist in the list (sometimes 'Test001_gt' appears)
    test_folders = [f for f in test_folders if "_gt" not in f]
    
    all_scores = []
    all_labels = []

    print(f"Found {len(test_folders)} test videos.")

    with torch.no_grad():
        for folder in tqdm(test_folders):
            folder_name = os.path.basename(folder)
            
            # Load frames
            frame_paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
            
            # Load Ground Truth Masks (to know which frames are actually anomalies)
            # UCSD dataset structure: Test001_gt/001.bmp ...
            gt_folder = folder + "_gt"
            
            # Some UCSD versions have a single .m script defining anomalies, 
            # but usually they provide pixel masks or frame ranges.
            # For simpler Frame-Level AUC, we often rely on the provided label files 
            # or simply check if a GT mask exists and is non-zero.
            
            # Try to load pixel masks
            gt_paths = sorted(glob.glob(os.path.join(gt_folder, "*.bmp")))
            
            # Process video
            # We skip first 4 frames (history)
            for i in range(4, len(frame_paths)):
                # --- A. INFERENCE ---
                input_imgs = [transform(Image.open(frame_paths[i-4+j])) for j in range(4)]
                x = torch.cat(input_imgs, dim=0).unsqueeze(0).to(DEVICE)
                y_real = transform(Image.open(frame_paths[i])).unsqueeze(0).to(DEVICE)
                
                y_pred = model(x)
                
                # --- B. SCORING (Top-50 + Motion Mask) ---
                diff = torch.abs(y_real - y_pred).squeeze().cpu().numpy()
                
                curr_np = y_real.squeeze().cpu().numpy()
                prev_np = input_imgs[-1].squeeze().cpu().numpy()
                motion_mask = (np.abs(curr_np - prev_np) > 0.05).astype(float)
                masked_diff = diff * motion_mask
                
                # Score = Mean of Top 50 pixels
                top_k = np.sort(masked_diff.flatten())[::-1][:50]
                score = np.mean(top_k)
                all_scores.append(score)
                
                # --- C. GROUND TRUTH LABEL ---
                # Check if this frame is an anomaly
                # In UCSD, we check the corresponding GT mask file
                # If gt_folder has fewer files, we map by index
                is_anomaly = 0
                if i < len(gt_paths):
                    gt_mask = cv2.imread(gt_paths[i], 0) # Read as grayscale
                    if gt_mask is not None and np.sum(gt_mask) > 0:
                        is_anomaly = 1
                
                all_labels.append(is_anomaly)

    # 3. Calculate Metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Handle NaNs
    if np.isnan(all_scores).any():
        print("Warning: NaNs found in scores. Replacing with 0.")
        all_scores = np.nan_to_num(all_scores)

    # AUC-ROC
    auc = roc_auc_score(all_labels, all_scores)
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("="*40)
    print(f"Frame-Level AUC: {auc:.4f}")
    print("="*40)
    
    # 4. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved ROC curve to roc_curve.png")

if __name__ == "__main__":
    evaluate()