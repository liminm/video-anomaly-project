import torch
import cv2
import glob
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet_model import UNet
from unet_lstm import RecurrentUNet

# Config
TEST_ROOT = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
MODEL_UNET = "models/unet_video.pth"
MODEL_LSTM = "models/unet_lstm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print("Loading models...")
    # Load Standard U-Net
    unet = UNet().to(DEVICE)
    unet.load_state_dict(torch.load(MODEL_UNET, map_location=DEVICE))
    unet.eval()
    
    # Load LSTM U-Net
    lstm = RecurrentUNet().to(DEVICE)
    lstm.load_state_dict(torch.load(MODEL_LSTM, map_location=DEVICE))
    lstm.eval()
    
    return unet, lstm

def calculate_score(pred, target, prev_frame):
    # 1. Raw Error
    diff = torch.abs(target - pred).squeeze().cpu().numpy()
    
    # 2. Motion Mask (Ignore static background)
    curr_np = target.squeeze().cpu().numpy()
    prev_np = prev_frame.squeeze().cpu().numpy()
    mask = (np.abs(curr_np - prev_np) > 0.05).astype(float)
    masked_diff = diff * mask
    
    # 3. Top-50 Score
    score = np.mean(np.sort(masked_diff.flatten())[::-1][:50])
    return score

def compare():
    unet, lstm = load_models()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    test_folders = sorted(glob.glob(os.path.join(TEST_ROOT, "Test*")))
    test_folders = [f for f in test_folders if "_gt" not in f]
    
    scores_unet = []
    scores_lstm = []
    labels = []
    
    print(f"Comparing models on {len(test_folders)} videos...")

    with torch.no_grad():
        for folder in tqdm(test_folders):
            frame_paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
            gt_folder = folder + "_gt"
            gt_paths = sorted(glob.glob(os.path.join(gt_folder, "*.bmp")))
            
            # Reset LSTM state for new video
            hidden_state = None
            
            # We iterate through the video
            for i in range(1, len(frame_paths)):
                # Load Tensors
                img_prev = transform(Image.open(frame_paths[i-1])).unsqueeze(0).to(DEVICE)
                img_curr = transform(Image.open(frame_paths[i])).unsqueeze(0).to(DEVICE)
                
                # --- LSTM INFERENCE ---
                # We run this for EVERY frame (1 to N) to maintain memory state
                pred_lstm, hidden_state = lstm.step(img_prev, hidden_state)
                
                # --- COMPARISON START ---
                # We can only run U-Net starting at frame 4 (needs 4 history frames)
                if i >= 4:
                    # 1. Prepare U-Net Input (t-4 to t-1)
                    stack = []
                    for j in range(4):
                        stack.append(transform(Image.open(frame_paths[i-4+j])))
                    x_unet = torch.cat(stack, dim=0).unsqueeze(0).to(DEVICE)
                    
                    # 2. U-Net Inference
                    pred_unet = unet(x_unet)
                    
                    # 3. Calculate Scores
                    # Both models try to predict 'img_curr'
                    # We pass 'img_prev' for motion masking logic
                    s_unet = calculate_score(pred_unet, img_curr, img_prev)
                    s_lstm = calculate_score(pred_lstm, img_curr, img_prev)
                    
                    scores_unet.append(s_unet)
                    scores_lstm.append(s_lstm)
                    
                    # 4. Get Ground Truth
                    is_anomaly = 0
                    if i < len(gt_paths):
                        mask = cv2.imread(gt_paths[i], 0)
                        if mask is not None and np.sum(mask) > 0:
                            is_anomaly = 1
                    labels.append(is_anomaly)

    # --- METRICS & PLOTTING ---
    scores_unet = np.array(scores_unet)
    scores_lstm = np.array(scores_lstm)
    labels = np.array(labels)
    
    # Replace NaNs if any
    scores_unet = np.nan_to_num(scores_unet)
    scores_lstm = np.nan_to_num(scores_lstm)

    auc_unet = roc_auc_score(labels, scores_unet)
    auc_lstm = roc_auc_score(labels, scores_lstm)
    
    print("\n" + "="*40)
    print(f"COMPARISON RESULTS")
    print("="*40)
    print(f"Standard U-Net AUC: {auc_unet:.4f}")
    print(f"LSTM U-Net AUC:     {auc_lstm:.4f}")
    print("="*40)
    
    # Plot
    fpr_u, tpr_u, _ = roc_curve(labels, scores_unet)
    fpr_l, tpr_l, _ = roc_curve(labels, scores_lstm)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_u, tpr_u, label=f'Standard U-Net (AUC={auc_unet:.3f})', linewidth=2)
    plt.plot(fpr_l, tpr_l, label=f'LSTM U-Net (AUC={auc_lstm:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Comparison: TCN vs LSTM')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = "model_comparison.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    compare()