import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from src.model import VideoPredictor

# Configuration
TEST_VIDEO_PATH = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"
MODEL_PATH = "models/lstm_ucsd_cosine.pth"
SEQ_LEN = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def load_models():
    print(f"Loading models on {DEVICE}...")
    
    # 1. Encoder (ResNet)
    encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder.to(DEVICE).eval()
    
    # 2. Predictor (LSTM)
    lstm = VideoPredictor().to(DEVICE)
    lstm.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    lstm.eval()
    
    return encoder, lstm

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def run_inference():
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"Error: Test video path {TEST_VIDEO_PATH} not found.")
        return

    encoder, lstm = load_models()
    transform = get_transforms()
    
    # 1. Load Frames
    print(f"Analyzing {TEST_VIDEO_PATH}...")
    frame_paths = sorted(glob.glob(os.path.join(TEST_VIDEO_PATH, "*.tif")))
    frames = []
    
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        frames.append(transform(img))
        
    # Stack: (Total_Frames, 3, 224, 224)
    video_tensor = torch.stack(frames).to(DEVICE)
    
    # 2. Extract Features
    print("Extracting features...")
    features = []
    BATCH_SIZE = 32
    with torch.no_grad():
        for i in range(0, len(video_tensor), BATCH_SIZE):
            batch = video_tensor[i : i+BATCH_SIZE]
            feat = encoder(batch)
            features.append(feat)
    
    # Shape: (Total_Frames, 512)
    features = torch.cat(features)
    
    # 3. Calculate Anomaly Scores
    print("Calculating anomaly scores...")
    scores = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # Iterate through video using sliding window
        for i in range(len(features) - SEQ_LEN):
            # Input: Frames i to i+9
            input_seq = features[i : i+SEQ_LEN].unsqueeze(0) # (1, 10, 512)
            
            # Predict: Frame i+10
            prediction = lstm(input_seq) # (1, 512)
            
            # Actual: Frame i+10
            actual = features[i+SEQ_LEN].unsqueeze(0) # (1, 512)
            
            # Error
            loss = criterion(prediction, actual)
            scores.append(loss.item())

    # 4. Visualization
    plot_results(scores)

def plot_results(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Reconstruction Error (MSE)", color="red")
    
    # Add a threshold line (visual estimation)
    plt.axhline(y=0.005, color='blue', linestyle='--', label="Normal Threshold")
    
    plt.title(f"Anomaly Detection: UCSD Ped2 Test001")
    plt.xlabel("Frame Number")
    plt.ylabel("Model Surprise (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "anomaly_plot.png"
    plt.savefig(output_file)
    print(f"Inference complete! Plot saved to {output_file}")
    print("Check the plot: Peaks indicate anomalies (bikers/carts).")

if __name__ == "__main__":
    run_inference()