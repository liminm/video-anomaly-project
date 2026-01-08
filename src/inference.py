import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2
import os

from src.model import VideoPredictor

class AnomalyDetector:
    def __init__(self, model_path="models/lstm_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Loading inference pipeline on {self.device}...")

        # 1. Load the ResNet Feature Extractor
        self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Identity() # Remove classification layer
        self.encoder.to(self.device)
        self.encoder.eval()

        # 2. Load the LSTM Predictor
        self.predictor = VideoPredictor().to(self.device)
        self.predictor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictor.eval()

        # 3. Define the Transform (Same as training!)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_video(self, video_path):
        """
        Reads a video file (mp4, avi) and converts it to a tensor of shape (Seq, 3, 224, 224)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transforms
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
            
        cap.release()
        
        if not frames:
            raise ValueError("No frames read from video")

        # Stack into (Seq, 3, 224, 224)
        return torch.stack(frames)

    def predict(self, video_path):
        """
        Returns a list of anomaly scores for the video.
        """
        # 1. Get raw frames
        video_tensor = self.preprocess_video(video_path)
        seq_len = len(video_tensor)
        
        if seq_len < 2:
            return {"error": "Video too short. Need at least 2 frames."}

        # 2. Extract Features
        # Process in batches to avoid OOM
        features = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, seq_len, batch_size):
                batch = video_tensor[i:i+batch_size].to(self.device)
                feat = self.encoder(batch)
                features.append(feat)
        
        features = torch.cat(features) # Shape: (Seq_Len, 512)

        # 3. Run LSTM Prediction
        # We need to feed frames 0..N-1 to predict 1..N
        input_seq = features[:-1].unsqueeze(0) # Add batch dim: (1, Seq-1, 512)
        target_seq = features[1:].unsqueeze(0) # Add batch dim: (1, Seq-1, 512)

        with torch.no_grad():
            # LSTM returns predictions for the entire sequence
            predictions = self.predictor(input_seq) 
            # Note: Our model definition currently returns only the *last* step.
            # We need to tweak it slightly if we want the full sequence, 
            # but for now, let's just predict the NEXT frame based on the window.
            
            # For simplicity in this project version:
            # Let's just predict the final frame surprise to keep it simple.
            pass

        # Calculate Error (MSE) between Prediction and Reality
        # We will iterate through the video with a sliding window of 10 frames
        scores = []
        window_size = 10
        
        if seq_len <= window_size:
             return {"error": f"Video too short. Need > {window_size} frames."}

        with torch.no_grad():
            for t in range(seq_len - window_size):
                # Input: Frames t to t+9
                window = features[t : t+window_size].unsqueeze(0) 
                
                # Predict Frame t+10
                predicted_vector = self.predictor(window)
                
                # Actual Frame t+10
                actual_vector = features[t+window_size].unsqueeze(0)
                
                # Calculate distance (MSE)
                loss = nn.MSELoss()(predicted_vector, actual_vector)
                scores.append(loss.item())

        return {"anomaly_scores": scores, "max_anomaly": max(scores)}