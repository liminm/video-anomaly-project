import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from src.model import VideoPredictor, VideoDecoder
from src.dataset import MovingMNISTDataset

# Paths
DATA_PATH = "data/mnist_test_seq.npy"
LSTM_PATH = "models/lstm_model.pth"
DECODER_PATH = "models/decoder.pth"
OUTPUT_DIR = "generated_results"

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_prediction():
    print(f"Running generation on {DEVICE}...")
    ensure_dir(OUTPUT_DIR)

    # 1. Load Models
    print("Loading models...")
    
    # Encoder (ResNet)
    encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder.to(DEVICE)
    encoder.eval()
    
    # Predictor (LSTM)
    lstm = VideoPredictor().to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
    lstm.eval()
    
    # Decoder (Generator)
    decoder = VideoDecoder().to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    decoder.eval()

    # 2. Get a Sample Video
    # We grab index 0 just for a consistent test, or use random.randint
    dataset = MovingMNISTDataset(DATA_PATH)
    sample_idx = np.random.randint(0, len(dataset))
    video_tensor = dataset[sample_idx] # Shape: (20, 3, 224, 224)
    print(f"Generating forecast for video index {sample_idx}...")

    # 3. The "Past" (First 10 frames)
    # We feed these into the LSTM to build up memory
    past_frames = video_tensor[:10].to(DEVICE)
    
    with torch.no_grad():
        # Encode Past Frames -> Vectors
        # Process in batch: (10, 3, 224, 224) -> (10, 512)
        past_vectors = encoder(past_frames)
        
        # Add batch dimension: (1, 10, 512)
        past_vectors = past_vectors.unsqueeze(0)
        
        # Run LSTM on the past to warm up the hidden state
        # We don't care about the output yet, just the internal state
        _, (hidden, cell) = lstm.lstm(past_vectors)
        
        # The last vector from the past is our starting point for the future
        current_vector = past_vectors[:, -1, :] # Shape (1, 512)

    # 4. The "Future" (Generate next 10 frames)
    generated_frames = []
    
    print("Hallucinating future frames...")
    with torch.no_grad():
        for i in range(10):
            # A. Predict NEXT vector using LSTM head
            # We pass the hidden state manually to step forward one by one
            # Note: Our VideoPredictor class simplifies this, but for autoregression
            # we need to feed the prediction back in. 
            
            # Simple step: Pass current vector + hidden state
            # Reshape input to (Batch, Seq, Feature) -> (1, 1, 512)
            lstm_input = current_vector.unsqueeze(1)
            
            lstm_out, (hidden, cell) = lstm.lstm(lstm_input, (hidden, cell))
            
            # Map LSTM output to Feature Vector
            next_vector = lstm.head(lstm_out[:, -1, :])
            
            # B. Decode Vector -> Image
            generated_image = decoder(next_vector) # Shape (1, 3, 64, 64)
            
            # Save for GIF (Convert to numpy, 0-255 format)
            img_np = generated_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            generated_frames.append(img_np)
            
            # C. Update current_vector for the next step
            # This is "Autoregressive" generation: the model consumes its own prediction
            current_vector = next_vector

    # 5. Save as GIF
    save_gif(sample_idx, dataset, generated_frames)

def save_gif(idx, dataset, generated_frames):
    # Get the ACTUAL future frames for comparison
    original_video = dataset.data[idx] # (20, 64, 64) raw numpy
    
    # Prepare comparison images
    frames_for_gif = []
    
    for i in range(10):
        # Top: Actual Future Frame (Frame 10 to 19)
        # Convert raw (64,64) -> RGB (64,64,3)
        real_frame = original_video[10 + i]
        real_frame = np.stack([real_frame]*3, axis=-1) 
        
        # Bottom: Generated Frame
        gen_frame = generated_frames[i]
        
        # Stitch them vertically
        # Add labels (optional/simple text)
        combined = np.vstack((real_frame, gen_frame))
        frames_for_gif.append(combined)

    # Save using OpenCV. Use MJPG/AVI for broader codec compatibility.
    out_path = os.path.join(OUTPUT_DIR, f"prediction_{idx}.avi")
    height, width, layers = frames_for_gif[0].shape
    
    # MJPG is widely supported and avoids missing mp4v/ffmpeg issues
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(out_path, fourcc, 5, (width, height)) # 5 FPS
    if not video.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check OpenCV codecs/FFmpeg support.")

    for frame in frames_for_gif:
        # OpenCV expects BGR
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Generated video saved to {out_path}")
    print("Top half = REAL, Bottom half = GENERATED")

if __name__ == "__main__":
    generate_prediction()