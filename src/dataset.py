import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MovingMNISTDataset(Dataset[Tensor]):
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the .npy file containing the dataset.
        """
        # Load the data. Shape is (20, 10000, 64, 64) -> (Seq_Len, Num_Videos, H, W)
        # We transpose it to (Num_Videos, Seq_Len, H, W) for easier indexing
        self.data = np.load(data_path).transpose(1, 0, 2, 3)

        # Define the preprocessing pipeline for ResNet18
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # 1 -> 3 channels
                transforms.Resize((224, 224)),  # Resize to ResNet standard
                transforms.ToTensor(),  # Convert to Tensor (0-1)
                transforms.Normalize(  # ImageNet Normalization
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Get the sequence of frames for the video at 'idx'
        # Shape: (20, 64, 64)
        video_seq = self.data[idx]

        processed_frames = []

        # Process each frame individually
        for frame in video_seq:
            # frame is (64, 64), transform expects (H, W, C) or (H, W)
            # The transform pipeline handles the conversion
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)

        # Stack them into a single tensor: (Seq_Len, 3, 224, 224)
        return torch.stack(processed_frames)


if __name__ == "__main__":
    # Quick test to verify shapes
    dataset = MovingMNISTDataset("data/mnist_test_seq.npy")
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    # Expected: torch.Size([20, 3, 224, 224])
