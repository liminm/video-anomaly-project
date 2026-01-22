import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from unet_lstm import RecurrentUNet

# Defaults
DATA_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
MODEL_PATH = "models/unet_lstm.pth"
ONNX_PATH = "models/unet_lstm.onnx"
BATCH_SIZE = 4   # LSTMs use more VRAM, so we lower the batch size
SEQ_LEN = 8      # Train on clips of 8 frames
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UCSDLSTMDataset(Dataset):
    def __init__(self, root_dir: str | Path, seq_len: int):
        self.samples = []
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        root_dir = Path(root_dir)
        folders = sorted([p for p in root_dir.iterdir() if p.is_dir()])
        for folder in folders:
            files = sorted(folder.glob("*.tif"))
            if len(files) <= seq_len:
                continue
            # Sliding window of SEQ_LEN + 1 (Input + Target)
            for i in range(len(files) - seq_len - 1):
                self.samples.append(files[i : i + seq_len + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        frames = [self.transform(Image.open(p)) for p in paths]
        
        # Stack into (Seq_Len+1, 1, H, W)
        video_tensor = torch.stack(frames, dim=0)
        
        # Input: Frames 0 to N-1
        # Target: Frames 1 to N
        x = video_tensor[:-1]
        y = video_tensor[1:]

        return x, y

@dataclass
class TrainConfig:
    data_dir: str
    model_path: str
    onnx_path: str
    batch_size: int
    seq_len: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    val_split: float
    seed: int
    num_workers: int
    hidden_channels: int
    lstm_layers: int
    dropout: float
    max_steps: int | None
    save_dir: str | None


def _resolve_paths(config: TrainConfig):
    model_path = Path(config.model_path)
    onnx_path = Path(config.onnx_path)
    metrics_path = None
    if config.save_dir:
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "unet_lstm.pth"
        onnx_path = save_dir / "unet_lstm.onnx"
        metrics_path = save_dir / "metrics.json"
    return model_path, onnx_path, metrics_path


def _epoch_loss(model, loader, device, criterion, train=True):
    total_loss = 0.0
    count = 0
    if train:
        model.train()
    else:
        model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            preds = model(x)
        else:
            with torch.no_grad():
                preds = model(x)
        loss = criterion(preds, y)
        if train:
            loss.backward()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def train_model(config: TrainConfig) -> dict:
    device = config.device
    print(f"Training Recurrent U-Net (LSTM) on {device}...")

    dataset = UCSDLSTMDataset(config.data_dir, config.seq_len)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {config.data_dir}")

    if config.val_split < 0 or config.val_split >= 1:
        raise ValueError("val_split must be in [0, 1)")

    generator = torch.Generator().manual_seed(config.seed)
    val_len = int(len(dataset) * config.val_split)
    train_len = len(dataset) - val_len

    if val_len > 0:
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    else:
        train_set = dataset
        val_loader = None

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = RecurrentUNet(
        hidden_channels=config.hidden_channels,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    history = []
    best_val = None

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if i % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.5f}")
            if config.max_steps is not None and i + 1 >= config.max_steps:
                break

        train_loss = total_loss / max(steps, 1)
        val_loss = None
        if val_loader is not None:
            val_loss = _epoch_loss(model, val_loader, device, criterion, train=False)
            best_val = val_loss if best_val is None else min(best_val, val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"Epoch {epoch+1} Average Loss: {train_loss:.5f}")
        if val_loss is not None:
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.5f}")

    model_path, onnx_path, metrics_path = _resolve_paths(config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Export final model to ONNX
    model.eval()
    model_cpu = model.to("cpu")
    dummy_input = torch.zeros(1, config.seq_len, 1, 256, 256, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )

    summary = {
        "config": asdict(config),
        "history": history,
        "best_val_loss": best_val,
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if metrics_path:
        metrics_path.write_text(json.dumps(summary, indent=2))

    print(f"ONNX model saved to {onnx_path}")
    return summary


def default_config() -> TrainConfig:
    return TrainConfig(
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        onnx_path=ONNX_PATH,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        epochs=EPOCHS,
        lr=5e-4,
        weight_decay=0.0,
        device=DEVICE,
        val_split=0.1,
        seed=42,
        num_workers=0,
        hidden_channels=128,
        lstm_layers=2,
        dropout=0.0,
        max_steps=None,
        save_dir=None,
    )
    print(f"ONNX model saved to {ONNX_PATH}")

if __name__ == "__main__":
    train_model(default_config())
