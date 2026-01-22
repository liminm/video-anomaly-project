import csv
from itertools import product
from pathlib import Path

import torch

from train import TrainConfig, train_model

ROOT = Path(__file__).resolve().parents[0]

print(f"Using ROOT directory: {ROOT}")

DATA_DIR = (
    ROOT / "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    if (ROOT / "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train").exists()
    else ROOT / "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
)
SAVE_ROOT = ROOT / "generated_results/experiments"

EPOCHS = 2
VAL_SPLIT = 0.1
MAX_STEPS = None
DEVICE = None


SEARCH_SPACE = {
    "hidden_channels": [128, 256],
    "lstm_layers": [1, 2],
    "dropout": [0.0, 0.3],
    "lr": [5e-4, 1e-3],
    "weight_decay": [0.0, 1e-4],
    "batch_size": [4],
    "seq_len": [8],
}

def main() -> None:
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    grid = product(
        SEARCH_SPACE["hidden_channels"],
        SEARCH_SPACE["lstm_layers"],
        SEARCH_SPACE["dropout"],
        SEARCH_SPACE["lr"],
        SEARCH_SPACE["weight_decay"],
        SEARCH_SPACE["batch_size"],
        SEARCH_SPACE["seq_len"],
    )

    summary_rows = []

    for hidden_ch, layers, dropout, lr, weight_decay, batch_size, seq_len in grid:
        run_name = (
            f"hc{hidden_ch}_l{layers}_do{dropout:.2f}_"
            f"lr{lr:.0e}_wd{weight_decay:.0e}_bs{batch_size}_seq{seq_len}"
        )
        run_dir = SAVE_ROOT / run_name
        metrics_path = run_dir / "metrics.json"

        config = TrainConfig(
            data_dir=str(DATA_DIR),
            model_path=str(ROOT / "models/unet_lstm.pth"),
            onnx_path=str(ROOT / "models/unet_lstm.onnx"),
            batch_size=batch_size,
            seq_len=seq_len,
            epochs=EPOCHS,
            lr=lr,
            weight_decay=weight_decay,
            device=DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"),
            val_split=VAL_SPLIT,
            seed=42,
            num_workers=0,
            hidden_channels=hidden_ch,
            lstm_layers=layers,
            dropout=dropout,
            max_steps=MAX_STEPS,
            save_dir=str(run_dir),
        )
        metrics = train_model(config)
        history = metrics.get("history", [])
        summary_rows.append(
            {
                "run": run_name,
                "hidden_channels": hidden_ch,
                "lstm_layers": layers,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "final_val_loss": history[-1]["val_loss"] if history else None,
                "best_val_loss": metrics.get("best_val_loss"),
                "metrics_path": str(metrics_path),
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = SAVE_ROOT / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
