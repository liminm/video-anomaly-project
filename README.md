# UCSD LSTM Video Anomaly Detection

Detect unusual events in surveillance footage by learning normal pedestrian
motion and flagging frames where reconstruction error spikes.

Deployed app: https://video-anomaly-app-277455458775.europe-west3.run.app

![Clip selection](images/selection.gif)
Caption: Select a test clip from the dropdown and launch detection.

![Results panel](images/result.png)
Caption: Results summary with max score, alarm frames, and the score timeline.

![Anomaly visualization](images/video.gif)
Caption: The video overlay highlights detected anomalies (e.g., a bike or cart).

## Problem description
Manual monitoring of CCTV streams is error-prone and expensive. This project
targets automated anomaly detection in pedestrian walkways (UCSD Ped2), where
anomalies include bikes, carts, and other non-pedestrian motion. The system is
trained only on normal pedestrian behavior and flags abnormal events by
measuring how poorly a model can predict the next frame. The output is a
frame-level anomaly score plus visual overlays (heatmap and bounding cues) to
help operators quickly review suspicious segments.

## Dataset
- UCSD Anomaly Detection Dataset (Ped2 by default; Ped1 also supported).
- Train: normal pedestrian-only clips.
- Test: contains anomalous events plus ground-truth masks.

Download and extract:
```bash
python download_ucsd.py
```

Expected layout:
```
data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/
  Train/Train001/*.tif
  Test/Test001/*.tif
  Test/Test001_gt/*.bmp   # depending on dataset layout
```

## Dataset details
- UCSD provides multiple subsets (Ped1, Ped2) with different scenes.
- Each clip is a short sequence of frames; test clips include anomaly masks.
- Ground truth masks label anomalous objects, not just lighting shifts.

## Approach
- Model: Recurrent U-Net with ConvLSTM bottleneck (`unet_lstm.py`).
- Training: predict the next frame from a sequence of previous frames.
- Scoring: pixel reconstruction error masked by motion, smoothed into a density
  map; thresholds trigger alarms and visual overlays.
- Serving: export to ONNX and run inference with `onnxruntime` for fast API
  responses.

## Application architecture
- Data layer: UCSD frames in `data/`.
- Notebook layer: `notebooks/notebook.ipynb` for EDA, feature analysis, and
  model experiments.
- Model layer: ConvLSTM U-Net exported to `models/unet_lstm.onnx`.
- Inference helper: `src/lstm_detector.py` runs ONNX inference and produces
  scores, overlays, and GIFs.
- API layer: FastAPI in `app/main.py`.
- UI layer: Streamlit in `app/streamlit_app.py`.
- Container: `Dockerfile` runs API + UI via `scripts/start.sh`.

## Why a ConvLSTM U-Net
- The task is temporal: anomalies are defined by unusual motion patterns.
- ConvLSTM captures sequence dynamics while preserving spatial structure.
- Reconstruction error is a natural signal for "unexpected" frames.

## Repository layout
- `data/` - UCSD Ped1/Ped2 frames (downloaded).
- `notebooks/notebook.ipynb` - EDA, feature analysis, training, evaluation.
- `train.py` - training + ONNX export (mirrors notebook).
- `run_experiments.py` - hyperparameter sweep (hardcoded search space).
- `unet_lstm.py`, `conv_lstm.py` - model definition.
- `src/lstm_detector.py` - inference + GIF generation.
- `app/main.py` - FastAPI web service.
- `app/streamlit_app.py` - Streamlit UI.
- `models/` - model weights and ONNX export.
- `generated_results/` - GIFs and experiment outputs.
- `requirements.txt`, `requirements.app.txt` - dependencies.

## Results and model selection
- Frame-level ROC AUC is computed in the notebook evaluation section.
- Example result on a small subset of clips: AUC ~= 1.00 (see notebook cell).
- Model selection uses validation loss from the hyperparameter sweep.

Search space (current sweep):
- `hidden_channels`: [128, 256]
- `lstm_layers`: [1, 2]
- `dropout`: [0.0, 0.3]
- `lr`: [5e-4, 1e-3]
- `weight_decay`: [0.0, 1e-4]
- `batch_size`: [4]
- `seq_len`: [8]

## Setup
Python 3.11+ recommended.

Full dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API/UI only:
```bash
pip install -r requirements.app.txt
```

## EDA and feature analysis
```bash
jupyter notebook notebooks/notebook.ipynb
```

## Training
```bash
python train.py
```
Outputs:
- `models/unet_lstm.pth`
- `models/unet_lstm.onnx`

## Hyperparameter sweep
```bash
python run_experiments.py
```
Outputs:
- `generated_results/experiments/summary.csv`
- per-run `metrics.json` and `unet_lstm.onnx`

## Evaluation
Use the evaluation section in `notebooks/notebook.ipynb` to compute frame-level
ROC AUC using ground-truth masks.

## Web service (local)
Start the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Start the UI in another terminal:
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Sample API calls:
```bash
curl http://127.0.0.1:8001/clips
curl -X POST http://127.0.0.1:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"clip": "Test001", "save_gif": true, "stride": 1}'
```

Environment variables (optional):
- `UCSD_CLIP_ROOT`: override dataset path.
- `LSTM_MODEL_PATH`: ONNX model path (default `models/unet_lstm.onnx`).
- `LSTM_OUTPUT_DIR`: output folder for GIFs.
- `GCS_BUCKET`, `GCS_PREFIX`, `GCS_CACHE_DIR`: load clips from Google Cloud
  Storage instead of local disk.

## Docker
Build:
```bash
docker build -t video-anomaly-app .
```

Run (mount data + outputs, serve API + UI):
```bash
docker run --rm \
  -p 8001:8001 -p 8501:8501 \
  -e UCSD_CLIP_ROOT=/app/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test \
  -v "$PWD/data:/app/data" \
  -v "$PWD/generated_results:/app/generated_results" \
  video-anomaly-app
```

Open the UI at:
```
http://localhost:8501
```

Optional (load clips from GCS instead of local disk):
```
-e GCS_BUCKET=... -e GCS_PREFIX=... -e GCS_CACHE_DIR=/app/data/gcs_cache
```

## Cloud deployment (Google Cloud Run)
Example deploy:
```bash
gcloud run deploy video-anomaly-app \
  --region europe-west3 \
  --source . \
  --port 8501 \
  --memory 4Gi \
  --timeout 900 \
  --concurrency 10 \
  --min-instances 1 \
  --set-env-vars API_PORT=8001,UI_PORT=8501,GCS_BUCKET=YOUR_BUCKET,GCS_PREFIX=YOUR_PREFIX
```

Fetch the service URL:
```bash
gcloud run services describe video-anomaly-app --region europe-west3 --format='value(status.url)'
```

## Notes
- The dataset is not committed; use `download_ucsd.py` or your own copy.
- Generated GIFs and experiment outputs live under `generated_results/`.
