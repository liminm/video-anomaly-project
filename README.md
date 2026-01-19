# Video Anomaly Detection on UCSD Ped2

Detect unusual events (bikers, carts, non-pedestrian motion) in surveillance
footage by learning to predict the next frame and flagging large reconstruction
errors.

## Problem
Manual monitoring of CCTV streams is error-prone and expensive. The goal is to
automatically detect anomalous activity in pedestrian walkways. The model is
trained only on normal behavior and should highlight frames where the predicted
next frame deviates from reality.

## Dataset
UCSD Anomaly Detection Dataset (Ped2).
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
  Test/Test001_gt/*.bmp
```

## Approach
- Model: Recurrent U-Net with ConvLSTM bottleneck (`unet_lstm.py`).
- Training: predict the next frame from a sequence of previous frames.
- Scoring: pixel reconstruction error masked by motion, smoothed into a density
  map; thresholds trigger alarms and visual overlays.
- Serving: export to ONNX and run inference with `onnxruntime` for fast API
  responses.

## Repository Highlights
- `notebooks/notebook.ipynb`: EDA, feature analysis, model experiments.
- `train_lstm.py`: trains the ConvLSTM U-Net and exports `models/unet_lstm.onnx`.
- `detect_lstm.py`: offline GIF visualization on a test clip.
- `app/main.py`: FastAPI service for batch clip analysis.
- `app/streamlit_app.py`: Streamlit UI to explore results.
- `Dockerfile`, `scripts/start.sh`: containerized API + UI.

## Setup
Python 3.11+ recommended.

Full development dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API/UI only:
```bash
pip install -r requirements.app.txt
```

## EDA and Feature Analysis
Open the notebook:
```bash
jupyter notebook notebooks/notebook.ipynb
```
The notebook includes data exploration, frame statistics, and model comparison
experiments.

## Training
```bash
python train_lstm.py
```
Outputs:
- `models/unet_lstm.pth`
- `models/unet_lstm.onnx` (used by the web service)

## Offline Inference Demo
Edit `TEST_DIR` in `detect_lstm.py` if needed, then:
```bash
python detect_lstm.py
```
This produces a GIF with heatmaps and bounding boxes for detected anomalies.

## Web Service (Local)
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
docker build -t video-anomaly .
```

Run (mount data and models):
```bash
docker run --rm \
  -p 8001:8001 -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  video-anomaly
```

## Notes on Reproducibility
- Dataset is not committed; use `download_ucsd.py` or provide your own copy.
- All training and inference scripts assume the UCSD Ped2 folder layout shown
  above.
- Models in `models/` can be overwritten by running training scripts.
