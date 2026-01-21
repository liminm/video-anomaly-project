# UCSD LSTM Video Anomaly Detection

Detect unusual events in surveillance footage by learning normal pedestrian
motion and flagging frames where reconstruction error spikes.

Deployed app: https://video-anomaly-app-277455458775.europe-west3.run.app
Warning: The Cloud Run deployment can be slow for full-resolution clips and GIF
generation. For faster runs, lower the GIF scale or enable quick mode in the UI.

![Clip selection](images/selection.gif)
_Select a test clip from the dropdown and launch detection._

![Results panel](images/result.png)
_Results summary with max score, alarm frames, and the score timeline._

![Anomaly visualization](images/video.gif)
_The video overlay highlights detected anomalies (e.g., a bike or cart)._

![Anomaly visualization 2](images/video2.gif)
_Another detection example highlighting anomalous motion in the scene._

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
The script downloads the UCSD tarball (with a mirror fallback) and extracts it
to `data/UCSD_Anomaly_Dataset.v1p2/`.

Expected layout:
```
data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/
  Train/Train001/*.tif
  Test/Test001/*.tif
  Test/Test001_gt/*.bmp   # depending on dataset layout
```

## Dataset details
- UCSD provides multiple subsets (Ped1, Ped2) with different scenes.
- Each clip is a short sequence of frame images stored in a folder.
- Test clips include per-frame anomaly masks in matching `_gt` folders.
- Ground truth masks label anomalous objects, not just lighting shifts.

## Approach
- Model: Recurrent U-Net with ConvLSTM bottleneck (`unet_lstm.py`).
- Training: predict the next frame from a sequence of previous frames.
- Scoring: pixel reconstruction error masked by motion, smoothed into a density
  map; thresholds trigger alarms and visual overlays.
- Serving: export to ONNX and run inference with `onnxruntime` for fast API
  responses.

Model architecture (high level):
- Encoder: stacked conv blocks downsample the input frames to a compact
  spatiotemporal representation.
- Bottleneck: ConvLSTM layers model temporal dynamics across the input sequence.
- Decoder: upsampling + skip connections reconstruct the next frame.
- Output: predicted next frame, compared to the true next frame to compute
  reconstruction error.

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

Training parameters and what was tried:
- Hidden channels (model capacity): 128 vs. 256 to trade accuracy vs. memory.
- LSTM depth: 1 vs. 2 layers to test temporal modeling depth.
- Dropout: 0.0 vs. 0.3 to test regularization.
- Learning rate: 5e-4 and 1e-3 to balance stability vs. convergence speed.
- Weight decay: 0.0 vs. 1e-4 to reduce overfitting.
- Batch size: 4 due to GPU memory constraints for ConvLSTM.
- Sequence length: 8 frames to capture short-term motion patterns.

Selection criteria:
- The best run is chosen by lowest validation loss in the sweep output
  (`generated_results/experiments/summary.csv`).
- The notebook then evaluates the selected model with frame-level ROC AUC.

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

The notebook covers:
- Data preparation and cleaning.
- EDA (frame statistics, motion analysis, ground-truth inspection).
- Feature/target analysis and rationale for preprocessing choices.
- Model training, hyperparameter sweep, evaluation, and inference demo.

## Training
```bash
python train.py
```
Outputs:
- `models/unet_lstm.pth`
- `models/unet_lstm.onnx`

This script is the exported version of the notebook training logic.

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
The deployment API is built with FastAPI. The service entrypoint is
`app/main.py`.

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

Prediction flow:
- `app/main.py` loads the ONNX model and exposes `/analyze`.
- The Streamlit UI calls this endpoint and renders the results.

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
Detailed steps:
1) Set project + region:
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region YOUR_REGION
```

2) Enable required services (one-time):
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

3) Create an Artifact Registry repo (one-time):
```bash
gcloud artifacts repositories create anomaly-repo \
  --repository-format=docker \
  --location=YOUR_REGION \
  --description="Video anomaly app images"
```

4) Authenticate Docker to Artifact Registry:
```bash
gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
```

5) Build + push the image:
```bash
docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/anomaly-repo/video-anomaly-app:latest .
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/anomaly-repo/video-anomaly-app:latest
```

6) Deploy to Cloud Run:
```bash
gcloud run deploy video-anomaly-app \
  --region europe-west3 \
  --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/anomaly-repo/video-anomaly-app:latest \
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

Deployed URL:
- https://video-anomaly-app-277455458775.europe-west3.run.app

## Zoomcamp criteria checklist
- Problem description: see “Problem description”.
- EDA: see `notebooks/notebook.ipynb` (EDA section with plots + notes).
- Model training + tuning: `train.py` and `run_experiments.py`.
- Exported training script: `train.py` mirrors the notebook.
- Reproducibility: `download_ucsd.py` + notebook + training scripts.
- Model deployment: FastAPI (`app/main.py`) + Streamlit UI.
- Dependency management: `requirements.txt` / `requirements.app.txt`.
- Containerization: `Dockerfile` + run instructions above.
- Cloud deployment: Cloud Run commands + deployed URL above.

## Notes
- The dataset is not committed; use `download_ucsd.py` or your own copy.
- Generated GIFs and experiment outputs live under `generated_results/`.
