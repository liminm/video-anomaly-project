import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.lstm_detector import LSTMAnomalyDetector, list_available_clips, resolve_clip_dir

CLIP_ROOT = Path(os.getenv("UCSD_CLIP_ROOT", "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"))
MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/unet_lstm.onnx")
OUTPUT_DIR = Path(os.getenv("LSTM_OUTPUT_DIR", "generated_results"))
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX")
GCS_CACHE_DIR = os.getenv("GCS_CACHE_DIR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GCS_BUCKET and not CLIP_ROOT.exists():
        print(f"Warning: clip root not found: {CLIP_ROOT}")
    app.state.detector = LSTMAnomalyDetector(model_path=MODEL_PATH)
    yield


app = FastAPI(title="UCSD LSTM Anomaly Service", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    clip: str
    save_gif: bool = True
    stride: int = 1
    gif_stride: int = 1
    max_frames: int | None = None


@app.get("/clips")
def get_clips():
    if not GCS_BUCKET and not CLIP_ROOT.exists():
        raise HTTPException(status_code=404, detail=f"Clip root not found: {CLIP_ROOT}")
    return {"clips": list_available_clips(CLIP_ROOT, gcs_bucket=GCS_BUCKET, gcs_prefix=GCS_PREFIX)}


@app.post("/analyze")
def analyze_clip(req: AnalyzeRequest):
    try:
        clip_dir = resolve_clip_dir(
            req.clip,
            CLIP_ROOT,
            gcs_bucket=GCS_BUCKET,
            gcs_prefix=GCS_PREFIX,
            cache_dir=Path(GCS_CACHE_DIR) if GCS_CACHE_DIR else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not clip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Clip not found: {req.clip}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = app.state.detector.analyze_clip(
        clip_dir,
        OUTPUT_DIR,
        save_gif=req.save_gif,
        stride=req.stride,
        gif_stride=req.gif_stride,
        max_frames=req.max_frames,
    )

    if result.get("gif_path"):
        filename = Path(result["gif_path"]).name
        result["gif_url"] = f"/outputs/{filename}"

    return result


@app.get("/outputs/{filename}")
def get_output(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
