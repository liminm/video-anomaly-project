import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.lstm_detector import LSTMAnomalyDetector, list_clips

CLIP_ROOT = Path(os.getenv("UCSD_CLIP_ROOT", "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"))
MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/unet_lstm.onnx")
OUTPUT_DIR = Path(os.getenv("LSTM_OUTPUT_DIR", "generated_results"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CLIP_ROOT.exists():
        print(f"Warning: clip root not found: {CLIP_ROOT}")
    app.state.detector = LSTMAnomalyDetector(model_path=MODEL_PATH)
    yield


app = FastAPI(title="UCSD LSTM Anomaly Service", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    clip: str
    save_gif: bool = True
    stride: int = 1


@app.get("/clips")
def get_clips():
    if not CLIP_ROOT.exists():
        raise HTTPException(status_code=404, detail=f"Clip root not found: {CLIP_ROOT}")
    return {"clips": list_clips(CLIP_ROOT)}


@app.post("/analyze")
def analyze_clip(req: AnalyzeRequest):
    clip_dir = CLIP_ROOT / req.clip
    if not clip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Clip not found: {req.clip}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = app.state.detector.analyze_clip(
        clip_dir,
        OUTPUT_DIR,
        save_gif=req.save_gif,
        stride=req.stride,
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
