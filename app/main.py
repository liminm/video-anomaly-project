import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import AnomalyDetector

app = FastAPI(title="Video Anomaly Detector")

# Global model instance
detector = None

@app.on_event("startup")
def load_model():
    global detector
    # Load the model once when the server starts
    detector = AnomalyDetector()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Run Inference
        result = detector.predict(tmp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 3. Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
def root():
    return {"message": "Video Anomaly Service is Running"}
