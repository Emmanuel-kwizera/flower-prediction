import os
# Fix for macOS TensorFlow/OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import subprocess
import sys
import uvicorn
import pathlib
import json
import time
import psutil

# Global Metrics
START_TIME = time.time()
TOTAL_PREDICTIONS = 0
TOTAL_INFERENCE_TIME = 0.0

app = FastAPI(
    title="Flower Prediction API",
    description="API for classifying flowers and triggering model retraining.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / '../models/model.tflite'
VIS_DIR = BASE_DIR / '../visualizations'
WEB_DIR = BASE_DIR / '../web'

print(f"DEBUG: BASE_DIR={BASE_DIR}")
print(f"DEBUG: WEB_DIR={WEB_DIR}")
print(f"DEBUG: WEB_DIR exists={WEB_DIR.exists()}")

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Mount static directories
VIS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/visualizations", StaticFiles(directory=str(VIS_DIR)), name="visualizations")
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# Load model on startup
model = None

# Load model on demand - REMOVED (using subprocess)
# model = None

from fastapi.responses import RedirectResponse

# ... imports ...

@app.get("/health")
async def health_check():
    """Health check and model status."""
    # Check if model file exists
    status = "available" if MODEL_PATH.exists() else "missing"
    
    # Calculate metrics
    uptime = time.time() - START_TIME
    avg_inference = (TOTAL_INFERENCE_TIME / TOTAL_PREDICTIONS) if TOTAL_PREDICTIONS > 0 else 0
    
    return {
        "message": "Flower Prediction API is running",
        "model_status": status,
        "model_path": str(MODEL_PATH),
        "uptime": uptime,
        "total_predictions": TOTAL_PREDICTIONS,
        "avg_inference": avg_inference,
        "cpu_usage": psutil.cpu_percent(interval=None)
    }

@app.get("/")
async def root():
    """Redirect to Web UI."""
    return RedirectResponse(url="/web/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of a flower image.
    """
    # Save uploaded file temporarily
    import tempfile
    import shutil
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Run prediction script
        # Ensure MODEL_PATH is absolute
        model_path_abs = MODEL_PATH.resolve()
        
        cmd = [sys.executable, "scripts/predict.py", tmp_path, "--model_path", str(model_path_abs)]
        
        # Run from project root
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR.parent),
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        # Cleanup
        os.unlink(tmp_path)
        
        if result.returncode != 0:
            # Print stderr to API logs for debugging
            print(f"Prediction script failed. Stderr: {result.stderr}")
            raise Exception(result.stderr)
            
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            raise Exception(f"Invalid JSON output from prediction script: {result.stdout}")
        
        if "error" in output:
            raise HTTPException(status_code=500, detail=output["error"])
            
        # Update metrics
        global TOTAL_PREDICTIONS, TOTAL_INFERENCE_TIME
        TOTAL_PREDICTIONS += 1
        TOTAL_INFERENCE_TIME += duration
        
        return output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/train")
async def train_model(force: bool = False):
    """
    Triggers the model training script.
    """
    try:
        cmd = [sys.executable, "scripts/train.py"]
        if force:
            cmd.append("--force")
            
        process = subprocess.Popen(
            cmd, 
            cwd="..", 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        return {"message": "Training triggered successfully", "pid": process.pid}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger training: {str(e)}")

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Uploads a ZIP file of training data (images) and extracts it to the data directory.
    """
    import zipfile
    import shutil
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")
        
    try:
        # Save zip temporarily
        temp_zip = BASE_DIR / "temp_data.zip"
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Define data directory
        data_dir = BASE_DIR.parent / "data" / "flowers"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Cleanup
        os.remove(temp_zip)
        
        return {"message": f"Data uploaded and extracted to {data_dir}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
