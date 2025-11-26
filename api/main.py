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
MODEL_PATH = BASE_DIR / '../models/model.h5'
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

@app.get("/")
async def root():
    """Health check and model status."""
    # Check if model file exists
    status = "available" if MODEL_PATH.exists() else "missing"
    return {
        "message": "Flower Prediction API is running",
        "model_status": status,
        "model_path": str(MODEL_PATH)
    }

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
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR.parent),
            capture_output=True,
            text=True
        )
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
