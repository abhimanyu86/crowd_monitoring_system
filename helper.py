import os
from pathlib import Path
from ultralytics import YOLO

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)
EXTS = {'.pt', '.onnx'}

def list_models():
    models = [f.name for f in MODEL_DIR.iterdir() if f.suffix.lower() in EXTS]
    return models or ['yolov8n.pt']   # fallback to built‑in

def load_model(model_name):
    local = MODEL_DIR / model_name
    path = str(local) if local.exists() else model_name
    model = YOLO(path)
    return model, model.names
