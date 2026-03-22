"""
FastAPI server for Deep Ensemble inference.

Endpoints:
    POST /predict  — Upload an image, get prediction + uncertainty metrics
    GET  /health   — Health check

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from model import get_model
from utils import (
    ensemble_predict, mean_prediction, predictive_entropy,
    mutual_information, CIFAR10_CLASSES,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
ensemble_models: list = []
DEVICE: Optional[torch.device] = None
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "5"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")

# Image preprocessing (CIFAR-10 normalization, 32×32 resize)
TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)),
])


# ---------------------------------------------------------------------------
# Lifespan: load models on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble_models, DEVICE

    # Select device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"[startup] Device: {DEVICE}")

    # Load ensemble models
    loaded = 0
    for i in range(ENSEMBLE_SIZE):
        path = os.path.join(CHECKPOINT_DIR, f"model_{i}.pt")
        if not os.path.exists(path):
            print(f"[startup] Checkpoint not found: {path} — skipping")
            continue
        model = get_model(num_classes=10, depth=16, widen_factor=2)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        ensemble_models.append(model)
        loaded += 1
        print(f"[startup] Loaded model_{i} "
              f"(acc: {checkpoint.get('test_acc', '?'):.2f}%)")

    print(f"[startup] Loaded {loaded}/{ENSEMBLE_SIZE} ensemble members")

    if loaded == 0:
        print("[startup] WARNING: No models loaded! /predict will use random "
              "predictions. Train models first with train.py.")

    yield

    # Cleanup
    ensemble_models.clear()
    print("[shutdown] Models unloaded")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI That Knows When It's Wrong",
    description="Deep Ensemble uncertainty quantification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class ClassProbability(BaseModel):
    class_name: str
    class_index: int
    probability: float


class UncertaintyMetrics(BaseModel):
    predictive_entropy: float
    mutual_information: float
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    top5: list[ClassProbability]
    uncertainty: UncertaintyMetrics
    ensemble_predictions: list[list[float]]
    num_models: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    device: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        models_loaded=len(ensemble_models),
        device=str(DEVICE) if DEVICE else "unknown",
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Run ensemble inference on an uploaded image."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, etc.)")

    # Read and preprocess image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0)  # (1, 3, 32, 32)
    except Exception as e:
        raise HTTPException(400, f"Failed to process image: {str(e)}")

    if not ensemble_models:
        # Fallback: return random predictions when no models are loaded
        num_fake = 5
        fake_probs = np.random.dirichlet(np.ones(10), size=num_fake)
        probs = fake_probs.reshape(num_fake, 1, 10)
    else:
        # Ensemble inference
        probs = ensemble_predict(ensemble_models, input_tensor, DEVICE)  # (M, 1, K)

    # Compute metrics
    mean_p = mean_prediction(probs)[:, :]  # (1, K) → squeeze to (K,)
    mean_p_sample = mean_p[0]
    entropy = float(predictive_entropy(mean_p)[0])
    mi = float(mutual_information(probs[:, :1, :])[0])

    # Top-5 classes
    top5_indices = np.argsort(mean_p_sample)[::-1][:5]
    top5 = [
        ClassProbability(
            class_name=CIFAR10_CLASSES[idx],
            class_index=int(idx),
            probability=float(mean_p_sample[idx]),
        )
        for idx in top5_indices
    ]

    predicted_idx = int(top5_indices[0])
    confidence = float(mean_p_sample[predicted_idx])

    # Per-model predictions for the predicted class (for histogram)
    ensemble_preds = probs[:, 0, :].tolist()  # list of M lists of K probs

    return PredictionResponse(
        predicted_class=CIFAR10_CLASSES[predicted_idx],
        predicted_index=predicted_idx,
        confidence=confidence,
        top5=top5,
        uncertainty=UncertaintyMetrics(
            predictive_entropy=entropy,
            mutual_information=mi,
            confidence=confidence,
        ),
        ensemble_predictions=ensemble_preds,
        num_models=len(ensemble_models) if ensemble_models else 5,
    )

# ---------------------------------------------------------------------------
# Mount Frontend Static Files
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "out")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"[warning] Static frontend build not found at {FRONTEND_DIR}")

