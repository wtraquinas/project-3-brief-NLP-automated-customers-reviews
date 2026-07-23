import os
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_MODEL = os.getenv(
    "HF_MODEL",
    "AntonioTrx99/sentiment_model_quantized"
)

HF_TOKEN = os.getenv("HF_TOKEN")

CACHE_DIR = "/tmp/hf_cache"

LABELS = [
    "negative",
    "neutral",
    "positive",
]

app = FastAPI()

tokenizer = None
session = None
input_names = None

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def load_model():
    global tokenizer
    global session
    global input_names

    if tokenizer is not None and session is not None:
        return

    try:

        print("Downloading tokenizer...")

        tokenizer_dir = hf_hub_download(
            repo_id=HF_MODEL,
            filename="tokenizer.json",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )

        print("Downloading ONNX model...")

        model_path = hf_hub_download(
            repo_id=HF_MODEL,
            filename="model_quantized.onnx",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )

        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            Path(tokenizer_dir).parent
        )

        print("Loading ONNX Runtime...")

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        input_names = {
            i.name
            for i in session.get_inputs()
        }

        print("Model loaded successfully!")

    except Exception:

        print("========== STARTUP ERROR ==========")
        traceback.print_exc()
        raise


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup():

    load_model()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
async def root():

    return {
        "status": "running",
        "model": HF_MODEL,
    }


@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "model": HF_MODEL,
    }


@app.post("/api/predict")
async def predict(request: Request):

    load_model()

    body = await request.json()

    text = body.get("text", "").strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Missing 'text' field"
        )

    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=256,
    )

    ort_inputs = {
        k: v
        for k, v in inputs.items()
        if k in input_names
    }

    logits = session.run(
        None,
        ort_inputs,
    )[0]

    probs = softmax(logits)[0]

    idx = int(np.argmax(probs))

    return {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "scores": {
            LABELS[i]: float(probs[i])
            for i in range(len(LABELS))
        }
    }