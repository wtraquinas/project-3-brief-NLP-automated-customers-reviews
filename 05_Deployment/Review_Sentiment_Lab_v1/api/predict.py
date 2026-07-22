import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Model files live in /model at the project root (copied there from the
# `sentiment_model_onnx` export in the notebook - see README.md).
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
ONNX_FILE = os.path.join(MODEL_DIR, "model_quantized.onnx")
LABELS = ["negative", "neutral", "positive"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
input_names = {i.name for i in session.get_inputs()}

app = FastAPI()

# Only needed if the frontend is deployed on a different domain than this
# API (e.g. a separate Vercel project, or a non-Vercel host). Same-origin
# requests (frontend + api in this same project) don't need CORS at all -
# tighten allow_origins to your actual frontend domain before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    text = (body or {}).get("text", "").strip()
    if not text:
        return {"error": "Missing 'text' field"}

    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding=True)
    ort_inputs = {k: v for k, v in inputs.items() if k in input_names}
    logits = session.run(None, ort_inputs)[0]
    probs = softmax(logits)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": LABELS[pred_idx],
        "scores": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
    }


@app.get("/api/predict")
async def health():
    return {"status": "ok", "model": "distilbert-sentiment-onnx"}
