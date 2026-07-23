import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from fastapi import FastAPI, Request


os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

REPO_ID = os.environ.get("HF_MODEL_REPO", "AntonioTrx99/sentiment_model_quantized")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the repo is private
MODEL_DIR = "/tmp/model"  # /tmp is writable and persists across warm invocations
ONNX_FILE = os.path.join(MODEL_DIR, "model_quantized.onnx")
LABELS = ["negative", "neutral", "positive"]

# Downloads once per cold start; warm invocations reuse the cached files.
if not os.path.exists(ONNX_FILE):
    snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR, token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
input_names = {i.name for i in session.get_inputs()}

app = FastAPI()


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


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": REPO_ID}

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/index.html", status_code=307)
