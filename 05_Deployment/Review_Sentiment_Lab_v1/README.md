# Sentiment model — Vercel deployment

Serves the fine-tuned DistilBERT sentiment classifier (negative/neutral/positive)
as a FastAPI serverless function on Vercel, exported to ONNX + INT8 dynamic
quantization so it fits comfortably inside Vercel's function bundle size and
execution-time limits (raw PyTorch + fp32 weights would not).

## 1. Export the model (in the notebook)

Run the "Export to ONNX" cell appended to `dataprep2_sentiment.ipynb` after
training. It converts `./sentiment_model_final` into a quantized ONNX model
at `./sentiment_model_onnx/` (~65 MB vs. ~260 MB for the PyTorch checkpoint).

## 2. Copy the exported files here

Copy every file from `sentiment_model_onnx/` (`model_quantized.onnx`,
`config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, etc.)
into this project's `model/` folder.

## 3. Project structure

```
vercel-deploy/
├── api/
│   └── predict.py       # FastAPI inference endpoint
├── model/                # <- exported ONNX model + tokenizer files go here
├── index.html            # demo frontend, calls /api/predict
├── requirements.txt
├── vercel.json
└── README.md
```

## 4. Frontend integration

`index.html` is a static file at the project root. Deployed in the **same**
Vercel project as `api/predict.py`, it's automatically served at `/`, and
`fetch("/api/predict", ...)` in its script is same-origin - no CORS, no
extra config needed. This is the simplest setup: one project, one domain,
frontend and API together.

If you'd rather build a real frontend (Next.js, React, etc.) instead of the
plain HTML demo, keep it in the same repo/project and call the same relative
path:

```js
const res = await fetch("/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: reviewText }),
});
const { label, scores } = await res.json();
```

**Separate projects/domains:** if the frontend is deployed elsewhere (a
different Vercel project, or another host entirely), call the full API URL
instead (`https://<api-project>.vercel.app/api/predict`) and lock down the
CORS origin in `api/predict.py` - `allow_origins=["*"]` is fine for testing
but replace it with your actual frontend domain before going live.

## 5. Deploy

```bash
npm i -g vercel        # if not already installed
cd vercel-deploy
vercel                 # first deploy, follow the prompts
vercel --prod           # promote to production
```

## 6. Test

```bash
curl -X POST https://<your-project>.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This tablet is fantastic, battery lasts all day"}'

# {"label": "positive", "scores": {"negative": 0.01, "neutral": 0.02, "positive": 0.97}}
```

## Notes / limits

- `maxDuration: 30` in `vercel.json` requires a **Pro** plan; on the Hobby
  plan functions are capped at 10s (cold start + inference should still fit,
  but test it).
- Vercel's Python bundle limit is 500 MB uncompressed (5 GB with Fluid
  compute). The ONNX + `onnxruntime` + `transformers` (tokenizer only, no
  torch) stack stays well under this — do **not** add `torch` to
  `requirements.txt`, it isn't needed for ONNX inference and will blow the
  budget.
- The function has a read-only filesystem except `/tmp`, so the model is
  loaded once at cold start from `model/` and reused across warm invocations.
