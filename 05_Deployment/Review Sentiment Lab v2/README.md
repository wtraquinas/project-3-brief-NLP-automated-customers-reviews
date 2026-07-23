
## Deployment

✅ Frontend:
https://p3nlp-customers-reviews.vercel.app/

✅ Health endpoint:
https://p3nlp-customers-reviews.vercel.app/api/health

✅ Prediction endpoint:
https://p3nlp-customers-reviews.vercel.app/api/predict


--- 

## Workflow

Browser (HTML/CSS/JS)
        │
        ▼
Vercel Static Hosting
        │
        ▼
FastAPI (Vercel Serverless)
        │
        ▼
Downloads ONNX model from Hugging Face Hub
        │
        ▼
ONNX Runtime
        │
        ▼
JSON Response


--- 

## Portfolio

Frontend development
REST API design
FastAPI
Serverless deployment on Vercel
ONNX model inference
Hugging Face Hub integration
Machine learning deployment


---

## Next Improvements:

🎨 Better UI with positive/neutral/negative color badges.
📊 Confidence bar for each sentiment score.
📋 Example review buttons to let users test the API quickly.
🌙 Dark mode toggle.
📈 Response time indicator (e.g. "Inference completed in 42 ms").
📄 Copy JSON response button for developers.
📚 Interactive API documentation linked from the homepage (/docs).
📱 Responsive layout for mobile devices.
🔄 Loading spinner while waiting for the prediction.


---

# Option A: model weights on the Hub, inference in your own Vercel function

## 1. Push the model to a Hub repo
Edit `REPO_ID` in `push_to_hub.py`, then:
```bash
pip install huggingface_hub
huggingface-cli login          # paste a token with write access
python push_to_hub.py
```

## 2. Set environment variables (Vercel dashboard → Settings → Environment Variables)
- `HF_MODEL_REPO` = the repo id you used above (e.g. `yourname/sentiment-model-onnx`)
- `HF_TOKEN` = only needed if you set `private=True` when creating the repo

## 3. Test locally first
```bash
pip install -r requirements.txt
export HF_MODEL_REPO=yourname/sentiment-model-onnx
uvicorn api.predict:app --reload --port 8000
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"text": "Great battery life"}'
```
First request downloads the model into `/tmp/model` - watch for that delay, then confirm the response looks right.

## 4. Deploy
```bash
vercel --prod
```

## Notes
- `/tmp` on Vercel is ephemeral per instance but persists across warm
  invocations of the *same* instance, so only the first request after a
  cold start pays the download cost - expect that first request to be
  noticeably slower.
- If the Hobby plan's 10s timeout gets hit by cold-start downloads, either
  upgrade to Pro (60s) or fall back to committing the model into `model/`
  in the repo (the version from the earlier answer) - that trades a slightly
  bigger deployment bundle for a faster, more predictable cold start.
- Remove the old `model/` folder and its files from this project if you're
  switching to this approach - they're no longer used.
