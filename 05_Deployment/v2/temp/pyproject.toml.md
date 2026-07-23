[tool.vercel]
entrypoint = "api.predict:app"

[project]
name = "sentiment-api"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "fastapi",
    "uvicorn",
    "onnxruntime",
    "numpy",
    "tokenizers",
    "huggingface_hub"
]