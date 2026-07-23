from huggingface_hub import HfApi, create_repo

# Run `huggingface-cli login` first (or set the HF_TOKEN env var) so this
# script is authenticated to create/push to the repo.

REPO_ID = "AntonioTrx99/sentiment_model_quantized"   # change this
LOCAL_DIR = "./sentiment_model_onnx"                # from the notebook export cell

api = HfApi()
create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)  # set private=True to require a token to download
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"Pushed -> https://huggingface.co/{REPO_ID}")
