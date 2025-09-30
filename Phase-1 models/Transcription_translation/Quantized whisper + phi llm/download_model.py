from huggingface_hub import snapshot_download

# Download the pre-converted CTranslate2 model for faster-whisper large-v3
snapshot_download(
    repo_id="Systran/faster-whisper-large-v3",
    local_dir="transcription_model",
    local_dir_use_symlinks=False  # Ensure actual files are downloaded
)

print("Model downloaded to 'transcription_model' directory.")