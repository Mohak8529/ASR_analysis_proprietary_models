# download_model.py
import requests
import os

def download_file(url, dest_path):
    """Download a file from a URL to the specified destination path."""
    try:
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Write the file in chunks to handle large files
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Model successfully downloaded to {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")

# Model details
model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
model_name = "Phi-3-mini-4k-instruct-q4.gguf"
model_path = os.path.join("models", model_name)

# Download the model
download_file(model_url, model_path)