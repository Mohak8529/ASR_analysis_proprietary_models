# Promise to Pay Model

A multihead transformer model for classifying Promise to Pay categories and extracting Critical Keywords from audio and text data.

## Directory Structure
- `dataset/`: Contains audio, transcription, translation, category, and keyword files.
- `src/`: Source code (dataset.py, model.py, train.py, utils.py, inference.py).
- `checkpoints/`: Saved model checkpoints.
- `logs/`: Training logs.

## Setup
1. Install Python 3.12 and create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate