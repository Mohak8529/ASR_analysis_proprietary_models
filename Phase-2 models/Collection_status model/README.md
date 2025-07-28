Collection Status Model
Predicts collection categories from bank call audio and translations using a CNN+Transformer.
Setup

Install Python 3.12 on Ubuntu.
Install dependencies: pip install -r requirements.txt
Organize dataset in dataset/:
audio/: .wav files (e.g., abc.wav)
transcription/: .txt files (e.g., abc_transcription.txt)
translation/: .txt files (e.g., abc_translation.txt)
collection_status/: .txt files (e.g., abc_collection_status.txt)



Training
Run: python src/train.py

Resumes from checkpoints/latest.pth if interrupted (Ctrl+C).
Saves per-epoch checkpoints in checkpoints/epoch_XXX.pth.
Logs training progress in logs/training_log.txt.

Project Structure

src/: Code (dataset.py, model.py, train.py, utils.py)
dataset/: Data folders
checkpoints/: Model checkpoints
logs/: Training logs

