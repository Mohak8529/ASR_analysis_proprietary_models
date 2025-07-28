Collection Scoring Criteria Model
Setup

Create project directory: mkdir CollectionScoringModel && cd CollectionScoringModel
Create virtual environment: python3.12 -m venv venv && source venv/bin/activate
Install dependencies: pip install -r requirements.txt
Place Dataset/ folder in the project root with subfolders: audio/, transcription/, translation/, collection_scoring_criteria/.
Ensure filenames align (e.g., ac.wav, ac_transcription.txt, ac_translation.txt, ac_criteria.json).

Training
Run the training script:
python src/train.py


Checkpoints:
Per-epoch checkpoints saved as checkpoints/checkpoint_epoch_{NNN}.pth.
Latest checkpoint saved as checkpoints/latest_checkpoint.pth for resuming.
Best model (lowest validation loss) saved as checkpoints/best_model.pth.


Resuming: If interrupted (Ctrl+C), training saves the current state and resumes from the last epoch on restart.
Fallback: Use per-epoch checkpoints to revert to a previous epoch if training fails (e.g., torch.load('checkpoints/checkpoint_epoch_005.pth')).
Adjust hyperparameters in config.yaml.

Inference
Run inference for a single sample:
python src/inference.py


Outputs prediction.json with 22 true/false labels.
Modify input paths in inference.py for different samples.

Requirements

Python 3.12
Ubuntu
GPU recommended (CUDA-compatible)
