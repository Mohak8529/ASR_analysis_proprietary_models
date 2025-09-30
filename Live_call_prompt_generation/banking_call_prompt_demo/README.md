# Banking Call Prompt Demo

Generates a static prompt for a single 5-10s banking call utterance based on emotion and situation.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   NLTK data is already present in `./nltk_data/` and configured in `main.py`.

2. **Prepare Models**:
   - Ensure `emotion-english-distilroberta-base` and `hubert-large-superb-er` are in `./emotion_models/`.
   - Download from Hugging Face if needed.

3. **Prepare Input**:
   - Place `utterance.wav` (5-10s) in `./audio/`.
   - Place `utterance.json` in `./transcription/` (format: `[{"startTime": 0.0, "endTime": float, "dialogue": str}]`).

4. **Prompt Directory**:
   - Already created in `prompts/` with static prompt JSON files.
   - Run `main.py` to generate `keyword_index.json`.

## Running

```bash
python main.py
```

Outputs `utterance_results.json` in `./output/` with emotion, metrics, situation, and static prompt.

## Customization

- Update `prompts/` with new static prompt JSON files.
- Modify `SITUATION_REGEX` in `main.py` for new banking scenarios.

## Requirements
- torch==2.4.1
- torchaudio==2.4.1
- transformers==4.44.2
- librosa==0.10.2.post1
- numpy==1.26.4
- scipy==1.13.1
- nltk==3.9.1
- textstat==0.7.7
- langid==1.1.6
- textblob==0.17.1
- vaderSentiment==3.3.2
- soundfile==0.12.1
- statsmodels==0.14.1
