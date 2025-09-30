from presidio_analyzer.nlp_engine import SpacyNlpEngine
import spacy

class CustomSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, model_path: str):
        # Load the fine-tuned model
        try:
            nlp = spacy.load(model_path)
            print(f"Loaded fine-tuned model from {model_path}")
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            raise
        # Configure the model for 'tl' (Tagalog/Filipino) language
        models = [{"lang_code": "tl", "model_name": model_path}]
        super().__init__(models=models)