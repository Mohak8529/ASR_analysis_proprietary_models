from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

def download_and_save(local_dir: str = "speecht5_model"):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    processor.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    print(f"Saved processor and model to '{local_dir}'")

if __name__ == "__main__":
    download_and_save()
