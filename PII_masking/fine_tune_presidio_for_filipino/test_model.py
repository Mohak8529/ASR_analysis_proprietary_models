import spacy
import json

nlp = spacy.load("fine_tuned_xx_ent_wiki_sm")
with open("test_main/taglish_calls/ashu_test/Payment Plan - Approved_Marjorie Casas_Clarissa Elizan_613834760548_2025-01-24_10.30.39_transcription.json", "r", encoding="utf-8") as f:
    transcription = json.load(f)
    text = " ".join([seg["dialogue"] for seg in transcription])
doc = nlp(text)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])