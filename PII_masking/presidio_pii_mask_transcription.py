import json
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import List, Dict, Tuple

class PresidioPIIMasker:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.masked_transcription = []
        self.pii_mappings = []  # List of (masked_text, mappings) for unmasking

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in transcription and store mappings for unmasking."""
        self.masked_transcription = []
        self.pii_mappings = []
        for entry in transcription:
            text = entry["dialogue"]
            # Analyze for PERSON and PHONE_NUMBER
            analyzer_results = self.analyzer.analyze(
                text=text, entities=["PERSON", "PHONE_NUMBER"], language="en"
            )
            # Create mappings of original PII to placeholders
            mappings = {}
            for result in analyzer_results:
                original_pii = text[result.start:result.end]
                placeholder = "[PERSON]" if result.entity_type == "PERSON" else "[PHONE]"
                mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
            # Anonymize with replace operator
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators={
                    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                },
            )
            masked_entry = entry.copy()
            masked_entry["dialogue"] = anonymized_result.text
            self.masked_transcription.append(masked_entry)
            self.pii_mappings.append((anonymized_result.text, mappings))
        return self.masked_transcription

    def unmask_transcription(self) -> List[Dict]:
        """Unmask PII using stored PII mappings."""
        unmasked_transcription = []
        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            unmasked_text = masked_text
            # Replace placeholders with original PII
            for placeholder, original_values in mappings.items():
                for original_pii in original_values:
                    if placeholder in unmasked_text:
                        unmasked_text = unmasked_text.replace(placeholder, original_pii, 1)
            unmasked_entry = masked_entry.copy()
            unmasked_entry["dialogue"] = unmasked_text
            unmasked_transcription.append(unmasked_entry)
        return unmasked_transcription

def main():
    # Load transcription
    try:
        with open("transcription.json", "r") as f:
            transcription = json.load(f)
    except FileNotFoundError:
        print("Error: transcription.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in transcription.json.")
        return

    # Initialize masker
    masker = PresidioPIIMasker()

    # Mask transcription
    masked_transcription = masker.mask_transcription(transcription)
    with open("presidio_masked_transcription.json", "w", encoding="utf-8") as f:
        json.dump(masked_transcription, f, indent=2, ensure_ascii=False)

    # Unmask transcription
    unmasked_transcription = masker.unmask_transcription()
    with open("presidio_unmasked_transcription.json", "w", encoding="utf-8") as f:
        json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)

    # Print results
    print("Masked Transcription:")
    print(json.dumps(masked_transcription, indent=2, ensure_ascii=False))
    print("\nUnmasked Transcription:")
    print(json.dumps(unmasked_transcription, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()