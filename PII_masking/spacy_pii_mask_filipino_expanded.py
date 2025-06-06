import json
import re
import spacy
from typing import List, Dict, Tuple

class SpacyPIIMasker:
    def __init__(self):
        self.nlp = spacy.blank("en")  # Blank model for tokenization, no NER
        self.masked_transcription = []
        self.pii_mappings = []

        # Comprehensive regex patterns for PII (universal and Filipino-specific)
        self.patterns = [
            # Universal PII
            (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "CREDIT_CARD"),
            (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL_ADDRESS"),
            (r"\b\+?[1-9]\d{1,14}\b", "PHONE_NUMBER"),
            (r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", "IP_ADDRESS"),
            (r"\bhttps?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*\b", "URL"),
            (r"\b(?:bc1|[13])[a-zA-Z0-9]{25,39}\b|\b0x[a-fA-F0-9]{40}\b", "CRYPTO_ADDRESS"),
            (r"\b[A-Z]{2}[0-9]{2}(?:[A-Z0-9]){9,30}\b", "IBAN_CODE"),
            (r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", "DOMAIN_NAME"),
            # Filipino-specific PII
            (r"(\+639\d{9}|09\d{9})", "PH_PHONE_NUMBER"),
            (r"\d{3}-\d{3}-\d{3}-\d{3}", "PH_TIN"),
            (r"\d{2}-\d{7}-\d", "PH_SSS"),
            (r"\d{11}", "PH_GSIS"),
            (r"\d{4}-\d{4}-\d{4}", "PH_PAGIBIG"),
            (r"\d{2}-\d{9}-\d", "PH_PHILHEALTH"),
            (r"(?:credit card na nagtatapos sa|credit card ending with) \d{4}", "CREDIT_CARD_ENDING"),
            (r"\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b", "FILIPINO_PERSON"),
            # Other PII (for international contexts)
            (r"\b\d{3}-\d{2}-\d{4}\b", "US_SSN"),
            (r"\b[A-Z]\d{8}\b", "US_PASSPORT"),
            (r"\b\d{8,12}\b", "US_BANK_NUMBER"),
            (r"\b\d{3}\s\d{3}\s\d{3}\b", "AU_TFN"),
            (r"\b\d{3}\s\d{3}\s\d{4}\b", "UK_NHS"),
        ]

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in transcription using regex and store mappings for unmasking."""
        self.masked_transcription = []
        self.pii_mappings = []
        for entry in transcription:
            text = entry["dialogue"]
            mappings = {}
            masked_text = text
            pii_results = []

            # Apply regex patterns
            for pattern, entity_type in self.patterns:
                for match in re.finditer(pattern, text):
                    start, end = match.start(), match.end()
                    original_pii = match.group(0)
                    placeholder = f"[{entity_type}]"
                    pii_results.append((start, end, original_pii, entity_type))
                    mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]

            # Sort and mask (replace in reverse to preserve indices)
            pii_results.sort(key=lambda x: x[0])
            for start, end, original_pii, entity_type in reversed(pii_results):
                masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]

            masked_entry = entry.copy()
            masked_entry["dialogue"] = masked_text
            self.masked_transcription.append(masked_entry)
            self.pii_mappings.append((masked_text, mappings))
        return self.masked_transcription

    def unmask_transcription(self) -> List[Dict]:
        """Unmask PII using stored PII mappings."""
        unmasked_transcription = []
        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            unmasked_text = masked_text
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
        with open("transcription.json", "r", encoding="utf-8") as f:
            transcription = json.load(f)
    except FileNotFoundError:
        print("Error: transcription.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in transcription.json.")
        return

    # Initialize masker
    masker = SpacyPIIMasker()

    # Mask transcription
    masked_transcription = masker.mask_transcription(transcription)
    with open("spacy_masked_transcription.json", "w", encoding="utf-8") as f:
        json.dump(masked_transcription, f, indent=2, ensure_ascii=False)

    # Unmask transcription
    unmasked_transcription = masker.unmask_transcription()
    with open("spacy_unmasked_transcription.json", "w", encoding="utf-8") as f:
        json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)

    # Print results
    print("Masked Transcription:")
    print(json.dumps(masked_transcription, indent=2, ensure_ascii=False))
    print("\nUnmasked Transcription:")
    print(json.dumps(unmasked_transcription, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()