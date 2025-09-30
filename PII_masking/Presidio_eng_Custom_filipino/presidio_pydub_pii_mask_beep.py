import json
import os
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydub import AudioSegment

def generate_beep_wave(beep_path: str, frequency: float = 1000.0, duration: float = 1.0, sample_rate: int = 44100, amplitude: float = 0.8) -> None:
    """Generate a sine wave beep sound and save as WAV file."""
    try:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
        wavfile.write(beep_path, sample_rate, audio)
    except Exception as e:
        raise RuntimeError(f"Failed to generate beep.wav: {e}")

class PresidioPIIMasker:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []

        # Custom recognizer for partial credit card numbers
        credit_card_ending_pattern = Pattern(
            name="credit_card_ending_pattern",
            regex=r"credit card ending with \d{4}",
            score=0.85
        )
        credit_card_ending_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD_ENDING",
            patterns=[credit_card_ending_pattern]
        )
        self.analyzer.registry.add_recognizer(credit_card_ending_recognizer)

        # Custom recognizer for contract reference numbers
        contract_ref_pattern = Pattern(
            name="contract_ref_pattern",
            regex=r"\bCF[- ]?\d{4,}(?:\s*dash\s*\d+)?\b|\bCF\b|\b\d{4,}(?:\s*dash\s*\d+)?\b",
            score=0.95
        )
        contract_ref_recognizer = PatternRecognizer(
            supported_entity="CONTRACT_REF",
            patterns=[contract_ref_pattern]
        )
        self.analyzer.registry.add_recognizer(contract_ref_recognizer)

        # Custom recognizer for account numbers
        account_number_pattern = Pattern(
            name="account_number_pattern",
            regex=r"\b\d{3,}(?:[- ]\d+)*\b(?=.*\b(account|number)\b)",
            score=0.9
        )
        account_number_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[account_number_pattern]
        )
        self.analyzer.registry.add_recognizer(account_number_recognizer)

        # Custom recognizer for monetary amounts
        monetary_amount_pattern = Pattern(
            name="monetary_amount_pattern",
            regex=r"\b\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?\b(?=.*\b(USD|SGD|buy|sell|amount|dollar)\b)",
            score=0.9
        )
        monetary_amount_recognizer = PatternRecognizer(
            supported_entity="MONETARY_AMOUNT",
            patterns=[monetary_amount_pattern]
        )
        self.analyzer.registry.add_recognizer(monetary_amount_recognizer)

        # Custom recognizer for numbers in words (FIGURE)
        number_words_pattern = Pattern(
            name="number_words_pattern",
            regex=r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and\s+)?(?:\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion))*\b",
            score=0.9
        )
        number_words_recognizer = PatternRecognizer(
            supported_entity="FIGURE",
            patterns=[number_words_pattern]
        )
        self.analyzer.registry.add_recognizer(number_words_recognizer)

        # All supported Presidio entities plus custom
        self.entities = [
            "CREDIT_CARD", "CRYPTO", "DATE_TIME", "DOMAIN_NAME", "EMAIL_ADDRESS",
            "IBAN_CODE", "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER",
            "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER", "US_LICENSE_PLATE",
            "US_ITIN", "US_PASSPORT", "US_SSN", "SG_NRIC_FIN", "AU_ABN",
            "AU_ACN", "AU_TFN", "AU_MEDICARE", "UK_NHS", "CREDIT_CARD_ENDING",
            "CONTRACT_REF", "ACCOUNT_NUMBER", "MONETARY_AMOUNT", "FIGURE"
        ]

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in transcription, store mappings, and calculate PII timelines."""
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        for entry in transcription:
            text = entry["dialogue"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            # Analyze for all supported entities
            analyzer_results = self.analyzer.analyze(
                text=text, entities=self.entities, language="en"
            )
            # Create mappings of original PII to placeholders
            mappings = {}
            for result in analyzer_results:
                original_pii = text[result.start:result.end]
                placeholder = f"[{result.entity_type}]"
                mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
                # Calculate PII timeline
                text_length = len(text)
                if text_length > 0:
                    duration = end_time - start_time
                    pii_start_time = max(start_time, start_time + (result.start / text_length) * duration)
                    pii_end_time = min(end_time, start_time + (result.end / text_length) * duration)
                    self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, result.entity_type))
            # Anonymize with replace operator for all entities
            operators = {entity: OperatorConfig("replace", {"new_value": f"[{entity}]"}) for entity in self.entities}
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators,
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
            for placeholder, original_values in mappings.items():
                for original_pii in original_values:
                    if placeholder in unmasked_text:
                        unmasked_text = unmasked_text.replace(placeholder, original_pii, 1)
            unmasked_entry = masked_entry.copy()
            unmasked_entry["dialogue"] = unmasked_text
            unmasked_transcription.append(unmasked_entry)
        return unmasked_transcription

    def get_pii_timelines(self) -> List[Tuple[float, float, str, str]]:
        """Return timelines of PII occurrences."""
        return self.pii_timelines

class PydubPIIMasker:
    """PII audio masker using pydub instead of Audacity."""
    
    def mask_audio(self, audio_path: str, beep_path: str, pii_timelines: List[Tuple[float, float, str, str]]) -> str:
        """
        Mask PII segments in audio using a beep sound using pydub.
        
        Args:
            audio_path: Path to the audio file to mask
            beep_path: Path to the beep sound file
            pii_timelines: List of tuples (start_time, end_time, text, entity_type)
            
        Returns:
            Path to the output masked audio file
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep file not found: {beep_path}")
            
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        
        print(f"Loading beep file: {beep_path}")
        beep = AudioSegment.from_wav(beep_path)
        
        # Sort PII timelines by start time to process them in order
        pii_timelines_sorted = sorted(pii_timelines, key=lambda x: x[0])
        
        print(f"Processing {len(pii_timelines_sorted)} PII occurrences...")
        
        # Convert to milliseconds for pydub
        processed_segments = []
        last_end = 0
        
        for i, (start_time, end_time, text, entity_type) in enumerate(pii_timelines_sorted):
            # Convert to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            print(f"Processing PII segment {i+1}/{len(pii_timelines_sorted)}: {entity_type} - '{text}'")
            print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s ({start_ms}ms - {end_ms}ms)")
            
            # Keep audio before current PII
            if start_ms > last_end:
                print(f"  Adding segment from {last_end}ms to {start_ms}ms")
                segment = audio[last_end:start_ms]
                processed_segments.append(segment)
            
            # Calculate duration of PII segment
            duration_ms = end_ms - start_ms
            
            # If beep is shorter, extend it
            if len(beep) < duration_ms:
                print(f"  Extending beep to match PII duration: {duration_ms}ms")
                beep_segment = beep * (int(duration_ms / len(beep)) + 1)
                beep_segment = beep_segment[:duration_ms]
            else:
                print(f"  Trimming beep to match PII duration: {duration_ms}ms")
                beep_segment = beep[:duration_ms]
            
            # Match volume
            segment_volume = audio[start_ms:end_ms].dBFS
            if segment_volume > -float('inf'):  # Check if segment is not silent
                print(f"  Matching volume: {segment_volume:.2f} dBFS")
                beep_segment = beep_segment.apply_gain(segment_volume - beep_segment.dBFS)
            
            # Add beep to processed segments
            processed_segments.append(beep_segment)
            
            # Update last_end position
            last_end = end_ms
        
        # Add remaining audio
        if last_end < len(audio):
            print(f"  Adding final segment from {last_end}ms to end")
            processed_segments.append(audio[last_end:])
        
        # Combine all segments
        print("Combining all audio segments...")
        masked_audio = sum(processed_segments)
        
        # Save output
        output_path = audio_path.replace('.wav', '_masked.wav')
        print(f"Exporting masked audio to: {output_path}")
        masked_audio.export(output_path, format="wav")
        
        return output_path

def main():
    # Input files
    transcription_path = "test_audi_trans/Third_test/transcription.json"
    audio_path = "test_audi_trans/Third_test/1709786425Call 3.wav"
    beep_path = "beep.wav"

    # Generate beep sound
    try:
        generate_beep_wave(beep_path, frequency=1000.0, duration=1.0, sample_rate=44100, amplitude=0.8)
        print(f"Generated beep sound: {beep_path}")
    except Exception as e:
        print(f"Error generating beep.wav: {e}")
        return

    # Load transcription
    try:
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription = json.load(f)
    except FileNotFoundError:
        print(f"Error: {transcription_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {transcription_path}.")
        return

    # Initialize Presidio masker
    presidio_masker = PresidioPIIMasker()

    # Mask transcription and get PII timelines
    try:
        masked_transcription = presidio_masker.mask_transcription(transcription)
        with open("presidio_masked_transcription.json", "w", encoding="utf-8") as f:
            json.dump(masked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error masking transcription: {e}")
        return

    # Unmask transcription
    try:
        unmasked_transcription = presidio_masker.unmask_transcription()
        with open("presidio_unmasked_transcription.json", "w", encoding="utf-8") as f:
            json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error unmasking transcription: {e}")
        return

    # Get PII timelines
    pii_timelines = presidio_masker.get_pii_timelines()

    # Initialize Pydub masker and mask audio
    pydub_masker = PydubPIIMasker()

    try:
        output_path = pydub_masker.mask_audio(audio_path, beep_path, pii_timelines)
        print(f"Successfully masked audio file: {output_path}")
    except Exception as e:
        print(f"Error processing audio with Pydub: {e}")
        return

    # Print results
    print("\nMasked Transcription:")
    print(json.dumps(masked_transcription, indent=2, ensure_ascii=False))
    print("\nUnmasked Transcription:")
    print(json.dumps(unmasked_transcription, indent=2, ensure_ascii=False))
    print("\nPII Timelines (start, end, text, entity_type):")
    print(json.dumps(pii_timelines, indent=2, ensure_ascii=False))
    
    print(f"\nSuccessfully processed {len(pii_timelines)} PII occurrences in the audio.")
    print(f"Masked audio saved as: {output_path}")

if __name__ == "__main__":
    main()