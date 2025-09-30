import json
import os
import re
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
from pydub import AudioSegment
#Generate beep sound
def generate_beep_wave(beep_path: str, frequency: float = 1000.0, duration: float = 1.0, sample_rate: int = 44100, amplitude: float = 0.8) -> None:
    """Generate a sine wave beep sound and save as WAV file."""
    try:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
        wavfile.write(beep_path, sample_rate, audio)
    except Exception as e:
        raise RuntimeError(f"Failed to generate beep.wav: {e}")

class FilipinoPIIMasker:
    def __init__(self, surname_file: str = "filipino_surnamess.txt"):
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        self.surnames = self.load_surnames(surname_file)

        # Define PII patterns for Filipino text (FIGURE, PERSON relies on surname file, EMAIL added)
        self.pii_patterns = [
            {
                "entity": "FIGURE",
                "pattern": r"\b(?:(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)|(?:isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampo|labing-isa|labindalawa|labintatlo|labing-apat|labing-lima|labing-anim|labing-pito|labing-walo|labing-siyam|daan|sandaan|libo|milyon))(?:\s*(?:[,;\-]|\b(?:and|at|na|ma\'am|ah|centavos)\b)?\s*(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)|(?:isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampo|labing-isa|labindalawa|labintatlo|labing-apat|labing-lima|labing-anim|labing-pito|labing-walo|labing-siyam|daan|sandaan|libo|milyon)))*)\b",
                "context": None  # No context required; match all number groups
            },
            {
                "entity": "EMAIL",
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "context": None  # No context required; match valid email addresses
            }
        ]

    def load_surnames(self, surname_file: str) -> set:
        """Load surnames from a text file into a set for efficient lookup."""
        surnames = set()
        try:
            if os.path.exists(surname_file):
                with open(surname_file, "r", encoding="utf-8") as f:
                    surnames = {line.strip().lower() for line in f if line.strip()}
                print(f"Loaded {len(surnames)} surnames from {surname_file}")
            else:
                print(f"Warning: Surname file {surname_file} not found.")
        except Exception as e:
            print(f"Error loading surnames: {e}")
        return surnames

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in Filipino transcription, store mappings, and calculate PII timelines."""
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []

        for entry in transcription:
            text = entry["dialogue"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            mappings = {}
            modified_text = text
            pii_results = []

            # Step 1: List-based surname matching (exact whole-word match)
            # Use regex to find whole words, ensuring proper boundaries
            words = re.findall(r'\b\w+\b', modified_text, re.UNICODE)
            char_index = 0
            for i, word in enumerate(words):
                word_lower = word.lower()
                # Exact match for whole word (case-insensitive)
                if word_lower in self.surnames:
                    original_pii = word
                    placeholder = "[PERSON]"

                    # Find the exact position of the word in the text
                    word_pattern = r'\b' + re.escape(word) + r'\b'
                    match = re.search(word_pattern, modified_text, re.IGNORECASE)
                    if match:
                        name_start = match.start()
                        name_end = match.end()
                        mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]

                        # Calculate timeline
                        text_length = len(modified_text)
                        if text_length > 0:
                            duration = end_time - start_time
                            pii_start_time = max(start_time, start_time + (name_start / text_length) * duration)
                            pii_end_time = min(end_time, start_time + (name_end / text_length) * duration)
                            self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, "PERSON"))
                        pii_results.append((name_start, name_end, original_pii, "PERSON"))

                # Update char_index for the next word
                char_index += len(word) + 1 if i < len(words) - 1 else len(word)

            # Step 2: Regex-based matching (for FIGURE and EMAIL)
            for pii_type in self.pii_patterns:
                entity = pii_type["entity"]
                pattern = pii_type["pattern"]
                context = pii_type["context"]

                # Check context if required
                if context is None or re.search(context, modified_text, re.IGNORECASE):
                    matches = re.finditer(pattern, modified_text, re.IGNORECASE)
                    for match in matches:
                        original_pii = match.group(0)
                        placeholder = f"[{entity}]"
                        mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
                        pii_start = match.start()
                        pii_end = match.end()
                        text_length = len(modified_text)
                        if text_length > 0:
                            duration = end_time - start_time
                            pii_start_time = max(start_time, start_time + (pii_start / text_length) * duration)
                            pii_end_time = min(end_time, start_time + (pii_end / text_length) * duration)
                            self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, entity))
                        pii_results.append((pii_start, pii_end, original_pii, entity))

            # Sort results by start position (descending) to avoid index issues
            pii_results.sort(key=lambda x: x[0], reverse=True)
            # Replace PII with placeholders
            for start, end, original_pii, entity in pii_results:
                placeholder = f"[{entity}]"
                modified_text = modified_text[:start] + placeholder + modified_text[end:]

            masked_entry = entry.copy()
            masked_entry["dialogue"] = modified_text
            self.masked_transcription.append(masked_entry)
            self.pii_mappings.append((modified_text, mappings))

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
    """PII audio masker using pydub."""

    def mask_audio(self, audio_path: str, beep_path: str, pii_timelines: List[Tuple[float, float, str, str]]) -> str:
        """ Mask PII segments in audio using a beep sound using pydub.

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
            print(f" Time range: {start_time:.2f}s - {end_time:.2f}s ({start_ms}ms - {end_ms}ms)")

            # Keep audio before current PII
            if start_ms > last_end:
                print(f" Adding segment from {last_end}ms to {start_ms}ms")
                segment = audio[last_end:start_ms]
                processed_segments.append(segment)

            # Calculate duration of PII segment
            duration_ms = end_ms - start_ms

            # If beep is shorter, extend it
            if len(beep) < duration_ms:
                print(f" Extending beep to match PII duration: {duration_ms}ms")
                beep_segment = beep * (int(duration_ms / len(beep)) + 1)
                beep_segment = beep_segment[:duration_ms]
            else:
                print(f" Trimming beep to match PII duration: {duration_ms}ms")
                beep_segment = beep[:duration_ms]

            # Match volume
            segment_volume = audio[start_ms:end_ms].dBFS
            if segment_volume > -float('inf'):  # Check if segment is not silent
                print(f" Matching volume: {segment_volume:.2f} dBFS")
                beep_segment = beep_segment.apply_gain(segment_volume - beep_segment.dBFS)

            # Add beep to processed segments
            processed_segments.append(beep_segment)

            # Update last_end position
            last_end = end_ms

        # Add remaining audio
        if last_end < len(audio):
            print(f" Adding final segment from {last_end}ms to end")
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
    # Input files (updated to Second_test directory)
    transcription_path = "test_audi_trans_philipino/Third_test/transcription.json"
    audio_path = "test_audi_trans_philipino/Third_test/1679044095malay_collection.wav"
    beep_path = "beep.wav"
    surname_file = "filipino_surnames.txt"

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

    # Initialize Filipino PII masker
    filipino_masker = FilipinoPIIMasker(surname_file=surname_file)

    # Mask transcription and get PII timelines
    try:
        masked_transcription = filipino_masker.mask_transcription(transcription)
        with open("filipino_masked_transcription.json", "w", encoding="utf-8") as f:
            json.dump(masked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error masking transcription: {e}")
        return

    # Unmask Transcription
    try:
        unmasked_transcription = filipino_masker.unmask_transcription()
        with open("filipino_unmasked_transcription.json", "w", encoding="utf-8") as f:
            json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error unmasking transcription: {e}")
        return

    # Get PII timelines
    pii_timelines = filipino_masker.get_pii_timelines()

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