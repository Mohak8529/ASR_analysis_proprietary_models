# Note: Ensure Audacity is installed and GTK dependencies are met:
# sudo apt-get install audacity libgtk2.0-0 gtk2-engines-murrine libcanberra-gtk-module libatk-adaptor
# Enable 'mod-script-pipe' in Audacity: Tools > Add/Remove Plug-ins > Scripting > Enable

import json
import os
import time
import subprocess
import shutil
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

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

        # All supported Presidio entities plus custom
        self.entities = [
            "CREDIT_CARD", "CRYPTO", "DATE_TIME", "DOMAIN_NAME", "EMAIL_ADDRESS",
            "IBAN_CODE", "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER",
            "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER", "US_LICENSE_PLATE",
            "US_ITIN", "US_PASSPORT", "US_SSN", "SG_NRIC_FIN", "AU_ABN",
            "AU_ACN", "AU_TFN", "AU_MEDICARE", "UK_NHS", "CREDIT_CARD_ENDING"
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

class AudacityPIIMasker:
    def __init__(self, audacity_pipe_path: str = "/tmp/audacity_script_pipe"):
        self.pipe_path = audacity_pipe_path
        self.to_pipe_path = f"{audacity_pipe_path}.to"
        self.from_pipe_path = f"{audacity_pipe_path}.from"
        self.process = None

    def start_audacity(self):
        """Start Audacity with scripting enabled."""
        audacity_exe = os.environ.get("AUDACITY_PATH")
        if not audacity_exe:
            audacity_exe = shutil.which("audacity")
        if not audacity_exe:
            raise FileNotFoundError(
                "Audacity executable not found. Install Audacity with 'sudo apt-get install audacity' "
                "and ensure GTK dependencies: 'sudo apt-get install libgtk2.0-0 gtk2-engines-murrine libcanberra-gtk-module libatk-adaptor'."
            )
        
        # Use environment with correct GTK_PATH setting
        cmd_env = os.environ.copy()
        cmd_env["GTK_PATH"] = "/usr/lib/x86_64-linux-gnu/gtk-2.0"
        
        cmd = ["snap", "run", "audacity"] if audacity_exe.endswith("snap/bin/audacity") else [audacity_exe]
        try:
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=cmd_env
            )
            # Wait for Audacity to start up and initialize
            print("Starting Audacity...")
            for _ in range(15):  # Try for 15 seconds
                time.sleep(1)
                if os.path.exists(self.to_pipe_path) and os.path.exists(self.from_pipe_path):
                    print("Audacity pipes found and ready!")
                    break
            else:
                # If we exit the loop normally (not by break), pipes weren't found
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
                    raise RuntimeError(f"Audacity failed to start. Return code: {self.process.returncode}, Error: {stderr_str}")
                    
                raise RuntimeError(
                    f"Audacity scripting pipes not found at {self.to_pipe_path} and {self.from_pipe_path}. "
                    "Make sure 'mod-script-pipe' is enabled in Audacity: "
                    "Tools > Add/Remove Plug-ins > Scripting > Enable."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to start Audacity: {e}")

    def send_command(self, command: str) -> str:
        """Send a command to Audacity via pipe."""
        try:
            # Check if pipes exist
            if not os.path.exists(self.to_pipe_path) or not os.path.exists(self.from_pipe_path):
                raise FileNotFoundError(f"Audacity pipes not found at {self.pipe_path}. "
                                      f"Make sure mod-script-pipe is enabled in Audacity.")
            
            # Write command to the "to" pipe
            with open(self.to_pipe_path, 'w') as pipe:
                pipe.write(command + '\n')
                pipe.flush()
                
            # Read response from the "from" pipe
            time.sleep(0.2)  # Give Audacity time to process
            with open(self.from_pipe_path, 'r') as pipe:
                response = pipe.read()
                
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Audacity pipe: {e}")

    def mask_audio(self, audio_path: str, beep_path: str, pii_timelines: List[Tuple[float, float, str, str]]) -> None:
        """Mask PII segments in audio using a beep sound in Audacity."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep file not found: {beep_path}")
        
        # Ensure paths use forward slashes for consistency
        audio_path = os.path.abspath(audio_path).replace('\\', '/')
        beep_path = os.path.abspath(beep_path).replace('\\', '/')
        
        # Import audio and beep files
        print(f"Importing audio file: {audio_path}")
        response = self.send_command(f'Import2: Filename="{audio_path}"')
        print(f"Audacity response: {response}")
        
        print(f"Importing beep file: {beep_path}")
        response = self.send_command(f'Import2: Filename="{beep_path}"')
        print(f"Audacity response: {response}")
        
        # Apply beep over PII segments
        for i, (start_time, end_time, text, entity_type) in enumerate(pii_timelines):
            print(f"Processing PII segment {i+1}/{len(pii_timelines)}: {entity_type} - '{text}'")
            duration = end_time - start_time
            
            # Select the portion of the beep we need
            response = self.send_command(f'SelectTime: Start=0 End={duration} Track=1')
            print(f"Select beep response: {response}")
            
            response = self.send_command('Copy:')
            print(f"Copy beep response: {response}")
            
            # Select where to paste in the original audio
            response = self.send_command(f'SelectTime: Start={start_time} End={end_time} Track=0')
            print(f"Select audio segment response: {response}")
            
            response = self.send_command('Paste:')
            print(f"Paste beep response: {response}")
        
        # Export the masked audio
        output_path = audio_path.replace('.wav', '_masked.wav')
        print(f"Exporting masked audio to: {output_path}")
        response = self.send_command(f'Export2: Filename="{output_path}" Format="WAV"')
        print(f"Export response: {response}")
        
        # Close the project
        response = self.send_command('Close:')
        print(f"Close project response: {response}")

    def close_audacity(self):
        """Close Audacity."""
        try:
            print("Closing Audacity...")
            if os.path.exists(self.to_pipe_path) and os.path.exists(self.from_pipe_path):
                self.send_command('Exit:')
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("Audacity closed successfully.")
        except Exception as e:
            print(f"Error while closing Audacity: {e}")
            # Force kill if needed
            if self.process and self.process.poll() is None:
                self.process.kill()

def main():
    # Input files
    transcription_path = "test_audi_trans/First_test/transcription.json"
    audio_path = "test_audi_trans/First_test/1709786890Call 4.wav"
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

    # Initialize Audacity masker
    audacity_masker = AudacityPIIMasker()
    try:
        audacity_masker.start_audacity()
        audacity_masker.mask_audio(audio_path, beep_path, pii_timelines)
    except Exception as e:
        print(f"Error processing audio with Audacity: {e}")
        return
    finally:
        audacity_masker.close_audacity()

    # Print results
    print("Masked Transcription:")
    print(json.dumps(masked_transcription[:5], indent=2, ensure_ascii=False))  # Show just first 5 for brevity
    print("\nUnmasked Transcription:")
    print(json.dumps(unmasked_transcription[:5], indent=2, ensure_ascii=False))  # Show just first 5 for brevity
    print("\nPII Timelines (start, end, text, entity_type):")
    print(json.dumps(pii_timelines[:5], indent=2, ensure_ascii=False))  # Show just first 5 for brevity
    
    print(f"\nSuccessfully processed {len(pii_timelines)} PII occurrences in the audio.")
    print(f"Masked audio saved as: {audio_path.replace('.wav', '_masked.wav')}")

if __name__ == "__main__":
    main()