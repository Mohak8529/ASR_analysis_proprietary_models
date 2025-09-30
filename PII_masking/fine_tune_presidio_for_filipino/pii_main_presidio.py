import json
import os
import re
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
import random

from pydub import AudioSegment
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
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

def detect_language(transcription: List[Dict], surname_file: str = "filipino_surnames.txt") -> str:
    """Detect language based on percentage of Filipino words in transcription."""
    # Load Filipino surnames
    surnames = set()
    try:
        if os.path.exists(surname_file):
            with open(surname_file, "r", encoding="utf-8") as f:
                surnames = {line.strip().lower() for line in f if line.strip()}
    except Exception as e:
        print(f"Error loading surnames: {e}")

    # Common Filipino words (partial list for detection)
    filipino_words = {
        "ako", "ikaw", "siya", "kami", "kayo", "sila", "ito", "iyan", "iyon",
        "sa", "ng", "at", "para", "mga", "nang", "din", "rin", "ang", "na",
        "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam",
        "po", "opo", "hindi", "oo", "salamat", "magandang", "araw", "gabi",
        "pa", "ba", "kung", "dahil", "pero", "subalit", "kasi", "eh"
    } | surnames  # Include surnames as Filipino words

    total_words = 0
    filipino_word_count = 0

    for entry in transcription:
        text = entry.get("dialogue", "")
        words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
        total_words += len(words)
        filipino_word_count += sum(1 for word in words if word in filipino_words)

    # Calculate percentage
    filipino_percentage = (filipino_word_count / total_words * 100) if total_words > 0 else 0
    print(f"Filipino words: {filipino_word_count}/{total_words} ({filipino_percentage:.2f}%)")

    # Threshold: ≥30% Filipino words → Filipino call
    return ("filipino" if filipino_percentage >= 3 else "english", filipino_percentage)

# from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
# from presidio_analyzer.nlp_engine import NlpEngineProvider
# from presidio_anonymizer import AnonymizerEngine
# from presidio_anonymizer.entities import OperatorConfig
# import re
# import random
# import os
# from typing import List, Dict, Tuple

class FilipinoPIIMasker:
    def __init__(self, surname_file: str = "filipino_surnames.txt"):
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        self.surnames = self.load_surnames(surname_file)
        self.first_names = [
            "Mark", "Anna", "Jose", "Maria", "John", "Clara", "Pedro", "Liza",
            "Ramon", "Teresa", "Ben", "Sofia", "Luis", "Emma", "Carlos"
        ]

        # Import and use the fine-tuned model via CustomSpacyNlpEngine
        from custom_nlp_engine import CustomSpacyNlpEngine

        try:
            nlp_engine = CustomSpacyNlpEngine(model_path="fine_tuned_xx_ent_wiki_sm")
            print("Loaded fine-tuned model with CustomSpacyNlpEngine")
        except Exception as e:
            print(f"Error initializing CustomSpacyNlpEngine: {e}. Falling back to default AnalyzerEngine.")
            nlp_engine = None
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["tl"]
        ) if nlp_engine else AnalyzerEngine(supported_languages=["tl"])
        self.anonymizer = AnonymizerEngine()
        print(f"Analyzer engine: {type(self.analyzer.nlp_engine).__name__}")

        # Define custom recognizers for FIGURE, EMAIL_ADDRESS, and MONTH
        figure_pattern = Pattern(
            name="figure_pattern",
            regex=r"\b(?:(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)|(?:isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampo|labing-isa|labindalawa|labintatlo|labing-apat|labing-lima|labing-anim|labing-pito|labing-walo|labing-siyam|daan|sandaan|libo|milyon))(?:\s*(?:[,;\-]|\b(?:and|at|na|ma\'am|ah|centavos)\b)?\s*(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)|(?:isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampo|labing-isa|labindalawa|labintatlo|labing-apat|labing-lima|labing-anim|labing-pito|labing-walo|labing-siyam|daan|sandaan|libo|milyon)))*|\d+(?:st|nd|rd|th))\b",
            score=0.9
        )
        figure_recognizer = PatternRecognizer(
            supported_entity="FIGURE",
            patterns=[figure_pattern],
            supported_language="tl",
            context=["bilang", "number", "halaga", "amount", "peso", "suweldo"]
        )

        email_pattern = Pattern(
            name="email_pattern",
            regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            score=0.85
        )
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[email_pattern],
            supported_language="tl",
            context=["email", "correo", "e-mail", "koreo"]
        )

        month_pattern = Pattern(
            name="month_pattern",
            regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Enero|Pebrero|Marso|Abril|Mayo|Hunyo|Hulyo|Agosto|Setyembre|Oktubre|Nobyembre|Disyembre)\b",
            score=0.95
        )
        month_recognizer = PatternRecognizer(
            supported_entity="MONTH",
            patterns=[month_pattern],
            supported_language="tl",
            context=["buwan", "month", "petsa", "date", "araw", "day"]
        )

        # Add custom recognizers to registry
        registry = RecognizerRegistry()
        registry.add_recognizer(figure_recognizer)
        registry.add_recognizer(email_recognizer)
        registry.add_recognizer(month_recognizer)
        self.analyzer.registry = registry

        # Supported entities (use Presidio's PERSON and LOCATION, plus custom entities)
        self.entities = ["PERSON", "LOCATION", "FIGURE", "EMAIL_ADDRESS", "MONTH"]

    def load_surnames(self, surname_file: str) -> set:
        """Load surnames from a text file for fictitious name generation."""
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
            pii_results = []

            # Analyze text with Presidio
            try:
                analyzer_results = self.analyzer.analyze(text=text, entities=self.entities, language="tl")
            except Exception as e:
                print(f"Error analyzing text with Presidio: {e}")
                analyzer_results = []

            # Collect PII results, mapping LOCATION to PHYSICAL_ADDRESS
            for result in analyzer_results:
                entity_type = result.entity_type
                if entity_type == "LOCATION":
                    entity_type = "PHYSICAL_ADDRESS"  # Map to match original behavior
                original_pii = text[result.start:result.end]
                placeholder = f"[{entity_type}]"
                mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
                pii_results.append((result.start, result.end, original_pii, entity_type))

                # Calculate timeline
                text_length = len(text)
                if text_length > 0:
                    duration = end_time - start_time
                    pii_start_time = max(start_time, start_time + (result.start / text_length) * duration)
                    pii_end_time = min(end_time, start_time + (result.end / text_length) * duration)
                    self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, entity_type))

            # Perform initial anonymization
            operators = {
                "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[PHYSICAL_ADDRESS]"}),
                "FIGURE": OperatorConfig("replace", {"new_value": "[FIGURE]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_ADDRESS]"}),
                "MONTH": OperatorConfig("replace", {"new_value": "[MONTH]"})
            }
            try:
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=operators
                )
                modified_text = anonymized_result.text
            except Exception as e:
                print(f"Error anonymizing text: {e}")
                modified_text = text

            # Post-process for DOB ([FIGURE][MONTH][FIGURE])
            dob_pattern = r'\[FIGURE\][ ,./]*\[MONTH\][ ,./]*\[FIGURE\]'
            dob_matches = list(re.finditer(dob_pattern, modified_text))
            if dob_matches:
                dob_matches.reverse()  # Process in reverse to avoid index issues
                for match in dob_matches:
                    dob_start = match.start()
                    dob_end = match.end()
                    dob_text = match.group(0)

                    # Reconstruct original DOB text
                    figure1_start = dob_start
                    figure1_end = figure1_start + len("[FIGURE]")
                    month_start = modified_text.find("[MONTH]", figure1_end)
                    month_end = month_start + len("[MONTH]")
                    figure2_start = modified_text.find("[FIGURE]", month_end)
                    figure2_end = figure2_start + len("[FIGURE]")

                    original_components = []
                    for start, end, orig_pii, entity in pii_results:
                        if (start == figure1_start and entity == "FIGURE") or \
                           (start == month_start and entity == "MONTH") or \
                           (start == figure2_start and entity == "FIGURE"):
                            original_components.append(orig_pii)

                    original_dob = dob_text
                    if len(original_components) == 3:
                        original_dob = modified_text[figure1_start:figure2_end].replace(
                            "[FIGURE]", original_components[0], 1).replace(
                            "[MONTH]", original_components[1], 1).replace(
                            "[FIGURE]", original_components[2], 1)

                    mappings["[DOB]"] = mappings.get("[DOB]", []) + [original_dob]

                    # Calculate DOB timeline
                    text_length = len(modified_text)
                    if text_length > 0:
                        duration = end_time - start_time
                        dob_start_time = max(start_time, start_time + (dob_start / text_length) * duration)
                        dob_end_time = min(end_time, start_time + (dob_end / text_length) * duration)
                        self.pii_timelines.append((dob_start_time, dob_end_time, original_dob, "DOB"))

                    pii_results.append((dob_start, dob_end, original_dob, "DOB"))
                    pii_results = [(s, e, p, ent) for s, e, p, ent in pii_results
                                   if not (s in (figure1_start, month_start, figure2_start) and ent in ("FIGURE", "MONTH"))]
                    self.pii_timelines = [(s, e, p, ent) for s, e, p, ent in self.pii_timelines
                                          if not (ent in ("FIGURE", "MONTH") and
                                                  figure1_start <= s < figure2_end)]

                    modified_text = modified_text[:dob_start] + "[DOB]" + modified_text[dob_end:]

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

    def create_fictitious_transcription(self) -> List[Dict]:
        """Create a transcription with [PERSON] replaced by consistent fictitious Filipino names and [EMAIL_ADDRESS] partially masked."""
        fictitious_transcription = []
        available_surnames = list(self.surnames)
        name_mapping = {}

        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            fictitious_text = masked_text
            if "[PERSON]" in mappings:
                for original_name in mappings["[PERSON]"]:
                    if "[PERSON]" in fictitious_text:
                        original_name_key = original_name.lower()
                        if original_name_key not in name_mapping:
                            first_name = random.choice(self.first_names)
                            surname = random.choice(available_surnames).capitalize()
                            name_mapping[original_name_key] = f"{first_name} {surname}"
                        fictitious_name = name_mapping[original_name_key]
                        fictitious_text = fictitious_text.replace("[PERSON]", fictitious_name, 1)
            if "[EMAIL_ADDRESS]" in mappings:
                for original_email in mappings["[EMAIL_ADDRESS]"]:
                    if "[EMAIL_ADDRESS]" in fictitious_text:
                        parts = original_email.split('@')
                        if len(parts) == 2:
                            local, domain = parts
                            domain_parts = domain.rsplit('.', 1)
                            if len(domain_parts) == 2:
                                domain_name, extension = domain_parts
                                masked_local = local[0] + '*' * (len(local) - 1)
                                masked_domain = domain_name[0] + '*' * (len(domain_name) - 1)
                                masked_email = f"{masked_local}@{masked_domain}.{extension}"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                            else:
                                masked_email = f"{local[0] + '*' * (len(local) - 1)}@*****"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                        else:
                            masked_email = "*****@*****"
                            fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
            fictitious_entry = masked_entry.copy()
            fictitious_entry["dialogue"] = fictitious_text
            fictitious_transcription.append(fictitious_entry)
        return fictitious_transcription

    def get_pii_timelines(self) -> List[Tuple[float, float, str, str]]:
        """Return timelines of PII occurrences."""
        return self.pii_timelines

class PresidioPIIMasker:
    def __init__(self, regions_file: str = "philippines_regions.txt"):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        # List of common Filipino first names for fictitious name substitution
        self.first_names = [
            "Mark", "Anna", "Jose", "Maria", "John", "Clara", "Pedro", "Liza",
            "Ramon", "Teresa", "Ben", "Sofia", "Luis", "Emma", "Carlos"
        ]
        # Surnames and regions will be loaded dynamically if needed
        self.surnames = set()
        self.regions = self.load_regions(regions_file)

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

        # Custom recognizer for physical addresses (city/area names)
        physical_address_pattern = Pattern(
            name="physical_address_pattern",
            regex=r'\b(?:' + '|'.join(re.escape(region) for region in self.regions) + r')\b',
            score=0.95
        )
        physical_address_recognizer = PatternRecognizer(
            supported_entity="PHYSICAL_ADDRESS",
            patterns=[physical_address_pattern]
        )
        self.analyzer.registry.add_recognizer(physical_address_recognizer)

        # All supported Presidio entities plus custom
        self.entities = [
            "CREDIT_CARD", "CRYPTO", "DATE_TIME", "DOMAIN_NAME", "EMAIL_ADDRESS",
            "IBAN_CODE", "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER",
            "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER", "US_LICENSE_PLATE",
            "US_ITIN", "US_PASSPORT", "US_SSN", "SG_NRIC_FIN", "AU_ABN",
            "AU_ACN", "AU_TFN", "AU_MEDICARE", "UK_NHS", "CREDIT_CARD_ENDING",
            "CONTRACT_REF", "ACCOUNT_NUMBER", "FIGURE", "PHYSICAL_ADDRESS"
        ]

    def load_surnames(self, surname_file: str) -> None:
        """Load surnames for fictitious name substitution if needed."""
        try:
            if os.path.exists(surname_file):
                with open(surname_file, "r", encoding="utf-8") as f:
                    self.surnames = {line.strip().lower() for line in f if line.strip()}
                print(f"Loaded {len(self.surnames)} surnames for fictitious names")
            else:
                print(f"Warning: Surname file {surname_file} not found for fictitious names.")
        except Exception as e:
            print(f"Error loading surnames for fictitious names: {e}")

    def load_regions(self, regions_file: str) -> set:
        """Load city/area names from a text file into a set for physical address matching."""
        regions = set()
        try:
            if os.path.exists(regions_file):
                with open(regions_file, "r", encoding="utf-8") as f:
                    regions = {line.strip() for line in f if line.strip()}
                print(f"Loaded {len(regions)} regions from {regions_file}")
            else:
                print(f"Warning: Regions file {regions_file} not found.")
        except Exception as e:
            print(f"Error loading regions: {e}")
        return regions

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in transcription, store mappings, and calculate PII timelines."""
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        for entry in transcription:
            text = entry["dialogue"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            analyzer_results = self.analyzer.analyze(
                text=text, entities=self.entities, language="en"
            )
            mappings = {}
            for result in analyzer_results:
                original_pii = text[result.start:result.end]
                placeholder = f"[{result.entity_type}]"
                mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
                text_length = len(text)
                if text_length > 0:
                    duration = end_time - start_time
                    pii_start_time = max(start_time, start_time + (result.start / text_length) * duration)
                    pii_end_time = min(end_time, start_time + (result.end / text_length) * duration)
                    self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, result.entity_type))
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
        """Return timestamps of PII occurrences."""
        return self.pii_timelines

    def create_fictitious_transcription(self, surname_file: str) -> List[Dict]:
        """Create a transcription with [PERSON] replaced by consistent fictitious Filipino names and [EMAIL_ADDRESS] partially masked."""
        if not self.surnames:
            self.load_surnames(surname_file)
        fictitious_transcription = []
        available_surnames = list(self.surnames)  # Convert set to list for random selection
        name_mapping = {}  # Dictionary to store original name to fictitious name mappings

        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            fictitious_text = masked_text
            # Replace [PERSON] with consistent fictitious names
            if "[PERSON]" in mappings:
                for original_name in mappings["[PERSON]"]:
                    if "[PERSON]" in fictitious_text:
                        # Normalize original name to lowercase for consistent mapping
                        original_name_key = original_name.lower()
                        # Check if we already have a fictitious name for this original name
                        if original_name_key not in name_mapping:
                            # Generate a new fictitious name
                            first_name = random.choice(self.first_names)
                            surname = random.choice(available_surnames).capitalize()
                            name_mapping[original_name_key] = f"{first_name} {surname}"
                        # Use the mapped fictitious name
                        fictitious_name = name_mapping[original_name_key]
                        fictitious_text = fictitious_text.replace("[PERSON]", fictitious_name, 1)
            # Replace [EMAIL_ADDRESS] with partially masked email
            if "[EMAIL_ADDRESS]" in mappings:
                for original_email in mappings["[EMAIL_ADDRESS]"]:
                    if "[EMAIL_ADDRESS]" in fictitious_text:
                        # Split email into local part, domain, and extension
                        parts = original_email.split('@')
                        if len(parts) == 2:
                            local, domain = parts
                            domain_parts = domain.rsplit('.', 1)
                            if len(domain_parts) == 2:
                                domain_name, extension = domain_parts
                                # Keep first character of local part and domain name, mask the rest
                                masked_local = local[0] + '*' * (len(local) - 1)
                                masked_domain = domain_name[0] + '*' * (len(domain_name) - 1)
                                masked_email = f"{masked_local}@{masked_domain}.{extension}"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                            else:
                                # Fallback if no extension
                                masked_email = f"{local[0] + '*' * (len(local) - 1)}@*****"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                        else:
                            # Fallback for malformed email
                            masked_email = "*****@*****"
                            fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
            fictitious_entry = masked_entry.copy()
            fictitious_entry["dialogue"] = fictitious_text
            fictitious_transcription.append(fictitious_entry)
        return fictitious_transcription

class PydubPIIMasker:
    """PII audio masker using pydub."""
    
    def mask_audio(self, audio_path: str, beep_path: str, pii_timelines: List[Tuple[float, float, str, str]]) -> str:
        """Mask PII segments in audio using a beep sound using pydub."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep file not found: {beep_path}")
            
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        
        print(f"Loading beep file: {beep_path}")
        beep = AudioSegment.from_wav(beep_path)
        
        pii_timelines_sorted = sorted(pii_timelines, key=lambda x: x[0])
        
        print(f"Processing {len(pii_timelines_sorted)} PII occurrences...")
        
        processed_segments = []
        last_end = 0
        
        for i, (start_time, end_time, text, entity_type) in enumerate(pii_timelines_sorted):
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            print(f"Processing PII segment {i+1}/{len(pii_timelines_sorted)}: {entity_type} - '{text}'")
            print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s ({start_ms}ms - {end_ms}ms)")
            
            if start_ms > last_end:
                print(f"  Adding segment from {last_end}ms to {start_ms}ms")
                segment = audio[last_end:start_ms]
                processed_segments.append(segment)
            
            duration_ms = end_ms - start_ms
            
            if len(beep) < duration_ms:
                print(f"  Extending beep to match PII duration: {duration_ms}ms")
                beep_segment = beep * (int(duration_ms / len(beep)) + 1)
                beep_segment = beep_segment[:duration_ms]
            else:
                print(f"  Trimming beep to match PII duration: {duration_ms}ms")
                beep_segment = beep[:duration_ms]
            
            segment_volume = audio[start_ms:end_ms].dBFS
            if segment_volume > -float('inf'):
                print(f"  Matching volume: {segment_volume:.2f} dBFS")
                beep_segment = beep_segment.apply_gain(segment_volume - beep_segment.dBFS)
            
            processed_segments.append(beep_segment)
            
            last_end = end_ms
        
        if last_end < len(audio):
            print(f"  Adding final segment from {last_end}ms to end")
            processed_segments.append(audio[last_end:])
        
        print("Combining all audio segments...")
        masked_audio = sum(processed_segments)
        
        output_path = audio_path.replace('.wav', '_masked.wav')
        print(f"Exporting masked audio to: {output_path}")
        masked_audio.export(output_path, format="wav")
        
        return output_path

def main():
    # Input files (example paths, can be modified)
    transcription_path = "test_main/taglish_calls/ashu_test/Payment Plan - Approved_Marjorie Casas_Clarissa Elizan_613834760548_2025-01-24_10.30.39_transcription.json"
    audio_path = "test_main/taglish_calls/ashu_test/Payment Plan - Approved_Marjorie Casas_Clarissa Elizan_613834760548_2025-01-24_10.30.39.wav"
    beep_path = "beep.wav"
    surname_file = "filipino_surnames.txt"
    regions_file = "philippines_regions.txt"

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

    # Detect language
    language, filipino_percentage = detect_language(transcription, surname_file)
    print(f"Detected language: {language}")

    # Initialize appropriate masker
    if language == "filipino":
        masker = FilipinoPIIMasker(surname_file=surname_file)
        masked_output = "filipino_masked_transcription.json"
        unmasked_output = "filipino_unmasked_transcription.json"
    else:
        masker = PresidioPIIMasker(regions_file=regions_file)
        masked_output = "presidio_masked_transcription.json"
        unmasked_output = "presidio_unmasked_transcription.json"

    # Mask transcription and get PII timelines
    try:
        # First call mask_transcription to populate pii_mappings and pii_timelines
        _ = masker.mask_transcription(transcription)
        # Use fictitious transcription for masked output
        masked_transcription = masker.create_fictitious_transcription() if language == "filipino" else masker.create_fictitious_transcription(surname_file)
        with open(masked_output, "w", encoding="utf-8") as f:
            json.dump(masked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error masking transcription: {e}")
        return

    # Unmask transcription
    try:
        unmasked_transcription = masker.unmask_transcription()
        with open(unmasked_output, "w", encoding="utf-8") as f:
            json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error unmasking transcription: {e}")
        return

    # Get PII timelines
    pii_timelines = masker.get_pii_timelines()

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
    # print("\nUnmasked Transcription:")
    # print(json.dumps(unmasked_transcription, indent=2, ensure_ascii=False))
    # print("\nPII Timelines (start, end, text, entity_type):")
    # print(json.dumps(pii_timelines, indent=2, ensure_ascii=False))
    
    # print(f"\nSuccessfully processed {len(pii_timelines)} PII occurrences in the audio.")
    # print(f"Masked audio saved as: {output_path}")
    # print(f"Detected language: {language}")
    # print(f"Language percentage: {filipino_percentage}")
    
if __name__ == "__main__":
    main()