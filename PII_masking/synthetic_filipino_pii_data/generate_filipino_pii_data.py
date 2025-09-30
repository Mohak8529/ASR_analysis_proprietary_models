import json
import os
import random
from typing import List, Tuple
from faker import Faker
import re
import spacy
from spacy.training import offsets_to_biluo_tags
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker and spaCy
fake = Faker()
nlp = spacy.blank("tl")  # Blank Tagalog model

# Custom tokenizer for Filipino
def custom_tokenizer(nlp):
    special_cases = {
        "ng": [{"ORTH": "ng"}],
        "sa": [{"ORTH": "sa"}],
        "ay": [{"ORTH": "ay"}],
        "po": [{"ORTH": "po"}],
        "opo": [{"ORTH": "opo"}]
    }
    return spacy.tokenizer.Tokenizer(nlp.vocab, rules=special_cases)

nlp.tokenizer = custom_tokenizer(nlp)

# Define paths
SURNAME_FILE = "filipino_surnames.txt"
REGIONS_FILE = "philippines_regions.txt"
OUTPUT_FILE = "synthetic_filipino_pii.json"
TEMPLATE_FILE = "updated_templates.txt"

# Load templates from updated_templates.txt
try:
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        templates = eval(f.read().replace("templates = ", ""))
    if not isinstance(templates, list):
        raise ValueError("Templates file must contain a valid Python list")
    logger.info(f"Loaded {len(templates)} templates from {TEMPLATE_FILE}")
except Exception as e:
    logger.error(f"Error loading templates from {TEMPLATE_FILE}: {e}")
    exit(1)

# Load surnames and regions
def load_surnames(surname_file: str) -> set:
    surnames = set()
    try:
        if os.path.exists(surname_file):
            with open(surname_file, "r", encoding="utf-8") as f:
                surnames = {line.strip().lower() for line in f if line.strip()}
            logger.info(f"Loaded {len(surnames)} surnames from {surname_file}")
        else:
            logger.warning(f"Surname file {surname_file} not found.")
    except Exception as e:
        logger.error(f"Error loading surnames: {e}")
    return surnames

def load_regions(regions_file: str) -> set:
    regions = set()
    try:
        if os.path.exists(regions_file):
            with open(regions_file, "r", encoding="utf-8") as f:
                regions = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(regions)} regions from {regions_file}")
        else:
            logger.warning(f"Regions file {regions_file} not found.")
    except Exception as e:
        logger.error(f"Error loading regions: {e}")
    return regions

# Normalize text to prevent extra spaces
def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())

# Load data
first_names = [
    "Mark", "Anna", "Jose", "Maria", "John", "Clara", "Pedro", "Liza",
    "Ramon", "Teresa", "Ben", "Sofia", "Luis", "Emma", "Carlos", "Marjorie", "Clarissa"
]
surnames = load_surnames(SURNAME_FILE)
regions = load_regions(REGIONS_FILE)

# Generate complex addresses
def generate_complex_addresses(num_samples: int, regions: set) -> List[str]:
    address_templates = [
        "Block {block} Lot {lot} {region}",
        "{street} Street {region}",
        "{region} {city}",
        "Unit {unit} {region} City",
    ]
    streets = ["Rizal", "Mabini", "Bonifacio", "Luna", "Quezon", "Palau", "Riel"]
    cities = ["Calamba", "Manila", "Quezon City", "Davao", "Cebu", "Laguna"]
    addresses = list(regions)
    for _ in range(num_samples // 2):
        template = random.choice(address_templates)
        address = template.format(
            block=random.randint(1, 50),
            lot=random.randint(1, 100),
            region=random.choice(list(regions)),
            city=random.choice(cities),
            street=random.choice(streets),
            unit=random.randint(1, 100)
        )
        addresses.append(normalize_text(address))
    return addresses

# Generate phone numbers in Philippine formats
def generate_ph_phone_numbers(num_samples: int) -> List[str]:
    formats = [
        "09{0}",  # 09XXXXXXXXX
        "+639{0}",  # +639XXXXXXXXX
        "0{0}{1}"  # 0XXXXXXXXXX (provincial)
    ]
    mobile_prefixes = ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"]
    provincial_codes = ["32", "33", "34", "35", "36", "37", "38", "42", "43", "44"]
    phone_numbers = []
    
    for _ in range(num_samples):
        fmt = random.choice(formats)
        if fmt == "09{0}":
            prefix = random.choice(mobile_prefixes)
            digits = "".join([str(random.randint(0, 9)) for _ in range(7)])
            number = fmt.format(prefix + digits)
        elif fmt == "+639{0}":
            prefix = random.choice(mobile_prefixes)
            digits = "".join([str(random.randint(0, 9)) for _ in range(7)])
            number = fmt.format(prefix + digits)
        elif fmt == "0{0}{1}":
            area_code = random.choice(provincial_codes)
            digits = "".join([str(random.randint(0, 9)) for _ in range(7)])
            number = fmt.format(area_code, digits)
        phone_numbers.append(number)
    
    return phone_numbers

# Fake data mappings
fake_data = {
    "name": [f"{first_name} {surname.capitalize()}" for first_name in first_names for surname in surnames] + first_names,
    "address": generate_complex_addresses(5000, regions),
    "phone_number": generate_ph_phone_numbers(5000),
    "email": [
        f"{first_name.lower()}.{surname.lower()}@example.com" for first_name in first_names for surname in surnames
    ] + [
        f"{random.choice(['info', 'contact', 'support', 'admin'])}@{random.choice(['mgcollect.ph', 'maya.ph', 'example.com', 'gmail.com'])}"
        for _ in range(500)
    ],
    "date": [
        f"{month} {random.randint(1, 28)} {random.randint(1970, 2005)}"
        for month in [
            "Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo", "Hulyo",
            "Agosto", "Setyembre", "Oktubre", "Nobyembre", "Disyembre",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        ]
        for _ in range(50)
    ]
}

# Adjust entity spans to align with token boundaries
def adjust_entity_spans(doc, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    adjusted_entities = []
    for start, end, label in sorted(entities, key=lambda x: x[0]):
        start_token = None
        end_token = None
        for token in doc:
            token_end = token.idx + len(token.text)
            # Allow off-by-one character misalignment
            if token.idx <= start <= token_end + 1:
                start_token = token
            if token.idx - 1 <= end <= token_end or (end == token_end and token.i == len(doc) - 1):
                end_token = token
        if start_token and end_token:
            adjusted_start = start_token.idx
            adjusted_end = end_token.idx + len(end_token.text)
            if adjusted_start < adjusted_end <= len(doc.text):
                adjusted_entities.append((adjusted_start, adjusted_end, label))
            else:
                logger.debug(f"Invalid adjusted span ({adjusted_start}, {adjusted_end}, {label}) for text: {doc.text[:50]}...")
        else:
            logger.debug(f"Could not adjust entity: ({start}, {end}, {label}) in text: {doc.text[:50]}...")
    return adjusted_entities

# Check for overlapping entities
def has_overlapping_entities(entities: List[Tuple[int, int, str]]) -> bool:
    sorted_entities = sorted(entities, key=lambda x: x[0])
    for i in range(len(sorted_entities) - 1):
        current_end = sorted_entities[i][1]
        next_start = sorted_entities[i + 1][0]
        if current_end > next_start:
            logger.debug(f"Overlapping entities detected: {sorted_entities[i]} and {sorted_entities[i + 1]}")
            return True
    return False

# Validate entity spans and store in doc.spans
def validate_entities(doc, entities: List[Tuple[int, int, str]]) -> bool:
    try:
        # Check basic span validity
        for start, end, label in entities:
            if start < 0 or end > len(doc.text) or start >= end:
                logger.debug(f"Invalid entity span: ({start}, {end}, {label}) in text: {doc.text[:50]}...")
                return False
        
        # Check for overlaps
        if has_overlapping_entities(entities):
            return False

        # Validate alignment with BILUO tags, allowing one misaligned tag
        biluo_tags = offsets_to_biluo_tags(doc, [(start, end, label) for start, end, label in entities])
        if '-' in biluo_tags and sum(1 for tag in biluo_tags if tag == '-') <= 1:
            logger.debug(f"Allowing minor misalignment: {biluo_tags}")
            return True
        elif '-' in biluo_tags:
            logger.debug(f"Misaligned entities detected: {biluo_tags} in text: {doc.text[:50]}...")
            return False

        # Store entities in doc.spans to handle potential overlaps during training
        doc.spans["entities"] = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is None:
                logger.debug(f"Could not create span for ({start}, {end}, {label}) in text: {doc.text[:50]}...")
                return False
            doc.spans["entities"].append(span)
        
        return True
    except ValueError as e:
        logger.debug(f"Validation error: {e} in text: {doc.text[:50]}...")
        return False

# Add Taglish noise with alignment preservation
def add_noise(text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
    original_text = text
    original_entities = entities.copy()
    
    doc = nlp.make_doc(text)
    if not validate_entities(doc, entities):
        logger.debug(f"Skipping noise addition due to initial misalignment: {text[:50]}...")
        return original_text, original_entities

    # Add Taglish conversational noise (20% chance)
    if random.random() < 0.2:
        noise = random.choice([" po", " opo", " naman", " eh", " ha", " Apo", " siguro", " kasi"])
        text += noise
        doc = nlp.make_doc(text)
        if not validate_entities(doc, entities):
            logger.debug(f"Noise '{noise}' caused misalignment in: {text[:50]}...")
            return original_text, original_entities

    # Add code-switching prefixes (20% chance)
    if random.random() < 0.2:
        prefix = random.choice(["Okay ", "Hello ", "So ", "Good morning "])
        text = prefix + text
        entities = [(start + len(prefix), end + len(prefix), label) for start, end, label in entities]
        doc = nlp.make_doc(text)
        if not validate_entities(doc, entities):
            logger.debug(f"Prefix '{prefix}' caused misalignment in: {text[:50]}...")
            return original_text, original_entities

    # Add mid-sentence noise in non-entity regions (20% chance)
    if random.random() < 0.2:
        non_entity_ranges = []
        last_end = 0
        sorted_entities = sorted(entities, key=lambda x: x[0])
        for start, end, _ in sorted_entities:
            if last_end < start:
                non_entity_ranges.append((last_end, start))
            last_end = end
        if last_end < len(text):
            non_entity_ranges.append((last_end, len(text)))
        if non_entity_ranges:
            valid_range = random.choice(non_entity_ranges)
            if valid_range[1] - valid_range[0] > 1:
                idx = random.randint(valid_range[0] + 1, valid_range[1] - 1)
                noise = random.choice([" eh ", " Apo ", " siguro ", " kasi "])
                text = text[:idx] + noise + text[idx:]
                entities = [
                    (start + len(noise) if start >= idx else start, end + len(noise) if end >= idx else end, label)
                    for start, end, label in entities
                ]
                doc = nlp.make_doc(text)
                if not validate_entities(doc, entities):
                    logger.debug(f"Mid-sentence noise '{noise}' at index {idx} caused misalignment in: {text[:50]}...")
                    return original_text, original_entities

    # Add typos in non-entity regions (10% chance)
    if random.random() < 0.1 and len(text) > 5:
        non_entity_ranges = []
        last_end = 0
        sorted_entities = sorted(entities, key=lambda x: x[0])
        for start, end, _ in sorted_entities:
            if last_end < start:
                non_entity_ranges.append((last_end, start))
            last_end = end
        if last_end < len(text):
            non_entity_ranges.append((last_end, len(text)))
        if non_entity_ranges:
            valid_range = random.choice(non_entity_ranges)
            if valid_range[1] - valid_range[0] > 1:
                idx = random.randint(valid_range[0] + 1, valid_range[1] - 1)
                text = text[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[idx + 1:]
                entities = [
                    (start + 1 if start > idx else start, end + 1 if end > idx else end, label)
                    for start, end, label in entities
                ]
                doc = nlp.make_doc(text)
                if not validate_entities(doc, entities):
                    logger.debug(f"Typo at index {idx} caused misalignment in: {text[:50]}...")
                    return original_text, original_entities

    # Final alignment adjustment
    doc = nlp.make_doc(text)
    adjusted_entities = adjust_entity_spans(doc, entities)
    if not adjusted_entities:
        logger.debug(f"Failed to adjust entities for: {text[:50]}...")
        return original_text, original_entities

    if not validate_entities(doc, adjusted_entities):
        logger.debug(f"Adjusted entities still misaligned for: {text[:50]}...")
        return original_text, original_entities

    return text, adjusted_entities

# Generate synthetic data with balanced entity distribution and dynamic target
def generate_synthetic_data(num_samples: int, templates: List[str], fake_data: dict) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    spacy_data = []
    entity_counts = {"PERSON": 0, "ADDRESS": 0, "PHONE_NUMBER": 0, "EMAIL": 0, "DATE": 0}
    misaligned_samples = 0
    max_attempts = num_samples * 20  # Increased to allow more attempts
    attempts = 0
    
    # Estimate average entities per template
    total_entities_per_template = 0
    for template in templates:
        entities = len(re.findall(r"\{\{(\w+)\}\}", template))
        total_entities_per_template += entities
    avg_entities_per_sample = total_entities_per_template / len(templates) if templates else 1
    # Initial target per entity to approach num_samples, capped to prevent extremes
    target_per_entity = max(200, min(int(num_samples / (5 * avg_entities_per_sample)) * 2, num_samples // 2))
    
    template_groups = {
        "PERSON": [t for t in templates if "{{name}}" in t],
        "ADDRESS": [t for t in templates if "{{address}}" in t],
        "PHONE_NUMBER": [t for t in templates if "{{phone_number}}" in t],
        "EMAIL": [t for t in templates if "{{email}}" in t],
        "DATE": [t for t in templates if "{{date}}" in t]
    }
    
    while len(spacy_data) < num_samples and attempts < max_attempts:
        attempts += 1
        # Recalculate target_per_entity dynamically to approach num_samples
        if spacy_data:
            current_total_entities = sum(entity_counts.values())
            current_samples = len(spacy_data)
            if current_total_entities > 0 and current_samples > 0:
                avg_entities_per_sample = current_total_entities / current_samples
                remaining_samples = num_samples - current_samples
                if avg_entities_per_sample > 0:
                    target_per_entity = max(200, min(int(remaining_samples * avg_entities_per_sample / 5), num_samples // 2))
        
        # Select entity with furthest-from-target count, with fallback to any available entity
        candidate_entities = [
            entity for entity in entity_counts
            if template_groups[entity]  # Ensure entity has available templates
        ]
        if not candidate_entities:
            logger.warning("No templates available for any entity. Stopping generation.")
            break
        
        max_deviation = max((target_per_entity - entity_counts[entity]) / target_per_entity for entity in candidate_entities) if candidate_entities else 0
        eligible_entities = [
            entity for entity in candidate_entities
            if (target_per_entity - entity_counts[entity]) / target_per_entity >= max_deviation * 0.9
        ] or candidate_entities  # Fallback to any available entity if empty
        min_entity = random.choice(eligible_entities)
        
        template = random.choice(template_groups[min_entity])
        sample_entities = set(re.findall(r"\{\{(\w+)\}\}", template))
        if not sample_entities:
            continue
        
        data = {
            "name": random.choice(fake_data["name"]),
            "address": random.choice(fake_data["address"]),
            "phone_number": random.choice(fake_data["phone_number"]),
            "email": random.choice(fake_data["email"]),
            "date": random.choice(fake_data["date"])
        }
        
        text = template
        entities = []
        for entity_type, value in data.items():
            placeholder = "{{" + entity_type + "}}"
            if placeholder in text:
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                end = start + len(value)
                entity_label = "PERSON" if entity_type == "name" else entity_type.upper()
                entities.append((start, end, entity_label))
        
        text = normalize_text(text)
        doc = nlp.make_doc(text)
        entities = adjust_entity_spans(doc, entities)
        if not entities:
            logger.debug(f"Skipping sample due to unadjustable entities: {text[:50]}...")
            misaligned_samples += 1
            continue
        
        if has_overlapping_entities(entities):
            logger.debug(f"Skipping sample due to overlapping entities: {text[:50]}... Entities: {entities}")
            misaligned_samples += 1
            continue
        
        if not validate_entities(doc, entities):
            logger.debug(f"Skipping sample due to invalid entities: {text[:50]}... Entities: {entities}")
            misaligned_samples += 1
            continue
        
        text, entities = add_noise(text, entities)
        doc = nlp.make_doc(text)
        if not validate_entities(doc, entities):
            logger.debug(f"Skipping sample after noise addition: {text[:50]}... Entities: {entities}")
            misaligned_samples += 1
            continue
        
        # Add sample and update counts
        spacy_data.append((text, entities))
        for _, _, entity_label in entities:
            entity_counts[entity_label] += 1
        
        # Log progress every 1000 samples
        if len(spacy_data) % 1000 == 0:
            logger.info(f"Progress: {len(spacy_data)} samples, Entity counts: {entity_counts}, Current target_per_entity: {target_per_entity}")
    
    if len(spacy_data) < num_samples:
        logger.warning(f"Generated only {len(spacy_data)} samples out of {num_samples} due to strict validation or insufficient valid samples")
    
    logger.info(f"Generated {len(spacy_data)} samples. Misaligned samples skipped: {misaligned_samples}")
    logger.info(f"Entity distribution: {entity_counts}")
    return spacy_data

# Generate and save data
num_samples = 10000
synthetic_data = generate_synthetic_data(num_samples, templates, fake_data)
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Synthetic data saved to {OUTPUT_FILE} with {len(synthetic_data)} samples")
except Exception as e:
    logger.error(f"Error saving synthetic data: {e}")

# Validate the generated dataset
def validate_dataset(data: List[Tuple[str, List[Tuple[int, int, str]]]]):
    invalid_samples = []
    for text, entities in data:
        doc = nlp.make_doc(text)
        if not validate_entities(doc, entities):
            invalid_samples.append((text, entities))
    if invalid_samples:
        logger.warning(f"Found {len(invalid_samples)} invalid samples in the dataset")
        for text, entities in invalid_samples[:5]:  # Log first 5 for brevity
            logger.debug(f"Invalid sample: {text[:50]}... Entities: {entities}")
    else:
        logger.info("All samples in the dataset are valid")

validate_dataset(synthetic_data)

# Print the number of templates at the end of execution
print(f"Number of templates loaded from {TEMPLATE_FILE}: {len(templates)}")