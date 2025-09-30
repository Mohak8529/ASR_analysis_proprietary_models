import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_FILE = "synthetic_filipino_pii.json"
OUTPUT_MODEL_PATH = "fine_tuned_xx_ent_wiki_sm"

# Load synthetic data
try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"Loaded {len(train_data)} training examples")
except FileNotFoundError:
    logger.error(f"{DATA_FILE} not found.")
    exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON format in {DATA_FILE}: {e}")
    exit(1)
except Exception as e:
    logger.error(f"Error loading data: {e}")
    exit(1)

# Map synthetic data labels to fine-tuned model labels
label_mapping = {
    "PERSON": "PERSON",
    "ADDRESS": "LOCATION",
    "PHONE_NUMBER": "FIGURE",
    "EMAIL": "EMAIL_ADDRESS",
    "DATE": "MONTH"
}

# Process data to align labels
processed_data = []
for text, annotations in train_data:
    new_annotations = []
    for start, end, label in annotations:
        new_label = label_mapping.get(label, None)
        if new_label:
            new_annotations.append([start, end, new_label])
        else:
            logger.warning(f"Unknown label {label} in annotations for text: {text[:50]}...")
    if new_annotations:  # Only include samples with valid annotations
        processed_data.append([text, new_annotations])

# Split data into train (80%) and validation (20%)
random.seed(42)
random.shuffle(processed_data)
train_size = int(0.8 * len(processed_data))
train_set = processed_data[:train_size]
val_set = processed_data[train_size:]
logger.info(f"Training set: {len(train_set)} samples, Validation set: {len(val_set)} samples")

# Load the base model
try:
    nlp = spacy.load("xx_ent_wiki_sm")
    logger.info("Loaded xx_ent_wiki_sm model")
except Exception as e:
    logger.error(f"Error loading model: {e}. Please install with: pip install spacy==3.4.4 && python -m spacy download xx_ent_wiki_sm")
    exit(1)

# Add PII labels to the NER pipeline
ner = nlp.get_pipe("ner")
labels = ["PERSON", "LOCATION", "FIGURE", "EMAIL_ADDRESS", "MONTH"]  # Removed DOB
for label in labels:
    ner.add_label(label)

# Disable other pipelines to focus on NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    n_iterations = 30  # Increased from 20
    best_f1 = 0.0
    patience = 5
    no_improvement = 0
    
    for itn in range(n_iterations):
        logger.info(f"Iteration {itn + 1}/{n_iterations}")
        random.shuffle(train_set)
        losses = {}
        batches = minibatch(train_set, size=compounding(8.0, 128.0, 1.001))  # Increased max batch size
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                spans = [(start, end, label) for start, end, label in annotations]
                try:
                    example = Example.from_dict(doc, {"entities": spans})
                    examples.append(example)
                except ValueError as e:
                    logger.warning(f"Skipping invalid example: {text[:50]}... Error: {e}")
                    continue
            try:
                nlp.update(examples, drop=0.4, sgd=optimizer, losses=losses)
            except ValueError as e:
                logger.warning(f"Error in nlp.update: {e}. Skipping batch.")
                continue
        
        # Evaluate on validation set
        val_examples = []
        for text, annotations in val_set:
            doc = nlp.make_doc(text)
            spans = [(start, end, label) for start, end, label in annotations]
            try:
                example = Example.from_dict(doc, {"entities": spans})
                val_examples.append(example)
            except ValueError as e:
                logger.warning(f"Skipping invalid validation example: {text[:50]}... Error: {e}")
                continue
        
        try:
            scores = nlp.evaluate(val_examples)
            f1 = scores.get("ents_f", 0.0)
            logger.info(f"Training Losses: {losses}")
            logger.info(f"Validation Scores - F1: {f1:.4f}, Precision: {scores.get('ents_p', 0.0):.4f}, Recall: {scores.get('ents_r', 0.0):.4f}")
            logger.info(f"Per-entity F1: {scores.get('ents_per_type', {})}")
            
            # Early stopping
            if f1 > best_f1 + 0.01:  # Significant improvement
                best_f1 = f1
                no_improvement = 0
                # Save best model so far
                nlp.to_disk(OUTPUT_MODEL_PATH + "_best")
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping at iteration {itn + 1}: No F1 improvement for {patience} iterations")
                    break
        except Exception as e:
            logger.warning(f"Error in validation: {e}")
            continue

# Save the final fine-tuned model
try:
    nlp.to_disk(OUTPUT_MODEL_PATH)
    logger.info(f"Final fine-tuned model saved to {OUTPUT_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    exit(1)