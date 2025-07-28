import os
import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split

# Define project root and data directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
opus_dir = os.path.join(project_root, "Dataset", "opus")
taglish_dir = os.path.join(project_root, "Dataset", "taglish_bank_data", "Taglish")
english_dir = os.path.join(project_root, "Dataset", "taglish_bank_data", "English")
processed_dir = os.path.join(project_root, "Dataset", "processed")
os.makedirs(processed_dir, exist_ok=True)

# Define OPUS file pairs
opus_datasets = [
    (os.path.join(opus_dir, "Tatoeba.en-tl"), "tatoeba"),
    (os.path.join(opus_dir, "ParaCrawl.en-tl"), "paracrawl"),
    (os.path.join(opus_dir, "WikiMatrix.en-tl"), "wikimatrix")
]

# Load OPUS data
all_data = []
for base_file, prefix in opus_datasets:
    en_file = f"{base_file}.en"
    tl_file = f"{base_file}.tl"
    print(f"Checking OPUS files: {en_file}, {tl_file}")
    if not os.path.exists(en_file) or not os.path.exists(tl_file):
        print(f"Warning: Missing file(s) for {prefix}")
        continue
    with open(en_file, "r", encoding="utf-8") as en_f, open(tl_file, "r", encoding="utf-8") as tl_f:
        en_lines = en_f.readlines()
        tl_lines = tl_f.readlines()
        min_len = min(len(en_lines), len(tl_lines))
        print(f"Loaded {min_len} lines for {prefix}")
        df = pd.DataFrame({
            "source": [line.strip() for line in tl_lines[:min_len]],
            "target": [line.strip() for line in en_lines[:min_len]],
            "dataset": prefix
        })
        all_data.append(df)

# Load Taglish data
taglish_files = [f for f in os.listdir(taglish_dir) if f.endswith(".txt")]
for taglish_file in taglish_files:
    taglish_path = os.path.join(taglish_dir, taglish_file)
    english_path = os.path.join(english_dir, taglish_file.replace("taglish", "english"))
    print(f"Checking Taglish files: {taglish_path}, {english_path}")
    if not os.path.exists(english_path):
        print(f"Warning: Missing English file for {taglish_file}")
        continue
    with open(taglish_path, "r", encoding="utf-8") as tl_f, open(english_path, "r", encoding="utf-8") as en_f:
        tl_lines = tl_f.readlines()
        en_lines = en_f.readlines()
        min_len = min(len(tl_lines), len(en_lines))
        print(f"Loaded {min_len} lines for Taglish {taglish_file}")
        df = pd.DataFrame({
            "source": [line.strip() for line in tl_lines[:min_len]],
            "target": [line.strip() for line in en_lines[:min_len]],
            "dataset": f"taglish_{taglish_file}"
        })
        all_data.append(df)

if not all_data:
    raise FileNotFoundError("No valid data files found.")

# Combine and clean data
data = pd.concat(all_data, ignore_index=True)
data = data.dropna()
data = data[data["source"].str.strip() != ""]
data = data[data["target"].str.strip() != ""]

# Subsample ParaCrawl if too large
para_crawl_data = data[data["dataset"] == "paracrawl"]
para_crawl_size = len(para_crawl_data)
sample_size = min(150000, para_crawl_size) if para_crawl_size > 0 else para_crawl_size
if sample_size > 0 and sample_size < para_crawl_size:
    para_crawl = para_crawl_data.sample(n=sample_size, random_state=42)
else:
    para_crawl = para_crawl_data
other_data = data[data["dataset"] != "paracrawl"]
final_data = pd.concat([other_data, para_crawl]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train, val, test
train_data, temp_data = train_test_split(final_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save preprocessed data
train_data.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
val_data.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
test_data.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

# Train SentencePiece tokenizer
input_files = [os.path.join(opus_dir, f"{base}.tl") for base, _ in opus_datasets if os.path.exists(f"{base}.tl")]
input_files += [os.path.join(opus_dir, f"{base}.en") for base, _ in opus_datasets if os.path.exists(f"{base}.en")]
input_files += [os.path.join(taglish_dir, f) for f in taglish_files]
input_files += [os.path.join(english_dir, f) for f in os.listdir(english_dir) if f.endswith(".txt")]
spm.SentencePieceTrainer.train(
    input=",".join(input_files),
    model_prefix=os.path.join(project_root, "models", "spm"),
    vocab_size=32000,
    model_type="bpe",
    user_defined_symbols=["<start>", "<end>"]
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
print("SentencePiece tokenizer trained and saved to models/spm.model")