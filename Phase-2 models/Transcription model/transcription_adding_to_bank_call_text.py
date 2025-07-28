import os
from pathlib import Path

def extract_statements_from_transcriptions(trans_dir):
    """Extract statements from transcription files."""
    statements = set()  # Use a set to avoid duplicates
    trans_files = sorted(Path(trans_dir).glob("*_transcription.txt"))
    
    for trans_path in trans_files:
        with open(trans_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            # Replace newlines with spaces, split on periods to approximate sentences
            text = text.replace('\n', ' ')
            trans_sentences = text.split('.')
            # Clean and add non-empty sentences
            for sentence in trans_sentences:
                sentence = sentence.strip()
                if sentence:  # Ignore empty strings
                    statements.add(sentence)
    
    return statements

def append_to_bank_call_text(text_dir, new_statements):
    """Append new statements to bank_call_text.txt, avoiding duplicates."""
    text_file = Path(text_dir) / "bank_call_text.txt"
    
    # Read existing statements to avoid duplicates
    existing_statements = set()
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            existing_statements.update(line.strip() for line in lines if line.strip())
    
    # Add new statements that aren't already in the file
    statements_to_add = new_statements - existing_statements
    if not statements_to_add:
        print("No new statements to add to bank_call_text.txt.")
        return
    
    # Append new statements
    with open(text_file, 'a', encoding='utf-8') as f:
        for statement in sorted(statements_to_add):  # Sort for consistency
            f.write(statement + '\n')
    
    print(f"Added {len(statements_to_add)} new statements to {text_file}.")

def main():
    trans_dir = "Dataset/Transcription"
    text_dir = "Dataset/Text_data"
    
    # Extract statements from transcriptions
    new_statements = extract_statements_from_transcriptions(trans_dir)
    
    # Append to bank_call_text.txt
    append_to_bank_call_text(text_dir, new_statements)

if __name__ == "__main__":
    main()