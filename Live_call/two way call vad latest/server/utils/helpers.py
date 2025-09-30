# server/utils/helpers.py

def save_transcript(text: str, file_path: str):
    """Save transcript text to file"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()

def is_meaningful_text(text: str) -> bool:
    """
    Accept any non-empty transcript as meaningful (no minimum character restriction,
    no word or number restrictions).
    """
    if not text or not text.strip():
        return False
    return True
