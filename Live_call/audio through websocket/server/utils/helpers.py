def save_transcript(text, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
