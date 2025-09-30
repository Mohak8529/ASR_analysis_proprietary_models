def save_transcript(text: str, file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
