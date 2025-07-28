import torch
from dataset import PromiseToPayDataset
from model import PromiseToPayModel

def decode_keywords(tokens, keyword_logits, vocab):
    inverse_vocab = {v: k for k, v in vocab.items()}
    keyword_labels = torch.argmax(keyword_logits, dim=-1)
    keywords = []
    current_keyword = []
    for token, label in zip(tokens, keyword_labels):
        token_str = inverse_vocab.get(token.item(), "<unk>")
        if label == 1:  # B-KEY
            if current_keyword:
                keywords.append(" ".join(current_keyword))
                current_keyword = []
            current_keyword.append(token_str)
        elif label == 2:  # I-KEY
            current_keyword.append(token_str)
        else:  # O
            if current_keyword:
                keywords.append(" ".join(current_keyword))
                current_keyword = []
    if current_keyword:
        keywords.append(" ".join(current_keyword))
    return [kw for kw in keywords if kw]

def infer(file_id, data_dir, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = PromiseToPayDataset(data_dir)
    model = PromiseToPayModel(vocab_size=len(dataset.vocab)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    audio, audio_len, trans_tokens, transcr_tokens, trans_len, _, _ = dataset[dataset.files.index(file_id)]
    audio, audio_len, trans_tokens, transcr_tokens, trans_len = (
        audio.unsqueeze(0).to(device),
        torch.tensor([audio_len]).to(device),
        trans_tokens.unsqueeze(0).to(device),
        transcr_tokens.unsqueeze(0).to(device),
        torch.tensor([trans_len]).to(device)
    )

    with torch.no_grad():
        class_logits, keyword_logits = model(audio, audio_len, trans_tokens, transcr_tokens, trans_len)
        class_pred = torch.argmax(class_logits, dim=-1).item()
        keywords = decode_keywords(trans_tokens[0], keyword_logits[0], dataset.vocab)

    class_map = {v: k for k, v in dataset.label_map.items()}
    return class_map[class_pred], keywords

if __name__ == "__main__":
    file_id = "abc"  # Example file ID
    result, keywords = infer(file_id, "../dataset", "../checkpoints/latest.pth")
    print(f"Predicted Category: {result}")
    print(f"Critical Keywords: {keywords}")