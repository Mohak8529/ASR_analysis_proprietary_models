from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer  # Add this import

# Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = Whitespace()  # Split on whitespace, like Whisper

# Define special tokens (mimicking Whisper)
special_tokens = [
    "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|tl|>", "[PAD]", "[UNK]"
]

# Train the tokenizer using bank_call_text.txt
trainer = BpeTrainer(vocab_size=50000, special_tokens=special_tokens)
tokenizer.train(files=["Dataset/Text_data/bank_call_text.txt"], trainer=trainer)

# Save the tokenizer
tokenizer.save("custom_taglish_tokenizer.json")
print("Custom tokenizer saved to custom_taglish_tokenizer.json")