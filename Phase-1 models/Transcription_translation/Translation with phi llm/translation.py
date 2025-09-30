# translate.py
from llama_cpp import Llama
import os

# Specify the path to the local model
model_path = os.path.join("models", "Phi-3-mini-4k-instruct-q4.gguf")

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please run download_model.py first.")
    exit(1)

# Load the model
try:
    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length for input/output
        n_threads=os.cpu_count() or 4,  # Use available CPU cores
        n_gpu_layers=0,  # CPU-only inference
        verbose=False  # Reduce logging
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define the translation prompt template (based on Phi-3 model card)
prompt_template = (
    "<|system|> You are a native from Phillipines and you understand the natural, candid and usual filipino+english dialect accurately. Your task is to translate Filipino-English text to English, strictly maintaining the context.First study the input and understand the natural code switching from filipino to English and vice versa, then translate it. Provide only the translated text in the response. <|end|>\n"
    "<|user|> Translate the following Filipino-English code-switched text to English:\n"
    "Input: {}\n"
    "<|end|>\n"
    "<|assistant|>"
)

# Example input (replace with your transcript)
input_text = "Then, pakisagot po yung mga tawag po namin sa kanya dahil tinatawagan po namin siya. Hindi niya po nasasagot po yung mga tawag namin maghapon. Pasabi na lang po sa kanya, Ma'am."

# Generate translation
try:
    prompt = prompt_template.format(input_text)
    output = llm(
        prompt,
        max_tokens=100,  # Limit output length
        temperature=0.7,  # Balance creativity and determinism
        stop=["<|end|>"],  # Stop at end token
        echo=False  # Exclude prompt from output
    )
    response = output["choices"][0]["text"].strip()
    # Extract the response after <|assistant|> tag
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[1].strip()
    print(f"Translation: {response}")
except Exception as e:
    print(f"Error during inference: {e}")
finally:
    # Clean up
    llm.reset()