from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import os

# 1. Load the same dataset we used for training
print("Loading data for tokenizer training...")
dataset = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
dataset = dataset.shuffle(seed=42).select(range(5000))

# 2. Save the raw text to a temporary file so the trainer can read it
with open("python_corpus.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["content"] + "\n")

# 3. Initialize and train the tokenizer
# 8000 is our target vocab size. 
tokenizer = ByteLevelBPETokenizer()

print("Training tokenizer (this will be fast)...")
tokenizer.train(files=["python_corpus.txt"], vocab_size=8000, min_frequency=2,
                special_tokens=[
                    "<|endoftext|>",
                    "<s>",
                    "</s>",
                    "<pad>",
                    "<unk>"
                ])

# 4. Save the files
os.makedirs("python-tokenizer", exist_ok=True)
tokenizer.save_model("python-tokenizer")
# Save a fast tokenizer JSON so transformers can load it directly
# without needing to convert a slow tokenizer at runtime.
tokenizer.save("python-tokenizer/tokenizer.json")
print("Done! Tokenizer saved in folder: ./python-tokenizer")