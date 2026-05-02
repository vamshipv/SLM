import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("1. Prepping the Operating Table...")
model_name = "Qwen/Qwen2.5-0.5B"

# Load the Tokenizer and the Base Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# We load the model in standard precision (float32 or bfloat16) for the surgery
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Original Dictionary Size: {len(tokenizer)} tokens")
print(f"Original Matrix Size: {model.get_input_embeddings().weight.shape}")

print("\n2. Extracting Targets from surgery_list.txt...")
new_medical_words = []

# Read the text file we generated yesterday
with open("surgery_list.txt", "r", encoding="utf-8") as f:
    # Read all lines, but skip the first 2 lines (the headers we made)
    lines = f.readlines()[2:]
    
    for line in lines:
        # Split the line by the "|" character and grab the very first item (the word)
        word = line.split("|")[0].strip()
        new_medical_words.append(word)

print(f"Loaded {len(new_medical_words)} words for injection. (e.g., {new_medical_words[0]}, {new_medical_words[1]})")

print("\n3. Injecting Words into the Dictionary...")
# This fixes the BPE algorithm so it stops shattering these specific words
num_added_toks = tokenizer.add_tokens(new_medical_words)
print(f"Successfully added {num_added_toks} new tokens to the dictionary.")
print(f"New Dictionary Size: {len(tokenizer)} tokens")

print("\n4. Expanding the Neural Network (The Surgery)...")
# This is the actual matrix resizing. PyTorch welds new rows to the bottom.
model.resize_token_embeddings(len(tokenizer))

print(f"New Matrix Size: {model.get_input_embeddings().weight.shape}")

print("\n5. Saving the Frankenstein Model...")
# We save this modified, untrained model to a local folder
save_directory = "./qwen-medical-frankenstein"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Surgery Complete! Model saved to {save_directory}.")