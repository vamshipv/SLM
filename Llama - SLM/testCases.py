import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# 1. Setup path and device
# If your folder has multiple 'checkpoint-XXX' folders, pick the one with the highest number
model_path = "./baby-llama-pro/checkpoint-1689" 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the Tokenizer and the Baby Brain
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = LlamaForCausalLM.from_pretrained(model_path).to(device)

# 3. The Test Function
def generate_code(prompt, tokens=30, temp=0.4):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # We use 'do_sample' to allow for that 'creativity'
    # We use 'pad_token_id' to keep the model from complaining
    outputs = model.generate(
        **inputs, 
        max_new_tokens=tokens, 
        temperature=0.1, # Keep it VERY strict
        top_p=0.9, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. The Moment of Truth
test_prompt = "user_age = 25\nif user_age"
print(f"\n--- Prompt ---\n{test_prompt}")

result = generate_code(test_prompt)
print(f"\n--- Baby Llama Response ---\n{result}")

# 5. Bonus: Try a different one!
print("\n--- Bonus Test (Importing) ---")
print(generate_code("import torch\nimport", tokens=10))