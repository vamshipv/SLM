from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading your overfitted checkpoint...")
model_path = "./day3-overfit-test/checkpoint-50" # Make sure this matches your folder!
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure pad token is set so we don't get errors
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

prompt = "The game began development in"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: '{prompt}'\n")

# 1. GREEDY DECODING (Boring, prone to loops)
print("--- 1. GREEDY ---")
greedy_output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# 2. HIGH TEMPERATURE (Creative but risky)
print("\n--- 2. TEMPERATURE (High: 1.5) ---")
temp_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=1.5)
print(tokenizer.decode(temp_output[0], skip_special_tokens=True))

# 3. TOP-K (Safe creativity)
print("\n--- 3. TOP-K (K=50) ---")
topk_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, top_k=50)
print(tokenizer.decode(topk_output[0], skip_special_tokens=True))

# 4. TOP-P (Dynamic creativity - Industry Standard)
print("\n--- 4. TOP-P (P=0.9) ---")
topp_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, top_p=0.9)
print(tokenizer.decode(topp_output[0], skip_special_tokens=True))

# 5. THE HYBRID (What production models actually use)
print("\n--- 5. HYBRID (Temp=0.7, Top-P=0.9) ---")
hybrid_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(hybrid_output[0], skip_special_tokens=True))