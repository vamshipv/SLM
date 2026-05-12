import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("1. Waking up the Frankenstein Model...")
model_directory = "./qwen-medical-frankenstein"

# We load from your local folder, NOT from the Hugging Face cloud
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

print("Model loaded successfully! Type 'quit' to exit.")
print("-" * 50)

# This creates an interactive loop so you can keep chatting with it
while True:
    user_prompt = input("\nEnter your prompt: ")
    
    if user_prompt.lower() == 'quit':
        print("Shutting down...")
        break

    # Convert your English text into the model's tokens
    inputs = tokenizer(user_prompt, return_tensors="pt")

    print("Generating response (watch for the hallucinations)...")
    
    # Generate the output (we use torch.no_grad() to save memory since we aren't training)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Limit output so it doesn't ramble forever
            temperature=0.7,    # A little bit of creativity
            do_sample=True
        )

    # Decode the mathematical tokens back into readable English
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Model Output ---")
    print(response)
    print("--------------------")