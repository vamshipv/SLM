import torch
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset

print("1. Borrowing a Tokenizer (The Alphabet)...")
# We borrow a standard fast tokenizer just so the model knows how to read English letters.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("2. Spawning the Baby Llama...")
config = LlamaConfig(
    vocab_size=len(tokenizer),  # Dynamically match the tokenizer's dictionary size!
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    max_position_embeddings=512
)
model = LlamaForCausalLM(config)

print("3. Writing the 'Python Codebook'...")
# The "Code Textbook" - Common Python patterns
textbook = [
    "def hello_world():\n    print('hello')\n",
    "if x > 5:\n    return True\n",
    "for i in range(10):\n    print(i)\n",
    "class MyModel(nn.Module):\n    def __init__(self):\n",
    "import torch\nimport numpy as np\n"
] * 100  # Repeat 100 times so the model sees the 'def' and ':' patterns constantly

# Convert the text into numbers
dataset = Dataset.from_dict({"text": textbook})
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=16)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n4. Starting the Heartbeat (Training)...")
training_args = TrainingArguments(
    output_dir="./baby-llama",
    overwrite_output_dir=True,
    num_train_epochs=10,            # Read the tiny book 100 times
    per_device_train_batch_size=4,  # Read 4 sentences at a time
    learning_rate=1e-3,             # Massive learning rate! We WANT to violently overwrite the random static.
    logging_steps=10,               # Print vitals every 10 steps
    save_strategy="no"              # Don't bother saving checkpoints to your hard drive yet
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Start the math!
trainer.train()

print("\n5. Testing the Baby's Brain...")
# Let's see if it learned Python!
prompt = "print("
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a continuation
outputs = model.generate(**inputs, max_new_tokens=30, temperature=1.5, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nPrompt: {prompt}")
print(f"Baby Llama says: {response}")

