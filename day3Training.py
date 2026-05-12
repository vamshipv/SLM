import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

print("1. Wheeling Frankenstein into the OR...")
model_path = "./qwen-medical-frankenstein"

# Load the model and tokenizer from yesterday
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure the pad token is set properly so the Trainer doesn't complain
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n2. Prepping the Medical Text (The Medicine)...")
# Instead of streaming, we download a small, potent slice of 5,000 sentences
# This easily fits in your system RAM and gives the Trainer a concrete finish line
dataset = load_dataset("medmcqa", split="train[:5000]")

# We need to convert the English text into the model's numbers
def tokenize_function(examples):
    # Combine the question and explanation into one paragraph
    texts = [q + " " + (e if e else "") for q, e in zip(examples['question'], examples['exp'])]
    print(f"Texts to tokenize: {texts[:2]}")  # Print the first 2 for sanity check
    return tokenizer(texts, truncation=True, max_length=128)

print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# The Data Collator takes our tokenized sentences and perfectly formats them into GPU blocks
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n3. Setting up the Life Support (8GB VRAM Limits)...")
training_args = TrainingArguments(
    output_dir="./qwen-medical-cured",
    overwrite_output_dir=True,
    
    # --- HARDWARE SURVIVAL SETTINGS ---
    per_device_train_batch_size=2,  # Only 2 sentences in VRAM at a time
    gradient_accumulation_steps=4,  # Wait for 4 loops before updating (Simulates batch size of 8)
    fp16=True,                      # Cut memory size in half (16-bit precision)
    
    # --- BRAIN DAMAGE PREVENTION ---
    learning_rate=2e-5,             # Microscopic learning rate! Do not scramble the English grammar!
    num_train_epochs=1,             # We only read the 5,000 sentences once to avoid overfitting
    
    # --- LOGGING ---
    logging_steps=10,               # Print an update to the terminal every 10 steps
    save_steps=500,                 
)

print("\n4. Starting the Heart Defibrillator (Training)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# This is where the actual math happens!
trainer.train()

print("\n5. Surgery Complete. Saving the Cured Model...")
trainer.save_model("./qwen-medical-cured")
tokenizer.save_pretrained("./qwen-medical-cured")

print("Patient is stable and saved to ./qwen-medical-cured!")