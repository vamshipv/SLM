import torch
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 1. LOAD THE CUSTOM DICTIONARY
# This is the 8k 'Lego Box' we built earlier
tokenizer = PreTrainedTokenizerFast.from_pretrained("./python-tokenizer")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "<|endoftext|>"

# 2. PREPARE THE DATA (Massive Scaling)
print("Loading 5,000 scripts for the Racecar...")
raw_dataset = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
dataset = raw_dataset.shuffle(seed=42).select(range(5000))
dataset_split = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    # Add the 'Stop Sign' so it learns to stop talking
    texts = [t + tokenizer.eos_token for t in examples["content"]]
    return tokenizer(texts, truncation=True, max_length=512) # 512 token memory

tokenized_datasets = dataset_split.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["content"]
)

# 3. THE RACECAR BRAIN (19M Parameters)
# We freed up parameters from the vocab to beef up the layers and hidden size
config = LlamaConfig(
    vocab_size=8000,
    hidden_size=512,        # Higher resolution 'detail'
    intermediate_size=1024, # More 'scratchpad' space
    num_hidden_layers=8,    # Deeper logic processing
    num_attention_heads=8,
    max_position_embeddings=512
)
model = LlamaForCausalLM(config)

# Calculate parameters to confirm we are around 19M-20M
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters.")

# 4. THE COACHING PLAN (Training Arguments)
training_args = TrainingArguments(
    output_dir="./baby-llama-racecar",
    num_train_epochs=10,            # 10 rounds to deeply learn patterns
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,  # Effective batch size of 32 for stability
    eval_strategy="epoch",    
    learning_rate=8e-4,             # Slightly higher for small models
    weight_decay=0.05,              # Stronger 'rubber band' to prevent memorizing
    lr_scheduler_type="cosine",     # Slows down as it gets closer to the finish line
    logging_steps=50,
    save_strategy="epoch",
    fp16=True,                      # Use your 4060's speed
    report_to="none"
)

# 5. START TRAINING
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\nFiring up the Racecar Engine...")
trainer.train()