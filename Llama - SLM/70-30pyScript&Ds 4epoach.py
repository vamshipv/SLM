import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets, interleave_datasets

# 1. LOAD THE CUSTOM 8K TOKENIZER
tokenizer = PreTrainedTokenizerFast.from_pretrained("./python-tokenizer")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "<|endoftext|>"

# 2. PREPARE THE BALANCED DIET
print("Preparing the Balanced Diet (70% General / 30% DS)...")

# Load General Python
ds_general = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
ds_general = ds_general.shuffle(seed=42)

# Load Data Science
ds_specialist = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_specialist = ds_specialist.shuffle(seed=42).select(range(4285))

# --- THE FIX: STRIP TO CONTENT ONLY ---
# We force both datasets to have ONLY the 'content' column. 
# This makes their schemas identical and perfectly aligned.
ds_general = ds_general.select_columns(["content"])
ds_specialist = ds_specialist.select_columns(["content"])

# Now the shuffle will work perfectly!
balanced_dataset = interleave_datasets(
    [ds_general, ds_specialist], 
    probabilities=[0.7, 0.3], 
    seed=42
)

# Proceed with split and tokenize as before...
dataset_split = balanced_dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    texts = [(t if t is not None else "") + tokenizer.eos_token for t in examples["content"]]
    return tokenizer(texts, truncation=True, max_length=512)

tokenized_datasets = dataset_split.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset_split["train"].column_names
)

# 3. THE RACECAR BRAIN (Fresh Start)
# We are training from scratch to let the mix settle in the weights
config = LlamaConfig(
    vocab_size=8000,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=512
)
model = LlamaForCausalLM(config)

# 4. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./baby-llama-v3-balanced-3Ep",
    num_train_epochs=4, 
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, 
    learning_rate=6e-4, # A solid "middle ground" rate
    weight_decay=0.1,   # Higher decay to fight hallucinations
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Starting the Balanced 70/30 Training...")
trainer.train()