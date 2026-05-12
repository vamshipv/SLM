import torch
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 1. THE TOKENIZER & THE STOP SIGN
# We use the GPT-2 tokenizer. It uses <|endoftext|> as the EOS (End Of String) token.
# This is how the model learns to NOT repeat 'return True' forever.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. THE DATA PIPELINE (Massive Scaling)
print("Loading 5,000 real Python scripts...")
raw_dataset = load_dataset("huggingface-course/codeparrot-ds-train", split="train")

# We shuffle and pick 5,000 scripts to avoid overwhelming your 19M model
# Then we split it: 90% to learn, 10% to test (The Validation Drawer)
dataset = raw_dataset.shuffle(seed=42).select(range(5000))
dataset_split = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    # We add the EOS token to the end of every script!
    texts = [t + tokenizer.eos_token for t in examples["content"]]
    return tokenizer(texts, truncation=True, max_length=256)

tokenized_datasets = dataset_split.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["content"]
)

# 3. THE 19M BRAIN (Llama Architecture)
config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    max_position_embeddings=256
)
model = LlamaForCausalLM(config)

# 4. PROFESSIONAL TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./baby-llama-pro",
    overwrite_output_dir=True,
    num_train_epochs=3,             # 3 rounds is enough to start generalizing
    per_device_train_batch_size=8,  # Batching 8 scripts at once for stability
    eval_strategy="epoch",    # Test the model after every epoch
    learning_rate=5e-4,             # Lower than before to be more 'precise'
    weight_decay=0.01,              # Prevents memorization (The Rubber Band)
    logging_steps=50,
    save_strategy="epoch",
    fp16=True                       # Uses your 4060's Tensor Cores for speed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # Testing on the 'Locked Drawer'
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\nStarting Massive Scaling Training...")
trainer.train()

# Discarding the model because the GPT2 took too much space and the python script had very space to contruct the meaningful code.