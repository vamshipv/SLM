import torch
from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 1. LOAD THE EXISTING RACECAR
# We do NOT create a new LlamaConfig. We load the brain you already trained!
model_path = "./baby-llama-racecar/checkpoint-1410"
tokenizer_path = "./python-tokenizer"

print("Loading existing Racecar model...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# --> ADD THESE TWO LINES BACK IN <--
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "<|endoftext|>"

# 2. FREEZE THE FOUNDATION (The Add-on Strategy)
# Your model has 8 layers. We will freeze the first 4.
print("Freezing the bottom 4 layers...")
for layer_idx, layer in enumerate(model.model.layers):
    if layer_idx < 4:
        for param in layer.parameters():
            param.requires_grad = False # This locks the math!

# Verify what is frozen
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Frozen Parameters: {frozen_params:,}")
print(f"Trainable Parameters (The Add-on): {trainable_params:,}")

# 3. THE NEW 'CLEAN' DATASET (The Stack - General Python)
# We use 'the-stack-smol' which has a massive variety of normal python code
print("Loading General Python Dataset...")
raw_dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")

# Grab a fresh 10,000 scripts to overwrite the Data Science bias
dataset = raw_dataset.shuffle(seed=42).select(range(10000))
dataset_split = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    # This checks if 't' is actually text. If 'The Stack' gives us a blank None file, 
    # it turns it into an empty string before adding the Stop Sign.
    texts = [
        (t if t is not None else "") + tokenizer.eos_token 
        for t in examples["content"]
    ]
    return tokenizer(texts, truncation=True, max_length=512)

tokenized_datasets = dataset_split.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["content", "repository_name", "size"] # clean up extra columns
)

# 4. THE GENTLE COACHING PLAN
# We must use a MUCH smaller Learning Rate. We are doing 'surgery', not 'building'
training_args = TrainingArguments(
    output_dir="./baby-llama-generalist",
    num_train_epochs=5,             # 5 is enough since it already knows syntax
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,  
    eval_strategy="epoch",    
    learning_rate=5e-5,             # VERY LOW! We don't want to shock the brain.
    weight_decay=0.01,              
    logging_steps=50,
    save_strategy="epoch",
    fp16=True,                      
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\nStarting Add-on Training Phase...")
trainer.train()