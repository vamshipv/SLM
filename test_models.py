from transformers import pipeline

# 1. Point this to the checkpoint folder the Trainer auto-saved
model_path = "./day3-overfit-test/checkpoint-50" 

print(f"Loading broken model from {model_path}...")

# 2. Load the pipeline with your corrupted weights, but standard tokenizer
generator = pipeline(
    "text-generation", 
    model=model_path, 
    tokenizer="gpt2",
    # We use basic greedy decoding for now to see its true, unfiltered state
    do_sample=False 
)

# 3. Try two different prompts

# Prompt A: Something from the Wikipedia dataset
print("\n--- TEST 1: Memorization Test ---")
prompt_a = "The game began development in"
output_a = generator(prompt_a, max_new_tokens=30)
print(output_a[0]['generated_text'])

# Prompt B: Something completely unrelated
print("\n--- TEST 2: Confusion Test ---")
prompt_b = "My favorite recipe for chocolate cake is"
output_b = generator(prompt_b, max_new_tokens=30)
print(output_b[0]['generated_text'])

# Here you can see the model's "broken" behavior. It should have memorized the first prompt and produce a coherent continuation, 
# while the second prompt should yield a nonsensical output, 
# demonstrating that the model has overfitted to the training data and lost its generalization ability.