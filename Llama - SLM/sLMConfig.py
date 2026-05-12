from transformers import LlamaConfig, LlamaForCausalLM

print("1. Drafting the Blueprint (Micro-Llama)...")
# We design a completely custom, microscopic architecture
config = LlamaConfig(
    vocab_size=32000,           # Standard dictionary size
    hidden_size=256,            # The width of the matrix (Llama-3 is usually 4096)
    intermediate_size=512,      # The width of the SwiGLU hidden layer
    num_hidden_layers=4,        # Number of transformer blocks (Llama-3 has 32)
    num_attention_heads=8,      # Number of attention heads (Llama-3 has 32)
    max_position_embeddings=512 # Context window: How many words it can read at once
)

print("2. Spawning the Blank Matrix...")
# Notice we use 'from_config' instead of 'from_pretrained'
# This generates completely random numbers. The model is brain-dead.
model = LlamaForCausalLM(config)

print("\n--- The Vitals ---")
# This will print the actual PyTorch architecture so you can see the RMSNorm and RoPE modules
print(model)

print("\n--- The Scale ---")
# Count the exact number of random floating-point numbers we just created
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")