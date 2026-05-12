from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

# Load your brand new custom dictionary
tokenizer = PreTrainedTokenizerFast.from_pretrained("./python-tokenizer")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "<|endoftext|>"

# Build the beefier 19M brain
config = LlamaConfig(
    vocab_size=8000,
    hidden_size=512,        # Improvement from 256 to 512 for better understanding
    num_hidden_layers=8,    #  More layers to capture complex patterns
    num_attention_heads=8,
    intermediate_size=1024, # More processing power
    max_position_embeddings=512
)
model = LlamaForCausalLM(config)