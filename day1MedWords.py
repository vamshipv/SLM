import re
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

print("1. Loading Qwen's Tokenizer...")
# We use Qwen 2.5 (0.5B) to see exactly how its specific dictionary reacts
tokenizer_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print("2. Connecting to Hugging Face Medical Stream...")
# We use a medical Q&A dataset. 
# streaming=True means it acts like Netflix, downloading one chunk at a time.
dataset = load_dataset("medmcqa", split="train", streaming=True)

# We will store our word frequencies here
word_counter = Counter()

print("3. Scanning Medical Data (This will take a minute)...")
max_sentences_to_read = 50000  # We stop at 50k to save time. It's enough for a statistical sample.
sentences_read = 0

for row in dataset:
    # Grab the medical text from the row
    text = row['question'] + " " + row['exp'] if row['exp'] else row['question']
    
    # Clean the text: convert to lowercase and strip out punctuation
    clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
    # Split into words and keep only "jargon" (words longer than 7 letters)
    # This acts as a cheap filter to ignore words like "the", "and", "patient"
    words = [w for w in clean_text.split() if len(w) > 7]
    word_counter.update(words)
    
    sentences_read += 1
    if sentences_read >= max_sentences_to_read:
        break

print(f"Finished reading {sentences_read} sentences. Found {len(word_counter)} unique long words.")
print("4. Calculating ROI (Return on Investment) for Surgery...")

roi_scores = []

# We only calculate ROI for words that appeared at least 10 times to avoid rare typos
for word, freq in word_counter.items():
    if freq >= 10:
        # Ask Qwen to tokenize the word and see how many pieces it shatters into
        tokens = tokenizer.tokenize(word)
        fragments = len(tokens)
        
        # If it's already 1 token, ROI is 0 (it doesn't need fixing)
        if fragments > 1:
            # The Math: How many wasted compute cycles does this word cause?
            roi = freq * (fragments - 1)
            roi_scores.append({
                "word": word,
                "fragments": fragments,
                "frequency": freq,
                "roi": roi,
                "token_preview": tokens # So you can see how badly it broke
            })

# Sort the list so the most damaging words (Highest ROI) are at the very top
roi_scores.sort(key=lambda x: x['roi'], reverse=True)

print("5. Saving the Top 500 words to surgery_list.txt...")
top_500 = roi_scores[:500]

with open("surgery_list.txt", "w", encoding="utf-8") as f:
    f.write("Word | ROI | Freq | Fragments | Token Breakdown\n")
    f.write("-" * 80 + "\n")
    for item in top_500:
        f.write(f"{item['word']} | {item['roi']} | {item['frequency']} | {item['fragments']} | {item['token_preview']}\n")

print("Done! Open surgery_list.txt to see the targets.")