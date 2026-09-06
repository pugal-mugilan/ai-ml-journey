"""
Week 10 Day 4 — Part 1: Understanding the Tokenizer
=====================================================
Run: pip install transformers torch
Then: python w10d4_part1_tokenizer.py

This script shows what the tokenizer does to your text
before BERT ever sees it.
"""

from transformers import AutoTokenizer

# ── Step 1: Load the tokenizer ──────────────────────────────
# This downloads DistilBERT's vocabulary (30,522 tokens)
# and its rules for splitting text into subwords.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ── Step 2: Tokenize a sentence ─────────────────────────────
sentence = "I loved the unbelievable acting"

# tokenizer() does everything: lowercase → split → add [CLS]/[SEP] → convert to IDs
encoded = tokenizer(sentence, return_tensors="pt")  # "pt" = PyTorch tensors

print("=" * 60)
print("INPUT:", sentence)
print("=" * 60)

# What tokens did it produce?
tokens = tokenizer.tokenize(sentence)
print(f"\n1. Subword tokens: {tokens}")
print(f"   Count: {len(tokens)} tokens from {len(sentence.split())} words")

# What IDs do those tokens map to?
print(f"\n2. Token IDs:  {encoded['input_ids'].tolist()[0]}")
print(f"   [101] = [CLS], [102] = [SEP], rest = your words")

# What's the attention mask?
print(f"\n3. Attention mask: {encoded['attention_mask'].tolist()[0]}")
print(f"   1 = real token, 0 = padding (no padding here since single sentence)")

# ── Step 3: See subword splitting in action ─────────────────
print("\n" + "=" * 60)
print("SUBWORD SPLITTING EXAMPLES")
print("=" * 60)

test_words = [
    "unbelievable",    # un + ##bel + ##ie + ##va + ##ble
    "playing",         # play + ##ing
    "transformers",    # transform + ##ers
    "AI",              # ai (lowercased, uncased model)
    "chatbot",         # chat + ##bot
]

for word in test_words:
    subtokens = tokenizer.tokenize(word)
    ids = tokenizer.convert_tokens_to_ids(subtokens)
    print(f"\n  '{word}'")
    print(f"    → tokens: {subtokens}")
    print(f"    → IDs:    {ids}")

# ── Step 4: Batch tokenization (multiple sentences) ─────────
print("\n" + "=" * 60)
print("BATCH TOKENIZATION (padding + truncation)")
print("=" * 60)

sentences = [
    "I loved it",           # short sentence
    "This was the worst movie I have ever seen in my entire life"  # long sentence
]

# padding=True  → shorter sentences get [PAD] tokens (ID=0) to match longest
# truncation=True → sentences longer than max_length get cut
# max_length=16 → cap at 16 tokens
batch = tokenizer(sentences, padding=True, truncation=True,
                  max_length=16, return_tensors="pt")

for i, sent in enumerate(sentences):
    tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
    print(f"\n  Sentence {i+1}: '{sent}'")
    print(f"    Tokens: {tokens}")
    print(f"    IDs:    {batch['input_ids'][i].tolist()}")
    print(f"    Mask:   {batch['attention_mask'][i].tolist()}")

print("\n  Shape of input_ids:", batch['input_ids'].shape)
print("  → (2 sentences, 16 tokens each)")

# ── Key takeaway ────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
The tokenizer converts text → numbers that BERT understands.
It handles three things you'd otherwise have to do manually:
  1. Splitting words into subword pieces (vocabulary coverage)
  2. Adding special tokens ([CLS] at start, [SEP] at end)
  3. Padding shorter sentences so all have the same length

The output is a tensor of shape (batch_size, seq_length)
— exactly what the model expects as input.
""")