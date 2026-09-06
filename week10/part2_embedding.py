"""
Week 10 Day 4 — Part 2: Inside BERT's Embeddings
==================================================
Run: python w10d4_part2_embeddings.py

This script opens up DistilBERT and shows you the three
embedding tables and how the [CLS] vector changes through layers.
"""

import torch
from transformers import AutoTokenizer, AutoModel

# ── Load model and tokenizer ────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Put model in evaluation mode (no dropout during inference)
model.eval()

print("=" * 60)
print("EXPLORING DISTILBERT'S INTERNALS")
print("=" * 60)

# ── 1. The three embedding tables ───────────────────────────
print("\n1. EMBEDDING TABLES")
print("-" * 40)

embeddings = model.embeddings

# Token embedding: what word is this?
print(f"   Token embedding shape:    {embeddings.word_embeddings.weight.shape}")
print(f"   → 30,522 vocabulary words × 768 dimensions")

# Position embedding: where is this word?
print(f"   Position embedding shape: {embeddings.position_embeddings.weight.shape}")
print(f"   → 512 possible positions × 768 dimensions")
print(f"   → (max sequence length = 512 tokens)")

# No segment embedding in DistilBERT (simplification over full BERT)
# Full BERT has: embeddings.token_type_embeddings (2, 768)
print(f"   Note: DistilBERT skips segment embedding (simplification)")

# ── 2. Look up a specific word's vector ─────────────────────
print("\n2. WORD VECTOR LOOKUP")
print("-" * 40)

# Convert word to ID
word = "movie"
word_id = tokenizer.convert_tokens_to_ids(word)
print(f"   '{word}' → ID: {word_id}")

# Grab its vector from the embedding table
word_vector = embeddings.word_embeddings.weight[word_id]
print(f"   Vector shape: {word_vector.shape}")
print(f"   First 10 values: {word_vector[:10].tolist()}")
print(f"   → These 768 numbers encode everything BERT learned about '{word}'")

# ── 3. Similar words have similar vectors ───────────────────
print("\n3. WORD SIMILARITY (cosine similarity)")
print("-" * 40)


def get_word_vector(word):
    """Grab raw embedding vector for a word."""
    token_id = tokenizer.convert_tokens_to_ids(word)
    return embeddings.word_embeddings.weight[token_id].detach()


def cosine_sim(v1, v2):
    """How similar are two vectors? 1.0 = identical, 0.0 = unrelated."""
    return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# Compare word pairs
pairs = [
    ("movie", "film"),  # similar meaning → high similarity
    ("love", "hate"),  # opposite meaning but same category → medium
    ("love", "movie"),  # different categories → lower
    ("good", "great"),  # similar sentiment → high
    ("good", "terrible"),  # opposite sentiment → lower
]

for w1, w2 in pairs:
    v1, v2 = get_word_vector(w1), get_word_vector(w2)
    sim = cosine_sim(v1, v2)
    bar = "█" * int(sim * 20)
    print(f"   {w1:>10} ↔ {w2:<10}  sim={sim:.3f}  {bar}")

# ── 4. How token + position combine ────────────────────────
print("\n4. TOKEN + POSITION EMBEDDING ADDITION")
print("-" * 40)

sentence = "the movie was great"
encoded = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Get token embeddings (lookup by ID)
token_embeds = embeddings.word_embeddings(encoded['input_ids'])
print(f"   Token embeddings shape: {token_embeds.shape}")
print(f"   → (1 sentence, {token_embeds.shape[1]} tokens, 768 dims)")

# Get position embeddings (positions 0, 1, 2, ...)
seq_length = encoded['input_ids'].shape[1]
position_ids = torch.arange(seq_length).unsqueeze(0)
position_embeds = embeddings.position_embeddings(position_ids)
print(f"   Position embeddings shape: {position_embeds.shape}")

# Add them together (element-wise addition — same as your Week 2 broadcasting!)
combined = token_embeds + position_embeds
print(f"   Combined shape: {combined.shape}")
print(f"   → Same shape — just added element-wise, then layer-normed")

# ── 5. [CLS] output — the sentence summary ─────────────────
print("\n5. [CLS] OUTPUT — THE SENTENCE SUMMARY VECTOR")
print("-" * 40)

# Run the full model (all 6 transformer layers)
with torch.no_grad():  # no gradients needed for inference
    outputs = model(**encoded)

# outputs.last_hidden_state shape: (batch, seq_len, 768)
# Every token position has a 768-dim output vector
print(f"   Full output shape: {outputs.last_hidden_state.shape}")
print(f"   → (1 sentence, {outputs.last_hidden_state.shape[1]} tokens, 768 dims)")

# [CLS] is at position 0
cls_vector = outputs.last_hidden_state[0, 0, :]  # batch 0, token 0, all dims
print(f"   [CLS] vector shape: {cls_vector.shape}")
print(f"   First 10 values: {[round(v, 4) for v in cls_vector[:10].tolist()]}")

# ── 6. Same word, different context ─────────────────────────
print("\n6. CONTEXT CHANGES THE VECTOR")
print("-" * 40)
print("   Same word 'bank' in two different sentences:")

sentences = [
    "I went to the bank to deposit money",
    "I sat on the river bank and watched the water"
]

for sent in sentences:
    enc = tokenizer(sent, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)

    # Find where "bank" is in the token list
    token_list = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    bank_idx = token_list.index("bank")
    bank_vector = out.last_hidden_state[0, bank_idx, :]

    print(f"\n   '{sent}'")
    print(f"     'bank' at position {bank_idx}")
    print(f"     First 5 values: {[round(v, 4) for v in bank_vector[:5].tolist()]}")

# Compare the two "bank" vectors
enc1 = tokenizer(sentences[0], return_tensors="pt")
enc2 = tokenizer(sentences[1], return_tensors="pt")
with torch.no_grad():
    out1, out2 = model(**enc1), model(**enc2)

tok1 = tokenizer.convert_ids_to_tokens(enc1['input_ids'][0])
tok2 = tokenizer.convert_ids_to_tokens(enc2['input_ids'][0])
b1 = out1.last_hidden_state[0, tok1.index("bank"), :]
b2 = out2.last_hidden_state[0, tok2.index("bank"), :]

sim = cosine_sim(b1, b2)
print(f"\n   Similarity between the two 'bank' vectors: {sim:.3f}")
print(f"   → Less than 1.0! Self-attention made them different")
print(f"   → BERT understands 'bank' means different things in each sentence")

print("\n" + "=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
Before the Transformer layers: 'bank' has the SAME vector regardless of context.
After the Transformer layers:  'bank' has DIFFERENT vectors based on context.

This is the power of self-attention — it transforms static word vectors
into context-aware representations. The raw embedding table is just
the starting point. The Transformer layers do the real work.

For classification: we grab the [CLS] vector from the FINAL layer output.
That single 768-dim vector summarizes the entire sentence's meaning.
We feed it to a simple Linear(768 → 2) to predict positive/negative.
""")