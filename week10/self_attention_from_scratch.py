"""
Week 10 Day 2 — Self-Attention from Scratch (NumPy)
Single-head scaled dot-product attention on a toy sentence.
No PyTorch — every matrix multiply is visible.
"""

import numpy as np

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

# ============================================================
# STEP 1: Fake a sentence as embeddings
# ============================================================
# In reality, each word would go through an embedding layer (word → vector).
# We'll simulate that with random vectors.

sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)  # 6 tokens
d_model = 8  # embedding dimension (tiny for visibility; real models use 512-768)

# Each row is one token's embedding vector
# Shape: (seq_len, d_model) = (6, 8)
X = np.random.randn(seq_len, d_model)

print("=" * 60)
print("STEP 1: Input embeddings")
print(f"Sentence: {sentence}")
print(f"X shape: {X.shape}  →  ({seq_len} tokens, {d_model}-dim embeddings)")
print(f"Each row is one word's vector representation")
print()

# ============================================================
# STEP 2: Create Q, K, V weight matrices (these are LEARNED in real models)
# ============================================================
# d_k = dimension of queries and keys
# d_v = dimension of values (often same as d_k)
d_k = 4  # small for visibility
d_v = 4

# Weight matrices — in a real Transformer, these are nn.Linear layers trained via backprop
W_Q = np.random.randn(d_model, d_k)  # (8, 4)
W_K = np.random.randn(d_model, d_k)  # (8, 4)
W_V = np.random.randn(d_model, d_v)  # (8, 4)

print("=" * 60)
print("STEP 2: Projection weight matrices (learned parameters)")
print(f"W_Q shape: {W_Q.shape}  →  projects {d_model}-dim embeddings to {d_k}-dim queries")
print(f"W_K shape: {W_K.shape}  →  projects {d_model}-dim embeddings to {d_k}-dim keys")
print(f"W_V shape: {W_V.shape}  →  projects {d_model}-dim embeddings to {d_v}-dim values")
print()

# ============================================================
# STEP 3: Project input into Q, K, V
# ============================================================
# Every token gets THREE different representations
Q = X @ W_Q  # (6, 8) @ (8, 4) = (6, 4) — each token's "what am I looking for?"
K = X @ W_K  # (6, 8) @ (8, 4) = (6, 4) — each token's "what do I contain?"
V = X @ W_V  # (6, 8) @ (8, 4) = (6, 4) — each token's "what do I actually say?"

print("=" * 60)
print("STEP 3: Project X into Q, K, V")
print(f"Q shape: {Q.shape}  →  6 tokens, each with a 4-dim query vector")
print(f"K shape: {K.shape}  →  6 tokens, each with a 4-dim key vector")
print(f"V shape: {V.shape}  →  6 tokens, each with a 4-dim value vector")
print()
print("Example — token 'cat' (index 1):")
print(f"  Query: {Q[1]}  (what 'cat' is looking for)")
print(f"  Key:   {K[1]}  (what 'cat' advertises)")
print(f"  Value: {V[1]}  (what 'cat' actually contains)")
print()

# ============================================================
# STEP 4: Compute attention scores — Q @ K^T
# ============================================================
# Dot product between every query and every key
# High score = high similarity = "these tokens are relevant to each other"
scores = Q @ K.T  # (6, 4) @ (4, 6) = (6, 6)

print("=" * 60)
print("STEP 4: Raw attention scores (Q @ K^T)")
print(f"Shape: {scores.shape}  →  6×6 matrix")
print(f"Cell (i,j) = how much token i should attend to token j")
print()
print("Raw scores matrix:")
for i, word in enumerate(sentence):
    print(f"  {word:>4}: {scores[i]}")
print()

# ============================================================
# STEP 5: Scale by √d_k
# ============================================================
# Without scaling: large d_k → large dot products → softmax saturates → gradients die
scale = np.sqrt(d_k)
scaled_scores = scores / scale

print("=" * 60)
print(f"STEP 5: Scale by √d_k = √{d_k} = {scale:.2f}")
print(f"WHY: prevents softmax saturation (same reason you scale logits in classification)")
print()
print("Scaled scores matrix:")
for i, word in enumerate(sentence):
    print(f"  {word:>4}: {scaled_scores[i]}")
print()


# ============================================================
# STEP 6: Softmax — convert to probabilities (row-wise)
# ============================================================
def softmax(x):
    """Numerically stable softmax along last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


attention_weights = softmax(scaled_scores)  # (6, 6), each row sums to 1

print("=" * 60)
print("STEP 6: Attention weights (softmax of scaled scores)")
print(f"Each row sums to 1.0 — it's a probability distribution")
print(f"Row i = 'for token i, here's the % of attention given to every token'")
print()
print("Attention weight matrix:")
print(f"{'':>6}", end="")
for w in sentence:
    print(f"{w:>8}", end="")
print()
for i, word in enumerate(sentence):
    print(f"  {word:>4}:", end="")
    for j in range(seq_len):
        print(f"{attention_weights[i][j]:8.3f}", end="")
    print(f"  (sum={attention_weights[i].sum():.3f})")
print()

# ============================================================
# STEP 7: Weighted sum of values — attention_weights @ V
# ============================================================
# Each token's output = weighted average of ALL value vectors
# Weights = how much attention that token paid to each other token
output = attention_weights @ V  # (6, 6) @ (6, 4) = (6, 4)

print("=" * 60)
print("STEP 7: Output = attention_weights @ V")
print(f"Shape: {output.shape}  →  6 tokens, each now a {d_v}-dim context-aware vector")
print()
print("Before attention (raw value of 'cat'):", V[1])
print("After attention (context-aware 'cat'): ", output[1])
print()
print("MEANING: 'cat' started as just its own meaning.")
print("After attention, it's a weighted blend of ALL words in the sentence.")
print("It now 'knows about' the words around it.")
print()


# ============================================================
# STEP 8: Full function — reusable
# ============================================================
def self_attention(X, W_Q, W_K, W_V):
    """
    Single-head scaled dot-product self-attention.

    X:    (seq_len, d_model) — input embeddings
    W_Q:  (d_model, d_k)    — query projection
    W_K:  (d_model, d_k)    — key projection
    W_V:  (d_model, d_v)    — value projection

    Returns:
        output:  (seq_len, d_v) — context-aware representations
        weights: (seq_len, seq_len) — attention weight matrix
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V

    return output, weights


print("=" * 60)
print("STEP 8: Complete self_attention() function — 7 lines of math")
print()

# Verify it matches
output2, weights2 = self_attention(X, W_Q, W_K, W_V)
assert np.allclose(output, output2), "Mismatch!"
assert np.allclose(attention_weights, weights2), "Mismatch!"
print("✓ Function output matches step-by-step computation")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SELF-ATTENTION PIPELINE SUMMARY")
print("=" * 60)
print("""
Input X (6 tokens × 8-dim)
    ↓
Three projections:  Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
    ↓
Similarity scores:  scores = Q @ K^T          → (6×6) raw scores
    ↓
Scale:              scores / √d_k             → prevent softmax saturation
    ↓
Normalize:          softmax(scores)           → (6×6) attention weights (rows sum to 1)
    ↓
Weighted average:   output = weights @ V      → (6 tokens × 4-dim) context-aware vectors

Total: 2 matrix multiplies for the core attention, 3 projections on input.
Every token now contains information from EVERY other token — in ONE step, not sequentially.
""")