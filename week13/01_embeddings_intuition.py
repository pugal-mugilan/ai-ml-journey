"""
Week 13 - Day 1: Embeddings Intuition
Run this locally: pip install sentence-transformers numpy
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model (22MB, 384 dimensions, CPU-fast)
print("Loading all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 10 hand-crafted sentences across 3 topics + 1 outlier
sentences = [
    # Topic 1: Cats / pets
    "The cat sat on the mat",
    "A feline rested on the rug",
    "My kitten is sleeping on the sofa",

    # Topic 2: Finance
    "Interest rates rose sharply today",
    "The stock market crashed this morning",
    "Bond yields increased significantly",

    # Topic 3: Cooking
    "Chop the onions and fry them in oil",
    "Dice the vegetables and sauté in butter",
    "Slice the garlic and cook in olive oil",

    # Outlier
    "The weather in Tokyo is sunny",
]

# Generate embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)

# --- Basic info ---
print(f"\nShape of all embeddings: {embeddings.shape}")
print(f"Each sentence becomes a vector of {embeddings.shape[1]} numbers\n")

print(f"Sentence: '{sentences[0]}'")
print(f"First 10 values: {embeddings[0][:10].round(4)}")
print(f"Sum of squares (normalized): {np.sum(embeddings[0]**2):.4f}")

# --- Cosine similarity ---
# Since vectors are normalized, dot product = cosine similarity
similarity_matrix = embeddings @ embeddings.T

print("\n--- COSINE SIMILARITY (selected pairs) ---\n")

pairs = [
    (0, 1, "cat sat on mat  vs  feline rested on rug  (SAME topic)"),
    (0, 2, "cat sat on mat  vs  kitten sleeping        (SAME topic)"),
    (0, 3, "cat sat on mat  vs  interest rates rose     (DIFF topic)"),
    (0, 9, "cat sat on mat  vs  weather in Tokyo        (DIFF topic)"),
    (3, 5, "interest rates  vs  bond yields             (SAME topic)"),
    (6, 8, "chop onions     vs  slice garlic            (SAME topic)"),
    (6, 4, "chop onions     vs  stock market crashed    (DIFF topic)"),
]

for i, j, desc in pairs:
    score = similarity_matrix[i][j]
    print(f"  {score:.4f}  |  {desc}")

# --- Failure mode: opposite meaning, high similarity ---
print("\n--- FAILURE MODE: Opposite meaning ---\n")
opposite_sentences = [
    "I love this movie",
    "I hate this movie",
]
opp_emb = model.encode(opposite_sentences, normalize_embeddings=True)
opp_score = opp_emb[0] @ opp_emb[1]
print(f"  {opp_score:.4f}  |  'I love this movie' vs 'I hate this movie'")
print(f"  ^ High similarity despite OPPOSITE meaning!")
print(f"    Why? They share most vocabulary. Only one word differs.")