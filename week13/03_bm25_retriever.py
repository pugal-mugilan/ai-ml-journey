"""
Week 13 · Day 3 — BM25 Sparse Retriever (from scratch)
=======================================================
Keyword-based retrieval using the BM25 ranking function.
Same 50-doc corpus and interface as Day 2's TinyRetriever
so both can be fused on Day 5.

Includes text normalization:
  1. Punctuation stripping
  2. Lowercasing
  3. Stop word removal
  4. Stemming (basic suffix stripping)
"""

import math
import re
import numpy as np
from collections import Counter


# ── Stop words (high-frequency, low-information words) ───────────
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
    "or", "for", "with", "at", "by", "from", "as", "into", "that",
    "this", "its", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "not", "no", "but",
    "if", "so", "than", "too", "very", "just", "about", "up",
    "out", "how", "what", "which", "who", "when", "where", "why",
    "all", "each", "every", "your", "you", "we", "they", "i",
}


def basic_stem(word):
    """
    Minimal suffix-stripping stemmer.
    Not as accurate as Porter/Snowball, but zero dependencies.
    Production systems use nltk.stem.PorterStemmer or similar.
    """
    if len(word) <= 3:
        return word
    # Order matters: check longest suffixes first
    if word.endswith("tion"):
        return word[:-4]
    if word.endswith("sion"):
        return word[:-4]
    if word.endswith("ness"):
        return word[:-4]
    if word.endswith("ment"):
        return word[:-4]
    if word.endswith("ing"):
        # "cooking" → "cook", but "ring" stays "ring"
        stem = word[:-3]
        return stem if len(stem) >= 3 else word
    if word.endswith("ies"):
        return word[:-3] + "y"   # "recipes" → "recipy" (rough but consistent)
    if word.endswith("ed"):
        stem = word[:-2]
        return stem if len(stem) >= 3 else word
    if word.endswith("ly"):
        stem = word[:-2]
        return stem if len(stem) >= 3 else word
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text):
    """
    Full normalization pipeline:
      raw text → lowercase → strip punctuation → split → remove stops → stem
    """
    text = text.lower()                              # Step 1: lowercase
    text = re.sub(r"[^\w\s-]", "", text)             # Step 2: strip punctuation (keep hyphens)
    words = text.split()                             # Split on whitespace
    words = [w for w in words if w not in STOP_WORDS]  # Step 3: remove stop words
    words = [basic_stem(w) for w in words]           # Step 4: stem
    return words


class BM25Retriever:
    """Sparse retriever using BM25 scoring."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def index(self, texts):
        """Pre-compute document frequencies and average doc length."""
        self.texts = texts
        self.N = len(texts)

        # Tokenize with full normalization pipeline
        self.tokenized = [tokenize(doc) for doc in texts]

        # Average document length (post-normalization)
        self.avgdl = sum(len(doc) for doc in self.tokenized) / self.N

        # Document frequency: how many docs contain each word
        self.doc_freq = Counter()
        for doc in self.tokenized:
            for word in set(doc):
                self.doc_freq[word] += 1

    def query(self, q, k=3):
        """Score all docs against query, return top-k (text, score) pairs."""
        q_words = tokenize(q)   # Same normalization on query!
        scores = []

        for i, doc in enumerate(self.tokenized):
            print(doc)
            score = 0.0
            doc_len = len(doc)
            tf_counter = Counter(doc)

            for word in q_words:
                if word not in self.doc_freq:
                    continue

                f = tf_counter.get(word, 0)
                df = self.doc_freq[word]

                # IDF: rare words score higher
                idf = math.log(self.N / df)

                # TF with saturation + length normalization
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                tf_score = numerator / denominator

                score += idf * tf_score

            scores.append(score)

        top_k_idx = np.argsort(-np.array(scores))[:k]
        return [(self.texts[i], round(scores[i], 4)) for i in top_k_idx]


# ── 50-doc corpus (same as Day 2) ───────────────────────────────
CORPUS = [
    # Cooking (docs 0-16)
    "How to make a classic Italian pasta with tomato sauce",
    "The best way to season a cast iron skillet for cooking",
    "Tips for baking sourdough bread at home from scratch",
    "A simple recipe for homemade chicken tikka masala",
    "How to properly dice an onion without crying",
    "Guide to making perfect fluffy scrambled eggs every time",
    "The secret to a crispy golden fried chicken coating",
    "Easy vegetarian stir fry with tofu and mixed vegetables",
    "How to caramelize onions slowly for maximum sweetness",
    "Best practices for grilling steak to medium rare perfection",
    "Traditional Japanese ramen broth from scratch takes hours",
    "Quick weeknight dinner ideas using only pantry staples",
    "How to temper chocolate for professional looking desserts",
    "The art of making fresh handmade pasta dough at home",
    "Fermented hot sauce recipe with habanero and garlic",
    "Use a meat thermometer to check if the steak is medium rare",
    "Slow cooker pulled pork with homemade barbecue sauce",

    # Coding (docs 17-33)
    "Introduction to Python list comprehensions with examples",
    "How to set up a virtual environment in Python using venv",
    "Understanding Git rebase versus merge for clean history",
    "REST API design best practices for scalable web services",
    "Docker container basics and writing your first Dockerfile",
    "Guide to writing unit tests in Python with pytest framework",
    "How to handle exceptions and error logging in Python",
    "Setting up CI CD pipelines with GitHub Actions step by step",
    "Database indexing strategies for faster SQL query performance",
    "Introduction to async await patterns in modern JavaScript",
    "Version pinning in requirements.txt prevents dependency conflicts",
    "How to debug memory leaks in Node.js applications effectively",
    "Kubernetes pod scheduling and resource limit configuration",
    "Writing clean maintainable code with SOLID design principles",
    "SSH key authentication setup for secure remote server access",
    "Error code ERR-4829: connection timeout on port 5432",
    "Microservices communication patterns using message queues",

    # Weather (docs 34-49)
    "Severe thunderstorm warning issued for the northeast region",
    "How to prepare an emergency kit for hurricane season",
    "Understanding the difference between weather and climate change",
    "Heavy rainfall is expected across the southern coast tomorrow",
    "Tips for driving safely in foggy conditions on highways",
    "The science behind how tornadoes form in the great plains",
    "UV index explained and how to protect your skin from sunburn",
    "Winter storm preparedness checklist for homeowners",
    "How barometric pressure changes affect daily weather patterns",
    "Record breaking heat wave continues across the southwest region",
    "Monsoon season forecast predicts above average rainfall this year",
    "How to read weather radar maps for accurate local predictions",
    "The role of ocean currents in determining regional climate",
    "Frost advisory issued for overnight temperatures below freezing",
    "Wind chill factor calculation and its effect on human comfort",
    "El Nino and La Nina weather patterns affect global temperatures",
]


# ── Run tests ────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = BM25Retriever()
    retriever.index(CORPUS)

    print(f"Corpus: {retriever.N} docs, avgdl = {retriever.avgdl:.1f} words (after normalization)")
    print(f"Vocab size: {len(retriever.doc_freq)} unique words")
    print("=" * 65)

    # --- Test 1: BM25's strength — exact keyword match ---
    queries_bm25_wins = [
        "ERR-4829",
        "pytest framework",
        "habanero garlic",
        "El Nino La Nina",
    ]

    print("\n🔑 QUERIES WHERE BM25 SHINES (exact keywords / rare terms)")
    print("-" * 65)
    for q in queries_bm25_wins:
        print(f"\nQuery: '{q}'")
        results = retriever.query(q, k=3)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  #{rank} [{score:.4f}] {text[:70]}")

    # --- Test 2: BM25's weakness — semantic matching ---
    queries_bm25_weak = [
        "how to cook pasta",
        "tips for grilling meat",
        "bad weather coming soon",
    ]

    print("\n\n🧠 QUERIES WHERE DENSE RETRIEVAL WOULD WIN (meaning, not keywords)")
    print("-" * 65)
    for q in queries_bm25_weak:
        print(f"\nQuery: '{q}'")
        results = retriever.query(q, k=3)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  #{rank} [{score:.4f}] {text[:70]}")

    # --- Test 3: The antonym test ---
    print("\n\n🔄 ANTONYM TEST")
    print("-" * 65)
    q1_results = retriever.query("I love this product", k=1)
    q2_results = retriever.query("I hate this product", k=1)
    print(f"'I love this product' → top score: {q1_results[0][1]}")
    print(f"'I hate this product' → top score: {q2_results[0][1]}")
    print("Neither query word exists in corpus — BM25 scores are 0.")
    print("(Dense retrieval at least returned similar-meaning docs)")

    # --- Test 4: Multi-word query showing IDF in action ---
    print("\n\n📊 IDF IN ACTION — which query words matter most?")
    print("-" * 65)
    q = "python error connection"
    q_tokens = tokenize(q)
    print(f"Query: '{q}' → tokens after normalization: {q_tokens}")
    for token in q_tokens:
        df = retriever.doc_freq.get(token, 0)
        idf = math.log(retriever.N / df) if df > 0 else 0
        print(f"  doc_freq('{token}') = {df} docs → IDF = {idf:.3f}")
    results = retriever.query(q, k=3)
    for rank, (text, score) in enumerate(results, 1):
        print(f"  #{rank} [{score:.4f}] {text[:70]}")

    # --- Test 5: Stemming in action ---
    print("\n\n🌱 STEMMING IN ACTION")
    print("-" * 65)
    q = "cooking recipes"
    q_tokens = tokenize(q)
    print(f"Query: '{q}' → tokens: {q_tokens}")
    print("  'cooking' stems to 'cook', matches docs with 'cooking'/'cooker'")
    print("  'recipes' stems to 'recipe', matches docs with 'recipes'/'recipe'")
    results = retriever.query(q, k=3)
    for rank, (text, score) in enumerate(results, 1):
        print(f"  #{rank} [{score:.4f}] {text[:70]}")