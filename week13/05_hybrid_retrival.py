"""
Week 13 — Day 5: Reciprocal Rank Fusion (Hybrid Retrieval)
==========================================================
Combines BM25Retriever (Day 3) + TinyRetriever (Day 2) into a single
HybridRetriever using Reciprocal Rank Fusion (RRF).

Key idea: throw away raw scores (incomparable scales), use only RANKS.
RRF_score(doc) = Σ 1/(k + rank_i(doc))   where k=60 (default)

Run: pip install sentence-transformers numpy && python 05_hybrid_retrieval.py
"""

import re
import string
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. BM25Retriever (from Day 3 — 03_bm25_retriever.py)
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "if", "when", "where", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "i", "me", "my", "myself",
    "we", "our", "ours", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "they", "them", "their",
    "about", "up", "down", "here", "there", "all", "any", "every",
}

SUFFIXES = ["ing", "tion", "ed", "ly", "er", "est", "ness", "ment", "able", "ous", "ive", "al"]


def tokenize(text: str) -> list[str]:
    """4-step text normalization: lowercase → strip punctuation → remove stop words → suffix stemming."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    stemmed = []
    for t in tokens:
        for suffix in SUFFIXES:
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                t = t[: -len(suffix)]
                break
        stemmed.append(t)
    return stemmed


class BM25Retriever:
    """BM25 sparse retriever with .index() / .query() interface."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_texts = None
        self.tokenized_docs = None
        self.doc_freq = None
        self.avgdl = 0.0
        self.N = 0

    def index(self, texts: list[str]):
        self.corpus_texts = texts
        self.N = len(texts)
        self.tokenized_docs = [tokenize(t) for t in texts]
        self.avgdl = np.mean([len(d) for d in self.tokenized_docs])

        self.doc_freq = Counter()
        for doc_tokens in self.tokenized_docs:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                self.doc_freq[term] += 1

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = len(doc_tokens)
        tf_counter = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_counter.get(qt, 0)
            if tf == 0:
                continue
            idf = self._idf(qt)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def query(self, q: str, k: int = 5) -> list[tuple[str, float]]:
        query_tokens = tokenize(q)
        scores = np.array([self._score_doc(query_tokens, i) for i in range(self.N)])
        top_k_idx = np.argsort(-scores)[:k]
        return [(self.corpus_texts[i], scores[i]) for i in top_k_idx]


# ============================================================
# 2. TinyRetriever (from Day 2 — 02_tiny_retriever.py)
# ============================================================

class TinyRetriever:
    """Dense retriever using cosine similarity (normalized dot product)."""

    def __init__(self, model):
        self.model = model
        self.corpus_embeddings = None
        self.corpus_texts = None

    def index(self, texts: list[str]):
        self.corpus_texts = texts
        self.corpus_embeddings = self.model.encode(texts, normalize_embeddings=True)

    def query(self, q: str, k: int = 5) -> list[tuple[str, float]]:
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        scores = self.corpus_embeddings @ q_emb
        top_k_idx = np.argsort(-scores)[:k]
        return [(self.corpus_texts[i], float(scores[i])) for i in top_k_idx]


# ============================================================
# 3. HybridRetriever — RRF Fusion
# ============================================================

def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each ranked_list is [(doc_text, original_score), ...] in ranked order.
    Original scores are IGNORED — only rank position matters.

    RRF_score(doc) = Σ 1/(k + rank_i(doc))

    Args:
        ranked_lists: list of ranked result lists from different retrievers
        k: fusion constant (default 60). Higher k = less weight on top ranks.

    Returns:
        Fused list of (doc_text, rrf_score), sorted descending by RRF score.
    """
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank, (doc_text, _original_score) in enumerate(ranked_list, start=1):
            if doc_text not in rrf_scores:
                rrf_scores[doc_text] = 0.0
            rrf_scores[doc_text] += 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


class HybridRetriever:
    """
    Combines BM25 (sparse) + Dense (semantic) retrieval via Reciprocal Rank Fusion.

    Same .index() / .query() interface as BM25Retriever and TinyRetriever.
    Delegates indexing and querying to both sub-retrievers, then fuses results.
    """

    def __init__(self, model, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.bm25 = BM25Retriever(k1=k1, b=b)
        self.dense = TinyRetriever(model)
        self.rrf_k = rrf_k

    def index(self, texts: list[str]):
        """Index the corpus in both sub-retrievers."""
        self.bm25.index(texts)
        self.dense.index(texts)
        print(f"[HybridRetriever] Indexed {len(texts)} documents in both BM25 and Dense retrievers")

    def query(self, q: str, k: int = 5, sub_k: int = 50) -> list[tuple[str, float]]:
        """
        Query both retrievers, fuse with RRF, return top-k.

        Args:
            q: query string
            k: number of final results to return
            sub_k: number of results to fetch from each sub-retriever before fusion
        """
        bm25_results = self.bm25.query(q, k=sub_k)
        dense_results = self.dense.query(q, k=sub_k)

        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)

        return fused[:k]


# ============================================================
# 4. Corpus — Same 50 docs from Days 2 and 3
# ============================================================

CORPUS = [
    # --- Cooking (17 docs) ---
    "Dice the onions finely before adding them to the hot pan",
    "Use a meat thermometer to check if the steak is medium rare",
    "Marinate the chicken overnight for the best flavor",
    "Boil water with a pinch of salt before adding the pasta",
    "Fresh basil added at the end gives pasta a vibrant taste",
    "Toast the spices in a dry pan to release their aroma",
    "Knead the dough for ten minutes until it becomes smooth and elastic",
    "A sharp knife makes chopping vegetables much safer and faster",
    "Simmer the tomato sauce on low heat for at least thirty minutes",
    "Let the meat rest for five minutes before slicing it",
    "Roast the peppers until the skin is charred and blistered",
    "Add a splash of vinegar to brighten up any soup or stew",
    "Blanch the broccoli in boiling water then immediately ice bath it",
    "Season the cast iron skillet with oil after every use",
    "Use room temperature eggs for fluffier cakes and better emulsification",
    "Deglaze the pan with wine to create a rich sauce from the fond",
    "Fold the egg whites gently to keep the batter light and airy",

    # --- Coding (17 docs) ---
    "Use try except blocks to handle exceptions gracefully",
    "Write unit tests before refactoring legacy code",
    "Version pinning in requirements.txt prevents dependency conflicts",
    "Use list comprehensions for concise and readable Python loops",
    "A linter catches style issues before code review begins",
    "Git rebase keeps commit history clean compared to merge commits",
    "Logging with structured formats makes debugging production issues easier",
    "Type hints in Python improve IDE autocompletion and catch bugs early",
    "Use virtual environments to isolate project dependencies",
    "Profile your code before optimizing to find actual bottlenecks",
    "Database indexes speed up queries but slow down write operations",
    "Cache frequently accessed data to reduce database round trips",
    "Use environment variables for secrets instead of hardcoding them",
    "Async functions in Python handle IO bound tasks more efficiently",
    "Write docstrings for every public function to help future developers",
    "Continuous integration runs tests automatically on every commit",
    "Use a debugger instead of print statements for complex bugs",

    # --- Weather (16 docs) ---
    "Heavy rainfall is expected across the southern coast this weekend",
    "Morning fog will clear by noon in most valley regions",
    "The temperature dropped below freezing overnight in the mountains",
    "Winds are gusting up to sixty miles per hour near the ridgeline",
    "Humidity levels above eighty percent make the heat feel unbearable",
    "A cold front moving in from the north will bring snow by Tuesday",
    "Clear skies and mild temperatures are forecast for the rest of the week",
    "The UV index is extremely high today so wear sunscreen outdoors",
    "Thunderstorms are likely this afternoon with a chance of hail",
    "Barometric pressure is dropping which often signals incoming storms",
    "Pollen counts are very high this spring causing widespread allergies",
    "The drought has lasted three months with no rain in the forecast",
    "Dense fog advisory issued for all major highways until ten AM",
    "Tropical storm warnings have been issued for the entire gulf coast",
    "Overnight lows will dip into the mid twenties across the plains",
    "El Nino patterns are expected to bring a wetter than average winter",
]


# ============================================================
# 5. Test Queries — designed to show where hybrid wins
# ============================================================

TEST_QUERIES = [
    # --- Semantic queries (Dense should do well) ---
    {
        "query": "how to cook pasta",
        "type": "semantic",
        "note": "Dense matches 'cook pasta' semantically; BM25 needs exact word overlap"
    },
    {
        "query": "best practices for error handling in code",
        "type": "semantic",
        "note": "Dense matches meaning; BM25 hits 'handle exceptions' only if keywords overlap"
    },
    {
        "query": "will it rain tomorrow",
        "type": "semantic",
        "note": "Dense matches weather/rain concepts; BM25 needs 'rain' keyword"
    },

    # --- Keyword queries (BM25 should do well) ---
    {
        "query": "requirements.txt dependency",
        "type": "keyword",
        "note": "Exact term match — BM25 finds 'requirements.txt' directly"
    },
    {
        "query": "cast iron skillet seasoning",
        "type": "keyword",
        "note": "Specific product term — BM25 matches 'cast iron skillet' exactly"
    },
    {
        "query": "El Nino winter forecast",
        "type": "keyword",
        "note": "Rare term 'El Nino' — BM25 scores high via IDF; Dense may miss"
    },

    # --- Hybrid queries (both contribute useful results) ---
    {
        "query": "keep code dependencies isolated",
        "type": "hybrid",
        "note": "Dense matches 'isolated' ≈ 'virtual environments'; BM25 matches 'dependencies'"
    },
    {
        "query": "preparing meat properly",
        "type": "hybrid",
        "note": "Dense matches cooking concept; BM25 matches 'meat' across multiple docs"
    },
    {
        "query": "storm warning coastal areas",
        "type": "hybrid",
        "note": "Dense matches weather/storm semantics; BM25 matches 'storm' and 'coast'"
    },
    {
        "query": "debugging production issues",
        "type": "hybrid",
        "note": "Dense matches the concept; BM25 matches exact phrase in logging doc"
    },
]


# ============================================================
# 6. Main — Run all three retrievers and compare
# ============================================================

def format_result(rank: int, text: str, score: float, max_len: int = 70) -> str:
    """Format a single result for display."""
    truncated = text[:max_len] + "..." if len(text) > max_len else text
    return f"  #{rank}: [{score:.4f}] {truncated}"


def run_comparison():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize all three retrievers
    bm25 = BM25Retriever()
    dense = TinyRetriever(model)
    hybrid = HybridRetriever(model)

    # Index the corpus
    print(f"\nIndexing {len(CORPUS)} documents...")
    bm25.index(CORPUS)
    dense.index(CORPUS)
    hybrid.index(CORPUS)

    print(f"\n{'='*80}")
    print("RETRIEVER COMPARISON: BM25 vs Dense vs Hybrid (RRF k=60)")
    print(f"{'='*80}")

    for i, tq in enumerate(TEST_QUERIES, 1):
        q = tq["query"]
        qtype = tq["type"]
        note = tq["note"]

        bm25_results = bm25.query(q, k=3)
        dense_results = dense.query(q, k=3)
        hybrid_results = hybrid.query(q, k=3)

        print(f"\n{'─'*80}")
        print(f"Query {i}: \"{q}\"")
        print(f"Type: {qtype} | {note}")
        print(f"{'─'*80}")

        print("\n  BM25 Top-3:")
        for rank, (text, score) in enumerate(bm25_results, 1):
            print(format_result(rank, text, score))

        print("\n  Dense Top-3:")
        for rank, (text, score) in enumerate(dense_results, 1):
            print(format_result(rank, text, score))

        print("\n  Hybrid (RRF) Top-3:")
        for rank, (text, score) in enumerate(hybrid_results, 1):
            print(format_result(rank, text, score))

        # Show overlap analysis
        bm25_top3_texts = {text for text, _ in bm25_results}
        dense_top3_texts = {text for text, _ in dense_results}
        overlap = bm25_top3_texts & dense_top3_texts
        only_bm25 = bm25_top3_texts - dense_top3_texts
        only_dense = dense_top3_texts - bm25_top3_texts

        print(f"\n  Overlap: {len(overlap)} docs in both | {len(only_bm25)} BM25-only | {len(only_dense)} Dense-only")

    # --- Summary statistics ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Corpus size:    {len(CORPUS)} documents")
    print(f"Test queries:   {len(TEST_QUERIES)}")
    print(f"RRF k:          60 (default)")
    print(f"Sub-retriever k: 50 (fetch top-50 from each before fusion)")
    print(f"\nKey insight: Hybrid RRF surfaces docs that BOTH retrievers agree on,")
    print(f"while still including strong unique finds from either side.")
    print(f"No score normalization needed — only ranks matter.")


if __name__ == "__main__":
    run_comparison()