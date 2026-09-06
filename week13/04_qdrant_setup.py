"""
04_qdrant_setup.py — Week 13 Day 4
Load 10K Stack Overflow posts, chunk, embed, upsert into Qdrant, query with metadata filters.

Prerequisites:
    docker compose up -d          # Qdrant running on localhost:6333
    pip install qdrant-client datasets sentence-transformers

"""

import time
import uuid
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


# ── Step 1: Load Stack Overflow posts & deduplicate ───────────────
print("Step 1: Loading Stack Overflow dataset...")
# Load more rows than we need — many questions repeat with different answers.
# We'll deduplicate by question_id and keep the best answer.
ds = load_dataset(
    "koutch/stackoverflow_python",
    split="train[:30000]",  # load 30K rows to get ~10K unique questions
)
print(f"  Loaded {len(ds)} raw rows")
print(f"  Columns: {ds.column_names}")

# Deduplicate: group by question_id, keep the answer with highest score
best_by_question: dict[int, dict] = {}
for row in ds:
    qid = row["question_id"]
    if qid not in best_by_question or row["answer_score"] > best_by_question[qid]["answer_score"]:
        best_by_question[qid] = row

# Take first 10K unique questions
unique_posts = list(best_by_question.values())[:10000]
print(f"  Unique questions: {len(best_by_question)}")
print(f"  Using first: {len(unique_posts)}")
print(f"  Sample title: {unique_posts[0].get('title', 'N/A')}")


# ── Step 2: Chunk long posts ──────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    - chunk_size: max words per chunk
    - overlap: words shared between consecutive chunks
    Returns list of chunk strings (skips chunks < 20 words).
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]  # short text → one chunk, no splitting needed

    chunks = []
    start = 0
    step = chunk_size - overlap  # how far to advance each time

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(words[start:end]) >= 20:  # skip tiny leftover fragments
            chunks.append(chunk)
        start += step

    return chunks


print("\nStep 2: Chunking posts...")
chunks = []  # list of dicts: {"text": ..., "title": ..., "tags": ..., "question_id": ...}

for row in unique_posts:
    title = row.get("title", "")
    question = row.get("question_body", "")     # ← FIXED: correct column name
    answer = row.get("answer_body", "")          # ← FIXED: include best answer
    tags = row.get("tags", "")
    qid = row.get("question_id", 0)

    # Combine title + question + best answer for rich context
    full_text = f"{title}\n{question}\n{answer}".strip()

    if not full_text:
        continue

    text_chunks = chunk_text(full_text)
    for j, chunk in enumerate(text_chunks):
        chunks.append({
            "text": chunk,
            "title": title,
            "tags": tags,
            "question_id": qid,
            "chunk_index": j,
        })

print(f"  Total chunks: {len(chunks)}")
print(f"  Posts with multiple chunks: {sum(1 for c in chunks if c['chunk_index'] > 0)}")
print(f"  Sample chunk (first 100 chars): {chunks[0]['text'][:100]}...")


# ── Step 3: Embed all chunks ──────────────────────────────────────
print("\nStep 3: Embedding chunks (this may take a few minutes)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Extract just the text for embedding
texts = [c["text"] for c in chunks]

# Batch encoding — sentence-transformers handles batching internally
# normalize_embeddings=True so dot product = cosine similarity
t0 = time.time()
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=256,  # larger batches = faster on CPU
)
embed_time = time.time() - t0
print(f"  Embedded {len(embeddings)} chunks in {embed_time:.1f}s")
print(f"  Embedding shape: {embeddings.shape}")  # (N, 384)
print(f"  Speed: {len(embeddings) / embed_time:.0f} chunks/sec")


# ── Step 4: Upsert into Qdrant ────────────────────────────────────
print("\nStep 4: Connecting to Qdrant and upserting...")
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "stackoverflow_python"

# Delete collection if it already exists (clean slate for re-runs)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing collection '{COLLECTION_NAME}'")

# Create collection — this is where you define the vector config
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,              # must match embedding model output dim
        distance=Distance.COSINE,  # Qdrant normalizes internally for cosine
    ),
)
print(f"  Created collection '{COLLECTION_NAME}' (384 dims, cosine distance)")

# Upsert in batches of 500 (avoid OOM on large payloads)
BATCH_SIZE = 500
t0 = time.time()

for batch_start in range(0, len(chunks), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(chunks))
    points = []

    for idx in range(batch_start, batch_end):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # unique ID per chunk
                vector=embeddings[idx].tolist(),
                payload={  # metadata — searchable and filterable
                    "text": chunks[idx]["text"],
                    "title": chunks[idx]["title"],
                    "tags": chunks[idx]["tags"],
                    "question_id": chunks[idx]["question_id"],
                    "chunk_index": chunks[idx]["chunk_index"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)

upsert_time = time.time() - t0
print(f"  Upserted {len(chunks)} points in {upsert_time:.1f}s")
print(f"  Speed: {len(chunks) / upsert_time:.0f} points/sec")

# Verify count
info = client.get_collection(COLLECTION_NAME)
print(f"  Collection point count: {info.points_count}")


# ── Step 5: Query — basic semantic search ─────────────────────────
print("\n" + "=" * 60)
print("Step 5: Querying Qdrant")
print("=" * 60)


def search(query: str, k: int = 3, tag_filter: str | None = None):
    """Search Qdrant with optional metadata filter."""
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Build filter if tag specified
    search_filter = None
    if tag_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="tags",
                    match=MatchValue(value=tag_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=k,
        query_filter=search_filter,
    )

    return results.points


# --- Query 1: Basic semantic search ---
print("\n🔍 Query 1: 'how to read a CSV file in pandas'")
results = search("how to read a CSV file in pandas")
for i, r in enumerate(results):
    print(f"\n  [{i+1}] Score: {r.score:.3f}")
    print(f"      Title: {r.payload['title'][:80]}")
    print(f"      Tags:  {r.payload['tags']}")
    print(f"      Text:  {r.payload['text'][:120]}...")

# --- Query 2: Rare/technical term ---
print("\n\n🔍 Query 2: 'asyncio gather RuntimeError'")
results = search("asyncio gather RuntimeError")
for i, r in enumerate(results):
    print(f"\n  [{i+1}] Score: {r.score:.3f}")
    print(f"      Title: {r.payload['title'][:80]}")
    print(f"      Tags:  {r.payload['tags']}")

# --- Query 3: With metadata filter ---
print("\n\n🔍 Query 3: 'sort a list' (filtered to posts tagged 'python-3.x')")
results = search("sort a list", tag_filter="python-3.x")
for i, r in enumerate(results):
    print(f"\n  [{i+1}] Score: {r.score:.3f}")
    print(f"      Title: {r.payload['title'][:80]}")
    print(f"      Tags:  {r.payload['tags']}")

if not results:
    print("  (No results — tag 'python-3.x' may not exist in this subset. Try without filter.)")


# ── Step 6: Compare brute-force vs Qdrant speed ──────────────────
print("\n" + "=" * 60)
print("Step 6: Speed comparison — brute force vs Qdrant")
print("=" * 60)

import numpy as np

query = "how to handle exceptions in Python"
q_emb = model.encode([query], normalize_embeddings=True)[0]

# Brute force: dot product against all embeddings
t0 = time.time()
for _ in range(100):  # 100 queries to get stable timing
    scores = embeddings @ q_emb
    top_k = np.argsort(-scores)[:3]
brute_time = (time.time() - t0) / 100

# Qdrant: HNSW search
t0 = time.time()
for _ in range(100):
    _ = search(query, k=3)
qdrant_time = (time.time() - t0) / 100

print(f"\n  Brute force (NumPy):  {brute_time*1000:.2f} ms/query")
print(f"  Qdrant (HNSW):        {qdrant_time*1000:.2f} ms/query")
print(f"\n  Note: At {len(chunks)} chunks, brute force is still fast.")
print(f"  The gap widens dramatically at 1M+ documents.")
print(f"  Qdrant's HNSW stays ~constant time regardless of corpus size.")