

My first week on the AI/ML engineering path — focused on reading real ML source code and mastering NumPy vectorization.

## What's in this repo

### 1. scikit-learn Source Code Reading: How `fit()` Works
**File:** [`fit_summary.md`](fit_summary.md)

I opened scikit-learn's `LinearRegression` source code and traced how `fit()` and `predict()` actually work under the hood. Key discovery: `fit()` does all the heavy lifting and stores results (`coef_`, `rank_`, `singular_`), while `predict()` is just a thin wrapper that reuses what was already learned. This **fit-store-predict** pattern is the backbone of sklearn's design.

### 2. NumPy Vectorization Benchmarks
**File:** [`vectorization_benchmarks.py`](vectorization_benchmarks.py)

Rewrote 5 common Python loop patterns as vectorized NumPy operations and benchmarked each:

| Loop | Pattern | Speedup |
|------|---------|---------|
| 1 | Element-wise computation | ~14x |
| 2 | Conditional replacement | ~17x |
| 3 | Outer product (2D broadcasting) | ~50x |
| 4 | Pairwise Euclidean distances (3D broadcasting) | ~50x |
| 5 | Row-wise statistics | ~500x |

**Biggest learning:** Loop 4 required reshaping a `(N, 2)` array into 3D — `(N, 1, 2)` and `(1, N, 2)` — to broadcast point pairs while keeping coordinates together. The pattern: identify what gets paired (split it across dimensions) and what stays together (keep it at the end).

## Key Learnings

- **`__getattr__`** triggers only as a last resort — normal attribute lookup happens first
- **Trailing underscore convention** in sklearn (`coef_`, `rank_`) signals attributes created by `fit()` that don't exist before training
- **Broadcasting** stretch direction is controlled by reshape — this applies to both 2D and 3D cases
- **Axis collapse** removes the dimension entirely; use `keepdims=True` to preserve it as size 1
- **Vectorization** isn't just faster — it forces you to think in terms of array operations, which is how NumPy and ML libraries are designed to work

## Tools Used

- Python 3.x
- NumPy
- scikit-learn (source code reading)
- timeit (benchmarking)
