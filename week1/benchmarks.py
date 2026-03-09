"""
 NumPy Vectorization Benchmarks
========================================
5 common loop patterns rewritten as vectorized NumPy operations.
Each benchmark compares the loop version against the vectorized version.

Key takeaway: Vectorized NumPy avoids Python-level loops by pushing
operations down to optimized C code, resulting in 14x-500x speedups.
"""

import numpy as np
import timeit


# ============================================================
# Loop 1: Element-wise computation (~14x speedup)
# Pattern: Apply a formula to every element in an array
# ============================================================

def loop1_loop():
    data = np.random.rand(100_000)
    result = np.empty(len(data))
    for i in range(len(data)):
        result[i] = (data[i] * 2) + 5
    return result


def loop1_vectorized():
    data = np.random.rand(100_000)
    result = (data * 2) + 5
    return result


# ============================================================
# Loop 2: Conditional replacement (~17x speedup)
# Pattern: Replace values based on a condition
# ============================================================

def loop2_loop():
    data = np.random.rand(100_000)
    result = np.copy(data)
    for i in range(len(data)):
        if result[i] > 0.5:
            result[i] = 1.0
        else:
            result[i] = 0.0
    return result


def loop2_vectorized():
    data = np.random.rand(100_000)
    result = np.where(data > 0.5, 1.0, 0.0)
    return result


# ============================================================
# Loop 3: Outer product using broadcasting (~50x speedup)
# Pattern: Compute every pair combination of two 1D arrays
# Reshape controls which array stretches as rows vs columns
# ============================================================

def loop3_loop():
    a = np.random.rand(1000)
    b = np.random.rand(1000)
    result = np.empty((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            result[i, j] = a[i] * b[j]
    return result


def loop3_vectorized():
    a = np.random.rand(1000)
    b = np.random.rand(1000)
    result = a.reshape(1000, 1) * b.reshape(1, 1000)
    return result


# ============================================================
# Loop 4: Pairwise Euclidean distances (~50x speedup)
# Pattern: 3D reshape + broadcasting for pairwise operations
# The coordinate dimension (2) stays untouched at the end,
# while the points dimension (1000) gets split for broadcasting
# ============================================================

def loop4_loop():
    points = np.random.rand(1000, 2)
    dist = np.empty((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist[i, j] = np.sqrt(dx ** 2 + dy ** 2)
    return dist


def loop4_vectorized():
    points = np.random.rand(1000, 2)
    rows = points.reshape(1000, 1, 2)
    cols = points.reshape(1, 1000, 2)
    diff = rows - cols
    sq = diff ** 2
    sum_sq = np.sum(sq, axis=2)
    dist = np.sqrt(sum_sq)
    return dist


# ============================================================
# Loop 5: Row-wise statistics (~500x speedup)
# Pattern: Compute stats for each row across columns
# axis parameter controls which dimension gets collapsed
# ============================================================

def loop5_loop():
    data = np.random.rand(10_000, 100)
    means = np.empty(10_000)
    stds = np.empty(10_000)
    for i in range(10_000):
        total = 0.0
        for j in range(100):
            total += data[i, j]
        means[i] = total / 100

        var_total = 0.0
        for j in range(100):
            var_total += (data[i, j] - means[i]) ** 2
        stds[i] = np.sqrt(var_total / 100)
    return means, stds


def loop5_vectorized():
    data = np.random.rand(10_000, 100)
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    return means, stds


# ============================================================
# Run all benchmarks
# ============================================================

if __name__ == "__main__":
    benchmarks = [
        ("Loop 1: Element-wise computation", loop1_loop, loop1_vectorized, 10),
        ("Loop 2: Conditional replacement", loop2_loop, loop2_vectorized, 10),
        ("Loop 3: Outer product", loop3_loop, loop3_vectorized, 10),
        ("Loop 4: Pairwise distances", loop4_loop, loop4_vectorized, 10),
        ("Loop 5: Row-wise statistics", loop5_loop, loop5_vectorized, 10),
    ]

    print("NumPy Vectorization Benchmarks")
    print("=" * 60)

    for name, loop_fn, vec_fn, runs in benchmarks:
        loop_time = timeit.timeit(loop_fn, number=runs)
        vec_time = timeit.timeit(vec_fn, number=runs)
        speedup = loop_time / vec_time
        print(f"\n{name}")
        print(f"  Loop:       {loop_time:.4f}s ({runs} runs)")
        print(f"  Vectorized: {vec_time:.4f}s ({runs} runs)")
        print(f"  Speedup:    {speedup:.1f}x")