"""Operations, FLOP / byte models and timing helpers for the roofline study."""
import time
import numpy as np

DTYPES = {"float32": np.float32, "float64": np.float64}


def matvec_model(n, itemsize):
    """y = A x.  FLOPs: n*n multiplications + n*(n-1) additions ~ 2 n^2.
    Ideal traffic: read A (n^2), read x (n), write y (n)."""
    flops = 2 * n * n
    nbytes = (n * n + 2 * n) * itemsize
    return flops, nbytes


def matmat_model(n, itemsize):
    """C = A B.  FLOPs: n^3 multiplications + n^2(n-1) additions ~ 2 n^3.
    Ideal traffic (perfect reuse, infinite cache): read A, B and write C."""
    flops = 2 * n ** 3
    nbytes = 3 * n * n * itemsize
    return flops, nbytes


def make_matvec(n, dtype):
    A = np.full((n, n), 1.0, dtype=dtype)
    x = np.full(n, 3.0, dtype=dtype)
    y = np.empty(n, dtype=dtype)
    return lambda: np.matmul(A, x, out=y)


def make_matmat(n, dtype):
    A = np.full((n, n), 1.0, dtype=dtype)
    B = np.full((n, n), 2.0, dtype=dtype)
    C = np.empty((n, n), dtype=dtype)
    return lambda: np.matmul(A, B, out=C)


def time_kernel(fn, repeats=10, min_time=0.2, warmup=2):
    """Time fn(). Allocation/initialisation happen outside (in make_*).
    Each sample runs fn() `inner` times so that a sample lasts >= min_time/ repeats-ish;
    returns the list of per-call times (seconds)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
    inner = max(1, int(min_time / max(one, 1e-9)))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples
