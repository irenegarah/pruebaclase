"""Benchmark y=Ax and C=AB for float32/float64 and write results/results.csv."""
import argparse, csv, os, platform
import numpy as np
from roofline_ops import *

MATVEC_N = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MATMAT_N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def stream_bandwidth(dtype, n=50_000_000, repeats=10):
    """Rough peak memory bandwidth (GB/s): a = b + s*c  (STREAM triad style, 3 arrays moved)."""
    b = np.ones(n, dtype=dtype); c = np.ones(n, dtype=dtype); a = np.empty(n, dtype=dtype)
    tmp = np.empty(n, dtype=dtype)
    def triad():
        np.multiply(c, dtype(2.0), out=tmp); np.add(b, tmp, out=a)
    # numpy needs two passes; count real traffic: read c, write tmp, read b, read tmp, write a = 5 arrays
    t = min(time_kernel(triad, repeats=repeats, min_time=0.5))
    return 5 * n * np.dtype(dtype).itemsize / t / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--out", default="results")
    p.add_argument("--max-matvec", type=int, default=MATVEC_N[-1])
    p.add_argument("--max-matmat", type=int, default=MATMAT_N[-1])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for op, sizes, model, make in [
        ("matvec", [n for n in MATVEC_N if n <= a.max_matvec], matvec_model, make_matvec),
        ("matmat", [n for n in MATMAT_N if n <= a.max_matmat], matmat_model, make_matmat)]:
        for dname, dt in DTYPES.items():
            for n in sizes:
                fn = make(n, dt)
                s = time_kernel(fn, repeats=a.repeats)
                flops, nbytes = model(n, np.dtype(dt).itemsize)
                tmin, tmed = min(s), float(np.median(s))
                rows.append(dict(op=op, dtype=dname, N=n, t_min=tmin, t_median=tmed,
                                 t_std=float(np.std(s)), flops=flops, bytes=nbytes,
                                 gflops_med=flops / tmed / 1e9, gflops_max=flops / tmin / 1e9,
                                 gbs_med=nbytes / tmed / 1e9, gbs_max=nbytes / tmin / 1e9,
                                 ai=flops / nbytes))
                r = rows[-1]
                print(f"{op:6s} {dname} N={n:6d} t={tmed:.3e}s {r['gflops_med']:8.2f} GFLOP/s {r['gbs_med']:7.2f} GB/s", flush=True)
                del fn
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    bw = {d: stream_bandwidth(dt) for d, dt in DTYPES.items()}
    with open(os.path.join(a.out, "machine.txt"), "w") as f:
        f.write(f"platform: {platform.platform()}\nprocessor: {platform.processor()}\n"
                f"numpy: {np.__version__}\nstream_like_GBs: {bw}\n")
    print("triad-like bandwidth (GB/s):", bw)


if __name__ == "__main__":
    main()
