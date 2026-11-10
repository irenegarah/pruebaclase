# Exploring the Roofline Model: y = Ax and C = AB

## 1. Methodology
Both operations use `np.matmul` with preallocated output arrays (`out=`), inputs filled with constants (A=1, B=2, x=3). Allocation/initialisation is outside the timed region; after 2 warm-up calls each sample averages several calls (>= 0.2 s of work) and 7 samples are taken; the **median** is reported. Machine: Apple M1 Pro, 16 GB, NumPy 2.3.2 with Apple Accelerate (BLAS). Peak bandwidth (~81 GB/s) was estimated with a STREAM-like triad; the compute peak of each precision is the best matmul GFLOP/s measured (float32 ~2.3 TFLOP/s, float64 ~0.6 TFLOP/s).

## 2. Theoretical estimates (s = bytes per element: 4 or 8)
| Operation | FLOPs | Bytes (ideal) | AI (FLOP/byte) | float32 | float64 |
|---|---|---|---|---|---|
| y = Ax | 2N² | (N²+2N)·s | ≈ 2/s | 0.5 | 0.25 |
| C = AB | 2N³ | 3N²·s | 2N/(3s) = N/6 (f32), N/12 (f64) | N/6 | N/12 |

Assumptions: the matvec reads A once and has no reuse of A (each element used once); x and y are small. For matmat the bound assumes perfect reuse (infinite cache): A, B read once, C written once. Real traffic is larger (finite cache, blocking, write-allocate of C), so the matmat AI is an upper bound; matvec's is nearly exact for large N.

## 3. Results
Figures: `results/plots/time_vs_N.png`, `perf_bandwidth_vs_N.png`, `roofline.png`, `speedup_f32.png`; raw data in `results/results.csv`.

Key numbers (median):
- matvec, N=16384: f64 6.2 GFLOP/s (25 GB/s), f32 14.5 GFLOP/s (29 GB/s).
- matvec peak at small N (cache resident): f32 85 GFLOP/s (169 GB/s, N=1024), f64 38 GFLOP/s (153 GB/s, N=512).
- matmat, N≥1024: f32 ≈ 1.6–2.2 TFLOP/s, f64 ≈ 0.41–0.60 TFLOP/s.

Roofline interpretation: matvec sits at AI 0.25–0.5, far left of the ridge point (~7 FLOP/B for f64, ~29 for f32) → memory-bound. For N ≥ 2048 it reaches only ~30–40 % of the STREAM bandwidth (25–32 GB/s vs 81), and it is almost flat in N, as expected for a bandwidth-limited kernel. Matmat has AI ≫ ridge point for N ≳ 64 → compute-bound, and reaches ~70–95 % of the best measured peak. Note the "effective GB/s" of matmat decreases with N (1.2 GB/s at N=8192): it is not memory-limited; the model bytes are minimal traffic and data reuse makes actual traffic per FLOP tiny.

## 4. Discussion
1. **Scaling.** matvec time grows ∝ N² (4× per doubling of N for N≥2048, e.g. 2.1→8.4→37 ms in f32) and matmat ∝ N³ (≈8×; 80→684 ms in f32). Small sizes deviate: call overhead (~µs) dominates for N≤256 and the working set fits in cache.
2. **matvec vs matmat.** matmat achieves 100–300× more GFLOP/s because its AI grows with N (each loaded element is reused O(N) times in registers/cache), while in matvec every element of A is used exactly once, so performance = AI × bandwidth.
3. **float32 vs float64.** matvec: ≈2.3× faster in f32 at large N (2× from half the bytes, plus slightly better efficiency); GFLOP/s doubles because AI doubles. matmat: ≈3.9–4× faster (e.g. 80 ms vs 321 ms at N=4096): the matrix units/SIMD process twice as many f32 lanes and Accelerate's f64 path is much less optimised on this chip; the expected 2× is exceeded.
4. **Unexpected results.** (a) matvec throughput *drops* sharply between N=1024 and N=2048 (f32: 85→27 GFLOP/s): A exceeds the cache (4 MB f32 → 16 MB) and the benchmark switches from cache-resident to DRAM bandwidth; points at small N above the DRAM roof are legitimate because they are served from cache. (b) matvec reaches only ~35 % of the triad bandwidth (single-threaded/poorly parallelised GEMV, plus triad measured with a different access pattern). (c) matmat peaks near N=512–1024 and then falls ~25 %, likely from cache blocking/thermal or power limits; f32 N=1024 point even approaches the roof. (d) The float64 matmat roof is itself an empirical estimate, so being "above" it is not possible by construction. (e) Measured traffic was not read from hardware counters; GB/s is model bytes/time, not actual traffic.

> Check the numbers above against `results/results.csv` before submitting; edit conclusions to your own wording.
