"""Build report.html (and report.pdf via headless Chrome if available) from results/."""
import os, subprocess, pandas as pd
df = pd.read_csv("results/results.csv")

def table(op):
    rows = ""
    for _, r in df[df.op == op].iterrows():
        rows += (f"<tr><td>{r.dtype}</td><td>{r.N}</td><td>{r.t_median:.3e}</td><td>{r.gflops_med:.1f}</td>"
                 f"<td>{r.gbs_med:.1f}</td><td>{r.ai:.2f}</td></tr>")
    return ("<table><tr><th>dtype</th><th>N</th><th>median time (s)</th><th>GFLOP/s</th>"
            "<th>GB/s (model)</th><th>AI (FLOP/B)</th></tr>" + rows + "</table>")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Roofline report</title>
<style>body{{font-family:Georgia,serif;max-width:800px;margin:auto;font-size:11pt;line-height:1.4}}
table{{border-collapse:collapse;font-size:9pt;margin:8px 0}}td,th{{border:1px solid #999;padding:2px 8px;text-align:right}}
img{{max-width:100%}}h1{{text-align:center}}.pb{{page-break-before:always}}</style></head><body>
<h1>Homework: Exploring the Roofline Model<br><small>y = Ax and C = AB with NumPy</small></h1>

<h2>1. Methodology</h2>
<p>Both operations are implemented with <code>np.matmul</code> writing into preallocated output arrays (<code>out=</code>); inputs are constant-filled (A=1, B=2, x=3) as requested. Allocation and initialisation lie outside the timed region. After two warm-up calls, each sample averages several calls (&ge;0.2&nbsp;s of work) and 7 samples are taken per point; the <b>median</b> is reported. Sizes: N=128&hellip;16384 (matvec) and N=64&hellip;8192 (matmat), for float32 and float64. Machine: Apple M1 Pro, 16&nbsp;GB, NumPy 2.3.2 linked to Apple Accelerate. Memory-bandwidth roof: STREAM-like triad, &asymp;81&nbsp;GB/s. Compute roof per precision: best matmul GFLOP/s observed (float32 &asymp;2.3&nbsp;TFLOP/s, float64 &asymp;0.6&nbsp;TFLOP/s). Code and scripts: <code>benchmark.py</code>, <code>plot_results.py</code>.</p>

<h2>2. Arithmetic intensity (s = 4 B for float32, 8 B for float64)</h2>
<table><tr><th>Operation</th><th>FLOPs</th><th>Bytes</th><th>AI float32</th><th>AI float64</th></tr>
<tr><td>y = Ax</td><td>2N&sup2; (N&sup2; mul + N(N&minus;1) add)</td><td>(N&sup2;+2N)s</td><td>&asymp;0.5</td><td>&asymp;0.25</td></tr>
<tr><td>C = AB</td><td>2N&sup3;</td><td>3N&sup2;s</td><td>N/6</td><td>N/12</td></tr></table>
<p><b>Assumptions.</b> Matvec: every element of A is read once and never reused (only x is reused, and it is small and cached); traffic &asymp; size of A, so the estimate is almost exact. Matmat: perfect reuse &ndash; A and B are read once and C written once, as if the cache were infinite. This is an <i>upper bound</i> on AI: real blocked implementations reload tiles, and C writes may incur write-allocate reads. Hence the model bytes are a lower bound of actual traffic, which we did not measure with hardware counters.</p>

<h2 class="pb">3. Results</h2>
<h3>3.1 Matrix&ndash;vector</h3>{table("matvec")}
<h3>3.2 Matrix&ndash;matrix</h3>{table("matmat")}
<img src="results/plots/time_vs_N.png"><p><i>Fig. 1 &ndash; Execution time vs N with O(N&sup2;)/O(N&sup3;) reference.</i></p>
<img src="results/plots/perf_bandwidth_vs_N.png"><p><i>Fig. 2 &ndash; GFLOP/s and effective bandwidth (model bytes / time).</i></p>
<img src="results/plots/roofline.png" style="max-width:70%"><p><i>Fig. 3 &ndash; Roofline with measured points.</i></p>
<img src="results/plots/speedup_f32.png" style="max-width:60%"><p><i>Fig. 4 &ndash; float64/float32 time ratio.</i></p>

<h3>3.3 Roofline interpretation</h3>
<p><b>Matvec</b> has AI 0.25&ndash;0.5 FLOP/B, far left of the ridge point (&asymp;7 FLOP/B in float64, &asymp;29 in float32): it is <b>memory-bound</b>. For N&ge;2048 its performance is almost independent of N (6&ndash;7 GFLOP/s float64, 14&ndash;16 float32), equal to AI &times; bandwidth, and it reaches only 30&ndash;40&nbsp;% of the triad bandwidth (25&ndash;32 vs 81&nbsp;GB/s), i.e. it does not hit the roof. For N&le;1024 the matrix fits in cache, so points lie <i>above</i> the DRAM roof (up to 169&nbsp;GB/s): the DRAM roof does not apply to cache-resident data.<br>
<b>Matmat</b> has AI &ge;5 (float64) and growing &prop; N, well beyond the ridge point once N&gt;~64: it is <b>compute-bound</b> and reaches 70&ndash;95&nbsp;% of the best observed peak (0.41&ndash;0.60 TFLOP/s float64, 1.6&ndash;2.2 TFLOP/s float32). Its effective GB/s <i>falls</i> with N (down to 0.6&ndash;1.2 GB/s at N=8192) because data reuse makes the actual traffic per FLOP tiny: the kernel is limited by arithmetic units, not memory.<br>
<b>Data reuse.</b> In matvec each loaded byte supports ~1 FLOP; in matmat each element of A and B is used O(N) times from registers/cache, so AI grows linearly with N. This is the only reason matmat can approach the compute roof.</p>

<h2>4. Discussion</h2>
<p><b>1. Scaling with N.</b> Matvec time grows as N&sup2; (about 4&times; per doubling for N&ge;2048: 2.1&rarr;8.4&rarr;37&nbsp;ms in float32) and matmat as N&sup3; (about 8&times;: 80&rarr;684&nbsp;ms float32 from N=4096 to 8192), as expected from the FLOP counts (and, for matvec, from the bytes moved). At small N, call overhead (&micro;s) and cache effects break the pure power law; the matvec slope is also steeper than 2 between N=1024 and 2048, when A leaves the cache (float32: 85&rarr;27 GFLOP/s).</p>
<p><b>2. Matvec vs matmat.</b> Matmat reaches ~100&ndash;300&times; more GFLOP/s (e.g. 1.7 TFLOP/s vs 16 GFLOP/s in float32 at N=4096) because its AI is O(N) versus O(1) for matvec: matvec performance = AI&times;bandwidth, matmat performance = compute peak.</p>
<p><b>3. float64 &rarr; float32.</b> Matvec is &asymp;2.3&times; faster at large N: half the bytes per element halve the traffic (2&times;), and AI doubles (0.25&rarr;0.5), so GFLOP/s doubles at equal bandwidth. Matmat is &asymp;3.9&ndash;4&times; faster (321 vs 80&nbsp;ms at N=4096), more than the 2&times; expected from SIMD width, so the float32 path of the library is evidently better optimised than its float64 path on this chip (the compute roofs differ by 3.9&times;). The expected benefit is therefore observed, and exceeded.</p>
<p><b>4. Unexpected results / deviations.</b> (a) Matvec is far from the bandwidth roof (30&ndash;40&nbsp;%): GEMV seems unable to saturate DRAM here (likely limited threading/streaming in the library; not verified). (b) Matvec points above the roof at N&le;1024 are cache effects, not violations. (c) Matmat performance peaks around N=512&ndash;1024 and drops ~25&nbsp;% for larger N (cache blocking/working-set effects, possibly power/thermal limits; not verified). (d) The compute roofs are empirical, so by construction no point exceeds them. (e) Effective GB/s uses model bytes, not hardware-counter traffic, so for matmat it underestimates the real traffic. (f) A single machine, run once, with background load possible: run-to-run variation of ~10&nbsp;% should be assumed.</p>
</body></html>"""
open("report.html", "w").write(html)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if os.path.exists(chrome):
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath('report.pdf')}", "file://" + os.path.abspath("report.html")],
                   check=True, capture_output=True)
    print("report.pdf written")
