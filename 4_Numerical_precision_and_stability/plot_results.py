"""Generate plots from results/results.csv."""
import os, ast, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
df = pd.read_csv(os.path.join(R, "results.csv"))
bw = ast.literal_eval(open(os.path.join(R, "machine.txt")).read().split("stream_like_GBs:")[1].strip())
col = {"float32": "tab:blue", "float64": "tab:red"}; mk = {"matvec": "o", "matmat": "s"}
os.makedirs(os.path.join(R, "plots"), exist_ok=True)

def sub(op, d): return df[(df.op == op) & (df.dtype == d)]

# 1. time vs N with reference slopes
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (op, p) in enumerate([("matvec", 2), ("matmat", 3)]):
    for d in col:
        s = sub(op, d); ax[i].loglog(s.N, s.t_median, mk[op] + "-", c=col[d], label=d)
    s = sub(op, "float64"); ref = s.t_median.iloc[-1] * (s.N / s.N.iloc[-1]) ** p
    ax[i].loglog(s.N, ref, "k--", lw=1, label=f"O(N^{p})")
    ax[i].set(title=op, xlabel="N", ylabel="time (s)"); ax[i].legend(); ax[i].grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/plots/time_vs_N.png", dpi=150); plt.close()

# 2. GFLOP/s and GB/s vs N
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for op in mk:
    for d in col:
        s = sub(op, d)
        ax[0].semilogx(s.N, s.gflops_med, mk[op], c=col[d], ls="-" if op == "matmat" else "--", label=f"{op} {d}")
        ax[1].semilogx(s.N, s.gbs_med, mk[op], c=col[d], ls="-" if op == "matmat" else "--", label=f"{op} {d}")
ax[0].set(xlabel="N", ylabel="GFLOP/s", title="Performance"); ax[1].set(xlabel="N", ylabel="GB/s (model bytes / time)", title="Effective bandwidth")
for a in ax: a.legend(fontsize=7); a.grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/plots/perf_bandwidth_vs_N.png", dpi=150); plt.close()

# 3. Roofline
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ai_grid = np.logspace(-2, 4, 200)
for d in col:
    peak = df[(df.dtype == d)].gflops_max.max()
    ax.loglog(ai_grid, np.minimum(peak, bw[d] * ai_grid), c=col[d], lw=1, alpha=.6,
              label=f"roof {d}: {peak:.0f} GFLOP/s, {bw[d]:.0f} GB/s")
    for op in mk:
        s = sub(op, d); ax.loglog(s.ai, s.gflops_med, mk[op], c=col[d], mfc="none" if op == "matvec" else col[d], label=f"{op} {d}")
ax.set(xlabel="Arithmetic intensity (FLOP/byte, model)", ylabel="GFLOP/s", title="Roofline"); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{R}/plots/roofline.png", dpi=150); plt.close()

# 4. float32 / float64 speedup
fig, ax = plt.subplots(figsize=(6, 4))
for op in mk:
    a, b = sub(op, "float64").set_index("N"), sub(op, "float32").set_index("N")
    ax.semilogx(a.index, a.t_median / b.t_median, mk[op] + "-", label=op)
ax.axhline(2, c="k", ls="--", lw=1); ax.set(xlabel="N", ylabel="time(f64)/time(f32)", title="float32 speedup"); ax.legend(); ax.grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.savefig(f"{R}/plots/speedup_f32.png", dpi=150); plt.close()
print("plots written to", f"{R}/plots")
