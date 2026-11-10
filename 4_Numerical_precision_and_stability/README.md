# Roofline model: matrix-vector vs matrix-matrix multiplication (NumPy)

## Files
- `roofline_ops.py` – operations (`y=Ax`, `C=AB`), FLOP/byte models, timing helper (allocation excluded from the timed region, `out=` buffers preallocated).
- `benchmark.py` – runs all sizes for float32/float64, writes `results/results.csv` and `results/machine.txt` (incl. a STREAM-like bandwidth estimate).
- `plot_results.py` – generates `results/plots/*.png` (time vs N, GFLOP/s and GB/s vs N, roofline, float32 speedup).
- `build_report.py` – builds `report.html` and `report.pdf` (the technical report submitted to Atenea) from `results/`.

## Reproduce
```bash
pip install -r requirements.txt
python benchmark.py            # ~1 min; use --max-matmat 4096 --max-matvec 8192 for a quicker run
python plot_results.py
```
Needs ~3 GB of free RAM for the largest sizes (N=16384 float64 matvec, N=8192 float64 matmat).
Results in this repo were measured on an Apple M1 Pro (NumPy 2.3.2 linked to Apple Accelerate).
