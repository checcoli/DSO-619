# Decision-Focused Kernel Metrics

Public replication package for the DSO 619 final project report
`reports/decision_focused_kernel_report.tex` (Kaiwen Li, Woojin Chae, Soochan
Kim). It contains the code, cached experiment outputs, and report assets
needed to reproduce the figures and tables in the paper.

## What Is Included

Source code:

- `src/experiment_suite.py`: primary synthetic benchmarks A, B, and C
  (nuisance-removal, rotated heteroscedastic newsvendor, rotated allocation).
- `src/representation_aware_sof.py`: representation-aware SOF transfer and the
  $2\times 2$ regret decomposition.

Runner and asset scripts:

- `scripts/run_main_benchmarks.py`: cache-aware runner for Benchmarks A, B,
  and C.
- `scripts/service_level_experiment.py`: standalone runner for Benchmark D
  (cost-dependent embedding; sweeps the newsvendor service level $\alpha$ and
  re-learns a diagonal Mahalanobis metric per $\alpha$).
- `scripts/build_main_report_assets.py`: rebuilds the main report tables and
  figures from the primary benchmark CSV files.
- `scripts/generate_report_artifacts.py`: rebuilds SOF-transfer tables and
  figures, using cached rows when available.
- `scripts/verify_report_claims.py`: checks headline numeric claims in the
  report against cached artifacts.

Report and replication:

- `reports/decision_focused_kernel_report.tex`: the LaTeX source.
- `reports/decision_focused_kernel_report.pdf`: the final compiled report.
- `reports/assets/`: cached CSV files, TeX tables, and figures.
- `replication_package_MS-BDA-20-02949/newsvendor/`: minimal official SOF
  newsvendor code used by the experiments.

Generated build products, Python caches, and unrelated large source data are
excluded.

## Environment

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The LaTeX report build also requires a local TeX distribution with `latexmk`.
The SOF newsvendor files import `mkl` and `gurobipy`; the public loader stubs
those modules because the newsvendor experiments in this project do not need a
commercial Gurobi solve.

## Quick Checks

Verify that cached artifacts reproduce the report's headline conclusions:

```bash
make verify
```

Build the report from cached tables and figures:

```bash
make report-only
```

## Regenerating Results

Regenerate the primary benchmark (A, B, C) CSV files:

```bash
make main-results
```

Regenerate Benchmark D (cost-dependent embedding):

```bash
make service-level
```

Regenerate the main report tables and figures from the CSV files above:

```bash
make main-assets
```

Regenerate the SOF-transfer and decomposition artifacts:

```bash
make artifacts
```

Run a smaller subset of A/B/C while developing:

```bash
make main-results BENCHMARKS=B
make artifacts BENCHMARKS=B
```

The full benchmark suite is computationally expensive. All runners are
cache-aware and reuse complete rows already present in `reports/assets/`.

## Replication Settings
<<<<<<< Updated upstream

All experiments are controlled by parameter grids defined in source files. Below are the 
primary benchmark settings used in the report.

### Benchmark A: Core Synthetic Grid

| Parameter | Values | Notes |
|-----------|--------|-------|
| `d` (dimension) | 5, 10, 20, 50, 100 | Problem size |
| `k` (rank) | 1, 3, 5 | Low-rank structure |
| `N` (samples) | 100, 200, 400, 800 | Training set size |
| Replications | 6 | Independent random seeds |
| Seed offset | `seed_base + replication_id` | Deterministic and reproducible |

**Total combinations:** 5 × 3 × 4 × 6 = 360 rows per configuration.

### Benchmark B & C: Extended Replication

| Parameter | Values |
|-----------|--------|
| Replications | 20 | 
| Seed pattern | Deterministic (fixed base + offset) |

### SOF-Transfer Experiments

The SOF-transfer and decomposition experiments in `src/representation_aware_sof.py` 
support two performance profiles:

| Profile | Setting | Use Case |
|---------|---------|----------|
| **Fast** | Reduced grid, fewer replications | Development and testing |
| **Full** | Complete grid (report results) | Final reproduction |

Switch profiles via environment variable:
```bash
export SOF_PROFILE=full  # Use report settings (default)
export SOF_PROFILE=fast  # Use reduced settings
```

The cached public artifacts (`reports/assets/`) correspond to the **full** profile.

### Reproducibility Notes

- All synthetic data generators use **fixed deterministic seeds** for exact reproducibility.
- Each benchmark run reuses complete rows already present in `reports/assets/` (cache-aware).
- Random number generation is seeded at the start of each benchmark iteration.
- The seed progression is: `seed_base + replication_id`, where `seed_base` is set per benchmark.
=======

The benchmark grids and replication counts used in the report are:

| Benchmark | Grid / parameters                                                          | Replications |
| --------- | -------------------------------------------------------------------------- | ------------ |
| A         | $d \in \{5,10,20,50,100\}$, $k \in \{1,3,5\}$, $N \in \{100,200,400,800\}$ | 6 per cell   |
| B         | rotated heteroscedastic newsvendor, $\alpha = 0.8$                          | 20           |
| C         | rotated two-action allocation                                              | 20           |
| D         | $\alpha \in \{0.50, 0.70, 0.85, 0.90, 0.95, 0.99\}$, $N_{\mathrm{tr}}=250$, $N_{\mathrm{val}}=150$, $N_{\mathrm{te}}=2000$ | 20           |

The SOF-transfer experiments can be switched between a fast profile and a
fuller profile via the `EXPERIMENT_PROFILE` environment variable, documented
in `src/representation_aware_sof.py`. The cached public artifacts correspond
to the report numbers.

## Reproducibility Notes

- All synthetic data generators use fixed deterministic seeds (Benchmarks A--C
  in `src/experiment_suite.py`; Benchmark D in
  `scripts/service_level_experiment.py`).
- The kernel-SAA training pipeline learns $\phi_{\mathrm{pred}}$ and
  $\phi_{\mathrm{dec}}$ within the same training set as the downstream solver
  (apples-to-apples; no extra data is used for embedding learning).
- For the diagonal Mahalanobis families, raw $\log s_j$ parameters are
  mean-centered before exponentiation to remove the $(L,h)$ scale
  non-identifiability with the bandwidth.
>>>>>>> Stashed changes

## Suggested Workflow

1. Run `make verify`.
2. Run `make report-only` if LaTeX is installed.
<<<<<<< Updated upstream
3. For a full rerun, run `make main-results`, `make main-assets`, `make artifacts`, and then `make report-only`.


=======
3. For a full rerun, run `make main-results`, `make service-level`,
   `make main-assets`, `make artifacts`, and then `make report-only`.
>>>>>>> Stashed changes
