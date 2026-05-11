# Decision-Focused Kernel Metrics

This is the public replication package for `reports/decision_focused_kernel_report.tex`.
It contains the code, cached experiment outputs, and report assets needed to check
and rebuild the paper-style report.

## What Is Included

- `src/experiment_suite.py`: primary synthetic benchmarks A, B, and C.
- `src/representation_aware_sof.py`: representation-aware SOF transfer and decomposition experiments.
- `scripts/run_main_benchmarks.py`: cache-aware runner for the primary benchmark CSV files.
- `scripts/build_main_report_assets.py`: rebuilds the main report tables and figures from primary benchmark CSV files.
- `scripts/generate_report_artifacts.py`: rebuilds SOF-transfer tables and figures, using cached rows when available.
- `scripts/verify_report_claims.py`: checks headline numeric claims in the report against cached artifacts.
- `replication_package_MS-BDA-20-02949/newsvendor/`: minimal official SOF newsvendor code used by the experiments.

Generated build products, Python caches, and unrelated large source data are excluded.

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

Regenerate the primary benchmark CSV files:

```bash
make main-results
```

Regenerate the main report tables and figures from those CSV files:

```bash
make main-assets
```

Regenerate the SOF-transfer and decomposition artifacts:

```bash
make artifacts
```

Run a smaller subset while developing:

```bash
make main-results BENCHMARKS=B
make artifacts BENCHMARKS=B
```

The full benchmark suite is computationally expensive. All runners are
cache-aware and reuse complete rows already present in `reports/assets/`.

## Replication Settings

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

## Suggested Open-Source Workflow

1. Run `make verify`.
2. Run `make report-only` if LaTeX is installed.
3. For a full rerun, run `make main-results`, `make main-assets`, `make artifacts`, and then `make report-only`.


