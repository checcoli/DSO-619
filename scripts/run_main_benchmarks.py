"""Run the three primary benchmark suites and refresh their raw CSV files.

The full grid is computationally expensive. The functions are cache-aware:
existing rows in ``reports/assets/benchmark_*_runs.csv`` are reused unless
required columns are missing. Delete a CSV if you want a completely fresh run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_suite import run_benchmark_a, run_benchmark_b, run_benchmark_c  # noqa: E402

ASSET_DIR = PROJECT_ROOT / "reports" / "assets"


def load_existing(name: str) -> pd.DataFrame | None:
    path = ASSET_DIR / name
    return pd.read_csv(path) if path.exists() else None


def parse_benchmarks(specification: str) -> set[str]:
    if specification.lower() == "all":
        return {"A", "B", "C"}
    selected = {token.strip().upper() for token in specification.split(",") if token.strip()}
    invalid = selected - {"A", "B", "C"}
    if invalid:
        raise ValueError(f"Unknown benchmark selection: {', '.join(sorted(invalid))}")
    if not selected:
        raise ValueError("At least one benchmark must be selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default="all", help="Comma-separated subset: A, B, C, or all.")
    args = parser.parse_args()
    selected = parse_benchmarks(args.benchmarks)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    if "A" in selected:
        print("[main] running Benchmark A", flush=True)
        frame_a = run_benchmark_a(load_existing("benchmark_a_runs.csv"), cache_path=ASSET_DIR / "benchmark_a_runs.csv")
        frame_a.to_csv(ASSET_DIR / "benchmark_a_runs.csv", index=False)

    if "B" in selected:
        print("[main] running Benchmark B", flush=True)
        frame_b = run_benchmark_b(load_existing("benchmark_b_runs.csv"))
        frame_b.to_csv(ASSET_DIR / "benchmark_b_runs.csv", index=False)

    if "C" in selected:
        print("[main] running Benchmark C", flush=True)
        existing_boundary = load_existing("benchmark_c_boundary_curve.csv")
        frame_c, boundary_curve = run_benchmark_c(load_existing("benchmark_c_runs.csv"), existing_boundary=existing_boundary)
        frame_c.to_csv(ASSET_DIR / "benchmark_c_runs.csv", index=False)
        if not boundary_curve.empty:
            boundary_curve.to_csv(ASSET_DIR / "benchmark_c_boundary_curve.csv", index=False)

    print("[main] raw benchmark CSV files are ready", flush=True)


if __name__ == "__main__":
    main()
