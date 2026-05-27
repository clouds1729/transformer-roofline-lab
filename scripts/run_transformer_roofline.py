from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.transformer_workloads import get_transformer_workloads
from src.hardware_profiles import get_hardware_profiles
from src.transformer_roofline import analyze_workload_on_hardware


RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "transformer_roofline_results.csv"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_results(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    headers = [
        "workload",
        "hardware",
        "TFLOPs",
        "GB moved",
        "AI FLOP/B",
        "achv TFLOP/s",
        "lat ms",
        "bound",
    ]
    print("Transformer Roofline Summary")
    print("=" * 119)
    print(
        f"{headers[0]:<20} {headers[1]:<26} {headers[2]:>10} {headers[3]:>10} "
        f"{headers[4]:>12} {headers[5]:>14} {headers[6]:>10} {headers[7]:<14}"
    )
    print("-" * 119)
    for row in rows:
        print(
            f"{row['workload']:<20} {row['hardware']:<26} "
            f"{format_float(float(row['total_tflops'])):>10} "
            f"{format_float(float(row['memory_traffic_gb'])):>10} "
            f"{format_float(float(row['arithmetic_intensity_flops_per_byte'])):>12} "
            f"{format_float(float(row['achievable_tflops'])):>14} "
            f"{format_float(float(row['estimated_latency_ms'])):>10} "
            f"{row['bound']:<14}"
        )
    print("=" * 119)
    print(f"Saved CSV: {RESULTS_CSV}")


def main() -> int:
    rows = [
        analyze_workload_on_hardware(workload, hardware)
        for workload in get_transformer_workloads()
        for hardware in get_hardware_profiles()
    ]
    write_results(rows, RESULTS_CSV)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
