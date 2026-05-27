from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hardware_profiles import HARDWARE_PROFILES


RESULTS_CSV = REPO_ROOT / "results" / "transformer_roofline_results.csv"
PLOTS_DIR = REPO_ROOT / "results" / "plots"
PLOT_PATH = PLOTS_DIR / "transformer_roofline.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_CSV}. Run scripts/run_transformer_roofline.py first."
        )

    rows = read_rows(RESULTS_CSV)
    rows_by_hardware: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_hardware[row["hardware"]].append(row)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(rows_by_hardware), figsize=(17, 5), sharey=True)
    if len(rows_by_hardware) == 1:
        axes = [axes]

    for axis, (hardware_name, hardware_rows) in zip(axes, rows_by_hardware.items()):
        hardware = HARDWARE_PROFILES[hardware_name]
        intensities = [
            float(row["arithmetic_intensity_flops_per_byte"]) for row in hardware_rows
        ]
        min_x = max(min(intensities) / 4, 0.1)
        max_x = max(max(intensities) * 4, hardware.ridge_point_flops_per_byte * 2)
        xs = [10 ** x for x in frange(math.log10(min_x), math.log10(max_x), 100)]
        ys = [
            min(hardware.peak_tflops, x * hardware.memory_bandwidth_bytes_s / 1e12)
            for x in xs
        ]

        axis.plot(xs, ys, color="#2f6f73", linewidth=2, label="roofline")
        axis.axhline(
            hardware.peak_tflops,
            color="#334155",
            linestyle="--",
            linewidth=1,
            label="peak compute",
        )
        axis.axvline(
            hardware.ridge_point_flops_per_byte,
            color="#9a6a00",
            linestyle=":",
            linewidth=1,
            label="ridge point",
        )

        for row in hardware_rows:
            x = float(row["arithmetic_intensity_flops_per_byte"])
            y = float(row["achievable_tflops"])
            axis.scatter(x, y, s=52)
            axis.annotate(
                row["workload"],
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(hardware_name)
        axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
        axis.grid(True, which="both", linestyle=":", linewidth=0.5)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Achievable performance (TFLOP/s)")
    fig.suptitle("Transformer Roofline Lab")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    print(f"Saved plot: {PLOT_PATH}")
    return 0


def frange(start: float, stop: float, steps: int) -> list[float]:
    if steps <= 1:
        return [start]
    step = (stop - start) / (steps - 1)
    return [start + i * step for i in range(steps)]


if __name__ == "__main__":
    raise SystemExit(main())
