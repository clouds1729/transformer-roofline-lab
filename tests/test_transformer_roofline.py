from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from examples.transformer_workloads import TRANSFORMER_WORKLOADS
from src.hardware_profiles import HARDWARE_PROFILES
from src.transformer_roofline import (
    TransformerWorkload,
    arithmetic_intensity,
    estimate_memory_traffic_bytes,
    estimate_transformer_flops,
    roofline_performance,
)


def test_flop_estimation_matches_closed_form_for_tiny_transformer() -> None:
    workload = TransformerWorkload(
        name="unit",
        batch_size=1,
        sequence_length=2,
        hidden_dim=4,
        num_layers=1,
        num_attention_heads=2,
        mlp_expansion_ratio=2,
        bytes_per_element=2,
    )

    expected = 2 * 2 * 4 * 12
    expected += 2 * 1 * 2 * 2 * 2 * 2
    expected += 2 * 1 * 2 * 2 * 2 * 2
    expected += 2 * 2 * 4 * 4
    expected += 2 * 2 * 4 * 8 + 2 * 2 * 8 * 4

    assert estimate_transformer_flops(workload) == expected


def test_arithmetic_intensity_calculation() -> None:
    assert arithmetic_intensity(200.0, 50.0) == 4.0
    with pytest.raises(ValueError):
        arithmetic_intensity(200.0, 0.0)


def test_roofline_bound_classification() -> None:
    hardware = HARDWARE_PROFILES["generic_mobile_npu"]

    low_intensity_perf, low_intensity_bound = roofline_performance(1.0, hardware)
    high_intensity_perf, high_intensity_bound = roofline_performance(10_000.0, hardware)

    assert low_intensity_bound == "memory-bound"
    assert low_intensity_perf < hardware.peak_tflops
    assert high_intensity_bound == "compute-bound"
    assert high_intensity_perf == hardware.peak_tflops


def test_example_workloads_have_positive_metrics() -> None:
    for workload in TRANSFORMER_WORKLOADS.values():
        assert estimate_transformer_flops(workload) > 0
        assert estimate_memory_traffic_bytes(workload) > 0


def test_roofline_script_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_transformer_roofline.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Transformer Roofline Summary" in result.stdout

    csv_path = repo_root / "results" / "transformer_roofline_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(TRANSFORMER_WORKLOADS) * len(HARDWARE_PROFILES)
