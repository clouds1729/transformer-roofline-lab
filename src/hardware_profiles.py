"""Illustrative hardware profiles for roofline examples.

These are intentionally generic educational profiles. They are not vendor
specifications and should not be presented as Qualcomm, NVIDIA, AMD, Intel, or
Apple product numbers.
"""

from __future__ import annotations

from src.transformer_roofline import HardwareProfile


HARDWARE_PROFILES = {
    "generic_mobile_npu": HardwareProfile(
        name="generic_mobile_npu",
        peak_tflops=8.0,
        memory_bandwidth_gb_s=68.0,
        notes="Illustrative mobile-class accelerator profile, not a vendor spec.",
    ),
    "generic_desktop_gpu": HardwareProfile(
        name="generic_desktop_gpu",
        peak_tflops=80.0,
        memory_bandwidth_gb_s=900.0,
        notes="Illustrative desktop GPU profile for comparison.",
    ),
    "generic_edge_accelerator": HardwareProfile(
        name="generic_edge_accelerator",
        peak_tflops=16.0,
        memory_bandwidth_gb_s=120.0,
        notes="Illustrative edge accelerator profile with moderate bandwidth.",
    ),
}


def get_hardware_profiles() -> list[HardwareProfile]:
    return list(HARDWARE_PROFILES.values())
