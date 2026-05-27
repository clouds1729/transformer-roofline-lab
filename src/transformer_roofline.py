"""Small transformer roofline estimators used by the demo scripts.

The original FRAME operator-level model remains in ``src/operators.py`` and
``src/operator_base.py``. This module adds a compact, educational workload-level
view for transformer blocks so examples can run without notebook state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerWorkload:
    name: str
    batch_size: int
    sequence_length: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    mlp_expansion_ratio: float
    bytes_per_element: int
    notes: str = ""

    @property
    def head_dim(self) -> int:
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"{self.name}: hidden_dim must be divisible by num_attention_heads"
            )
        return self.hidden_dim // self.num_attention_heads


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    peak_tflops: float
    memory_bandwidth_gb_s: float
    notes: str = ""

    @property
    def peak_flops(self) -> float:
        return self.peak_tflops * 1e12

    @property
    def memory_bandwidth_bytes_s(self) -> float:
        return self.memory_bandwidth_gb_s * 1e9

    @property
    def ridge_point_flops_per_byte(self) -> float:
        return self.peak_flops / self.memory_bandwidth_bytes_s


def estimate_transformer_flops(workload: TransformerWorkload) -> float:
    """Estimate dense forward-pass FLOPs for a decoder/encoder-like block stack.

    The estimate counts multiply-add as two FLOPs. Per layer it includes QKV
    projections, attention score/value matmuls, output projection, and two MLP
    matmuls. Softmax, normalization, activation functions, and embedding tables
    are intentionally omitted to keep the roofline math focused and explainable.
    """

    _ = workload.head_dim
    batch = workload.batch_size
    seq = workload.sequence_length
    hidden = workload.hidden_dim
    mlp_hidden = int(hidden * workload.mlp_expansion_ratio)

    qkv_projection = 2 * batch * seq * hidden * (3 * hidden)
    attention_scores = 2 * batch * workload.num_attention_heads * seq * seq * workload.head_dim
    attention_values = 2 * batch * workload.num_attention_heads * seq * seq * workload.head_dim
    output_projection = 2 * batch * seq * hidden * hidden
    mlp = 2 * batch * seq * hidden * mlp_hidden + 2 * batch * seq * mlp_hidden * hidden

    return workload.num_layers * (
        qkv_projection
        + attention_scores
        + attention_values
        + output_projection
        + mlp
    )


def estimate_memory_traffic_bytes(workload: TransformerWorkload) -> float:
    """Approximate bytes moved for one forward pass.

    This simple model counts each layer's main weights once and counts a small
    set of activation reads/writes around projections, attention matrices, MLP
    intermediates, and residual outputs. It is deliberately conservative and
    hardware-agnostic: cache reuse, tiling, quantization metadata, KV-cache
    behavior, and fusion are outside this first-pass educational model.
    """

    _ = workload.head_dim
    batch = workload.batch_size
    seq = workload.sequence_length
    hidden = workload.hidden_dim
    heads = workload.num_attention_heads
    mlp_hidden = int(hidden * workload.mlp_expansion_ratio)
    elem = workload.bytes_per_element

    qkv_weights = 3 * hidden * hidden
    output_weights = hidden * hidden
    mlp_weights = hidden * mlp_hidden + mlp_hidden * hidden
    weights_per_layer = qkv_weights + output_weights + mlp_weights

    token_activations = batch * seq * hidden
    qkv_activations = 3 * batch * seq * hidden
    attention_matrix = batch * heads * seq * seq
    mlp_activations = batch * seq * mlp_hidden

    activations_per_layer = (
        token_activations
        + qkv_activations
        + 2 * attention_matrix
        + 2 * mlp_activations
        + token_activations
    )

    return workload.num_layers * (weights_per_layer + activations_per_layer) * elem


def arithmetic_intensity(flops: float, memory_traffic_bytes: float) -> float:
    if memory_traffic_bytes <= 0:
        raise ValueError("memory_traffic_bytes must be positive")
    return flops / memory_traffic_bytes


def roofline_performance(
    arithmetic_intensity_flops_per_byte: float, hardware: HardwareProfile
) -> tuple[float, str]:
    """Return achievable TFLOP/s and the active roofline bound."""

    memory_limited_flops = (
        arithmetic_intensity_flops_per_byte * hardware.memory_bandwidth_bytes_s
    )
    achievable_flops = min(hardware.peak_flops, memory_limited_flops)
    bound = "compute-bound" if hardware.peak_flops <= memory_limited_flops else "memory-bound"
    return achievable_flops / 1e12, bound


def analyze_workload_on_hardware(
    workload: TransformerWorkload, hardware: HardwareProfile
) -> dict[str, float | str | int]:
    flops = estimate_transformer_flops(workload)
    memory_bytes = estimate_memory_traffic_bytes(workload)
    intensity = arithmetic_intensity(flops, memory_bytes)
    achievable_tflops, bound = roofline_performance(intensity, hardware)
    estimated_seconds = flops / (achievable_tflops * 1e12)

    return {
        "workload": workload.name,
        "hardware": hardware.name,
        "batch_size": workload.batch_size,
        "sequence_length": workload.sequence_length,
        "hidden_dim": workload.hidden_dim,
        "num_layers": workload.num_layers,
        "num_attention_heads": workload.num_attention_heads,
        "mlp_expansion_ratio": workload.mlp_expansion_ratio,
        "bytes_per_element": workload.bytes_per_element,
        "total_flops": flops,
        "total_tflops": flops / 1e12,
        "memory_traffic_bytes": memory_bytes,
        "memory_traffic_gb": memory_bytes / 1e9,
        "arithmetic_intensity_flops_per_byte": intensity,
        "peak_tflops": hardware.peak_tflops,
        "memory_bandwidth_gb_s": hardware.memory_bandwidth_gb_s,
        "ridge_point_flops_per_byte": hardware.ridge_point_flops_per_byte,
        "achievable_tflops": achievable_tflops,
        "estimated_latency_ms": estimated_seconds * 1e3,
        "bound": bound,
        "hardware_notes": hardware.notes,
        "workload_notes": workload.notes,
    }
