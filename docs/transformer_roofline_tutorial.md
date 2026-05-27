# Transformer Roofline Tutorial

FRAME is a roofline-oriented modeling project: it compares how much computation
a workload needs with how much data the hardware must move to feed that
computation. This tutorial explains the simplified transformer demo added in
this repository.

## What Is A Roofline Model?

A roofline model is a first-order performance model with two ceilings:

- **Peak compute throughput**: the fastest rate at which the hardware can do
  arithmetic, usually expressed in FLOP/s or TFLOP/s.
- **Memory bandwidth**: the fastest rate at which data can be moved from memory
  to the compute units, usually expressed in GB/s.

The model asks: for each byte loaded from memory, how many floating-point
operations does the workload perform? That ratio is arithmetic intensity.

## Arithmetic Intensity

Arithmetic intensity is:

```text
arithmetic intensity = total FLOPs / total bytes moved
```

High arithmetic intensity means the workload reuses data well and is more
likely to be limited by peak compute. Low arithmetic intensity means the
workload moves a lot of data per operation and is more likely to be limited by
memory bandwidth.

## Compute-Bound vs Memory-Bound

For a given hardware profile, the roofline estimate is:

```text
achievable FLOP/s = min(peak FLOP/s, arithmetic intensity * memory bandwidth)
```

If `arithmetic intensity * memory bandwidth` is below peak compute, the workload
is memory-bound. If it reaches or exceeds peak compute, the workload is
compute-bound.

The arithmetic intensity where the two ceilings meet is called the ridge point:

```text
ridge point = peak FLOP/s / memory bandwidth
```

## Why Transformers Stress Compute And Memory

Transformer blocks are dominated by matrix multiplications:

- Q, K, and V projections
- attention score computation
- attention-value multiplication
- output projection
- MLP expansion and projection

These operations can produce high FLOP counts, especially as hidden dimension
and layer count grow. At the same time, transformers move large activation
tensors, weight matrices, and attention matrices. Long sequence lengths are
especially costly because attention contains terms that scale roughly with
`sequence_length^2`.

That mix makes transformers a useful educational workload for accelerator
analysis: they reward high matrix throughput, but they also expose bandwidth,
tiling, cache, and data reuse limits.

## Reading The Generated Plot

Run:

```bash
python scripts/run_transformer_roofline.py
python scripts/plot_roofline.py
```

The plot places each workload at its arithmetic intensity and estimated
achievable throughput.

- The sloped line is the memory-bandwidth limit.
- The flat line is the peak-compute limit.
- Points on the slope are memory-bound.
- Points on the flat ceiling are compute-bound.
- The vertical ridge point marks the transition between the two regimes.

The included hardware profiles are illustrative only. They are useful for
comparing architectural tradeoffs, not for making vendor-specific performance
claims.

## Limitations

This demo intentionally keeps the model lightweight. It does not model:

- kernel launch overhead
- cache hierarchy details
- operator fusion
- tensor layout effects
- sparsity
- quantization metadata
- softmax, normalization, and activation costs
- autoregressive KV-cache behavior
- measured utilization from a real compiler or runtime

Those details matter in production performance work. The simplified model is
best used as a clear baseline for reasoning about FLOPs, bytes, arithmetic
intensity, and roofline limits.
