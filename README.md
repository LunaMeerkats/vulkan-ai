# Vulkan AI

Vulkan AI is an early-stage research project for making neural-network training
experiments practical on Vulkan-capable hardware. The first milestone validates
Burn's Vulkan/SPIR-V backend with deterministic forward and backward checks.

The project deliberately starts with a small compatibility probe. Reliable
device discovery, gradients, and reproducible tests are prerequisites for later
work on custom operations, profiling, and Vulkan interoperability.

The rationale for building on Burn rather than starting with a raw Vulkan
abstraction is recorded in
[ADR 0001](docs/adr/0001-burn-vulkan-backend.md). The extension boundary is
recorded in [ADR 0002](docs/adr/0002-extension-boundary.md): prefer Burn
primitives, use CubeCL for custom kernels, and require measured evidence plus
a separate design before introducing direct Vulkan interoperability. The first
custom forward kernel follows that boundary: Vulkan dispatches `x² + x` through
CubeCL while Flex retains the portable Burn reference.

## Status

Pre-alpha. There is no stable public API or release yet.

## Quick start

The default test suite uses Burn's portable Flex CPU backend, so it can run in
CI without a GPU:

```shell
cargo test
```

To execute the compatibility probe through Vulkan and compare a deterministic
linear-model training step with the CPU backend:

```shell
cargo run --features vulkan --bin vulkan-ai-probe
```

You need a Vulkan-capable device and a working Vulkan driver. The probe forces
Burn's Vulkan graphics API and uses its SPIR-V compiler path.

The probe first reports the selected adapter, driver, subgroup range, and
effective compute and buffer limits. It then prints the calculation results
and fails if the linear model's predictions, loss, or parameter gradients
differ from the CPU reference beyond an absolute plus relative tolerance of
`1e-5` each. It also validates a custom element-wise `x² + x` operation and
its explicitly registered `2x + 1` backward rule on CPU and Vulkan. The
probe's Vulkan `f32` forward path is one CubeCL kernel, including under Burn
fusion; the parity input is a transposed two-dimensional tensor so the probe
also exercises the documented contiguous-copy path for non-contiguous layouts.

The probe also times the same training workload with five warm-up iterations
and 20 measured iterations. Every iteration ends with an explicit Burn backend
synchronization, and host readback is excluded. The report records the build
profile, fusion state, command task batch limit, raw samples, and min, median,
P95, and max latency. Use a release build for comparable measurements:

```shell
cargo run --release --features vulkan --bin vulkan-ai-probe
cargo run --release --features vulkan-fusion --bin vulkan-ai-probe
```

The first command measures the unfused backend; the second enables Burn's
fusion optimizer. Timing output includes a single-line JSON record suitable
for storing and comparing runs.

Both commands also benchmark the custom CubeCL quadratic kernel against the
portable Burn `multiply + add` reference over 1, 256, 4,096, 65,536, and
1,048,576 `f32` elements. The fixed protocol uses 20 warm-ups and 20
synchronized samples per implementation and size, alternates measurement
order, and excludes input allocation and host readback. Wall-clock samples
include managed output allocation or reuse, dispatch, and synchronization.
The unfused command also records CubeCL runtime profiles and reports whether
the runtime used device or system timing; nested runtime profiling is
intentionally disabled with Burn fusion because it cannot safely wrap a
Fusion synchronization on the same runtime.

The schema-2 JSON report includes every raw wall-clock and available profile
sample, medians, and reference-to-kernel ratios. A ratio above `1.0` means the
custom kernel's median was lower for that measurement scope. Compare the
profile and wall-clock ratios before attributing a result to kernel execution:
a fast device profile can still have little end-to-end impact when managed
allocation, submission, or synchronization dominates. Treat every result as
evidence for that device, driver, build, and fusion mode, not a general
performance claim.

```text
Vulkan forward output: [8.0, 18.0]
Vulkan weight gradient: [4.0, 6.0]
Vulkan custom quadratic output: [2.0, -0.25, 0.0, 3.75]
Vulkan custom quadratic input gradient: [-3.0, 0.0, 1.0, 4.0]
```

## Near-term roadmap

- Use the size-sweep evidence to test a representative compound element-wise
  training operation where fusion or batching can amortize managed allocation,
  submission, and synchronization overhead. The quadratic experiment does not
  justify lower-level Vulkan interoperability.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. This project is
licensed under the Apache License 2.0.
