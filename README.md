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
primitives, use CubeCL for custom kernels, and require measured evidence plus a
separate design before introducing direct Vulkan interoperability.

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
its explicitly registered `2x + 1` backward rule on CPU and Vulkan.

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

```text
Vulkan forward output: [8.0, 18.0]
Vulkan weight gradient: [4.0, 6.0]
Vulkan custom quadratic output: [2.0, -0.25, 0.0, 3.75]
Vulkan custom quadratic input gradient: [-3.0, 0.0, 1.0, 4.0]
```

## Near-term roadmap

- Replace the composed custom quadratic forward path with a CubeCL kernel and
  benchmark it against the Burn reference while retaining autodiff parity.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. This project is
licensed under the Apache License 2.0.
