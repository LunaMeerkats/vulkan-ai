# Vulkan AI

Vulkan AI is an early-stage research project for making neural-network training
experiments practical on Vulkan-capable hardware. The first milestone validates
Burn's Vulkan/SPIR-V backend with deterministic forward and backward checks.

The project deliberately starts with a small compatibility probe. Reliable
device discovery, gradients, and reproducible tests are prerequisites for later
work on custom operations, profiling, and Vulkan interoperability.

The rationale for building on Burn rather than starting with a raw Vulkan
abstraction is recorded in
[ADR 0001](docs/adr/0001-burn-vulkan-backend.md).

## Status

Pre-alpha. There is no stable public API or release yet.

## Quick start

The default test suite uses Burn's portable Flex CPU backend, so it can run in
CI without a GPU:

```shell
cargo test
```

To execute the same forward and backward calculation through Vulkan:

```shell
cargo run --no-default-features --features vulkan --bin vulkan-ai-probe
```

You need a Vulkan-capable device and a working Vulkan driver. The probe forces
Burn's Vulkan graphics API and uses its SPIR-V compiler path.

Expected values are:

```text
Vulkan forward output: [8.0, 18.0]
Vulkan weight gradient: [4.0, 6.0]
```

## Near-term roadmap

- Report the selected Vulkan adapter and relevant device capabilities.
- Add CPU/Vulkan gradient parity tests for a small trainable model.
- Record reproducible timing, fusion, and synchronization diagnostics.
- Implement one custom operation with an explicit backward rule.
- Document the boundary between Burn/CubeCL extensions and direct Vulkan
  interoperability in an architecture decision record.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. This project is
licensed under the Apache License 2.0.
