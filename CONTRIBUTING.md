# Contributing to Vulkan AI

Thanks for helping improve Vulkan AI. The project is pre-alpha, so small,
well-tested changes are especially valuable.

## Development workflow

1. Open an issue before work that changes public APIs or architecture.
2. Create a focused branch from `main`.
3. Keep commits atomic and describe the reason for each change.
4. Add or update tests and documentation with the implementation.
5. Open a pull request and complete the provided checklist.

## Backend extensions

Follow [ADR 0002](docs/adr/0002-extension-boundary.md) when adding an operation:
start with Burn primitives, use CubeCL for a custom kernel, and keep a portable
reference implementation. Direct WGPU or Vulkan interoperability requires a
concrete capability or measured performance gap, a design issue, and a
follow-up ADR covering resource ownership, synchronization, safety, fallback,
and validation.

The quadratic example demonstrates the expected shape: a backend trait, a
portable Burn reference, a CubeCL implementation for `CubeBackend`, a custom
fusion-stream registration, and an explicit Burn autodiff rule. Its current
kernel supports `f32` tensors of any rank and shape; non-contiguous inputs are
copied to contiguous storage before dispatch, empty tensors are returned
without dispatch, and other floating-point storage types fall back to the Burn
reference.

## Local checks

Run the checks used by continuous integration:

```shell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --features vulkan --bin vulkan-ai-probe
cargo test --features vulkan-fusion --bin vulkan-ai-probe
```

The Vulkan test command compiles the GPU path and runs unit tests that do not
initialize a GPU. Running the probe itself requires a compatible device and
driver and compares its training and custom-operation results with the CPU
backend. Use
`cargo run --release --features vulkan` and repeat with `vulkan-fusion` when
changing timing, fusion, or synchronization behavior.

Custom-kernel changes must run both release commands. The probe benchmarks the
CubeCL quadratic path against its Burn reference with synchronized,
alternating-order samples and verifies the custom forward and backward results
on a non-contiguous input.

## Compatibility and releases

- `main` must remain releasable.
- Public releases follow Semantic Versioning.
- User-visible changes belong in `CHANGELOG.md` under `Unreleased`.
- Maintainers create annotated version tags and matching GitHub Releases.

Please keep discussions professional, specific, and focused on the technical
tradeoffs involved.
