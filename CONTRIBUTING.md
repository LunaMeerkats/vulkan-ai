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
driver and compares its single-step gradients, 20-step SGD loss trajectory,
final parameters, and custom-operation results with the CPU backend. Use
`cargo run --release --features vulkan` and repeat with `vulkan-fusion` when
changing timing, fusion, or synchronization behavior.

Custom-kernel changes must run both release commands. The probe benchmarks the
CubeCL quadratic path against its Burn reference at five representative tensor
sizes with 20 warm-ups, 20 synchronized alternating-order wall-clock samples,
and raw JSON output. The unfused run also collects CubeCL runtime profiles so
device execution can be distinguished from managed allocation, submission,
and synchronization overhead. Nested CubeCL profiling is intentionally
disabled for the fused run; its synchronized wall-clock sweep remains
required. The probe also verifies custom forward and backward results on a
non-contiguous input.

The same release commands also compare the custom quadratic forward and
explicit backward rule with the portable Burn autodiff graph at those five
sizes. This second sweep uses preallocated inputs, a mean-squared output loss,
balanced ordering, 20 warm-ups, 20 synchronized samples, and no host readback.
Report both unfused and fused results when changing the operation or its
autodiff integration.

Optimizer or model changes must preserve the deterministic fixed
initialization, batch, update count, and learning rate unless the protocol
change is documented. Run both release commands and report loss reduction plus
CPU/Vulkan final-parameter parity. Checkpoint changes must serialize the model
and optimizer with full precision, restore both into fresh instances, and
compare the complete resumed trajectory and final parameters with uninterrupted
training on CPU and Vulkan. The current stateful protocol checkpoints after
step 10 of 20 using SGD momentum `0.9`, dampening `0.1`, and no Nesterov
momentum. Model changes must exercise both the fixed linear probe and the
`2 -> 3 -> 1` tanh probe for the nonlinear product target, including every
parameter in uninterrupted/resumed and CPU/Vulkan comparisons.

Mini-batch changes must also preserve the deterministic batch sequence and
full-dataset evaluation trajectory. The current five-batch protocol uses
SplitMix64 with seed `0x5eedcafed15ca11e` to generate each epoch permutation.
Follow [ADR 0003](docs/adr/0003-deterministic-multibatch-sampler.md) for the
multi-batch contract: forward Fisher-Yates order, rejection-sampled bounded
indices, and the fixed five-batch conformance vectors are compatibility
requirements rather than implementation details.
Keep ordering and checkpoint invariants behind the internal data-module
boundary; training code must consume batch identifiers through the sampler
rather than mutate its serialized generator, permutation, or cursor directly.
Keep the fixed product inputs and targets behind `InMemoryProductDataset`, use
that dataset's batch count to configure the sampler, and materialize individual
batches and the full evaluation set from that single source. Dataset refactors
must preserve all ten examples and their five two-example groupings.
Checkpoint after step 11 so the generator state, current permutation
`[4, 1, 0, 3, 2]`, and next epoch position `1` are captured inside an epoch with
the model and optimizer; restore all three records into fresh state; and verify
the resumed batch sequence, final sampler state, and CPU/Vulkan parity. The
post-checkpoint order must diverge if either the current permutation or
generator is reset.

## Compatibility and releases

- `main` must remain releasable.
- Public releases follow Semantic Versioning.
- User-visible changes belong in `CHANGELOG.md` under `Unreleased`.
- Maintainers create annotated version tags and matching GitHub Releases.

Please keep discussions professional, specific, and focused on the technical
tradeoffs involved.
