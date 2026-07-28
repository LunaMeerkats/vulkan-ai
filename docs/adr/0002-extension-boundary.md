# ADR 0002: Escalate from Burn primitives to CubeCL before direct Vulkan

- Status: Accepted
- Date: 2026-07-28

## Context

ADR 0001 chose Burn's Vulkan backend so that experiments retain Burn's tensor,
autodiff, model, and training abstractions. In Burn 0.21, the `vulkan` feature
routes tensor work through `burn-wgpu`, CubeCL's WGPU runtime and SPIR-V
compiler, WGPU, and finally the Vulkan driver.

The first custom operation in this project deliberately separates two concerns:

- its public operation and explicit backward rule extend Burn; and
- its forward implementation currently composes Burn primitive operations.

That implementation validates the extension shape, but it is not a custom GPU
kernel. Future experiments need a clear rule for when to keep composing Burn
operations, when to add a CubeCL kernel, and when Vulkan interoperability is
worth the extra ownership and synchronization risk.

## Decision

Use the following extension ladder, stopping at the first layer that satisfies
the experiment.

### 1. Burn tensor operations

Prefer backend-generic Burn operations for model code, reference
implementations, backward rules, and operations whose performance is already
adequate. This keeps CPU/Vulkan parity straightforward and allows Burn's fusion
and backend selection to work.

### 2. CubeCL custom kernels

Use a CubeCL kernel when an experiment needs a fused operation, workgroup or
memory-layout control, or an operation that Burn does not provide. Integrate it
through Burn's `CubeBackend` and the tensor's existing CubeCL compute client so
that Burn and CubeCL continue to own tensor allocation, command submission, and
device synchronization.

Every CubeCL operation must:

- keep a Burn-level reference implementation for correctness tests;
- expose a backend trait and tensor API rather than leaking runtime handles;
- define or register its autodiff behavior explicitly when used in training;
- test expected values on CPU and parity on Vulkan;
- benchmark against the reference path before claiming a performance benefit;
- document supported element types, shapes, layouts, and fallback behavior.

CubeCL remains the default custom-kernel layer even when its WGPU runtime emits
SPIR-V. Generating SPIR-V or tuning dispatch geometry is not, by itself, a
reason to use Vulkan directly.

### 3. Direct GPU interoperability

Do not access raw WGPU HAL or Vulkan handles in the current architecture.
Sharing a WGPU device through supported CubeCL initialization remains inside
the managed path only while CubeCL owns command submission for project tensors.
Submitting separate WGPU command buffers or accessing WGPU HAL/Vulkan handles
counts as direct interoperability.

Direct interoperability may be proposed only for a measured requirement that
Burn and CubeCL cannot satisfy, such as:

- external memory or semaphore exchange with another Vulkan producer;
- a required Vulkan extension, queue operation, or instrumentation feature that
  WGPU/CubeCL does not expose; or
- a demonstrated performance limit that remains after a comparable CubeCL
  implementation and profiling.

The proposal must be a separate issue and follow-up ADR. It must specify:

- ownership and lifetime of the instance, device, queues, buffers, and fences;
- resource state transitions, barriers, synchronization, and device-loss
  behavior across the boundary;
- whether data is copied or shared, including measured transfer cost;
- how Burn tensor shape, type, layout, and allocation invariants are preserved;
- an isolated safety boundary, because WGPU HAL access is unsafe and this crate
  currently forbids unsafe code;
- a portable reference or fallback and the supported OS, driver, and device
  matrix;
- correctness, validation-layer, and performance evidence.

Direct Vulkan must remain an operation-level adapter; it must not create a
second tensor, autodiff, optimizer, or training framework alongside Burn.

## Consequences

- The first custom-kernel experiment follows this path: the quadratic forward
  implementation uses CubeCL on Vulkan while retaining its Burn reference,
  fusion-stream integration, explicit backward rule, parity coverage, and
  synchronized comparative benchmark.
- Most code remains portable across CPU and Vulkan, and the current GPU-free CI
  strategy remains viable.
- CubeCL version changes may require kernel API maintenance, so custom kernels
  must stay small, tested, and pinned through the lockfile.
- Some Vulkan-only capabilities remain unavailable until a concrete experiment
  justifies and designs the interoperability boundary.
- Raw Vulkan interoperability cannot be added without revisiting the current
  `unsafe_code = "forbid"` policy.

## References

- [ADR 0001](0001-burn-vulkan-backend.md)
- [Burn 0.21 custom CubeCL kernel example](https://github.com/tracel-ai/burn/tree/release/0.21/examples/custom-cubecl-kernel)
- [Burn custom CubeCL kernel guide](https://burn.dev/books/burn/advanced/backend-extension/custom-cubecl-kernel.html)
- [Burn WGPU 0.21 backend](https://docs.rs/burn-wgpu/0.21.0/burn_wgpu/)
- [WGPU 29 raw HAL access](https://docs.rs/wgpu/29.0.4/wgpu/struct.Device.html#method.as_hal)
