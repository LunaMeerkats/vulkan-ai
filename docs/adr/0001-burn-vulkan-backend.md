# ADR 0001: Start with Burn's Vulkan backend

- Status: Accepted
- Date: 2026-07-23

## Context

The project needs a research-friendly training loop on broadly available
Vulkan hardware. Direct Vulkan libraries provide excellent control over queues,
descriptors, synchronization, and shaders, but they do not normally provide
automatic differentiation, optimizers, model abstractions, or training-loop
infrastructure.

Burn already provides those training semantics and supports Vulkan through its
WGPU/CubeCL stack. Burn 0.21 also offers a Vulkan feature that uses its SPIR-V
compiler path. This lets the project validate real training workloads before
assuming the cost of a custom framework layer.

## Decision

Use Burn's Vulkan backend as the first execution path and keep core experiments
generic over Burn's backend traits. Use Burn's Flex backend for deterministic
tests that must run on hosts without a GPU.

Do not introduce direct Vulkan ownership or a second compute abstraction until
a concrete experiment demonstrates that Burn/CubeCL cannot express the needed
interop, profiling, synchronization, or custom-kernel behavior.

## Consequences

- Forward passes, gradients, and future optimizer work can share one model
  implementation across CPU and Vulkan.
- CI can validate mathematical behavior on CPU and compile the Vulkan path
  without assuming that hosted runners expose a GPU.
- Vulkan execution is mediated by Burn, WGPU, and CubeCL rather than direct
  command-buffer or descriptor management.
- A future direct-Vulkan extension must include an interoperability design and
  measurements showing why the additional complexity is justified.
