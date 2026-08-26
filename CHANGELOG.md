# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial Rust project structure.
- Backend-independent autodiff compatibility probe.
- Explicit Burn Vulkan/SPIR-V probe binary.
- CPU-based test coverage and continuous integration checks.
- Architecture decision record for the initial Burn/Vulkan backend strategy.
- Vulkan adapter, driver, subgroup, and compute-limit reporting.
- CPU/Vulkan prediction, loss, and parameter-gradient parity checks for a
  deterministic linear model.
- Reproducible synchronized Vulkan timing reports with raw JSON samples,
  summary statistics, runtime settings, and optional Burn fusion.
- Backend-generic custom `x² + x` operation with an explicit `2x + 1`
  backward rule and CPU/Vulkan parity coverage.
- Architecture decision record defining the staged boundary between Burn
  primitives, CubeCL custom kernels, and direct Vulkan interoperability.
- CubeCL forward kernel and fusion-stream integration for the custom quadratic
  operation, with a portable Burn reference, non-contiguous layout parity, and
  synchronized kernel-versus-reference benchmark reporting.
- Reproducible custom-kernel size sweep with raw wall-clock samples, CubeCL
  device profiles for the unfused backend, and explicit fused-profile
  limitations for separating execution from managed runtime overhead.
- Balanced unfused/fused quadratic autodiff size sweep comparing the custom
  forward and explicit backward rule with Burn's portable primitive graph,
  including preallocated inputs, raw synchronized samples, and median ratios.
- Deterministic 20-step full-batch SGD probe with loss-reduction checks and
  CPU/Vulkan parity for the complete loss trajectory and final parameters.
- Full-precision model and stateful momentum-SGD checkpoint round trip after
  step 10, with uninterrupted/resumed and CPU/Vulkan trajectory and parameter
  parity checks.
- Deterministic two-layer tanh training for a nonlinear product target, with
  full-parameter CPU/Vulkan and serialized checkpoint/resume parity.
- Deterministic two-batch nonlinear training with checkpointed data-position
  state, consumed-order checks, full-dataset loss trajectories, and
  uninterrupted/resumed CPU/Vulkan parity.
- Fixed-seed epoch permutations with checkpointed generator, current-order,
  and epoch-position state plus explicit generator-reset detection.
- Inside-epoch mini-batch checkpoint/resume coverage that proves the current
  permutation and cursor both survive serialization.
- Internal seeded batch-sampler boundary that separates checkpointable data
  ordering from model and optimizer execution without changing the fixed run.
- Deterministic multi-batch forward Fisher-Yates shuffle protocol with unbiased
  bounded sampling, fixed five-batch vectors, and inside-epoch resume coverage
  while preserving the existing two-batch training order and generator states.
- Five-batch nonlinear product training fixture with deterministic data-order,
  full-dataset loss, serialized mid-epoch resume, and CPU/Vulkan parameter parity.
- Backend-independent in-memory dataset boundary for the fixed five-batch
  inputs, targets, batch lookup, and full-dataset evaluation source.

### Fixed

- Custom quadratic autodiff forward execution and checkpoint recomputation now
  delegate to the backend-specific implementation, so Vulkan training and
  parity checks exercise the CubeCL kernel instead of the portable reference.

[Unreleased]: https://github.com/LunaMeerkats/vulkan-ai/commits/main
