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

[Unreleased]: https://github.com/LunaMeerkats/vulkan-ai/commits/main
