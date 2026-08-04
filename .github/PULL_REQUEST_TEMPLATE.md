## Summary

Describe what changed and why.

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test --all-targets`
- [ ] `cargo test --features vulkan --bin vulkan-ai-probe`
- [ ] `cargo test --features vulkan-fusion --bin vulkan-ai-probe`
- [ ] Vulkan-specific behavior was tested or clearly marked as compile-only
- [ ] Custom kernels retain a reference path, parity coverage, and a synchronized benchmark
- [ ] Optimizer/model/data-order changes preserve uninterrupted, checkpoint-resumed, and CPU/Vulkan parity
- [ ] `CHANGELOG.md` was updated when the change is user-visible
