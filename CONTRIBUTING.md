# Contributing to Vulkan AI

Thanks for helping improve Vulkan AI. The project is pre-alpha, so small,
well-tested changes are especially valuable.

## Development workflow

1. Open an issue before work that changes public APIs or architecture.
2. Create a focused branch from `main`.
3. Keep commits atomic and describe the reason for each change.
4. Add or update tests and documentation with the implementation.
5. Open a pull request and complete the provided checklist.

## Local checks

Run the checks used by continuous integration:

```shell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --no-default-features --features vulkan --bin vulkan-ai-probe
```

The Vulkan test command compiles the GPU path and runs unit tests that do not
initialize a GPU. Running the probe itself requires a compatible device and
driver.

## Compatibility and releases

- `main` must remain releasable.
- Public releases follow Semantic Versioning.
- User-visible changes belong in `CHANGELOG.md` under `Unreleased`.
- Maintainers create annotated version tags and matching GitHub Releases.

Please keep discussions professional, specific, and focused on the technical
tradeoffs involved.
