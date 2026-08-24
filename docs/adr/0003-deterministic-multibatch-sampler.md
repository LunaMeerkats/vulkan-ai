# ADR 0003: Specify deterministic multi-batch shuffle and checkpoint state

- Status: Accepted
- Date: 2026-08-25

## Context

The checkpoint probe currently splits its fixed nonlinear dataset into two
mini-batches. Its sampler uses one SplitMix64 output bit to decide whether each
two-item epoch is swapped. That is sufficient evidence for an inside-epoch
checkpoint, but it does not define how three or more batches are shuffled.

Data order affects every optimizer update, loss value, and parameter. Relying on
a library shuffle would also make the probe's persisted state sensitive to that
library's algorithm or version. The ordering protocol must therefore be fixed
before the sampler is connected to a representative multi-batch training
fixture or an external data source.

## Decision

Use SplitMix64 with the existing seed `0x5eedcafed15ca11e`. Create each epoch as
the identity batch identifiers `0..batch_count`, then apply forward Fisher-Yates:

1. Visit positions from zero through `batch_count - 2`.
2. Let `bound` be the number of positions remaining, including the current one.
3. Draw a SplitMix64 value. Set `threshold` to `(-bound mod 2^64) mod bound` and
   redraw while the value is below that threshold.
4. Swap the current position with `position + (value mod bound)`.

The rejection step avoids modulo bias without depending on a random-number
library. Forward iteration deliberately preserves the original two-batch rule:
an odd value swaps `[0, 1]`, while an even value leaves it unchanged.

Checkpoint state consists of exactly:

- the next SplitMix64 generator state;
- the complete current epoch permutation; and
- the position of the next batch in that permutation.

The batch count is derived from the permutation length. A valid state contains
at least one batch, contains every identifier from zero through `batch_count - 1`
exactly once, and has a next position no greater than `batch_count`. A new
permutation is generated only when that position equals `batch_count`.

The conformance fixture uses five batches and the same seed. Its first three
epoch permutations are:

```text
[3, 4, 2, 1, 0]
[4, 2, 3, 0, 1]
[4, 1, 0, 3, 2]
```

After seven batches, an inside-epoch checkpoint must contain permutation
`[4, 2, 3, 0, 1]`, next position `2`, and generator state
`0x50a998cacbb081c6`. Resuming that state must reproduce the uninterrupted
sequence.

This decision generalizes only the internal sampler protocol. The training
probe remains on its established two-batch fixture until a separate change
extends the model-level CPU/Vulkan and checkpoint-resume evidence.

## Consequences

- Shuffle and resume behavior is reproducible across supported platforms and
  independent of third-party RNG or shuffle implementations.
- The existing two-batch consumed order, checkpoint fields, and SplitMix64
  states remain unchanged.
- Rejection sampling can consume more than one generator value for a swap, so
  the generator state is part of the compatibility contract and must always be
  checkpointed.
- Future batch-count changes must update conformance evidence deliberately;
  they cannot silently substitute another shuffle algorithm.
- External data loading, worker concurrency, and distributed sampling remain
  out of scope and require separate evidence and design.

## References

- [README checkpoint protocol](../../README.md)
- [Contributor mini-batch protocol](../../CONTRIBUTING.md)
