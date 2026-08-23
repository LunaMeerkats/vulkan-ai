use burn::record::Record;
use std::{error::Error, fmt};

const MINI_BATCH_COUNT: usize = 2;

/// Checkpointable deterministic sampler for the probe's fixed mini-batches.
#[derive(Record, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeededBatchSampler {
    generator_state: u64,
    current_permutation: Vec<usize>,
    next_position: usize,
}

/// Invalid state recovered from a sampler checkpoint.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SamplerError {
    InvalidPosition(usize),
    InvalidPermutation(Vec<usize>),
}

impl SeededBatchSampler {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            generator_state: seed,
            current_permutation: (0..MINI_BATCH_COUNT).collect(),
            next_position: MINI_BATCH_COUNT,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Result<usize, SamplerError> {
        if self.next_position > MINI_BATCH_COUNT {
            return Err(SamplerError::InvalidPosition(self.next_position));
        }
        if !is_valid_permutation(&self.current_permutation) {
            return Err(SamplerError::InvalidPermutation(
                self.current_permutation.clone(),
            ));
        }
        if self.next_position == MINI_BATCH_COUNT {
            self.current_permutation = seeded_epoch_permutation(&mut self.generator_state);
            self.next_position = 0;
        }

        let batch_index = self.current_permutation[self.next_position];
        self.next_position += 1;
        Ok(batch_index)
    }

    pub(crate) fn generator_state(&self) -> u64 {
        self.generator_state
    }

    pub(crate) fn current_permutation(&self) -> &[usize] {
        &self.current_permutation
    }

    pub(crate) fn next_position(&self) -> usize {
        self.next_position
    }
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosition(position) => {
                write!(
                    formatter,
                    "sampler position {position} is outside its epoch"
                )
            }
            Self::InvalidPermutation(permutation) => {
                write!(formatter, "sampler permutation {permutation:?} is invalid")
            }
        }
    }
}

impl Error for SamplerError {}

fn seeded_epoch_permutation(generator_state: &mut u64) -> Vec<usize> {
    let mut permutation: Vec<_> = (0..MINI_BATCH_COUNT).collect();
    if splitmix64(generator_state) & 1 == 1 {
        permutation.swap(0, 1);
    }
    permutation
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn is_valid_permutation(permutation: &[usize]) -> bool {
    permutation.len() == MINI_BATCH_COUNT
        && (0..MINI_BATCH_COUNT).all(|batch| permutation.contains(&batch))
}

#[cfg(test)]
mod tests {
    use super::{SamplerError, SeededBatchSampler};

    const SEED: u64 = 0x5EED_CAFE_D15C_A11E;
    const CHECKPOINT_STEP: usize = 11;
    const TOTAL_STEPS: usize = 20;

    #[test]
    fn seeded_epochs_are_reproducible_and_checkpoint_sensitive() {
        let mut sampler = SeededBatchSampler::new(SEED);
        let before_checkpoint: Vec<_> = (0..CHECKPOINT_STEP)
            .map(|_| sampler.next_batch().unwrap())
            .collect();
        let mut resumed_sampler = sampler.clone();
        let after_checkpoint: Vec<_> = (CHECKPOINT_STEP..TOTAL_STEPS)
            .map(|_| resumed_sampler.next_batch().unwrap())
            .collect();
        let mut reset_generator_sampler = sampler.clone();
        reset_generator_sampler.generator_state = SEED;
        let reset_generator_second_half: Vec<_> = (CHECKPOINT_STEP..TOTAL_STEPS)
            .map(|_| reset_generator_sampler.next_batch().unwrap())
            .collect();
        let mut reset_permutation_sampler = sampler.clone();
        reset_permutation_sampler.current_permutation = vec![0, 1];
        let reset_permutation_next_batch = reset_permutation_sampler.next_batch().unwrap();

        assert_eq!(before_checkpoint, [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]);
        assert_eq!(sampler.next_position(), 1);
        assert_eq!(sampler.current_permutation(), [1, 0]);
        assert_eq!(after_checkpoint, [0, 0, 1, 1, 0, 0, 1, 0, 1]);
        assert_ne!(after_checkpoint, reset_generator_second_half);
        assert_ne!(after_checkpoint[0], reset_permutation_next_batch);
    }

    #[test]
    fn rejects_invalid_checkpoint_position() {
        let mut sampler = SeededBatchSampler::new(SEED);
        sampler.next_position = 3;

        assert_eq!(sampler.next_batch(), Err(SamplerError::InvalidPosition(3)));
    }

    #[test]
    fn rejects_invalid_checkpoint_permutation() {
        let mut sampler = SeededBatchSampler::new(SEED);
        sampler.current_permutation = vec![0, 0];

        assert_eq!(
            sampler.next_batch(),
            Err(SamplerError::InvalidPermutation(vec![0, 0]))
        );
    }
}
