use burn::record::Record;
use std::{error::Error, fmt};

/// Checkpointable deterministic sampler for the probe's mini-batches.
#[derive(Record, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeededBatchSampler {
    generator_state: u64,
    current_permutation: Vec<usize>,
    next_position: usize,
}

/// Invalid sampler configuration or state recovered from a checkpoint.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SamplerError {
    BatchCount(usize),
    Position(usize),
    Permutation(Vec<usize>),
}

impl SeededBatchSampler {
    pub(crate) fn new(seed: u64, batch_count: usize) -> Result<Self, SamplerError> {
        if batch_count == 0 {
            return Err(SamplerError::BatchCount(batch_count));
        }

        Ok(Self {
            generator_state: seed,
            current_permutation: (0..batch_count).collect(),
            next_position: batch_count,
        })
    }

    pub(crate) fn next_batch(&mut self) -> Result<usize, SamplerError> {
        let batch_count = self.current_permutation.len();
        if self.next_position > batch_count {
            return Err(SamplerError::Position(self.next_position));
        }
        if !is_valid_permutation(&self.current_permutation) {
            return Err(SamplerError::Permutation(self.current_permutation.clone()));
        }
        if self.next_position == batch_count {
            self.current_permutation =
                seeded_epoch_permutation(&mut self.generator_state, batch_count);
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
            Self::BatchCount(batch_count) => {
                write!(
                    formatter,
                    "sampler batch count {batch_count} must be positive"
                )
            }
            Self::Position(position) => {
                write!(
                    formatter,
                    "sampler position {position} is outside its epoch"
                )
            }
            Self::Permutation(permutation) => {
                write!(formatter, "sampler permutation {permutation:?} is invalid")
            }
        }
    }
}

impl Error for SamplerError {}

fn seeded_epoch_permutation(generator_state: &mut u64, batch_count: usize) -> Vec<usize> {
    let mut permutation: Vec<_> = (0..batch_count).collect();

    // Forward Fisher-Yates preserves the original two-batch low-bit decision:
    // an odd SplitMix64 value swaps [0, 1], while an even value leaves it intact.
    for position in 0..batch_count.saturating_sub(1) {
        let remaining = batch_count - position;
        let swap_position = position + sample_bounded(generator_state, remaining);
        permutation.swap(position, swap_position);
    }

    permutation
}

fn sample_bounded(generator_state: &mut u64, upper_bound: usize) -> usize {
    let bound = u64::try_from(upper_bound).expect("batch count must fit in u64");
    let rejection_threshold = bound.wrapping_neg() % bound;

    loop {
        let value = splitmix64(generator_state);
        if value >= rejection_threshold {
            return usize::try_from(value % bound).expect("bounded sample must fit in usize");
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn is_valid_permutation(permutation: &[usize]) -> bool {
    !permutation.is_empty() && (0..permutation.len()).all(|batch| permutation.contains(&batch))
}

#[cfg(test)]
mod tests {
    use super::{SamplerError, SeededBatchSampler};

    const SEED: u64 = 0x5EED_CAFE_D15C_A11E;
    const CHECKPOINT_STEP: usize = 11;
    const TOTAL_STEPS: usize = 20;

    #[test]
    fn seeded_epochs_are_reproducible_and_checkpoint_sensitive() {
        let mut sampler = SeededBatchSampler::new(SEED, 2).unwrap();
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
    fn five_batch_vectors_survive_an_inside_epoch_resume() {
        const BATCH_COUNT: usize = 5;
        const CHECKPOINT_STEP: usize = 7;
        const TOTAL_STEPS: usize = 17;

        let mut sampler = SeededBatchSampler::new(SEED, BATCH_COUNT).unwrap();
        let before_checkpoint: Vec<_> = (0..CHECKPOINT_STEP)
            .map(|_| sampler.next_batch().unwrap())
            .collect();
        let checkpoint = sampler.clone();
        let uninterrupted_after_checkpoint: Vec<_> = (CHECKPOINT_STEP..TOTAL_STEPS)
            .map(|_| sampler.next_batch().unwrap())
            .collect();
        let mut resumed_sampler = checkpoint.clone();
        let resumed_after_checkpoint: Vec<_> = (CHECKPOINT_STEP..TOTAL_STEPS)
            .map(|_| resumed_sampler.next_batch().unwrap())
            .collect();
        let mut reset_generator_sampler = checkpoint.clone();
        reset_generator_sampler.generator_state = SEED;
        let reset_generator_after_checkpoint: Vec<_> = (CHECKPOINT_STEP..TOTAL_STEPS)
            .map(|_| reset_generator_sampler.next_batch().unwrap())
            .collect();
        let mut reset_permutation_sampler = checkpoint.clone();
        reset_permutation_sampler.current_permutation = (0..BATCH_COUNT).collect();
        let reset_permutation_next_batch = reset_permutation_sampler.next_batch().unwrap();

        assert_eq!(before_checkpoint, [3, 4, 2, 1, 0, 4, 2]);
        assert_eq!(checkpoint.current_permutation(), [4, 2, 3, 0, 1]);
        assert_eq!(checkpoint.next_position(), 2);
        assert_eq!(checkpoint.generator_state(), 0x50A9_98CA_CBB0_81C6);
        assert_eq!(
            uninterrupted_after_checkpoint,
            [3, 0, 1, 4, 1, 0, 3, 2, 0, 4]
        );
        assert_eq!(resumed_after_checkpoint, uninterrupted_after_checkpoint);
        assert_ne!(
            reset_generator_after_checkpoint,
            uninterrupted_after_checkpoint
        );
        assert_ne!(
            reset_permutation_next_batch,
            uninterrupted_after_checkpoint[0]
        );
        assert_eq!(resumed_sampler.current_permutation(), [0, 4, 1, 3, 2]);
        assert_eq!(resumed_sampler.next_position(), 2);
        assert_eq!(resumed_sampler.generator_state(), 0x4265_6696_C604_626E);
    }

    #[test]
    fn rejects_invalid_checkpoint_position() {
        let mut sampler = SeededBatchSampler::new(SEED, 2).unwrap();
        sampler.next_position = 3;

        assert_eq!(sampler.next_batch(), Err(SamplerError::Position(3)));
    }

    #[test]
    fn rejects_invalid_checkpoint_permutation() {
        let mut sampler = SeededBatchSampler::new(SEED, 2).unwrap();
        sampler.current_permutation = vec![0, 0];

        assert_eq!(
            sampler.next_batch(),
            Err(SamplerError::Permutation(vec![0, 0]))
        );
    }

    #[test]
    fn rejects_an_empty_batch_set() {
        assert_eq!(
            SeededBatchSampler::new(SEED, 0),
            Err(SamplerError::BatchCount(0))
        );
    }
}
