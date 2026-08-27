#![recursion_limit = "256"]

//! Small, backend-independent checks for Vulkan AI research workflows.

mod data;
mod ops;

use data::{
    FIXED_PRODUCT_DATASET, InMemoryProductDataset, ProductBatchInputs, ProductBatchTargets,
    SampledDatasetCursor, SamplerError,
};

use burn::{
    module::{AutodiffModule, Module, Param},
    nn::{
        Linear,
        loss::{MseLoss, Reduction},
    },
    optim::{GradientsParams, Optimizer, SgdConfig, momentum::MomentumConfig},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use std::{error::Error, fmt};

pub use ops::{CustomOpsBackend, quadratic, quadratic_reference};

/// Values produced by a forward pass and its gradient calculation.
#[derive(Debug, PartialEq)]
pub struct ProbeResult {
    /// Result of multiplying the input matrix by the trainable weights.
    pub output: Vec<f32>,
    /// Gradient of the summed output with respect to the weights.
    pub weight_gradient: Vec<f32>,
}

/// Values produced by a deterministic linear-model training step.
#[derive(Debug, PartialEq)]
pub struct TrainingProbeResult {
    /// Predictions from the linear model before an optimizer update.
    pub predictions: Vec<f32>,
    /// Mean squared error for the training batch.
    pub loss: f32,
    /// Gradient of the loss with respect to the linear weights.
    pub weight_gradient: Vec<f32>,
    /// Gradient of the loss with respect to the linear bias.
    pub bias_gradient: Vec<f32>,
}

/// Values produced by deterministic multi-step optimizer training.
#[derive(Debug, PartialEq)]
pub struct OptimizerProbeResult {
    /// Mean squared error before training and after every optimizer step.
    pub losses: Vec<f32>,
    /// Linear weights after the final optimizer step.
    pub final_weights: Vec<f32>,
    /// Linear bias after the final optimizer step.
    pub final_bias: Vec<f32>,
}

/// Values produced by uninterrupted and checkpoint-resumed optimizer training.
#[derive(Debug, PartialEq)]
pub struct OptimizerCheckpointProbeResult {
    /// Result from training straight through without interruption.
    pub uninterrupted: OptimizerProbeResult,
    /// Result from serializing and restoring the model and optimizer mid-run.
    pub resumed: OptimizerProbeResult,
    /// Size of the full-precision named `MessagePack` model checkpoint.
    pub model_checkpoint_bytes: usize,
    /// Size of the full-precision named `MessagePack` optimizer checkpoint.
    pub optimizer_checkpoint_bytes: usize,
}

/// Values produced by deterministic multi-step nonlinear-model training.
#[derive(Debug, PartialEq)]
pub struct NonlinearOptimizerProbeResult {
    /// Mean squared error before training and after every optimizer step.
    pub losses: Vec<f32>,
    /// Final hidden weights, hidden bias, output weights, and output bias in that order.
    pub final_parameters: Vec<f32>,
}

/// Values produced by uninterrupted and checkpoint-resumed nonlinear training.
#[derive(Debug, PartialEq)]
pub struct NonlinearCheckpointProbeResult {
    /// Result from training straight through without interruption.
    pub uninterrupted: NonlinearOptimizerProbeResult,
    /// Result from serializing and restoring the model and optimizer mid-run.
    pub resumed: NonlinearOptimizerProbeResult,
    /// Size of the full-precision named `MessagePack` model checkpoint.
    pub model_checkpoint_bytes: usize,
    /// Size of the full-precision named `MessagePack` optimizer checkpoint.
    pub optimizer_checkpoint_bytes: usize,
}

/// Values produced by deterministic nonlinear mini-batch training.
#[derive(Debug, PartialEq)]
pub struct MiniBatchOptimizerProbeResult {
    /// Full-dataset mean squared error before training and after every optimizer step.
    pub losses: Vec<f32>,
    /// Mini-batch identifiers consumed by each optimizer step.
    pub batch_sequence: Vec<usize>,
    /// Final hidden weights, hidden bias, output weights, and output bias in that order.
    pub final_parameters: Vec<f32>,
    /// Position of the next mini-batch in the current epoch permutation.
    pub final_data_position: usize,
    /// Deterministic generator state after the final epoch permutation was created.
    pub final_generator_state: u64,
}

/// Values produced by uninterrupted and checkpoint-resumed mini-batch training.
#[derive(Debug, PartialEq)]
pub struct MiniBatchCheckpointProbeResult {
    /// Result from training straight through without interruption.
    pub uninterrupted: MiniBatchOptimizerProbeResult,
    /// Result from restoring model, optimizer, and sampler state mid-run.
    pub resumed: MiniBatchOptimizerProbeResult,
    /// Size of the full-precision named `MessagePack` model checkpoint.
    pub model_checkpoint_bytes: usize,
    /// Size of the full-precision named `MessagePack` optimizer checkpoint.
    pub optimizer_checkpoint_bytes: usize,
    /// Size of the named `MessagePack` sampler-state checkpoint.
    pub data_checkpoint_bytes: usize,
    /// Epoch position restored after the checkpoint.
    pub checkpoint_data_position: usize,
    /// Current epoch permutation restored after the checkpoint.
    pub checkpoint_epoch_permutation: Vec<usize>,
    /// Seeded permutation generator state restored after the checkpoint.
    pub checkpoint_generator_state: u64,
}

#[derive(Module, Debug)]
struct NonlinearModel<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> NonlinearModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.output.forward(self.hidden.forward(input).tanh())
    }
}

trait OptimizerProbeModel<B: AutodiffBackend>: AutodiffModule<B> {
    fn probe_forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

impl<B: AutodiffBackend> OptimizerProbeModel<B> for Linear<B> {
    fn probe_forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

impl<B: AutodiffBackend> OptimizerProbeModel<B> for NonlinearModel<B> {
    fn probe_forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

struct CheckpointProbeExecution<R> {
    uninterrupted: R,
    resumed: R,
    model_checkpoint_bytes: usize,
    optimizer_checkpoint_bytes: usize,
}

/// Number of full-batch SGD updates used by [`run_optimizer_probe`].
pub const OPTIMIZER_PROBE_STEPS: usize = 20;
/// Learning rate used by [`run_optimizer_probe`].
pub const OPTIMIZER_PROBE_LEARNING_RATE: f64 = 0.05;
/// Update after which [`run_optimizer_checkpoint_probe`] saves and restores state.
pub const OPTIMIZER_CHECKPOINT_STEP: usize = OPTIMIZER_PROBE_STEPS / 2;
/// Update after which [`run_minibatch_checkpoint_probe`] saves sampler state.
///
/// The odd step deliberately places the checkpoint inside a five-batch epoch so
/// restoring only the generator, permutation, or epoch position cannot
/// reproduce the run.
pub const MINI_BATCH_CHECKPOINT_STEP: usize = OPTIMIZER_CHECKPOINT_STEP + 1;
/// Momentum factor used by the stateful checkpoint/resume probe.
pub const OPTIMIZER_CHECKPOINT_MOMENTUM: f64 = 0.9;
/// Dampening factor used by the stateful checkpoint/resume probe.
pub const OPTIMIZER_CHECKPOINT_DAMPENING: f64 = 0.1;
/// Fixed seed used to generate deterministic mini-batch permutations for each epoch.
pub const MINI_BATCH_EPOCH_SEED: u64 = 0x5EED_CAFE_D15C_A11E;

/// Values produced by the custom quadratic operation and its backward rule.
#[derive(Debug, PartialEq)]
pub struct CustomOpProbeResult {
    /// Result of applying `x² + x` element-wise.
    pub output: Vec<f32>,
    /// Gradient of the summed output with respect to the input.
    pub input_gradient: Vec<f32>,
}

/// Forward implementation used by the quadratic autodiff benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadraticTrainingPath {
    /// Use the backend-specific custom operation and its explicit backward rule.
    Custom,
    /// Compose portable Burn primitives and let Burn derive their backward graph.
    Reference,
}

/// Error returned when a backend cannot complete the probe.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// The autodiff backend did not return the requested weight gradient.
    MissingGradient,
    /// The autodiff backend did not return the requested bias gradient.
    MissingBiasGradient,
    /// The deterministic linear model did not contain its expected bias parameter.
    MissingBiasParameter,
    /// A layer in the deterministic nonlinear model did not contain its expected bias.
    MissingNonlinearBiasParameter(&'static str),
    /// The autodiff backend did not return the custom operation's input gradient.
    MissingInputGradient,
    /// Tensor data could not be converted to the expected `f32` values.
    DataConversion(String),
    /// Model or optimizer state could not be serialized or restored.
    CheckpointSerialization(String),
    /// The mini-batch sampler was configured without any batches.
    InvalidMiniBatchCount(usize),
    /// Restored mini-batch state does not describe the paired dataset.
    MiniBatchDatasetMismatch {
        /// Number of batches exposed by the dataset.
        dataset: usize,
        /// Number of batches encoded by the sampler permutation.
        sampler: usize,
    },
    /// Restored mini-batch state did not identify a valid epoch position.
    InvalidMiniBatchPosition(usize),
    /// Restored mini-batch state did not contain each batch identifier exactly once.
    InvalidMiniBatchPermutation(Vec<usize>),
    /// The backend could not synchronize a measured workload.
    Synchronization(String),
    /// `CubeCL` could not profile a measured workload consistently.
    Profiling(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGradient => formatter.write_str("the required weight gradient is missing"),
            Self::MissingBiasGradient => {
                formatter.write_str("the required bias gradient is missing")
            }
            Self::MissingBiasParameter => {
                formatter.write_str("the deterministic linear model bias is missing")
            }
            Self::MissingNonlinearBiasParameter(layer) => {
                write!(
                    formatter,
                    "the deterministic nonlinear model {layer} bias is missing"
                )
            }
            Self::MissingInputGradient => {
                formatter.write_str("the custom operation input gradient is missing")
            }
            Self::DataConversion(message) => {
                write!(formatter, "could not read probe tensor data: {message}")
            }
            Self::CheckpointSerialization(message) => {
                write!(
                    formatter,
                    "could not round-trip checkpoint state: {message}"
                )
            }
            Self::InvalidMiniBatchCount(batch_count) => {
                write!(formatter, "mini-batch count {batch_count} must be positive")
            }
            Self::MiniBatchDatasetMismatch { dataset, sampler } => {
                write!(
                    formatter,
                    "mini-batch sampler count {sampler} does not match dataset count {dataset}"
                )
            }
            Self::InvalidMiniBatchPosition(position) => {
                write!(
                    formatter,
                    "mini-batch data position {position} is outside the current epoch permutation"
                )
            }
            Self::InvalidMiniBatchPermutation(permutation) => {
                write!(
                    formatter,
                    "mini-batch epoch permutation {permutation:?} does not contain every batch exactly once"
                )
            }
            Self::Synchronization(message) => {
                write!(
                    formatter,
                    "could not synchronize the backend workload: {message}"
                )
            }
            Self::Profiling(message) => {
                write!(
                    formatter,
                    "could not profile the backend workload: {message}"
                )
            }
        }
    }
}

/// Run the custom quadratic operation and its explicit backward rule.
///
/// The fixed transposed input exercises negative, zero, and positive
/// derivatives plus a non-contiguous two-dimensional layout while remaining
/// directly comparable across backends.
///
/// # Errors
///
/// Returns [`ProbeError`] if the backend omits the input gradient or if tensor
/// data cannot be converted to `f32` values.
pub fn run_custom_op_probe<B>(device: &B::Device) -> Result<CustomOpProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32> + CustomOpsBackend,
{
    let input = Tensor::<B, 2>::from_floats([[-2.0, 0.0], [-0.5, 1.5]], device)
        .swap_dims(0, 1)
        .require_grad();
    let output = quadratic(input.clone());
    let gradients = output.clone().sum().backward();
    let input_gradient = input
        .grad(&gradients)
        .ok_or(ProbeError::MissingInputGradient)?;

    Ok(CustomOpProbeResult {
        output: output
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
        input_gradient: input_gradient
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
    })
}

impl Error for ProbeError {}

impl From<SamplerError> for ProbeError {
    fn from(error: SamplerError) -> Self {
        match error {
            SamplerError::BatchCount(batch_count) => Self::InvalidMiniBatchCount(batch_count),
            SamplerError::DatasetBatchCount { dataset, sampler } => {
                Self::MiniBatchDatasetMismatch { dataset, sampler }
            }
            SamplerError::Position(position) => Self::InvalidMiniBatchPosition(position),
            SamplerError::Permutation(permutation) => {
                Self::InvalidMiniBatchPermutation(permutation)
            }
        }
    }
}

struct TrainingProbeTensors<B>
where
    B: AutodiffBackend,
{
    predictions: Tensor<B, 2>,
    loss: Tensor<B, 1>,
    weight_gradient: Tensor<B::InnerBackend, 2>,
    bias_gradient: Tensor<B::InnerBackend, 1>,
}

/// Run a deterministic matrix multiplication and backward pass.
///
/// The calculation is intentionally tiny. Its purpose is to verify that a
/// backend can allocate tensors, execute compute work, and return gradients.
///
/// # Errors
///
/// Returns [`ProbeError`] if the backend omits the requested gradient or if
/// its tensor data cannot be read as `f32` values.
pub fn run_autodiff_probe<B>(device: &B::Device) -> Result<ProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let input = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], device);
    let weights = Tensor::<B, 2>::from_floats([[2.0], [3.0]], device).require_grad();

    let output = input.matmul(weights.clone());
    let gradients = output.clone().sum().backward();
    let weight_gradient = weights
        .grad(&gradients)
        .ok_or(ProbeError::MissingGradient)?;

    Ok(ProbeResult {
        output: output
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
        weight_gradient: weight_gradient
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
    })
}

/// Run a deterministic forward and backward pass through a trainable linear model.
///
/// The fixed two-input, one-output model and three-example batch make results
/// directly comparable across backends without relying on random initialization.
///
/// # Errors
///
/// Returns [`ProbeError`] if the backend omits a parameter gradient or if its
/// tensor data cannot be read as `f32` values.
pub fn run_training_probe<B>(device: &B::Device) -> Result<TrainingProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let result = execute_training_probe::<B>(device)?;

    let loss_values = result
        .loss
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| ProbeError::DataConversion(error.to_string()))?;
    let loss = loss_values
        .first()
        .copied()
        .ok_or_else(|| ProbeError::DataConversion("loss tensor was empty".to_owned()))?;

    Ok(TrainingProbeResult {
        predictions: result
            .predictions
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
        loss,
        weight_gradient: result
            .weight_gradient
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
        bias_gradient: result
            .bias_gradient
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| ProbeError::DataConversion(error.to_string()))?,
    })
}

/// Run deterministic full-batch SGD updates through a trainable linear model.
///
/// The fixed model, batch, learning rate, and update count make the complete
/// loss trajectory and final parameters directly comparable across backends.
/// The returned losses contain the initial loss followed by the loss after
/// each optimizer step.
///
/// # Errors
///
/// Returns [`ProbeError`] if the model is missing its bias or tensor data
/// cannot be converted to `f32` values.
pub fn run_optimizer_probe<B>(device: &B::Device) -> Result<OptimizerProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let (input, target) = training_batch::<B>(device);
    let model = training_model::<B>(device);
    let mut optimizer = SgdConfig::new().init::<B, Linear<B>>();
    let mut losses = Vec::with_capacity(OPTIMIZER_PROBE_STEPS + 1);
    let model = run_optimizer_updates(
        model,
        &mut optimizer,
        &input,
        &target,
        OPTIMIZER_PROBE_STEPS,
        &mut losses,
    )?;

    optimizer_probe_result(&model, losses)
}

/// Compare uninterrupted stateful SGD with a serialized checkpoint/resume run.
///
/// Both paths use the fixed model, batch, learning rate, update count, and
/// momentum settings. The resumed path records full-precision named
/// `MessagePack` bytes for the model and optimizer after
/// [`OPTIMIZER_CHECKPOINT_STEP`] updates, restores both into fresh instances,
/// and completes the remaining updates.
///
/// # Errors
///
/// Returns [`ProbeError`] if tensor data cannot be read, the model is missing
/// its bias, or either checkpoint cannot be serialized or restored.
pub fn run_optimizer_checkpoint_probe<B>(
    device: &B::Device,
) -> Result<OptimizerCheckpointProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let (input, target) = training_batch::<B>(device);
    let execution = execute_checkpoint_probe(
        device,
        &input,
        &target,
        training_model::<B>,
        optimizer_probe_result,
    )?;

    Ok(OptimizerCheckpointProbeResult {
        uninterrupted: execution.uninterrupted,
        resumed: execution.resumed,
        model_checkpoint_bytes: execution.model_checkpoint_bytes,
        optimizer_checkpoint_bytes: execution.optimizer_checkpoint_bytes,
    })
}

/// Compare uninterrupted nonlinear training with a serialized checkpoint/resume run.
///
/// The fixed two-layer tanh model learns a product target that a single linear
/// layer cannot represent. Both paths reuse the established 20-step stateful
/// SGD protocol, checkpoint after step 10, and record the complete loss
/// trajectory plus every trainable parameter.
///
/// # Errors
///
/// Returns [`ProbeError`] if tensor data cannot be read, a layer is missing its
/// bias, or either checkpoint cannot be serialized or restored.
pub fn run_nonlinear_checkpoint_probe<B>(
    device: &B::Device,
) -> Result<NonlinearCheckpointProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let (input, target) = nonlinear_training_batch::<B>(device);
    let execution = execute_checkpoint_probe(
        device,
        &input,
        &target,
        nonlinear_training_model::<B>,
        nonlinear_optimizer_probe_result,
    )?;

    Ok(NonlinearCheckpointProbeResult {
        uninterrupted: execution.uninterrupted,
        resumed: execution.resumed,
        model_checkpoint_bytes: execution.model_checkpoint_bytes,
        optimizer_checkpoint_bytes: execution.optimizer_checkpoint_bytes,
    })
}

/// Compare uninterrupted mini-batch training with a complete checkpoint/resume run.
///
/// The nonlinear product dataset is a fixed `5 x 2` input grid split into five
/// two-example batches. A fixed seed drives a new permutation for every epoch.
/// The checkpoint after step 11 records the generator state, current permutation,
/// and next position from inside an epoch alongside the model and optimizer. The
/// resumed path compares the full-dataset loss trajectory, consumed batch
/// sequence, final parameters, sampler state, and generator state with
/// uninterrupted training.
///
/// # Errors
///
/// Returns [`ProbeError`] if tensor data cannot be read, a layer is missing its
/// bias, checkpoint state cannot be serialized or restored, or restored sampler
/// state is invalid.
pub fn run_minibatch_checkpoint_probe<B>(
    device: &B::Device,
) -> Result<MiniBatchCheckpointProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let dataset = &FIXED_PRODUCT_DATASET;
    let (evaluation_input, evaluation_target) = multibatch_training_dataset::<B>(dataset, device);

    let mut uninterrupted_optimizer = checkpoint_optimizer_config().init::<B, NonlinearModel<B>>();
    let mut uninterrupted_cursor = SampledDatasetCursor::new(dataset, MINI_BATCH_EPOCH_SEED)?;
    let mut uninterrupted_losses = Vec::with_capacity(OPTIMIZER_PROBE_STEPS + 1);
    let mut uninterrupted_batches = Vec::with_capacity(OPTIMIZER_PROBE_STEPS);
    let uninterrupted_model = run_minibatch_optimizer_updates(
        nonlinear_training_model::<B>(device),
        &mut uninterrupted_optimizer,
        &mut uninterrupted_cursor,
        &evaluation_input,
        &evaluation_target,
        device,
        OPTIMIZER_PROBE_STEPS,
        &mut uninterrupted_losses,
        &mut uninterrupted_batches,
    )?;
    let uninterrupted = minibatch_optimizer_probe_result(
        &uninterrupted_model,
        uninterrupted_losses,
        uninterrupted_batches,
        uninterrupted_cursor.sampler().next_position(),
        uninterrupted_cursor.sampler().generator_state(),
    )?;

    let mut resumed_optimizer = checkpoint_optimizer_config().init::<B, NonlinearModel<B>>();
    let mut resumed_cursor = SampledDatasetCursor::new(dataset, MINI_BATCH_EPOCH_SEED)?;
    let mut resumed_losses = Vec::with_capacity(OPTIMIZER_PROBE_STEPS + 1);
    let mut resumed_batches = Vec::with_capacity(OPTIMIZER_PROBE_STEPS);
    let resumed_model = run_minibatch_optimizer_updates(
        nonlinear_training_model::<B>(device),
        &mut resumed_optimizer,
        &mut resumed_cursor,
        &evaluation_input,
        &evaluation_target,
        device,
        MINI_BATCH_CHECKPOINT_STEP,
        &mut resumed_losses,
        &mut resumed_batches,
    )?;

    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let model_checkpoint =
        Recorder::<B>::record(&recorder, resumed_model.clone().into_record(), ())
            .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let optimizer_checkpoint = Recorder::<B>::record(&recorder, resumed_optimizer.to_record(), ())
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let data_checkpoint = Recorder::<B>::record(&recorder, resumed_cursor.sampler().clone(), ())
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let model_checkpoint_bytes = model_checkpoint.len();
    let optimizer_checkpoint_bytes = optimizer_checkpoint.len();
    let data_checkpoint_bytes = data_checkpoint.len();

    let restored_model_record = Recorder::<B>::load(&recorder, model_checkpoint, device)
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let restored_model = nonlinear_training_model::<B>(device).load_record(restored_model_record);
    let restored_optimizer_record = Recorder::<B>::load(&recorder, optimizer_checkpoint, device)
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let mut restored_optimizer = checkpoint_optimizer_config()
        .init::<B, NonlinearModel<B>>()
        .load_record(restored_optimizer_record);
    let restored_sampler = Recorder::<B>::load(&recorder, data_checkpoint, device)
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let mut restored_cursor = SampledDatasetCursor::from_sampler(dataset, restored_sampler)?;
    let checkpoint_data_position = restored_cursor.sampler().next_position();
    let checkpoint_epoch_permutation = restored_cursor.sampler().current_permutation().to_vec();
    let checkpoint_generator_state = restored_cursor.sampler().generator_state();

    let resumed_model = run_minibatch_optimizer_updates(
        restored_model,
        &mut restored_optimizer,
        &mut restored_cursor,
        &evaluation_input,
        &evaluation_target,
        device,
        OPTIMIZER_PROBE_STEPS - MINI_BATCH_CHECKPOINT_STEP,
        &mut resumed_losses,
        &mut resumed_batches,
    )?;
    let resumed = minibatch_optimizer_probe_result(
        &resumed_model,
        resumed_losses,
        resumed_batches,
        restored_cursor.sampler().next_position(),
        restored_cursor.sampler().generator_state(),
    )?;

    Ok(MiniBatchCheckpointProbeResult {
        uninterrupted,
        resumed,
        model_checkpoint_bytes,
        optimizer_checkpoint_bytes,
        data_checkpoint_bytes,
        checkpoint_data_position,
        checkpoint_epoch_permutation,
        checkpoint_generator_state,
    })
}

fn execute_checkpoint_probe<B, M, R, F, G>(
    device: &B::Device,
    input: &Tensor<B, 2>,
    target: &Tensor<B, 2>,
    model_factory: F,
    result_factory: G,
) -> Result<CheckpointProbeExecution<R>, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
    M: OptimizerProbeModel<B>,
    F: Fn(&B::Device) -> M,
    G: Fn(&M, Vec<f32>) -> Result<R, ProbeError>,
{
    let mut uninterrupted_optimizer = checkpoint_optimizer_config().init::<B, M>();
    let mut uninterrupted_losses = Vec::with_capacity(OPTIMIZER_PROBE_STEPS + 1);
    let uninterrupted_model = run_optimizer_updates(
        model_factory(device),
        &mut uninterrupted_optimizer,
        input,
        target,
        OPTIMIZER_PROBE_STEPS,
        &mut uninterrupted_losses,
    )?;
    let uninterrupted = result_factory(&uninterrupted_model, uninterrupted_losses)?;

    let mut resumed_optimizer = checkpoint_optimizer_config().init::<B, M>();
    let mut resumed_losses = Vec::with_capacity(OPTIMIZER_PROBE_STEPS + 1);
    let resumed_model = run_optimizer_updates(
        model_factory(device),
        &mut resumed_optimizer,
        input,
        target,
        OPTIMIZER_CHECKPOINT_STEP,
        &mut resumed_losses,
    )?;

    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let model_checkpoint =
        Recorder::<B>::record(&recorder, resumed_model.clone().into_record(), ())
            .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let optimizer_checkpoint = Recorder::<B>::record(&recorder, resumed_optimizer.to_record(), ())
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let model_checkpoint_bytes = model_checkpoint.len();
    let optimizer_checkpoint_bytes = optimizer_checkpoint.len();

    let restored_model_record = Recorder::<B>::load(&recorder, model_checkpoint, device)
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let restored_model = model_factory(device).load_record(restored_model_record);
    let restored_optimizer_record = Recorder::<B>::load(&recorder, optimizer_checkpoint, device)
        .map_err(|error| ProbeError::CheckpointSerialization(error.to_string()))?;
    let mut restored_optimizer = checkpoint_optimizer_config()
        .init::<B, M>()
        .load_record(restored_optimizer_record);

    let resumed_model = run_optimizer_updates(
        restored_model,
        &mut restored_optimizer,
        input,
        target,
        OPTIMIZER_PROBE_STEPS - OPTIMIZER_CHECKPOINT_STEP,
        &mut resumed_losses,
    )?;
    let resumed = result_factory(&resumed_model, resumed_losses)?;

    Ok(CheckpointProbeExecution {
        uninterrupted,
        resumed,
        model_checkpoint_bytes,
        optimizer_checkpoint_bytes,
    })
}

fn checkpoint_optimizer_config() -> SgdConfig {
    SgdConfig::new().with_momentum(Some(MomentumConfig {
        momentum: OPTIMIZER_CHECKPOINT_MOMENTUM,
        dampening: OPTIMIZER_CHECKPOINT_DAMPENING,
        nesterov: false,
    }))
}

fn run_optimizer_updates<B, M, O>(
    mut model: M,
    optimizer: &mut O,
    input: &Tensor<B, 2>,
    target: &Tensor<B, 2>,
    steps: usize,
    losses: &mut Vec<f32>,
) -> Result<M, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
    M: OptimizerProbeModel<B>,
    O: Optimizer<M, B>,
{
    let mut loss = MseLoss::new().forward(
        model.probe_forward(input.clone()),
        target.clone(),
        Reduction::Mean,
    );
    if losses.is_empty() {
        losses.push(read_loss(loss.clone())?);
    }

    for _ in 0..steps {
        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(OPTIMIZER_PROBE_LEARNING_RATE, model, gradients);
        loss = MseLoss::new().forward(
            model.probe_forward(input.clone()),
            target.clone(),
            Reduction::Mean,
        );
        losses.push(read_loss(loss.clone())?);
    }

    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn run_minibatch_optimizer_updates<B, O>(
    mut model: NonlinearModel<B>,
    optimizer: &mut O,
    cursor: &mut SampledDatasetCursor<'_>,
    evaluation_input: &Tensor<B, 2>,
    evaluation_target: &Tensor<B, 2>,
    device: &B::Device,
    steps: usize,
    losses: &mut Vec<f32>,
    batch_sequence: &mut Vec<usize>,
) -> Result<NonlinearModel<B>, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
    O: Optimizer<NonlinearModel<B>, B>,
{
    if losses.is_empty() {
        losses.push(read_loss(MseLoss::new().forward(
            model.forward(evaluation_input.clone()),
            evaluation_target.clone(),
            Reduction::Mean,
        ))?);
    }

    for _ in 0..steps {
        let (batch_index, inputs, targets) = cursor.next_batch()?;
        let (input, target) = nonlinear_mini_batch::<B>(inputs, targets, device);
        let loss = MseLoss::new().forward(model.forward(input), target, Reduction::Mean);
        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(OPTIMIZER_PROBE_LEARNING_RATE, model, gradients);
        batch_sequence.push(batch_index);

        losses.push(read_loss(MseLoss::new().forward(
            model.forward(evaluation_input.clone()),
            evaluation_target.clone(),
            Reduction::Mean,
        ))?);
    }

    Ok(model)
}

fn read_loss<B>(loss: Tensor<B, 1>) -> Result<f32, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    loss.into_data()
        .into_vec::<f32>()
        .map_err(|error| ProbeError::DataConversion(error.to_string()))?
        .first()
        .copied()
        .ok_or_else(|| ProbeError::DataConversion("loss tensor was empty".to_owned()))
}

fn optimizer_probe_result<B>(
    model: &Linear<B>,
    losses: Vec<f32>,
) -> Result<OptimizerProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let final_weights = model
        .weight
        .val()
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| ProbeError::DataConversion(error.to_string()))?;
    let final_bias = model
        .bias
        .as_ref()
        .ok_or(ProbeError::MissingBiasParameter)?
        .val()
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| ProbeError::DataConversion(error.to_string()))?;

    Ok(OptimizerProbeResult {
        losses,
        final_weights,
        final_bias,
    })
}

fn nonlinear_optimizer_probe_result<B>(
    model: &NonlinearModel<B>,
    losses: Vec<f32>,
) -> Result<NonlinearOptimizerProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    Ok(NonlinearOptimizerProbeResult {
        losses,
        final_parameters: nonlinear_parameters(model)?,
    })
}

fn minibatch_optimizer_probe_result<B>(
    model: &NonlinearModel<B>,
    losses: Vec<f32>,
    batch_sequence: Vec<usize>,
    final_data_position: usize,
    final_generator_state: u64,
) -> Result<MiniBatchOptimizerProbeResult, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    Ok(MiniBatchOptimizerProbeResult {
        losses,
        batch_sequence,
        final_parameters: nonlinear_parameters(model)?,
        final_data_position,
        final_generator_state,
    })
}

fn nonlinear_parameters<B>(model: &NonlinearModel<B>) -> Result<Vec<f32>, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let mut parameters = read_values(model.hidden.weight.val())?;
    parameters.extend(read_values(
        model
            .hidden
            .bias
            .as_ref()
            .ok_or(ProbeError::MissingNonlinearBiasParameter("hidden-layer"))?
            .val(),
    )?);
    parameters.extend(read_values(model.output.weight.val())?);
    parameters.extend(read_values(
        model
            .output
            .bias
            .as_ref()
            .ok_or(ProbeError::MissingNonlinearBiasParameter("output-layer"))?
            .val(),
    )?);
    Ok(parameters)
}

fn read_values<B, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>, ProbeError>
where
    B: Backend<FloatElem = f32>,
{
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| ProbeError::DataConversion(error.to_string()))
}

/// Run the deterministic training workload and wait for backend completion.
///
/// Unlike [`run_training_probe`], this function does not read tensor values
/// back to the host. It is intended for timing the allocation, forward, loss,
/// backward, and explicit synchronization path without transfer overhead.
///
/// # Errors
///
/// Returns [`ProbeError`] if the backend omits a parameter gradient or cannot
/// synchronize the workload.
pub fn run_synchronized_training_workload<B>(device: &B::Device) -> Result<(), ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let result = execute_training_probe::<B>(device)?;
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;
    std::hint::black_box(result);
    Ok(())
}

/// Run a quadratic forward, mean-squared loss, and input-gradient backward pass.
///
/// The caller supplies a preallocated input so timing can exclude host upload.
/// The custom path uses [`quadratic`] and its explicit backward rule; the
/// reference path composes Burn primitives and uses their generated autodiff
/// graph. Both paths synchronize without reading values back to the host.
///
/// # Errors
///
/// Returns [`ProbeError`] if the backend omits the input gradient or cannot
/// synchronize the workload.
pub fn run_synchronized_quadratic_training_workload<B>(
    input: Tensor<B, 1>,
    path: QuadraticTrainingPath,
    device: &B::Device,
) -> Result<(), ProbeError>
where
    B: AutodiffBackend<FloatElem = f32> + CustomOpsBackend,
{
    let input = input.require_grad();
    let output = match path {
        QuadraticTrainingPath::Custom => quadratic(input.clone()),
        QuadraticTrainingPath::Reference => quadratic_reference(input.clone()),
    };
    let loss = output.clone().mul(output).mean();
    let gradients = loss.clone().backward();
    let input_gradient = input
        .grad(&gradients)
        .ok_or(ProbeError::MissingInputGradient)?;

    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;
    std::hint::black_box((loss, input_gradient));
    Ok(())
}

fn execute_training_probe<B>(device: &B::Device) -> Result<TrainingProbeTensors<B>, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let (input, target) = training_batch::<B>(device);
    let model = training_model::<B>(device);

    let weight = model.weight.val();
    let Some(bias) = model.bias.as_ref().map(Param::val) else {
        return Err(ProbeError::MissingBiasGradient);
    };
    let predictions = model.forward(input);
    let loss = MseLoss::new().forward(predictions.clone(), target, Reduction::Mean);
    let gradients = loss.clone().backward();
    let weight_gradient = weight.grad(&gradients).ok_or(ProbeError::MissingGradient)?;
    let bias_gradient = bias
        .grad(&gradients)
        .ok_or(ProbeError::MissingBiasGradient)?;

    Ok(TrainingProbeTensors {
        predictions,
        loss,
        weight_gradient,
        bias_gradient,
    })
}

fn training_batch<B>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    (
        Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], device),
        Tensor::<B, 2>::from_floats([[1.0], [2.0], [-0.5]], device),
    )
}

fn training_model<B>(device: &B::Device) -> Linear<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_floats([[0.5], [-0.25]], device)),
        bias: Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
            [0.1],
            device,
        ))),
    }
}

fn nonlinear_training_batch<B>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    (
        Tensor::<B, 2>::from_floats([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], device),
        Tensor::<B, 2>::from_floats([[1.0], [-1.0], [-1.0], [1.0]], device),
    )
}

fn nonlinear_mini_batch<B>(
    inputs: ProductBatchInputs,
    targets: ProductBatchTargets,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    (
        Tensor::<B, 2>::from_floats(inputs, device),
        Tensor::<B, 2>::from_floats(targets, device),
    )
}

fn multibatch_training_dataset<B>(
    dataset: &InMemoryProductDataset,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    (
        Tensor::<B, 3>::from_floats(dataset.inputs(), device).reshape([dataset.example_count(), 2]),
        Tensor::<B, 3>::from_floats(dataset.targets(), device)
            .reshape([dataset.example_count(), 1]),
    )
}

fn nonlinear_training_model<B>(device: &B::Device) -> NonlinearModel<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    NonlinearModel {
        hidden: Linear {
            weight: Param::from_tensor(Tensor::<B, 2>::from_floats(
                [[0.4, -0.3, 0.2], [-0.2, 0.5, 0.3]],
                device,
            )),
            bias: Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
                [0.1, -0.1, 0.05],
                device,
            ))),
        },
        output: Linear {
            weight: Param::from_tensor(Tensor::<B, 2>::from_floats([[0.3], [-0.4], [0.2]], device)),
            bias: Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
                [0.0],
                device,
            ))),
        },
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use burn::{
        backend::{Autodiff, Flex, flex::FlexDevice},
        tensor::Tensor,
    };

    use super::{
        CustomOpProbeResult, MINI_BATCH_CHECKPOINT_STEP, OPTIMIZER_CHECKPOINT_STEP,
        OPTIMIZER_PROBE_STEPS, ProbeResult, QuadraticTrainingPath, TrainingProbeResult,
        quadratic_reference, run_autodiff_probe, run_custom_op_probe,
        run_minibatch_checkpoint_probe, run_nonlinear_checkpoint_probe,
        run_optimizer_checkpoint_probe, run_optimizer_probe,
        run_synchronized_quadratic_training_workload, run_synchronized_training_workload,
        run_training_probe,
    };

    #[test]
    fn computes_expected_output_and_gradient() {
        type Backend = Autodiff<Flex>;

        let result = run_autodiff_probe::<Backend>(&FlexDevice).unwrap();

        assert_eq!(
            result,
            ProbeResult {
                output: vec![8.0, 18.0],
                weight_gradient: vec![4.0, 6.0],
            }
        );
    }

    #[test]
    fn computes_expected_linear_model_gradients() {
        type Backend = Autodiff<Flex>;

        let result = run_training_probe::<Backend>(&FlexDevice).unwrap();
        let expected = TrainingProbeResult {
            predictions: vec![0.1, 0.6, -0.525],
            loss: 0.923_541_67,
            weight_gradient: vec![-3.383_333_4, -4.941_667],
            bias_gradient: vec![-1.55],
        };

        assert_values_close(&result.predictions, &expected.predictions);
        assert_values_close(&[result.loss], &[expected.loss]);
        assert_values_close(&result.weight_gradient, &expected.weight_gradient);
        assert_values_close(&result.bias_gradient, &expected.bias_gradient);
    }

    #[test]
    fn optimizer_reduces_loss_to_expected_parameters() {
        type Backend = Autodiff<Flex>;

        let result = run_optimizer_probe::<Backend>(&FlexDevice).unwrap();

        assert_eq!(result.losses.len(), OPTIMIZER_PROBE_STEPS + 1);
        assert!(result.losses.windows(2).all(|losses| losses[1] < losses[0]));
        assert_values_close(&[result.losses[0]], &[0.923_541_67]);
        assert_values_close(&[result.losses[OPTIMIZER_PROBE_STEPS]], &[0.013_698_871]);
        assert_values_close(&result.final_weights, &[0.640_245_14, -0.012_434_039]);
        assert_values_close(&result.final_bias, &[0.209_908_92]);
    }

    #[test]
    fn optimizer_checkpoint_resume_matches_uninterrupted_training() {
        type Backend = Autodiff<Flex>;

        let result = run_optimizer_checkpoint_probe::<Backend>(&FlexDevice).unwrap();

        assert_eq!(OPTIMIZER_CHECKPOINT_STEP, 10);
        assert_eq!(result.uninterrupted.losses.len(), OPTIMIZER_PROBE_STEPS + 1);
        assert_eq!(result.uninterrupted, result.resumed);
        assert!(result.model_checkpoint_bytes > 0);
        assert!(result.optimizer_checkpoint_bytes > 0);
        assert!(
            result.uninterrupted.losses[OPTIMIZER_PROBE_STEPS] < result.uninterrupted.losses[0]
        );
    }

    #[test]
    fn nonlinear_checkpoint_resume_matches_uninterrupted_training() {
        type Backend = Autodiff<Flex>;

        let result = run_nonlinear_checkpoint_probe::<Backend>(&FlexDevice).unwrap();

        assert_eq!(result.uninterrupted.losses.len(), OPTIMIZER_PROBE_STEPS + 1);
        assert_eq!(result.uninterrupted, result.resumed);
        assert!(result.model_checkpoint_bytes > 0);
        assert!(result.optimizer_checkpoint_bytes > 0);
        assert!(
            result.uninterrupted.losses[OPTIMIZER_PROBE_STEPS] < result.uninterrupted.losses[0]
        );
        assert_values_close(&[result.uninterrupted.losses[0]], &[1.067_263_4]);
        assert_values_close(
            &[result.uninterrupted.losses[OPTIMIZER_PROBE_STEPS]],
            &[0.955_212_5],
        );
        assert_values_close(
            &result.uninterrupted.final_parameters,
            &[
                0.229_769_95,
                -0.345_785_77,
                -0.026_133_817,
                -0.172_173_17,
                0.504_222_5,
                0.426_487_74,
                0.244_003_82,
                -0.537_556_05,
                -0.082_056,
                -0.063_568_816,
                -0.292_702_76,
                0.184_942_65,
                -0.015_700_676,
            ],
        );
    }

    #[test]
    fn minibatch_checkpoint_restores_data_position_and_training_state() {
        type Backend = Autodiff<Flex>;

        let result = run_minibatch_checkpoint_probe::<Backend>(&FlexDevice).unwrap();
        assert_eq!(result.uninterrupted, result.resumed);
        assert_eq!(result.uninterrupted.losses.len(), OPTIMIZER_PROBE_STEPS + 1);
        assert_eq!(
            result.uninterrupted.batch_sequence,
            [3, 4, 2, 1, 0, 4, 2, 3, 0, 1, 4, 1, 0, 3, 2, 0, 4, 1, 3, 2]
        );
        assert_eq!(MINI_BATCH_CHECKPOINT_STEP, 11);
        assert_eq!(result.checkpoint_data_position, 1);
        assert_eq!(result.checkpoint_epoch_permutation, [4, 1, 0, 3, 2]);
        assert_eq!(result.checkpoint_generator_state, 0xC987_7FB0_C8DA_721A);
        assert_eq!(result.uninterrupted.final_data_position, 5);
        assert_eq!(
            result.uninterrupted.final_generator_state,
            0x4265_6696_C604_626E
        );
        assert!(result.model_checkpoint_bytes > 0);
        assert!(result.optimizer_checkpoint_bytes > 0);
        assert!(result.data_checkpoint_bytes > 0);
        assert!(
            result.uninterrupted.losses[OPTIMIZER_PROBE_STEPS] < result.uninterrupted.losses[0]
        );
        assert_values_close(&[result.uninterrupted.losses[0]], &[0.552_909_14]);
        assert_values_close(
            &[result.uninterrupted.losses[OPTIMIZER_PROBE_STEPS]],
            &[0.499_564_65],
        );
        assert_values_close(
            &result.uninterrupted.final_parameters,
            &[
                0.323_447_26,
                -0.286_213_43,
                0.036_966_48,
                0.006_780_087,
                0.218_331_47,
                0.189_097_76,
                0.144_874_41,
                -0.215_510_09,
                -0.020_334_823,
                -0.102_201_834,
                -0.139_567_6,
                -0.060_481_966,
                0.018_556_682,
            ],
        );
    }

    #[test]
    fn computes_custom_quadratic_output_and_gradient() {
        type Backend = Autodiff<Flex>;

        let result = run_custom_op_probe::<Backend>(&FlexDevice).unwrap();

        assert_eq!(
            result,
            CustomOpProbeResult {
                output: vec![2.0, -0.25, 0.0, 3.75],
                input_gradient: vec![-3.0, 0.0, 1.0, 4.0],
            }
        );
    }

    #[test]
    fn computes_portable_quadratic_reference() {
        type Backend = Flex;

        let input = Tensor::<Backend, 1>::from_floats([-2.0, -0.5, 0.0, 1.5], &FlexDevice);
        let output = quadratic_reference(input)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        assert_eq!(output, vec![2.0, -0.25, 0.0, 3.75]);
    }

    #[test]
    fn synchronizes_training_workload_without_readback() {
        type Backend = Autodiff<Flex>;

        run_synchronized_training_workload::<Backend>(&FlexDevice).unwrap();
    }

    #[test]
    fn synchronizes_both_quadratic_training_paths_without_readback() {
        type Backend = Autodiff<Flex>;

        let input = Tensor::<Backend, 1>::ones([256], &FlexDevice);
        for path in [
            QuadraticTrainingPath::Custom,
            QuadraticTrainingPath::Reference,
        ] {
            run_synchronized_quadratic_training_workload(input.clone(), path, &FlexDevice).unwrap();
        }
    }

    fn assert_values_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "expected {expected}, got {actual}"
            );
        }
    }
}
