#![recursion_limit = "256"]

//! Small, backend-independent checks for Vulkan AI research workflows.

use burn::{
    module::Param,
    nn::{
        Linear,
        loss::{MseLoss, Reduction},
    },
    tensor::{Tensor, backend::AutodiffBackend},
};
use std::{error::Error, fmt};

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

/// Error returned when a backend cannot complete the probe.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// The autodiff backend did not return the requested weight gradient.
    MissingGradient,
    /// The autodiff backend did not return the requested bias gradient.
    MissingBiasGradient,
    /// Tensor data could not be converted to the expected `f32` values.
    DataConversion(String),
    /// The backend could not synchronize the training workload.
    Synchronization(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGradient => formatter.write_str("the required weight gradient is missing"),
            Self::MissingBiasGradient => {
                formatter.write_str("the required bias gradient is missing")
            }
            Self::DataConversion(message) => {
                write!(formatter, "could not read probe tensor data: {message}")
            }
            Self::Synchronization(message) => {
                write!(
                    formatter,
                    "could not synchronize the training workload: {message}"
                )
            }
        }
    }
}

impl Error for ProbeError {}

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

fn execute_training_probe<B>(device: &B::Device) -> Result<TrainingProbeTensors<B>, ProbeError>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let input = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], device);
    let target = Tensor::<B, 2>::from_floats([[1.0], [2.0], [-0.5]], device);
    let model = Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_floats([[0.5], [-0.25]], device)),
        bias: Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
            [0.1],
            device,
        ))),
    };

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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use burn::backend::{Autodiff, Flex, flex::FlexDevice};

    use super::{
        ProbeResult, TrainingProbeResult, run_autodiff_probe, run_synchronized_training_workload,
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
    fn synchronizes_training_workload_without_readback() {
        type Backend = Autodiff<Flex>;

        run_synchronized_training_workload::<Backend>(&FlexDevice).unwrap();
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
