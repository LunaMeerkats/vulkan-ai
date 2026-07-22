#![recursion_limit = "256"]

//! Small, backend-independent checks for Vulkan AI research workflows.

use burn::tensor::{Tensor, backend::AutodiffBackend};
use std::{error::Error, fmt};

/// Values produced by a forward pass and its gradient calculation.
#[derive(Debug, PartialEq)]
pub struct ProbeResult {
    /// Result of multiplying the input matrix by the trainable weights.
    pub output: Vec<f32>,
    /// Gradient of the summed output with respect to the weights.
    pub weight_gradient: Vec<f32>,
}

/// Error returned when a backend cannot complete the probe.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// The autodiff backend did not return the requested weight gradient.
    MissingGradient,
    /// Tensor data could not be converted to the expected `f32` values.
    DataConversion(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGradient => formatter.write_str("the required weight gradient is missing"),
            Self::DataConversion(message) => {
                write!(formatter, "could not read probe tensor data: {message}")
            }
        }
    }
}

impl Error for ProbeError {}

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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use burn::backend::{Autodiff, Flex, flex::FlexDevice};

    use super::{ProbeResult, run_autodiff_probe};

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
}
