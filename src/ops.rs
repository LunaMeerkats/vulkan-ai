//! Backend extensions used to validate custom forward and backward operations.

use burn::{
    backend::autodiff::{
        Autodiff, NodeId,
        checkpoint::{
            base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
            strategy::CheckpointStrategy,
        },
        grads::Gradients,
        ops::{Backward, Ops, OpsKind, unary},
    },
    tensor::{Tensor, TensorPrimitive, backend::Backend as BurnBackend, ops::FloatTensor},
};
use std::marker::PhantomData;

#[cfg(feature = "cpu")]
use burn::backend::Flex;

/// Backend support for Vulkan AI custom tensor operations.
///
/// Base backends use the default forward implementation. The autodiff
/// decorator overrides it to register one graph node with an explicit backward
/// rule.
pub trait CustomOpsBackend: BurnBackend {
    /// Calculate `x² + x` without adding autodiff graph nodes for its component
    /// operations.
    fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
        quadratic_primitive::<Self>(input)
    }
}

#[cfg(feature = "cpu")]
impl CustomOpsBackend for Flex {}

#[cfg(feature = "vulkan")]
impl CustomOpsBackend for burn::backend::Vulkan {}

/// Apply the custom element-wise quadratic operation `x² + x`.
pub fn quadratic<B: CustomOpsBackend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let output = B::quadratic(input.into_primitive().tensor());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn quadratic_primitive<B: BurnBackend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let squared = B::float_mul(input.clone(), input.clone());
    B::float_add(squared, input)
}

#[derive(Debug)]
struct QuadraticBackward;

impl<B: CustomOpsBackend> Backward<B, 1> for QuadraticBackward {
    type State = NodeId;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let input = checkpointer.retrieve_node_output(ops.state);
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let derivative =
                B::float_add_scalar(B::float_mul_scalar(input, 2.0.into()), 1.0.into());
            B::float_mul(grad, derivative)
        });
    }
}

#[derive(Debug)]
struct RetroQuadratic<B: BurnBackend> {
    input_id: NodeId,
    backend: PhantomData<B>,
}

impl<B: BurnBackend> RetroQuadratic<B> {
    fn new(input_id: NodeId) -> Self {
        Self {
            input_id,
            backend: PhantomData,
        }
    }
}

impl<B: BurnBackend> RetroForward for RetroQuadratic<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
        states.save(out_node, quadratic_primitive::<B>(input));
    }
}

impl<B: CustomOpsBackend, C: CheckpointStrategy> CustomOpsBackend for Autodiff<B, C> {
    fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
        match QuadraticBackward
            .prepare::<C>([input.node.clone()])
            .memory_bound()
            .retro_forward(RetroQuadratic::<B>::new(input.node.id))
            .parents([&input])
            .stateful()
        {
            OpsKind::Tracked(mut preparation) => {
                let state = preparation.checkpoint(&input);
                let output = quadratic_primitive::<B>(input.primitive);
                preparation.finish(state, output)
            }
            OpsKind::UnTracked(preparation) => {
                preparation.finish(quadratic_primitive::<B>(input.primitive))
            }
        }
    }
}
