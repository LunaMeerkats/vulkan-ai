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

#[cfg(all(feature = "cpu", test))]
std::thread_local! {
    static FLEX_QUADRATIC_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Backend support for Vulkan AI custom tensor operations.
///
/// Base backends use the default forward implementation. The autodiff
/// decorator overrides it to register one graph node with an explicit backward
/// rule.
pub trait CustomOpsBackend: BurnBackend {
    /// Calculate `x² + x` without adding autodiff graph nodes for its component operations.
    ///
    /// Portable backends use the Burn reference implementation. `CubeCL` backends
    /// override this method with a custom element-wise kernel.
    fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
        quadratic_primitive::<Self>(input)
    }
}

#[cfg(all(feature = "cpu", not(test)))]
impl CustomOpsBackend for Flex {}

#[cfg(all(feature = "cpu", test))]
impl CustomOpsBackend for Flex {
    fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
        // Let the regression test distinguish backend delegation from the
        // mathematically equivalent portable implementation.
        FLEX_QUADRATIC_CALLS.with(|calls| calls.set(calls.get() + 1));
        quadratic_primitive::<Self>(input)
    }
}

/// Apply the custom element-wise quadratic operation `x² + x`.
pub fn quadratic<B: CustomOpsBackend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let output = B::quadratic(input.into_primitive().tensor());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Apply the portable Burn reference for the element-wise operation `x² + x`.
pub fn quadratic_reference<B: BurnBackend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let output = quadratic_primitive::<B>(input.into_primitive().tensor());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn quadratic_primitive<B: BurnBackend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let squared = B::float_mul(input.clone(), input.clone());
    B::float_add(squared, input)
}

#[cfg(feature = "vulkan")]
mod cubecl_forward {
    use super::{CustomOpsBackend, quadratic_primitive};
    use burn::{
        backend::wgpu::{
            BoolElement, CubeBackend, CubeTensor, FloatElement, IntElement, WgpuRuntime,
        },
        tensor::ops::FloatTensor,
    };
    use burn_cubecl::kernel::into_contiguous;
    use cubecl::{calculate_cube_count_elemwise, cube, prelude::*};

    #[allow(clippy::trivially_copy_pass_by_ref)]
    #[cube(launch)]
    fn quadratic_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
        if ABSOLUTE_POS >= output.len() {
            terminate!();
        }

        let value = input[ABSOLUTE_POS];
        output[ABSOLUTE_POS] = value * value + value;
    }

    impl<F, I, BT> CustomOpsBackend for CubeBackend<WgpuRuntime, F, I, BT>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
            if input.dtype != F::dtype() {
                return quadratic_primitive::<Self>(input);
            }
            if input.meta.num_elements() == 0 {
                return input;
            }

            let input = into_contiguous(input);
            let client = input.client.clone();
            let num_elements = input.meta.num_elements();
            let cube_dim = CubeDim::new(&client, num_elements);
            let cube_count = calculate_cube_count_elemwise(&client, num_elements, cube_dim);
            let output = CubeTensor::new_contiguous(
                client.clone(),
                input.device.clone(),
                input.meta.shape().clone(),
                client.empty(num_elements * input.dtype.size()),
                input.dtype,
            );

            quadratic_kernel::launch::<F, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                input.into_tensor_arg(),
                output.clone().into_tensor_arg(),
            );

            output
        }
    }
}

#[cfg(feature = "vulkan-fusion")]
mod fusion_forward {
    use super::CustomOpsBackend;
    use burn::tensor::ops::FloatTensor;
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
    use std::marker::PhantomData;

    #[derive(Debug)]
    struct QuadraticOperation<B> {
        description: CustomOpIr,
        backend: PhantomData<B>,
    }

    impl<B> QuadraticOperation<B> {
        fn new(description: CustomOpIr) -> Self {
            Self {
                description,
                backend: PhantomData,
            }
        }
    }

    impl<B> Operation<B::FusionRuntime> for QuadraticOperation<B>
    where
        B: FusionBackend + CustomOpsBackend,
    {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let ([input], [output]) = self.description.as_fixed();
            let input = handles.get_float_tensor::<B>(input);
            let output_tensor = B::quadratic(input);

            handles.register_float_tensor::<B>(&output.id, output_tensor);
        }
    }

    impl<B> CustomOpsBackend for Fusion<B>
    where
        B: FusionBackend + CustomOpsBackend,
    {
        fn quadratic(input: FloatTensor<Self>) -> FloatTensor<Self> {
            let client = input.client.clone();
            let streams = OperationStreams::with_inputs([&input]);
            let output = TensorIr::uninit(
                client.create_empty_handle(),
                input.shape.clone(),
                input.dtype,
            );
            let description = CustomOpIr::new("vulkan_ai_quadratic", &[input.into_ir()], &[output]);

            client
                .register(
                    streams,
                    OperationIr::Custom(description.clone()),
                    QuadraticOperation::<B>::new(description),
                )
                .output()
        }
    }
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
struct RetroQuadratic<B: CustomOpsBackend> {
    input_id: NodeId,
    backend: PhantomData<B>,
}

impl<B: CustomOpsBackend> RetroQuadratic<B> {
    fn new(input_id: NodeId) -> Self {
        Self {
            input_id,
            backend: PhantomData,
        }
    }
}

impl<B: CustomOpsBackend> RetroForward for RetroQuadratic<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
        states.save(out_node, B::quadratic(input));
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
                let output = B::quadratic(input.primitive);
                preparation.finish(state, output)
            }
            OpsKind::UnTracked(preparation) => preparation.finish(B::quadratic(input.primitive)),
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use burn::{
        backend::{Autodiff, Flex, flex::FlexDevice},
        tensor::Tensor,
    };

    use super::{FLEX_QUADRATIC_CALLS, quadratic};

    #[test]
    fn autodiff_forward_delegates_to_the_inner_backend() {
        type Backend = Autodiff<Flex>;

        FLEX_QUADRATIC_CALLS.with(|calls| calls.set(0));
        let input =
            Tensor::<Backend, 1>::from_floats([-2.0, -0.5, 0.0, 1.5], &FlexDevice).require_grad();
        let output = quadratic(input).into_data().into_vec::<f32>().unwrap();

        assert_eq!(output, vec![2.0, -0.25, 0.0, 3.75]);
        FLEX_QUADRATIC_CALLS.with(|calls| assert_eq!(calls.get(), 1));
    }
}
