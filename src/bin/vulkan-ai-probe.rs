#![recursion_limit = "256"]

use burn::backend::{
    Autodiff, Vulkan as VulkanBackend,
    wgpu::{WgpuDevice, graphics::Vulkan as VulkanGraphics, init_setup},
};
use vulkan_ai::run_autodiff_probe;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = Autodiff<VulkanBackend>;

    let device = WgpuDevice::DefaultDevice;
    init_setup::<VulkanGraphics>(&device, Default::default());

    let result = run_autodiff_probe::<Backend>(&device)?;
    println!("Vulkan forward output: {:?}", result.output);
    println!("Vulkan weight gradient: {:?}", result.weight_gradient);

    Ok(())
}
