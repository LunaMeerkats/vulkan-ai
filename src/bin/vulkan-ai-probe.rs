#![recursion_limit = "256"]

use burn::backend::{
    Autodiff, Flex, Vulkan as VulkanBackend,
    flex::FlexDevice,
    wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan as VulkanGraphics, init_setup},
};
use std::fmt;
use vulkan_ai::{TrainingProbeResult, run_autodiff_probe, run_training_probe};

const PARITY_ABSOLUTE_TOLERANCE: f32 = 1.0e-5;
const PARITY_RELATIVE_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, PartialEq, Eq)]
struct VulkanAdapterReport {
    name: String,
    backend: String,
    device_type: String,
    vendor_id: u32,
    device_id: u32,
    pci_bus_id: String,
    driver: String,
    driver_info: String,
    subgroup_min_size: u32,
    subgroup_max_size: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroup_size: [u32; 3],
    max_compute_workgroups_per_dimension: u32,
    max_compute_workgroup_storage_size: u32,
    max_storage_buffer_binding_size: u64,
    max_storage_buffers_per_shader_stage: u32,
    max_buffer_size: u64,
}

impl fmt::Display for VulkanAdapterReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Vulkan adapter:")?;
        writeln!(formatter, "  Name: {}", self.name)?;
        writeln!(formatter, "  Backend: {}", self.backend)?;
        writeln!(formatter, "  Device type: {}", self.device_type)?;
        writeln!(
            formatter,
            "  Vendor/device ID: {:#06x}/{:#06x}",
            self.vendor_id, self.device_id
        )?;
        writeln!(
            formatter,
            "  PCI bus: {}",
            value_or_unavailable(&self.pci_bus_id)
        )?;
        writeln!(
            formatter,
            "  Driver: {}",
            value_or_unavailable(&self.driver)
        )?;
        writeln!(
            formatter,
            "  Driver info: {}",
            value_or_unavailable(&self.driver_info)
        )?;
        writeln!(formatter, "Vulkan compute capabilities:")?;
        writeln!(
            formatter,
            "  Subgroup size: {}..={}",
            self.subgroup_min_size, self.subgroup_max_size
        )?;
        writeln!(
            formatter,
            "  Max invocations/workgroup: {}",
            self.max_compute_invocations_per_workgroup
        )?;
        writeln!(
            formatter,
            "  Max workgroup size: {} x {} x {}",
            self.max_compute_workgroup_size[0],
            self.max_compute_workgroup_size[1],
            self.max_compute_workgroup_size[2]
        )?;
        writeln!(
            formatter,
            "  Max workgroups/dimension: {}",
            self.max_compute_workgroups_per_dimension
        )?;
        writeln!(
            formatter,
            "  Max workgroup storage: {} bytes",
            self.max_compute_workgroup_storage_size
        )?;
        writeln!(
            formatter,
            "  Max storage buffer binding: {} bytes",
            self.max_storage_buffer_binding_size
        )?;
        writeln!(
            formatter,
            "  Max storage buffers/shader stage: {}",
            self.max_storage_buffers_per_shader_stage
        )?;
        write!(
            formatter,
            "  Max buffer size: {} bytes",
            self.max_buffer_size
        )
    }
}

fn value_or_unavailable(value: &str) -> &str {
    if value.is_empty() {
        "unavailable"
    } else {
        value
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type CpuBackend = Autodiff<Flex>;
    type VulkanAutodiffBackend = Autodiff<VulkanBackend>;

    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup::<VulkanGraphics>(&device, RuntimeOptions::default());
    let adapter_info = setup.adapter.get_info();
    let device_limits = setup.device.limits();

    let adapter_report = VulkanAdapterReport {
        name: adapter_info.name,
        backend: format!("{:?}", adapter_info.backend),
        device_type: format!("{:?}", adapter_info.device_type),
        vendor_id: adapter_info.vendor,
        device_id: adapter_info.device,
        pci_bus_id: adapter_info.device_pci_bus_id,
        driver: adapter_info.driver,
        driver_info: adapter_info.driver_info,
        subgroup_min_size: adapter_info.subgroup_min_size,
        subgroup_max_size: adapter_info.subgroup_max_size,
        max_compute_invocations_per_workgroup: device_limits.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size: [
            device_limits.max_compute_workgroup_size_x,
            device_limits.max_compute_workgroup_size_y,
            device_limits.max_compute_workgroup_size_z,
        ],
        max_compute_workgroups_per_dimension: device_limits.max_compute_workgroups_per_dimension,
        max_compute_workgroup_storage_size: device_limits.max_compute_workgroup_storage_size,
        max_storage_buffer_binding_size: device_limits.max_storage_buffer_binding_size,
        max_storage_buffers_per_shader_stage: device_limits.max_storage_buffers_per_shader_stage,
        max_buffer_size: device_limits.max_buffer_size,
    };

    let result = run_autodiff_probe::<VulkanAutodiffBackend>(&device)?;
    let cpu_training_result = run_training_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_training_result = run_training_probe::<VulkanAutodiffBackend>(&device)?;
    check_training_parity(&cpu_training_result, &vulkan_training_result)?;

    println!("{adapter_report}");
    println!("Vulkan forward output: {:?}", result.output);
    println!("Vulkan weight gradient: {:?}", result.weight_gradient);
    println!(
        "Vulkan training predictions: {:?}",
        vulkan_training_result.predictions
    );
    println!("Vulkan training loss: {}", vulkan_training_result.loss);
    println!(
        "Vulkan training weight gradient: {:?}",
        vulkan_training_result.weight_gradient
    );
    println!(
        "Vulkan training bias gradient: {:?}",
        vulkan_training_result.bias_gradient
    );
    println!(
        "CPU/Vulkan training parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );

    Ok(())
}

fn check_training_parity(
    cpu: &TrainingProbeResult,
    vulkan: &TrainingProbeResult,
) -> Result<(), String> {
    check_values("predictions", &cpu.predictions, &vulkan.predictions)?;
    check_values("loss", &[cpu.loss], &[vulkan.loss])?;
    check_values(
        "weight gradient",
        &cpu.weight_gradient,
        &vulkan.weight_gradient,
    )?;
    check_values("bias gradient", &cpu.bias_gradient, &vulkan.bias_gradient)
}

fn check_values(name: &str, cpu: &[f32], vulkan: &[f32]) -> Result<(), String> {
    if cpu.len() != vulkan.len() {
        return Err(format!(
            "CPU/Vulkan {name} length differs: {} versus {}",
            cpu.len(),
            vulkan.len()
        ));
    }

    for (index, (&cpu_value, &vulkan_value)) in cpu.iter().zip(vulkan).enumerate() {
        if !cpu_value.is_finite() || !vulkan_value.is_finite() {
            return Err(format!(
                "CPU/Vulkan {name} contains a non-finite value at index {index}: {cpu_value} versus {vulkan_value}"
            ));
        }

        let tolerance = PARITY_ABSOLUTE_TOLERANCE + PARITY_RELATIVE_TOLERANCE * cpu_value.abs();
        if (cpu_value - vulkan_value).abs() > tolerance {
            return Err(format!(
                "CPU/Vulkan {name} differs at index {index}: {cpu_value} versus {vulkan_value} (tolerance {tolerance})"
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TrainingProbeResult, VulkanAdapterReport, check_training_parity};

    #[test]
    fn formats_vulkan_adapter_report() {
        let report = VulkanAdapterReport {
            name: "Example GPU".to_owned(),
            backend: "Vulkan".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
            vendor_id: 0x1234,
            device_id: 0x5678,
            pci_bus_id: String::new(),
            driver: "example".to_owned(),
            driver_info: "1.2.3".to_owned(),
            subgroup_min_size: 32,
            subgroup_max_size: 64,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size: [1024, 1024, 64],
            max_compute_workgroups_per_dimension: 65_535,
            max_compute_workgroup_storage_size: 32_768,
            max_storage_buffer_binding_size: 134_217_728,
            max_storage_buffers_per_shader_stage: 8,
            max_buffer_size: 268_435_456,
        };

        assert_eq!(
            report.to_string(),
            "\
Vulkan adapter:
  Name: Example GPU
  Backend: Vulkan
  Device type: DiscreteGpu
  Vendor/device ID: 0x1234/0x5678
  PCI bus: unavailable
  Driver: example
  Driver info: 1.2.3
Vulkan compute capabilities:
  Subgroup size: 32..=64
  Max invocations/workgroup: 1024
  Max workgroup size: 1024 x 1024 x 64
  Max workgroups/dimension: 65535
  Max workgroup storage: 32768 bytes
  Max storage buffer binding: 134217728 bytes
  Max storage buffers/shader stage: 8
  Max buffer size: 268435456 bytes"
        );
    }

    #[test]
    fn accepts_training_results_within_tolerance() {
        let cpu = training_result(0.5);
        let vulkan = training_result(0.500_009);

        check_training_parity(&cpu, &vulkan).unwrap();
    }

    #[test]
    fn rejects_training_results_outside_tolerance() {
        let cpu = training_result(0.5);
        let vulkan = training_result(0.51);

        let error = check_training_parity(&cpu, &vulkan).unwrap_err();

        assert_eq!(
            error,
            "CPU/Vulkan loss differs at index 0: 0.5 versus 0.51 (tolerance 0.000015)"
        );
    }

    #[test]
    fn rejects_non_finite_training_results() {
        let cpu = training_result(0.5);
        let vulkan = training_result(f32::NAN);

        let error = check_training_parity(&cpu, &vulkan).unwrap_err();

        assert_eq!(
            error,
            "CPU/Vulkan loss contains a non-finite value at index 0: 0.5 versus NaN"
        );
    }

    fn training_result(loss: f32) -> TrainingProbeResult {
        TrainingProbeResult {
            predictions: vec![0.1, 0.6, -0.525],
            loss,
            weight_gradient: vec![-3.383_333_4, -4.941_667],
            bias_gradient: vec![-1.55],
        }
    }
}
