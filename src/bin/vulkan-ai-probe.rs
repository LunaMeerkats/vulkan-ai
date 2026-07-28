#![recursion_limit = "256"]

use burn::backend::{
    Autodiff, Flex, Vulkan as VulkanBackend,
    flex::FlexDevice,
    wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan as VulkanGraphics, init_setup},
};
use burn::tensor::{Tensor, backend::Backend};
use std::{
    fmt,
    time::{Duration, Instant},
};
use vulkan_ai::{
    CustomOpProbeResult, CustomOpsBackend, ProbeError, TrainingProbeResult, quadratic,
    quadratic_reference, run_autodiff_probe, run_custom_op_probe,
    run_synchronized_training_workload, run_training_probe,
};

const PARITY_ABSOLUTE_TOLERANCE: f32 = 1.0e-5;
const PARITY_RELATIVE_TOLERANCE: f32 = 1.0e-5;
const TIMING_WARMUP_ITERATIONS: usize = 5;
const TIMING_MEASURED_ITERATIONS: usize = 20;
const TIMING_SCOPE: &str = "allocation+forward+mse+backward; host readback excluded";
const TIMING_SYNCHRONIZATION: &str = "Burn Backend::sync after every iteration";
const CUSTOM_TIMING_ELEMENTS: usize = 1_048_576;
const CUSTOM_TIMING_SCOPE: &str =
    "quadratic forward over 1048576 f32 elements; input allocation and host readback excluded";

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

#[derive(Debug, PartialEq)]
struct TimingSummary {
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
}

#[derive(Debug, PartialEq)]
struct VulkanTimingReport {
    build_profile: &'static str,
    fusion: &'static str,
    tasks_max: usize,
    warmup_iterations: usize,
    samples_ms: Vec<f64>,
    summary: TimingSummary,
}

#[derive(Debug, PartialEq)]
struct VulkanCustomTimingReport {
    build_profile: &'static str,
    fusion: &'static str,
    elements: usize,
    warmup_iterations: usize,
    kernel_samples_ms: Vec<f64>,
    kernel_summary: TimingSummary,
    reference_samples_ms: Vec<f64>,
    reference_summary: TimingSummary,
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

impl fmt::Display for VulkanTimingReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Vulkan timing protocol:")?;
        writeln!(formatter, "  Build profile: {}", self.build_profile)?;
        writeln!(formatter, "  Fusion: {}", self.fusion)?;
        writeln!(formatter, "  Command task batch limit: {}", self.tasks_max)?;
        writeln!(formatter, "  Workload: {TIMING_SCOPE}")?;
        writeln!(
            formatter,
            "  Warm-up iterations: {}",
            self.warmup_iterations
        )?;
        writeln!(
            formatter,
            "  Measured iterations: {}",
            self.samples_ms.len()
        )?;
        writeln!(formatter, "  Synchronization: {TIMING_SYNCHRONIZATION}")?;
        writeln!(formatter, "Vulkan synchronized training timing:")?;
        writeln!(formatter, "  Min: {:.6} ms", self.summary.min)?;
        writeln!(formatter, "  Median: {:.6} ms", self.summary.median)?;
        writeln!(formatter, "  P95: {:.6} ms", self.summary.p95)?;
        writeln!(formatter, "  Max: {:.6} ms", self.summary.max)?;
        write!(
            formatter,
            "Vulkan timing JSON: {{\"schema\":1,\"build_profile\":\"{}\",\"fusion\":\"{}\",\"tasks_max\":{},\"scope\":\"{TIMING_SCOPE}\",\"warmup_iterations\":{},\"synchronization\":\"{TIMING_SYNCHRONIZATION}\",\"samples_ms\":[",
            self.build_profile, self.fusion, self.tasks_max, self.warmup_iterations
        )?;
        for (index, sample) in self.samples_ms.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            write!(formatter, "{sample:.6}")?;
        }
        write!(
            formatter,
            "],\"min_ms\":{:.6},\"median_ms\":{:.6},\"p95_ms\":{:.6},\"max_ms\":{:.6}}}",
            self.summary.min, self.summary.median, self.summary.p95, self.summary.max
        )
    }
}

impl fmt::Display for VulkanCustomTimingReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Vulkan custom quadratic benchmark:")?;
        writeln!(formatter, "  Build profile: {}", self.build_profile)?;
        writeln!(formatter, "  Fusion: {}", self.fusion)?;
        writeln!(formatter, "  Elements: {}", self.elements)?;
        writeln!(formatter, "  Workload: {CUSTOM_TIMING_SCOPE}")?;
        writeln!(
            formatter,
            "  Warm-up iterations per implementation: {}",
            self.warmup_iterations
        )?;
        writeln!(
            formatter,
            "  Measured iterations per implementation: {}",
            self.kernel_samples_ms.len()
        )?;
        writeln!(formatter, "  Synchronization: {TIMING_SYNCHRONIZATION}")?;
        writeln!(
            formatter,
            "  CubeCL kernel median: {:.6} ms",
            self.kernel_summary.median
        )?;
        writeln!(
            formatter,
            "  Burn reference median: {:.6} ms",
            self.reference_summary.median
        )?;
        writeln!(
            formatter,
            "  Reference/kernel median ratio: {:.3}x",
            self.reference_summary.median / self.kernel_summary.median
        )?;
        write!(
            formatter,
            "Vulkan custom timing JSON: {{\"schema\":1,\"build_profile\":\"{}\",\"fusion\":\"{}\",\"elements\":{},\"scope\":\"{CUSTOM_TIMING_SCOPE}\",\"warmup_iterations\":{},\"synchronization\":\"{TIMING_SYNCHRONIZATION}\",\"kernel_samples_ms\":[",
            self.build_profile, self.fusion, self.elements, self.warmup_iterations
        )?;
        write_samples(formatter, &self.kernel_samples_ms)?;
        formatter.write_str("],\"reference_samples_ms\":[")?;
        write_samples(formatter, &self.reference_samples_ms)?;
        write!(
            formatter,
            "],\"kernel_median_ms\":{:.6},\"reference_median_ms\":{:.6},\"reference_kernel_median_ratio\":{:.6}}}",
            self.kernel_summary.median,
            self.reference_summary.median,
            self.reference_summary.median / self.kernel_summary.median
        )
    }
}

fn write_samples(formatter: &mut fmt::Formatter<'_>, samples: &[f64]) -> fmt::Result {
    for (index, sample) in samples.iter().enumerate() {
        if index > 0 {
            formatter.write_str(",")?;
        }
        write!(formatter, "{sample:.6}")?;
    }

    Ok(())
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
    let runtime_options = RuntimeOptions::default();
    let tasks_max = runtime_options.tasks_max;
    let setup = init_setup::<VulkanGraphics>(&device, runtime_options);
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
    let cpu_custom_result = run_custom_op_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_custom_result = run_custom_op_probe::<VulkanAutodiffBackend>(&device)?;
    check_custom_op_parity(&cpu_custom_result, &vulkan_custom_result)?;
    let timing_report = measure_training_timing::<VulkanAutodiffBackend>(&device, tasks_max)?;
    let custom_timing_report = measure_custom_op_timing::<VulkanBackend>(&device)?;

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
    println!(
        "Vulkan custom quadratic output: {:?}",
        vulkan_custom_result.output
    );
    println!(
        "Vulkan custom quadratic input gradient: {:?}",
        vulkan_custom_result.input_gradient
    );
    println!(
        "CPU/Vulkan custom operation parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );
    println!("{timing_report}");
    println!("{custom_timing_report}");

    Ok(())
}

fn measure_custom_op_timing<B>(device: &B::Device) -> Result<VulkanCustomTimingReport, ProbeError>
where
    B: Backend<FloatElem = f32> + CustomOpsBackend,
{
    let input = Tensor::<B, 1>::ones([CUSTOM_TIMING_ELEMENTS], device);
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;

    for _ in 0..TIMING_WARMUP_ITERATIONS {
        measure_quadratic_iteration(input.clone(), true, device)?;
        measure_quadratic_iteration(input.clone(), false, device)?;
    }

    let mut kernel_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    let mut reference_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    for iteration in 0..TIMING_MEASURED_ITERATIONS {
        if iteration.is_multiple_of(2) {
            kernel_samples.push(measure_quadratic_iteration(input.clone(), true, device)?);
            reference_samples.push(measure_quadratic_iteration(input.clone(), false, device)?);
        } else {
            reference_samples.push(measure_quadratic_iteration(input.clone(), false, device)?);
            kernel_samples.push(measure_quadratic_iteration(input.clone(), true, device)?);
        }
    }

    let (kernel_samples_ms, kernel_summary) = summarize_samples(&kernel_samples);
    let (reference_samples_ms, reference_summary) = summarize_samples(&reference_samples);

    Ok(VulkanCustomTimingReport {
        build_profile: build_profile(),
        fusion: fusion_state(),
        elements: CUSTOM_TIMING_ELEMENTS,
        warmup_iterations: TIMING_WARMUP_ITERATIONS,
        kernel_samples_ms,
        kernel_summary,
        reference_samples_ms,
        reference_summary,
    })
}

fn measure_quadratic_iteration<B>(
    input: Tensor<B, 1>,
    use_kernel: bool,
    device: &B::Device,
) -> Result<Duration, ProbeError>
where
    B: Backend<FloatElem = f32> + CustomOpsBackend,
{
    let start = Instant::now();
    let output = if use_kernel {
        quadratic(input)
    } else {
        quadratic_reference(input)
    };
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;
    std::hint::black_box(output);

    Ok(start.elapsed())
}

fn measure_training_timing<B>(
    device: &B::Device,
    tasks_max: usize,
) -> Result<VulkanTimingReport, ProbeError>
where
    B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>,
{
    for _ in 0..TIMING_WARMUP_ITERATIONS {
        run_synchronized_training_workload::<B>(device)?;
    }

    let mut samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    for _ in 0..TIMING_MEASURED_ITERATIONS {
        let start = Instant::now();
        run_synchronized_training_workload::<B>(device)?;
        samples.push(start.elapsed());
    }

    Ok(timing_report(&samples, tasks_max))
}

fn timing_report(samples: &[Duration], tasks_max: usize) -> VulkanTimingReport {
    let (samples_ms, summary) = summarize_samples(samples);

    VulkanTimingReport {
        build_profile: build_profile(),
        fusion: fusion_state(),
        tasks_max,
        warmup_iterations: TIMING_WARMUP_ITERATIONS,
        samples_ms,
        summary,
    }
}

fn summarize_samples(samples: &[Duration]) -> (Vec<f64>, TimingSummary) {
    assert!(!samples.is_empty(), "timing requires at least one sample");

    let samples_ms = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[middle - 1], sorted[middle])
    } else {
        sorted[middle]
    };
    let p95_index = (sorted.len() * 95).div_ceil(100) - 1;
    (
        samples_ms,
        TimingSummary {
            min: sorted[0],
            median,
            p95: sorted[p95_index],
            max: sorted[sorted.len() - 1],
        },
    )
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn fusion_state() -> &'static str {
    if cfg!(feature = "vulkan-fusion") {
        "enabled"
    } else {
        "disabled"
    }
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

fn check_custom_op_parity(
    cpu: &CustomOpProbeResult,
    vulkan: &CustomOpProbeResult,
) -> Result<(), String> {
    check_values("custom operation output", &cpu.output, &vulkan.output)?;
    check_values(
        "custom operation input gradient",
        &cpu.input_gradient,
        &vulkan.input_gradient,
    )
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
    use std::time::Duration;

    use super::{
        CUSTOM_TIMING_SCOPE, TIMING_SCOPE, TIMING_SYNCHRONIZATION, TimingSummary,
        TrainingProbeResult, VulkanAdapterReport, VulkanCustomTimingReport, check_training_parity,
        timing_report,
    };

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

    #[test]
    fn summarizes_and_serializes_timing_samples() {
        let report = timing_report(
            &[
                Duration::from_millis(4),
                Duration::from_millis(1),
                Duration::from_millis(3),
                Duration::from_millis(2),
            ],
            32,
        );

        assert_eq!(
            report.summary,
            TimingSummary {
                min: 1.0,
                median: 2.5,
                p95: 4.0,
                max: 4.0,
            }
        );
        let formatted = report.to_string();
        assert!(formatted.contains("  Warm-up iterations: 5"));
        assert!(formatted.contains("  Measured iterations: 4"));
        assert!(formatted.contains(&format!("  Workload: {TIMING_SCOPE}")));
        assert!(formatted.contains(&format!("  Synchronization: {TIMING_SYNCHRONIZATION}")));
        assert!(formatted.contains(
            "\"samples_ms\":[4.000000,1.000000,3.000000,2.000000],\"min_ms\":1.000000,\"median_ms\":2.500000,\"p95_ms\":4.000000,\"max_ms\":4.000000"
        ));
    }

    #[test]
    fn formats_custom_kernel_benchmark() {
        let report = VulkanCustomTimingReport {
            build_profile: "release",
            fusion: "disabled",
            elements: 1_048_576,
            warmup_iterations: 5,
            kernel_samples_ms: vec![1.0, 2.0],
            kernel_summary: TimingSummary {
                min: 1.0,
                median: 1.5,
                p95: 2.0,
                max: 2.0,
            },
            reference_samples_ms: vec![3.0, 4.0],
            reference_summary: TimingSummary {
                min: 3.0,
                median: 3.5,
                p95: 4.0,
                max: 4.0,
            },
        };

        let formatted = report.to_string();

        assert!(formatted.contains(&format!("  Workload: {CUSTOM_TIMING_SCOPE}")));
        assert!(formatted.contains("  CubeCL kernel median: 1.500000 ms"));
        assert!(formatted.contains("  Burn reference median: 3.500000 ms"));
        assert!(formatted.contains("  Reference/kernel median ratio: 2.333x"));
        assert!(formatted.contains(
            "\"kernel_samples_ms\":[1.000000,2.000000],\"reference_samples_ms\":[3.000000,4.000000]"
        ));
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
