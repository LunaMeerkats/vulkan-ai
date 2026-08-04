#![recursion_limit = "256"]

use burn::backend::{
    Autodiff, Flex, Vulkan as VulkanBackend,
    flex::FlexDevice,
    wgpu::{
        RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::Vulkan as VulkanGraphics, init_setup,
    },
};
use burn::tensor::{Tensor, backend::Backend};
use cubecl::Runtime;
use std::{
    fmt,
    time::{Duration, Instant},
};
use vulkan_ai::{
    CustomOpProbeResult, CustomOpsBackend, MiniBatchCheckpointProbeResult,
    MiniBatchOptimizerProbeResult, NonlinearCheckpointProbeResult, NonlinearOptimizerProbeResult,
    OPTIMIZER_CHECKPOINT_DAMPENING, OPTIMIZER_CHECKPOINT_MOMENTUM, OPTIMIZER_CHECKPOINT_STEP,
    OPTIMIZER_PROBE_LEARNING_RATE, OPTIMIZER_PROBE_STEPS, OptimizerCheckpointProbeResult,
    OptimizerProbeResult, ProbeError, QuadraticTrainingPath, TrainingProbeResult, quadratic,
    quadratic_reference, run_autodiff_probe, run_custom_op_probe, run_minibatch_checkpoint_probe,
    run_nonlinear_checkpoint_probe, run_optimizer_checkpoint_probe, run_optimizer_probe,
    run_synchronized_quadratic_training_workload, run_synchronized_training_workload,
    run_training_probe,
};

const PARITY_ABSOLUTE_TOLERANCE: f32 = 1.0e-5;
const PARITY_RELATIVE_TOLERANCE: f32 = 1.0e-5;
const TIMING_WARMUP_ITERATIONS: usize = 5;
const TIMING_MEASURED_ITERATIONS: usize = 20;
const TIMING_SCOPE: &str = "allocation+forward+mse+backward; host readback excluded";
const TIMING_SYNCHRONIZATION: &str = "Burn Backend::sync after every iteration";
const CUSTOM_TIMING_ELEMENTS: [usize; 5] = [1, 256, 4_096, 65_536, 1_048_576];
const CUSTOM_TIMING_WARMUP_ITERATIONS: usize = 20;
const CUSTOM_TIMING_SCOPE: &str = "quadratic forward over preallocated f32 input; output allocation/reuse, dispatch, and synchronization included; host readback excluded";
const CUSTOM_PROFILE_SCOPE: &str =
    "CubeCL runtime profile around forward and synchronization; host readback excluded";
const CUSTOM_TRAINING_WARMUP_ITERATIONS: usize = 20;
const CUSTOM_TRAINING_SCOPE: &str = "quadratic forward + mean-squared output loss + input-gradient backward over preallocated f32 input; graph/output allocation, dispatch, reduction, and synchronization included; host readback excluded";

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
    warmup_iterations: usize,
    profile_timing_method: Option<String>,
    measurements: Vec<CustomTimingMeasurement>,
}

#[derive(Debug, PartialEq)]
struct CustomTimingMeasurement {
    elements: usize,
    kernel_samples_ms: Vec<f64>,
    kernel_summary: TimingSummary,
    reference_samples_ms: Vec<f64>,
    reference_summary: TimingSummary,
    kernel_profile_samples_ms: Option<Vec<f64>>,
    kernel_profile_summary: Option<TimingSummary>,
    reference_profile_samples_ms: Option<Vec<f64>>,
    reference_profile_summary: Option<TimingSummary>,
}

#[derive(Debug, PartialEq)]
struct VulkanCustomTrainingTimingReport {
    build_profile: &'static str,
    fusion: &'static str,
    warmup_iterations: usize,
    measurements: Vec<CustomTrainingTimingMeasurement>,
}

#[derive(Debug, PartialEq)]
struct CustomTrainingTimingMeasurement {
    elements: usize,
    custom_samples_ms: Vec<f64>,
    custom_summary: TimingSummary,
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
        writeln!(formatter, "Vulkan custom quadratic size sweep:")?;
        writeln!(formatter, "  Build profile: {}", self.build_profile)?;
        writeln!(formatter, "  Fusion: {}", self.fusion)?;
        write!(formatter, "  Elements: ")?;
        for (index, measurement) in self.measurements.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{}", measurement.elements)?;
        }
        formatter.write_str("\n")?;
        writeln!(formatter, "  Wall-clock scope: {CUSTOM_TIMING_SCOPE}")?;
        if let Some(timing_method) = &self.profile_timing_method {
            writeln!(formatter, "  Profile scope: {CUSTOM_PROFILE_SCOPE}")?;
            writeln!(formatter, "  Profile timing method: {timing_method}")?;
        } else {
            writeln!(
                formatter,
                "  Profile timing: unavailable with fusion; nested CubeCL profiling is intentionally disabled"
            )?;
        }
        writeln!(
            formatter,
            "  Warm-up iterations per implementation: {}",
            self.warmup_iterations
        )?;
        writeln!(
            formatter,
            "  Measured iterations per implementation: {}",
            self.measurements
                .first()
                .map_or(0, |measurement| measurement.kernel_samples_ms.len())
        )?;
        writeln!(formatter, "  Synchronization: {TIMING_SYNCHRONIZATION}")?;
        writeln!(formatter, "Vulkan custom quadratic wall-clock medians:")?;
        writeln!(
            formatter,
            "  Elements | CubeCL kernel | Burn reference | Reference/kernel"
        )?;
        for measurement in &self.measurements {
            writeln!(
                formatter,
                "  {:>8} | {:>11.6} ms | {:>13.6} ms | {:>15.3}x",
                measurement.elements,
                measurement.kernel_summary.median,
                measurement.reference_summary.median,
                measurement.reference_summary.median / measurement.kernel_summary.median,
            )?;
        }
        if let Some(timing_method) = &self.profile_timing_method {
            writeln!(
                formatter,
                "Vulkan custom quadratic profiled medians ({timing_method}):"
            )?;
            writeln!(
                formatter,
                "  Elements | CubeCL kernel | Burn reference | Reference/kernel"
            )?;
            for measurement in &self.measurements {
                if let (Some(kernel), Some(reference)) = (
                    &measurement.kernel_profile_summary,
                    &measurement.reference_profile_summary,
                ) {
                    writeln!(
                        formatter,
                        "  {:>8} | {:>11.6} ms | {:>13.6} ms | {:>15.3}x",
                        measurement.elements,
                        kernel.median,
                        reference.median,
                        reference.median / kernel.median,
                    )?;
                }
            }
        }
        self.write_json(formatter)
    }
}

impl VulkanCustomTimingReport {
    fn write_json(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Vulkan custom timing JSON: {{\"schema\":2,\"build_profile\":\"{}\",\"fusion\":\"{}\",\"wall_clock_scope\":\"{CUSTOM_TIMING_SCOPE}\",\"profile_scope\":\"{CUSTOM_PROFILE_SCOPE}\",\"profile_timing_method\":\"{}\",\"warmup_iterations\":{},\"synchronization\":\"{TIMING_SYNCHRONIZATION}\",\"measurements\":[",
            self.build_profile,
            self.fusion,
            self.profile_timing_method
                .as_deref()
                .unwrap_or("unavailable"),
            self.warmup_iterations
        )?;
        for (index, measurement) in self.measurements.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            write!(
                formatter,
                "{{\"elements\":{},\"kernel_wall_samples_ms\":[",
                measurement.elements
            )?;
            write_samples(formatter, &measurement.kernel_samples_ms)?;
            formatter.write_str("],\"reference_wall_samples_ms\":[")?;
            write_samples(formatter, &measurement.reference_samples_ms)?;
            formatter.write_str("],\"kernel_profile_samples_ms\":")?;
            write_optional_samples(formatter, measurement.kernel_profile_samples_ms.as_deref())?;
            formatter.write_str(",\"reference_profile_samples_ms\":")?;
            write_optional_samples(
                formatter,
                measurement.reference_profile_samples_ms.as_deref(),
            )?;
            write!(
                formatter,
                ",\"kernel_wall_median_ms\":{:.6},\"reference_wall_median_ms\":{:.6},\"wall_reference_kernel_median_ratio\":{:.6}",
                measurement.kernel_summary.median,
                measurement.reference_summary.median,
                measurement.reference_summary.median / measurement.kernel_summary.median,
            )?;
            if let (Some(kernel), Some(reference)) = (
                &measurement.kernel_profile_summary,
                &measurement.reference_profile_summary,
            ) {
                write!(
                    formatter,
                    ",\"kernel_profile_median_ms\":{:.6},\"reference_profile_median_ms\":{:.6},\"profile_reference_kernel_median_ratio\":{:.6}",
                    kernel.median,
                    reference.median,
                    reference.median / kernel.median,
                )?;
            } else {
                formatter.write_str(
                    ",\"kernel_profile_median_ms\":null,\"reference_profile_median_ms\":null,\"profile_reference_kernel_median_ratio\":null",
                )?;
            }
            formatter.write_str("}")?;
        }
        formatter.write_str("]}")
    }
}

impl fmt::Display for VulkanCustomTrainingTimingReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Vulkan quadratic autodiff size sweep:")?;
        writeln!(formatter, "  Build profile: {}", self.build_profile)?;
        writeln!(formatter, "  Fusion: {}", self.fusion)?;
        write!(formatter, "  Elements: ")?;
        for (index, measurement) in self.measurements.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{}", measurement.elements)?;
        }
        formatter.write_str("\n")?;
        writeln!(formatter, "  Wall-clock scope: {CUSTOM_TRAINING_SCOPE}")?;
        writeln!(
            formatter,
            "  Warm-up iterations per implementation: {}",
            self.warmup_iterations
        )?;
        writeln!(
            formatter,
            "  Measured iterations per implementation: {}",
            self.measurements
                .first()
                .map_or(0, |measurement| measurement.custom_samples_ms.len())
        )?;
        writeln!(formatter, "  Synchronization: {TIMING_SYNCHRONIZATION}")?;
        writeln!(formatter, "Vulkan quadratic autodiff wall-clock medians:")?;
        writeln!(
            formatter,
            "  Elements | Custom forward/backward | Burn autodiff reference | Reference/custom"
        )?;
        for measurement in &self.measurements {
            writeln!(
                formatter,
                "  {:>8} | {:>19.6} ms | {:>21.6} ms | {:>16.3}x",
                measurement.elements,
                measurement.custom_summary.median,
                measurement.reference_summary.median,
                measurement.reference_summary.median / measurement.custom_summary.median,
            )?;
        }

        write!(
            formatter,
            "Vulkan quadratic autodiff timing JSON: {{\"schema\":1,\"build_profile\":\"{}\",\"fusion\":\"{}\",\"wall_clock_scope\":\"{CUSTOM_TRAINING_SCOPE}\",\"warmup_iterations\":{},\"synchronization\":\"{TIMING_SYNCHRONIZATION}\",\"measurements\":[",
            self.build_profile, self.fusion, self.warmup_iterations
        )?;
        for (index, measurement) in self.measurements.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            write!(
                formatter,
                "{{\"elements\":{},\"custom_samples_ms\":[",
                measurement.elements
            )?;
            write_samples(formatter, &measurement.custom_samples_ms)?;
            formatter.write_str("],\"reference_samples_ms\":[")?;
            write_samples(formatter, &measurement.reference_samples_ms)?;
            write!(
                formatter,
                "],\"custom_median_ms\":{:.6},\"reference_median_ms\":{:.6},\"reference_custom_median_ratio\":{:.6}}}",
                measurement.custom_summary.median,
                measurement.reference_summary.median,
                measurement.reference_summary.median / measurement.custom_summary.median,
            )?;
        }
        formatter.write_str("]}")
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

fn write_optional_samples(
    formatter: &mut fmt::Formatter<'_>,
    samples: Option<&[f64]>,
) -> fmt::Result {
    let Some(samples) = samples else {
        return formatter.write_str("null");
    };

    formatter.write_str("[")?;
    write_samples(formatter, samples)?;
    formatter.write_str("]")
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
    let cpu_optimizer_result = run_optimizer_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_optimizer_result = run_optimizer_probe::<VulkanAutodiffBackend>(&device)?;
    check_optimizer_parity(&cpu_optimizer_result, &vulkan_optimizer_result)?;
    let cpu_checkpoint_result = run_optimizer_checkpoint_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_checkpoint_result =
        run_optimizer_checkpoint_probe::<VulkanAutodiffBackend>(&device)?;
    check_optimizer_checkpoint_parity(&cpu_checkpoint_result, &vulkan_checkpoint_result)?;
    let cpu_nonlinear_checkpoint_result =
        run_nonlinear_checkpoint_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_nonlinear_checkpoint_result =
        run_nonlinear_checkpoint_probe::<VulkanAutodiffBackend>(&device)?;
    check_nonlinear_checkpoint_parity(
        &cpu_nonlinear_checkpoint_result,
        &vulkan_nonlinear_checkpoint_result,
    )?;
    let vulkan_minibatch_checkpoint_result = run_minibatch_checkpoint_checks(&device)?;
    let cpu_custom_result = run_custom_op_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan_custom_result = run_custom_op_probe::<VulkanAutodiffBackend>(&device)?;
    check_custom_op_parity(&cpu_custom_result, &vulkan_custom_result)?;
    let timing_report = measure_training_timing::<VulkanAutodiffBackend>(&device, tasks_max)?;
    let custom_timing_report = measure_custom_op_timing::<VulkanBackend>(&device)?;
    let custom_training_timing_report =
        measure_quadratic_training_timing::<VulkanAutodiffBackend>(&device)?;

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
    print_optimizer_results(&vulkan_optimizer_result, &vulkan_checkpoint_result);
    print_nonlinear_checkpoint_results(&vulkan_nonlinear_checkpoint_result);
    print_minibatch_checkpoint_results(&vulkan_minibatch_checkpoint_result);
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
    println!("{custom_training_timing_report}");

    Ok(())
}

fn run_minibatch_checkpoint_checks(
    device: &WgpuDevice,
) -> Result<MiniBatchCheckpointProbeResult, Box<dyn std::error::Error>> {
    type CpuBackend = Autodiff<Flex>;
    type VulkanAutodiffBackend = Autodiff<VulkanBackend>;

    let cpu = run_minibatch_checkpoint_probe::<CpuBackend>(&FlexDevice)?;
    let vulkan = run_minibatch_checkpoint_probe::<VulkanAutodiffBackend>(device)?;
    check_minibatch_checkpoint_parity(&cpu, &vulkan)?;
    Ok(vulkan)
}

fn print_optimizer_results(
    optimizer: &OptimizerProbeResult,
    checkpoint: &OptimizerCheckpointProbeResult,
) {
    println!(
        "Vulkan optimizer loss: {} -> {} over {OPTIMIZER_PROBE_STEPS} SGD steps at learning rate {OPTIMIZER_PROBE_LEARNING_RATE}",
        optimizer.losses[0], optimizer.losses[OPTIMIZER_PROBE_STEPS]
    );
    println!(
        "Vulkan optimizer final weights: {:?}",
        optimizer.final_weights
    );
    println!("Vulkan optimizer final bias: {:?}", optimizer.final_bias);
    println!(
        "CPU/Vulkan optimizer loss and parameter parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );
    println!(
        "Vulkan checkpoint/resume loss: {} -> {} over {OPTIMIZER_PROBE_STEPS} momentum SGD steps at learning rate {OPTIMIZER_PROBE_LEARNING_RATE}; checkpoint restored after step {OPTIMIZER_CHECKPOINT_STEP} (momentum {OPTIMIZER_CHECKPOINT_MOMENTUM}, dampening {OPTIMIZER_CHECKPOINT_DAMPENING})",
        checkpoint.resumed.losses[0], checkpoint.resumed.losses[OPTIMIZER_PROBE_STEPS]
    );
    println!(
        "Vulkan checkpoint size: model {} bytes, optimizer {} bytes",
        checkpoint.model_checkpoint_bytes, checkpoint.optimizer_checkpoint_bytes
    );
    println!(
        "CPU/Vulkan uninterrupted and checkpoint-resumed parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );
}

fn print_nonlinear_checkpoint_results(checkpoint: &NonlinearCheckpointProbeResult) {
    println!(
        "Vulkan nonlinear checkpoint/resume loss: {} -> {} over {OPTIMIZER_PROBE_STEPS} momentum SGD steps at learning rate {OPTIMIZER_PROBE_LEARNING_RATE}; checkpoint restored after step {OPTIMIZER_CHECKPOINT_STEP} (momentum {OPTIMIZER_CHECKPOINT_MOMENTUM}, dampening {OPTIMIZER_CHECKPOINT_DAMPENING})",
        checkpoint.resumed.losses[0], checkpoint.resumed.losses[OPTIMIZER_PROBE_STEPS]
    );
    println!(
        "Vulkan nonlinear final parameters: {:?}",
        checkpoint.resumed.final_parameters
    );
    println!(
        "Vulkan nonlinear checkpoint size: model {} bytes, optimizer {} bytes",
        checkpoint.model_checkpoint_bytes, checkpoint.optimizer_checkpoint_bytes
    );
    println!(
        "CPU/Vulkan nonlinear uninterrupted and checkpoint-resumed parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );
}

fn print_minibatch_checkpoint_results(checkpoint: &MiniBatchCheckpointProbeResult) {
    println!(
        "Vulkan mini-batch checkpoint/resume loss: {} -> {} over {OPTIMIZER_PROBE_STEPS} momentum SGD steps at learning rate {OPTIMIZER_PROBE_LEARNING_RATE}; checkpoint restored after step {OPTIMIZER_CHECKPOINT_STEP}",
        checkpoint.resumed.losses[0], checkpoint.resumed.losses[OPTIMIZER_PROBE_STEPS]
    );
    println!(
        "Vulkan mini-batch schedule: {:?} repeating; restored data position {}",
        &checkpoint.resumed.batch_sequence[..4],
        checkpoint.checkpoint_data_position
    );
    println!(
        "Vulkan mini-batch checkpoint size: model {} bytes, optimizer {} bytes, data position {} bytes",
        checkpoint.model_checkpoint_bytes,
        checkpoint.optimizer_checkpoint_bytes,
        checkpoint.data_checkpoint_bytes
    );
    println!(
        "CPU/Vulkan mini-batch uninterrupted and checkpoint-resumed parity: passed (absolute tolerance {PARITY_ABSOLUTE_TOLERANCE}, relative tolerance {PARITY_RELATIVE_TOLERANCE})"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CustomBenchmark {
    Kernel,
    Reference,
}

const CUSTOM_BENCHMARK_ORDERS: [[CustomBenchmark; 2]; 2] = [
    [CustomBenchmark::Kernel, CustomBenchmark::Reference],
    [CustomBenchmark::Reference, CustomBenchmark::Kernel],
];

impl CustomBenchmark {
    const fn profile_name(self) -> &'static str {
        match self {
            Self::Kernel => "vulkan_ai_quadratic_kernel",
            Self::Reference => "vulkan_ai_quadratic_reference",
        }
    }
}

fn measure_custom_op_timing<B>(device: &WgpuDevice) -> Result<VulkanCustomTimingReport, ProbeError>
where
    B: Backend<Device = WgpuDevice, FloatElem = f32> + CustomOpsBackend,
{
    let mut measurements = Vec::with_capacity(CUSTOM_TIMING_ELEMENTS.len());
    let mut profile_timing_method = None;
    for elements in CUSTOM_TIMING_ELEMENTS {
        let (measurement, timing_method) = measure_custom_op_size::<B>(elements, device)?;
        if let Some(timing_method) = timing_method {
            record_profile_timing_method(&mut profile_timing_method, timing_method)?;
        }
        measurements.push(measurement);
    }
    if !cfg!(feature = "vulkan-fusion") && profile_timing_method.is_none() {
        return Err(ProbeError::Profiling(
            "the unfused size sweep did not collect profile samples".to_owned(),
        ));
    }

    Ok(VulkanCustomTimingReport {
        build_profile: build_profile(),
        fusion: fusion_state(),
        warmup_iterations: CUSTOM_TIMING_WARMUP_ITERATIONS,
        profile_timing_method,
        measurements,
    })
}

fn measure_custom_op_size<B>(
    elements: usize,
    device: &WgpuDevice,
) -> Result<(CustomTimingMeasurement, Option<String>), ProbeError>
where
    B: Backend<Device = WgpuDevice, FloatElem = f32> + CustomOpsBackend,
{
    let input = Tensor::<B, 1>::ones([elements], device);
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;

    for iteration in 0..CUSTOM_TIMING_WARMUP_ITERATIONS {
        for benchmark in custom_benchmark_order(iteration) {
            measure_custom_benchmark(input.clone(), benchmark, device)?;
        }
    }

    let mut kernel_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    let mut reference_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    for iteration in 0..TIMING_MEASURED_ITERATIONS {
        for benchmark in custom_benchmark_order(iteration) {
            let sample = measure_custom_benchmark(input.clone(), benchmark, device)?;
            match benchmark {
                CustomBenchmark::Kernel => kernel_samples.push(sample),
                CustomBenchmark::Reference => reference_samples.push(sample),
            }
        }
    }

    let (
        kernel_profile_samples_ms,
        kernel_profile_summary,
        reference_profile_samples_ms,
        reference_profile_summary,
        profile_timing_method,
    ) = if cfg!(feature = "vulkan-fusion") {
        (None, None, None, None, None)
    } else {
        let mut profile_timing_method = None;
        for iteration in 0..CUSTOM_TIMING_WARMUP_ITERATIONS {
            for benchmark in custom_benchmark_order(iteration) {
                let (_, timing_method) =
                    measure_profiled_custom_benchmark(input.clone(), benchmark, device)?;
                record_profile_timing_method(&mut profile_timing_method, timing_method)?;
            }
        }

        let mut kernel_profile_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
        let mut reference_profile_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
        for iteration in 0..TIMING_MEASURED_ITERATIONS {
            for benchmark in custom_benchmark_order(iteration) {
                let (sample, timing_method) =
                    measure_profiled_custom_benchmark(input.clone(), benchmark, device)?;
                record_profile_timing_method(&mut profile_timing_method, timing_method)?;
                match benchmark {
                    CustomBenchmark::Kernel => kernel_profile_samples.push(sample),
                    CustomBenchmark::Reference => reference_profile_samples.push(sample),
                }
            }
        }

        let (kernel_profile_samples_ms, kernel_profile_summary) =
            summarize_samples(&kernel_profile_samples);
        let (reference_profile_samples_ms, reference_profile_summary) =
            summarize_samples(&reference_profile_samples);
        (
            Some(kernel_profile_samples_ms),
            Some(kernel_profile_summary),
            Some(reference_profile_samples_ms),
            Some(reference_profile_summary),
            Some(profile_timing_method.ok_or_else(|| {
                ProbeError::Profiling("the profiled sample set was empty".to_owned())
            })?),
        )
    };
    let (kernel_samples_ms, kernel_summary) = summarize_samples(&kernel_samples);
    let (reference_samples_ms, reference_summary) = summarize_samples(&reference_samples);
    Ok((
        CustomTimingMeasurement {
            elements,
            kernel_samples_ms,
            kernel_summary,
            reference_samples_ms,
            reference_summary,
            kernel_profile_samples_ms,
            kernel_profile_summary,
            reference_profile_samples_ms,
            reference_profile_summary,
        },
        profile_timing_method,
    ))
}

fn custom_benchmark_order(iteration: usize) -> [CustomBenchmark; 2] {
    CUSTOM_BENCHMARK_ORDERS[iteration % CUSTOM_BENCHMARK_ORDERS.len()]
}

fn measure_custom_benchmark<B>(
    input: Tensor<B, 1>,
    benchmark: CustomBenchmark,
    device: &WgpuDevice,
) -> Result<Duration, ProbeError>
where
    B: Backend<Device = WgpuDevice, FloatElem = f32> + CustomOpsBackend,
{
    let start = Instant::now();
    let output = match benchmark {
        CustomBenchmark::Kernel => quadratic(input),
        CustomBenchmark::Reference => quadratic_reference(input),
    };
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;
    std::hint::black_box(output);

    Ok(start.elapsed())
}

fn measure_profiled_custom_benchmark<B>(
    input: Tensor<B, 1>,
    benchmark: CustomBenchmark,
    device: &WgpuDevice,
) -> Result<(Duration, String), ProbeError>
where
    B: Backend<Device = WgpuDevice, FloatElem = f32> + CustomOpsBackend,
{
    let client = WgpuRuntime::client(device);
    let device = device.clone();
    let (output, duration) = client
        .profile(
            move || {
                let output = match benchmark {
                    CustomBenchmark::Kernel => quadratic(input),
                    CustomBenchmark::Reference => quadratic_reference(input),
                };
                B::sync(&device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;
                Ok::<_, ProbeError>(output)
            },
            benchmark.profile_name(),
        )
        .map_err(|error| ProbeError::Profiling(error.to_string()))?;
    let output = output?;
    let timing_method = duration.timing_method().to_string();
    let duration = futures_lite::future::block_on(duration.resolve()).duration();
    std::hint::black_box(output);

    Ok((duration, timing_method))
}

fn record_profile_timing_method(
    current: &mut Option<String>,
    observed: String,
) -> Result<(), ProbeError> {
    match current {
        Some(current) if current != &observed => Err(ProbeError::Profiling(format!(
            "CubeCL profile timing method changed from {current} to {observed}"
        ))),
        Some(_) => Ok(()),
        None => {
            *current = Some(observed);
            Ok(())
        }
    }
}

const QUADRATIC_TRAINING_ORDERS: [[QuadraticTrainingPath; 2]; 2] = [
    [
        QuadraticTrainingPath::Custom,
        QuadraticTrainingPath::Reference,
    ],
    [
        QuadraticTrainingPath::Reference,
        QuadraticTrainingPath::Custom,
    ],
];

fn measure_quadratic_training_timing<B>(
    device: &WgpuDevice,
) -> Result<VulkanCustomTrainingTimingReport, ProbeError>
where
    B: burn::tensor::backend::AutodiffBackend<Device = WgpuDevice, FloatElem = f32>
        + CustomOpsBackend,
{
    let mut measurements = Vec::with_capacity(CUSTOM_TIMING_ELEMENTS.len());
    for elements in CUSTOM_TIMING_ELEMENTS {
        measurements.push(measure_quadratic_training_size::<B>(elements, device)?);
    }

    Ok(VulkanCustomTrainingTimingReport {
        build_profile: build_profile(),
        fusion: fusion_state(),
        warmup_iterations: CUSTOM_TRAINING_WARMUP_ITERATIONS,
        measurements,
    })
}

fn measure_quadratic_training_size<B>(
    elements: usize,
    device: &WgpuDevice,
) -> Result<CustomTrainingTimingMeasurement, ProbeError>
where
    B: burn::tensor::backend::AutodiffBackend<Device = WgpuDevice, FloatElem = f32>
        + CustomOpsBackend,
{
    let input = Tensor::<B, 1>::ones([elements], device);
    B::sync(device).map_err(|error| ProbeError::Synchronization(error.to_string()))?;

    for iteration in 0..CUSTOM_TRAINING_WARMUP_ITERATIONS {
        for path in quadratic_training_order(iteration) {
            measure_quadratic_training_benchmark(input.clone(), path, device)?;
        }
    }

    let mut custom_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    let mut reference_samples = Vec::with_capacity(TIMING_MEASURED_ITERATIONS);
    for iteration in 0..TIMING_MEASURED_ITERATIONS {
        for path in quadratic_training_order(iteration) {
            let sample = measure_quadratic_training_benchmark(input.clone(), path, device)?;
            match path {
                QuadraticTrainingPath::Custom => custom_samples.push(sample),
                QuadraticTrainingPath::Reference => reference_samples.push(sample),
            }
        }
    }

    let (custom_samples_ms, custom_summary) = summarize_samples(&custom_samples);
    let (reference_samples_ms, reference_summary) = summarize_samples(&reference_samples);
    Ok(CustomTrainingTimingMeasurement {
        elements,
        custom_samples_ms,
        custom_summary,
        reference_samples_ms,
        reference_summary,
    })
}

fn quadratic_training_order(iteration: usize) -> [QuadraticTrainingPath; 2] {
    QUADRATIC_TRAINING_ORDERS[iteration % QUADRATIC_TRAINING_ORDERS.len()]
}

fn measure_quadratic_training_benchmark<B>(
    input: Tensor<B, 1>,
    path: QuadraticTrainingPath,
    device: &WgpuDevice,
) -> Result<Duration, ProbeError>
where
    B: burn::tensor::backend::AutodiffBackend<Device = WgpuDevice, FloatElem = f32>
        + CustomOpsBackend,
{
    let start = Instant::now();
    run_synchronized_quadratic_training_workload(input, path, device)?;
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

fn check_optimizer_parity(
    cpu: &OptimizerProbeResult,
    vulkan: &OptimizerProbeResult,
) -> Result<(), String> {
    check_loss_reduction("CPU", cpu)?;
    check_loss_reduction("Vulkan", vulkan)?;
    check_optimizer_results("CPU", cpu, "Vulkan", vulkan)
}

fn check_optimizer_checkpoint_parity(
    cpu: &OptimizerCheckpointProbeResult,
    vulkan: &OptimizerCheckpointProbeResult,
) -> Result<(), String> {
    check_loss_reduction("CPU uninterrupted", &cpu.uninterrupted)?;
    check_loss_reduction("CPU resumed", &cpu.resumed)?;
    check_loss_reduction("Vulkan uninterrupted", &vulkan.uninterrupted)?;
    check_loss_reduction("Vulkan resumed", &vulkan.resumed)?;
    check_optimizer_results(
        "CPU uninterrupted",
        &cpu.uninterrupted,
        "CPU resumed",
        &cpu.resumed,
    )?;
    check_optimizer_results(
        "Vulkan uninterrupted",
        &vulkan.uninterrupted,
        "Vulkan resumed",
        &vulkan.resumed,
    )?;
    check_optimizer_results(
        "CPU resumed",
        &cpu.resumed,
        "Vulkan resumed",
        &vulkan.resumed,
    )
}

fn check_nonlinear_checkpoint_parity(
    cpu: &NonlinearCheckpointProbeResult,
    vulkan: &NonlinearCheckpointProbeResult,
) -> Result<(), String> {
    check_nonlinear_loss_reduction("CPU uninterrupted", &cpu.uninterrupted)?;
    check_nonlinear_loss_reduction("CPU resumed", &cpu.resumed)?;
    check_nonlinear_loss_reduction("Vulkan uninterrupted", &vulkan.uninterrupted)?;
    check_nonlinear_loss_reduction("Vulkan resumed", &vulkan.resumed)?;
    check_nonlinear_optimizer_results(
        "CPU uninterrupted",
        &cpu.uninterrupted,
        "CPU resumed",
        &cpu.resumed,
    )?;
    check_nonlinear_optimizer_results(
        "Vulkan uninterrupted",
        &vulkan.uninterrupted,
        "Vulkan resumed",
        &vulkan.resumed,
    )?;
    check_nonlinear_optimizer_results(
        "CPU resumed",
        &cpu.resumed,
        "Vulkan resumed",
        &vulkan.resumed,
    )
}

fn check_minibatch_checkpoint_parity(
    cpu: &MiniBatchCheckpointProbeResult,
    vulkan: &MiniBatchCheckpointProbeResult,
) -> Result<(), String> {
    check_minibatch_loss_reduction("CPU uninterrupted", &cpu.uninterrupted)?;
    check_minibatch_loss_reduction("CPU resumed", &cpu.resumed)?;
    check_minibatch_loss_reduction("Vulkan uninterrupted", &vulkan.uninterrupted)?;
    check_minibatch_loss_reduction("Vulkan resumed", &vulkan.resumed)?;
    check_minibatch_optimizer_results(
        "CPU uninterrupted",
        &cpu.uninterrupted,
        "CPU resumed",
        &cpu.resumed,
    )?;
    check_minibatch_optimizer_results(
        "Vulkan uninterrupted",
        &vulkan.uninterrupted,
        "Vulkan resumed",
        &vulkan.resumed,
    )?;
    check_minibatch_optimizer_results(
        "CPU resumed",
        &cpu.resumed,
        "Vulkan resumed",
        &vulkan.resumed,
    )?;
    check_index_values(
        "mini-batch checkpoint data position",
        "CPU",
        &[cpu.checkpoint_data_position],
        "Vulkan",
        &[vulkan.checkpoint_data_position],
    )
}

fn check_optimizer_results(
    left_name: &str,
    left: &OptimizerProbeResult,
    right_name: &str,
    right: &OptimizerProbeResult,
) -> Result<(), String> {
    check_named_values(
        "optimizer loss trajectory",
        left_name,
        &left.losses,
        right_name,
        &right.losses,
    )?;
    check_named_values(
        "optimizer final weights",
        left_name,
        &left.final_weights,
        right_name,
        &right.final_weights,
    )?;
    check_named_values(
        "optimizer final bias",
        left_name,
        &left.final_bias,
        right_name,
        &right.final_bias,
    )?;
    Ok(())
}

fn check_nonlinear_optimizer_results(
    left_name: &str,
    left: &NonlinearOptimizerProbeResult,
    right_name: &str,
    right: &NonlinearOptimizerProbeResult,
) -> Result<(), String> {
    check_named_values(
        "nonlinear optimizer loss trajectory",
        left_name,
        &left.losses,
        right_name,
        &right.losses,
    )?;
    check_named_values(
        "nonlinear optimizer final parameters",
        left_name,
        &left.final_parameters,
        right_name,
        &right.final_parameters,
    )
}

fn check_minibatch_optimizer_results(
    left_name: &str,
    left: &MiniBatchOptimizerProbeResult,
    right_name: &str,
    right: &MiniBatchOptimizerProbeResult,
) -> Result<(), String> {
    check_index_values(
        "mini-batch sequence",
        left_name,
        &left.batch_sequence,
        right_name,
        &right.batch_sequence,
    )?;
    check_index_values(
        "mini-batch final data position",
        left_name,
        &[left.final_data_position],
        right_name,
        &[right.final_data_position],
    )?;
    check_named_values(
        "mini-batch evaluation loss trajectory",
        left_name,
        &left.losses,
        right_name,
        &right.losses,
    )?;
    check_named_values(
        "mini-batch final parameters",
        left_name,
        &left.final_parameters,
        right_name,
        &right.final_parameters,
    )
}

fn check_index_values(
    description: &str,
    left_name: &str,
    left: &[usize],
    right_name: &str,
    right: &[usize],
) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "{left_name}/{right_name} {description} length differs: {} versus {}",
            left.len(),
            right.len()
        ));
    }
    for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
        if left_value != right_value {
            return Err(format!(
                "{left_name}/{right_name} {description} differs at index {index}: {left_value} versus {right_value}"
            ));
        }
    }
    Ok(())
}

fn check_loss_reduction(backend: &str, result: &OptimizerProbeResult) -> Result<(), String> {
    check_loss_trajectory_reduction(backend, &result.losses)
}

fn check_nonlinear_loss_reduction(
    backend: &str,
    result: &NonlinearOptimizerProbeResult,
) -> Result<(), String> {
    check_loss_trajectory_reduction(backend, &result.losses)
}

fn check_minibatch_loss_reduction(
    backend: &str,
    result: &MiniBatchOptimizerProbeResult,
) -> Result<(), String> {
    check_loss_trajectory_reduction(backend, &result.losses)
}

fn check_loss_trajectory_reduction(backend: &str, losses: &[f32]) -> Result<(), String> {
    let Some((&initial_loss, remaining_losses)) = losses.split_first() else {
        return Err(format!("{backend} optimizer loss trajectory is empty"));
    };
    let Some(&final_loss) = remaining_losses.last() else {
        return Err(format!(
            "{backend} optimizer loss trajectory has no post-update value"
        ));
    };
    if !initial_loss.is_finite() || !final_loss.is_finite() {
        return Err(format!(
            "{backend} optimizer loss is non-finite: {initial_loss} -> {final_loss}"
        ));
    }
    if final_loss >= initial_loss {
        return Err(format!(
            "{backend} optimizer did not reduce loss: {initial_loss} -> {final_loss}"
        ));
    }

    Ok(())
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
    check_named_values(name, "CPU", cpu, "Vulkan", vulkan)
}

fn check_named_values(
    name: &str,
    left_name: &str,
    left: &[f32],
    right_name: &str,
    right: &[f32],
) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "{left_name}/{right_name} {name} length differs: {} versus {}",
            left.len(),
            right.len()
        ));
    }

    for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
        if !left_value.is_finite() || !right_value.is_finite() {
            return Err(format!(
                "{left_name}/{right_name} {name} contains a non-finite value at index {index}: {left_value} versus {right_value}"
            ));
        }

        let tolerance = PARITY_ABSOLUTE_TOLERANCE + PARITY_RELATIVE_TOLERANCE * left_value.abs();
        if (left_value - right_value).abs() > tolerance {
            return Err(format!(
                "{left_name}/{right_name} {name} differs at index {index}: {left_value} versus {right_value} (tolerance {tolerance})"
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        CUSTOM_PROFILE_SCOPE, CUSTOM_TIMING_SCOPE, CUSTOM_TRAINING_SCOPE, CustomBenchmark,
        CustomTimingMeasurement, CustomTrainingTimingMeasurement, TIMING_SCOPE,
        TIMING_SYNCHRONIZATION, TimingSummary, TrainingProbeResult, VulkanAdapterReport,
        VulkanCustomTimingReport, VulkanCustomTrainingTimingReport,
        check_minibatch_checkpoint_parity, check_nonlinear_checkpoint_parity,
        check_optimizer_checkpoint_parity, check_optimizer_parity, check_training_parity,
        custom_benchmark_order, quadratic_training_order, timing_report,
    };
    use vulkan_ai::{
        MiniBatchCheckpointProbeResult, MiniBatchOptimizerProbeResult,
        NonlinearCheckpointProbeResult, NonlinearOptimizerProbeResult,
        OptimizerCheckpointProbeResult, OptimizerProbeResult, QuadraticTrainingPath,
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
    fn accepts_optimizer_loss_reduction_and_parameter_parity() {
        let cpu = optimizer_result(0.01);
        let vulkan = optimizer_result(0.010_000_1);

        check_optimizer_parity(&cpu, &vulkan).unwrap();
    }

    #[test]
    fn rejects_optimizer_without_loss_reduction() {
        let cpu = optimizer_result(1.0);
        let vulkan = optimizer_result(0.01);

        let error = check_optimizer_parity(&cpu, &vulkan).unwrap_err();

        assert_eq!(error, "CPU optimizer did not reduce loss: 1 -> 1");
    }

    #[test]
    fn accepts_uninterrupted_and_checkpoint_resumed_parity() {
        let cpu = checkpoint_result(0.01);
        let vulkan = checkpoint_result(0.010_000_1);

        check_optimizer_checkpoint_parity(&cpu, &vulkan).unwrap();
    }

    #[test]
    fn rejects_checkpoint_resume_divergence() {
        let cpu = checkpoint_result(0.1);
        let vulkan = checkpoint_result(0.01);

        let error = check_optimizer_checkpoint_parity(&cpu, &vulkan).unwrap_err();

        assert!(error.starts_with(
            "CPU uninterrupted/CPU resumed optimizer loss trajectory differs at index 1"
        ));
    }

    #[test]
    fn accepts_nonlinear_uninterrupted_and_checkpoint_resumed_parity() {
        let cpu = nonlinear_checkpoint_result(0.95);
        let vulkan = nonlinear_checkpoint_result(0.950_001);

        check_nonlinear_checkpoint_parity(&cpu, &vulkan).unwrap();
    }

    #[test]
    fn rejects_nonlinear_checkpoint_parameter_divergence() {
        let cpu = nonlinear_checkpoint_result(0.95);
        let mut vulkan = nonlinear_checkpoint_result(0.95);
        vulkan.resumed.final_parameters[0] = 0.1;

        let error = check_nonlinear_checkpoint_parity(&cpu, &vulkan).unwrap_err();

        assert!(error.starts_with(
            "Vulkan uninterrupted/Vulkan resumed nonlinear optimizer final parameters differs at index 0"
        ));
    }

    #[test]
    fn accepts_minibatch_order_and_checkpoint_parity() {
        let cpu = minibatch_checkpoint_result(0.95);
        let vulkan = minibatch_checkpoint_result(0.950_001);

        check_minibatch_checkpoint_parity(&cpu, &vulkan).unwrap();
    }

    #[test]
    fn rejects_minibatch_data_order_divergence() {
        let cpu = minibatch_checkpoint_result(0.95);
        let mut vulkan = minibatch_checkpoint_result(0.95);
        vulkan.resumed.batch_sequence[1] = 0;

        let error = check_minibatch_checkpoint_parity(&cpu, &vulkan).unwrap_err();

        assert_eq!(
            error,
            "Vulkan uninterrupted/Vulkan resumed mini-batch sequence differs at index 1: 1 versus 0"
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
            warmup_iterations: 5,
            profile_timing_method: Some("device".to_owned()),
            measurements: vec![CustomTimingMeasurement {
                elements: 256,
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
                kernel_profile_samples_ms: Some(vec![0.5, 0.75]),
                kernel_profile_summary: Some(TimingSummary {
                    min: 0.5,
                    median: 0.625,
                    p95: 0.75,
                    max: 0.75,
                }),
                reference_profile_samples_ms: Some(vec![1.0, 1.5]),
                reference_profile_summary: Some(TimingSummary {
                    min: 1.0,
                    median: 1.25,
                    p95: 1.5,
                    max: 1.5,
                }),
            }],
        };

        let formatted = report.to_string();

        assert!(formatted.contains(&format!("  Wall-clock scope: {CUSTOM_TIMING_SCOPE}")));
        assert!(formatted.contains(&format!("  Profile scope: {CUSTOM_PROFILE_SCOPE}")));
        assert!(formatted.contains("  Profile timing method: device"));
        assert!(formatted.contains("       256 |    1.500000 ms |      3.500000 ms |"));
        assert!(formatted.contains("|           2.333x"));
        assert!(formatted.contains("       256 |    0.625000 ms |      1.250000 ms |"));
        assert!(formatted.contains("|           2.000x"));
        assert!(formatted.contains("\"schema\":2"));
        assert!(formatted.contains(
            "\"kernel_wall_samples_ms\":[1.000000,2.000000],\"reference_wall_samples_ms\":[3.000000,4.000000],\"kernel_profile_samples_ms\":[0.500000,0.750000],\"reference_profile_samples_ms\":[1.000000,1.500000]"
        ));
        assert!(formatted.contains("\"profile_reference_kernel_median_ratio\":2.000000"));
    }

    #[test]
    fn formats_fused_benchmark_without_nested_profile() {
        let report = VulkanCustomTimingReport {
            build_profile: "release",
            fusion: "enabled",
            warmup_iterations: 20,
            profile_timing_method: None,
            measurements: vec![CustomTimingMeasurement {
                elements: 1,
                kernel_samples_ms: vec![1.0],
                kernel_summary: TimingSummary {
                    min: 1.0,
                    median: 1.0,
                    p95: 1.0,
                    max: 1.0,
                },
                reference_samples_ms: vec![2.0],
                reference_summary: TimingSummary {
                    min: 2.0,
                    median: 2.0,
                    p95: 2.0,
                    max: 2.0,
                },
                kernel_profile_samples_ms: None,
                kernel_profile_summary: None,
                reference_profile_samples_ms: None,
                reference_profile_summary: None,
            }],
        };

        let formatted = report.to_string();

        assert!(formatted.contains(
            "Profile timing: unavailable with fusion; nested CubeCL profiling is intentionally disabled"
        ));
        assert!(!formatted.contains("Vulkan custom quadratic profiled medians"));
        assert!(formatted.contains("\"profile_timing_method\":\"unavailable\""));
        assert!(formatted.contains("\"kernel_profile_samples_ms\":null"));
        assert!(formatted.contains("\"profile_reference_kernel_median_ratio\":null"));
    }

    #[test]
    fn rotates_custom_benchmark_order() {
        assert_eq!(
            custom_benchmark_order(0),
            [CustomBenchmark::Kernel, CustomBenchmark::Reference]
        );
        assert_eq!(
            custom_benchmark_order(1),
            [CustomBenchmark::Reference, CustomBenchmark::Kernel]
        );
        assert_eq!(custom_benchmark_order(2), custom_benchmark_order(0));
    }

    #[test]
    fn formats_quadratic_training_benchmark() {
        let report = VulkanCustomTrainingTimingReport {
            build_profile: "release",
            fusion: "enabled",
            warmup_iterations: 20,
            measurements: vec![CustomTrainingTimingMeasurement {
                elements: 4_096,
                custom_samples_ms: vec![1.0, 2.0],
                custom_summary: TimingSummary {
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
            }],
        };

        let formatted = report.to_string();

        assert!(formatted.contains(&format!("  Wall-clock scope: {CUSTOM_TRAINING_SCOPE}")));
        assert!(
            formatted.contains("      4096 |            1.500000 ms |              3.500000 ms |")
        );
        assert!(formatted.contains("2.333x"));
        assert!(formatted.contains("\"schema\":1"));
        assert!(formatted.contains(
            "\"custom_samples_ms\":[1.000000,2.000000],\"reference_samples_ms\":[3.000000,4.000000]"
        ));
        assert!(formatted.contains("\"reference_custom_median_ratio\":2.333333"));
    }

    #[test]
    fn rotates_quadratic_training_order() {
        assert_eq!(
            quadratic_training_order(0),
            [
                QuadraticTrainingPath::Custom,
                QuadraticTrainingPath::Reference,
            ]
        );
        assert_eq!(
            quadratic_training_order(1),
            [
                QuadraticTrainingPath::Reference,
                QuadraticTrainingPath::Custom,
            ]
        );
        assert_eq!(quadratic_training_order(2), quadratic_training_order(0));
    }

    fn training_result(loss: f32) -> TrainingProbeResult {
        TrainingProbeResult {
            predictions: vec![0.1, 0.6, -0.525],
            loss,
            weight_gradient: vec![-3.383_333_4, -4.941_667],
            bias_gradient: vec![-1.55],
        }
    }

    fn optimizer_result(final_loss: f32) -> OptimizerProbeResult {
        OptimizerProbeResult {
            losses: vec![1.0, final_loss],
            final_weights: vec![0.6, -0.01],
            final_bias: vec![0.2],
        }
    }

    fn checkpoint_result(resumed_final_loss: f32) -> OptimizerCheckpointProbeResult {
        OptimizerCheckpointProbeResult {
            uninterrupted: optimizer_result(0.01),
            resumed: optimizer_result(resumed_final_loss),
            model_checkpoint_bytes: 100,
            optimizer_checkpoint_bytes: 200,
        }
    }

    fn nonlinear_checkpoint_result(resumed_final_loss: f32) -> NonlinearCheckpointProbeResult {
        NonlinearCheckpointProbeResult {
            uninterrupted: nonlinear_optimizer_result(0.95),
            resumed: nonlinear_optimizer_result(resumed_final_loss),
            model_checkpoint_bytes: 300,
            optimizer_checkpoint_bytes: 400,
        }
    }

    fn nonlinear_optimizer_result(final_loss: f32) -> NonlinearOptimizerProbeResult {
        NonlinearOptimizerProbeResult {
            losses: vec![1.0, final_loss],
            final_parameters: vec![
                0.2, -0.3, 0.4, -0.1, 0.5, 0.3, 0.1, -0.2, 0.05, 0.3, -0.4, 0.2, 0.0,
            ],
        }
    }

    fn minibatch_checkpoint_result(final_loss: f32) -> MiniBatchCheckpointProbeResult {
        MiniBatchCheckpointProbeResult {
            uninterrupted: minibatch_optimizer_result(0.95),
            resumed: minibatch_optimizer_result(final_loss),
            model_checkpoint_bytes: 300,
            optimizer_checkpoint_bytes: 400,
            data_checkpoint_bytes: 100,
            checkpoint_data_position: 2,
        }
    }

    fn minibatch_optimizer_result(final_loss: f32) -> MiniBatchOptimizerProbeResult {
        MiniBatchOptimizerProbeResult {
            losses: vec![1.0, final_loss],
            batch_sequence: vec![0, 1],
            final_parameters: vec![
                0.2, -0.3, 0.4, -0.1, 0.5, 0.3, 0.1, -0.2, 0.05, 0.3, -0.4, 0.2, 0.0,
            ],
            final_data_position: 2,
        }
    }
}
