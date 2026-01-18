// gpu_sift/mod.rs
// WebGPU-based SIFT implementation

use crate::keypoints::KeyPoint;
use crate::utils::*;
use std::sync::{Arc, Mutex};
use wgpu;

// ===== Configuration =====
pub struct GpuSiftConfig {
    pub octaves: u32,
    pub scales: u32,             // includes +2 extra scales (e.g., 5 total)
    pub base_sigma: f32,         // initial blur (e.g., 1.6)
    pub contrast_threshold: f32, // e.g., 0.03
    pub edge_threshold: f32,     // e.g., 10.0
}

impl Default for GpuSiftConfig {
    fn default() -> Self {
        Self {
            octaves: 4,
            scales: 5, // 5 scales → 4 DoG layers
            base_sigma: 1.6,
            contrast_threshold: 0.04, // Match CPU default (0.04 / num_intervals)
            edge_threshold: 10.0,
        }
    }
}

// ===== GPU Resources =====
pub struct GpuSiftContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: GpuPipelines,
    #[allow(dead_code)]
    kernels: GpuKernels,
    buffers: Mutex<GpuSiftBuffers>,
    #[allow(dead_code)]
    config: GpuSiftConfig,
}

#[allow(dead_code)]
struct GpuPipelines {
    upload: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    dog: wgpu::ComputePipeline,
    extrema: wgpu::ComputePipeline,
    orientation: wgpu::ComputePipeline,
    descriptor: wgpu::ComputePipeline,
}

#[allow(dead_code)]
struct GpuKernels {
    // Precomputed Gaussian kernels for each scale
    kernels: Vec<Vec<f32>>, // kernels[scale_idx] = weights
}

struct GpuSiftBuffers {
    // Pyramid heap
    heap: wgpu::Buffer,
    heap_capacity: u64,

    // Metadata
    meta_buffer: wgpu::Buffer,
    level_offsets: wgpu::Buffer,
    level_widths: wgpu::Buffer,
    level_heights: wgpu::Buffer,

    // Gaussian kernel weights (one buffer per scale)
    #[allow(dead_code)]
    kernel_buffers: Vec<wgpu::Buffer>,

    // Keypoint buffers
    extrema_counter: wgpu::Buffer,
    keypoints_staging: wgpu::Buffer,
    orientation_counter: wgpu::Buffer,
    keypoints_final: wgpu::Buffer,
    descriptors: wgpu::Buffer,

    // Readback buffers
    readback_counters: wgpu::Buffer,
    readback_keypoints: wgpu::Buffer,
    readback_descriptors: wgpu::Buffer,

    // Current image dimensions
    current_width: u32,
    current_height: u32,
}

// Lightweight clone for async operations (no mutex held)
struct GpuRunContext {
    heap: wgpu::Buffer,
    meta_buffer: wgpu::Buffer,
    level_offsets: wgpu::Buffer,
    level_widths: wgpu::Buffer,
    level_heights: wgpu::Buffer,
    #[allow(dead_code)]
    kernel_buffers: Vec<wgpu::Buffer>,
    extrema_counter: wgpu::Buffer,
    keypoints_staging: wgpu::Buffer,
    orientation_counter: wgpu::Buffer,
    keypoints_final: wgpu::Buffer,
    descriptors: wgpu::Buffer,
}

// ===== Public API =====
impl GpuSiftContext {
    pub async fn new(config: GpuSiftConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Request WebGPU device/queue
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await;

        let adapter = match adapter {
            Ok(a) => a,
            Err(_) => return Err("No suitable GPU adapter found".into()),
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("SIFT GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Precompute Gaussian kernels
        let kernels = Self::compute_kernels(&config);

        // Create compute pipelines
        let pipelines = Self::create_pipelines(&device)?;

        // Initialize empty buffers
        let mut buffers = GpuSiftBuffers::new(&device, 0, 0);

        // Initialize kernel weight buffers
        buffers.initialize_kernel_buffers(&device, &queue, &kernels);

        let buffers = Mutex::new(buffers);

        Ok(Self {
            device,
            queue,
            pipelines,
            kernels,
            buffers,
            config,
        })
    }

    pub async fn detect(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<KeyPoint>, Vec<[u8; 128]>), Box<dyn std::error::Error>> {
        let profile = std::env::var("SIFT_PROFILE").is_ok();
        let total_start = web_time::Instant::now();

        // 1. Ensure buffers are sized correctly
        let t0 = web_time::Instant::now();
        {
            let mut buffers = self.buffers.lock().unwrap();
            buffers.ensure_capacity(&self.device, width, height, &self.config);
        }
        if profile {
            eprintln!("  [GPU] Buffer setup: {:?}", t0.elapsed());
        }

        // 2. Clone buffer handles (release lock before async)
        let run_ctx = {
            let buffers = self.buffers.lock().unwrap();
            GpuRunContext {
                heap: buffers.heap.clone(),
                meta_buffer: buffers.meta_buffer.clone(),
                level_offsets: buffers.level_offsets.clone(),
                level_widths: buffers.level_widths.clone(),
                level_heights: buffers.level_heights.clone(),
                kernel_buffers: buffers.kernel_buffers.clone(),
                extrema_counter: buffers.extrema_counter.clone(),
                keypoints_staging: buffers.keypoints_staging.clone(),
                orientation_counter: buffers.orientation_counter.clone(),
                keypoints_final: buffers.keypoints_final.clone(),
                descriptors: buffers.descriptors.clone(),
            }
        };

        // 3. Build DoG pyramid on CPU (hybrid approach for now)
        let t1 = web_time::Instant::now();
        let gaussian_pyramid = self.build_pyramid_cpu(image, width, height);
        if profile {
            eprintln!("  [GPU] Gaussian pyramid (CPU): {:?}", t1.elapsed());
        }

        let t2 = web_time::Instant::now();
        let dog_pyramid = self.compute_dog_cpu(&gaussian_pyramid, width, height);
        if profile {
            eprintln!("  [GPU] DoG computation (CPU): {:?}", t2.elapsed());
        }

        // 4. Upload DoG pyramid to GPU
        let t3 = web_time::Instant::now();
        self.upload_dog_pyramid(&dog_pyramid, &run_ctx);
        if profile {
            eprintln!("  [GPU] Upload to GPU: {:?}", t3.elapsed());
        }

        // 5. Execute GPU pipeline (extrema detection, orientation, descriptors)
        let t4 = web_time::Instant::now();
        self.execute_pipeline(width, height, &run_ctx).await?;
        if profile {
            eprintln!("  [GPU] GPU pipeline: {:?}", t4.elapsed());
        }

        // 6. Readback results
        let t5 = web_time::Instant::now();
        let (keypoints, descriptors) = self.readback_results(&run_ctx).await?;
        if profile {
            eprintln!("  [GPU] Readback: {:?}", t5.elapsed());
            eprintln!("  [GPU] Total: {:?}", total_start.elapsed());
        }

        Ok((keypoints, descriptors))
    }

    fn compute_kernels(config: &GpuSiftConfig) -> GpuKernels {
        let mut kernels = Vec::new();
        let k = 2.0_f32.powf(1.0 / (config.scales as f32 - 2.0));

        for s in 0..config.scales {
            let sigma = config.base_sigma * k.powi(s as i32);
            let radius = (4.0 * sigma).ceil() as usize;
            let size = 2 * radius + 1;

            let mut weights = vec![0.0; size];
            let two_sigma_sq = 2.0 * sigma * sigma;
            let mut sum = 0.0;

            for (i, weight) in weights.iter_mut().enumerate() {
                let x = (i as f32) - (radius as f32);
                *weight = (-x * x / two_sigma_sq).exp();
                sum += *weight;
            }

            // Normalize
            for weight in weights.iter_mut() {
                *weight /= sum;
            }

            kernels.push(weights);
        }

        GpuKernels { kernels }
    }

    fn create_pipelines(device: &wgpu::Device) -> Result<GpuPipelines, Box<dyn std::error::Error>> {
        // Load shaders
        let upload_src = include_str!("shaders/upload.wgsl");
        let blur_src = include_str!("shaders/gaussian_blur.wgsl");
        let downsample_src = include_str!("shaders/downsample.wgsl");
        let dog_src = include_str!("shaders/dog.wgsl");
        let extrema_src = include_str!("shaders/extrema_detect.wgsl");
        let orientation_src = include_str!("shaders/orientation.wgsl");
        let descriptor_src = include_str!("shaders/descriptor.wgsl");

        let upload_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upload Shader"),
            source: wgpu::ShaderSource::Wgsl(upload_src.into()),
        });

        let blur_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(blur_src.into()),
        });

        let downsample_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(downsample_src.into()),
        });

        let dog_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoG Shader"),
            source: wgpu::ShaderSource::Wgsl(dog_src.into()),
        });

        let extrema_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Extrema Shader"),
            source: wgpu::ShaderSource::Wgsl(extrema_src.into()),
        });

        let orientation_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Orientation Shader"),
            source: wgpu::ShaderSource::Wgsl(orientation_src.into()),
        });

        let descriptor_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Descriptor Shader"),
            source: wgpu::ShaderSource::Wgsl(descriptor_src.into()),
        });

        // Create bind group layouts
        // Upload pipeline: @group(0) params (uniform), input_u8, heap
        let upload_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Upload BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Blur/downsample: @group(0) heap_in/out, @group(1) params + weights
        let blur_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // DoG: @group(0) header, params, level_offsets, level_widths, level_heights, heap_in
        //      @group(1) heap_out
        let dog_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoG BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let dog_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoG BGL 1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Extrema: @group(0) meta, @group(1) heap, @group(2) output
        let extrema_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Extrema BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let extrema_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Extrema BGL 1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let extrema_bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Extrema BGL 2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipelines
        let upload_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Upload Layout"),
            bind_group_layouts: &[&upload_bgl0],
            push_constant_ranges: &[],
        });

        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Layout"),
            bind_group_layouts: &[&blur_bgl0, &blur_bgl1],
            push_constant_ranges: &[],
        });

        let dog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoG Layout"),
            bind_group_layouts: &[&dog_bgl0, &dog_bgl1],
            push_constant_ranges: &[],
        });

        let extrema_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Extrema Layout"),
            bind_group_layouts: &[&extrema_bgl0, &extrema_bgl1, &extrema_bgl2],
            push_constant_ranges: &[],
        });

        let upload = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Upload Pipeline"),
            layout: Some(&upload_layout),
            module: &upload_module,
            entry_point: Some("upload_grayscale"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blur H Pipeline"),
            layout: Some(&blur_layout),
            module: &blur_module,
            entry_point: Some("gaussian_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blur V Pipeline"),
            layout: Some(&blur_layout),
            module: &blur_module,
            entry_point: Some("gaussian_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Downsample Pipeline"),
            layout: Some(&blur_layout),
            module: &downsample_module,
            entry_point: Some("downsample"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dog = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DoG Pipeline"),
            layout: Some(&dog_layout),
            module: &dog_module,
            entry_point: Some("compute_dog"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extrema = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extrema Pipeline"),
            layout: Some(&extrema_layout),
            module: &extrema_module,
            entry_point: Some("detect_extrema"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Orientation and descriptor pipelines (simplified, need proper layouts)
        let orientation = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Orientation Pipeline"),
            layout: None, // auto-layout for now
            module: &orientation_module,
            entry_point: Some("compute_orientation"),
            compilation_options: Default::default(),
            cache: None,
        });

        let descriptor = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Descriptor Pipeline"),
            layout: None, // auto-layout for now
            module: &descriptor_module,
            entry_point: Some("compute_descriptor"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(GpuPipelines {
            upload,
            blur_h,
            blur_v,
            downsample,
            dog,
            extrema,
            orientation,
            descriptor,
        })
    }

    #[allow(dead_code)]
    fn upload_image(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        ctx: &GpuRunContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Write image data to a temporary staging area at the END of heap
        // The upload shader will convert u8->f16 and write to the beginning
        let image_size = (width * height) as usize;
        let staging_offset = ctx.heap.size() as usize - ((image_size + 3) / 4) * 4; // Aligned

        // Create a padded buffer for u8 data (4-byte aligned)
        let mut padded_image = vec![0u8; ((image_size + 3) / 4) * 4];
        padded_image[..image_size].copy_from_slice(image);

        self.queue
            .write_buffer(&ctx.heap, staging_offset as u64, &padded_image);
        Ok(())
    }

    /// Build Gaussian scale space on CPU and upload to GPU
    /// This is a hybrid approach: CPU builds pyramid, GPU does extrema detection
    /// Uses INCREMENTAL blurring for efficiency (each scale from previous)
    fn build_pyramid_cpu(&self, image: &[u8], width: u32, height: u32) -> Vec<f32> {
        // k is the scale multiplier between adjacent scales
        // Standard SIFT uses scales-3 intervals
        let intervals = (self.config.scales as f32 - 3.0).max(1.0);
        let k = 2.0_f32.powf(1.0 / intervals);

        // Precompute differential sigmas for incremental blurring
        // sigma_total[s] = base_sigma * k^s
        // sigma_diff[s] = sqrt(sigma_total[s]^2 - sigma_total[s-1]^2)
        let mut diff_sigmas = vec![0.0f32; self.config.scales as usize];
        let assumed_blur = 0.5f32; // assumed initial blur of input image

        for s in 0..self.config.scales as usize {
            if s == 0 {
                // First scale: blur from assumed_blur to base_sigma
                let sigma_target = self.config.base_sigma;
                if sigma_target > assumed_blur {
                    diff_sigmas[s] =
                        (sigma_target * sigma_target - assumed_blur * assumed_blur).sqrt();
                } else {
                    diff_sigmas[s] = 0.0;
                }
            } else {
                // Incremental blur from previous scale
                let sigma_prev = self.config.base_sigma * k.powi((s - 1) as i32);
                let sigma_curr = self.config.base_sigma * k.powi(s as i32);
                diff_sigmas[s] = (sigma_curr * sigma_curr - sigma_prev * sigma_prev).sqrt();
            }
        }

        let mut pyramid_data = Vec::new();
        let mut current_img: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();
        let mut w = width as usize;
        let mut h = height as usize;

        for octave in 0..self.config.octaves {
            if w < 8 || h < 8 {
                break;
            }

            // Build scales for this octave using incremental blur
            let mut octave_images: Vec<Vec<f32>> = Vec::with_capacity(self.config.scales as usize);

            for s in 0..self.config.scales as usize {
                let blurred = if s == 0 {
                    if octave == 0 && diff_sigmas[0] > 0.01 {
                        // First octave, first scale: blur from input
                        self.gaussian_blur_cpu(&current_img, w, h, diff_sigmas[0])
                    } else {
                        // Other octaves: first scale comes from downsampling (already at correct blur)
                        current_img.clone()
                    }
                } else {
                    // Incremental blur from previous scale in this octave
                    let prev_scale = &octave_images[s - 1];
                    if diff_sigmas[s] > 0.01 {
                        self.gaussian_blur_cpu(prev_scale, w, h, diff_sigmas[s])
                    } else {
                        prev_scale.clone()
                    }
                };

                pyramid_data.extend_from_slice(&blurred);
                octave_images.push(blurred);
            }

            // Downsample from scale (scales-3) for next octave
            // This is the scale with 2x the base blur
            let downsample_idx = (self.config.scales as usize).saturating_sub(3);
            current_img = self.downsample_2x(&octave_images[downsample_idx], w, h);
            w /= 2;
            h /= 2;
        }

        pyramid_data
    }

    /// Fast 2x downsample by taking every other pixel
    fn downsample_2x(&self, img: &[f32], width: usize, height: usize) -> Vec<f32> {
        let new_w = width / 2;
        let new_h = height / 2;
        let mut result = vec![0.0f32; new_w * new_h];

        result
            .par_chunks_mut(new_w)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..new_w {
                    row[x] = img[(y * 2) * width + (x * 2)];
                }
            });

        result
    }

    fn gaussian_blur_cpu(&self, img: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
        // Handle edge case of very small sigma
        if sigma < 0.1 {
            return img.to_vec();
        }

        // Use smaller radius (sigma * 2.5 is sufficient for SIFT, saves computation)
        let radius = (sigma * 2.5).ceil() as i32;
        let size = (2 * radius + 1).max(1) as usize;

        // Build symmetric kernel (only store half + center)
        let mut kernel = vec![0.0f32; size];
        let mut sum = 0.0f32;
        let two_sigma_sq = 2.0 * sigma * sigma;
        for i in 0..size {
            let x = (i as i32 - radius) as f32;
            kernel[i] = (-x * x / two_sigma_sq).exp();
            sum += kernel[i];
        }
        let norm = 1.0 / sum;
        for k in kernel.iter_mut() {
            *k *= norm;
        }

        // For small kernels, use simple approach
        // For larger kernels, use optimized symmetric approach
        if size <= 5 {
            return self.gaussian_blur_simple(img, width, height, &kernel, radius);
        }

        // Optimized: exploit symmetry - kernel[i] == kernel[size-1-i]
        // Horizontal pass - parallel over rows
        let mut temp = vec![0.0f32; width * height];
        temp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            let row_start = y * width;
            for x in 0..width {
                // Center weight
                let mut val = img[row_start + x] * kernel[radius as usize];

                // Symmetric pairs
                for i in 1..=radius as usize {
                    let left = if x >= i { x - i } else { 0 };
                    let right = (x + i).min(width - 1);
                    val += (img[row_start + left] + img[row_start + right])
                        * kernel[radius as usize + i];
                }
                row[x] = val;
            }
        });

        // Vertical pass - parallel over rows (better cache locality)
        let mut result = vec![0.0f32; width * height];

        // Process in chunks for better cache utilization
        let chunk_height = 64.min(height);
        result
            .par_chunks_mut(chunk_height * width)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let y_start = chunk_idx * chunk_height;
                let y_end: usize = (y_start + chunk_height).min(height); // Fix E0282

                for local_y in 0..(y_end - y_start) {
                    let y = y_start + local_y;
                    let row_offset = local_y * width;

                    for x in 0..width {
                        // Center weight
                        let mut val = temp[y * width + x] * kernel[radius as usize];

                        // Symmetric pairs
                        for i in 1..=radius as usize {
                            let top = if y >= i { y - i } else { 0 };
                            let bottom: usize = (y + i).min(height - 1); // Fix E0282
                            val += (temp[top * width + x] + temp[bottom * width + x])
                                * kernel[radius as usize + i];
                        }
                        chunk[row_offset + x] = val;
                    }
                }
            });

        result
    }

    /// Simple blur for small kernels (avoids overhead of symmetric optimization)
    fn gaussian_blur_simple(
        &self,
        img: &[f32],
        width: usize,
        height: usize,
        kernel: &[f32],
        radius: i32,
    ) -> Vec<f32> {
        let size = kernel.len();

        // Horizontal pass
        let mut temp = vec![0.0f32; width * height];
        temp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for x in 0..width {
                let mut val = 0.0f32;
                for i in 0..size {
                    let sx = (x as i32 + i as i32 - radius).clamp(0, width as i32 - 1) as usize;
                    val += img[y * width + sx] * kernel[i];
                }
                row[x] = val;
            }
        });

        // Vertical pass
        let mut result = vec![0.0f32; width * height];
        result
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let mut val = 0.0f32;
                    for i in 0..size {
                        let sy =
                            (y as i32 + i as i32 - radius).clamp(0, height as i32 - 1) as usize;
                        val += temp[sy * width + x] * kernel[i];
                    }
                    row[x] = val;
                }
            });

        result
    }

    /// Compute DoG from Gaussian pyramid (parallelized)
    fn compute_dog_cpu(&self, gaussian_pyramid: &[f32], width: u32, height: u32) -> Vec<f32> {
        let scales = self.config.scales as usize;
        let dog_scales = scales - 1;

        // First pass: collect octave info
        let mut octave_info = Vec::new();
        let mut w = width as usize;
        let mut h = height as usize;
        let mut offset = 0usize;

        for _ in 0..self.config.octaves {
            if w < 8 || h < 8 {
                break;
            }
            let level_size = w * h;
            octave_info.push((offset, level_size, w, h));
            offset += scales * level_size;
            w /= 2;
            h /= 2;
        }

        // Compute total DoG size
        let total_dog_size: usize = octave_info
            .iter()
            .map(|(_, level_size, _, _)| level_size * dog_scales)
            .sum();

        let mut dog_data = vec![0.0f32; total_dog_size];

        // Parallel computation of DoG for each octave
        let mut dog_offset = 0usize;
        for (gauss_offset, level_size, _, _) in &octave_info {
            for d in 0..dog_scales {
                let scale1_start = gauss_offset + d * level_size;
                let scale2_start = gauss_offset + (d + 1) * level_size;
                let dog_start = dog_offset + d * level_size;

                dog_data[dog_start..dog_start + level_size]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, dog_val)| {
                        *dog_val =
                            gaussian_pyramid[scale2_start + i] - gaussian_pyramid[scale1_start + i];
                    });
            }
            dog_offset += dog_scales * level_size;
        }

        dog_data
    }

    /// Upload DoG pyramid to GPU in f16 format (parallelized conversion)
    fn upload_dog_pyramid(&self, dog_data: &[f32], ctx: &GpuRunContext) {
        // Convert f32 to f16 packed as u32 (parallel)
        let packed_data: Vec<u32> = dog_data
            .par_chunks(2)
            .map(|chunk| {
                let v0 = chunk[0];
                let v1 = if chunk.len() > 1 { chunk[1] } else { 0.0 };
                half::f16::from_f32(v0).to_bits() as u32
                    | ((half::f16::from_f32(v1).to_bits() as u32) << 16)
            })
            .collect();

        let bytes: Vec<u8> = packed_data
            .iter()
            .flat_map(|v: &u32| v.to_le_bytes())
            .collect(); // Fix E0282
        self.queue.write_buffer(&ctx.heap, 0, &bytes);
    }

    async fn execute_pipeline(
        &self,
        width: u32,
        height: u32,
        ctx: &GpuRunContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // DoG pyramid has (scales-1) layers per octave
        let dog_scales = self.config.scales - 1;

        // Compute metadata: level offsets, widths, heights for DoG pyramid
        let mut level_offsets_data = Vec::new();
        let mut level_widths_data = Vec::new();
        let mut level_heights_data = Vec::new();

        let mut offset = 0u32;
        let mut w = width;
        let mut h = height;
        let mut actual_octaves = 0u32;

        for octave in 0..self.config.octaves {
            if w < 8 || h < 8 {
                break;
            }
            actual_octaves = octave + 1;

            // DoG has (scales-1) layers per octave
            for _scale in 0..dog_scales {
                level_offsets_data.push(offset);
                level_widths_data.push(w);
                level_heights_data.push(h);

                let pixels = w * h;
                offset += (pixels + 1) / 2; // f16 packed as u32
            }

            w /= 2;
            h /= 2;
        }

        // Write metadata to GPU
        let offsets_bytes: Vec<u8> = level_offsets_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let widths_bytes: Vec<u8> = level_widths_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let heights_bytes: Vec<u8> = level_heights_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        self.queue
            .write_buffer(&ctx.level_offsets, 0, &offsets_bytes);
        self.queue.write_buffer(&ctx.level_widths, 0, &widths_bytes);
        self.queue
            .write_buffer(&ctx.level_heights, 0, &heights_bytes);

        // Write pyramid metadata
        // Note: extrema shader expects dog_scales (scales-1), not scales
        let meta_data = [
            actual_octaves,
            dog_scales,     // Number of DoG scales per octave
            dog_scales - 2, // Usable DoG layers for extrema (need 3 adjacent)
            width,
            height,
            self.config.base_sigma.to_bits(),
            self.config.contrast_threshold.to_bits(),
            self.config.edge_threshold.to_bits(),
        ];
        let meta_bytes: Vec<u8> = meta_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&ctx.meta_buffer, 0, &meta_bytes);

        // Clear counters
        self.queue.write_buffer(&ctx.extrema_counter, 0, &[0u8; 4]);
        self.queue
            .write_buffer(&ctx.orientation_counter, 0, &[0u8; 4]);

        // Create all bind groups upfront
        // ===== Extrema Detection Bind Groups =====
        let extrema_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extrema BG0"),
            layout: &self.pipelines.extrema.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.level_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ctx.level_widths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ctx.level_heights.as_entire_binding(),
                },
            ],
        });

        let extrema_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extrema BG1"),
            layout: &self.pipelines.extrema.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ctx.heap.as_entire_binding(),
            }],
        });

        let extrema_bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extrema BG2"),
            layout: &self.pipelines.extrema.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.extrema_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.keypoints_staging.as_entire_binding(),
                },
            ],
        });

        // ===== Orientation Bind Groups =====
        let orient_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Orientation BG0"),
            layout: &self.pipelines.orientation.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.level_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ctx.level_widths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ctx.level_heights.as_entire_binding(),
                },
            ],
        });

        let orient_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Orientation BG1"),
            layout: &self.pipelines.orientation.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ctx.heap.as_entire_binding(),
            }],
        });

        let orient_bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Orientation BG2"),
            layout: &self.pipelines.orientation.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.keypoints_staging.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.extrema_counter.as_entire_binding(),
                },
            ],
        });

        let orient_bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Orientation BG3"),
            layout: &self.pipelines.orientation.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.orientation_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.keypoints_final.as_entire_binding(),
                },
            ],
        });

        // ===== Descriptor Bind Groups =====
        let desc_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Descriptor BG0"),
            layout: &self.pipelines.descriptor.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.level_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ctx.level_widths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ctx.level_heights.as_entire_binding(),
                },
            ],
        });

        let desc_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Descriptor BG1"),
            layout: &self.pipelines.descriptor.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ctx.heap.as_entire_binding(),
            }],
        });

        let desc_bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Descriptor BG2"),
            layout: &self.pipelines.descriptor.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.keypoints_final.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.orientation_counter.as_entire_binding(),
                },
            ],
        });

        let desc_bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Descriptor BG3"),
            layout: &self.pipelines.descriptor.get_bind_group_layout(3),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ctx.descriptors.as_entire_binding(),
            }],
        });

        // ===== SINGLE ENCODER - All stages in one command buffer =====
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SIFT Full Pipeline Encoder"),
            });

        // ===== STAGE 1: Extrema Detection =====
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Extrema Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.extrema);
            compute_pass.set_bind_group(0, &extrema_bg0, &[]);
            compute_pass.set_bind_group(1, &extrema_bg1, &[]);
            compute_pass.set_bind_group(2, &extrema_bg2, &[]);

            // Calculate total z workgroups: octaves * (dog_scales - 2)
            let usable_dog_scales = dog_scales.saturating_sub(2).max(1);
            let total_z = actual_octaves * usable_dog_scales;

            let workgroups_x = (width + 15) / 16;
            let workgroups_y = (height + 15) / 16;

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, total_z);
        }
        // Pass ends here - implicit barrier between compute passes

        // ===== STAGE 2: Orientation Assignment =====
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Orientation Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.orientation);
            compute_pass.set_bind_group(0, &orient_bg0, &[]);
            compute_pass.set_bind_group(1, &orient_bg1, &[]);
            compute_pass.set_bind_group(2, &orient_bg2, &[]);
            compute_pass.set_bind_group(3, &orient_bg3, &[]);

            // Dispatch orientation computation
            let max_keypoints = 1024;
            let workgroups = (max_keypoints * 36 + 35) / 36;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        // Pass ends here - implicit barrier

        // ===== STAGE 3: Descriptor Computation =====
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Descriptor Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.descriptor);
            compute_pass.set_bind_group(0, &desc_bg0, &[]);
            compute_pass.set_bind_group(1, &desc_bg1, &[]);
            compute_pass.set_bind_group(2, &desc_bg2, &[]);
            compute_pass.set_bind_group(3, &desc_bg3, &[]);

            // Each descriptor needs 4 threads
            let max_final_keypoints = 2048;
            let workgroups = (max_final_keypoints * 4 + 3) / 4;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // SINGLE SUBMIT - all stages in one command buffer
        self.queue.submit(Some(encoder.finish()));

        // Wait for all GPU work to complete (only ONE wait)
        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        Ok(())
    }

    async fn readback_results(
        &self,
        ctx: &GpuRunContext,
    ) -> Result<(Vec<KeyPoint>, Vec<[u8; 128]>), Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        let buffers = self.buffers.lock().unwrap();

        // Copy counters
        encoder.copy_buffer_to_buffer(&ctx.extrema_counter, 0, &buffers.readback_counters, 0, 4);
        encoder.copy_buffer_to_buffer(
            &ctx.orientation_counter,
            0,
            &buffers.readback_counters,
            4,
            4,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read counters
        let counters_slice = buffers.readback_counters.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        counters_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        rx.recv()??;

        let counters_data = counters_slice.get_mapped_range();
        let orientation_count = u32::from_le_bytes([
            counters_data[4],
            counters_data[5],
            counters_data[6],
            counters_data[7],
        ]);
        drop(counters_data);
        buffers.readback_counters.unmap();

        let num_keypoints = orientation_count.min(65536);

        // Early return if no keypoints found
        if num_keypoints == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Copy keypoints and descriptors
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback KP Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &ctx.keypoints_final,
            0,
            &buffers.readback_keypoints,
            0,
            (num_keypoints as u64) * 16,
        );

        encoder.copy_buffer_to_buffer(
            &ctx.descriptors,
            0,
            &buffers.readback_descriptors,
            0,
            (num_keypoints as u64) * 128,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map keypoints
        let kp_slice = buffers
            .readback_keypoints
            .slice(..(num_keypoints as u64 * 16));
        let (tx, rx) = std::sync::mpsc::channel();
        kp_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        rx.recv()??;

        let kp_data = kp_slice.get_mapped_range();
        let mut keypoints = Vec::with_capacity(num_keypoints as usize);

        for i in 0..num_keypoints as usize {
            let offset = i * 16;
            let x = f32::from_le_bytes([
                kp_data[offset],
                kp_data[offset + 1],
                kp_data[offset + 2],
                kp_data[offset + 3],
            ]);
            let y = f32::from_le_bytes([
                kp_data[offset + 4],
                kp_data[offset + 5],
                kp_data[offset + 6],
                kp_data[offset + 7],
            ]);
            let size = f32::from_le_bytes([
                kp_data[offset + 8],
                kp_data[offset + 9],
                kp_data[offset + 10],
                kp_data[offset + 11],
            ]);
            let angle = f32::from_le_bytes([
                kp_data[offset + 12],
                kp_data[offset + 13],
                kp_data[offset + 14],
                kp_data[offset + 15],
            ]);

            keypoints.push(KeyPoint {
                x,
                y,
                size,
                angle,
                response: 0.0,
                octave: 0,
                layer: 0,
            });
        }
        drop(kp_data);
        buffers.readback_keypoints.unmap();

        // Map descriptors
        let desc_slice = buffers
            .readback_descriptors
            .slice(..(num_keypoints as u64 * 128));
        let (tx, rx) = std::sync::mpsc::channel();
        desc_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        rx.recv()??;

        let desc_data = desc_slice.get_mapped_range();
        let mut descriptors = Vec::with_capacity(num_keypoints as usize);

        for i in 0..num_keypoints as usize {
            let offset = i * 128;
            let mut desc = [0u8; 128];
            desc.copy_from_slice(&desc_data[offset..offset + 128]);
            descriptors.push(desc);
        }
        drop(desc_data);
        buffers.readback_descriptors.unmap();

        Ok((keypoints, descriptors))
    }
}

impl GpuSiftBuffers {
    fn new(device: &wgpu::Device, _width: u32, _height: u32) -> Self {
        // Initialize with minimal buffers
        let heap = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pyramid Heap"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Metadata"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let level_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Level Offsets"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let level_widths = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Level Widths"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let level_heights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Level Heights"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let extrema_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Extrema Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keypoints_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Keypoints Staging"),
            size: 32768 * 16, // 32768 keypoints × 4 f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let orientation_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orientation Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keypoints_final = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Keypoints Final"),
            size: 65536 * 16, // 65536 keypoints × 4 f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let descriptors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Descriptors"),
            size: 65536 * 128, // 65536 descriptors × 128 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Counters"),
            size: 8, // 2 u32 counters
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_keypoints = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Keypoints"),
            size: 65536 * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_descriptors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Descriptors"),
            size: 65536 * 128,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            heap,
            heap_capacity: 1024,
            meta_buffer,
            level_offsets,
            level_widths,
            level_heights,
            kernel_buffers: Vec::new(), // Will be initialized later with actual kernels
            extrema_counter,
            keypoints_staging,
            orientation_counter,
            keypoints_final,
            descriptors,
            readback_counters,
            readback_keypoints,
            readback_descriptors,
            current_width: 0,
            current_height: 0,
        }
    }

    fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        config: &GpuSiftConfig,
    ) {
        // Check if we need to reallocate
        if width == self.current_width && height == self.current_height {
            return;
        }

        // Compute required heap size
        let mut total_pixels = 0u64;
        let mut w = width;
        let mut h = height;

        for _ in 0..config.octaves {
            for _ in 0..config.scales {
                total_pixels += (w * h) as u64;
            }
            w /= 2;
            h /= 2;
            if w < 8 || h < 8 {
                break;
            }
        }

        let heap_size = total_pixels * 2; // 2 bytes per f16 pixel

        if heap_size > self.heap_capacity {
            // Reallocate heap
            self.heap = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pyramid Heap"),
                size: heap_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.heap_capacity = heap_size;
        }

        self.current_width = width;
        self.current_height = height;
    }

    fn initialize_kernel_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernels: &GpuKernels,
    ) {
        self.kernel_buffers = kernels
            .kernels
            .iter()
            .enumerate()
            .map(|(i, weights)| {
                let weights_bytes: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();

                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Kernel Weights {}", i)),
                    size: weights_bytes.len() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                queue.write_buffer(&buffer, 0, &weights_bytes);
                buffer
            })
            .collect();
    }
}
