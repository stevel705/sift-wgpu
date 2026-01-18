// Full GPU SIFT implementation using texture-based pipeline
// Targeting <20ms performance like VulkanSift

use crate::keypoints::KeyPoint;
use std::sync::Arc;
use wgpu;

pub struct GpuSiftConfigV2 {
    pub octaves: u32,
    pub scales_per_octave: u32, // S in SIFT paper (typically 3)
    pub base_sigma: f32,
    pub contrast_threshold: f32,
    pub edge_threshold: f32,
    pub max_keypoints: u32, // Maximum number of keypoints to detect
}

impl Default for GpuSiftConfigV2 {
    fn default() -> Self {
        Self {
            octaves: 4,
            scales_per_octave: 3, // Results in S+3=6 Gaussian images per octave
            base_sigma: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            max_keypoints: 4096, // Reasonable default for most images
        }
    }
}

/// Full GPU SIFT using texture-based pipeline
pub struct GpuSiftV2 {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Pipelines
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    dog_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    extrema_pipeline: wgpu::ComputePipeline,
    orientation_pipeline: wgpu::ComputePipeline,
    descriptor_pipeline: wgpu::ComputePipeline,
    prepare_orient_indirect_pipeline: wgpu::ComputePipeline,
    prepare_desc_indirect_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    blur_bind_group_layout: wgpu::BindGroupLayout,
    dog_bind_group_layout: wgpu::BindGroupLayout,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    extrema_bind_group_layout: wgpu::BindGroupLayout,
    prepare_indirect_bind_group_layout: wgpu::BindGroupLayout,

    // Sampler
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,

    // Preallocated resources (resized per image)
    resources: Option<GpuResources>,

    config: GpuSiftConfigV2,
}

struct GpuResources {
    width: u32,
    height: u32,

    // Gaussian pyramid textures: [octave][scale]
    // Each octave has scales_per_octave + 3 images
    gaussian_textures: Vec<Vec<wgpu::Texture>>,
    gaussian_views: Vec<Vec<wgpu::TextureView>>,

    // DoG pyramid: [octave][scale] - one less than Gaussian per octave
    dog_textures: Vec<Vec<wgpu::Texture>>,
    dog_views: Vec<Vec<wgpu::TextureView>>,

    // Temporary textures for blur passes
    temp_textures: Vec<wgpu::Texture>,
    temp_views: Vec<wgpu::TextureView>,

    // Keypoint buffers
    keypoint_counter: wgpu::Buffer,
    keypoints: wgpu::Buffer,
    oriented_keypoint_counter: wgpu::Buffer,
    oriented_keypoints: wgpu::Buffer,
    descriptors: wgpu::Buffer,

    // Indirect dispatch buffers (12 bytes each: x, y, z workgroups)
    orientation_indirect: wgpu::Buffer,
    descriptor_indirect: wgpu::Buffer,

    // Readback
    readback_counters: wgpu::Buffer,
    readback_keypoints: wgpu::Buffer,
    readback_descriptors: wgpu::Buffer,
}

impl GpuSiftV2 {
    pub async fn new(config: GpuSiftConfigV2) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::default();
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::Instance::default(); // Browser WebGPU default is usually fine (backends: BROWSER_WEBGPU)

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find an appropriate adapter: {:?}", e))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("SIFT V2 Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Failed to create device: {:?}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create samplers
        // Note: R32Float doesn't support filtering on most hardware, so use Nearest
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest, // R32Float doesn't support linear
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layouts
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blur BGL"),
                entries: &[
                    // Input texture - R32Float is not filterable by default on some hardware
                    // Use unfilterable sample type and separate sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler - use NonFiltering since R32Float doesn't support filtering
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // Output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Kernel weights
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
                ],
            });

        let dog_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoG BGL"),
                entries: &[
                    // Input texture 1 (higher blur)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Input texture 2 (lower blur)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let downsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Downsample BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let extrema_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Extrema BGL"),
                entries: &[
                    // DoG below
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // DoG current
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // DoG above
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Keypoint counter
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Keypoints output
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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

        // Load shaders
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_blur.wgsl").into()),
        });

        let dog_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoG Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_dog.wgsl").into()),
        });

        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Downsample Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_downsample.wgsl").into()),
        });

        let extrema_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Extrema Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_extrema.wgsl").into()),
        });

        let orientation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Orientation Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_orientation.wgsl").into()),
        });

        let descriptor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Descriptor Shader V2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_descriptor.wgsl").into()),
        });

        let prepare_indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prepare Indirect Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/gpu_prepare_indirect.wgsl").into(),
            ),
        });

        // Prepare indirect bind group layout
        let prepare_indirect_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Prepare Indirect BGL"),
                entries: &[
                    // Counter (read via atomicLoad, needs read_write for atomic)
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
                    // Indirect dispatch buffer (write)
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

        // Create pipelines
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blur H Pipeline V2"),
            layout: Some(&blur_layout),
            module: &blur_shader,
            entry_point: Some("blur_horizontal"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blur V Pipeline V2"),
            layout: Some(&blur_layout),
            module: &blur_shader,
            entry_point: Some("blur_vertical"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoG Layout"),
            bind_group_layouts: &[&dog_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DoG Pipeline V2"),
            layout: Some(&dog_layout),
            module: &dog_shader,
            entry_point: Some("compute_dog"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Downsample Layout"),
            bind_group_layouts: &[&downsample_bind_group_layout],
            push_constant_ranges: &[],
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Downsample Pipeline V2"),
                layout: Some(&downsample_layout),
                module: &downsample_shader,
                entry_point: Some("downsample_2x"),
                compilation_options: Default::default(),
                cache: None,
            });

        let extrema_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Extrema Layout"),
            bind_group_layouts: &[&extrema_bind_group_layout],
            push_constant_ranges: &[],
        });

        let extrema_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extrema Pipeline V2"),
            layout: Some(&extrema_layout),
            module: &extrema_shader,
            entry_point: Some("detect_extrema"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Orientation and descriptor use auto-layout for now
        let orientation_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Orientation Pipeline V2"),
                layout: None,
                module: &orientation_shader,
                entry_point: Some("compute_orientation"),
                compilation_options: Default::default(),
                cache: None,
            });

        let descriptor_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Descriptor Pipeline V2"),
                layout: None,
                module: &descriptor_shader,
                entry_point: Some("compute_descriptor"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Prepare indirect dispatch pipelines
        let prepare_indirect_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Prepare Indirect Layout"),
                bind_group_layouts: &[&prepare_indirect_bind_group_layout],
                push_constant_ranges: &[],
            });

        let prepare_orient_indirect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Prepare Orient Indirect Pipeline"),
                layout: Some(&prepare_indirect_layout),
                module: &prepare_indirect_shader,
                entry_point: Some("prepare_orientation_indirect"),
                compilation_options: Default::default(),
                cache: None,
            });

        let prepare_desc_indirect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Prepare Desc Indirect Pipeline"),
                layout: Some(&prepare_indirect_layout),
                module: &prepare_indirect_shader,
                entry_point: Some("prepare_descriptor_indirect"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            blur_h_pipeline,
            blur_v_pipeline,
            dog_pipeline,
            downsample_pipeline,
            extrema_pipeline,
            orientation_pipeline,
            descriptor_pipeline,
            prepare_orient_indirect_pipeline,
            prepare_desc_indirect_pipeline,
            blur_bind_group_layout,
            dog_bind_group_layout,
            downsample_bind_group_layout,
            extrema_bind_group_layout,
            prepare_indirect_bind_group_layout,
            linear_sampler,
            nearest_sampler,
            resources: None,
            config,
        })
    }

    fn ensure_resources(&mut self, width: u32, height: u32) {
        if let Some(ref res) = self.resources {
            if res.width == width && res.height == height {
                return;
            }
        }

        let scales_per_octave = self.config.scales_per_octave + 3; // Total Gaussian images per octave
        let dog_scales = scales_per_octave - 1;

        let mut gaussian_textures = Vec::new();
        let mut gaussian_views = Vec::new();
        let mut dog_textures = Vec::new();
        let mut dog_views = Vec::new();
        let mut temp_textures = Vec::new();
        let mut temp_views = Vec::new();

        let mut w = width;
        let mut h = height;

        for octave in 0..self.config.octaves {
            if w < 8 || h < 8 {
                break;
            }

            let mut octave_gaussian = Vec::new();
            let mut octave_gaussian_views = Vec::new();
            let mut octave_dog = Vec::new();
            let mut octave_dog_views = Vec::new();

            // Gaussian textures for this octave
            for s in 0..scales_per_octave {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Gaussian O{}S{}", octave, s)),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                octave_gaussian.push(tex);
                octave_gaussian_views.push(view);
            }

            // DoG textures for this octave
            for d in 0..dog_scales {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("DoG O{}D{}", octave, d)),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                octave_dog.push(tex);
                octave_dog_views.push(view);
            }

            // Temp texture for blur intermediate
            let temp = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Temp O{}", octave)),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let temp_view = temp.create_view(&Default::default());
            temp_textures.push(temp);
            temp_views.push(temp_view);

            gaussian_textures.push(octave_gaussian);
            gaussian_views.push(octave_gaussian_views);
            dog_textures.push(octave_dog);
            dog_views.push(octave_dog_views);

            w /= 2;
            h /= 2;
        }

        // Keypoint buffers
        let max_keypoints = self.config.max_keypoints as u64;

        let keypoint_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Keypoint Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keypoints = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Keypoints"),
            size: max_keypoints * 32, // x, y, sigma, response, octave, scale, pad, pad
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let oriented_keypoint_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Oriented Keypoint Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let oriented_keypoints = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Oriented Keypoints"),
            size: max_keypoints * 2 * 16, // x, y, sigma, angle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let descriptors = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Descriptors"),
            size: max_keypoints * 2 * 128, // 128 bytes per descriptor
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Indirect dispatch buffers (12 bytes each: x, y, z as u32)
        let orientation_indirect = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orientation Indirect"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let descriptor_indirect = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Descriptor Indirect"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // Readback buffers
        let readback_counters = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Counters"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_keypoints = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Keypoints"),
            size: max_keypoints * 2 * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_descriptors = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Descriptors"),
            size: max_keypoints * 2 * 128,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.resources = Some(GpuResources {
            width,
            height,
            gaussian_textures,
            gaussian_views,
            dog_textures,
            dog_views,
            temp_textures,
            temp_views,
            keypoint_counter,
            keypoints,
            oriented_keypoint_counter,
            oriented_keypoints,
            descriptors,
            orientation_indirect,
            descriptor_indirect,
            readback_counters,
            readback_keypoints,
            readback_descriptors,
        });
    }

    pub async fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<KeyPoint>, Vec<[u8; 128]>), Box<dyn std::error::Error>> {
        let profile = std::env::var("SIFT_PROFILE").is_ok();
        let total_start = web_time::Instant::now();

        // Ensure resources
        let t0 = web_time::Instant::now();
        self.ensure_resources(width, height);
        if profile {
            eprintln!("  [GPU V2] Resource setup: {:?}", t0.elapsed());
        }

        let _res = self.resources.as_ref().unwrap();

        // Upload image to first Gaussian texture
        let t1 = web_time::Instant::now();
        self.upload_image(image, width, height)?;
        if profile {
            eprintln!("  [GPU V2] Upload: {:?}", t1.elapsed());
        }

        // Build Gaussian pyramid on GPU
        let t2 = web_time::Instant::now();
        self.build_gaussian_pyramid(width, height)?;
        if profile {
            eprintln!("  [GPU V2] Gaussian pyramid: {:?}", t2.elapsed());
        }

        // Compute DoG
        let t3 = web_time::Instant::now();
        self.compute_dog(width, height)?;
        if profile {
            eprintln!("  [GPU V2] DoG: {:?}", t3.elapsed());
        }

        // Detect extrema
        let t4 = web_time::Instant::now();
        self.detect_extrema(width, height)?;
        if profile {
            eprintln!("  [GPU V2] Extrema: {:?}", t4.elapsed());
        }

        // Orientation assignment
        let t5 = web_time::Instant::now();
        self.compute_orientation(width, height)?;
        if profile {
            eprintln!("  [GPU V2] Orientation: {:?}", t5.elapsed());
        }

        // Descriptors
        let t6 = web_time::Instant::now();
        self.compute_descriptors(width, height)?;
        if profile {
            eprintln!("  [GPU V2] Descriptors: {:?}", t6.elapsed());
        }

        // Readback
        let t7 = web_time::Instant::now();
        let result = self.readback_results().await?;
        if profile {
            eprintln!("  [GPU V2] Readback: {:?}", t7.elapsed());
            eprintln!("  [GPU V2] Total: {:?}", total_start.elapsed());
        }

        Ok(result)
    }

    fn upload_image(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        // Convert u8 to f32
        let image_f32: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();
        let bytes: Vec<u8> = image_f32.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &res.gaussian_textures[0][0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Helper: create a uniform buffer with initial data
    fn create_uniform_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Helper: create a storage buffer with initial data
    fn create_storage_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn build_gaussian_pyramid(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();
        let scales_per_octave = self.config.scales_per_octave + 3;

        // k is the multiplier between scales
        let k = 2.0_f32.powf(1.0 / self.config.scales_per_octave as f32);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gaussian Pyramid Encoder"),
            });

        let mut w = width;
        let mut h = height;

        for octave in 0..res.gaussian_textures.len() {
            if w < 8 || h < 8 {
                break;
            }

            for s in 0..scales_per_octave as usize {
                // Compute sigma for this scale
                let sigma = if s == 0 && octave == 0 {
                    // First scale of first octave: blur from assumed blur to base_sigma
                    let assumed_blur = 0.5f32;
                    if self.config.base_sigma > assumed_blur {
                        (self.config.base_sigma * self.config.base_sigma
                            - assumed_blur * assumed_blur)
                            .sqrt()
                    } else {
                        0.0
                    }
                } else if s == 0 {
                    // First scale of other octaves: already blurred from downsample
                    0.0
                } else {
                    // Incremental blur
                    let sigma_prev = self.config.base_sigma * k.powi((s - 1) as i32);
                    let sigma_curr = self.config.base_sigma * k.powi(s as i32);
                    (sigma_curr * sigma_curr - sigma_prev * sigma_prev).sqrt()
                };

                if sigma < 0.1 {
                    // Skip blur, just copy
                    if s > 0 {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &res.gaussian_textures[octave][s - 1],
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: &res.gaussian_textures[octave][s],
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: w,
                                height: h,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                    continue;
                }

                // Build kernel
                let radius = (sigma * 2.5).ceil() as i32;
                let kernel_size = (2 * radius + 1) as usize;
                let mut kernel = vec![0.0f32; kernel_size];
                let mut sum = 0.0f32;
                let two_sigma_sq = 2.0 * sigma * sigma;

                for (i, kv) in kernel.iter_mut().enumerate() {
                    let x = (i as i32 - radius) as f32;
                    *kv = (-x * x / two_sigma_sq).exp();
                    sum += *kv;
                }
                for kv in kernel.iter_mut() {
                    *kv /= sum;
                }

                // Create kernel buffer for this pass
                let kernel_bytes: Vec<u8> = kernel.iter().flat_map(|f| f.to_le_bytes()).collect();
                let kernel_buffer = self.create_storage_buffer(&kernel_bytes, "Kernel Buffer");

                // Source texture
                let src_view = if s == 0 {
                    if octave == 0 {
                        &res.gaussian_views[0][0]
                    } else {
                        &res.gaussian_views[octave][0]
                    }
                } else {
                    &res.gaussian_views[octave][s - 1]
                };

                // Horizontal pass - create params buffer for this specific pass
                let params_h = [w, h, radius as u32, 0u32]; // direction 0 = horizontal
                let params_h_bytes: Vec<u8> =
                    params_h.iter().flat_map(|v| v.to_le_bytes()).collect();
                let params_h_buffer = self.create_uniform_buffer(&params_h_bytes, "Blur H Params");

                let blur_h_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blur H BG"),
                    layout: &self.blur_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&res.temp_views[octave]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_h_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: kernel_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Blur H Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.blur_h_pipeline);
                    pass.set_bind_group(0, &blur_h_bg, &[]);
                    pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
                }

                // Vertical pass - create params buffer for this specific pass
                let params_v = [w, h, radius as u32, 1u32]; // direction 1 = vertical
                let params_v_bytes: Vec<u8> =
                    params_v.iter().flat_map(|v| v.to_le_bytes()).collect();
                let params_v_buffer = self.create_uniform_buffer(&params_v_bytes, "Blur V Params");

                let blur_v_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blur V BG"),
                    layout: &self.blur_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&res.temp_views[octave]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &res.gaussian_views[octave][s],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_v_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: kernel_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Blur V Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.blur_v_pipeline);
                    pass.set_bind_group(0, &blur_v_bg, &[]);
                    pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
                }
            }

            // Downsample for next octave
            if octave + 1 < res.gaussian_textures.len() {
                let next_w = w / 2;
                let next_h = h / 2;

                // Source from scale (scales_per_octave - 3) which has 2x base blur
                let src_scale = (scales_per_octave as usize).saturating_sub(3);

                let params = [w, h, next_w, next_h];
                let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
                let params_buffer = self.create_uniform_buffer(&params_bytes, "Downsample Params");

                let ds_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Downsample BG"),
                    layout: &self.downsample_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &res.gaussian_views[octave][src_scale],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &res.gaussian_views[octave + 1][0],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Downsample Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.downsample_pipeline);
                    pass.set_bind_group(0, &ds_bg, &[]);
                    pass.dispatch_workgroups((next_w + 15) / 16, (next_h + 15) / 16, 1);
                }
            }

            w /= 2;
            h /= 2;
        }

        self.queue.submit(Some(encoder.finish()));
        // No poll here - let GPU work continue
        Ok(())
    }

    fn compute_dog(&self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DoG Encoder"),
            });

        let mut w = width;
        let mut h = height;

        for octave in 0..res.dog_textures.len() {
            if w < 8 || h < 8 {
                break;
            }

            for d in 0..res.dog_views[octave].len() {
                let params = [w, h, 0u32, 0u32];
                let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
                let params_buffer = self.create_uniform_buffer(&params_bytes, "DoG Params");

                let dog_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DoG BG"),
                    layout: &self.dog_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &res.gaussian_views[octave][d + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &res.gaussian_views[octave][d],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&res.dog_views[octave][d]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("DoG Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.dog_pipeline);
                    pass.set_bind_group(0, &dog_bg, &[]);
                    pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
                }
            }

            w /= 2;
            h /= 2;
        }

        self.queue.submit(Some(encoder.finish()));
        // No poll here - let GPU work continue
        Ok(())
    }

    fn detect_extrema(&self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        // Clear counter
        self.queue.write_buffer(&res.keypoint_counter, 0, &[0u8; 4]);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Extrema Encoder"),
            });

        let mut w = width;
        let mut h = height;
        let k = 2.0_f32.powf(1.0 / self.config.scales_per_octave as f32);

        for octave in 0..res.dog_textures.len() {
            if w < 8 || h < 8 {
                break;
            }

            // Need 3 adjacent DoG scales for extrema detection
            for d in 1..(res.dog_views[octave].len() - 1) {
                let sigma = self.config.base_sigma * k.powi(d as i32) * (1 << octave) as f32;

                #[repr(C)]
                struct ExtremaParams {
                    width: u32,
                    height: u32,
                    octave: u32,
                    scale: u32,
                    contrast_threshold: f32,
                    edge_threshold: f32,
                    sigma: f32,
                    _pad: u32,
                }

                let params = ExtremaParams {
                    width: w,
                    height: h,
                    octave: octave as u32,
                    scale: d as u32,
                    // SIFT paper: threshold = 0.04 / num_intervals
                    contrast_threshold: self.config.contrast_threshold
                        / self.config.scales_per_octave as f32,
                    edge_threshold: self.config.edge_threshold,
                    sigma,
                    _pad: 0,
                };

                let params_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &params as *const _ as *const u8,
                        std::mem::size_of::<ExtremaParams>(),
                    )
                };
                let params_buffer = self.create_uniform_buffer(params_bytes, "Extrema Params");

                let extrema_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Extrema BG"),
                    layout: &self.extrema_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &res.dog_views[octave][d - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&res.dog_views[octave][d]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &res.dog_views[octave][d + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: res.keypoint_counter.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: res.keypoints.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Extrema Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.extrema_pipeline);
                    pass.set_bind_group(0, &extrema_bg, &[]);
                    pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
                }
            }

            w /= 2;
            h /= 2;
        }

        self.queue.submit(Some(encoder.finish()));
        // No poll - next stage uses indirect dispatch

        Ok(())
    }

    fn compute_orientation(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        // Clear output counter
        self.queue
            .write_buffer(&res.oriented_keypoint_counter, 0, &[0u8; 4]);

        // Create bind group for prepare indirect dispatch
        let prepare_indirect_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prepare Orient Indirect BG"),
            layout: &self.prepare_indirect_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.keypoint_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res.orientation_indirect.as_entire_binding(),
                },
            ],
        });

        // Max keypoints for params (shader will check bounds)
        #[repr(C)]
        struct OrientParams {
            width: u32,
            height: u32,
            octave: u32,
            num_keypoints: u32,
        }

        let params = OrientParams {
            width,
            height,
            octave: 0,
            num_keypoints: 32768, // Max, shader checks actual count
        };

        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const _ as *const u8,
                std::mem::size_of::<OrientParams>(),
            )
        };
        let params_buffer = self.create_uniform_buffer(params_bytes, "Orientation Params");

        // Use the first scale of first octave for gradient computation
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Orientation BG"),
            layout: &self.orientation_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&res.gaussian_views[0][0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: res.keypoints.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: res.oriented_keypoint_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res.oriented_keypoints.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Orientation Encoder"),
            });

        // First: prepare indirect dispatch buffer from keypoint counter
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prepare Orient Indirect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_orient_indirect_pipeline);
            pass.set_bind_group(0, &prepare_indirect_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Then: run orientation using indirect dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Orientation Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.orientation_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups_indirect(&res.orientation_indirect, 0);
        }

        self.queue.submit(Some(encoder.finish()));
        // No poll - descriptor stage uses indirect dispatch too

        Ok(())
    }

    fn compute_descriptors(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        // Create bind group for prepare indirect dispatch
        let prepare_indirect_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prepare Desc Indirect BG"),
            layout: &self.prepare_indirect_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.oriented_keypoint_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res.descriptor_indirect.as_entire_binding(),
                },
            ],
        });

        #[repr(C)]
        struct DescParams {
            width: u32,
            height: u32,
            octave: u32,
            num_keypoints: u32,
        }

        let params = DescParams {
            width,
            height,
            octave: 0,
            num_keypoints: 65536, // Max, shader checks actual count
        };

        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const _ as *const u8,
                std::mem::size_of::<DescParams>(),
            )
        };
        let params_buffer = self.create_uniform_buffer(params_bytes, "Descriptor Params");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Descriptor BG"),
            layout: &self.descriptor_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&res.gaussian_views[0][0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: res.oriented_keypoints.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: res.descriptors.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Descriptor Encoder"),
            });

        // First: prepare indirect dispatch buffer from oriented keypoint counter
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prepare Desc Indirect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_desc_indirect_pipeline);
            pass.set_bind_group(0, &prepare_indirect_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Then: run descriptors using indirect dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Descriptor Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.descriptor_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups_indirect(&res.descriptor_indirect, 0);
        }

        self.queue.submit(Some(encoder.finish()));
        // No poll - readback will sync

        Ok(())
    }

    async fn readback_results(
        &self,
    ) -> Result<(Vec<KeyPoint>, Vec<[u8; 128]>), Box<dyn std::error::Error>> {
        let res = self.resources.as_ref().unwrap();

        // Copy counter and all data in ONE submit
        let max_keypoints = self.config.max_keypoints as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        // Copy counter
        encoder.copy_buffer_to_buffer(
            &res.oriented_keypoint_counter,
            0,
            &res.readback_counters,
            0,
            4,
        );

        // Copy keypoints and descriptors
        encoder.copy_buffer_to_buffer(
            &res.oriented_keypoints,
            0,
            &res.readback_keypoints,
            0,
            max_keypoints * 16,
        );
        encoder.copy_buffer_to_buffer(
            &res.descriptors,
            0,
            &res.readback_descriptors,
            0,
            max_keypoints * 128,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map ALL buffers in parallel - single poll for everything
        let counter_slice = res.readback_counters.slice(..4);
        let kp_slice = res.readback_keypoints.slice(..(max_keypoints * 16));
        let desc_slice = res.readback_descriptors.slice(..(max_keypoints * 128));

        let (tx1, rx1) = futures::channel::oneshot::channel();
        let (tx2, rx2) = futures::channel::oneshot::channel();
        let (tx3, rx3) = futures::channel::oneshot::channel();

        counter_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx1.send(result);
        });
        kp_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx2.send(result);
        });
        desc_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx3.send(result);
        });

        // Ensure works are submitted and mapping process started
        // On native: Wait blocks until done.
        // On Web: Wait is no-op or poll. We need to await the channels to let event loop run.
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native polling usually requires Maintain::Wait but wgpu 25+ might have changed api.
            // For now commenting out to avoid IDE errors if Maintain is missing.
            // self.device.poll(wgpu::Maintain::Wait);
            // self.device.poll(wgpu::Maintain::Poll);
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Give the browser a chance to poll if needed, though await below does it implicitly
            // self.device.poll(wgpu::Maintain::Poll);
        }

        // Await results (non-blocking yield)
        rx1.await??;
        rx2.await??;
        rx3.await??;

        // Read counter first
        let counter_data = counter_slice.get_mapped_range();
        let num_keypoints = u32::from_le_bytes([
            counter_data[0],
            counter_data[1],
            counter_data[2],
            counter_data[3],
        ])
        .min(self.config.max_keypoints) as usize;
        drop(counter_data);
        res.readback_counters.unmap();

        if num_keypoints == 0 {
            res.readback_keypoints.unmap();
            res.readback_descriptors.unmap();
            return Ok((Vec::new(), Vec::new()));
        }

        // Read keypoints
        let kp_data = kp_slice.get_mapped_range();
        let mut keypoints = Vec::with_capacity(num_keypoints);

        for i in 0..num_keypoints {
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
            let sigma = f32::from_le_bytes([
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
                size: sigma * 2.0,
                angle,
                response: 0.0,
                octave: 0,
                layer: 0,
            });
        }
        drop(kp_data);
        res.readback_keypoints.unmap();

        // Read descriptors
        let desc_data = desc_slice.get_mapped_range();
        let mut descriptors = Vec::with_capacity(num_keypoints);

        for i in 0..num_keypoints {
            let offset = i * 128;
            let mut desc = [0u8; 128];
            desc.copy_from_slice(&desc_data[offset..offset + 128]);
            descriptors.push(desc);
        }
        drop(desc_data);
        res.readback_descriptors.unmap();

        Ok((keypoints, descriptors))
    }
}
