// src/gpu_sift.rs

use crate::keypoints::KeyPoint;
use crate::sift::Sift; // Импортируем структуру Sift

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba};
use std::sync::{mpsc, Arc};
use wgpu::Adapter; // Используем Arc для Device/Queue, если нужно будет передавать

// --- Структура для параметров шейдера ---
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    width: u32,
    height: u32,
    sigma: f32,     // Sigma для Гаусса или другой параметр
    step_x: u32,    // 1 для горизонтального, 0 для вертикального/другого
    step_y: u32,    // 0 для горизонтального, 1 для вертикального/другого
    _padding1: u32, // Паддинг для выравнивания std140/std430
    _padding2: u32,
}

// GPU путь пока не задействован в основном pipeline, поэтому подавляем предупреждения о неиспользуемом коде.
#[allow(dead_code)]
struct GpuSiftContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    param_buffer: wgpu::Buffer, // Буфер для ComputeParams
    sampler: wgpu::Sampler,
    texture_bind_group_layout: wgpu::BindGroupLayout, // Лэйаут для (in_tex, out_tex, sampler, params)
    blur_pipeline_h: wgpu::ComputePipeline,           // Горизонтальный блюр
    blur_pipeline_v: wgpu::ComputePipeline,           // Вертикальный блюр
    subtract_pipeline: wgpu::ComputePipeline,         // Вычитание текстур
    downsample_pipeline: wgpu::ComputePipeline,       // Пайплайн для downsample (если нужно)
}

#[allow(dead_code)]
impl GpuSiftContext {
    async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default(); // Используем дефолтные бэкенды

        // Запрашиваем адаптер
        let adapter: Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await // Option<Adapter>
            .expect("Failed to find an appropriate adapter");

        let required_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

        // Дескриптор устройства для wgpu
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("SIFT GPU Device"),
            required_features: required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off, // Дефолтные лимиты
        };

        // Запрашиваем устройство
        let (device, queue) = adapter
            .request_device(&descriptor)
            .await
            .map_err(|e| format!("Failed to get device: {}", e))?;

        // --- Создание ресурсов ---

        // 1. Буфер для параметров
        let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SIFT Compute Params Buffer"),
            size: std::mem::size_of::<ComputeParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. Сэмплер (например, билинейный с зажимом по краям)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        // 3. Лэйаут для биндингов (одинаковый для всех наших пайплайнов) для float текстур
        // binding 0: params (uniform buffer)
        // binding 1: input texture 1 (texture_2d)
        // binding 2: input texture 2 / sampler (зависит от шейдера)
        // binding 3: output texture (storage texture)
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SIFT Texture Bind Group Layout"),
                entries: &[
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<ComputeParams>() as _,
                            ),
                        },
                        count: None,
                    },
                    // Input Texture 1
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
                    // Input Texture 2 (для вычитания) OR Sampler (для блюра/даунсэмплинга)
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
                    // Добавим сэмплер отдельно на binding 4
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, // Используем другой индекс
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Output Texture (Storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            // Формат должен совпадать с форматом создаваемых текстур
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let texture_format = wgpu::TextureFormat::Rgba32Float; // Храним float значения
        let features = adapter.get_texture_format_features(texture_format);
        if !features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::STORAGE_READ_WRITE)
        {
            return Err(format!(
                "GPU does not support writing to {:?} storage texture",
                texture_format
            ));
        }

        // 4. Загрузка WGSL шейдеров и создание пайплайнов
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SIFT Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout], // Используем наш лэйаут
            push_constant_ranges: &[],                         // Push константы не используем
        });

        // --- Шейдер для Гаусса (сепарабельный) ---
        // Мы передаем sigma и направление (step_x, step_y) в uniform
        // Радиус ядра определяется в шейдере на основе sigma
        // TODO: Написать gaussian_blur.wgsl
        let blur_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian_blur.wgsl"));

        let blur_pipeline_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Blur Pipeline H"),
            layout: Some(&pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("main_blur"), // Точка входа для блюра
            compilation_options: Default::default(),
            cache: None,
        });
        // Вертикальный пайплайн использует тот же шейдер, но другие параметры step_x/step_y
        let blur_pipeline_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Blur Pipeline V"),
            layout: Some(&pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("main_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Шейдер для вычитания ---
        // TODO: Написать subtract.wgsl
        let subtract_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("shaders/subtract.wgsl"));
        let subtract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Subtract Pipeline"),
            layout: Some(&pipeline_layout), // Используем тот же лэйаут
            module: &subtract_shader_module,
            entry_point: Some("main_subtract"), // Точка входа для вычитания
            compilation_options: Default::default(),
            cache: None,
        });

        // TODO: Создать downsample_pipeline аналогично
        let downsample_shader_module =
            device.create_shader_module(wgpu::include_wgsl!("shaders/downsample.wgsl"));
        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Downsample Pipeline"),
                layout: Some(&pipeline_layout), // Используем тот же лэйаут? (Да, если читает 1 текстуру и пишет в другую)
                module: &downsample_shader_module,
                entry_point: Some("main_downsample"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Возвращаем контекст ---
        Ok(GpuSiftContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            param_buffer,
            sampler,
            texture_bind_group_layout,
            blur_pipeline_h,
            blur_pipeline_v,
            subtract_pipeline,
            downsample_pipeline,
        })
    }

    async fn read_texture_to_imagebuffer(
        &self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<Rgba<f32>, Vec<f32>>, String> {
        let texture_format = texture.format();
        let bytes_per_pixel = match texture_format {
            wgpu::TextureFormat::Rgba32Float => 16,
            _ => return Err(format!("Unsupported texture format for readback: {:?}", texture_format)),
        };

        let padded_bytes_per_row =
            ((width * bytes_per_pixel as u32 + 255) / 256) * 256; // wgpu требует кратности 256
        let buffer_size = padded_bytes_per_row as wgpu::BufferAddress * height as wgpu::BufferAddress;
        let buffer_desc = wgpu::BufferDescriptor {
            label: Some("Texture Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        };
        let readback_buffer = self.device.create_buffer(&buffer_desc);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Texture Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                // Updated from ImageCopyBuffer
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    // Updated from ImageDataLayout
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            texture.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // --- Ожидание и чтение буфера с mpsc ---
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();

        // Запускаем маппинг с колбэком
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // Добавляем колбэк
            let _ = sender.send(result);
        });

        // Запускаем обработку GPU и ждем
        if let Err(e) = self.device.poll(wgpu::MaintainBase::Wait) {
            return Err(format!("Device poll failed: {:?}", e));
        }

        // Ожидаем результат из канала
        match receiver.recv() {
            // Убираем .await
            Ok(Ok(())) => {
                // Успешный маппинг
                let data = buffer_slice.get_mapped_range();
                let result_buffer = data.to_vec();
                drop(data);
                readback_buffer.unmap();

                // Учитываем паддинг при формировании выходного буфера
                let mut pixels: Vec<f32> = Vec::with_capacity((width * height * 4) as usize);
                for row in 0..height {
                    let start = (row * padded_bytes_per_row) as usize;
                    let end = start + (width * bytes_per_pixel as u32) as usize;
                    let row_slice = &result_buffer[start..end];
                    // Преобразуем u8 -> f32 (little endian)
                    let mut chunk = row_slice
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect::<Vec<f32>>();
                    pixels.append(&mut chunk);
                }

                match ImageBuffer::<Rgba<f32>, Vec<f32>>::from_raw(width, height, pixels) {
                    Some(image) => Ok(image),
                    None => Err("Failed to create ImageBuffer from raw data.".to_string()),
                }
            }
            Ok(Err(e)) => Err(format!("Failed to map buffer: {:?}", e)), // Ошибка маппинга
            Err(e) => Err(format!("Channel receive error: {:?}", e)),    // Ошибка канала
        }
    }
    // ... (convert_rgba8_to_f32_gray, convert_rgba8_to_luma8) ...
    fn convert_rgba_f32_to_luma_f32(
        img_rgba: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    ) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let (width, height) = img_rgba.dimensions();
        let mut img_f32 = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel_rgba = img_rgba.get_pixel(x, y);
                let gray = pixel_rgba[0];
                img_f32.put_pixel(x, y, Luma([gray]));
            }
        }
        img_f32
    }

    fn convert_luma_f32_to_u8(img_f32: &ImageBuffer<Luma<f32>, Vec<f32>>) -> GrayImage {
        let (width, height) = img_f32.dimensions();
        let mut img_u8 = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = img_f32.get_pixel(x, y)[0].clamp(0.0, 1.0);
                img_u8.put_pixel(x, y, Luma([(val * 255.0).round() as u8]));
            }
        }
        img_u8
    }

    // --- Функция для запуска compute shader ---
    fn run_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
        label: &str,
    ) {
        // Рассчитываем количество рабочих групп
        // TODO: Получить workgroup_size из пайплайна или задать константой
        let workgroup_size_x = 8;
        let workgroup_size_y = 8;
        let workgroups_x = (width + workgroup_size_x - 1) / workgroup_size_x;
        let workgroups_y = (height + workgroup_size_y - 1) / workgroup_size_y;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            // Начало compute pass
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None, // Таймстемпы пока не используем
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]); // group index 0
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1); // Запускаем
        } // Конец compute pass (дропаем compute_pass)

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    async fn build_pyramids_gpu(
        &self,
        base_image_dyn: &DynamicImage, // Принимаем DynamicImage
        num_octaves: u32,
        num_intervals: u32,
        initial_sigma: f32, // sigma базового изображения для 0-й октавы
        assumed_blur: f32,  // Пре-блюр исходного изображения
    ) -> Result<(Vec<Vec<wgpu::Texture>>, Vec<Vec<wgpu::Texture>>), String> {
        // 1. Подготовка базового изображения и параметров CPU
        let base_gray = base_image_dyn.to_luma8(); // Для получения размеров
        let (mut current_width, mut current_height) = base_gray.dimensions();

        // Применяем начальный блюр на CPU (пока что, идеально - делать на GPU)
        let initial_blur_amount = if initial_sigma > assumed_blur {
            (initial_sigma.powi(2) - assumed_blur.powi(2)).sqrt()
        } else {
            0.0
        };
        let base_image_blurred_cpu = if initial_blur_amount > 1e-4 {
            imageproc::filter::gaussian_blur_f32(&base_gray, initial_blur_amount)
        } else {
            base_gray.clone()
        };

        // Конвертируем в RGBA для загрузки на GPU (float)
        let base_rgba_u8 = DynamicImage::ImageLuma8(base_image_blurred_cpu).to_rgba8();
        let mut base_rgba_f32: Vec<f32> =
            Vec::with_capacity((base_rgba_u8.width() * base_rgba_u8.height() * 4) as usize);
        for (_x, _y, pixel) in base_rgba_u8.enumerate_pixels() {
            base_rgba_f32.push(pixel[0] as f32 / 255.0);
            base_rgba_f32.push(pixel[1] as f32 / 255.0);
            base_rgba_f32.push(pixel[2] as f32 / 255.0);
            base_rgba_f32.push(pixel[3] as f32 / 255.0);
        }

        let k = 2.0_f32.powf(1.0 / num_intervals as f32);
        // Сигмы *относительно начала текущей октавы*
        let mut octave_sigmas = Vec::with_capacity((num_intervals + 3) as usize);
        octave_sigmas.push(initial_sigma); // sigma для слоя 0
        for s_idx in 1..(num_intervals + 3) {
            octave_sigmas.push(octave_sigmas[s_idx as usize - 1] * k);
        }

        // Формат текстур
        let texture_format = wgpu::TextureFormat::Rgba32Float; // Храним float для точности/знака
        let texture_usage = wgpu::TextureUsages::TEXTURE_BINDING |
                              wgpu::TextureUsages::STORAGE_BINDING | // Для записи из шейдера
                              wgpu::TextureUsages::COPY_DST |       // Для write_texture и readback
                              wgpu::TextureUsages::COPY_SRC; // Для копирования между текстурами, если нужно

        // 2. Загрузка базового изображения на GPU
        let base_texture_desc = wgpu::TextureDescriptor {
            label: Some("Octave 0 Base Texture"),
            size: wgpu::Extent3d {
                width: current_width,
                height: current_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: texture_usage,
            view_formats: &[], // Для wgpu 0.19+
        };
        let mut current_octave_base_texture = self.device.create_texture(&base_texture_desc);

        let bytes_per_row = current_width * 16; // 4 компонента * 4 байта
        self.queue.write_texture(
            current_octave_base_texture.as_image_copy(),
            bytemuck::cast_slice(&base_rgba_f32), // Данные RGBA32F
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(current_height),
            },
            base_texture_desc.size,
        );

        let mut gaussian_pyramid_gpu: Vec<Vec<wgpu::Texture>> =
            Vec::with_capacity(num_octaves as usize);
        let mut dog_pyramid_gpu: Vec<Vec<wgpu::Texture>> = Vec::with_capacity(num_octaves as usize);

        // Переменная для хранения базовой текстуры для следующей октавы
        // Начинаем с первой загруженной текстуры
        let mut next_octave_base_texture =
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Initial Base Texture Copy"), // Копия для первой октавы
                size: base_texture_desc.size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture_format,
                usage: texture_usage,
                view_formats: &[],
            }));

        // Копируем начальное изображение в next_octave_base_texture
        let mut initial_copy_encoder = self.device.create_command_encoder(&Default::default());
        initial_copy_encoder.copy_texture_to_texture(
            current_octave_base_texture.as_image_copy(), // Используем только что загруженную
            next_octave_base_texture
                .as_ref()
                .expect("next_octave_base_texture is None")
                .as_image_copy(),
            base_texture_desc.size,
        );
        self.queue
            .submit(std::iter::once(initial_copy_encoder.finish()));

        for o_idx in 0..num_octaves {
            println!("GPU: Building Octave {}", o_idx);
            let mut current_gauss_octave: Vec<wgpu::Texture> =
                Vec::with_capacity((num_intervals + 3) as usize);
            let mut current_dog_octave: Vec<wgpu::Texture> =
                Vec::with_capacity((num_intervals + 2) as usize);

            // Начинаем октаву с базовой текстуры, подготовленной на предыдущем шаге
            current_octave_base_texture = next_octave_base_texture
                .take()
                .expect("next_octave_base_texture should not be None"); // Переносим владение
            let current_size = current_octave_base_texture.size();
            current_width = current_size.width;
            current_height = current_size.height;

            current_gauss_octave.push(current_octave_base_texture); // Добавляем базовую текстуру (слой 0)

            let mut prev_sigma_in_octave = initial_sigma;

            for s_idx in 1..(num_intervals + 3) {
                let target_sigma = octave_sigmas[s_idx as usize];
                let blur_sigma_step = (target_sigma.powi(2).max(prev_sigma_in_octave.powi(2))
                    - prev_sigma_in_octave.powi(2))
                .sqrt();
                let prev_gauss_texture = current_gauss_octave.last().unwrap();

                // Создаем выходную текстуру для текущего слоя Гаусса
                let gauss_texture_label = format!("Octave {} Gaussian {}", o_idx, s_idx);
                let current_gauss_texture_desc = wgpu::TextureDescriptor {
                    label: Some(&gauss_texture_label),
                    size: prev_gauss_texture.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: texture_format,
                    usage: texture_usage,
                    view_formats: &[],
                };
                let current_gauss_texture = self.device.create_texture(&current_gauss_texture_desc);

                // Создаем текстуру для DoG *заранее*, если она понадобится
                let dog_texture_label = format!("Octave {} DoG {}", o_idx, s_idx);
                let dog_texture: Option<wgpu::Texture> = if s_idx > 0 {
                    let dog_texture_desc = wgpu::TextureDescriptor {
                        label: Some(&dog_texture_label),
                        size: prev_gauss_texture.size(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        usage: texture_usage,
                        view_formats: &[],
                    };
                    Some(self.device.create_texture(&dog_texture_desc))
                } else {
                    None
                };

                // --- Выполнение блюра ---
                if blur_sigma_step > 1e-4 {
                    // Создаем временную текстуру для горизонтального прохода
                    // Используем ту же текстуру, что и для DoG, если она есть
                    let temp_blur_texture_label = format!("Octave {} Temp Blur {}", o_idx, s_idx);
                    let temp_blur_texture_desc = wgpu::TextureDescriptor {
                        label: Some(&temp_blur_texture_label),
                        size: prev_gauss_texture.size(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        usage: texture_usage,
                        view_formats: &[],
                    };
                    let temp_blur_texture = self.device.create_texture(&temp_blur_texture_desc);

                    // Горизонтальный проход
                    let params_h = ComputeParams {
                        /* ... */ width: current_width,
                        height: current_height,
                        sigma: blur_sigma_step,
                        step_x: 1,
                        step_y: 0,
                        _padding1: 0,
                        _padding2: 0,
                    };
                    self.queue
                        .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params_h));
                    let bind_group_h = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Blur H BindGroup O{} S{}", o_idx, s_idx)),
                        layout: &self.texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.param_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &prev_gauss_texture.create_view(&Default::default()),
                                ),
                            },
                            // Placeholder (same as input) to satisfy layout binding 2
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &prev_gauss_texture.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &temp_blur_texture.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                    self.run_compute(
                        &self.blur_pipeline_h,
                        &bind_group_h,
                        current_width,
                        current_height,
                        &format!("Blur H O{} S{}", o_idx, s_idx),
                    );

                    // Вертикальный проход
                    let params_v = ComputeParams {
                        /* ... */ width: current_width,
                        height: current_height,
                        sigma: blur_sigma_step,
                        step_x: 0,
                        step_y: 1,
                        _padding1: 0,
                        _padding2: 0,
                    };
                    self.queue
                        .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params_v));
                    let bind_group_v = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Blur V BindGroup O{} S{}", o_idx, s_idx)),
                        layout: &self.texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.param_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &temp_blur_texture.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &temp_blur_texture.create_view(&Default::default()),
                                ),
                            },
                            // !! Исправлено: Пишем в current_gauss_texture
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &current_gauss_texture.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                    self.run_compute(
                        &self.blur_pipeline_v,
                        &bind_group_v,
                        current_width,
                        current_height,
                        &format!("Blur V O{} S{}", o_idx, s_idx),
                    );
                } else {
                    // Копируем prev_gauss_texture в current_gauss_texture
                    println!("GPU: Copying texture for O{} S{}", o_idx, s_idx);
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some(&format!("Texture Copy O{} S{}", o_idx, s_idx)),
                            });
                    encoder.copy_texture_to_texture(
                        prev_gauss_texture.as_image_copy(),    // Source
                        current_gauss_texture.as_image_copy(), // Destination
                        prev_gauss_texture.size(),             // Extent
                    );
                    self.queue.submit(std::iter::once(encoder.finish()));
                }

                // --- Вычисление DoG ---
                if let Some(dog_tex) = &dog_texture {
                    // Используем dog_tex (текстуру, созданную ранее)
                    let params_sub = ComputeParams {
                        /* ... */ width: current_width,
                        height: current_height,
                        sigma: 0.0,
                        step_x: 0,
                        step_y: 0,
                        _padding1: 0,
                        _padding2: 0,
                    };
                    self.queue
                        .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params_sub));
                    let bind_group_sub =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("Subtract BindGroup O{} S{}", o_idx, s_idx - 1)),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.param_buffer.as_entire_binding(),
                                },
                                // !! Исправлено: Используем current_gauss_texture как Gauss[s]
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &current_gauss_texture.create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &prev_gauss_texture.create_view(&Default::default()),
                                    ),
                                }, // Gauss[s-1]
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &dog_tex.create_view(&Default::default()),
                                    ),
                                }, // Output DoG[s-1]
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                    self.run_compute(
                        &self.subtract_pipeline,
                        &bind_group_sub,
                        current_width,
                        current_height,
                        &format!("Subtract O{} S{}", o_idx, s_idx - 1),
                    );
                    // !! Исправлено: Добавляем саму dog_tex в список. Копирование не нужно.
                    current_dog_octave.push(dog_texture.unwrap()); // Мы знаем, что dog_texture is Some здесь
                }

                // Добавляем вычисленную Гауссову текстуру
                current_gauss_octave.push(current_gauss_texture);
                prev_sigma_in_octave = target_sigma;
            } // Конец цикла по s_idx

            gaussian_pyramid_gpu.push(current_gauss_octave);
            dog_pyramid_gpu.push(current_dog_octave);

            // --- Подготовка к следующей октаве: Downsample ---
            if o_idx < num_octaves - 1 {
                let downsample_source_idx = num_intervals as usize;
                // Проверяем, что индекс существует, прежде чем разыменовывать
                if let Some(source_texture) = gaussian_pyramid_gpu
                    .last()
                    .and_then(|octave| octave.get(downsample_source_idx))
                {
                    let next_width = current_width / 2;
                    let next_height = current_height / 2;
                    if next_width == 0 || next_height == 0 {
                        break;
                    }

                    // Создаем текстуру для следующей октавы
                    let next_octave_base_texture_label =
                        format!("Octave {} Base Texture", o_idx + 1);
                    let next_octave_base_texture_desc = wgpu::TextureDescriptor {
                        label: Some(&next_octave_base_texture_label),
                        size: wgpu::Extent3d {
                            width: next_width,
                            height: next_height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        usage: texture_usage,
                        view_formats: &[],
                    };
                    next_octave_base_texture =
                        Some(self.device.create_texture(&next_octave_base_texture_desc));

                    println!("GPU: Downsampling for Octave {}", o_idx + 1);
                    let params_down = ComputeParams {
                        /* ... */ width: next_width,
                        height: next_height,
                        sigma: 0.0,
                        step_x: 0,
                        step_y: 0,
                        _padding1: 0,
                        _padding2: 0,
                    };
                    self.queue.write_buffer(
                        &self.param_buffer,
                        0,
                        bytemuck::bytes_of(&params_down),
                    );
                    let bind_group_down =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("Downsample BindGroup O{}", o_idx + 1)),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.param_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &source_texture.create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &source_texture.create_view(&Default::default()),
                                    ),
                                },
                                // !! Исправлено: Пишем в next_octave_base_texture
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &next_octave_base_texture
                                            .as_ref()
                                            .expect("next_octave_base_texture is None")
                                            .create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                    self.run_compute(
                        &self.downsample_pipeline,
                        &bind_group_down,
                        next_width,
                        next_height,
                        &format!("Downsample O{}", o_idx + 1),
                    );
                    // current_width и current_height обновятся на следующей итерации o_idx
                } else {
                    eprintln!("Error: Could not get source texture for downsampling at octave {}, layer index {}", o_idx, downsample_source_idx);
                    break; // Прерываем, если не можем получить текстуру
                }
            } else {
                // Для последней октавы нам не нужна следующая базовая текстура
                // Можно здесь очистить next_octave_base_texture, если управление памятью важно
            }
        } // Конец цикла по o_idx
        Ok((gaussian_pyramid_gpu, dog_pyramid_gpu))
    }
}

// --- Обновленная функция верхнего уровня ---
pub fn sift_detect_and_compute_gpu(
    img: &DynamicImage,
    params: &Sift,
) -> Result<(Vec<KeyPoint>, Vec<Vec<f32>>), String> {
    let _ = env_logger::try_init();

    pollster::block_on(async {
        let gpu_context = GpuSiftContext::new().await?;

        let gray_img = img.to_luma8();
        let initial_blur_amount = if params.sigma > params.assumed_blur {
            (params.sigma.powi(2) - params.assumed_blur.powi(2)).sqrt()
        } else {
            0.0
        };
        let base_image_cpu = if initial_blur_amount > 1e-4 {
            imageproc::filter::gaussian_blur_f32(&gray_img, initial_blur_amount)
        } else {
            gray_img.clone()
        };
        let base_image_dyn = DynamicImage::ImageLuma8(base_image_cpu);

        // Заглушка для gpu_build_pyramids
        let (gpu_gauss_pyramid, gpu_dog_pyramid) = gpu_context
            .build_pyramids_gpu(
                &base_image_dyn,
                params.num_octaves,
                params.num_intervals,
                params.sigma,
                params.assumed_blur,
            )
            .await?;

        let mut cpu_gauss_pyramid: Vec<Vec<GrayImage>> = Vec::new();
        let mut cpu_dog_pyramid: Vec<Vec<ImageBuffer<Luma<f32>, Vec<f32>>>> = Vec::new();

        // --- Чтение пирамид с GPU (пустой цикл) ---
        for octave_textures in gpu_gauss_pyramid.iter() {
            let mut cpu_octave: Vec<GrayImage> = Vec::new();
            for texture in octave_textures {
                let (w, h) = (texture.width(), texture.height());
                let rgba_image = gpu_context
                    .read_texture_to_imagebuffer(texture, w, h)
                    .await?;
                let luma_f32 = GpuSiftContext::convert_rgba_f32_to_luma_f32(&rgba_image);
                cpu_octave.push(GpuSiftContext::convert_luma_f32_to_u8(&luma_f32));
            }
            cpu_gauss_pyramid.push(cpu_octave);
        }
        for octave_textures in gpu_dog_pyramid.iter() {
            let mut cpu_octave: Vec<ImageBuffer<Luma<f32>, Vec<f32>>> = Vec::new();
            for texture in octave_textures {
                let (w, h) = (texture.width(), texture.height());
                let rgba_image = gpu_context
                    .read_texture_to_imagebuffer(texture, w, h)
                    .await?;
                cpu_octave.push(GpuSiftContext::convert_rgba_f32_to_luma_f32(&rgba_image));
            }
            cpu_dog_pyramid.push(cpu_octave);
        }
        // --- Конец чтения ---

        let initial_keypoints = params.find_scale_space_extrema(&cpu_dog_pyramid);
        let refined_keypoints =
            params.refine_and_filter_extrema(&initial_keypoints, &cpu_dog_pyramid);
        let oriented_keypoints =
            params.assign_orientations(&refined_keypoints, &cpu_gauss_pyramid);
        let descriptors = params.compute(&cpu_gauss_pyramid, &oriented_keypoints);

        Ok((oriented_keypoints, descriptors))
    })
}
