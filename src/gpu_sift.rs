// src/gpu_sift.rs

use crate::keypoints::KeyPoint;
use crate::sift::Sift; // Импортируем структуру Sift

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba};
use wgpu::Adapter;
use std::sync::{mpsc, Arc}; // Используем Arc для Device/Queue, если нужно будет передавать

struct GpuSiftContext {
    device: Arc<wgpu::Device>, // Используем Arc для удобства
    queue: Arc<wgpu::Queue>,
    // ... (пайплайны и т.д. будут добавлены позже)
}


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

        // Дескриптор устройства для wgpu 
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("SIFT GPU Device"),
            required_features: wgpu::Features::empty(), // Начнем с минимума, добавим если нужно
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off, // Дефолтные лимиты
        };

        // Запрашиваем устройство
        let (device, queue) = adapter
            .request_device(
                &descriptor,
            )
            .await
            .map_err(|e| format!("Failed to get device: {}", e))?;

        Ok(GpuSiftContext {
             device: Arc::new(device),
             queue: Arc::new(queue),
        })
    }

    async fn build_pyramids_gpu(
        &self,
        _base_image: &DynamicImage,
        _num_octaves: u32,
        _num_intervals: u32,
        _initial_sigma: f32,
    ) -> Result<(Vec<Vec<wgpu::Texture>>, Vec<Vec<wgpu::Texture>>), String>
    {
        Err("build_pyramids_gpu not implemented".to_string())
    }

    async fn read_texture_to_imagebuffer(
        &self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, String> {
        let texture_format = texture.format();
        let bytes_per_pixel = match texture_format {
             wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
             _ => return Err(format!("Unsupported texture format for readback: {:?}", texture_format)),
        };

        let buffer_size = (width * height * bytes_per_pixel) as wgpu::BufferAddress;
        let buffer_desc = wgpu::BufferDescriptor {
            label: Some("Texture Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        };
        let readback_buffer = self.device.create_buffer(&buffer_desc);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Texture Readback Encoder"),
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { // Updated from ImageCopyBuffer
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout { // Updated from ImageDataLayout
                    offset: 0,
                    bytes_per_row: Some(width * bytes_per_pixel),
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
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { // Добавляем колбэк
             let _ = sender.send(result);
        });

        // Запускаем обработку GPU и ждем
        if let Err(e) = self.device.poll(wgpu::MaintainBase::Wait) {
            return Err(format!("Device poll failed: {:?}", e));
        }

        // Ожидаем результат из канала
        match receiver.recv() { // Убираем .await
             Ok(Ok(())) => { // Успешный маппинг
                 let data = buffer_slice.get_mapped_range();
                 let result_buffer = data.to_vec();
                 drop(data);
                 readback_buffer.unmap();

                 match ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, result_buffer) {
                    Some(image) => Ok(image),
                    None => Err("Failed to create ImageBuffer from raw data.".to_string()),
                 }
             }
             Ok(Err(e)) => Err(format!("Failed to map buffer: {:?}", e)), // Ошибка маппинга
             Err(e) => Err(format!("Channel receive error: {:?}", e)), // Ошибка канала
         }
    }
    // ... (convert_rgba8_to_f32_gray, convert_rgba8_to_luma8) ...
    fn convert_rgba8_to_f32_gray(img_rgba: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> ImageBuffer<Luma<f32>, Vec<f32>> { /* ... без изменений ... */
         let (width, height) = img_rgba.dimensions(); let mut img_f32 = ImageBuffer::new(width, height);
        for y in 0..height { for x in 0..width { let pixel_rgba = img_rgba.get_pixel(x, y); let gray_u8 = pixel_rgba[0]; img_f32.put_pixel(x, y, Luma([gray_u8 as f32 / 255.0])); } } img_f32
    }
     fn convert_rgba8_to_luma8(img_rgba: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> GrayImage { /* ... без изменений ... */
         let (width, height) = img_rgba.dimensions(); let mut img_luma = GrayImage::new(width, height);
        for y in 0..height { for x in 0..width { let pixel_rgba = img_rgba.get_pixel(x, y); img_luma.put_pixel(x, y, Luma([pixel_rgba[0]])); } } img_luma
    }
}


// --- Обновленная функция верхнего уровня ---
pub fn sift_detect_and_compute_gpu(
    img: &DynamicImage,
) -> Result<(Vec<KeyPoint>, Vec<Vec<f32>>), String>
{
    let _ = env_logger::try_init();

    pollster::block_on(async {
        let gpu_context = GpuSiftContext::new().await?;
        let sift_params = Sift::default();

        let gray_img = img.to_luma8();
        let initial_blur_amount = if sift_params.sigma > sift_params.assumed_blur {
            (sift_params.sigma.powi(2) - sift_params.assumed_blur.powi(2)).sqrt()
        } else { 0.0 };
        let base_image_cpu = if initial_blur_amount > 1e-4 {
            imageproc::filter::gaussian_blur_f32(&gray_img, initial_blur_amount)
        } else { gray_img.clone() };
        let base_image_dyn = DynamicImage::ImageLuma8(base_image_cpu);

        // Заглушка для gpu_build_pyramids
        let gpu_gauss_pyramid: Vec<Vec<wgpu::Texture>> = Vec::new();
        let gpu_dog_pyramid: Vec<Vec<wgpu::Texture>> = Vec::new();
        // let (gpu_gauss_pyramid, gpu_dog_pyramid) = gpu_context.build_pyramids_gpu(...).await?;


        let mut cpu_gauss_pyramid: Vec<Vec<GrayImage>> = Vec::new();
        let mut cpu_dog_pyramid: Vec<Vec<ImageBuffer<Luma<f32>, Vec<f32>>>> = Vec::new();

        // --- Чтение пирамид с GPU (пустой цикл) ---
         for octave_textures in gpu_gauss_pyramid.iter() {
            let mut cpu_octave: Vec<GrayImage> = Vec::new();
             for texture in octave_textures {
                 let (w,h) = (texture.width(), texture.height());
                 let rgba_image = gpu_context.read_texture_to_imagebuffer(texture, w, h).await?;
                 cpu_octave.push(GpuSiftContext::convert_rgba8_to_luma8(&rgba_image));
             }
             cpu_gauss_pyramid.push(cpu_octave);
         }
         for octave_textures in gpu_dog_pyramid.iter() {
            let mut cpu_octave: Vec<ImageBuffer<Luma<f32>, Vec<f32>>> = Vec::new();
             for texture in octave_textures {
                let (w,h) = (texture.width(), texture.height());
                 let rgba_image = gpu_context.read_texture_to_imagebuffer(texture, w, h).await?;
                 cpu_octave.push(GpuSiftContext::convert_rgba8_to_f32_gray(&rgba_image));
             }
             cpu_dog_pyramid.push(cpu_octave);
         }
        // --- Конец чтения ---


        // --- Используем CPU пирамиды для теста ---
        println!("Warning: Using CPU pyramid generation as GPU path is not implemented.");
        // Вызываем публичные методы (сделаем их pub(crate))
        let cpu_gauss_pyramid_real = sift_params.generate_gaussian_pyramid(&base_image_dyn.to_luma8());
        let cpu_dog_pyramid_real = sift_params.generate_dog_pyramid(&cpu_gauss_pyramid_real);

        // Вызываем публичные методы (сделаем их pub(crate))
        let initial_keypoints = sift_params.find_scale_space_extrema(&cpu_dog_pyramid_real);
        let refined_keypoints = sift_params.refine_and_filter_extrema(&initial_keypoints, &cpu_dog_pyramid_real);
        let oriented_keypoints = sift_params.assign_orientations(&refined_keypoints, &cpu_gauss_pyramid_real);
        let descriptors = sift_params.compute(&cpu_gauss_pyramid_real, &oriented_keypoints);

        Ok((oriented_keypoints, descriptors))
    })
}