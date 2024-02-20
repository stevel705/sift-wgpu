
use image::{open, DynamicImage, GrayImage, ImageBuffer, Luma};
use std::f32::consts::PI;
use image::imageops::FilterType;
use imageproc::filter::gaussian_blur_f32;


pub fn load_and_convert_image(path: &str) -> GrayImage {
    let img = open(path).expect("Failed to open image");
    img.into_luma8()
}

pub fn save_image(image: &GrayImage, path: &str) {
    image.save(path).expect("Failed to save image");
}


// Compute the keypoints and descriptors for a given image.
// Input:
// - image: The input image.
// - sigma: The initial value of sigma for Gaussian blur.
// - num_intervals: The number of intervals in each octave.
// - assumed_blur: The assumed blur for the image.
// - image_border_width: The width of the border to be ignored.
// Output:
// - keypoints: The computed keypoints.
// - descriptors: The computed descriptors.

// pub fn compute_keypoints_and_descriptors(image: &GrayImage, sigma: f32, num_intervals: u32, assumed_blur: f32, image_border_width: u32) -> (Vec<(u32, u32, usize, f32)>, Vec<Vec<f32>>) {
//     /// Compute SIFT keypoints and descriptors for an input image
//     let resized_image = resize_image(image, 320, 240);
//     let scales_per_octave = num_intervals as usize - 3;
//     let num_octaves = 4;
//     let scale_images = create_scale_images(&resized_image, num_octaves * scales_per_octave);
//     let blurred_images = apply_gaussian_blur_to_octave(&scale_images, scales_per_octave, sigma);
//     let dog_images = compute_dog(&blurred_images);
//     let keypoints = find_local_extrema(&dog_images);
//     let (magnitudes, orientations) = compute_gradients(&resized_image);
//     let keypoints_with_orientations = assign_orientations(&keypoints, &magnitudes, &orientations, resized_image.width());
//     let descriptors = create_descriptors(&keypoints_with_orientations, &magnitudes, &orientations, resized_image.width());
//     (keypoints_with_orientations, descriptors)
// }

/// Применяет Гауссово размытие к каждому изображению в октаве.
/// 
/// # Аргументы
/// 
/// * `octave_images` - вектор изображений одной октавы.
/// * `scales_per_octave` - количество масштабов на октаву.
/// * `initial_sigma` - начальное значение sigma для Гауссова размытия.
/// 
/// # Возвращает
/// 
/// Возвращает вектор размытых изображений для данной октавы.
pub fn apply_gaussian_blur_to_octave(octave_images: &[GrayImage], scales_per_octave: usize, initial_sigma: f32) -> Vec<GrayImage> {

    let mut blurred_images = Vec::with_capacity(scales_per_octave);

    // Коэффициент для увеличения sigma на каждом шаге, чтобы обеспечить перекрытие масштабов
    let k = (2f32).powf(1.0 / scales_per_octave as f32);

    for i in 0..scales_per_octave {
        let sigma = initial_sigma * k.powi(i as i32);
        let blurred_image = gaussian_blur_f32(&octave_images[i], sigma);
        blurred_images.push(blurred_image);
    }

    blurred_images
}


/// Изменяет размер изображения и возвращает его как GrayImage
pub fn resize_image(image: &GrayImage, new_width: u32, new_height: u32) -> GrayImage {
    // Преобразование GrayImage в DynamicImage для доступа к методу resize_exact
    let dyn_image = DynamicImage::ImageLuma8(image.clone());
    // Изменение размера с использованием желаемого метода фильтрации
    let resized = dyn_image.resize_exact(new_width, new_height, FilterType::Gaussian);
    // Конвертация обратно в GrayImage
    resized.to_luma8()
}

/// Создает изображения в разных масштабах.
pub fn create_scale_images(image: &GrayImage, scales: usize) -> Vec<GrayImage> {
    let mut images = Vec::new();
    for scale in 0..scales {
        // Для каждого масштаба изменяем размер, уменьшая размер вдвое относительно оригинала
        let scaled_width = image.width() >> scale;
        let scaled_height = image.height() >> scale;
        let scaled_image = resize_image(image, scaled_width, scaled_height);
        images.push(scaled_image);
    }
    images
}


/// Вычисляет разность Гауссовых размытий (DoG) для списка изображений в одной октаве.
pub fn compute_dog(images: &[GrayImage]) -> Vec<GrayImage> {
    let mut dogs = Vec::new();
    for i in 0..images.len() - 1 {
        let img1 = &images[i];
        let img2 = &images[i + 1];
        let dog = subtract_images(img1, img2);
        dogs.push(dog);
    }
    dogs
}

/// Вычитает два изображения и возвращает результат.
pub fn subtract_images(img1: &GrayImage, img2: &GrayImage) -> GrayImage {
    let (width, height) = img1.dimensions();
    let mut result_image = ImageBuffer::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let p1 = img1.get_pixel(x, y)[0] as i16;
            let p2 = img2.get_pixel(x, y)[0] as i16;
            // Гарантируем, что результат остается в пределах допустимых значений [0, 255]
            let diff = (p1 - p2).abs() as u8;
            result_image.put_pixel(x, y, Luma([diff]));
        }
    }

    result_image
}

/// Находит локальные экстремумы в списке DoG изображений.
pub fn find_local_extrema(dogs: &[GrayImage]) -> Vec<(u32, u32, usize)> {
    let mut keypoints = Vec::new();
    for i in 1..dogs.len() - 1 {
        let prev = &dogs[i - 1];
        let current = &dogs[i];
        let next = &dogs[i + 1];

        for x in 1..current.width() - 1 {
            for y in 1..current.height() - 1 {
                let p = current.get_pixel(x, y)[0];
                if is_local_extremum(x, y, p, prev, current, next) {
                    keypoints.push((x, y, i)); // Сохраняем координаты и индекс изображения
                }
            }
        }
    }
    keypoints
}

/// Проверяет, является ли точка локальным экстремумом.
pub fn is_local_extremum(x: u32, y: u32, p: u8, prev: &GrayImage, current: &GrayImage, next: &GrayImage) -> bool {
    let mut is_max = true;
    let mut is_min = true;

    for nx in x - 1..=x + 1 {
        for ny in y - 1..=y + 1 {
            if nx == x && ny == y { continue; }
            
            // Проверяем соседей в текущем изображении
            is_max &= p > current.get_pixel(nx, ny)[0];
            is_min &= p < current.get_pixel(nx, ny)[0];

            // Проверяем соседей в предыдущем и следующем масштабе
            is_max &= p > prev.get_pixel(nx, ny)[0] && p > next.get_pixel(nx, ny)[0];
            is_min &= p < prev.get_pixel(nx, ny)[0] && p < next.get_pixel(nx, ny)[0];

            if !is_max && !is_min { return false; }
        }
    }

    is_max || is_min
}


/// Вычисляет магнитуду и направление градиента для каждого пикселя изображения.
pub fn compute_gradients(image: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let mut magnitudes = vec![0f32; (image.width() * image.height()) as usize];
    let mut orientations = vec![0f32; (image.width() * image.height()) as usize];

    for x in 1..image.width() - 1 {
        for y in 1..image.height() - 1 {
            let gx = (image.get_pixel(x + 1, y)[0] as f32 - image.get_pixel(x - 1, y)[0] as f32).abs();
            let gy = (image.get_pixel(x, y + 1)[0] as f32 - image.get_pixel(x, y - 1)[0] as f32).abs();

            let magnitude = (gx.powi(2) + gy.powi(2)).sqrt();
            let orientation = (gy.atan2(gx) * 180.0 / PI) % 360.0; // Преобразование в градусы

            magnitudes[(y * image.width() + x) as usize] = magnitude;
            orientations[(y * image.width() + x) as usize] = orientation;
        }
    }

    (magnitudes, orientations)
}


/// Назначает основную ориентацию для каждой ключевой точки на основе гистограммы градиентов.
pub fn assign_orientations(keypoints: &[(u32, u32, usize)], magnitudes: &[f32], orientations: &[f32], width: u32) -> Vec<(u32, u32, usize, f32)> {
    let mut keypoints_with_orientations = Vec::new();

    // Размерность гистограммы (например, 36 бинов для 360 градусов)
    let bin_size = 10.0; // 360 градусов / 36 бинов = 10 градусов на бин
    let mut histogram = vec![0f32; 36];

    for &(x, y, scale) in keypoints {
        histogram.iter_mut().for_each(|v| *v = 0.0); // Очистка гистограммы

        // Рассчитываем гистограмму градиентов вокруг ключевой точки
        for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
            for ny in y.saturating_sub(1)..=(y + 1).min(width - 1) { // Предполагаем квадратные изображения для упрощения
                let index = (ny * width + nx) as usize;
                let magnitude = magnitudes[index];
                let orientation = orientations[index];

                let bin = (orientation / bin_size).floor() as usize % 36;
                histogram[bin] += magnitude;
            }
        }

        // Находим пик в гистограмме, который соответствует основной ориентации
        if let Some((bin, &max_magnitude)) = histogram.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            let dominant_orientation = bin as f32 * bin_size + bin_size / 2.0; // Середина бина
            keypoints_with_orientations.push((x, y, scale, dominant_orientation));
        }
    }

    keypoints_with_orientations
}


/// Создает дескрипторы для ключевых точек.
pub fn create_descriptors(keypoints_with_orientations: &[(u32, u32, usize, f32)], magnitudes: &[f32], orientations: &[f32], width: u32) -> Vec<Vec<f32>> {
    let mut descriptors = Vec::new();

    for &(x, y, _, orientation) in keypoints_with_orientations {
        let mut descriptor = vec![0f32; 128]; // Дескриптор размером 128 элементов
        let bin_size = 45.0; // 360 градусов / 8 направлений = 45 градусов на направление

        // Итерация по 16 подблокам (4x4) вокруг ключевой точки
        for sub_x in 0..4 {
            for sub_y in 0..4 {
                // Создание гистограммы направлений для подблока
                let mut sub_histogram = vec![0f32; 8];

                // Итерация по пикселям подблока
                for local_x in 0..4 {
                    for local_y in 0..4 {
                        let global_x = (x as usize) + sub_x * 4 + local_x - 8; // Смещение относительно ключевой точки
                        let global_y = (y as usize) + sub_y * 4 + local_y - 8;
                        let index = (global_y * width as usize + global_x) as usize;

                        let magnitude = magnitudes[index];
                        let mut direction = orientations[index] - orientation; // Учитываем ориентацию ключевой точки
                        if direction < 0.0 {
                            direction += 360.0;
                        }

                        let bin = (direction / bin_size).floor() as usize % 8;
                        sub_histogram[bin] += magnitude;
                    }
                }

                // Добавление гистограммы подблока к дескриптору
                for (i, &val) in sub_histogram.iter().enumerate() {
                    // Конвертируем индексы и размеры из u32 в usize для корректной индексации
                    let index = (sub_x * 32 + sub_y * 8 + i) as usize;
                    descriptor[index] = val;
                }
            }
        }

        // Нормализация дескриптора
        let norm = descriptor.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
        for val in &mut descriptor {
            *val /= norm;
        }

        descriptors.push(descriptor);
    }

    descriptors
}
