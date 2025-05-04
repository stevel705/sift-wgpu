pub mod gpu_sift;
pub mod keypoints;
pub mod sift;

// Реэкспорт основных типов
pub use gpu_sift::sift_detect_and_compute_gpu;
pub use keypoints::KeyPoint;
pub use sift::{convert_f32_to_grayimage_normalized, load_image_dyn, save_gray_image, Sift};

// Оригинальные функции из твоего примера, если они все еще нужны снаружи
use image::{open, DynamicImage, GrayImage, Rgb, RgbImage}; // Добавляем RgbImage, Rgb
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut}; // Добавляем функции рисования

/// Загружает изображение и преобразует его в оттенки серого.
pub fn load_and_convert_image(path: &str) -> GrayImage {
    let img = open(path).expect("Failed to open image");
    img.into_luma8()
}

/// Рисует ключевые точки на изображении.
///
/// # Arguments
/// * `img` - Исходное изображение (`DynamicImage`).
/// * `keypoints` - Срез ключевых точек для отрисовки.
/// * `color` - Цвет для отрисовки точек (например, `Rgb([255u8, 0, 0])` для красного).
///
/// # Returns
/// * `RgbImage` - Новое изображение с нарисованными точками.
pub fn draw_keypoints_to_image(
    img: &DynamicImage,
    keypoints: &[KeyPoint],
    color: Rgb<u8>,
) -> RgbImage {
    // Конвертируем в Rgb8 для возможности рисовать цветом
    let mut rgb_image = img.to_rgb8();

    // Создаем цветной RGB-изображение для рисования
    for kp in keypoints {
        let x = kp.x;
        let y = kp.y;
        let size: f32 = 2.0; //kp.size; // sigma точки
        let angle = kp.angle; // ориентация в радианах

        // Радиус круга пропорционален масштабу точки (sigma)
        // Множитель 3.0 - эмпирический, чтобы круг был виден
        let radius = (size * 3.0).round() as i32;
        // Минимальный радиус, чтобы очень маленькие точки были видны
        let display_radius = radius.max(2);

        // Рисуем круг
        draw_filled_circle_mut(&mut rgb_image, (x as i32, y as i32), display_radius, color);

        // Конечная точка для линии ориентации
        let x_end = x + (display_radius as f32 * angle.cos());
        let y_end = y + (display_radius as f32 * angle.sin());

        // Рисуем линию ориентации
        draw_line_segment_mut(
            &mut rgb_image,
            (x, y),         // Начало в центре
            (x_end, y_end), // Конец по направлению угла
            color,
        );

        // Опционально: можно нарисовать еще один круг поменьше в центре
        draw_filled_circle_mut(
            &mut rgb_image,
            (x as i32, y as i32),
            1, // Маленький радиус для центральной точки
            color,
        );
    }

    rgb_image
}
