// src/main.rs

// Используем имя библиотеки 'sift', а не имя пакета 'sift_rs'
use image::Rgb;
use sift::draw_keypoints_to_image; // Импортируем новую функцию
use sift::sift::{load_image_dyn, Sift}; // Импортируем Rgb для указания цвета
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = "data/1.jpg";

    if std::fs::create_dir_all("data").is_err() {
        eprintln!(
            "Warning: Could not create data directory. Ensure it exists or you have permissions."
        );
    }

    println!("Loading image from: {}", image_path);
    let img_dyn = match load_image_dyn(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error loading image '{}': {}", image_path, e);
            // ... (сообщения об ошибке) ...
            return Err(e.into());
        }
    };
    println!(
        "Image loaded successfully: {}x{}",
        img_dyn.width(),
        img_dyn.height()
    );

    // Используем параметры SIFT по умолчанию
    let sift = Sift::default();

    println!("Detecting SIFT keypoints and computing descriptors...");
    // Используем новый метод, возвращающий и точки, и дескрипторы
    let start_time = Instant::now();
    let (keypoints, descriptors) = sift.detect_and_compute(&img_dyn);
    let duration = start_time.elapsed();

    println!("SIFT processing took: {:?}", duration); // <--- Вывод времени
    println!("Found {} keypoints.", keypoints.len());
    println!("Computed {} descriptors.", descriptors.len());

    if !keypoints.is_empty() {
        // Выводим информацию о первой точке для примера
        println!("Example keypoint [0]: x={:.2}, y={:.2}, size={:.2}, angle={:.2}, response={:.4}, octave={}, layer={}",
                 keypoints[0].x, keypoints[0].y, keypoints[0].size, keypoints[0].angle.to_degrees(), // Угол в градусах для наглядности
                 keypoints[0].response, keypoints[0].octave, keypoints[0].layer);

        // Выводим размер первого дескриптора
        if !descriptors.is_empty() {
            println!("Example descriptor [0] length: {}", descriptors[0].len());
            // Можно вывести первые несколько значений дескриптора
            // println!("Example descriptor [0] values (first 10): {:?}", &descriptors[0][..10.min(descriptors[0].len())]);
        }

        // --- Визуализация ---
        println!("Drawing keypoints on image...");
        let color = Rgb([255u8, 0, 0]); // Красный цвет
        let image_with_keypoints = draw_keypoints_to_image(&img_dyn, &keypoints, color);

        let output_path = "data/output_with_keypoints.png";
        println!("Saving image with keypoints to: {}", output_path);
        if let Err(e) = image_with_keypoints.save(output_path) {
            eprintln!("Error saving output image: {}", e);
        } else {
            println!("PNG output image saved successfully.");
        }
        // --- Конец Визуализации ---
    } else {
        println!("No keypoints found.");
    }

    println!("SIFT process finished.");
    Ok(())
}
