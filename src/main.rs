
extern crate sift; // not needed since Rust edition 2018
use sift::*;


fn main() {
    let image_path = "./data/1.jpg";
    let gray_image = load_and_convert_image(image_path); // Шаг 1: Загрузка и преобразование в градации серого
    let resized_image = resize_image(&gray_image, 320, 240); // Изменение размера изображения
    let scales_per_octave = 3; // Количество масштабов (изображений) на октаву
    let num_octaves = 4; // Количество октав
    let initial_sigma = 1.6; // Начальное значение sigma для Гауссова размытия

    // Шаг 2: Построение масштабно-инвариантного пространства
    let scale_images = create_scale_images(&resized_image, num_octaves); // Создание изображений разных масштабов
    let blurred_images = apply_gaussian_blur_to_octave(&scale_images, scales_per_octave, initial_sigma); // Применение Гауссова размытия
    
    // for i in 0..blurred_images.len() {
    //     let image = &blurred_images[i];
    //     let path = format!("./data/octave_{}.jpg", i);
    //     save_image(image, &path);
    // }
    
    // // Шаг 3: Вычисление DoG
    // let dog_images = compute_dog(&blurred_images);
    
    // // Шаг 4: Поиск ключевых точек
    // let keypoints = find_local_extrema(&dog_images);

    // // Шаг 5: Вычисление градиентов и ориентаций
    // let (magnitudes, orientations) = compute_gradients(&gray_image);
    // let keypoints_with_orientations = assign_orientations(&keypoints, &magnitudes, &orientations, gray_image.width());

    // // Шаг 6: Создание дескрипторов
    // let descriptors = create_descriptors(&keypoints_with_orientations, &magnitudes, &orientations, gray_image.width());

    // // В этот момент `descriptors` содержит дескрипторы ключевых точек, которые могут быть использованы для сравнения с другими изображениями
    // println!("Generated {} descriptors for keypoints.", descriptors.len());
}