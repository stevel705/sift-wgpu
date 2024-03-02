
use image::{open, DynamicImage, GrayImage, ImageBuffer, Luma};
use std::f32::consts::PI;
use image::imageops::FilterType;
use imageproc::filter::gaussian_blur_f32;

/// Load an image and return it in its original format.
pub fn load_image(path: &str) -> DynamicImage {
    open(path).expect("Failed to open image")
}

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

/// Apply Gaussian blur to each image in an octave.
/// 
/// # Args
/// 
/// * `octave_images` - vector of images in an octave.
/// * `scales_per_octave` - number of scales in each octave.
/// * `initial_sigma` - initial value of sigma for Gaussian blur.
/// 
/// # Returns
/// 
/// Return a vector of blurred images.
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


/// Changes the size of an image.
pub fn resize_image(image: &GrayImage, new_width: u32, new_height: u32) -> GrayImage {
    // Преобразование GrayImage в DynamicImage для доступа к методу resize_exact
    let dyn_image = DynamicImage::ImageLuma8(image.clone());
    // Изменение размера с использованием желаемого метода фильтрации
    let resized = dyn_image.resize_exact(new_width, new_height, FilterType::Gaussian);
    // Конвертация обратно в GrayImage
    resized.to_luma8()
}

/// Substracts two images and returns the result.
pub fn subtract(img1: &GrayImage, img2: &GrayImage) -> GrayImage {
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

/// Creates a list of scaled images from the original image.
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


/// Calculate the difference of Gaussian (DoG) for a list of images in one octave.
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

/// Substracts two images and returns the result.
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
