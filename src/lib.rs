use ::image::imageops::FilterType;
use ::image::{open, DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
// use std::f32::consts::PI;
pub mod image;
use crate::image::{gaussian_blur, GrayFloatImage};

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeyPoint {
    /// The horizontal coordinate in a coordinate system is
    /// defined s.t. +x faces right and starts from the top
    /// of the image.
    /// the vertical coordinate in a coordinate system is defined
    /// s.t. +y faces toward the bottom of an image and starts
    /// from the left side of the image.
    pub point: (f32, f32),
    /// The magnitude of response from the detector.
    pub response: f32,

    /// The radius defining the extent of the keypoint, in pixel units
    pub size: f32,

    /// The level of scale space in which the keypoint was detected.
    pub octave: usize,

    /// A classification ID
    pub class_id: usize,

    /// The orientation angle
    pub angle: f32,
}

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

pub fn save_image_as_f32(image: &GrayFloatImage, path: &str) {
    // Преобразование f32 обратно в u8 для сохранения
    let (width, height) = image.dimensions();
    let mut img_u8 = ImageBuffer::new(width, height);

    for (x, y, pixel) in image.enumerate_pixels() {
        let Luma([value]) = pixel;
        // Нормализация и преобразование значения пикселя
        let value_u8 = (*value * 255.0) as u8;
        img_u8.put_pixel(x, y, Luma([value_u8]));
    }

    // Сохранение преобразованного изображения в формате TIFF
    img_u8.save(path).expect("Failed to save image");
}

/// Substracts two images and returns the result.
pub fn subtract_images(img1: &GrayFloatImage, img2: &GrayFloatImage) -> GrayFloatImage {
    let (width, height) = img1.dimensions();
    let mut result_image = ImageBuffer::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let p1 = img1.get_pixel(x, y)[0] as f32;
            let p2 = img2.get_pixel(x, y)[0] as f32;
            // Gaurantee that the result stays within the valid range [0, 255]
            let diff = (p1 - p2).abs() as f32;
            result_image.put_pixel(x, y, Luma([diff]));
        }
    }
    GrayFloatImage(result_image)
}

/// Generate base image from input image by upsampling by 2 in both directions and blurring
pub fn generate_base_image(image: &GrayImage, sigma: f32, assumed_blur: f32) -> GrayFloatImage {
    let (width, height) = image.dimensions();

    let dyn_resized_image = DynamicImage::ImageLuma8(image.clone()).resize_exact(
        width * 2,
        height * 2,
        FilterType::Gaussian,
    );
    let sigma_diff = ((sigma * sigma) - (2.0 * assumed_blur * assumed_blur))
        .sqrt()
        .max(0.01);
    let resized_image: GrayFloatImage = GrayFloatImage::from_dynamic(&dyn_resized_image);
    gaussian_blur(&resized_image, sigma_diff)
}

/// Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
pub fn compute_number_of_octaves(image_shape: (usize, usize)) -> u32 {
    ((image_shape.0.min(image_shape.1) as f32).log2() - 1.0).round() as u32
}

/// Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
pub fn generate_gaussian_kernels(sigma: f32, num_intervals: u32) -> Vec<f32> {
    let num_images_per_octave = num_intervals + 3;
    let k = 2f32.powf(1.0 / num_intervals as f32);
    let mut gaussian_kernels = Vec::with_capacity(num_images_per_octave as usize);
    gaussian_kernels.push(sigma);

    for image_index in 1..num_images_per_octave {
        let sigma_previous = k.powi((image_index - 1) as i32) * sigma;
        let sigma_total = k * sigma_previous;
        let kernel = (sigma_total.powi(2) - sigma_previous.powi(2)).sqrt();
        gaussian_kernels.push(kernel);
    }
    gaussian_kernels
}

/// Generate scale-space pyramid of Gaussian images
pub fn generate_gaussian_images(
    initial_image: &GrayFloatImage,
    num_octaves: u32,
    gaussian_kernels: &[f32],
) -> Vec<Vec<GrayFloatImage>> {
    let mut gaussian_images = Vec::new();
    // let mut current_image = initial_image.clone();

    let mut current_image: GrayFloatImage = initial_image.clone();
    for _ in 0..num_octaves {
        let mut gaussian_images_in_octave = Vec::new();
        gaussian_images_in_octave.push(current_image.clone()); // first image in octave already has the correct blur
        for &gaussian_kernel in gaussian_kernels.iter().skip(1) {
            let blurred_image = gaussian_blur(&current_image, gaussian_kernel);
            gaussian_images_in_octave.push(blurred_image);
        }
        gaussian_images.push(gaussian_images_in_octave.clone());

        // Подготовка изображения для следующей октавы
        if let Some(octave_base) =
            gaussian_images_in_octave.get(gaussian_images_in_octave.len() - 3)
        {
            current_image = octave_base.half_size();
        }
    }

    gaussian_images
}

/// Calculate the difference of Gaussian (DoG) for a list of images in one octave.
pub fn compute_dog(gaussian_images: &[Vec<GrayFloatImage>]) -> Vec<Vec<GrayFloatImage>> {
    let mut dog_images = Vec::new();

    for gaussian_images_in_octave in gaussian_images.iter() {
        let mut dog_images_in_octave = Vec::new();
        for (first_image, second_image) in gaussian_images_in_octave
            .iter()
            .zip(gaussian_images_in_octave.iter().skip(1))
        {
            let dog_image = subtract_images(second_image, first_image);
            dog_images_in_octave.push(dog_image);
        }
        dog_images.push(dog_images_in_octave.clone());
    }

    dog_images
}

fn find_scale_space_extrema(
    gaussian_images: &Vec<Vec<GrayFloatImage>>,
    dog_images: &Vec<Vec<GrayFloatImage>>,
    num_intervals: u32,
    sigma: f32,
    image_border_width: u32,
    contrast_threshold: f32,
) -> Vec<KeyPoint> {
    // Keypoint - структура, представляющая ключевую точку
    let threshold = (0.5 * contrast_threshold / num_intervals as f32 * 255.0).floor() as u8;
    let mut keypoints = Vec::new();

    for (octave_index, dog_images_in_octave) in dog_images.iter().enumerate() {
        for (image_index, window) in dog_images_in_octave.windows(3).enumerate() {
            let (first_image, second_image, third_image) = (&window[0], &window[1], &window[2]);
            for i in image_border_width..first_image.height() as u32 - image_border_width {
                for j in image_border_width..first_image.width() as u32 - image_border_width {
                    if is_pixel_an_extremum(first_image, second_image, third_image, i, j, threshold)
                    {
                        if let Some((keypoint, localized_image_index)) =
                            localize_extremum_via_quadratic_fit(
                                i,
                                j,
                                image_index + 1,
                                octave_index,
                                num_intervals,
                                &dog_images_in_octave,
                                sigma,
                                contrast_threshold,
                                image_border_width,
                            )
                        {
                            // Предполагаем, что compute_keypoints_with_orientations возвращает Vec<Keypoint>
                            let keypoints_with_orientations = compute_keypoints_with_orientations(
                                &keypoint,
                                octave_index,
                                &gaussian_images[octave_index][localized_image_index],
                            );
                            for keypoint_with_orientation in keypoints_with_orientations {
                                keypoints.push(keypoint_with_orientation);
                            }
                        }
                    }
                }
            }
        }
    }
    keypoints
}

fn is_pixel_an_extremum(
    first_image: &GrayFloatImage,
    second_image: &GrayFloatImage,
    third_image: &GrayFloatImage,
    x: u32,
    y: u32,
    threshold: u8,
) -> bool {
    let center_pixel_value = second_image.get_pixel(1, 1)[0];

    if center_pixel_value.abs() as u8 > threshold {
        let mut is_extremum = true;

        for i in x - 1..=x + 1 {
            for j in y - 1..=y + 1 {
                if i == x && j == y {
                    continue;
                } // Skip the center pixel itself

                // Check if the center pixel is not an extremum
                if center_pixel_value > 0.0 {
                    if center_pixel_value < first_image.get_pixel(i, j)[0]
                        || center_pixel_value < second_image.get_pixel(i, j)[0]
                        || center_pixel_value < third_image.get_pixel(i, j)[0]
                    {
                        is_extremum = false;
                        break;
                    }
                } else {
                    if center_pixel_value > first_image.get_pixel(i, j)[0]
                        || center_pixel_value > second_image.get_pixel(i, j)[0]
                        || center_pixel_value > third_image.get_pixel(i, j)[0]
                    {
                        is_extremum = false;
                        break;
                    }
                }
            }
            if !is_extremum {
                break;
            }
        }

        return is_extremum;
    }

    false
}

pub fn localize_extremum_via_quadratic_fit(
    i: u32,
    j: u32,
    image_index: usize,
    octave_index: usize,
    num_intervals: u32,
    dog_images_in_octave: &Vec<GrayFloatImage>,
    sigma: f32,
    contrast_threshold: f32,
    image_border_width: u32,
) -> Option<(KeyPoint, usize)> {
    panic!("Not implemented");
}

pub fn compute_keypoints_with_orientations(
    keypoint: &KeyPoint,
    octave_index: usize,
    image: &GrayFloatImage,
) -> Vec<KeyPoint> {
    panic!("Not implemented");
}
