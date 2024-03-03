use ::image::imageops::FilterType;
use sift::*;
// use ::image::DynamicImage;

fn main() {
    let image_path = "./data/box.png";
    let image = load_image(image_path); // Step 1: Load and convert the image to grayscale
    let resized_image = image.resize(320, 240, FilterType::Gaussian).to_luma8(); // Resize the image to a power of 2
    let scales_per_octave = 3; // Number of intervals in each octave
    // let num_octaves = 4; // Number of octaves
    let initial_sigma = 1.6; // Initial value of sigma for Gaussian blur
    
    
    // // Step 2: Build scale-space pyramid of Gaussian images
    let base_img = generate_base_image(&resized_image, initial_sigma, 1.0);
    let num_octaves = compute_number_of_octaves((base_img.width(), base_img.height()));
    let gaussian_kernels = generate_gaussian_kernels(initial_sigma, scales_per_octave);
    let gaussian_images = generate_gaussian_images(&base_img, num_octaves, &gaussian_kernels);
    
    // Step 3: Compute Difference of Gaussians
    let dog_images = compute_dog(&gaussian_images); 
    
    // Step 4: Find local extrema
    // let keypoints = find_local_extrema(&dog_images);

    // // Step 5: Assign an orientation to each keypoint (optional) 
    // let (magnitudes, orientations) = compute_gradients(&gray_image);
    // let keypoints_with_orientations = assign_orientations(&keypoints, &magnitudes, &orientations, gray_image.width());

    // // Step 6: Generate descriptors for each keypoint
    // let descriptors = create_descriptors(&keypoints_with_orientations, &magnitudes, &orientations, gray_image.width());

    // // В этот момент `descriptors` содержит дескрипторы ключевых точек, которые могут быть использованы для сравнения с другими изображениями
    // println!("Generated {} descriptors for keypoints.", descriptors.len());
}



 