use crate::utils::*;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
use std::f32::consts::PI;

use crate::{keypoints::KeyPoint, SiftBackend};
use log::warn;

// Sift parameters based on Lowe's paper and common implementations
const DEFAULT_SIGMA: f32 = 1.6;
const DEFAULT_NUM_OCTAVES: u32 = 4; // Can be computed based on image size
pub const DEFAULT_NUM_INTERVALS: u32 = 3; // S in Lowe's paper (number of DoG intervals for extrema detection)
const DEFAULT_ASSUMED_BLUR: f32 = 0.5; // Assumed blur of the input image
const DEFAULT_CONTRAST_THRESHOLD: f32 = 0.04;
const DEFAULT_EDGE_THRESHOLD: f32 = 10.0;
const DEFAULT_IMAGE_BORDER_WIDTH: u32 = 5; // Border width ignored when finding points
const MAX_INTERPOLATION_STEPS: usize = 5; // Max number of interpolation steps
const INTERPOLATION_OFFSET_THRESHOLD: f32 = 0.5; // Offset threshold for interpolation
                                                 // Other parameters that can be added later:
const ORIENTATION_HIST_BINS: usize = 36; // Number of bins in orientation histogram
const ORIENTATION_WINDOW_RADIUS_FACTOR: f32 = 3.0; // Window radius = factor * 1.5 * sigma_octave
const ORIENTATION_SMOOTHING_ITERATIONS: usize = 2; // Number of histogram smoothing passes
const ORIENTATION_PEAK_RATIO: f32 = 0.8; // Threshold for secondary orientation peaks
const ORIENTATION_GAUSSIAN_EXPANSION_FACTOR: f32 = 1.5; // sigma for Gaussian weighting = factor * sigma_octave

// Descriptor constants
const DESC_HIST_BINS: usize = 8; // Number of orientation bins in descriptor histogram
const DESC_WINDOW_WIDTH: usize = 4; // Descriptor grid width (4x4)
const DESC_MAG_THR: f32 = 0.2; // Threshold for magnitude clipping in descriptor
const DESC_PATCH_SCALE_FACTOR: f32 = 3.0;
// const DESC_INT_FACTOR: f32 = 512.0; // Multiplier for conversion to bytes (not used here, but often in impl)
// Coefficient for scaling descriptor window relative to point sigma
// Window will be DESC_WINDOW_WIDTH * patch_size_factor pixels wide in point sigma scale

pub struct Sift {
    pub sigma: f32,
    pub num_octaves: u32,
    pub num_intervals: u32, // S
    pub assumed_blur: f32,
    pub contrast_threshold: f32,
    pub edge_threshold: f32, // Not currently used
    image_border_width: u32,
    // Other parameters can be added as implementation progresses
}

impl Default for Sift {
    fn default() -> Self {
        Sift {
            sigma: DEFAULT_SIGMA,
            num_octaves: DEFAULT_NUM_OCTAVES,
            num_intervals: DEFAULT_NUM_INTERVALS,
            assumed_blur: DEFAULT_ASSUMED_BLUR,
            // Contrast threshold is often normalized by number of intervals, like in vlfeat
            contrast_threshold: DEFAULT_CONTRAST_THRESHOLD / DEFAULT_NUM_INTERVALS as f32,
            edge_threshold: DEFAULT_EDGE_THRESHOLD,
            image_border_width: DEFAULT_IMAGE_BORDER_WIDTH,
        }
    }
}

impl Sift {
    pub fn new(
        sigma: f32,
        num_octaves: u32,
        num_intervals: u32,
        assumed_blur: f32,
        contrast_threshold: f32,
        edge_threshold: f32,
    ) -> Self {
        Sift {
            sigma,
            num_octaves,
            num_intervals,
            assumed_blur,
            contrast_threshold: contrast_threshold / num_intervals as f32,
            edge_threshold,
            image_border_width: DEFAULT_IMAGE_BORDER_WIDTH,
        }
    }

    // Helper function: resize image
    fn resize_image(image: &GrayImage, new_width: u32, new_height: u32) -> GrayImage {
        let dyn_image = DynamicImage::ImageLuma8(image.clone());
        // Lanczos3 is well suited for downscaling, preserving details
        let resized = dyn_image.resize_exact(new_width, new_height, FilterType::Lanczos3);
        resized.into_luma8()
    }

    // Helper function: convert GrayImage (Luma<u8>) to ImageBuffer<Luma<f32>, Vec<f32>>
    // Pixel values are normalized to range [0.0, 1.0]
    fn convert_u8_to_f32_gray(img: &GrayImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let (width, height) = img.dimensions();
        let mut f32_img = ImageBuffer::new(width, height);
        for x in 0..width {
            for y in 0..height {
                f32_img.put_pixel(x, y, Luma([img.get_pixel(x, y)[0] as f32 / 255.0]));
            }
        }
        f32_img
    }

    // Helper function: subtract two Luma<f32> images (parallel)
    fn subtract_f32_images(
        img1: &ImageBuffer<Luma<f32>, Vec<f32>>,
        img2: &ImageBuffer<Luma<f32>, Vec<f32>>,
    ) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let (width, height) = img1.dimensions();
        assert_eq!(
            img1.dimensions(),
            img2.dimensions(),
            "Images must have the same dimensions for subtraction"
        );
        let pixels1 = img1.as_raw();
        let pixels2 = img2.as_raw();

        let result_pixels: Vec<f32> = pixels1
            .par_iter()
            .zip(pixels2.par_iter())
            .map(|(&p1, &p2)| p1 - p2)
            .collect();

        ImageBuffer::from_raw(width, height, result_pixels).expect("Failed to create result image")
    }

    // Build Gaussian Pyramid
    // base_image: initial image for pyramid (after preprocessing)
    // Returns: vector of octaves, where each octave is a vector of blurred GrayImages
    pub(crate) fn generate_gaussian_pyramid(&self, base_image: &GrayImage) -> Vec<Vec<GrayImage>> {
        let mut pyramid = Vec::with_capacity(self.num_octaves as usize);
        let mut current_octave_base_image = base_image.clone();
        let k = 2.0_f32.powf(1.0 / self.num_intervals as f32);
        let mut octave_target_sigmas = Vec::with_capacity((self.num_intervals + 3) as usize);
        for s_idx in 0..(self.num_intervals + 3) {
            octave_target_sigmas.push(self.sigma * k.powi(s_idx as i32));
        }

        for o_idx in 0..self.num_octaves {
            let mut octave_images = Vec::with_capacity((self.num_intervals + 3) as usize);
            octave_images.push(current_octave_base_image.clone());
            let mut prev_image_sigma_abs = self.sigma * (2.0_f32).powi(o_idx as i32); // Corrected sigma for octave base

            for s_idx in 1..(self.num_intervals + 3) {
                let target_sigma_abs =
                    octave_target_sigmas[s_idx as usize] * (2.0_f32).powi(o_idx as i32); // Absolute sigma for this level
                let blur_to_apply =
                    (target_sigma_abs.powi(2) - prev_image_sigma_abs.powi(2)).sqrt();

                let blurred_image = if blur_to_apply < 1e-4 {
                    octave_images.last().unwrap().clone()
                } else {
                    // Use the f32 version for blurring for better precision
                    // let prev_img_f32 = Sift::convert_u8_to_f32_gray(octave_images.last().unwrap());
                    // let blurred_f32 = gaussian_blur_f32(&prev_img_f32, blur_to_apply);
                    // Convert back to u8 - careful about normalization/clamping if blur introduces values outside [0,1]
                    // gaussian_blur_f32 from imageproc should handle this reasonably.
                    // Let's keep using the u8 version from imageproc for now if it exists and works.
                    // Revert: use GrayImage directly as input to gaussian_blur_f32
                    gaussian_blur_f32(octave_images.last().unwrap(), blur_to_apply)
                };
                octave_images.push(blurred_image);
                prev_image_sigma_abs = target_sigma_abs;
            }
            pyramid.push(octave_images);

            if o_idx < self.num_octaves - 1 {
                let next_octave_base_idx = self.num_intervals as usize;
                let image_for_downsample = &pyramid.last().unwrap()[next_octave_base_idx];
                current_octave_base_image = Sift::resize_image(
                    image_for_downsample,
                    image_for_downsample.width() / 2,
                    image_for_downsample.height() / 2,
                );
            }
        }
        pyramid
    }

    // Build Difference of Gaussians (DoG) pyramid - parallel over octaves
    // gaussian_pyramid: result of generate_gaussian_pyramid
    // Returns: vector of octaves, where each octave is a vector of DoG images (Luma<f32>)
    pub(crate) fn generate_dog_pyramid(
        &self,
        gaussian_pyramid: &[Vec<GrayImage>],
    ) -> Vec<Vec<ImageBuffer<Luma<f32>, Vec<f32>>>> {
        gaussian_pyramid
            .par_iter()
            .map(|octave_u8_images| {
                // Convert to f32 in parallel
                let octave_f32_images: Vec<_> = octave_u8_images
                    .par_iter()
                    .map(Sift::convert_u8_to_f32_gray)
                    .collect();

                // Compute DoG for adjacent pairs
                (0..(octave_f32_images.len() - 1))
                    .into_par_iter()
                    .map(|i| {
                        Sift::subtract_f32_images(&octave_f32_images[i + 1], &octave_f32_images[i])
                    })
                    .collect()
            })
            .collect()
    }

    // Helper function to get pixel value (boundary safe)
    #[inline(always)]
    pub(crate) fn get_pixel_value(img: &ImageBuffer<Luma<f32>, Vec<f32>>, x: i32, y: i32) -> f32 {
        // Simple boundary handling - clamping to edge pixel
        let (width, height) = img.dimensions();
        let x_clamp = x.clamp(0, width as i32 - 1) as u32;
        let y_clamp = y.clamp(0, height as i32 - 1) as u32;
        img.get_pixel(x_clamp, y_clamp)[0]
    }

    // Helper for Gaussian images (GrayImage -> f32)
    #[inline(always)]
    pub(crate) fn get_gauss_pixel_value(img: &GrayImage, x: i32, y: i32) -> f32 {
        let (width, height) = img.dimensions();
        let x_clamp = x.clamp(0, width as i32 - 1) as u32;
        let y_clamp = y.clamp(0, height as i32 - 1) as u32;
        img.get_pixel(x_clamp, y_clamp)[0] as f32 / 255.0 // Normalize
    }

    #[inline(always)]
    pub(crate) fn get_gauss_pixel_value_f32(
        img: &ImageBuffer<Luma<f32>, Vec<f32>>,
        x: i32,
        y: i32,
    ) -> f32 {
        let (width, height) = img.dimensions();
        let x_clamp = x.clamp(0, width as i32 - 1) as u32;
        let y_clamp = y.clamp(0, height as i32 - 1) as u32;
        img.get_pixel(x_clamp, y_clamp)[0]
    }

    #[inline(always)]
    pub(crate) fn get_gauss_pixel_bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
        // Ensure coordinates are within valid range for interpolation
        // Allow slightly outside [0, width/height - 1] to handle border cases, clamp later.
        let x_floor = x.floor();
        let y_floor = y.floor();
        let x_ceil = x_floor + 1.0;
        let y_ceil = y_floor + 1.0;

        let dx = x - x_floor;
        let dy = y - y_floor;

        let x0 = x_floor as i32;
        let y0 = y_floor as i32;
        let x1 = x_ceil as i32;
        let y1 = y_ceil as i32;

        // Use safe getter which clamps coordinates
        let q11 = Self::get_gauss_pixel_value(img, x0, y0);
        let q21 = Self::get_gauss_pixel_value(img, x1, y0);
        let q12 = Self::get_gauss_pixel_value(img, x0, y1);
        let q22 = Self::get_gauss_pixel_value(img, x1, y1);

        // Bilinear interpolation formula
        let val = q11 * (1.0 - dx) * (1.0 - dy)
            + q21 * dx * (1.0 - dy)
            + q12 * (1.0 - dx) * dy
            + q22 * dx * dy;
        val
    }

    #[inline(always)]
    pub(crate) fn get_gauss_pixel_bilinear_f32(
        img: &ImageBuffer<Luma<f32>, Vec<f32>>,
        x: f32,
        y: f32,
    ) -> f32 {
        let x_floor = x.floor();
        let y_floor = y.floor();
        let x_ceil = x_floor + 1.0;
        let y_ceil = y_floor + 1.0;

        let dx = x - x_floor;
        let dy = y - y_floor;

        let x0 = x_floor as i32;
        let y0 = y_floor as i32;
        let x1 = x_ceil as i32;
        let y1 = y_ceil as i32;

        let q11 = Self::get_gauss_pixel_value_f32(img, x0, y0);
        let q21 = Self::get_gauss_pixel_value_f32(img, x1, y0);
        let q12 = Self::get_gauss_pixel_value_f32(img, x0, y1);
        let q22 = Self::get_gauss_pixel_value_f32(img, x1, y1);

        q11 * (1.0 - dx) * (1.0 - dy)
            + q21 * dx * (1.0 - dy)
            + q12 * (1.0 - dx) * dy
            + q22 * dx * dy
    }

    /// Refines extrema positions, filters out low contrast points and edge points.
    pub(crate) fn refine_and_filter_extrema(
        &self,
        initial_keypoints: &[KeyPoint],
        dog_pyramid: &[Vec<ImageBuffer<Luma<f32>, Vec<f32>>>],
    ) -> Vec<KeyPoint> {
        let mut refined_keypoints = Vec::new();

        for kp in initial_keypoints {
            let octave_idx = kp.octave as usize;
            let layer_idx = kp.layer as usize; // Index in DoG pyramid
            let x_int = kp.x / (2.0_f32.powi(kp.octave)); // Coordinates in octave
            let y_int = kp.y / (2.0_f32.powi(kp.octave));
            let mut current_x = x_int as i32; // Use i32 for difference calculations
            let mut current_y = y_int as i32;
            let mut current_layer = layer_idx as i32;

            let dog_octave = &dog_pyramid[octave_idx];

            // Iterative interpolation to refine position
            let mut converged = false;
            let mut interpolated_kp_data = None;

            for _ in 0..MAX_INTERPOLATION_STEPS {
                // Check if we are within layer usage or image bounds
                if current_layer < 1 || current_layer >= (dog_octave.len() - 1) as i32 {
                    break; // Cannot compute scale derivatives
                }
                let img_prev = &dog_octave[current_layer as usize - 1];
                let img_curr = &dog_octave[current_layer as usize];
                let img_next = &dog_octave[current_layer as usize + 1];
                let (width, height) = img_curr.dimensions();
                if current_x < 1
                    || current_x >= (width - 1) as i32
                    || current_y < 1
                    || current_y >= (height - 1) as i32
                {
                    break; // Cannot compute spatial derivatives
                }

                // Compute gradient (g) and Hessian (H) using central differences
                let dx = (Self::get_pixel_value(img_curr, current_x + 1, current_y)
                    - Self::get_pixel_value(img_curr, current_x - 1, current_y))
                    / 2.0;
                let dy = (Self::get_pixel_value(img_curr, current_x, current_y + 1)
                    - Self::get_pixel_value(img_curr, current_x, current_y - 1))
                    / 2.0;
                let ds = (Self::get_pixel_value(img_next, current_x, current_y)
                    - Self::get_pixel_value(img_prev, current_x, current_y))
                    / 2.0;
                let gradient = [dx, dy, ds];

                let center_val = Self::get_pixel_value(img_curr, current_x, current_y);
                let dxx = Self::get_pixel_value(img_curr, current_x + 1, current_y)
                    + Self::get_pixel_value(img_curr, current_x - 1, current_y)
                    - 2.0 * center_val;
                let dyy = Self::get_pixel_value(img_curr, current_x, current_y + 1)
                    + Self::get_pixel_value(img_curr, current_x, current_y - 1)
                    - 2.0 * center_val;
                let dss = Self::get_pixel_value(img_next, current_x, current_y)
                    + Self::get_pixel_value(img_prev, current_x, current_y)
                    - 2.0 * center_val;

                let dxy = (Self::get_pixel_value(img_curr, current_x + 1, current_y + 1)
                    - Self::get_pixel_value(img_curr, current_x - 1, current_y + 1)
                    - Self::get_pixel_value(img_curr, current_x + 1, current_y - 1)
                    + Self::get_pixel_value(img_curr, current_x - 1, current_y - 1))
                    / 4.0;
                let dxs = (Self::get_pixel_value(img_next, current_x + 1, current_y)
                    - Self::get_pixel_value(img_next, current_x - 1, current_y)
                    - (Self::get_pixel_value(img_prev, current_x + 1, current_y)
                        - Self::get_pixel_value(img_prev, current_x - 1, current_y)))
                    / 4.0;
                let dys = (Self::get_pixel_value(img_next, current_x, current_y + 1)
                    - Self::get_pixel_value(img_next, current_x, current_y - 1)
                    - (Self::get_pixel_value(img_prev, current_x, current_y + 1)
                        - Self::get_pixel_value(img_prev, current_x, current_y - 1)))
                    / 4.0;

                let hessian = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];

                // Solve H * x_offset = -g for x_offset = [dx_hat, dy_hat, ds_hat]
                // Use formula for 3x3 matrix inversion or solve system
                if let Some(offset) =
                    Self::solve_linear_system(hessian, [-gradient[0], -gradient[1], -gradient[2]])
                {
                    let dx_hat = offset[0];
                    let dy_hat = offset[1];
                    let ds_hat = offset[2];

                    // If offset in all dimensions is small, consider converged
                    if dx_hat.abs() < INTERPOLATION_OFFSET_THRESHOLD
                        && dy_hat.abs() < INTERPOLATION_OFFSET_THRESHOLD
                        && ds_hat.abs() < INTERPOLATION_OFFSET_THRESHOLD
                    {
                        // Compute DoG value at interpolated point
                        let interpolated_dog_val = center_val
                            + 0.5
                                * (gradient[0] * dx_hat
                                    + gradient[1] * dy_hat
                                    + gradient[2] * ds_hat);

                        // 1. Contrast rejection
                        if interpolated_dog_val.abs() < self.contrast_threshold {
                            break; // Discard point
                        }

                        // 2. Edge rejection (use only 2x2 Hessian for x, y)
                        let hessian_xy = [[dxx, dxy], [dxy, dyy]];
                        let trace_sq = (hessian_xy[0][0] + hessian_xy[1][1]).powi(2);
                        let det = hessian_xy[0][0] * hessian_xy[1][1]
                            - hessian_xy[0][1] * hessian_xy[1][0];

                        if det <= 0.0 {
                            // Determinant <= 0 means different curvature signs (saddle point) or one curvature = 0
                            break; // Discard point
                        }

                        let edge_response_ratio = trace_sq / det;
                        let edge_threshold_sq =
                            (self.edge_threshold + 1.0).powi(2) / self.edge_threshold;

                        if edge_response_ratio >= edge_threshold_sq {
                            break; // Discard point (too edge-like)
                        }

                        // Point passed all checks! Saving data.
                        converged = true;
                        let scale_factor = 2.0_f32.powi(kp.octave);
                        let final_layer_float = current_layer as f32 + ds_hat;
                        // Effective sigma = sigma_0 * 2^(octave + layer_float / num_intervals)
                        // where sigma_0 = self.sigma
                        // Point size is usually related to the Gaussian sigma where it was found
                        let point_sigma_absolute = self.sigma
                            * 2.0_f32.powf(
                                kp.octave as f32 + final_layer_float / self.num_intervals as f32,
                            );

                        interpolated_kp_data = Some(KeyPoint {
                            x: (current_x as f32 + dx_hat) * scale_factor,
                            y: (current_y as f32 + dy_hat) * scale_factor,
                            // size: point_sigma_absolute * 2.0, // Size is often doubled for visualization
                            size: point_sigma_absolute, // Use sigma as size
                            angle: 0.0,                 // To be computed later
                            response: interpolated_dog_val,
                            octave: kp.octave,
                            layer: final_layer_float.round() as i32, // Store nearest integer layer for info
                        });
                        break; // Successful interpolation
                    } else {
                        // Offset too large, need to move to new nearest pixel
                        // and repeat interpolation (if step limit not exceeded)
                        // Update integer coordinates
                        current_x = (current_x as f32 + dx_hat).round() as i32;
                        current_y = (current_y as f32 + dy_hat).round() as i32;
                        current_layer = (current_layer as f32 + ds_hat).round() as i32;

                        // Check if we exited reasonable layer bounds after offset
                        if current_layer < 0 || current_layer >= dog_octave.len() as i32 {
                            break;
                        }
                    }
                } else {
                    // Failed to solve system (Hessian is singular)
                    break; // Discard point
                }
            } // end interpolation loop

            if converged {
                if let Some(final_kp) = interpolated_kp_data {
                    refined_keypoints.push(final_kp);
                }
            }
        } // end keypoints loop

        refined_keypoints
    }

    // Helper function to solve 3x3 system Ax = b (Cramer's rule or Gauss)
    // Returns Option<[f32; 3]> ([x0, x1, x2])
    fn solve_linear_system(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
        // Using simple Cramer's rule for 3x3
        let det_a = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det_a.abs() < 1e-10 {
            // Matrix is singular or close to singular
            return None;
        }

        // Compute determinants for Dx, Dy, Dz
        let det_x = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

        let det_y = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
            - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

        let det_z = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
            - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        Some([det_x / det_a, det_y / det_a, det_z / det_a])
    }

    // Search for extrema in scale-space (parallel over octaves)
    // dog_pyramid: result of generate_dog_pyramid
    // Returns: vector of keypoint candidates
    /// Finds initial keypoint candidates (DoG extrema).
    /// Returns `KeyPoint` with integer coordinates and layer.
    pub(crate) fn find_scale_space_extrema(
        &self,
        dog_pyramid: &[Vec<ImageBuffer<Luma<f32>, Vec<f32>>>],
    ) -> Vec<KeyPoint> {
        let num_intervals = self.num_intervals as usize;
        let border = self.image_border_width as i32;

        // Process octaves in parallel
        dog_pyramid
            .par_iter()
            .enumerate()
            .flat_map(|(o_idx, dog_octave)| {
                if dog_octave.is_empty() {
                    return Vec::new();
                }
                let (width, height) = dog_octave[0].dimensions();
                let width_i32 = width as i32;
                let height_i32 = height as i32;
                let scale_factor = 2.0_f32.powi(o_idx as i32);

                // Process scales in parallel within each octave
                (1..=num_intervals)
                    .into_par_iter()
                    .filter(|&s_idx| s_idx < dog_octave.len() - 1)
                    .flat_map(|s_idx| {
                        let img_prev = &dog_octave[s_idx - 1];
                        let img_curr = &dog_octave[s_idx];
                        let img_next = &dog_octave[s_idx + 1];

                        // Process rows in parallel
                        (border..(height_i32 - border))
                            .into_par_iter()
                            .flat_map(|y| {
                                let mut row_keypoints = Vec::new();
                                for x in border..(width_i32 - border) {
                                    let val = Self::get_pixel_value(img_curr, x, y);

                                    let mut is_max = true;
                                    let mut is_min = true;

                                    'check_neighbors: for dz_offset in -1..=1 {
                                        let current_s_offset_img = match dz_offset {
                                            -1 => img_prev,
                                            0 => img_curr,
                                            1 => img_next,
                                            _ => unreachable!(),
                                        };
                                        for dy_offset in -1..=1 {
                                            for dx_offset in -1..=1 {
                                                if dz_offset == 0
                                                    && dy_offset == 0
                                                    && dx_offset == 0
                                                {
                                                    continue;
                                                }
                                                let neighbor_val = Self::get_pixel_value(
                                                    current_s_offset_img,
                                                    x + dx_offset,
                                                    y + dy_offset,
                                                );

                                                if val <= neighbor_val {
                                                    is_max = false;
                                                }
                                                if val >= neighbor_val {
                                                    is_min = false;
                                                }
                                                if !is_max && !is_min {
                                                    break 'check_neighbors;
                                                }
                                            }
                                        }
                                    }

                                    if is_max || is_min {
                                        row_keypoints.push(KeyPoint {
                                            x: (x as f32 + 0.5) * scale_factor,
                                            y: (y as f32 + 0.5) * scale_factor,
                                            size: 0.0,
                                            angle: 0.0,
                                            response: val,
                                            octave: o_idx as i32,
                                            layer: s_idx as i32,
                                        });
                                    }
                                }
                                row_keypoints
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub(crate) fn assign_orientations(
        &self,
        keypoints: &[KeyPoint],
        gaussian_pyramid: &[Vec<GrayImage>],
    ) -> Vec<KeyPoint> {
        // Use parallel iterator from Rayon
        // collect() will gather results from different threads
        // flat_map is used because one input point can spawn multiple output points (with different angles)
        keypoints
            .par_iter() // <--- Replaces iter() with par_iter()
            .flat_map(|kp| {
                let mut oriented_keypoints_for_this_kp = Vec::new(); // Local vector for the point
                let octave_idx = kp.octave as usize;
                let gauss_layer_idx = (kp.layer).clamp(0, self.num_intervals as i32 + 2) as usize;

                if octave_idx >= gaussian_pyramid.len()
                    || gauss_layer_idx >= gaussian_pyramid[octave_idx].len()
                {
                    return oriented_keypoints_for_this_kp; // Return empty vector if index out of bounds
                }

                let gauss_image = &gaussian_pyramid[octave_idx][gauss_layer_idx];
                let (img_width, img_height) = gauss_image.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.size / scale_factor;

                if sigma_octave <= 0.0 {
                    return oriented_keypoints_for_this_kp;
                }

                let window_radius = (ORIENTATION_WINDOW_RADIUS_FACTOR
                    * ORIENTATION_GAUSSIAN_EXPANSION_FACTOR
                    * sigma_octave)
                    .round() as i32;
                let weight_sigma = ORIENTATION_GAUSSIAN_EXPANSION_FACTOR * sigma_octave;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let mut hist = [0.0f32; ORIENTATION_HIST_BINS];

                // --- Histogram building loop (remains sequential within task) ---
                for dy in -window_radius..=window_radius {
                    for dx in -window_radius..=window_radius {
                        let x_img = (x_octave + dx as f32).round() as i32;
                        let y_img = (y_octave + dy as f32).round() as i32;
                        if x_img < 1
                            || x_img >= (img_width - 1) as i32
                            || y_img < 1
                            || y_img >= (img_height - 1) as i32
                        {
                            continue;
                        }
                        let grad_x = Self::get_gauss_pixel_value(gauss_image, x_img + 1, y_img)
                            - Self::get_gauss_pixel_value(gauss_image, x_img - 1, y_img);
                        let grad_y = Self::get_gauss_pixel_value(gauss_image, x_img, y_img + 1)
                            - Self::get_gauss_pixel_value(gauss_image, x_img, y_img - 1);
                        let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                        let angle = grad_y.atan2(grad_x);
                        let weight =
                            (-(dx as f32 * dx as f32 + dy as f32 * dy as f32) / weight_denom).exp();
                        let angle_normalized = if angle < 0.0 { angle + 2.0 * PI } else { angle };
                        let bin_float =
                            angle_normalized * (ORIENTATION_HIST_BINS as f32) / (2.0 * PI);
                        let bin_idx = bin_float.floor() as usize % ORIENTATION_HIST_BINS;
                        hist[bin_idx] += magnitude * weight;
                    }
                }
                // --- End histogram building loop ---

                // --- Smoothing and peak finding (sequential) ---
                let mut smoothed_hist = hist;
                // ... (smoothing code unchanged) ...
                for _ in 0..ORIENTATION_SMOOTHING_ITERATIONS {
                    let prev_hist = smoothed_hist;
                    for i in 0..ORIENTATION_HIST_BINS {
                        let prev_bin = (i + ORIENTATION_HIST_BINS - 1) % ORIENTATION_HIST_BINS;
                        let next_bin = (i + 1) % ORIENTATION_HIST_BINS;
                        smoothed_hist[i] =
                            (prev_hist[prev_bin] + prev_hist[i] + prev_hist[next_bin]) / 3.0;
                    }
                }
                hist = smoothed_hist;

                let max_peak_val = hist.iter().fold(0.0_f32, |max, &val| max.max(val));
                let peak_threshold = max_peak_val * ORIENTATION_PEAK_RATIO;

                for i in 0..ORIENTATION_HIST_BINS {
                    let current_val = hist[i];
                    if current_val >= peak_threshold {
                        let prev_bin_idx = (i + ORIENTATION_HIST_BINS - 1) % ORIENTATION_HIST_BINS;
                        let next_bin_idx = (i + 1) % ORIENTATION_HIST_BINS;
                        let prev_val = hist[prev_bin_idx];
                        let next_val = hist[next_bin_idx];
                        if current_val > prev_val && current_val > next_val {
                            let interp_denom = prev_val - 2.0 * current_val + next_val;
                            let interpolated_offset = if interp_denom.abs() > 1e-5 {
                                0.5 * (prev_val - next_val) / interp_denom
                            } else {
                                0.0
                            };
                            let bin_center_angle =
                                (i as f32 + 0.5) * (2.0 * PI / ORIENTATION_HIST_BINS as f32);
                            let interpolated_angle = bin_center_angle
                                + interpolated_offset * (2.0 * PI / ORIENTATION_HIST_BINS as f32);
                            let final_angle = interpolated_angle.rem_euclid(2.0 * PI);
                            let final_angle = if final_angle > PI {
                                final_angle - 2.0 * PI
                            } else {
                                final_angle
                            };
                            let mut new_kp = kp.clone();
                            new_kp.angle = final_angle;
                            oriented_keypoints_for_this_kp.push(new_kp); // Add to local vector
                        }
                    }
                }
                // --- End peak finding ---

                oriented_keypoints_for_this_kp // Return result for this point
            })
            .collect() // Collect results from all threads into one Vec<KeyPoint>
    }

    #[allow(dead_code)]
    pub(crate) fn assign_orientations_f32(
        &self,
        keypoints: &[KeyPoint],
        gaussian_pyramid: &[Vec<ImageBuffer<Luma<f32>, Vec<f32>>>],
    ) -> Vec<KeyPoint> {
        keypoints
            .par_iter()
            .flat_map(|kp| {
                let mut oriented_keypoints_for_this_kp = Vec::new();
                let octave_idx = kp.octave as usize;
                let gauss_layer_idx = (kp.layer).clamp(0, self.num_intervals as i32 + 2) as usize;

                if octave_idx >= gaussian_pyramid.len()
                    || gauss_layer_idx >= gaussian_pyramid[octave_idx].len()
                {
                    return oriented_keypoints_for_this_kp;
                }

                let gauss_image = &gaussian_pyramid[octave_idx][gauss_layer_idx];
                let (img_width, img_height) = gauss_image.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.size / scale_factor;

                if sigma_octave <= 0.0 {
                    return oriented_keypoints_for_this_kp;
                }

                let window_radius = (ORIENTATION_WINDOW_RADIUS_FACTOR
                    * ORIENTATION_GAUSSIAN_EXPANSION_FACTOR
                    * sigma_octave)
                    .round() as i32;
                let weight_sigma = ORIENTATION_GAUSSIAN_EXPANSION_FACTOR * sigma_octave;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let mut hist = [0.0f32; ORIENTATION_HIST_BINS];

                for dy in -window_radius..=window_radius {
                    for dx in -window_radius..=window_radius {
                        let x_img = (x_octave + dx as f32).round() as i32;
                        let y_img = (y_octave + dy as f32).round() as i32;
                        if x_img < 1
                            || x_img >= (img_width - 1) as i32
                            || y_img < 1
                            || y_img >= (img_height - 1) as i32
                        {
                            continue;
                        }
                        let grad_x = Self::get_gauss_pixel_value_f32(gauss_image, x_img + 1, y_img)
                            - Self::get_gauss_pixel_value_f32(gauss_image, x_img - 1, y_img);
                        let grad_y = Self::get_gauss_pixel_value_f32(gauss_image, x_img, y_img + 1)
                            - Self::get_gauss_pixel_value_f32(gauss_image, x_img, y_img - 1);
                        let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                        let angle = grad_y.atan2(grad_x);
                        let weight =
                            (-(dx as f32 * dx as f32 + dy as f32 * dy as f32) / weight_denom).exp();
                        let angle_normalized = if angle < 0.0 { angle + 2.0 * PI } else { angle };
                        let bin_float =
                            angle_normalized * (ORIENTATION_HIST_BINS as f32) / (2.0 * PI);
                        let bin_idx = bin_float.floor() as usize % ORIENTATION_HIST_BINS;
                        hist[bin_idx] += magnitude * weight;
                    }
                }

                // Сглаживание гистограммы
                let mut smoothed_hist = hist;
                for _ in 0..ORIENTATION_SMOOTHING_ITERATIONS {
                    let prev_hist = smoothed_hist;
                    for i in 0..ORIENTATION_HIST_BINS {
                        let prev_bin = (i + ORIENTATION_HIST_BINS - 1) % ORIENTATION_HIST_BINS;
                        let next_bin = (i + 1) % ORIENTATION_HIST_BINS;
                        smoothed_hist[i] =
                            (prev_hist[prev_bin] + prev_hist[i] + prev_hist[next_bin]) / 3.0;
                    }
                }
                hist = smoothed_hist;

                let max_peak_val = hist.iter().fold(0.0_f32, |max, &val| max.max(val));
                let peak_threshold = max_peak_val * ORIENTATION_PEAK_RATIO;

                for i in 0..ORIENTATION_HIST_BINS {
                    let current_val = hist[i];
                    if current_val >= peak_threshold {
                        let prev_bin_idx = (i + ORIENTATION_HIST_BINS - 1) % ORIENTATION_HIST_BINS;
                        let next_bin_idx = (i + 1) % ORIENTATION_HIST_BINS;
                        let prev_val = hist[prev_bin_idx];
                        let next_val = hist[next_bin_idx];
                        if current_val > prev_val && current_val > next_val {
                            let interp_denom = prev_val - 2.0 * current_val + next_val;
                            let interpolated_offset = if interp_denom.abs() > 1e-5 {
                                0.5 * (prev_val - next_val) / interp_denom
                            } else {
                                0.0
                            };
                            let bin_center_angle =
                                (i as f32 + 0.5) * (2.0 * PI / ORIENTATION_HIST_BINS as f32);
                            let interpolated_angle = bin_center_angle
                                + interpolated_offset * (2.0 * PI / ORIENTATION_HIST_BINS as f32);
                            let final_angle = interpolated_angle.rem_euclid(2.0 * PI);
                            let final_angle = if final_angle > PI {
                                final_angle - 2.0 * PI
                            } else {
                                final_angle
                            };
                            let mut new_kp = kp.clone();
                            new_kp.angle = final_angle;
                            oriented_keypoints_for_this_kp.push(new_kp);
                        }
                    }
                }

                oriented_keypoints_for_this_kp
            })
            .collect()
    }

    /// Detects SIFT keypoints in an image.
    pub fn detect(&self, img: &DynamicImage) -> Vec<KeyPoint> {
        // 1. Convert to grayscale
        let gray_img = img.to_luma8();

        // 2. Prepare initial image
        let initial_blur_amount = if self.sigma > self.assumed_blur {
            (self.sigma.powi(2) - self.assumed_blur.powi(2)).sqrt()
        } else {
            0.0
        };
        let base_image = if initial_blur_amount > 1e-4 {
            gaussian_blur_f32(&gray_img, initial_blur_amount)
        } else {
            gray_img.clone()
        };

        // 3. Build Gaussian pyramid
        let gaussian_pyramid = self.generate_gaussian_pyramid(&base_image);

        // 4. Build Difference-of-Gaussians (DoG) pyramid
        let dog_pyramid = self.generate_dog_pyramid(&gaussian_pyramid);

        // 5. Find initial scale-space extrema
        let initial_keypoints = self.find_scale_space_extrema(&dog_pyramid);
        // println!("Found {} initial extrema.", initial_keypoints.len());

        // 6. Refine extrema location and filter by contrast & edge response
        let refined_keypoints = self.refine_and_filter_extrema(&initial_keypoints, &dog_pyramid);
        // println!("Found {} refined keypoints after filtering.", refined_keypoints.len());

        // 7. Assign orientations
        // Pass Gaussian pyramid since gradients are computed from it
        let oriented_keypoints = self.assign_orientations(&refined_keypoints, &gaussian_pyramid);

        println!(
            "Found {} final keypoints after orientation assignment.",
            oriented_keypoints.len()
        ); // Debug

        oriented_keypoints
    }

    /// Normalizes and clips a descriptor vector.
    pub(crate) fn normalize_and_clip_descriptor(desc: &mut [f32]) {
        let norm = desc.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            // Avoid division by zero
            return;
        }

        let norm_inv = 1.0 / norm;
        let mut new_norm_sq = 0.0;
        for val in desc.iter_mut() {
            *val *= norm_inv;
            *val = val.min(DESC_MAG_THR); // Clipping
            new_norm_sq += *val * *val;
        }

        // Second normalization after clipping
        let new_norm = new_norm_sq.sqrt();
        if new_norm < 1e-8 {
            return;
        }
        let new_norm_inv = 1.0 / new_norm;
        for val in desc.iter_mut() {
            *val *= new_norm_inv;
        }
    }

    /// Computes SIFT descriptors for the given keypoints.
    /// Uses Gaussian pyramid for gradient computation.
    pub fn compute(
        &self,
        gaussian_pyramid: &[Vec<GrayImage>],
        keypoints: &[KeyPoint],
    ) -> Vec<Vec<f32>> {
        let desc_len = DESC_WINDOW_WIDTH * DESC_WINDOW_WIDTH * DESC_HIST_BINS;

        // Use parallel iterator from Rayon
        // map() converts each keypoint to a descriptor
        keypoints
            .par_iter() // <--- Replaces iter() with par_iter()
            .map(|kp| {
                let mut hist = vec![0.0f32; desc_len]; // Local histogram for the point
                let octave_idx = kp.octave as usize;
                let gauss_layer_idx = (kp.layer).clamp(0, self.num_intervals as i32 + 2) as usize;

                // Check bounds (return zero descriptor if out of bounds)
                if octave_idx >= gaussian_pyramid.len()
                    || gauss_layer_idx >= gaussian_pyramid[octave_idx].len()
                {
                    // eprintln!("Warning: Keypoint octave/layer index out of bounds during descriptor computation. KP: {:?}", kp);
                    return hist; // Return zero vector
                }

                let gauss_image = &gaussian_pyramid[octave_idx][gauss_layer_idx];
                let (img_width, img_height) = gauss_image.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.size / scale_factor;

                if sigma_octave <= 0.0 {
                    // eprintln!("Warning: Non-positive sigma_octave encountered ({}) during descriptor computation for KP: {:?}", sigma_octave, kp);
                    return hist; // Return zero vector
                }

                let angle = kp.angle;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let bin_width_pixels = DESC_PATCH_SCALE_FACTOR * sigma_octave;
                let window_width_pixels = bin_width_pixels * (DESC_WINDOW_WIDTH as f32);
                let weight_sigma = 0.5 * window_width_pixels;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let sample_radius = (window_width_pixels * 2.0f32.sqrt() * 0.5).ceil() as i32;

                // --- Descriptor histogram building loop (sequential within task) ---
                for dy_img in -sample_radius..=sample_radius {
                    for dx_img in -sample_radius..=sample_radius {
                        let px = dx_img as f32;
                        let py = dy_img as f32;
                        let rx = cos_a * px + sin_a * py;
                        let ry = -sin_a * px + cos_a * py;
                        let x_bin_cont =
                            rx / bin_width_pixels + (DESC_WINDOW_WIDTH as f32) / 2.0 - 0.5;
                        let y_bin_cont =
                            ry / bin_width_pixels + (DESC_WINDOW_WIDTH as f32) / 2.0 - 0.5;

                        if x_bin_cont > -1.0
                            && x_bin_cont < (DESC_WINDOW_WIDTH as f32)
                            && y_bin_cont > -1.0
                            && y_bin_cont < (DESC_WINDOW_WIDTH as f32)
                        {
                            let x_sample = x_octave + px;
                            let y_sample = y_octave + py;
                            if x_sample < 0.0
                                || x_sample >= (img_width - 1) as f32
                                || y_sample < 0.0
                                || y_sample >= (img_height - 1) as f32
                            {
                                continue;
                            }

                            let grad_x = Self::get_gauss_pixel_bilinear(
                                gauss_image,
                                x_sample + 1.0,
                                y_sample,
                            ) - Self::get_gauss_pixel_bilinear(
                                gauss_image,
                                x_sample - 1.0,
                                y_sample,
                            );
                            let grad_y = Self::get_gauss_pixel_bilinear(
                                gauss_image,
                                x_sample,
                                y_sample + 1.0,
                            ) - Self::get_gauss_pixel_bilinear(
                                gauss_image,
                                x_sample,
                                y_sample - 1.0,
                            );
                            let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                            let pixel_angle = grad_y.atan2(grad_x);
                            let angle_relative = (pixel_angle - angle).rem_euclid(2.0 * PI);
                            let weight = (-(px * px + py * py) / weight_denom).exp();
                            let weighted_mag = magnitude * weight;
                            let angle_bin_cont =
                                angle_relative * (DESC_HIST_BINS as f32) / (2.0 * PI);
                            let x_bin_idx = x_bin_cont.floor() as i32;
                            let y_bin_idx = y_bin_cont.floor() as i32;
                            let angle_bin_idx = angle_bin_cont.floor() as i32;
                            let dx_interp = x_bin_cont - x_bin_idx as f32;
                            let dy_interp = y_bin_cont - y_bin_idx as f32;
                            let da_interp = angle_bin_cont - angle_bin_idx as f32;

                            for i in 0..2 {
                                for j in 0..2 {
                                    for k in 0..2 {
                                        let ix = x_bin_idx + i;
                                        let iy = y_bin_idx + j;
                                        let ia =
                                            (angle_bin_idx + k).rem_euclid(DESC_HIST_BINS as i32);
                                        if ix >= 0
                                            && ix < DESC_WINDOW_WIDTH as i32
                                            && iy >= 0
                                            && iy < DESC_WINDOW_WIDTH as i32
                                        {
                                            let weight_x =
                                                if i == 0 { 1.0 - dx_interp } else { dx_interp };
                                            let weight_y =
                                                if j == 0 { 1.0 - dy_interp } else { dy_interp };
                                            let weight_a =
                                                if k == 0 { 1.0 - da_interp } else { da_interp };
                                            let contribution =
                                                weighted_mag * weight_x * weight_y * weight_a;
                                            let hist_index = (iy * DESC_WINDOW_WIDTH as i32 + ix)
                                                * DESC_HIST_BINS as i32
                                                + ia;
                                            hist[hist_index as usize] += contribution;
                                            // Update local hist
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // --- End descriptor histogram building loop ---

                // Normalize local hist
                Self::normalize_and_clip_descriptor(&mut hist);
                hist // Return ready descriptor for this point
            })
            .collect() // Collect results from all threads into one Vec<Vec<f32>>
    }

    /// Computes SIFT descriptors using Gaussian pyramid in f32 format.
    pub fn compute_f32(
        &self,
        gaussian_pyramid: &[Vec<ImageBuffer<Luma<f32>, Vec<f32>>>],
        keypoints: &[KeyPoint],
    ) -> Vec<Vec<f32>> {
        let desc_len = DESC_WINDOW_WIDTH * DESC_WINDOW_WIDTH * DESC_HIST_BINS;

        keypoints
            .par_iter()
            .map(|kp| {
                let mut hist = vec![0.0f32; desc_len];
                let octave_idx = kp.octave as usize;
                let gauss_layer_idx = (kp.layer).clamp(0, self.num_intervals as i32 + 2) as usize;

                if octave_idx >= gaussian_pyramid.len()
                    || gauss_layer_idx >= gaussian_pyramid[octave_idx].len()
                {
                    return hist;
                }

                let gauss_image = &gaussian_pyramid[octave_idx][gauss_layer_idx];
                let (img_width, img_height) = gauss_image.dimensions();
                let scale_factor = 2.0_f32.powi(kp.octave);
                let x_octave = kp.x / scale_factor;
                let y_octave = kp.y / scale_factor;
                let sigma_octave = kp.size / scale_factor;

                if sigma_octave <= 0.0 {
                    return hist;
                }

                let angle = kp.angle;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let bin_width_pixels = DESC_PATCH_SCALE_FACTOR * sigma_octave;
                let window_width_pixels = bin_width_pixels * (DESC_WINDOW_WIDTH as f32);
                let weight_sigma = 0.5 * window_width_pixels;
                let weight_denom = 2.0 * weight_sigma * weight_sigma;
                let sample_radius = (window_width_pixels * 2.0f32.sqrt() * 0.5).ceil() as i32;

                for dy_img in -sample_radius..=sample_radius {
                    for dx_img in -sample_radius..=sample_radius {
                        let px = dx_img as f32;
                        let py = dy_img as f32;
                        let rx = cos_a * px + sin_a * py;
                        let ry = -sin_a * px + cos_a * py;
                        let x_bin_cont =
                            rx / bin_width_pixels + (DESC_WINDOW_WIDTH as f32) / 2.0 - 0.5;
                        let y_bin_cont =
                            ry / bin_width_pixels + (DESC_WINDOW_WIDTH as f32) / 2.0 - 0.5;

                        if x_bin_cont > -1.0
                            && x_bin_cont < (DESC_WINDOW_WIDTH as f32)
                            && y_bin_cont > -1.0
                            && y_bin_cont < (DESC_WINDOW_WIDTH as f32)
                        {
                            let x_sample = x_octave + px;
                            let y_sample = y_octave + py;
                            if x_sample < 0.0
                                || x_sample >= (img_width - 1) as f32
                                || y_sample < 0.0
                                || y_sample >= (img_height - 1) as f32
                            {
                                continue;
                            }

                            let grad_x = Self::get_gauss_pixel_bilinear_f32(
                                gauss_image,
                                x_sample + 1.0,
                                y_sample,
                            ) - Self::get_gauss_pixel_bilinear_f32(
                                gauss_image,
                                x_sample - 1.0,
                                y_sample,
                            );
                            let grad_y = Self::get_gauss_pixel_bilinear_f32(
                                gauss_image,
                                x_sample,
                                y_sample + 1.0,
                            ) - Self::get_gauss_pixel_bilinear_f32(
                                gauss_image,
                                x_sample,
                                y_sample - 1.0,
                            );
                            let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                            let pixel_angle = grad_y.atan2(grad_x);
                            let angle_relative = (pixel_angle - angle).rem_euclid(2.0 * PI);
                            let weight = (-(px * px + py * py) / weight_denom).exp();
                            let weighted_mag = magnitude * weight;
                            let angle_bin_cont =
                                angle_relative * (DESC_HIST_BINS as f32) / (2.0 * PI);
                            let x_bin_idx = x_bin_cont.floor() as i32;
                            let y_bin_idx = y_bin_cont.floor() as i32;
                            let angle_bin_idx = angle_bin_cont.floor() as i32;
                            let dx_interp = x_bin_cont - x_bin_idx as f32;
                            let dy_interp = y_bin_cont - y_bin_idx as f32;
                            let da_interp = angle_bin_cont - angle_bin_idx as f32;

                            for i in 0..2 {
                                for j in 0..2 {
                                    for k in 0..2 {
                                        let ix = x_bin_idx + i;
                                        let iy = y_bin_idx + j;
                                        let ia =
                                            (angle_bin_idx + k).rem_euclid(DESC_HIST_BINS as i32);
                                        if ix >= 0
                                            && ix < DESC_WINDOW_WIDTH as i32
                                            && iy >= 0
                                            && iy < DESC_WINDOW_WIDTH as i32
                                        {
                                            let weight_x =
                                                if i == 0 { 1.0 - dx_interp } else { dx_interp };
                                            let weight_y =
                                                if j == 0 { 1.0 - dy_interp } else { dy_interp };
                                            let weight_a =
                                                if k == 0 { 1.0 - da_interp } else { da_interp };
                                            let contribution =
                                                weighted_mag * weight_x * weight_y * weight_a;
                                            let hist_index = (iy * DESC_WINDOW_WIDTH as i32 + ix)
                                                * DESC_HIST_BINS as i32
                                                + ia;
                                            hist[hist_index as usize] += contribution;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Self::normalize_and_clip_descriptor(&mut hist);
                hist
            })
            .collect()
    }

    /// CPU path for SIFT: detection and descriptor computation.
    pub fn detect_and_compute_cpu(&self, img: &DynamicImage) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        // 1. Convert to grayscale
        let gray_img = img.to_luma8();

        // 2. Prepare initial image
        let initial_blur_amount = if self.sigma > self.assumed_blur {
            (self.sigma.powi(2) - self.assumed_blur.powi(2)).sqrt()
        } else {
            0.0
        };
        let base_image = if initial_blur_amount > 1e-4 {
            gaussian_blur_f32(&gray_img, initial_blur_amount)
        } else {
            gray_img.clone()
        };

        // 3. Build Gaussian pyramid
        let gaussian_pyramid = self.generate_gaussian_pyramid(&base_image);

        // 4. Build Difference-of-Gaussians (DoG) pyramid
        let dog_pyramid = self.generate_dog_pyramid(&gaussian_pyramid);

        // 5. Find initial scale-space extrema
        let initial_keypoints = self.find_scale_space_extrema(&dog_pyramid);

        // 6. Refine extrema location and filter by contrast & edge response
        let refined_keypoints = self.refine_and_filter_extrema(&initial_keypoints, &dog_pyramid);

        // 7. Assign orientations (uses Gaussian pyramid)
        let oriented_keypoints = self.assign_orientations(&refined_keypoints, &gaussian_pyramid);
        println!(
            "Found {} keypoints with orientation.",
            oriented_keypoints.len()
        ); // Debug

        // 8. Compute descriptors (uses Gaussian pyramid)
        let descriptors = self.compute(&gaussian_pyramid, &oriented_keypoints);
        println!("Computed {} descriptors.", descriptors.len()); // Debug

        (oriented_keypoints, descriptors)
    }

    /// Full SIFT process with backend selection. For backward compatibility, `detect_and_compute`
    /// uses the CPU, while this function allows trying WebGPU with fallback.
    pub fn detect_and_compute_with_backend(
        &self,
        img: &DynamicImage,
        backend: SiftBackend,
    ) -> Result<(Vec<KeyPoint>, Vec<Vec<f32>>), String> {
        match backend {
            SiftBackend::Cpu => Ok(self.detect_and_compute_cpu(img)),
            #[cfg(not(target_arch = "wasm32"))]
            SiftBackend::WebGpu => {
                // Use GPU implementation
                use crate::gpu_sift::{GpuSiftConfig, GpuSiftContext};

                let gray = img.to_luma8();
                let (width, height) = gray.dimensions();
                let pixels = gray.into_raw();

                let config = GpuSiftConfig {
                    octaves: self.num_octaves,
                    scales: self.num_intervals + 3, // SIFT uses s+3 scales per octave
                    base_sigma: self.sigma,
                    contrast_threshold: self.contrast_threshold,
                    edge_threshold: self.edge_threshold,
                };

                // Run GPU detection synchronously using tokio
                let result = std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new()
                        .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

                    rt.block_on(async {
                        let ctx = GpuSiftContext::new(config)
                            .await
                            .map_err(|e| format!("GPU init failed: {}", e))?;

                        ctx.detect(&pixels, width, height)
                            .await
                            .map_err(|e| format!("GPU detection failed: {}", e))
                    })
                })
                .join()
                .map_err(|_| "GPU thread panicked".to_string())??;

                // Convert [u8; 128] descriptors to Vec<f32>
                let (keypoints, descriptors_u8) = result;
                let descriptors: Vec<Vec<f32>> = descriptors_u8
                    .into_iter()
                    .map(|d| d.iter().map(|&v| v as f32 / 255.0).collect())
                    .collect();

                Ok((keypoints, descriptors))
            }
            #[cfg(not(target_arch = "wasm32"))]
            SiftBackend::WebGpuV2 => {
                // Use GPU V2 implementation (full texture-based pipeline)
                use crate::gpu_sift_v2::{GpuSiftConfigV2, GpuSiftV2};

                let gray = img.to_luma8();
                let (width, height) = gray.dimensions();
                let pixels = gray.into_raw();

                let config = GpuSiftConfigV2 {
                    octaves: self.num_octaves,
                    scales_per_octave: self.num_intervals,
                    base_sigma: self.sigma,
                    contrast_threshold: self.contrast_threshold,
                    edge_threshold: self.edge_threshold,
                    max_keypoints: 4096, // Default limit
                };

                // Run GPU V2 detection synchronously using tokio
                let result = std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new()
                        .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

                    rt.block_on(async {
                        let mut ctx = GpuSiftV2::new(config)
                            .await
                            .map_err(|e| format!("GPU V2 init failed: {}", e))?;

                        ctx.detect(&pixels, width, height)
                            .await
                            .map_err(|e| format!("GPU V2 detection failed: {}", e))
                    })
                })
                .join()
                .map_err(|_| "GPU V2 thread panicked".to_string())??;

                // Convert [u8; 128] descriptors to Vec<f32>
                let (keypoints, descriptors_u8) = result;
                let descriptors: Vec<Vec<f32>> = descriptors_u8
                    .into_iter()
                    .map(|d| d.iter().map(|&v| v as f32 / 255.0).collect())
                    .collect();

                Ok((keypoints, descriptors))
            }
            #[cfg(target_arch = "wasm32")]
            SiftBackend::WebGpu | SiftBackend::WebGpuV2 => {
                Err("Sync GPU backend is not supported on WASM. Use async API.".to_string())
            }
            SiftBackend::WebGpuWithCpuFallback => {
                // Try GPU, fallback to CPU
                match self.detect_and_compute_with_backend(img, SiftBackend::WebGpu) {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        warn!("WebGPU failed ({}), falling back to CPU.", e);
                        Ok(self.detect_and_compute_cpu(img))
                    }
                }
            }
        }
    }

    /// Backward compatibility: use the CPU path by default.
    pub fn detect_and_compute(&self, img: &DynamicImage) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        self.detect_and_compute_cpu(img)
    }
}

// Utility functions
pub fn load_image_dyn(path: &str) -> Result<DynamicImage, image::ImageError> {
    image::open(path)
}

pub fn save_gray_image(image: &GrayImage, path: &str) -> Result<(), image::ImageError> {
    image.save(path)
}

// Helper function for visualizing DoG images (Luma<f32>)
// Normalizes values to the range [0, 255] and saves as a GrayImage
pub fn convert_f32_to_grayimage_normalized(
    img_f32: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> GrayImage {
    let (width, height) = img_f32.dimensions();
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for pixel_val in img_f32.iter() {
        min_val = min_val.min(*pixel_val);
        max_val = max_val.max(*pixel_val);
    }

    let mut gray_img = GrayImage::new(width, height);
    let range = max_val - min_val;

    if range.abs() < 1e-6 {
        // If image is almost flat
        let fill_val = if min_val > 0.0 {
            255
        } else if min_val < 0.0 {
            0
        } else {
            128
        };
        for y_coord in 0..height {
            for x_coord in 0..width {
                gray_img.put_pixel(x_coord, y_coord, Luma([fill_val]));
            }
        }
    } else {
        for y_coord in 0..height {
            for x_coord in 0..width {
                let val_f32 = img_f32.get_pixel(x_coord, y_coord)[0];
                let normalized_val = (val_f32 - min_val) / range; // Normalize to [0, 1]
                gray_img.put_pixel(
                    x_coord,
                    y_coord,
                    Luma([(normalized_val * 255.0).round() as u8]),
                );
            }
        }
    }
    gray_img
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    fn test_solve_linear_system_solvable() {
        let a2 = [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]];
        let b2 = [1.0, -2.0, 0.0];
        let expected2 = [1.0, -2.0, -2.0];

        match Sift::solve_linear_system(a2, b2) {
            Some(solution) => {
                assert!((solution[0] - expected2[0]).abs() < 1e-5, "x mismatch");
                assert!((solution[1] - expected2[1]).abs() < 1e-5, "y mismatch");
                assert!((solution[2] - expected2[2]).abs() < 1e-5, "z mismatch");
            }
            None => panic!("Expected a solution, but got None"),
        }
    }

    #[test]
    fn test_solve_linear_system_singular() {
        // Create a singular matrix (e.g., two rows are linearly dependent)
        // 1x + 2y + 3z = 1
        // 2x + 4y + 6z = 2  (second row = 2 * first)
        // 0x + 1y + 1z = 3
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]];
        let b = [1.0, 2.0, 3.0]; // b does not matter, matrix a is important

        let result = Sift::solve_linear_system(a, b);
        assert!(
            result.is_none(),
            "Expected None for a singular matrix, but got a solution"
        );
    }

    #[test]
    fn test_get_gauss_pixel_bilinear() {
        // Create a simple 3x3 image
        // 10 20 30
        // 40 50 60
        // 70 80 90
        // Values are normalized / 255.0
        let img = GrayImage::from_raw(3, 3, vec![10, 20, 30, 40, 50, 60, 70, 80, 90]).unwrap();

        let val_at = |x, y| Sift::get_gauss_pixel_bilinear(&img, x, y);
        let pixel_val = |v| v as f32 / 255.0;

        // 1. Exactly at pixel center (1, 1) -> should be 50/255
        assert!(
            (val_at(1.0, 1.0) - pixel_val(50)).abs() < 1e-6,
            "Center pixel mismatch"
        );

        // 2. Exactly at corner pixel (0, 0) -> should be 10/255
        assert!(
            (val_at(0.0, 0.0) - pixel_val(10)).abs() < 1e-6,
            "Corner pixel (0,0) mismatch"
        );

        // 3. Exactly at corner pixel (2, 2) -> should be 90/255
        assert!(
            (val_at(2.0, 2.0) - pixel_val(90)).abs() < 1e-6,
            "Corner pixel (2,2) mismatch"
        );

        // 4. Exactly halfway between (0,0) and (1,0) -> (10+20)/2 = 15
        assert!(
            (val_at(0.5, 0.0) - pixel_val(15)).abs() < 1e-6,
            "Midpoint x=0.5, y=0 mismatch"
        );

        // 5. Exactly halfway between (0,0) and (0,1) -> (10+40)/2 = 25
        assert!(
            (val_at(0.0, 0.5) - pixel_val(25)).abs() < 1e-6,
            "Midpoint x=0, y=0.5 mismatch"
        );

        // 6. Exactly in the center of the square (0,0), (1,0), (0,1), (1,1) -> (10+20+40+50)/4 = 30
        assert!(
            (val_at(0.5, 0.5) - pixel_val(30)).abs() < 1e-6,
            "Center of square mismatch"
        );

        // 7. Outside image bounds (should use edge value)
        // x=-0.5, y=0.5 -> should interpolate between q11=10, q21=20, q12=40, q22=50, but x0=-1, x1=0
        // Uses get_gauss_pixel_value which clones boundary.
        // q11=val(-1,0)=10, q21=val(0,0)=10, q12=val(-1,1)=40, q22=val(0,1)=40
        // x=-0.5 -> x0=-1, dx=0.5. y=0.5 -> y0=0, dy=0.5
        // val = 10*(0.5)*(0.5) + 10*(0.5)*(0.5) + 40*(0.5)*(0.5) + 40*(0.5)*(0.5)
        // val = 2.5 + 2.5 + 10 + 10 = 25
        assert!(
            (val_at(-0.5, 0.5) - pixel_val(25)).abs() < 1e-6,
            "Outside boundary interpolation mismatch"
        );

        // 8. Very far outside (should return corner value)
        assert!(
            (val_at(-10.0, -10.0) - pixel_val(10)).abs() < 1e-6,
            "Far outside boundary mismatch (TL)"
        );
        assert!(
            (val_at(10.0, 10.0) - pixel_val(90)).abs() < 1e-6,
            "Far outside boundary mismatch (BR)"
        );
    }
}
