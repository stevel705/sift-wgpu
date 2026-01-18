pub mod gpu_sift;
pub mod gpu_sift_v2;
pub mod keypoints;
pub mod sift;
pub mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::str::FromStr;

// Re-export main types
pub use gpu_sift::{GpuSiftConfig, GpuSiftContext};
pub use gpu_sift_v2::{GpuSiftConfigV2, GpuSiftV2};
pub use keypoints::KeyPoint;
pub use sift::{convert_f32_to_grayimage_normalized, load_image_dyn, save_gray_image, Sift};

/// Backend selection for SIFT calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiftBackend {
    /// CPU only.
    Cpu,
    /// WebGPU only (errors are propagated up).
    WebGpu,
    /// Try WebGPU first, silently fallback to CPU on error.
    WebGpuWithCpuFallback,
    /// Full GPU V2 pipeline (texture-based, targeting <20ms).
    WebGpuV2,
}

impl Default for SiftBackend {
    fn default() -> Self {
        SiftBackend::WebGpuWithCpuFallback
    }
}

impl FromStr for SiftBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_lowercase();
        match normalized.as_str() {
            "cpu" => Ok(SiftBackend::Cpu),
            "webgpu" | "wgpu" | "gpu" => Ok(SiftBackend::WebGpu),
            "auto" | "fallback" | "prefer-gpu" => Ok(SiftBackend::WebGpuWithCpuFallback),
            "gpuv2" | "gpu-v2" | "webgpu-v2" | "v2" => Ok(SiftBackend::WebGpuV2),
            _ => Err(format!("Unknown SIFT backend: {value}")),
        }
    }
}

// Original functions from your example if they are still needed externally
use image::{open, DynamicImage, GrayImage, Rgb, RgbImage}; // Add RgbImage, Rgb
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut}; // Add drawing functions

/// Loads an image and converts it to grayscale.
pub fn load_and_convert_image(path: &str) -> GrayImage {
    let img = open(path).expect("Failed to open image");
    img.into_luma8()
}

/// Draws keypoints on an image.
///
/// # Arguments
/// * `img` - Source image (`DynamicImage`).
/// * `keypoints` - Slice of keypoints to draw.
/// * `color` - Color to draw points (e.g., `Rgb([255u8, 0, 0])` for red).
///
/// # Returns
/// * `RgbImage` - New image with drawn points.
pub fn draw_keypoints_to_image(
    img: &DynamicImage,
    keypoints: &[KeyPoint],
    color: Rgb<u8>,
) -> RgbImage {
    // Convert to Rgb8 to be able to draw in color
    let mut rgb_image = img.to_rgb8();

    // Create colored RGB image for drawing
    for kp in keypoints {
        let x = kp.x;
        let y = kp.y;
        let size: f32 = 2.0; //kp.size; // sigma of the point
        let angle = kp.angle; // orientation in radians

        // Circle radius is proportional to the point scale (sigma)
        // Multiplier 3.0 is empirical to make the circle visible
        let radius = (size * 3.0).round() as i32;
        // Minimum radius so very small points are visible
        let display_radius = radius.max(2);

        // Draw circle
        draw_filled_circle_mut(&mut rgb_image, (x as i32, y as i32), display_radius, color);

        // End point for orientation line
        let x_end = x + (display_radius as f32 * angle.cos());
        let y_end = y + (display_radius as f32 * angle.sin());

        // Draw orientation line
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
