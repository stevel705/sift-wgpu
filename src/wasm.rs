#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn init_logging() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct JsKeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub octave: i32,
    pub layer: i32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SiftResult {
    // We cannot return Vec<JsKeyPoint> directly without serde or wrapping,
    // so we provide getters or just raw arrays.
    // Simplifying: return flattened arrays or use helper methods.
    // Or we can return a JsValue using serde_wasm_bindgen if we added it.
    // But let's stick to simple getters or methods.
    keypoints: Vec<JsKeyPoint>,
    descriptors: Vec<f32>, // Flattened descriptors
    descriptor_width: u32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SiftResult {
    pub fn keypoint_count(&self) -> usize {
        self.keypoints.len()
    }

    pub fn get_keypoint(&self, index: usize) -> JsKeyPoint {
        let kp = &self.keypoints[index];
        JsKeyPoint {
            x: kp.x,
            y: kp.y,
            size: kp.size,
            angle: kp.angle,
            octave: kp.octave,
            layer: kp.layer,
        }
    }

    // Return all keypoints as a flat Float32Array [x, y, size, angle, octave, layer, ...]
    pub fn get_keypoints_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.keypoints.len() * 6);
        for kp in &self.keypoints {
            flat.push(kp.x);
            flat.push(kp.y);
            flat.push(kp.size);
            flat.push(kp.angle);
            flat.push(kp.octave as f32);
            flat.push(kp.layer as f32);
        }
        flat
    }

    pub fn get_descriptors(&self) -> Vec<f32> {
        self.descriptors.clone()
    }
}

#[cfg(target_arch = "wasm32")]
use crate::sift::Sift;
#[cfg(target_arch = "wasm32")]
use image::{GrayImage, Luma};

#[cfg(target_arch = "wasm32")]
fn img_from_raw(data: &[u8], width: u32, height: u32) -> GrayImage {
    // Assuming data is RGBA or just Gray?
    // Usually canvas gives RGBA.
    // If input is RGBA:
    if data.len() as u32 == width * height * 4 {
        let mut gray = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                // Simple luminosity
                let r = data[idx] as f32;
                let g = data[idx + 1] as f32;
                let b = data[idx + 2] as f32;
                let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                gray.put_pixel(x, y, Luma([luma as u8]));
            }
        }
        gray
    } else if data.len() as u32 == width * height {
        // Already gray
        image::GrayImage::from_raw(width, height, data.to_vec()).unwrap()
    } else {
        panic!("Invalid image data length");
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn detect_sift_cpu(image_data: &[u8], width: u32, height: u32) -> SiftResult {
    let img = img_from_raw(image_data, width, height);
    let dyn_img = image::DynamicImage::ImageLuma8(img);

    let sift = Sift::default();
    let (kps, descs) = sift.detect_and_compute(&dyn_img);

    let js_kps: Vec<JsKeyPoint> = kps
        .into_iter()
        .map(|kp| JsKeyPoint {
            x: kp.x,
            y: kp.y,
            size: kp.size,
            angle: kp.angle,
            octave: kp.octave,
            layer: kp.layer,
        })
        .collect();

    let flat_descs: Vec<f32> = descs.into_iter().flatten().collect();

    SiftResult {
        keypoints: js_kps,
        descriptors: flat_descs,
        descriptor_width: 128,
    }
}

#[cfg(target_arch = "wasm32")]
use crate::gpu_sift_v2::{GpuSiftConfigV2, GpuSiftV2};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SiftDetector {
    // We wrap it in Option to allow taking it out if needed (though not needed here)
    // or just direct field.
    gpu: GpuSiftV2,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SiftDetector {
    /// Initialize the SIFT detector with GPU resources.
    /// This should be called once.
    pub async fn new() -> Result<SiftDetector, JsValue> {
        let config = GpuSiftConfigV2::default();
        let gpu = GpuSiftV2::new(config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to init GPU SIFT: {}", e)))?;
        Ok(SiftDetector { gpu })
    }

    /// Run SIFT detection on the provided image data.
    /// Reuse the same detector instance for best performance.
    pub async fn detect(
        &mut self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<SiftResult, JsValue> {
        let img = img_from_raw(image_data, width, height);
        let pixels = img.into_raw();

        let (kps, descs_u8) = self
            .gpu
            .detect(&pixels, width, height)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to detect: {}", e)))?;

        let js_kps: Vec<JsKeyPoint> = kps
            .into_iter()
            .map(|kp| JsKeyPoint {
                x: kp.x,
                y: kp.y,
                size: kp.size,
                angle: kp.angle,
                octave: kp.octave,
                layer: kp.layer,
            })
            .collect();

        // GpuSiftV2 returns Vec<[u8; 128]>, normalize to f32
        let flat_descs: Vec<f32> = descs_u8
            .iter()
            .flat_map(|d| d.iter().map(|&v| v as f32 / 255.0))
            .collect();

        Ok(SiftResult {
            keypoints: js_kps,
            descriptors: flat_descs,
            descriptor_width: 128,
        })
    }
}

// Kept for backward compat or easy testing, but discouraged for loop
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn detect_sift_gpu(
    image_data: &[u8],
    width: u32,
    height: u32,
) -> Result<SiftResult, JsValue> {
    // Warning: Creates new device every time!
    let mut detector = SiftDetector::new().await?;
    detector.detect(image_data, width, height).await
}
