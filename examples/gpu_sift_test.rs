// Example: Test GPU SIFT implementation
use sift::{GpuSiftConfig, GpuSiftContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU SIFT Test ===");

    // Create test image (64x64 grayscale)
    let width = 64u32;
    let height = 64u32;
    let mut image = vec![0u8; (width * height) as usize];

    // Create simple pattern: gradient
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            image[idx] = ((x + y) * 255 / (width + height)) as u8;
        }
    }

    println!("Created {}×{} test image", width, height);

    // Initialize GPU SIFT context
    let config = GpuSiftConfig::default();
    println!("Initializing GPU context...");

    let ctx = GpuSiftContext::new(config).await?;
    println!("GPU context initialized successfully");

    // Detect keypoints and compute descriptors
    println!("Running GPU SIFT detection...");
    let (keypoints, descriptors) = ctx.detect(&image, width, height).await?;

    println!("Detected {} keypoints", keypoints.len());
    println!("Computed {} descriptors", descriptors.len());

    // Print first few keypoints
    for (i, kp) in keypoints.iter().take(5).enumerate() {
        println!(
            "  KP {}: ({:.2}, {:.2}) size={:.2} angle={:.2}",
            i, kp.x, kp.y, kp.size, kp.angle
        );
    }

    Ok(())
}
