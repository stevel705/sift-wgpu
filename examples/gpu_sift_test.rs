// Example: Test GPU SIFT implementation
use sift::{GpuSiftConfigV2, GpuSiftV2};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU SIFT Test ===");

    // Load real image for testing
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/lenna.png".to_string());

    let (image, width, height) = if std::path::Path::new(&img_path).exists() {
        println!("Loading image from: {}", img_path);
        let img = image::open(&img_path)?.to_luma8();
        let width = img.width();
        let height = img.height();
        (img.into_raw(), width, height)
    } else {
        // Fallback: Create test image with features (corners and blobs)
        let width = 256u32;
        let height = 256u32;
        let mut image = vec![128u8; (width * height) as usize];

        // Add some corner-like features
        for i in 0..10 {
            let cx = 30 + i * 20;
            let cy = 30 + i * 20;
            for dy in -5i32..=5 {
                for dx in -5i32..=5 {
                    let x = (cx as i32 + dx) as u32;
                    let y = (cy as i32 + dy) as u32;
                    if x < width && y < height {
                        let idx = (y * width + x) as usize;
                        image[idx] = if dx.abs() <= 2 && dy.abs() <= 2 {
                            255
                        } else {
                            64
                        };
                    }
                }
            }
        }

        println!(
            "Created {}×{} synthetic test image with features",
            width, height
        );
        (image, width, height)
    };

    println!("Image size: {}×{}", width, height);

    // Initialize GPU SIFT V2 context
    let config = GpuSiftConfigV2::default();
    println!("Initializing GPU context...");

    let mut ctx = GpuSiftV2::new(config).await?;
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
