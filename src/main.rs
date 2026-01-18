// src/main.rs

// Use library crate name 'sift' instead of package name 'sift_rs'
use image::Rgb;
use sift::draw_keypoints_to_image; // Import drawing function
use sift::sift::{load_image_dyn, Sift}; // Import Rgb for color
use sift::SiftBackend;
use std::env;
use std::io;
use std::time::Instant;

fn print_usage() {
    eprintln!("Usage: sift [--backend cpu|gpu|gpu-fallback] <image_path>");
    eprintln!("  --backend cpu          Use CPU backend (default)");
    eprintln!("  --backend gpu          Use GPU (WebGPU) backend");
    eprintln!("  --backend gpu-fallback Use GPU with CPU fallback");
    eprintln!();
    eprintln!("Environment variable SIFT_BACKEND can also be used.");
}

fn parse_args() -> Result<(SiftBackend, String), String> {
    let args: Vec<String> = env::args().collect();

    let mut backend: Option<SiftBackend> = None;
    let mut image_path: Option<String> = None;
    let mut i = 1;

    while i < args.len() {
        if args[i] == "--backend" {
            if i + 1 >= args.len() {
                return Err("--backend requires a value".to_string());
            }
            backend = Some(args[i + 1].parse().map_err(|e| format!("{}", e))?);
            i += 2;
        } else if args[i] == "--help" || args[i] == "-h" {
            print_usage();
            std::process::exit(0);
        } else if !args[i].starts_with('-') {
            image_path = Some(args[i].clone());
            i += 1;
        } else {
            return Err(format!("Unknown argument: {}", args[i]));
        }
    }

    // Fallback to environment variable if no CLI backend specified
    let backend = backend.unwrap_or_else(|| {
        env::var("SIFT_BACKEND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_default()
    });

    let image_path = image_path.ok_or_else(|| "No image path provided".to_string())?;

    Ok((backend, image_path))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, image_path) = parse_args().map_err(|e| {
        eprintln!("Error: {}", e);
        print_usage();
        io::Error::new(io::ErrorKind::InvalidInput, e)
    })?;

    if std::fs::create_dir_all("data").is_err() {
        eprintln!(
            "Warning: Could not create data directory. Ensure it exists or you have permissions."
        );
    }

    println!("Loading image from: {}", image_path);
    let img_dyn = match load_image_dyn(&image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error loading image '{}': {}", image_path, e);
            // ... (error messages) ...
            return Err(e.into());
        }
    };
    println!(
        "Image loaded successfully: {}x{}",
        img_dyn.width(),
        img_dyn.height()
    );
    // let img_dyn = img_dyn.resize_exact(320, 240, FilterType::Lanczos3);
    // println!(
    //     "Image resized to: {}x{}",
    //     img_dyn.width(),
    //     img_dyn.height()
    // );
    println!("Selected SIFT backend: {:?}", backend);

    // Use default SIFT parameters
    let sift = Sift::default();

    println!("Detecting SIFT keypoints and computing descriptors...");
    // Use new method returning both keypoints and descriptors
    let start_time = Instant::now();
    let (keypoints, descriptors) = sift
        .detect_and_compute_with_backend(&img_dyn, backend)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let duration = start_time.elapsed();

    println!("SIFT processing took: {:?}", duration); // <--- Output time
    println!("Found {} keypoints.", keypoints.len());
    println!("Computed {} descriptors.", descriptors.len());

    if !keypoints.is_empty() {
        // Print info about the first point as an example
        println!("Example keypoint [0]: x={:.2}, y={:.2}, size={:.2}, angle={:.2}, response={:.4}, octave={}, layer={}",
                 keypoints[0].x, keypoints[0].y, keypoints[0].size, keypoints[0].angle.to_degrees(), // Angle in degrees for clarity
                 keypoints[0].response, keypoints[0].octave, keypoints[0].layer);

        // Print size of the first descriptor
        if !descriptors.is_empty() {
            println!("Example descriptor [0] length: {}", descriptors[0].len());
            // Can print first few descriptor values
            println!(
                "Example descriptor [0] values (first 10): {:?}",
                &descriptors[0][..10.min(descriptors[0].len())]
            );
        }

        // --- Visualization ---
        println!("Drawing keypoints on image...");
        let color = Rgb([255u8, 0, 0]); // Red color
        let image_with_keypoints = draw_keypoints_to_image(&img_dyn, &keypoints, color);

        let output_path = format!("data/output_{:?}.png", backend);
        println!("Saving image with keypoints to: {}", output_path);
        if let Err(e) = image_with_keypoints.save(output_path) {
            eprintln!("Error saving output image: {}", e);
        } else {
            println!("PNG output image saved successfully.");
        }
        // --- End Visualization ---
    } else {
        println!("No keypoints found.");
    }

    println!("SIFT process finished.");
    Ok(())
}
