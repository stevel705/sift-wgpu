# SIFT-rs

A high-performance implementation of SIFT (David G. Lowe's Scale-Invariant Feature Transform) in Rust with CPU and GPU (WebGPU/wgpu) backends.

## Features

- **Multiple backends**: CPU, WebGPU (GPU), WebGPU V2 (optimized texture-based pipeline)
- **Automatic fallback**: GPU with automatic CPU fallback if GPU is unavailable
- **Full SIFT pipeline**: Gaussian pyramid, DoG, extrema detection, orientation assignment, 128-dimensional descriptors
- **Visualization**: Built-in keypoint drawing on images

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
sift-rs = "0.1.0"
```

## Usage

### Library API

```rust
use image::open;
use sift::{Sift, SiftBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img = open("path/to/image.jpg")?;
    
    // Create SIFT detector with default parameters
    let sift = Sift::default();
    
    // Detect keypoints and compute descriptors (CPU)
    let (keypoints, descriptors) = sift.detect_and_compute(&img);
    
    // Or use a specific backend
    let (keypoints, descriptors) = sift.detect_and_compute_with_backend(
        &img, 
        SiftBackend::WebGpuV2  // GPU V2 pipeline
    )?;
    
    println!("Found {} keypoints", keypoints.len());
    Ok(())
}
```

### Available Backends

| Backend | Description |
|---------|-------------|
| `SiftBackend::Cpu` | Pure CPU implementation |
| `SiftBackend::WebGpu` | GPU implementation using wgpu |
| `SiftBackend::WebGpuV2` | Optimized GPU pipeline (texture-based) |
| `SiftBackend::WebGpuWithCpuFallback` | Try GPU, fallback to CPU on failure (default) |

### Custom Parameters

```rust
use sift::Sift;

let sift = Sift::new(
    1.6,   // sigma (base blur)
    4,     // num_octaves
    3,     // num_intervals (scales per octave)
    0.5,   // assumed_blur
    0.04,  // contrast_threshold
    10.0,  // edge_threshold
);
```

### Visualization

```rust
use image::{open, Rgb};
use sift::{Sift, draw_keypoints_to_image};

let img = open("image.jpg")?;
let sift = Sift::default();
let (keypoints, _) = sift.detect_and_compute(&img);

// Draw keypoints on image
let result = draw_keypoints_to_image(&img, &keypoints, Rgb([255, 0, 0]));
result.save("output.png")?;
```

## CLI

```sh
# Build
cargo build --release

# Run with default backend (auto GPU/CPU fallback)
./target/release/sift data/lenna.png

# Specify backend
./target/release/sift --backend cpu data/lenna.png
./target/release/sift --backend gpu data/lenna.png
./target/release/sift --backend gpuv2 data/lenna.png

# Or use environment variable
SIFT_BACKEND=gpuv2 ./target/release/sift data/lenna.png
```

## Web / WASM Support

The library supports compilation to WebAssembly (WASM) for use in browsers. It includes both a CPU backend (single-threaded) and a WebGPU backend.

### Prerequisites

- Rust toolchain
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer.html)

### Building for Web

```sh
# optional: --out-dir to specify output folder
wasm-pack build --target web --release --out-dir www/pkg 
```

### Running the Web Demo

The repository includes a webcam demo in the `www` folder.

1. Build the WASM package:
   ```sh
   wasm-pack build --target web --release
   ```

2. Link the package to the web folder:
   ```sh
   cd www
   ln -s ../pkg pkg
   ```
   *(Or manually copy the `pkg` folder into `www` if you are on Windows)*

3. Serve the `www` folder with a local server (HTTPS or localhost required for Camera API):
   ```sh
   # Python
   python3 -m http.server 8000
   
   # Node
   npx serve .
   ```

4. Open `http://localhost:8000` in a browser with WebGPU support (Chrome 113+, Edge).
   - If using `localhost`, Camera API works.
   - If using a network IP (e.g. on mobile), you **must** use HTTPS (e.g. via `ngrok`) or the camera will fail.

### Web API Usage

```javascript
import init, { SiftDetector, detect_sift_cpu } from './pkg/sift.js';

async function run() {
    await init();

    // 1. GPU Backend (Async, Persistent)
    // Initialize once (compiles shaders, allocates resources)
    const detector = await SiftDetector.new();
    
    // Detect frame (RGBA or Grayscale buffer)
    // detector returns { keypoints: [...], descriptors: Float32Array }
    const result = await detector.detect(imageData.data, width, height);
    
    console.log(`Found ${result.keypoint_count()} keypoints`);
    
    // Access result data
    const kps = result.get_keypoint(0); // { x, y, size, angle, octave, layer }
    const descriptors = result.get_descriptors(); 
    
    
    // 2. CPU Backend (Sync)
    const resultCpu = detect_sift_cpu(imageData.data, width, height);
}

run();
```

## Performance Note

- **CPU**: Uses optimized SIMD (via `wasm-opt`) but is single-threaded in the browser. Fast for 320p/480p, slower for HD.
- **WebGPU**: High initialization cost but scales well with resolution (720p+). Requires optimized texture pipeline (V2) which is the default in the web binding.

### CLI Options

```
Usage: sift [--backend cpu|gpu|gpuv2|gpu-fallback] <image_path>

Options:
  --backend cpu          Use CPU backend
  --backend gpu          Use GPU (WebGPU) backend
  --backend gpuv2        Use GPU V2 (optimized) backend  
  --backend gpu-fallback Use GPU with CPU fallback (default)
  -h, --help             Show help
```

## GPU Example

```rust
use sift::{GpuSiftConfigV2, GpuSiftV2};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("image.jpg")?.to_luma8();
    let (width, height) = img.dimensions();
    let pixels = img.into_raw();

    let config = GpuSiftConfigV2::default();
    let mut ctx = GpuSiftV2::new(config).await?;
    
    let (keypoints, descriptors) = ctx.detect(&pixels, width, height).await?;
    println!("Found {} keypoints", keypoints.len());
    
    Ok(())
}
```

## Project Structure

```
src/
├── lib.rs          # Public API exports
├── main.rs         # CLI application
├── sift.rs         # Core SIFT implementation (CPU)
├── keypoints.rs    # KeyPoint struct
├── gpu_sift.rs     # GPU backend V1
├── gpu_sift_v2.rs  # GPU backend V2 (optimized)
└── shaders/        # WGSL compute shaders
    ├── gpu_blur.wgsl
    ├── gpu_dog.wgsl
    ├── gpu_extrema.wgsl
    ├── gpu_orientation.wgsl
    ├── gpu_descriptor.wgsl
    └── ...
```

## References

- [Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2), 91-110.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Lowe, D. G. (1999). Object recognition from local scale-invariant features. ICCV 1999.](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
- [Lowe, D. G. (2004). SIFT: The scale invariant feature transform.](https://www.cs.ubc.ca/~lowe/keypoints/)

## TODO

- [x] Implement SIFT (CPU)
- [x] Add support for different image types
- [x] Add tests
- [x] Add examples
- [x] Add WebGPU support (V1 & V2)
- [x] Add WASM support
- [x] Add Web Demo with Camera
- [ ] Add documentation
- [ ] Add benchmarks

## License

MIT

## Legal Notice

SIFT was patented, but the patent has expired. This repo is primarily meant for educational purposes, but feel free to use the code for any purpose, commercial or otherwise. All I ask is that you cite or share this repo.