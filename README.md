# SIFT-rs

This is an implementation of SIFT (David G. Lowe's scale-invariant feature transform) in Rust.

## Usage

```rust
use image::io::Reader as ImageReader;
use sift_rs::Sift;

fn main() {
    let img = ImageReader::open("path/to/image.jpg").unwrap().decode().unwrap().to_luma8();
    let sift = Sift::new();
    let keypoints = sift.detect(&img);
    let descriptors = sift.compute(&img, &keypoints);
}
```

## Example

```sh
cargo run --release --example sift path/to/image.jpg
```

```sh
cargo build --release 2>&1 | tail -2 && SIFT_PROFILE=1 ./target/release/sift --backend gpuv2 data/lenna.png 2>&1
```

## References

- [Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2), 91-110.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Lowe, D. G. (1999). Object recognition from local scale-invariant features. The proceedings of the seventh IEEE international conference on computer vision, 2, 1150-1157.](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
- [Lowe, D. G. (2004). SIFT: The scale invariant feature transform.](https://www.cs.ubc.ca/~lowe/keypoints/)


## TODO

- [x] Implement SIFT
- [x] Add support for different image types
- [x] Add tests
- [ ] Add documentation
- [x] Add examples
- [ ] Add benchmarks
- [ ] Add WASM support
- [ ] Add WebGPU support

## Legal Notice

SIFT was patented, but it has expired. This repo is primarily meant for educational purposes, but feel free to use my code any way you want, commercial or otherwise. All I ask is that you cite or share this repo.