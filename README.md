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

## References

- [Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2), 91-110.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Lowe, D. G. (1999). Object recognition from local scale-invariant features. The proceedings of the seventh IEEE international conference on computer vision, 2, 1150-1157.](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
- [Lowe, D. G. (2004). SIFT: The scale invariant feature transform.](https://www.cs.ubc.ca/~lowe/keypoints/)

