// gpu_extrema.wgsl
// Scale-space extrema detection on GPU
// Detects local minima/maxima in 3x3x3 neighborhood across DoG scales

@group(0) @binding(0) var dog_prev: texture_2d<f32>;
@group(0) @binding(1) var dog_curr: texture_2d<f32>;
@group(0) @binding(2) var dog_next: texture_2d<f32>;

struct ExtremaParams {
    width: u32,
    height: u32,
    octave: u32,
    scale: u32,
    contrast_threshold: f32,
    edge_threshold: f32,
    sigma: f32,
    _pad: u32,
}

struct Keypoint {
    x: f32,
    y: f32,
    sigma: f32,
    response: f32,
    octave: u32,
    scale: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: ExtremaParams;
@group(0) @binding(4) var<storage, read_write> keypoint_count: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> keypoints: array<Keypoint>;

const MAX_KEYPOINTS: u32 = 32768u;

fn load_dog(tex: texture_2d<f32>, x: i32, y: i32) -> f32 {
    let coord = vec2<i32>(
        clamp(x, 0, i32(params.width) - 1),
        clamp(y, 0, i32(params.height) - 1)
    );
    return textureLoad(tex, coord, 0).r;
}

fn is_extremum(center: f32, x: i32, y: i32) -> bool {
    // Skip low contrast candidates early
    if (abs(center) < params.contrast_threshold) {
        return false;
    }
    
    // Check if center is max or min in 3x3x3 neighborhood
    var is_max = true;
    var is_min = true;
    
    // Check all 26 neighbors (3x3x3 cube minus center)
    // Current scale neighbors (8 pixels)
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx != 0 || dy != 0) {
                let v = load_dog(dog_curr, x + dx, y + dy);
                if (v >= center) { is_max = false; }
                if (v <= center) { is_min = false; }
            }
        }
    }
    
    // Early exit if not extremum in 2D
    if (!is_max && !is_min) {
        return false;
    }
    
    // Previous scale neighbors (9 pixels)
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let v = load_dog(dog_prev, x + dx, y + dy);
            if (is_max && v >= center) { is_max = false; }
            if (is_min && v <= center) { is_min = false; }
        }
    }
    
    // Next scale neighbors (9 pixels)
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let v = load_dog(dog_next, x + dx, y + dy);
            if (is_max && v >= center) { is_max = false; }
            if (is_min && v <= center) { is_min = false; }
        }
    }
    
    return is_max || is_min;
}

fn check_edge_response(x: i32, y: i32) -> bool {
    // Compute Hessian matrix elements
    let center = load_dog(dog_curr, x, y);
    let dxx = load_dog(dog_curr, x + 1, y) + load_dog(dog_curr, x - 1, y) - 2.0 * center;
    let dyy = load_dog(dog_curr, x, y + 1) + load_dog(dog_curr, x, y - 1) - 2.0 * center;
    let dxy = (load_dog(dog_curr, x + 1, y + 1) - load_dog(dog_curr, x - 1, y + 1) 
             - load_dog(dog_curr, x + 1, y - 1) + load_dog(dog_curr, x - 1, y - 1)) * 0.25;
    
    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    
    // Reject edge responses
    if (det <= 0.0) {
        return false;
    }
    
    // Check edge ratio: (r+1)^2/r < threshold where r = edge_threshold
    let ratio = trace * trace / det;
    let r = params.edge_threshold;
    let threshold = (r + 1.0) * (r + 1.0) / r;
    
    return ratio < threshold;
}

@compute @workgroup_size(16, 16, 1)
fn detect_extrema(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    // Boundary check - need 1 pixel border for 3x3 neighborhood
    if (x < 1 || y < 1 || x >= i32(params.width) - 1 || y >= i32(params.height) - 1) {
        return;
    }
    
    let center = load_dog(dog_curr, x, y);
    
    // Check if this is an extremum
    if (!is_extremum(center, x, y)) {
        return;
    }
    
    // Check edge response (reject edge-like features)
    if (!check_edge_response(x, y)) {
        return;
    }
    
    // Found a valid keypoint - add it
    let idx = atomicAdd(&keypoint_count, 1u);
    if (idx < MAX_KEYPOINTS) {
        // Scale coordinates to original image space
        let scale_factor = f32(1u << params.octave);
        keypoints[idx].x = (f32(x) + 0.5) * scale_factor;
        keypoints[idx].y = (f32(y) + 0.5) * scale_factor;
        keypoints[idx].sigma = params.sigma;
        keypoints[idx].response = abs(center);
        keypoints[idx].octave = params.octave;
        keypoints[idx].scale = params.scale;
        keypoints[idx]._pad0 = 0u;
        keypoints[idx]._pad1 = 0u;
    }
}
