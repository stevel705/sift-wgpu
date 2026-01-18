// gpu_orientation.wgsl
// Compute dominant orientation for each keypoint using gradient histogram
// Based on VulkanSift approach: 36-bin histogram, gaussian weighting, parabolic interpolation

const PI: f32 = 3.14159265359;
const HIST_BINS: u32 = 36u;
const LAMBDA_ORIENTATION: f32 = 1.5;
const LOCAL_EXTREMA_THRESHOLD: f32 = 0.8;
const MAX_OUTPUT: u32 = 65536u;

struct OrientParams {
    width: u32,
    height: u32,
    octave: u32,
    num_keypoints: u32,
}

struct KeypointIn {
    x: f32,
    y: f32,
    sigma: f32,
    response: f32,
    octave: u32,
    scale: u32,
    _pad0: u32,
    _pad1: u32,
}

struct KeypointOut {
    x: f32,
    y: f32,
    sigma: f32,
    angle: f32,
}

@group(0) @binding(0) var gaussian_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: OrientParams;
@group(0) @binding(2) var<storage, read> keypoints_in: array<KeypointIn>;
@group(0) @binding(3) var<storage, read_write> output_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> keypoints_out: array<KeypointOut>;

// Workgroup-local storage
var<workgroup> histogram: array<atomic<u32>, 36>;
var<workgroup> smoothed_hist: array<u32, 36>;
var<workgroup> kp: KeypointIn;
var<workgroup> max_hist_value: atomic<u32>;

fn load_pixel(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return textureLoad(gaussian_texture, vec2<i32>(cx, cy), 0).r;
}

fn compute_gradient(x: i32, y: i32) -> vec2<f32> {
    let gx = 0.5 * (load_pixel(x + 1, y) - load_pixel(x - 1, y));
    let gy = 0.5 * (load_pixel(x, y + 1) - load_pixel(x, y - 1));
    return vec2<f32>(gx, gy);
}

// 32 threads per workgroup - each processes part of the circular region
@compute @workgroup_size(32, 1, 1)
fn compute_orientation(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let kp_idx = wg_id.x;
    let tid = local_id.x;
    
    // Load keypoint and initialize histogram
    if (tid == 0u) {
        kp = keypoints_in[kp_idx];
        atomicStore(&max_hist_value, 0u);
    }
    
    // Initialize histogram bins
    if (tid < HIST_BINS) {
        atomicStore(&histogram[tid], 0u);
    }
    workgroupBarrier();
    
    if (kp_idx >= params.num_keypoints) {
        return;
    }
    
    // Scale to current octave coordinates
    let scale_factor = f32(1u << kp.octave);
    let kp_x = kp.x / scale_factor;
    let kp_y = kp.y / scale_factor;
    let sigma = kp.sigma / scale_factor;
    
    // Window radius based on SIFT paper: 1.5 * sigma * 3 = 4.5 * sigma
    let scaled_lambda = LAMBDA_ORIENTATION * sigma;
    let box_radius = i32(floor(3.0 * scaled_lambda));
    let expf_scale = -1.0 / (2.0 * scaled_lambda * scaled_lambda);
    
    // Fixed-point conversion for atomic adds (like VulkanSift)
    let fp_scale = 65536.0;
    
    // Each thread processes a subset of the circular region
    let box_size = box_radius * 2 + 1;
    let total_pixels = box_size * box_size;
    
    for (var pix_idx = i32(tid); pix_idx < total_pixels; pix_idx += 32) {
        let delta_y = (pix_idx / box_size) - box_radius;
        let delta_x = (pix_idx % box_size) - box_radius;
        
        let sample_x = i32(round(kp_x)) + delta_x;
        let sample_y = i32(round(kp_y)) + delta_y;
        
        // Check bounds
        if (sample_x < 1 || sample_x >= i32(params.width) - 1 ||
            sample_y < 1 || sample_y >= i32(params.height) - 1) {
            continue;
        }
        
        // Check if in circular region
        let dist_sq = f32(delta_x * delta_x + delta_y * delta_y);
        if (dist_sq > f32(box_radius * box_radius)) {
            continue;
        }
        
        // Compute gradient
        let grad = compute_gradient(sample_x, sample_y);
        let mag = length(grad);
        var angle = atan2(grad.y, grad.x);
        
        // Normalize angle to [0, 2*PI)
        if (angle < 0.0) {
            angle += 2.0 * PI;
        }
        
        // Gaussian weight
        let weight = exp(dist_sq * expf_scale);
        let weighted_mag = mag * weight * fp_scale;
        
        // Compute bin index
        var bin_idx = i32(angle * f32(HIST_BINS) / (2.0 * PI));
        if (bin_idx < 0) {
            bin_idx += i32(HIST_BINS);
        } else if (bin_idx >= i32(HIST_BINS)) {
            bin_idx -= i32(HIST_BINS);
        }
        
        atomicAdd(&histogram[bin_idx], u32(weighted_mag));
    }
    
    workgroupBarrier();
    
    // Smooth histogram (6 passes like original SIFT)
    for (var smooth_iter = 0u; smooth_iter < 3u; smooth_iter++) {
        // First smoothing pass
        if (tid < HIST_BINS) {
            let prev_idx = (tid + HIST_BINS - 1u) % HIST_BINS;
            let next_idx = (tid + 1u) % HIST_BINS;
            let prev = atomicLoad(&histogram[prev_idx]);
            let curr = atomicLoad(&histogram[tid]);
            let next = atomicLoad(&histogram[next_idx]);
            smoothed_hist[tid] = (prev + curr + next) / 3u;
        }
        workgroupBarrier();
        
        // Second smoothing pass
        if (tid < HIST_BINS) {
            let prev_idx = (tid + HIST_BINS - 1u) % HIST_BINS;
            let next_idx = (tid + 1u) % HIST_BINS;
            let prev = smoothed_hist[prev_idx];
            let curr = smoothed_hist[tid];
            let next = smoothed_hist[next_idx];
            atomicStore(&histogram[tid], (prev + curr + next) / 3u);
        }
        workgroupBarrier();
    }
    
    // Find max value
    if (tid < HIST_BINS) {
        atomicMax(&max_hist_value, atomicLoad(&histogram[tid]));
    }
    workgroupBarrier();
    
    // Find peaks and output keypoints with orientations
    if (tid < HIST_BINS) {
        let curr_val = atomicLoad(&histogram[tid]);
        let max_val = atomicLoad(&max_hist_value);
        let threshold = u32(f32(max_val) * LOCAL_EXTREMA_THRESHOLD);
        
        if (curr_val >= threshold) {
            let prev_idx = (tid + HIST_BINS - 1u) % HIST_BINS;
            let next_idx = (tid + 1u) % HIST_BINS;
            let prev_val = atomicLoad(&histogram[prev_idx]);
            let next_val = atomicLoad(&histogram[next_idx]);
            
            // Check if local maximum
            if (curr_val > prev_val && curr_val > next_val) {
                // Parabolic interpolation for sub-bin accuracy
                let f_prev = f32(prev_val);
                let f_curr = f32(curr_val);
                let f_next = f32(next_val);
                var interp = 0.0;
                let denom = f_prev - 2.0 * f_curr + f_next;
                if (abs(denom) > 0.0001) {
                    interp = 0.5 * (f_prev - f_next) / denom;
                }
                
                var angle = (f32(tid) + 0.5 + interp) * (2.0 * PI) / f32(HIST_BINS);
                if (angle < 0.0) {
                    angle += 2.0 * PI;
                } else if (angle >= 2.0 * PI) {
                    angle -= 2.0 * PI;
                }
                
                // Output keypoint
                let out_idx = atomicAdd(&output_count, 1u);
                if (out_idx < MAX_OUTPUT) {
                    keypoints_out[out_idx].x = kp.x;
                    keypoints_out[out_idx].y = kp.y;
                    keypoints_out[out_idx].sigma = kp.sigma;
                    keypoints_out[out_idx].angle = angle;
                }
            }
        }
    }
}
