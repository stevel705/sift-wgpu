// gpu_descriptor.wgsl
// Compute 128-dimensional SIFT descriptors
// 4x4 spatial bins × 8 orientation bins = 128 dimensions
// Based on VulkanSift approach with fixed-point atomic accumulation

const PI: f32 = 3.14159265359;
const NB_HIST: u32 = 4u;     // 4x4 spatial histogram
const NB_ORI: u32 = 8u;      // 8 orientation bins
const DESC_DIM: u32 = 128u;  // 4*4*8 = 128
const LAMBDA_DESCRIPTOR: f32 = 3.0;
const L2_NORM_THRESHOLD: f32 = 0.2;
const MAX_KEYPOINTS: u32 = 65536u;

struct DescParams {
    width: u32,
    height: u32,
    octave: u32,
    num_keypoints: u32,
}

struct KeypointIn {
    x: f32,
    y: f32,
    sigma: f32,
    angle: f32,
}

@group(0) @binding(0) var gaussian_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: DescParams;
@group(0) @binding(2) var<storage, read> keypoints: array<KeypointIn>;
@group(0) @binding(3) var<storage, read_write> descriptors: array<u32>;

// Shared workgroup memory for descriptor computation
var<workgroup> work_desc: array<atomic<u32>, 128>;
var<workgroup> kp: KeypointIn;
var<workgroup> euclidean_norm_acc: atomic<u32>;
var<workgroup> euclidean_norm: f32;

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

// 64 threads per workgroup for efficient parallel computation
@compute @workgroup_size(64, 1, 1)
fn compute_descriptor(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let kp_idx = wg_id.x;
    let tid = local_id.x;
    
    // Load keypoint and initialize descriptor
    if (tid == 0u) {
        kp = keypoints[kp_idx];
        atomicStore(&euclidean_norm_acc, 0u);
    }
    
    // Initialize descriptor elements (128 / 64 = 2 per thread)
    if (tid < 64u) {
        atomicStore(&work_desc[tid], 0u);
        atomicStore(&work_desc[tid + 64u], 0u);
    }
    workgroupBarrier();
    
    if (kp_idx >= params.num_keypoints) {
        return;
    }
    
    // Scale to current octave coordinates
    let scale_factor = f32(1u << params.octave);
    let kp_x = kp.x / scale_factor;
    let kp_y = kp.y / scale_factor;
    let sigma = kp.sigma / scale_factor;
    let orientation = kp.angle;  // Already in radians
    
    // Descriptor window parameters
    let scaled_lambda = LAMBDA_DESCRIPTOR * sigma;
    let radius = sqrt(2.0) * scaled_lambda * (f32(NB_HIST) + 1.0) * 0.5;
    let int_radius = i32(floor(radius + 0.5));
    
    // Rotation factors
    let cos_angle = cos(orientation) / scaled_lambda;
    let sin_angle = sin(orientation) / scaled_lambda;
    
    // Gaussian weighting scale
    let expf_scale = -1.0 / (2.0 * (f32(NB_HIST) / 2.0) * (f32(NB_HIST) / 2.0));
    
    // Fixed-point scale (16-bit to avoid overflow in norm computation)
    let fp_scale = 4096.0;
    
    // Process pixels in the descriptor window
    let box_size = int_radius * 2 + 1;
    let total_pixels = box_size * box_size;
    
    for (var pix_idx = i32(tid); pix_idx < total_pixels; pix_idx += 64) {
        let delta_y = (pix_idx / box_size) - int_radius;
        let delta_x = (pix_idx % box_size) - int_radius;
        
        let sample_x = i32(round(kp_x)) + delta_x;
        let sample_y = i32(round(kp_y)) + delta_y;
        
        // Check bounds
        if (sample_x < 1 || sample_x >= i32(params.width) - 1 ||
            sample_y < 1 || sample_y >= i32(params.height) - 1) {
            continue;
        }
        
        // Subpixel delta
        let subpix_delta_x = (round(kp_x) + f32(delta_x)) - kp_x;
        let subpix_delta_y = (round(kp_y) + f32(delta_y)) - kp_y;
        
        // Rotate coordinates by keypoint orientation
        let oriented_x = cos_angle * subpix_delta_x + sin_angle * subpix_delta_y;
        let oriented_y = cos_angle * subpix_delta_y - sin_angle * subpix_delta_x;
        
        // Check if in descriptor window
        if (abs(oriented_x) > f32(NB_HIST) / 2.0 + 0.5 || 
            abs(oriented_y) > f32(NB_HIST) / 2.0 + 0.5) {
            continue;
        }
        
        // Compute gradient
        let grad = compute_gradient(sample_x, sample_y);
        var grad_angle = atan2(grad.y, grad.x);
        
        // Normalize angle
        if (grad_angle < 0.0) {
            grad_angle += 2.0 * PI;
        }
        
        // Rotate gradient angle by negative keypoint orientation
        grad_angle = grad_angle - orientation;
        if (grad_angle < 0.0) {
            grad_angle += 2.0 * PI;
        } else if (grad_angle >= 2.0 * PI) {
            grad_angle -= 2.0 * PI;
        }
        
        // Magnitude with Gaussian weighting
        let mag = exp(expf_scale * (oriented_x * oriented_x + oriented_y * oriented_y)) * length(grad);
        
        // Compute histogram indices with trilinear interpolation
        let fhist_x = oriented_x + f32(NB_HIST) / 2.0;
        let fhist_y = oriented_y + f32(NB_HIST) / 2.0;
        let fbin = grad_angle * f32(NB_ORI) / (2.0 * PI);
        
        let hist_x = i32(floor(fhist_x - 0.5));
        let hist_y = i32(floor(fhist_y - 0.5));
        let bin = i32(floor(fbin));
        
        let rhist_x = fhist_x - (f32(hist_x) + 0.5);
        let rhist_y = fhist_y - (f32(hist_y) + 0.5);
        let rbin = fbin - f32(bin);
        
        // Trilinear interpolation across 2x2x2 neighbors
        for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
                for (var k = 0; k < 2; k++) {
                    let hx = hist_x + i;
                    let hy = hist_y + j;
                    let hb = (bin + k) % i32(NB_ORI);
                    
                    if (hx >= 0 && hx < i32(NB_HIST) && hy >= 0 && hy < i32(NB_HIST)) {
                        let weight_x = select(rhist_x, 1.0 - rhist_x, i == 0);
                        let weight_y = select(rhist_y, 1.0 - rhist_y, j == 0);
                        let weight_b = select(rbin, 1.0 - rbin, k == 0);
                        
                        let val = mag * weight_x * weight_y * weight_b * fp_scale;
                        let desc_idx = hy * i32(NB_HIST) * i32(NB_ORI) + hx * i32(NB_ORI) + hb;
                        
                        if (desc_idx >= 0 && desc_idx < 128) {
                            atomicAdd(&work_desc[desc_idx], u32(val));
                        }
                    }
                }
            }
        }
    }
    
    workgroupBarrier();
    
    // Compute L2 norm
    if (tid < 128u) {
        let val = atomicLoad(&work_desc[tid]);
        atomicAdd(&euclidean_norm_acc, val * val);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        euclidean_norm = sqrt(f32(atomicLoad(&euclidean_norm_acc)));
    }
    workgroupBarrier();
    
    // Clamp values to threshold
    let threshold = euclidean_norm * L2_NORM_THRESHOLD;
    if (tid < 128u) {
        let val = atomicLoad(&work_desc[tid]);
        let clamped = min(val, u32(threshold));
        atomicStore(&work_desc[tid], clamped);
    }
    workgroupBarrier();
    
    // Recompute norm after clamping
    if (tid == 0u) {
        atomicStore(&euclidean_norm_acc, 0u);
    }
    workgroupBarrier();
    
    if (tid < 128u) {
        let val = atomicLoad(&work_desc[tid]);
        atomicAdd(&euclidean_norm_acc, val * val);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        euclidean_norm = sqrt(f32(atomicLoad(&euclidean_norm_acc)));
        if (euclidean_norm < 0.0001) {
            euclidean_norm = 1.0;  // Prevent division by zero
        }
    }
    workgroupBarrier();
    
    // Normalize and convert to uint8 (packed into u32)
    // 128 values = 32 u32s (4 bytes each)
    if (tid < 32u) {
        let base_idx = tid * 4u;
        var packed: u32 = 0u;
        
        for (var i = 0u; i < 4u; i++) {
            let desc_idx = base_idx + i;
            let val = f32(atomicLoad(&work_desc[desc_idx])) * 512.0 / euclidean_norm;
            var u8_val = u32(clamp(val, 0.0, 255.0));
            packed |= (u8_val << (i * 8u));
        }
        
        // Write to output buffer
        descriptors[kp_idx * 32u + tid] = packed;
    }
}

