// descriptor.wgsl
// Compute 128-dimensional SIFT descriptors for each keypoint
// Each descriptor uses 4 threads (2×2 quadrants), each maintaining local histogram

// ===== Bind Groups =====
struct DescriptorMeta {
    octaves: u32,
    scales: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> desc_meta: DescriptorMeta;
@group(0) @binding(1) var<storage, read> level_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> level_widths: array<u32>;
@group(0) @binding(3) var<storage, read> level_heights: array<u32>;

@group(1) @binding(0) var<storage, read> heap: array<u32>;

@group(2) @binding(0) var<storage, read> keypoints_final: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> num_final: u32;

@group(3) @binding(0) var<storage, read_write> descriptors: array<u32>; // packed 4×u8 per u32

// ===== Constants =====
const PI: f32 = 3.14159265359;
const DESC_BINS: u32 = 8u;     // orientation bins per histogram
const DESC_WIDTH: u32 = 4u;    // 4×4 grid
const DESC_SIZE: u32 = 128u;   // 4×4×8 = 128
const DESCRIPTOR_MAGNIFICATION: f32 = 3.0;

// ===== F16 Unpacking =====
fn read_pixel_f16(base_offset: u32, x: i32, y: i32, width: u32, height: u32) -> f32 {
    let cx = clamp(x, 0, i32(width) - 1);
    let cy = clamp(y, 0, i32(height) - 1);
    let idx = u32(cy) * width + u32(cx);
    let word_idx = idx >> 1u;
    let is_high = (idx & 1u) != 0u;
    let packed = heap[base_offset + word_idx];
    let unpacked = unpack2x16float(packed);
    return select(unpacked.x, unpacked.y, is_high);
}

// ===== Trilinear Interpolation =====
// Accumulate gradient into 4×4×8 descriptor with trilinear weights
fn accumulate_trilinear(
    local_hist: ptr<function, array<f32, 128>>,
    x_bin: f32,  // [0, 4)
    y_bin: f32,  // [0, 4)
    o_bin: f32,  // [0, 8)
    weight: f32,
) {
    // Compute integer bin indices and fractional parts
    let x0 = u32(floor(x_bin));
    let y0 = u32(floor(y_bin));
    let o0 = u32(floor(o_bin));
    
    let x1 = min(x0 + 1u, DESC_WIDTH - 1u);
    let y1 = min(y0 + 1u, DESC_WIDTH - 1u);
    let o1 = (o0 + 1u) % DESC_BINS;
    
    let fx = fract(x_bin);
    let fy = fract(y_bin);
    let fo = fract(o_bin);
    
    // Trilinear interpolation weights
    let w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fo) * weight;
    let w001 = (1.0 - fx) * (1.0 - fy) * fo * weight;
    let w010 = (1.0 - fx) * fy * (1.0 - fo) * weight;
    let w011 = (1.0 - fx) * fy * fo * weight;
    let w100 = fx * (1.0 - fy) * (1.0 - fo) * weight;
    let w101 = fx * (1.0 - fy) * fo * weight;
    let w110 = fx * fy * (1.0 - fo) * weight;
    let w111 = fx * fy * fo * weight;
    
    // Accumulate into 8 bins
    (*local_hist)[(y0 * DESC_WIDTH + x0) * DESC_BINS + o0] += w000;
    (*local_hist)[(y0 * DESC_WIDTH + x0) * DESC_BINS + o1] += w001;
    (*local_hist)[(y1 * DESC_WIDTH + x0) * DESC_BINS + o0] += w010;
    (*local_hist)[(y1 * DESC_WIDTH + x0) * DESC_BINS + o1] += w011;
    (*local_hist)[(y0 * DESC_WIDTH + x1) * DESC_BINS + o0] += w100;
    (*local_hist)[(y0 * DESC_WIDTH + x1) * DESC_BINS + o1] += w101;
    (*local_hist)[(y1 * DESC_WIDTH + x1) * DESC_BINS + o0] += w110;
    (*local_hist)[(y1 * DESC_WIDTH + x1) * DESC_BINS + o1] += w111;
}

// ===== Shared Memory for 4-Thread Reduction =====
var<workgroup> shared_descriptor: array<atomic<i32>, 128>; // Final descriptor (scaled by 1000)

@compute @workgroup_size(4, 1, 1)
fn compute_descriptor(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let keypoint_idx = global_id.x / 4u;
    let quadrant_idx = local_id.x; // 0..3 (maps to 2×2 quadrants)
    
    if (keypoint_idx >= num_final) {
        return;
    }
    
    // Initialize shared descriptor
    for (var i = quadrant_idx; i < DESC_SIZE; i += 4u) {
        atomicStore(&shared_descriptor[i], 0);
    }
    workgroupBarrier();
    
    // Load keypoint
    let kp = keypoints_final[keypoint_idx];
    let x = kp.x;
    let y = kp.y;
    let sigma = kp.z;
    let orientation = kp.w;
    
    // Compute octave and scale from sigma
    let octave_idx_f = log2(sigma / 1.6) - log2(f32(desc_meta.scales - 2u));
    let octave_idx = u32(clamp(octave_idx_f, 0.0, f32(desc_meta.octaves - 1u)));
    let scale_factor = pow(2.0, f32(octave_idx));
    
    // Map to octave coordinates
    let octave_x = x / scale_factor;
    let octave_y = y / scale_factor;
    let octave_sigma = sigma / scale_factor;
    
    // Find Gaussian level
    let scale_in_octave_f = log2(octave_sigma / 1.6) * f32(desc_meta.scales - 2u);
    let scale_idx = u32(clamp(scale_in_octave_f, 0.0, f32(desc_meta.scales - 1u)));
    let level_idx = octave_idx * desc_meta.scales + scale_idx;
    
    let offset = level_offsets[level_idx];
    let width = level_widths[level_idx];
    let height = level_heights[level_idx];
    
    // Descriptor window parameters
    let hist_width = DESCRIPTOR_MAGNIFICATION * octave_sigma;
    let radius = i32(ceil(hist_width * sqrt(2.0) * (f32(DESC_WIDTH) + 1.0) * 0.5));
    
    // Gaussian weight
    let desc_sigma = hist_width * f32(DESC_WIDTH) * 0.5;
    let two_desc_sigma_sq = 2.0 * desc_sigma * desc_sigma;
    
    // Rotation matrix (to align with dominant orientation)
    let cos_theta = cos(-orientation);
    let sin_theta = sin(-orientation);
    
    // Each thread maintains local histogram for its quadrant
    var local_hist: array<f32, 128>;
    for (var i = 0u; i < DESC_SIZE; i++) {
        local_hist[i] = 0.0;
    }
    
    // Compute quadrant bounds (2×2 quadrants)
    let qx = quadrant_idx % 2u;
    let qy = quadrant_idx / 2u;
    
    // Sample only pixels in this quadrant's region
    // Divide window into 2×2 regions, each thread processes 1
    let region_start_y = -radius + (i32(qy) * (2 * radius)) / 2;
    let region_end_y = -radius + (i32(qy + 1u) * (2 * radius)) / 2;
    let region_start_x = -radius + (i32(qx) * (2 * radius)) / 2;
    let region_end_x = -radius + (i32(qx + 1u) * (2 * radius)) / 2;
    
    for (var dy = region_start_y; dy < region_end_y; dy++) {
        for (var dx = region_start_x; dx < region_end_x; dx++) {
            // Rotate offset to keypoint's coordinate frame
            let dx_f = f32(dx);
            let dy_f = f32(dy);
            let rot_x = cos_theta * dx_f - sin_theta * dy_f;
            let rot_y = sin_theta * dx_f + cos_theta * dy_f;
            
            let sample_x = i32(round(octave_x + rot_x));
            let sample_y = i32(round(octave_y + rot_y));
            
            if (sample_x < 1 || sample_x >= i32(width) - 1 || 
                sample_y < 1 || sample_y >= i32(height) - 1) {
                continue;
            }
            
            // Compute gradient
            let gx = read_pixel_f16(offset, sample_x + 1, sample_y, width, height)
                   - read_pixel_f16(offset, sample_x - 1, sample_y, width, height);
            let gy = read_pixel_f16(offset, sample_x, sample_y + 1, width, height)
                   - read_pixel_f16(offset, sample_x, sample_y - 1, width, height);
            
            let mag = sqrt(gx * gx + gy * gy);
            var angle = atan2(gy, gx) - orientation; // relative to keypoint orientation
            if (angle < 0.0) {
                angle += 2.0 * PI;
            }
            if (angle >= 2.0 * PI) {
                angle -= 2.0 * PI;
            }
            
            // Gaussian weight based on distance from keypoint
            let dist_sq = rot_x * rot_x + rot_y * rot_y;
            let gaussian_weight = exp(-dist_sq / two_desc_sigma_sq);
            
            // Map to descriptor bins [0, 4) for spatial, [0, 8) for orientation
            let x_bin = (rot_x / hist_width) + 2.0; // offset by 2 to center at [0,4)
            let y_bin = (rot_y / hist_width) + 2.0;
            let o_bin = angle * f32(DESC_BINS) / (2.0 * PI);
            
            // Check if sample falls within descriptor grid
            if (x_bin >= -0.5 && x_bin < 4.5 && y_bin >= -0.5 && y_bin < 4.5) {
                accumulate_trilinear(&local_hist, x_bin, y_bin, o_bin, mag * gaussian_weight);
            }
        }
    }
    
    // Reduction: accumulate all 4 thread histograms into shared memory
    // Scale f32 to i32 by multiplying by 1000
    workgroupBarrier();
    for (var i = 0u; i < DESC_SIZE; i++) {
        let scaled_val = i32(local_hist[i] * 1000.0);
        atomicAdd(&shared_descriptor[i], scaled_val);
    }
    workgroupBarrier();
    
    // Thread 0 normalizes and writes final descriptor
    if (local_id.x == 0u) {
        // L2 normalization (convert back from i32 to f32)
        var norm_sq: f32 = 0.0;
        for (var i = 0u; i < DESC_SIZE; i++) {
            let val_f32 = f32(atomicLoad(&shared_descriptor[i])) / 1000.0;
            norm_sq += val_f32 * val_f32;
        }
        let norm = sqrt(norm_sq);
        
        // Normalize, clip, and renormalize (all working with f32 temp array)
        var normalized: array<f32, DESC_SIZE>;
        if (norm > 1e-6) {
            for (var i = 0u; i < DESC_SIZE; i++) {
                normalized[i] = f32(atomicLoad(&shared_descriptor[i])) / 1000.0 / norm;
            }
        } else {
            for (var i = 0u; i < DESC_SIZE; i++) {
                normalized[i] = 0.0;
            }
        }
        
        // Clip values > 0.2 and renormalize
        var clipped_norm_sq: f32 = 0.0;
        for (var i = 0u; i < DESC_SIZE; i++) {
            normalized[i] = min(normalized[i], 0.2);
            clipped_norm_sq += normalized[i] * normalized[i];
        }
        let clipped_norm = sqrt(clipped_norm_sq);
        
        if (clipped_norm > 1e-6) {
            for (var i = 0u; i < DESC_SIZE; i++) {
                normalized[i] /= clipped_norm;
            }
        }
        
        // Quantize to u8 and pack into u32 (4 bytes per word)
        let desc_base = keypoint_idx * 32u; // 128 bytes = 32 u32 words
        for (var i = 0u; i < 32u; i++) {
            let val0 = u32(clamp(normalized[i * 4u + 0u] * 512.0, 0.0, 255.0));
            let val1 = u32(clamp(normalized[i * 4u + 1u] * 512.0, 0.0, 255.0));
            let val2 = u32(clamp(normalized[i * 4u + 2u] * 512.0, 0.0, 255.0));
            let val3 = u32(clamp(normalized[i * 4u + 3u] * 512.0, 0.0, 255.0));
            
            descriptors[desc_base + i] = (val3 << 24u) | (val2 << 16u) | (val1 << 8u) | val0;
        }
    }
}
