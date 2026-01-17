// orientation.wgsl
// Compute dominant orientations for each keypoint
// Each keypoint spawns 36 threads (1 per histogram bin)
// Uses local histograms + reduction to avoid atomics

// ===== Bind Groups =====
struct OrientationMeta {
    octaves: u32,
    scales: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> orient_meta: OrientationMeta;
@group(0) @binding(1) var<storage, read> level_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> level_widths: array<u32>;
@group(0) @binding(3) var<storage, read> level_heights: array<u32>;

@group(1) @binding(0) var<storage, read> heap: array<u32>;

@group(2) @binding(0) var<storage, read> keypoints_staging: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> num_staging: u32;

@group(3) @binding(0) var<storage, read_write> orientation_counter: atomic<u32>;
@group(3) @binding(1) var<storage, read_write> keypoints_final: array<vec4<f32>>;
const MAX_FINAL_KEYPOINTS: u32 = 65536u;

// ===== Constants =====
const PI: f32 = 3.14159265359;
const HISTOGRAM_BINS: u32 = 36u;
const ORIENTATION_RADIUS_FACTOR: f32 = 3.0; // 3×sigma window
const PEAK_THRESHOLD: f32 = 0.8; // 80% of max peak

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

// ===== Shared Memory for Reduction =====
var<workgroup> local_histograms: array<f32, 36>; // 36 bins, each thread owns 1 bin

@compute @workgroup_size(36, 1, 1)
fn compute_orientation(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let keypoint_idx = global_id.x / HISTOGRAM_BINS;
    let bin_idx = local_id.x; // 0..35
    
    if (keypoint_idx >= num_staging) {
        return;
    }
    
    // Initialize local histogram bin
    local_histograms[bin_idx] = 0.0;
    workgroupBarrier();
    
    // Load keypoint
    let kp = keypoints_staging[keypoint_idx];
    let x = kp.x;
    let y = kp.y;
    let octave_idx = u32(kp.z);
    let scale_in_octave = kp.w;
    
    // Compute Gaussian level index (scale corresponds to G[s+1])
    let gaussian_scale_idx = u32(scale_in_octave); // 1, 2, 3...
    let level_idx = octave_idx * orient_meta.scales + gaussian_scale_idx;
    
    let offset = level_offsets[level_idx];
    let width = level_widths[level_idx];
    let height = level_heights[level_idx];
    
    // Compute sigma at this scale
    let sigma = 1.6 * pow(2.0, f32(octave_idx) + scale_in_octave / f32(orient_meta.scales - 2u));
    
    // Window radius = 3×sigma
    let radius = i32(ceil(ORIENTATION_RADIUS_FACTOR * sigma));
    
    // Gaussian weight denominator
    let sigma_weight = 1.5 * sigma;
    let two_sigma_sq = 2.0 * sigma_weight * sigma_weight;
    
    // Each thread accumulates contributions for its own bin
    var bin_sum: f32 = 0.0;
    
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let px = i32(round(x)) + dx;
            let py = i32(round(y)) + dy;
            
            if (px < 1 || px >= i32(width) - 1 || py < 1 || py >= i32(height) - 1) {
                continue;
            }
            
            // Compute gradient magnitude and direction
            let gx = read_pixel_f16(offset, px + 1, py, width, height) 
                   - read_pixel_f16(offset, px - 1, py, width, height);
            let gy = read_pixel_f16(offset, px, py + 1, width, height) 
                   - read_pixel_f16(offset, px, py - 1, width, height);
            
            let mag = sqrt(gx * gx + gy * gy);
            var angle = atan2(gy, gx); // [-π, π]
            if (angle < 0.0) {
                angle += 2.0 * PI;
            }
            
            // Convert to bin index [0, 36)
            let angle_bin_f = angle * f32(HISTOGRAM_BINS) / (2.0 * PI);
            let angle_bin = u32(angle_bin_f) % HISTOGRAM_BINS;
            
            // Gaussian weight
            let dist_sq = f32(dx * dx + dy * dy);
            let weight = exp(-dist_sq / two_sigma_sq);
            
            // Accumulate only if this thread owns the bin
            if (angle_bin == bin_idx) {
                bin_sum += mag * weight;
            }
        }
    }
    
    // Write thread's contribution to shared memory
    local_histograms[bin_idx] = bin_sum;
    workgroupBarrier();
    
    // Thread 0 performs smoothing and peak detection
    if (local_id.x == 0u) {
        // Smooth histogram with [1, 2, 1] kernel (2 passes)
        var smoothed: array<f32, 36>;
        
        // Pass 1
        for (var i = 0u; i < HISTOGRAM_BINS; i++) {
            let prev = (i + HISTOGRAM_BINS - 1u) % HISTOGRAM_BINS;
            let next = (i + 1u) % HISTOGRAM_BINS;
            smoothed[i] = (local_histograms[prev] + 2.0 * local_histograms[i] + local_histograms[next]) * 0.25;
        }
        
        // Pass 2
        for (var i = 0u; i < HISTOGRAM_BINS; i++) {
            let prev = (i + HISTOGRAM_BINS - 1u) % HISTOGRAM_BINS;
            let next = (i + 1u) % HISTOGRAM_BINS;
            local_histograms[i] = (smoothed[prev] + 2.0 * smoothed[i] + smoothed[next]) * 0.25;
        }
        
        // Find max peak
        var max_val: f32 = 0.0;
        for (var i = 0u; i < HISTOGRAM_BINS; i++) {
            max_val = max(max_val, local_histograms[i]);
        }
        
        let threshold = PEAK_THRESHOLD * max_val;
        
        // Find all peaks above threshold
        for (var i = 0u; i < HISTOGRAM_BINS; i++) {
            let val = local_histograms[i];
            let prev = local_histograms[(i + HISTOGRAM_BINS - 1u) % HISTOGRAM_BINS];
            let next = local_histograms[(i + 1u) % HISTOGRAM_BINS];
            
            if (val >= threshold && val >= prev && val >= next) {
                // Parabolic interpolation for sub-bin precision
                let bin_center = f32(i);
                let denom = 2.0 * val - prev - next;
                var refined_bin = bin_center;
                if (abs(denom) > 1e-6) {
                    let offset = 0.5 * (prev - next) / denom;
                    refined_bin = bin_center + offset;
                }
                
                // Convert bin to angle
                var orientation = refined_bin * 2.0 * PI / f32(HISTOGRAM_BINS);
                if (orientation < 0.0) {
                    orientation += 2.0 * PI;
                }
                if (orientation >= 2.0 * PI) {
                    orientation -= 2.0 * PI;
                }
                
                // Write final keypoint with orientation
                let slot = atomicAdd(&orientation_counter, 1u);
                if (slot < MAX_FINAL_KEYPOINTS) {
                    // Map back to original image coordinates (scale by 2^octave)
                    let scale_factor = pow(2.0, f32(octave_idx));
                    let original_x = x * scale_factor;
                    let original_y = y * scale_factor;
                    let original_sigma = sigma * scale_factor;
                    
                    // Store (x, y, sigma, orientation)
                    keypoints_final[slot] = vec4<f32>(
                        original_x,
                        original_y,
                        original_sigma,
                        orientation
                    );
                }
            }
        }
    }
}
