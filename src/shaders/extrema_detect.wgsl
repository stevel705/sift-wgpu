// extrema_detect.wgsl
// Detect local extrema (maxima/minima) in scale-space DoG pyramid
// Each extrema must be > or < all 26 neighbors (3x3x3 cube)

// ===== Bind Groups =====
// @group(0): metadata (read-only)
struct PyramidMetadata {
    octaves: u32,          // number of octaves
    dog_scales: u32,       // DoG layers per octave (scales-1 from Gaussian)
    usable_scales: u32,    // layers where we can detect extrema (dog_scales-2)
    base_width: u32,
    base_height: u32,
    sigma: f32,
    contrast_threshold: f32,
    edge_threshold: f32,
}

@group(0) @binding(0) var<storage, read> pyramid_meta: PyramidMetadata;
@group(0) @binding(1) var<storage, read> level_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> level_widths: array<u32>;
@group(0) @binding(3) var<storage, read> level_heights: array<u32>;

// @group(1): heap (read-only)
@group(1) @binding(0) var<storage, read> heap: array<u32>;

// @group(2): output buffers
@group(2) @binding(0) var<storage, read_write> extrema_counter: atomic<u32>;
@group(2) @binding(1) var<storage, read_write> keypoints_staging: array<vec4<f32>>;
const MAX_STAGING_KEYPOINTS: u32 = 32768u;

// ===== Push Constants / Specialization =====
// For dispatch: each invocation processes (octave_idx, dog_layer_idx)
// Dispatch as: (ceil(width/16), ceil(height/16), octaves * dog_scales)

// ===== F16 Unpacking =====
fn read_pixel_f16(base_offset: u32, x: u32, y: u32, width: u32) -> f32 {
    let idx = y * width + x;
    let word_idx = idx >> 1u;
    let is_high = (idx & 1u) != 0u;
    let packed = heap[base_offset + word_idx];
    let unpacked = unpack2x16float(packed);
    return select(unpacked.x, unpacked.y, is_high);
}

// ===== Tile-Based Detection =====
// Workgroup: 16×16 threads process 16×16 center pixels
// Shared memory: 18×18 tile for 1-pixel halo around center 16×16
// Load 3 DoG layers (prev, current, next) into shared memory
var<workgroup> tile_prev: array<f32, 324>; // 18×18
var<workgroup> tile_curr: array<f32, 324>;
var<workgroup> tile_next: array<f32, 324>;
var<workgroup> tile_count: atomic<u32>;

@compute @workgroup_size(16, 16, 1)
fn detect_extrema(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // workgroup_id.z encodes (octave, usable_layer)
    let octave_idx = workgroup_id.z / pyramid_meta.usable_scales;
    let usable_layer = workgroup_id.z % pyramid_meta.usable_scales;
    
    if (octave_idx >= pyramid_meta.octaves || usable_layer >= pyramid_meta.usable_scales) {
        return;
    }
    
    // Only first thread initializes tile counter
    if (local_id.x == 0u && local_id.y == 0u) {
        atomicStore(&tile_count, 0u);
    }
    workgroupBarrier();
    
    // Get dimensions for this octave (all DoG layers in an octave have same dimensions)
    let dog_scales = pyramid_meta.dog_scales;
    let level_w = level_widths[octave_idx * dog_scales];
    let level_h = level_heights[octave_idx * dog_scales];
    
    // Tile origin in output space (center 16×16 region)
    let tile_origin_x = workgroup_id.x * 16u;
    let tile_origin_y = workgroup_id.y * 16u;
    
    // Load 18×18 halo for 3 DoG layers
    // Each thread loads ~2 pixels (324 / 256 = 1.27)
    let thread_idx = local_id.y * 16u + local_id.x;
    for (var i = thread_idx; i < 324u; i += 256u) {
        let tile_y = i / 18u;
        let tile_x = i % 18u;
        
        // Map to global coordinates with halo (-1 to +16)
        let global_x = i32(tile_origin_x) + i32(tile_x) - 1;
        let global_y = i32(tile_origin_y) + i32(tile_y) - 1;
        
        // Clamp to valid range
        let clamped_x = clamp(global_x, 0, i32(level_w) - 1);
        let clamped_y = clamp(global_y, 0, i32(level_h) - 1);
        
        // Read from DoG pyramid (3 adjacent layers: usable_layer, usable_layer+1, usable_layer+2)
        // usable_layer is the MIDDLE layer we're checking for extrema
        let dog_prev_idx = octave_idx * dog_scales + usable_layer;
        let dog_curr_idx = dog_prev_idx + 1u;
        let dog_next_idx = dog_curr_idx + 1u;
        
        let offset_prev = level_offsets[dog_prev_idx];
        let offset_curr = level_offsets[dog_curr_idx];
        let offset_next = level_offsets[dog_next_idx];
        
        tile_prev[i] = read_pixel_f16(offset_prev, u32(clamped_x), u32(clamped_y), level_w);
        tile_curr[i] = read_pixel_f16(offset_curr, u32(clamped_x), u32(clamped_y), level_w);
        tile_next[i] = read_pixel_f16(offset_next, u32(clamped_x), u32(clamped_y), level_w);
    }
    workgroupBarrier();
    
    // Each thread checks 1 center pixel (16×16)
    let center_x = global_id.x;
    let center_y = global_id.y;
    
    if (center_x >= level_w || center_y >= level_h) {
        return;
    }
    
    // Map to tile coordinates (+1 offset for halo)
    let tile_x = local_id.x + 1u;
    let tile_y = local_id.y + 1u;
    let tile_idx = tile_y * 18u + tile_x;
    
    let center_val = tile_curr[tile_idx];
    
    // Quick contrast check
    if (abs(center_val) < pyramid_meta.contrast_threshold) {
        return;
    }
    
    // Check if extrema (max or min among 26 neighbors)
    var is_max = true;
    var is_min = true;
    
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                
                let neighbor_tile_x = i32(tile_x) + dx;
                let neighbor_tile_y = i32(tile_y) + dy;
                let neighbor_tile_idx = u32(neighbor_tile_y * 18 + neighbor_tile_x);
                
                var neighbor_val: f32;
                if (dz == -1) {
                    neighbor_val = tile_prev[neighbor_tile_idx];
                } else if (dz == 0) {
                    neighbor_val = tile_curr[neighbor_tile_idx];
                } else {
                    neighbor_val = tile_next[neighbor_tile_idx];
                }
                
                if (center_val <= neighbor_val) {
                    is_max = false;
                }
                if (center_val >= neighbor_val) {
                    is_min = false;
                }
            }
        }
    }
    
    if (!is_max && !is_min) {
        return;
    }
    
    // Edge suppression via Hessian ratio
    // Compute Hxx, Hyy, Hxy from current layer
    let idx_00 = tile_idx;
    let idx_m10 = tile_idx - 1u;
    let idx_p10 = tile_idx + 1u;
    let idx_0m1 = tile_idx - 18u;
    let idx_0p1 = tile_idx + 18u;
    
    let val_00 = tile_curr[idx_00];
    let val_m10 = tile_curr[idx_m10];
    let val_p10 = tile_curr[idx_p10];
    let val_0m1 = tile_curr[idx_0m1];
    let val_0p1 = tile_curr[idx_0p1];
    
    let Dxx = val_p10 + val_m10 - 2.0 * val_00;
    let Dyy = val_0p1 + val_0m1 - 2.0 * val_00;
    let Dxy = (tile_curr[idx_0p1 + 1u] - tile_curr[idx_0p1 - 1u] 
              - tile_curr[idx_0m1 + 1u] + tile_curr[idx_0m1 - 1u]) * 0.25;
    
    let trace = Dxx + Dyy;
    let det = Dxx * Dyy - Dxy * Dxy;
    
    if (det <= 0.0) {
        return; // not a well-defined extremum
    }
    
    let ratio = (trace * trace) / det;
    let r_threshold = (pyramid_meta.edge_threshold + 1.0) * (pyramid_meta.edge_threshold + 1.0) / pyramid_meta.edge_threshold;
    
    if (ratio > r_threshold) {
        return; // edge-like, reject
    }
    
    // Passed all tests, add to staging buffer
    let slot = atomicAdd(&extrema_counter, 1u);
    if (slot < MAX_STAGING_KEYPOINTS) {
        // Store (x, y, octave, scale_within_octave)
        // usable_layer + 1 is the middle layer index we're detecting extrema at
        let scale_in_octave = f32(usable_layer) + 1.0;
        keypoints_staging[slot] = vec4<f32>(
            f32(center_x),
            f32(center_y),
            f32(octave_idx),
            scale_in_octave
        );
    }
}
