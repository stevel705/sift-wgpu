// gaussian_blur.wgsl
// Separable Gaussian blur: applies 1D blur in X or Y direction
// Uses precomputed kernel weights from GPU buffer

// ===== Bind Groups =====
// @group(0): input/output heap regions
@group(0) @binding(0) var<storage, read> heap_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> heap_out: array<u32>;

// @group(1): blur parameters
struct BlurParams {
    offset_in: u32,    // base offset in heap_in
    offset_out: u32,   // base offset in heap_out
    width: u32,
    height: u32,
    kernel_size: u32,  // number of weights
    direction: u32,    // 0=horizontal, 1=vertical
}

@group(1) @binding(0) var<uniform> params: BlurParams;
@group(1) @binding(1) var<storage, read> kernel_weights: array<f32>; // precomputed Gaussian

// ===== F16 Packing =====
fn read_pixel_f16(base_offset: u32, x: u32, y: u32, width: u32) -> f32 {
    let idx = y * width + x;
    let word_idx = idx >> 1u;
    let is_high = (idx & 1u) != 0u;
    let packed = heap_in[base_offset + word_idx];
    let unpacked = unpack2x16float(packed);
    return select(unpacked.x, unpacked.y, is_high);
}

fn write_pixel_f16(base_offset: u32, x: u32, y: u32, width: u32, value: f32) {
    let idx = y * width + x;
    let word_idx = idx >> 1u;
    let is_high = (idx & 1u) != 0u;
    
    // Read existing word
    let old_packed = heap_out[base_offset + word_idx];
    let old_unpacked = unpack2x16float(old_packed);
    
    // Update appropriate half
    let new_unpacked = select(
        vec2<f32>(value, old_unpacked.y),
        vec2<f32>(old_unpacked.x, value),
        is_high
    );
    
    heap_out[base_offset + word_idx] = pack2x16float(new_unpacked);
}

@compute @workgroup_size(16, 16, 1)
fn gaussian_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let radius = i32(params.kernel_size / 2u);
    var sum: f32 = 0.0;
    
    if (params.direction == 0u) {
        // Horizontal blur
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - radius;
            let sample_x = clamp(i32(x) + offset, 0, i32(params.width) - 1);
            let pixel_val = read_pixel_f16(params.offset_in, u32(sample_x), y, params.width);
            sum += pixel_val * kernel_weights[i];
        }
    } else {
        // Vertical blur
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - radius;
            let sample_y = clamp(i32(y) + offset, 0, i32(params.height) - 1);
            let pixel_val = read_pixel_f16(params.offset_in, x, u32(sample_y), params.width);
            sum += pixel_val * kernel_weights[i];
        }
    }
    
    write_pixel_f16(params.offset_out, x, y, params.width, sum);
}
