// downsample.wgsl
// Simple 2× downsample: output[x,y] = input[2x, 2y]

// ===== Bind Groups =====
@group(0) @binding(0) var<storage, read> heap_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> heap_out: array<u32>;

struct DownsampleParams {
    offset_in: u32,     // base offset in heap_in
    offset_out: u32,    // base offset in heap_out
    width_in: u32,      // input dimensions
    height_in: u32,
    width_out: u32,     // output dimensions (width_in/2)
    height_out: u32,    // (height_in/2)
}

@group(1) @binding(0) var<uniform> params: DownsampleParams;

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
    
    let old_packed = heap_out[base_offset + word_idx];
    let old_unpacked = unpack2x16float(old_packed);
    
    let new_unpacked = select(
        vec2<f32>(value, old_unpacked.y),
        vec2<f32>(old_unpacked.x, value),
        is_high
    );
    
    heap_out[base_offset + word_idx] = pack2x16float(new_unpacked);
}

@compute @workgroup_size(16, 16, 1)
fn downsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_out = global_id.x;
    let y_out = global_id.y;
    
    if (x_out >= params.width_out || y_out >= params.height_out) {
        return;
    }
    
    let x_in = x_out * 2u;
    let y_in = y_out * 2u;
    
    let value = read_pixel_f16(params.offset_in, x_in, y_in, params.width_in);
    write_pixel_f16(params.offset_out, x_out, y_out, params.width_out, value);
}
