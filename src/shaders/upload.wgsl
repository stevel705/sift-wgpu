// upload.wgsl - Upload u8 grayscale image and optionally apply initial blur

struct UploadParams {
    width: u32,
    height: u32,
    target_offset_words: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: UploadParams;
@group(0) @binding(1) var<storage, read> input_u8: array<u32>;  // Packed grayscale (4 pixels per u32)
@group(0) @binding(2) var<storage, read_write> heap: array<u32>;

// Each thread processes 2 pixels = 1 u32 word (no race condition)
@compute @workgroup_size(16, 16)
fn upload_grayscale(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x_pair = global_id.x;  // Process pairs of pixels
    let y = global_id.y;
    
    let x0 = x_pair * 2u;
    let x1 = x0 + 1u;
    
    // Bounds check
    if (y >= params.height) {
        return;
    }
    
    // Handle odd width
    if (x0 >= params.width) {
        return;
    }
    
    // Read 2 pixels from input (4 pixels packed per u32)
    let px0_idx = y * params.width + x0;
    
    let byte0 = (input_u8[px0_idx / 4u] >> ((px0_idx % 4u) * 8u)) & 0xFFu;
    var byte1 = byte0;  // Default for odd width
    
    if (x1 < params.width) {
        let px1_idx = px0_idx + 1u;
        byte1 = (input_u8[px1_idx / 4u] >> ((px1_idx % 4u) * 8u)) & 0xFFu;
    }
    
    let f0 = f32(byte0) / 255.0;
    let f1 = f32(byte1) / 255.0;
    
    // Pack 2 f16 into 1 u32
    let packed = pack2x16float(vec2<f32>(f0, f1));
    
    // Write to heap (one thread = one u32, no race)
    let out_word_idx = params.target_offset_words + (px0_idx >> 1u);
    heap[out_word_idx] = packed;
}
