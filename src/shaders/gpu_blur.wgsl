// gpu_blur.wgsl
// Fast separable Gaussian blur using texture load (R32Float not filterable)
// Optimized for building scale-space pyramid on GPU

// Input texture (use textureLoad since R32Float doesn't support filtering on all hardware)
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;  // Unused but kept for bind group compatibility

// Output as storage texture
@group(0) @binding(2) var output_texture: texture_storage_2d<r32float, write>;

struct BlurParams {
    width: u32,
    height: u32,
    kernel_radius: u32,  // half-size of kernel
    direction: u32,      // 0 = horizontal, 1 = vertical
}

@group(0) @binding(3) var<uniform> params: BlurParams;
@group(0) @binding(4) var<storage, read> kernel: array<f32>; // kernel weights (full, not symmetric)

fn load_pixel(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return textureLoad(input_texture, vec2<i32>(cx, cy), 0).r;
}

@compute @workgroup_size(16, 16, 1)
fn blur_horizontal(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let radius = i32(params.kernel_radius);
    var sum: f32 = 0.0;
    
    for (var i = -radius; i <= radius; i++) {
        sum += load_pixel(x + i, y) * kernel[i + radius];
    }
    
    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(sum, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn blur_vertical(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let radius = i32(params.kernel_radius);
    var sum: f32 = 0.0;
    
    for (var i = -radius; i <= radius; i++) {
        sum += load_pixel(x, y + i) * kernel[i + radius];
    }
    
    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(sum, 0.0, 0.0, 1.0));
}

// Simple blur for any direction
@compute @workgroup_size(16, 16, 1)
fn blur_simple(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let radius = i32(params.kernel_radius);
    var sum: f32 = 0.0;
    
    if (params.direction == 0u) {
        // Horizontal
        for (var i = -radius; i <= radius; i++) {
            sum += load_pixel(x + i, y) * kernel[i + radius];
        }
    } else {
        // Vertical
        for (var i = -radius; i <= radius; i++) {
            sum += load_pixel(x, y + i) * kernel[i + radius];
        }
    }
    
    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(sum, 0.0, 0.0, 1.0));
}
