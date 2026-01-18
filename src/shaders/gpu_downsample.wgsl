// gpu_downsample.wgsl
// 2x downsampling for octave pyramid
// Uses textureLoad since R32Float doesn't support filtering

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;  // Unused, kept for compatibility
@group(0) @binding(2) var output_texture: texture_storage_2d<r32float, write>;

struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

@group(0) @binding(3) var<uniform> params: DownsampleParams;

@compute @workgroup_size(16, 16, 1)
fn downsample_2x(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }
    
    // Take pixel from even coordinates
    let src_coord = vec2<i32>(i32(x * 2u), i32(y * 2u));
    let value = textureLoad(input_texture, src_coord, 0).r;
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(value, 0.0, 0.0, 1.0));
}

// Average 2x2 block for smoother downsampling
@compute @workgroup_size(16, 16, 1)
fn downsample_avg(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }
    
    let sx = i32(x * 2u);
    let sy = i32(y * 2u);
    
    let v00 = textureLoad(input_texture, vec2<i32>(sx, sy), 0).r;
    let v10 = textureLoad(input_texture, vec2<i32>(sx + 1, sy), 0).r;
    let v01 = textureLoad(input_texture, vec2<i32>(sx, sy + 1), 0).r;
    let v11 = textureLoad(input_texture, vec2<i32>(sx + 1, sy + 1), 0).r;
    
    let value = (v00 + v10 + v01 + v11) * 0.25;
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(value, 0.0, 0.0, 1.0));
}
