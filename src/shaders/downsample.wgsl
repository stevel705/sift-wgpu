// src/shaders/downsample.wgsl

struct Params {
    width: u32,
    height: u32,
    sigma: f32,
    step_x: u32,
    step_y: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var texture_in: texture_2d<f32>;
@group(0) @binding(2) var texture_aux: texture_2d<f32>; // Заглушка для соответствия лэйауту
@group(0) @binding(3) var texture_out: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var samp: sampler; // Не используется

fn mirror_coord(c: i32, max_v: i32) -> i32 {
    if (c < 0) {
        return -c;
    }
    if (c >= max_v) {
        return 2 * max_v - 2 - c;
    }
    return c;
}

@compute @workgroup_size(8, 8, 1) // Размер группы для выходной текстуры
fn main_downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_coord = vec2<i32>(i32(id.x), i32(id.y));
    let out_size = textureDimensions(texture_out);

    // Проверка границ выходной текстуры
    if (out_coord.x >= i32(out_size.x) || out_coord.y >= i32(out_size.y)) {
        return;
    }

    // Предполагается, что вход уже предразмыт (σ≈1). Здесь только point-sample каждый второй пиксель с зеркальными границами.
    let size_in = vec2<i32>(textureDimensions(texture_in));
    let src_coord = out_coord * 2;
    let sx = mirror_coord(src_coord.x, size_in.x);
    let sy = mirror_coord(src_coord.y, size_in.y);
    let v = textureLoad(texture_in, vec2<i32>(sx, sy), 0).r;
    textureStore(texture_out, out_coord, vec4<f32>(v, 0.0, 0.0, 1.0));
}
