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
@group(0) @binding(3) var texture_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var samp: sampler; // Не используется

@compute @workgroup_size(8, 8, 1) // Размер группы для выходной текстуры
fn main_downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_coord = vec2<i32>(i32(id.x), i32(id.y));
    let out_size = textureDimensions(texture_out);

    // Проверка границ выходной текстуры
    if (out_coord.x >= i32(out_size.x) || out_coord.y >= i32(out_size.y)) {
        return;
    }

    // --- Вариант 1: Простое усреднение 4 пикселей (без сэмплера) ---
    // Координаты 4 пикселей во входной текстуре
    let in_coord_tl = out_coord * 2; // Top-left
    // Читаем 4 пикселя
    let p00 = textureLoad(texture_in, in_coord_tl + vec2<i32>(0, 0), 0);
    let p10 = textureLoad(texture_in, in_coord_tl + vec2<i32>(1, 0), 0);
    let p01 = textureLoad(texture_in, in_coord_tl + vec2<i32>(0, 1), 0);
    let p11 = textureLoad(texture_in, in_coord_tl + vec2<i32>(1, 1), 0);
    // Усредняем
    let avg_color = (p00 + p10 + p01 + p11) * 0.25;

    // --- Вариант 2: Использование линейного сэмплера (проще) ---
    // Координаты центра соответствующего блока 2x2 во входной текстуре
    // let in_coord_center = (vec2<f32>(out_coord) + vec2<f32>(0.5, 0.5)) * 2.0; // - ? Нет, просто +0.5
    // let in_coord_center = vec2<f32>(out_coord) * 2.0 + vec2<f32>(0.5, 0.5); // Центр пикселя (0,0) -> (0.5, 0.5), центр блока -> (1.0, 1.0)
    // Преобразуем в UV координаты для сэмплера
    // let in_size = textureDimensions(texture_in);
    // let in_uv = in_coord_center / vec2<f32>(in_size);
    // let avg_color = textureSampleLevel(texture_in, samp, in_uv, 0.0);

    // Записываем результат
    textureStore(texture_out, out_coord, avg_color);
}
