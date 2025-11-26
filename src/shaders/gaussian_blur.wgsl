// gaussian_blur.wgsl

struct Params {
    width: u32,
    height: u32,
    sigma: f32,
    step_x: u32,
    step_y: u32,
    _padding1: u32,
    _padding2: u32,
}; // Структура должна совпадать с host-стороной ComputeParams.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var texture_in: texture_2d<f32>;
// binding(2) оставляем как заглушку, чтобы совпасть с общим BindGroupLayout
@group(0) @binding(2) var texture_aux: texture_2d<f32>;
@group(0) @binding(3) var texture_out: texture_storage_2d<r32float, write>; // Float формат
@group(0) @binding(4) var samp: sampler; // Не используется, но оставлен для совместимости

fn mirror_coord(c: i32, max_v: i32) -> i32 {
    if (c < 0) {
        return -c;
    }
    if (c >= max_v) {
        return 2 * max_v - 2 - c;
    }
    return c;
}

// Динамический радиус ~ ceil(4*sigma)
fn kernel_radius(sigma: f32) -> i32 {
    let r = i32(ceil(4.0 * sigma));
    return max(r, 1);
}

@compute @workgroup_size(8, 8, 1) // Размер рабочей группы (можно настроить)
fn main_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_coord = vec2<i32>(i32(id.x), i32(id.y));

    // Проверка выхода за границы выходной текстуры
    if (out_coord.x >= i32(params.width) || out_coord.y >= i32(params.height)) {
        return;
    }

    if (params.sigma <= 0.0) {
        let mx = mirror_coord(out_coord.x, i32(params.width));
        let my = mirror_coord(out_coord.y, i32(params.height));
        let texel = textureLoad(texture_in, vec2<i32>(mx, my), 0);
        textureStore(texture_out, out_coord, vec4<f32>(texel.r, 0.0, 0.0, 1.0));
        return;
    }

    let radius = kernel_radius(params.sigma);
    let sigma2 = params.sigma * params.sigma;

    var accumulated: f32 = 0.0;
    var total_weight: f32 = 0.0;

    // Цикл по ядру свертки
    for (var i: i32 = -radius; i <= radius; i = i + 1) {
        // Координаты для чтения из входной текстуры
        let read_coord = out_coord + vec2<i32>(i * i32(params.step_x), i * i32(params.step_y));

        // Чтение без сэмплера, используя зеркальное отражение на границах
        let mx = mirror_coord(read_coord.x, i32(params.width));
        let my = mirror_coord(read_coord.y, i32(params.height));
        let texel = textureLoad(texture_in, vec2<i32>(mx, my), 0);

        // Гауссов вес
        let offset = f32(i);
        let weight: f32 = exp(-0.5 * (offset * offset) / sigma2);

        accumulated = accumulated + texel.r * weight;
        total_weight = total_weight + weight;
    }

    if (total_weight > 0.0) {
        accumulated = accumulated / total_weight;
    }

    textureStore(texture_out, out_coord, vec4<f32>(accumulated, 0.0, 0.0, 1.0));
}
