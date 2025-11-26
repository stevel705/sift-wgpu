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
@group(0) @binding(3) var texture_out: texture_storage_2d<rgba32float, write>; // Float формат
@group(0) @binding(4) var samp: sampler; // Не используется, но оставлен для совместимости

const KERNEL_RADIUS: i32 = 7; // Определяем радиус ядра (можно вычислить из sigma)
const KERNEL_SIZE: u32 = 2u * u32(KERNEL_RADIUS) + 1u;

// TODO: Предвычислить или вычислить веса Гаусса здесь на основе params.sigma

@compute @workgroup_size(8, 8, 1) // Размер рабочей группы (можно настроить)
fn main_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_coord = vec2<i32>(i32(id.x), i32(id.y));

    // Проверка выхода за границы выходной текстуры
    if (out_coord.x >= i32(params.width) || out_coord.y >= i32(params.height)) {
        return;
    }

    var accumulated_color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var total_weight: f32 = 0.0;

    // Цикл по ядру свертки
    for (var i: i32 = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i = i + 1) {
        // Координаты для чтения из входной текстуры
        let read_coord = out_coord + vec2<i32>(i * i32(params.step_x), i * i32(params.step_y));

        // Чтение без сэмплера, используя clamp в координатах
        let clamped = clamp(
            read_coord,
            vec2<i32>(0, 0),
            vec2<i32>(i32(params.width) - 1, i32(params.height) - 1),
        );
        let texel = textureLoad(texture_in, clamped, 0);

        // Гауссов вес
        let offset = f32(i);
        let sigma2 = params.sigma * params.sigma;
        let weight: f32 = exp(-0.5 * (offset * offset) / sigma2);

        accumulated_color = accumulated_color + texel * weight;
        total_weight = total_weight + weight;
    }

    if (total_weight > 0.0) {
        accumulated_color = accumulated_color / total_weight;
    }

    // Запись результата в выходную текстуру (формат rgba8unorm)
    // WGSL ожидает значение в диапазоне [0.0, 1.0] для unorm форматов
    textureStore(texture_out, out_coord, accumulated_color);
}
