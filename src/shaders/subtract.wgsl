// subtract.wgsl

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
@group(0) @binding(1) var texture_a: texture_2d<f32>; // Первое изображение
@group(0) @binding(2) var texture_b: texture_2d<f32>; // Второе изображение
@group(0) @binding(3) var texture_out: texture_storage_2d<r32float, write>; // Выход (DoG)
@group(0) @binding(4) var samp: sampler; // Заглушка для соответствия лэйауту

@compute @workgroup_size(8, 8, 1)
fn main_subtract(@builtin(global_invocation_id) id: vec3<u32>) {
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    // Проверка границ (используем размеры из textureDimensions, если params нет)
    let size = textureDimensions(texture_a);
     if (coord.x >= i32(size.x) || coord.y >= i32(size.y)) {
        return;
    }

    // Читаем значения из обеих текстур
    // textureLoad требует целочисленных координат и уровня мипмапа
    let val_a: vec4<f32> = textureLoad(texture_a, coord, 0); // 0 - mip level
    let val_b: vec4<f32> = textureLoad(texture_b, coord, 0);

    // Вычитаем (используем float формат, поэтому без смещения/клиппинга)
    let diff = val_a.r - val_b.r;

    // Записываем одинаковое значение во все каналы
    textureStore(texture_out, coord, vec4<f32>(diff, diff, diff, 1.0));
}
