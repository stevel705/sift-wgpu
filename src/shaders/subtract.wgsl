// subtract.wgsl

struct Params { // Может быть не нужен, если размеры одинаковые
    width: u32,
    height: u32,
    // ...
};

// @group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var texture_a: texture_2d<f32>; // Первое изображение
@group(0) @binding(2) var texture_b: texture_2d<f32>; // Второе изображение
@group(0) @binding(3) var texture_out: texture_storage_2d<rgba8unorm, write>; // Выход (DoG)

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

    // Вычитаем (например, только красный канал, т.к. работаем с grayscale)
    // Результат DoG может быть отрицательным. Мы должны его как-то представить
    // в формате rgba8unorm. Варианты:
    // 1. Сдвиг и масштабирование: `0.5 + diff * 0.5` -> [0, 1]
    // 2. Только абсолютное значение: `abs(diff)` -> [0, ?] -> нормализовать
    // Используем вариант 1 для визуализации, хотя для вычислений лучше float формат.
    let diff = val_a.r - val_b.r;
    let dog_value = clamp(0.5 + diff * 0.5, 0.0, 1.0); // Сдвигаем и ограничиваем [0, 1]

    // Записываем одинаковое значение во все каналы (кроме альфы)
    textureStore(texture_out, coord, vec4<f32>(dog_value, dog_value, dog_value, 1.0));
}