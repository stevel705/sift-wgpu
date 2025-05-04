#[derive(Debug, Clone, PartialEq)]
pub struct KeyPoint {
    pub x: f32,        // x-координата (в масштабе исходного изображения)
    pub y: f32,        // y-координата (в масштабе исходного изображения)
    pub size: f32,     // Характерный размер (связан с sigma в момент обнаружения)
    pub angle: f32,    // Ориентация в радианах (-PI до PI)
    pub response: f32, // Отклик точки (значение DoG в экстремуме)
    pub octave: i32,   // Индекс октавы, в которой точка была обнаружена
    pub layer: i32,    // Индекс слоя (масштаба) внутри октавы
}

impl KeyPoint {
    // Конструктор для удобства, если понадобится
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f32,
        y: f32,
        size: f32,
        angle: f32,
        response: f32,
        octave: i32,
        layer: i32,
    ) -> Self {
        KeyPoint {
            x,
            y,
            size,
            angle,
            response,
            octave,
            layer,
        }
    }
}
