// dog.wgsl - Compute Difference of Gaussians

struct DogParams {
    gaussian_level_1: u32,
    gaussian_level_2: u32,
    dog_level: u32,
    _padding: u32,
}

struct PyramidHeader {
    num_octaves: u32,
    num_scales_per_octave: u32,
    num_dog_per_octave: u32,
    base_width: u32,
    base_height: u32,
    total_gaussian_levels: u32,
    total_dog_levels: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> header: PyramidHeader;
@group(0) @binding(1) var<uniform> params: DogParams;
@group(0) @binding(2) var<storage, read> level_offsets_words: array<u32>;
@group(0) @binding(3) var<storage, read> level_widths: array<u32>;
@group(0) @binding(4) var<storage, read> level_heights: array<u32>;
@group(0) @binding(5) var<storage, read> heap_in: array<u32>;
@group(1) @binding(0) var<storage, read_write> heap_out: array<u32>;

fn read_pixel_f16(level: u32, x: u32, y: u32) -> f32 {
    let w = level_widths[level];
    let word_off = level_offsets_words[level];
    let px = y * w + x;
    let word = heap_in[word_off + (px >> 1u)];
    let lane = px & 1u;
    let v = unpack2x16float(word);
    return select(v.x, v.y, lane == 1u);
}

fn write_pixel_f16(level: u32, x: u32, y: u32, value: f32) {
    let w = level_widths[level];
    let word_off = level_offsets_words[level];
    let px = y * w + x;
    let word_idx = word_off + (px >> 1u);
    let lane = px & 1u;
    
    // Read-modify-write for the correct lane
    let old_word = heap_out[word_idx];
    let old_values = unpack2x16float(old_word);
    let new_values = select(
        vec2<f32>(value, old_values.y),
        vec2<f32>(old_values.x, value),
        lane == 1u
    );
    heap_out[word_idx] = pack2x16float(new_values);
}

@compute @workgroup_size(16, 16)
fn compute_dog(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    let width = level_widths[params.dog_level];
    let height = level_heights[params.dog_level];
    
    if (x >= width || y >= height) {
        return;
    }
    
    let g1 = read_pixel_f16(params.gaussian_level_1, x, y);
    let g2 = read_pixel_f16(params.gaussian_level_2, x, y);
    let dog = g2 - g1;
    
    write_pixel_f16(params.dog_level, x, y, dog);
}
