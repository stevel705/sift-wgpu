// gpu_dog.wgsl
// Difference of Gaussians computation on GPU
// Subtracts adjacent Gaussian scales to create DoG pyramid

@group(0) @binding(0) var scale_a: texture_2d<f32>;  // scale s+1 (higher blur)
@group(0) @binding(1) var scale_b: texture_2d<f32>;  // scale s (lower blur)
@group(0) @binding(2) var dog_output: texture_storage_2d<r32float, write>;

struct DogParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: DogParams;

@compute @workgroup_size(16, 16, 1)
fn compute_dog(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let coord = vec2<i32>(i32(x), i32(y));
    let val_a = textureLoad(scale_a, coord, 0).r;
    let val_b = textureLoad(scale_b, coord, 0).r;
    
    // DoG = Gaussian(s+1) - Gaussian(s)
    let dog = val_a - val_b;
    
    textureStore(dog_output, coord, vec4<f32>(dog, 0.0, 0.0, 1.0));
}