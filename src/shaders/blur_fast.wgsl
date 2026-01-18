// blur_fast.wgsl
// Fast separable Gaussian blur using shared memory
// Based on VulkanSift optimizations

// Workgroup size: 64 threads per row, process 2 rows at a time
const WG_SIZE_X: u32 = 64u;
const WG_SIZE_Y: u32 = 2u;
const HALO: u32 = 8u; // Support up to kernel radius of 8

// Shared memory for tile + halo
var<workgroup> shared_data: array<f32, (WG_SIZE_X + 2u * HALO) * WG_SIZE_Y>;

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

struct BlurParams {
    width: u32,
    height: u32,
    direction: u32,    // 0=horizontal, 1=vertical
    kernel_radius: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(1) @binding(0) var<uniform> params: BlurParams;
@group(1) @binding(1) var<storage, read> kernel: array<f32>;

fn load_pixel(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return input_data[u32(cy) * params.width + u32(cx)];
}

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1)
fn blur_horizontal(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    let lx = local_id.x;
    let ly = local_id.y;
    
    let shared_width = WG_SIZE_X + 2u * HALO;
    let shared_idx = ly * shared_width + lx + HALO;
    
    // Load center pixel
    if (y < params.height) {
        shared_data[shared_idx] = load_pixel(i32(x), i32(y));
        
        // Load left halo
        if (lx < HALO) {
            let halo_x = i32(wg_id.x * WG_SIZE_X) - i32(HALO) + i32(lx);
            shared_data[ly * shared_width + lx] = load_pixel(halo_x, i32(y));
        }
        
        // Load right halo
        if (lx >= WG_SIZE_X - HALO) {
            let offset = lx - (WG_SIZE_X - HALO);
            let halo_x = i32(wg_id.x * WG_SIZE_X + WG_SIZE_X) + i32(offset);
            shared_data[ly * shared_width + WG_SIZE_X + HALO + offset] = load_pixel(halo_x, i32(y));
        }
    }
    
    workgroupBarrier();
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // Compute blur using symmetric kernel
    var sum = shared_data[shared_idx] * kernel[0];
    for (var i = 1u; i <= params.kernel_radius; i++) {
        let left_idx = shared_idx - i;
        let right_idx = shared_idx + i;
        sum += (shared_data[left_idx] + shared_data[right_idx]) * kernel[i];
    }
    
    output_data[y * params.width + x] = sum;
}

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1)
fn blur_vertical(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // For vertical pass, we can't efficiently use shared memory across rows
    // So we just do direct reads with symmetric kernel optimization
    var sum = load_pixel(i32(x), i32(y)) * kernel[0];
    for (var i = 1u; i <= params.kernel_radius; i++) {
        let top = load_pixel(i32(x), i32(y) - i32(i));
        let bottom = load_pixel(i32(x), i32(y) + i32(i));
        sum += (top + bottom) * kernel[i];
    }
    
    output_data[y * params.width + x] = sum;
}
