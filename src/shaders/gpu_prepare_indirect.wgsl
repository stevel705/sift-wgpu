// gpu_prepare_indirect.wgsl
// Prepares indirect dispatch buffer from counter value
// This avoids CPU-GPU sync for reading keypoint counts

struct IndirectDispatch {
    x: u32,
    y: u32,
    z: u32,
}

@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> indirect: IndirectDispatch;

@compute @workgroup_size(1, 1, 1)
fn prepare_orientation_indirect() {
    // One workgroup per keypoint for orientation
    let count = min(atomicLoad(&counter), 32768u);
    indirect.x = count;
    indirect.y = 1u;
    indirect.z = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn prepare_descriptor_indirect() {
    // One workgroup per oriented keypoint for descriptors
    let count = min(atomicLoad(&counter), 65536u);
    indirect.x = count;
    indirect.y = 1u;
    indirect.z = 1u;
}
