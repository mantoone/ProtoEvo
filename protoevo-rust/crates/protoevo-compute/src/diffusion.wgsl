// Diffusion Compute Shader (WGSL)
// Ported from optimized CUDA kernel

struct Params {
    width: u32,
    height: u32,
    decay: f32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 16;
const FILTER_RADIUS: i32 = 1;
const TILE_SIZE: u32 = BLOCK_SIZE + 2u; // 16 + 2*1 = 18

var<workgroup> tile: array<array<vec4<f32>, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>) {
    
    let x = global_id.x;
    let y = global_id.y;
    let w = params.width;
    let h = params.height;

    // Load into shared memory (tile)
    // Each thread loads its corresponding pixel + potential halo pixels
    // Simple strategy: map local_id to tile coordinates, but tile is larger than block.
    // We need to load 18x18 pixels using 16x16 threads.
    // Some threads need to load multiple pixels.
    
    let lx = local_id.x;
    let ly = local_id.y;
    
    // Base coordinates in global input
    let base_x = i32(group_id.x * BLOCK_SIZE) - FILTER_RADIUS;
    let base_y = i32(group_id.y * BLOCK_SIZE) - FILTER_RADIUS;

    // Cooperative loading
    // Total pixels to load: 18*18 = 324
    // Total threads: 16*16 = 256
    // Each thread loads 1 pixel, first 68 threads load 2nd pixel.
    
    let tid = ly * BLOCK_SIZE + lx;
    let tile_pixels = TILE_SIZE * TILE_SIZE;
    
    for (var i: u32 = tid; i < tile_pixels; i += 256u) {
        let tx = i % TILE_SIZE;
        let ty = i / TILE_SIZE;
        
        let gx = base_x + i32(tx);
        let gy = base_y + i32(ty);
        
        if (gx >= 0 && gx < i32(w) && gy >= 0 && gy < i32(h)) {
            tile[ty][tx] = input[u32(gy) * w + u32(gx)];
        } else {
            tile[ty][tx] = vec4<f32>(0.0);
        }
    }
    
    workgroupBarrier();

    if (x >= w || y >= h) {
        return;
    }

    // Diffusion calculation
    // Access tile at [ly + FILTER_RADIUS][lx + FILTER_RADIUS]
    let cx = lx + u32(FILTER_RADIUS);
    let cy = ly + u32(FILTER_RADIUS);
    
    var sum = vec4<f32>(0.0);
    var alpha_sum = 0.0;
    
    // 3x3 Filter
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let pixel = tile[i32(cy) + dy][i32(cx) + dx];
            sum += pixel * pixel.a; // Pre-multiplied alpha weighting
            alpha_sum += pixel.a;
        }
    }
    
    // Normalize and apply decay
    let filter_area = 9.0;
    let decay = params.decay;
    
    var result = sum / filter_area * decay;
    result.a = alpha_sum / filter_area * decay;
    
    // Early termination for low alpha
    if (result.a < 0.001) {
        result = vec4<f32>(0.0);
    } else {
        // Recover RGB from pre-multiplied
        if (result.a > 0.0) {
            result.r /= result.a;
            result.g /= result.a;
            result.b /= result.a;
        }
    }

    output[y * w + x] = result;
}
