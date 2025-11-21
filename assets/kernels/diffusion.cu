// Optimized diffusion kernel with shared memory and coalesced access
// Constants in constant memory for faster access
__constant__ float WORLD_RADIUS = 30.0f;
__constant__ float VOID_P = 0.9f;
__constant__ float DECAY_NORMAL = 0.995f;
__constant__ float DECAY_VOID = 0.9995f;
__constant__ float MIN_ALPHA_THRESHOLD = 5.0f / 255.0f;

__device__ const int FILTER_SIZE = 3;
__device__ const int FILTER_RADIUS = 1;

extern "C"
__global__ void kernel(
    unsigned int width,
    unsigned int height,
    unsigned int channels,
    unsigned char *img,
    unsigned char *result)
{
    // Shared memory tile - includes halo for neighbors
    // Tile size is blockDim + 2*FILTER_RADIUS to include neighbors
    extern __shared__ unsigned char tile[];
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit for out-of-bounds threads
    if (x >= width || y >= height) return;
    
    // Shared memory dimensions
    const int tile_width = blockDim.x + 2 * FILTER_RADIUS;
    const int tile_height = blockDim.y + 2 * FILTER_RADIUS;
    const int tile_x = threadIdx.x + FILTER_RADIUS;
    const int tile_y = threadIdx.y + FILTER_RADIUS;
    
    // Load center pixel into shared memory
    int global_idx = (y * width + x) * channels;
    int tile_idx = (tile_y * tile_width + tile_x) * channels;
    
    #pragma unroll
    for (int c = 0; c < 4; c++) { // Assuming 4 channels (RGBA)
        if (c < channels) {
            tile[tile_idx + c] = img[global_idx + c];
        }
    }
    
    // Load halo regions (edges and corners)
    // Each thread loads additional pixels if on the edge
    if (threadIdx.x < FILTER_RADIUS) {
        // Left halo
        int halo_x = x - FILTER_RADIUS;
        if (halo_x >= 0) {
            int halo_global = (y * width + halo_x) * channels;
            int halo_tile = (tile_y * tile_width + threadIdx.x) * channels;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c < channels) tile[halo_tile + c] = img[halo_global + c];
            }
        }
        // Right halo
        int halo_x_right = x + blockDim.x;
        if (halo_x_right < width && threadIdx.x + blockDim.x < tile_width - FILTER_RADIUS) {
            int halo_global = (y * width + halo_x_right) * channels;
            int halo_tile = (tile_y * tile_width + tile_x + blockDim.x) * channels;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c < channels) tile[halo_tile + c] = img[halo_global + c];
            }
        }
    }
    
    if (threadIdx.y < FILTER_RADIUS) {
        // Top halo
        int halo_y = y - FILTER_RADIUS;
        if (halo_y >= 0) {
            int halo_global = (halo_y * width + x) * channels;
            int halo_tile = (threadIdx.y * tile_width + tile_x) * channels;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c < channels) tile[halo_tile + c] = img[halo_global + c];
            }
        }
        // Bottom halo
        int halo_y_bottom = y + blockDim.y;
        if (halo_y_bottom < height && threadIdx.y + blockDim.y < tile_height - FILTER_RADIUS) {
            int halo_global = (halo_y_bottom * width + x) * channels;
            int halo_tile = ((tile_y + blockDim.y) * tile_width + tile_x) * channels;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c < channels) tile[halo_tile + c] = img[halo_global + c];
            }
        }
    }
    
    __syncthreads(); // Wait for all tiles to load
    
    // Calculate decay factor based on distance to world center
    float cellSizeX = 2.0f * WORLD_RADIUS / ((float) width);
    float cellSizeY = 2.0f * WORLD_RADIUS / ((float) height);
    float world_x = -WORLD_RADIUS + cellSizeX * x;
    float world_y = -WORLD_RADIUS + cellSizeY * y;
    float dist2_to_world_centre = world_x * world_x + world_y * world_y;
    float void_threshold2 = VOID_P * VOID_P * WORLD_RADIUS * WORLD_RADIUS;
    
    float decay;
    if (dist2_to_world_centre > void_threshold2) {
        float dist_to_world_centre = sqrtf(dist2_to_world_centre);
        float void_edge = VOID_P * WORLD_RADIUS;
        float void_width = (1.0f - VOID_P) * WORLD_RADIUS;
        decay = DECAY_VOID * fmaxf(0.0f, 1.0f - (dist_to_world_centre - void_edge) / void_width);
    } else {
        decay = DECAY_NORMAL;
    }
    
    const int alpha_channel = channels - 1;
    const float inv_filter_size2 = 1.0f / (float)(FILTER_SIZE * FILTER_SIZE);
    
    // Process alpha channel first with manual loop unrolling
    float final_alpha = 0.0f;
    
    // Unrolled 3x3 loop for better performance
    #pragma unroll
    for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
        #pragma unroll
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            int tx = tile_x + i;
            int ty = tile_y + j;
            int t_idx = (ty * tile_width + tx) * channels + alpha_channel;
            final_alpha += tile[t_idx] * (1.0f / 255.0f);
        }
    }
    
    final_alpha *= decay * inv_filter_size2;
    unsigned char alpha_byte = (unsigned char)(final_alpha * 255.0f);
    result[global_idx + alpha_channel] = alpha_byte;
    
    // Early termination for very low alpha values
    if (final_alpha < MIN_ALPHA_THRESHOLD) {
        #pragma unroll
        for (int c = 0; c < channels - 1; c++) {
            result[global_idx + c] = 0;
        }
        return;
    }
    
    // Process color channels
    float inv_final_alpha = 1.0f / final_alpha;
    
    #pragma unroll
    for (int c = 0; c < channels - 1; c++) {
        float final_value = 0.0f;
        
        #pragma unroll
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            #pragma unroll
            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                int tx = tile_x + i;
                int ty = tile_y + j;
                int t_idx = (ty * tile_width + tx) * channels;
                
                float alpha = decay * tile[t_idx + alpha_channel] * (1.0f / 255.0f);
                float val = tile[t_idx + c] * (1.0f / 255.0f);
                final_value += val * alpha;
            }
        }
        
        final_value *= inv_filter_size2 * decay * 255.0f * inv_final_alpha;
        result[global_idx + c] = (unsigned char)(final_value);
    }
}