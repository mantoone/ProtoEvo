extern "C"
__global__ void kernel(
    int width, int height,
    float* x_out,
    float* x_in,
    float* b,
    float a,
    float c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int i_L = (x > 0) ? idx - 1 : idx;
    int i_R = (x < width - 1) ? idx + 1 : idx;
    int i_B = (y > 0) ? idx - width : idx;
    int i_T = (y < height - 1) ? idx + width : idx;

    // For boundary conditions, we might need to be more careful, 
    // but for now let's use the clamped indices.
    // Ideally we should handle boundaries separately or use a mask.
    
    float neighbors = x_in[i_L] + x_in[i_R] + x_in[i_B] + x_in[i_T];
    
    x_out[idx] = (b[idx] + a * neighbors) / c;
}
