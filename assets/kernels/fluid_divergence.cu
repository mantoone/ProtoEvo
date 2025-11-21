extern "C"
__global__ void kernel(
    int width, int height,
    float* div,
    float* vx,
    float* vy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int i_L = (x > 0) ? idx - 1 : idx;
    int i_R = (x < width - 1) ? idx + 1 : idx;
    int i_B = (y > 0) ? idx - width : idx;
    int i_T = (y < height - 1) ? idx + width : idx;

    // div[i] = -0.5 * (vx[i+1] - vx[i-1] + vy[j+1] - vy[j-1]) / h
    // Assuming h = 1.0 for grid cell size, or handled in scaling.
    // Standard implementation often uses h = 1/N. 
    // Let's assume we just want the finite difference.
    
    float val = -0.5f * ( (vx[i_R] - vx[i_L]) + (vy[i_T] - vy[i_B]) );
    
    div[idx] = val;
}
