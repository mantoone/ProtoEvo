extern "C"
__global__ void kernel(
    int width, int height,
    float* vx,
    float* vy,
    float* p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int i_L = (x > 0) ? idx - 1 : idx;
    int i_R = (x < width - 1) ? idx + 1 : idx;
    int i_B = (y > 0) ? idx - width : idx;
    int i_T = (y < height - 1) ? idx + width : idx;

    // vx[i] -= 0.5 * (p[i+1] - p[i-1]) * N
    // vy[i] -= 0.5 * (p[j+1] - p[j-1]) * N
    // We'll assume the scaling factor is passed or handled.
    // Let's just do the gradient subtraction.
    
    vx[idx] -= 0.5f * (p[i_R] - p[i_L]);
    vy[idx] -= 0.5f * (p[i_T] - p[i_B]);
}
