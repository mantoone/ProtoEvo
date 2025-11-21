extern "C"
__global__ void kernel(
    int width, int height,
    float* d_vx, float* d_vy,
    float* d_vx0, float* d_vy0,
    float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float dt0 = dt * width; // Assuming square grid for simplicity in scaling
    float x_f = x - dt0 * d_vx0[idx];
    float y_f = y - dt0 * d_vy0[idx];

    if (x_f < 0.5f) x_f = 0.5f;
    if (x_f > width - 1.5f) x_f = width - 1.5f;
    if (y_f < 0.5f) y_f = 0.5f;
    if (y_f > height - 1.5f) y_f = height - 1.5f;

    int i0 = (int)x_f;
    int i1 = i0 + 1;
    int j0 = (int)y_f;
    int j1 = j0 + 1;

    float s1 = x_f - i0;
    float s0 = 1.0f - s1;
    float t1 = y_f - j0;
    float t0 = 1.0f - t1;

    int i0j0 = j0 * width + i0;
    int i0j1 = j1 * width + i0;
    int i1j0 = j0 * width + i1;
    int i1j1 = j1 * width + i1;

    d_vx[idx] = s0 * (t0 * d_vx0[i0j0] + t1 * d_vx0[i0j1]) +
                s1 * (t0 * d_vx0[i1j0] + t1 * d_vx0[i1j1]);
    d_vy[idx] = s0 * (t0 * d_vy0[i0j0] + t1 * d_vy0[i0j1]) +
                s1 * (t0 * d_vy0[i1j0] + t1 * d_vy0[i1j1]);
}
