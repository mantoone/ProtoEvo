extern "C"
__global__ void kernel(
    int width, int height,
    int* d_colors,
    int* d_colors0,
    float* d_vx, float* d_vy,
    float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float dt0 = dt * width;
    float x_f = x - dt0 * d_vx[idx];
    float y_f = y - dt0 * d_vy[idx];

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

    // Interpolate colors (RGBA packed in int)
    // This is tricky because we need to interpolate each channel.
    // Unpack
    int c00 = d_colors0[i0j0];
    int c01 = d_colors0[i0j1];
    int c10 = d_colors0[i1j0];
    int c11 = d_colors0[i1j1];

    float r = s0 * (t0 * ((c00 >> 24) & 0xFF) + t1 * ((c01 >> 24) & 0xFF)) +
              s1 * (t0 * ((c10 >> 24) & 0xFF) + t1 * ((c11 >> 24) & 0xFF));
    float g = s0 * (t0 * ((c00 >> 16) & 0xFF) + t1 * ((c01 >> 16) & 0xFF)) +
              s1 * (t0 * ((c10 >> 16) & 0xFF) + t1 * ((c11 >> 16) & 0xFF));
    float b = s0 * (t0 * ((c00 >> 8) & 0xFF) + t1 * ((c01 >> 8) & 0xFF)) +
              s1 * (t0 * ((c10 >> 8) & 0xFF) + t1 * ((c11 >> 8) & 0xFF));
    float a = s0 * (t0 * (c00 & 0xFF) + t1 * (c01 & 0xFF)) +
              s1 * (t0 * (c10 & 0xFF) + t1 * (c11 & 0xFF));

    d_colors[idx] = ((int)r << 24) | ((int)g << 16) | ((int)b << 8) | (int)a;
}
