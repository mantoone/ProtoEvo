package com.protoevo.env;

import com.badlogic.gdx.math.Vector2;
import com.protoevo.core.Simulation;
import com.protoevo.utils.DebugMode;
import com.protoevo.utils.JCudaUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class FluidField {

    private final int width, height;
    private final int size;
    private boolean initialized = false;

    // CUDA variables
    private CUcontext context;
    private java.util.List<CUmodule> modules = new java.util.ArrayList<>();
    private CUfunction advectKernel, diffuseKernel, divergenceKernel, pressureKernel, subtractGradientKernel;
    
    // Device pointers
    private CUdeviceptr d_vx, d_vy;
    private CUdeviceptr d_vx0, d_vy0;
    private CUdeviceptr d_div, d_p;
    
    // Host variables (for initialization and debug)
    private float[] vx, vy;

    public FluidField(int width, int height) {
        this.width = width;
        this.height = height;
        this.size = width * height;
        this.vx = new float[size];
        this.vy = new float[size];
        
        // Don't initialize CUDA in constructor - will be done lazily on first step()
        // to ensure it happens in the correct thread
    }

    private CUfunction chemicalAdvectKernel;

    private void initCUDA() throws IOException {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load kernels
        advectKernel = loadKernel("fluid_advect");
        diffuseKernel = loadKernel("fluid_lin_solve"); // Reusing lin_solve for diffuse
        divergenceKernel = loadKernel("fluid_divergence");
        pressureKernel = loadKernel("fluid_lin_solve"); // Reusing lin_solve for pressure
        subtractGradientKernel = loadKernel("fluid_subtract_gradient");
        chemicalAdvectKernel = loadKernel("chemical_advect");

        // Allocate device memory
        d_vx = allocateDevice(size * Sizeof.FLOAT);
        d_vy = allocateDevice(size * Sizeof.FLOAT);
        d_vx0 = allocateDevice(size * Sizeof.FLOAT);
        d_vy0 = allocateDevice(size * Sizeof.FLOAT);
        d_div = allocateDevice(size * Sizeof.FLOAT);
        d_p = allocateDevice(size * Sizeof.FLOAT);

        initialized = true;
    }

    public void advectChemicals(byte[] pixels, int w, int h, float dt) {
        // Lazy initialization - initialize CUDA on first call to ensure we're in the correct thread
        if (!initialized && Environment.settings.misc.useCUDA.get()) {
            try {
                initCUDA();
            } catch (Exception e) {
                System.err.println("Failed to initialize CUDA for FluidField: " + e.getMessage());
                e.printStackTrace();
                return;
            }
        }
        
        if (!initialized) return;
        
        // Ensure context is current on this thread
        cuCtxSetCurrent(context);
        
        // We assume pixels is RGBA8888 (4 bytes per pixel)
        // The kernel expects int* (4 bytes per pixel)
        
        // Allocate device memory for pixels
        CUdeviceptr d_pixels = allocateDevice((long) pixels.length * Sizeof.BYTE);
        CUdeviceptr d_pixels0 = allocateDevice((long) pixels.length * Sizeof.BYTE);
        
        // Copy host to device
        cuMemcpyHtoD(d_pixels0, Pointer.to(pixels), (long) pixels.length * Sizeof.BYTE);
        
        // Launch kernel
        Pointer params = Pointer.to(
            Pointer.to(new int[]{w}),
            Pointer.to(new int[]{height}),
            Pointer.to(d_pixels),   // dest
            Pointer.to(d_pixels0),  // source
            Pointer.to(d_vx),
            Pointer.to(d_vy),
            Pointer.to(new float[]{dt})
        );
        
        int blockSizeX = 32;
        int blockSizeY = 32;
        int gridSizeX = (w + blockSizeX - 1) / blockSizeX;
        int gridSizeY = (h + blockSizeY - 1) / blockSizeY;

        cuLaunchKernel(chemicalAdvectKernel,
            gridSizeX, gridSizeY, 1,
            blockSizeX, blockSizeY, 1,
            0, null,
            params, null
        );
        
        // Copy back
        cuMemcpyDtoH(Pointer.to(pixels), d_pixels, (long) pixels.length * Sizeof.BYTE);
        
        // Free
        cuMemFree(d_pixels);
        cuMemFree(d_pixels0);
    }


    private CUfunction loadKernel(String name) throws IOException {
        String ptxFile = JCudaUtils.preparePtxFile("kernels/" + name + ".cu");
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFile);
        modules.add(module); // Keep reference to prevent garbage collection
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "kernel");
        return function;
    }

    private CUdeviceptr allocateDevice(long bytes) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuMemAlloc(ptr, bytes);
        return ptr;
    }

    public void step(float dt) {
        // Lazy initialization - initialize CUDA on first call to ensure we're in the correct thread
        if (!initialized && Environment.settings.misc.useCUDA.get()) {
            try {
                initCUDA();
            } catch (Exception e) {
                System.err.println("Failed to initialize CUDA for FluidField: " + e.getMessage());
                e.printStackTrace();
                return;
            }
        }
        
        if (!initialized) return;

        // Ensure context is current on this thread
        cuCtxSetCurrent(context);

        // 1. Diffuse (Viscosity)
        // We use Jacobi iteration.
        // Copy vx to vx0 to use as source
        copyDevice(d_vx, d_vx0, size * Sizeof.FLOAT);
        copyDevice(d_vy, d_vy0, size * Sizeof.FLOAT);
        
        float diff = 0.0001f; // Viscosity
        float a = dt * diff * (width - 2) * (height - 2);
        float c = 1 + 4 * a;
        
        // Iterations for diffusion - reduced from 20 to 4 for performance
        // 4 iterations is sufficient for real-time fluid sim
        for (int i = 0; i < 4; i++) {
            // vx
            launchLinSolve(d_vx, d_vx, d_vx0, a, c);
            // vy
            launchLinSolve(d_vy, d_vy, d_vy0, a, c);
        }

        // 2. Project (Mass Conservation)
        project();

        // 3. Advect
        // Advect vx and vy along the velocity field.
        // We copy current vx/vy to vx0/vy0 to serve as the source field to be advected.
        copyDevice(d_vx, d_vx0, size * Sizeof.FLOAT);
        copyDevice(d_vy, d_vy0, size * Sizeof.FLOAT);
        
        // Advect velocity field (both components at once)
        launchAdvectVelocity(dt);


        // 4. Project again
        project();
    }
    
    public void syncVelocityToCPU() {
        if (!initialized) return;
        cuCtxSetCurrent(context);
        
        // Copy velocity data to host for CPU-side queries
        cuMemcpyDtoH(Pointer.to(vx), d_vx, (long) size * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(vy), d_vy, (long) size * Sizeof.FLOAT);
    }

    private void project() {
        // Calculate divergence
        launchDivergence(d_div, d_vx, d_vy);
        
        // Set pressure to 0
        cuMemsetD32(d_p, 0, size);
        
        // Solve pressure (Poisson equation)
        // Reduced from 20 to 8 iterations for performance
        // 8 is enough to maintain reasonable incompressibility
        for (int i = 0; i < 8; i++) {
            launchLinSolve(d_p, d_p, d_div, 1, 4);
        }
        
        // Subtract gradient
        launchSubtractGradient(d_vx, d_vy, d_p);
    }

    private void launchLinSolve(CUdeviceptr x_out, CUdeviceptr x_in, CUdeviceptr b, float a, float c) {
        Pointer params = Pointer.to(
            Pointer.to(new int[]{width}),
            Pointer.to(new int[]{height}),
            Pointer.to(x_out),
            Pointer.to(x_in),
            Pointer.to(b),
            Pointer.to(new float[]{a}),
            Pointer.to(new float[]{c})
        );
        launch(diffuseKernel, params);
    }
    
    private void launchAdvectVelocity(float dt) {
         Pointer params = Pointer.to(
            Pointer.to(new int[]{width}),
            Pointer.to(new int[]{height}),
            Pointer.to(d_vx),
            Pointer.to(d_vy),
            Pointer.to(d_vx0),
            Pointer.to(d_vy0),
            Pointer.to(new float[]{dt})
        );
        launch(advectKernel, params);
    }

    private void launchDivergence(CUdeviceptr div, CUdeviceptr u, CUdeviceptr v) {
        Pointer params = Pointer.to(
            Pointer.to(new int[]{width}),
            Pointer.to(new int[]{height}),
            Pointer.to(div),
            Pointer.to(u),
            Pointer.to(v)
        );
        launch(divergenceKernel, params);
    }

    private void launchSubtractGradient(CUdeviceptr u, CUdeviceptr v, CUdeviceptr p) {
        Pointer params = Pointer.to(
            Pointer.to(new int[]{width}),
            Pointer.to(new int[]{height}),
            Pointer.to(u),
            Pointer.to(v),
            Pointer.to(p)
        );
        launch(subtractGradientKernel, params);
    }

    private void launch(CUfunction function, Pointer params) {
        int blockSizeX = 32;
        int blockSizeY = 32;
        int gridSizeX = (width + blockSizeX - 1) / blockSizeX;
        int gridSizeY = (height + blockSizeY - 1) / blockSizeY;

        cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,
            blockSizeX, blockSizeY, 1,
            0, null,
            params, null
        );
    }
    
    private void copyDevice(CUdeviceptr dst, CUdeviceptr src, long bytes) {
        cuMemcpyDtoD(dst, src, bytes);
    }
    
    public void dispose() {
        if (initialized) {
            cuMemFree(d_vx);
            cuMemFree(d_vy);
            cuMemFree(d_vx0);
            cuMemFree(d_vy0);
            cuMemFree(d_div);
            cuMemFree(d_p);
            cuCtxDestroy(context);
        }
    }
    
    public CUdeviceptr getVx() { return d_vx; }
    public CUdeviceptr getVy() { return d_vy; }

    public Vector2 getVelocityAt(float worldX, float worldY, float fieldRadius) {
        if (!initialized) return Vector2.Zero;

        // Map world coordinates to grid coordinates
        // World: [-radius, radius] -> Grid: [0, width]
        float u = (worldX + fieldRadius) / (2 * fieldRadius);
        float v = (worldY + fieldRadius) / (2 * fieldRadius);
        
        float gridX = u * width;
        float gridY = v * height;
        
        // Bilinear interpolation
        int x0 = (int) gridX;
        int y0 = (int) gridY;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float sx = gridX - x0;
        float sy = gridY - y0;
        
        // Clamp coordinates
        x0 = Math.max(0, Math.min(x0, width - 1));
        y0 = Math.max(0, Math.min(y0, height - 1));
        x1 = Math.max(0, Math.min(x1, width - 1));
        y1 = Math.max(0, Math.min(y1, height - 1));
        
        int idx00 = x0 + y0 * width;
        int idx10 = x1 + y0 * width;
        int idx01 = x0 + y1 * width;
        int idx11 = x1 + y1 * width;
        
        float vx00 = vx[idx00];
        float vx10 = vx[idx10];
        float vx01 = vx[idx01];
        float vx11 = vx[idx11];
        
        float vy00 = vy[idx00];
        float vy10 = vy[idx10];
        float vy01 = vy[idx01];
        float vy11 = vy[idx11];
        
        float lerpVx1 = (1 - sx) * vx00 + sx * vx10;
        float lerpVx2 = (1 - sx) * vx01 + sx * vx11;
        float finalVx = (1 - sy) * lerpVx1 + sy * lerpVx2;
        
        float lerpVy1 = (1 - sx) * vy00 + sx * vy10;
        float lerpVy2 = (1 - sx) * vy01 + sx * vy11;
        float finalVy = (1 - sy) * lerpVy1 + sy * lerpVy2;
        
        return new Vector2(finalVx, finalVy);
    }

    public void injectVelocity(int x, int y, float u, float v) {
        if (!initialized) return;
        if (x < 0 || x >= width || y < 0 || y >= height) return;
        
        // Ensure context is current on this thread
        cuCtxSetCurrent(context);
        
        int idx = x + y * width;
        cuMemcpyHtoD(d_vx.withByteOffset((long) idx * Sizeof.FLOAT), Pointer.to(new float[]{u}), Sizeof.FLOAT);
        cuMemcpyHtoD(d_vy.withByteOffset((long) idx * Sizeof.FLOAT), Pointer.to(new float[]{v}), Sizeof.FLOAT);
    }
}
