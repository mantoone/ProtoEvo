package com.protoevo.utils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.File;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;


public class JCudaKernelRunner {

    static final int FILTER_SIZE = 3;

    private CUfunction function;
    private final int blockSizeX, blockSizeY;
    private final String kernelName, functionName;
    private CUdeviceptr devicePixels = new CUdeviceptr();
    private CUdeviceptr deviceOutput = new CUdeviceptr();

    public static boolean cudaAvailable() {
        try {
            new JCudaKernelRunner("diffusion");
            return true;
        } catch (RuntimeException ignored) {
            if (DebugMode.isDebugMode())
                System.out.println("CUDA is not available.");
            return false;
        }
    }

    public JCudaKernelRunner(String kernelName) {
        this(kernelName, "kernel", 32, 32);
    }

    public JCudaKernelRunner(String kernelName, String functionName, int blockSizeX, int blockSizeY) {
        this.blockSizeX = blockSizeX;
        this.blockSizeY = blockSizeY;
        this.kernelName = kernelName;
        this.functionName = functionName;

        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        initialise();
    }

    private void initialise() {
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        String kernelPath = "kernels/" + kernelName + ".cu";
        try {
            String ptxFile = JCudaUtils.preparePtxFile(kernelPath);

            // Load the ptx file. Make sure to have compiled the cu files first.
            // e.g.: > nvcc -m64 -ptx kernel.cu
            // This step should now be handled by the preparePtxFile
            CUmodule module = new CUmodule();
            cuModuleLoad(module,  ptxFile);

            // Obtain a function pointer to the function to run.
            function = new CUfunction();
            cuModuleGetFunction(function, module, functionName);

            // Allocate the device input data, and copy the
            // host input data to the device
            /*
            int pxLen = 2048*2048;
            cuMemAlloc(this.devicePixels, (long) pxLen * Sizeof.BYTE);

            // Allocate device output memory
            int resLen = 2048*2048;
            cuMemAlloc(this.deviceOutput, (long) resLen * Sizeof.BYTE);
            System.out.println("Mallocs done");
             */

        } catch (IOException e) {
            if (!new File(kernelName).exists())
                throw new RuntimeException("Was unable to compile " + kernelName + ":\n" + e);
            else {
                System.out.println("error\n" + e);
            };
        }
    }

    public CUdeviceptr[] getAdditionalParameters() {
        return null;
    }

    public byte[] processImage(byte[] pixels, int w, int h) {
        return processImage(pixels, pixels, w, h, 4);
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h) {
        return processImage(pixels, result, w, h, 4);
    }

    /*
    public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c) {
        cuMemcpyHtoD(this.devicePixels, Pointer.to(pixels), (long) pixels.length * Sizeof.BYTE);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{w}),
                Pointer.to(new int[]{h}),
                Pointer.to(new int[]{c}),
                Pointer.to(this.devicePixels),
                Pointer.to(this.deviceOutput)
        );


        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.

        // Call the kernel function.
        int gridSizeX = (int) Math.ceil((double) w / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) h / blockSizeY);
        cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,      // Grid dimension
                blockSizeX, blockSizeY, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        cuMemcpyDtoH(Pointer.to(result), deviceOutput, (long) result.length * Sizeof.BYTE);
        //cuMemFree(devicePixels);
        //cuMemFree(deviceOutput);

        return result;
    }
    */

    public static void kernelCPU(
            int width,
            int height,
            int channels,
            byte[] img,
            byte[] result)
    {
        float world_radius = 30.0f;

        float cellSizeX = 2 * world_radius / ((float) width);
        float cellSizeY = 2 * world_radius / ((float) height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float world_x = -world_radius + cellSizeX * x;
                float world_y = -world_radius + cellSizeY * y;
                float dist2_to_world_centre = world_x*world_x + world_y*world_y;

                float decay = 0.0f;

                float void_p = 0.9f;
                if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
                    float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
                    decay = 0.9995f * (1.0f - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
                    if (decay < 0.0f) {
                        decay = 0.0f;
                    }
                } else {
                    decay = 0.995f;
                }

                int alpha_channel = channels - 1;

                float final_alpha = 0.0f;
                int radius = (FILTER_SIZE - 1) / 2;
                for (int i = -radius; i <= radius; i++) {
                    for (int j = -radius; j <= radius; j++) {
                        int x_ = x + i;
                        int y_ = y + j;
                        if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                            continue;
                        }
                        float val = (float) (img[(y_*width + x_)*channels + alpha_channel] & 0xFF);
                        final_alpha += val / 255.0f;
                    }
                }
                final_alpha = decay * final_alpha / ((float) (FILTER_SIZE*FILTER_SIZE));
                result[(y*width + x)*channels + alpha_channel] = (byte) (255 * final_alpha);

                if (final_alpha < 5.0f / 255.0f) {
                    for (int i = 0; i < channels - 1; i++) {
                        result[(y*width + x)*channels + i] = 0;
                    }
                }

                float final_value = 0.0f;
                for (int c = 0; c < channels - 1; c++) {
                    final_value = 0.0f;
                    for (int i = -radius; i <= radius; i++) {
                        for (int j = -radius; j <= radius; j++) {
                            int x_ = x + i;
                            int y_ = y + j;
                            if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                                continue;
                            }
                            float alpha = decay * ((float) (img[(y_*width + x_)*channels + alpha_channel] & 0xFF)) / 255.0f;
                            float val = ((float) (img[(y_*width + x_)*channels + c] & 0xFF)) / 255.0f;
                            final_value += val * alpha;
                        }
                    }
                    final_value = final_value / ((float) (FILTER_SIZE*FILTER_SIZE));
                    final_value = decay * 255 * final_value / final_alpha;

                    result[(y*width + x)*channels + c] = (byte) (final_value);
                }
            }
        }
    }

public static void kernel(
    int width,
    int height,
    int channels,
    byte[] img,
    byte[] result)
{
    float world_radius = 30.0f;

    Quadtree quadtree = new Quadtree(0, new Rectangle(0, 0, width, height));

    // Insert all pixels into the quadtree
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            quadtree.insert(new Rectangle(x, y, 1, 1));
        }
    }

    // Traverse the quadtree and process each node
    processNode(quadtree, width, height, channels, img, result, world_radius);
}

private static void processNode(
    Quadtree node,
    int width,
    int height,
    int channels,
    byte[] img,
    byte[] result,
    float world_radius)
{
    Rectangle bounds = node.getBounds();

    // If the node is a leaf node, process each pixel in the node
    if (node.isLeaf()) {
        for (int y = bounds.y; y < bounds.y + bounds.height; y++) {
            for (int x = bounds.x; x < bounds.x + bounds.width; x++) {
                processPixel(x, y, width, height, channels, img, result, world_radius);
            }
        }
    } else {
        // Otherwise, process each child node
        for (Quadtree child : node.getChildren()) {
            processNode(child, width, height, channels, img, result, world_radius);
        }
    }
}

private static void processPixel(
    int x,
    int y,
    int width,
    int height,
    int channels,
    byte[] img,
    byte[] result,
    float world_radius)
{
    float cellSizeX = 2 * world_radius / ((float) width);
    float cellSizeY = 2 * world_radius / ((float) height);
    float world_x = -world_radius + cellSizeX * x;
    float world_y = -world_radius + cellSizeY * y;
    float dist2_to_world_centre = world_x*world_x + world_y*world_y;

    // set alpha decay to zero as we approach the void
    float decay = 0.0f;

    float void_p = 0.9f;
    if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
        float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
        // lerp from 1.0 to 0.0 for distance between void_p*world_radius and world_radius
        decay = 0.9995f * (1.0f - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
        if (decay < 0.0f) {
            decay = 0.0f;
        }
    } else {
        decay = 0.995f;
    }

    int alpha_channel = channels - 1;

    float final_alpha = 0.0f;
    int radius = (FILTER_SIZE - 1) / 2;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int x_ = x + i;
            int y_ = y + j;
            if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                continue;
            }
            float val = (float) (img[(y_*width + x_)*channels + alpha_channel] & 0xFF);
            final_alpha += val / 255.0f;
        }
    }
    final_alpha = decay * final_alpha / ((float) (FILTER_SIZE*FILTER_SIZE));
    result[(y*width + x)*channels + alpha_channel] = (byte) (255 * final_alpha);

    if (final_alpha < 5.0f / 255.0f) {
        for (int i = 0; i < channels - 1; i++) {
            result[(y*width + x)*channels + i] = 0;
        }
    }

    float final_value = 0.0f;
    // assume that the last channel is alpha
    for (int c = 0; c < channels - 1; c++) {
        final_value = 0.0f;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int x_ = x + i;
                int y_ = y + j;
                if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                    continue;
                }
                float alpha = decay * ((float) (img[(y_*width + x_)*channels + alpha_channel] & 0xFF)) / 255.0f;
                float val = ((float) (img[(y_*width + x_)*channels + c] & 0xFF)) / 255.0f;
                final_value += val * alpha;
            }
        }
        final_value = final_value / ((float) (FILTER_SIZE*FILTER_SIZE));
        final_value = decay * 255 * final_value / final_alpha;

        result[(y*width + x)*channels + c] = (byte) (final_value);
    }
}

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c) {
        kernel(w, h, c, pixels, result);
        //kernelCPU(w, h, c, pixels, result);

        return result;
    }

}
