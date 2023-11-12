package com.protoevo.utils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;
import java.util.concurrent.ExecutionException;

import static jcuda.driver.JCudaDriver.*;

public class JCudaKernelRunner {

    static final int FILTER_SIZE = 3;

    private CUfunction function;
    private final int blockSizeX, blockSizeY;
    private final String kernelName, functionName;
    private int executions = 0;
    private long totalTime = 0;
    private CUdeviceptr devicePixels = new CUdeviceptr();
    private CUdeviceptr deviceOutput = new CUdeviceptr();
    private ExecutorService executor;
    private static final int numProcessors = Runtime.getRuntime().availableProcessors();

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
        this.executor = Executors.newFixedThreadPool(JCudaKernelRunner.numProcessors);

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
            cuModuleLoad(module, ptxFile);

            // Obtain a function pointer to the function to run.
            function = new CUfunction();
            cuModuleGetFunction(function, module, functionName);

            // Allocate the device input data, and copy the
            // host input data to the device
            /*
             * int pxLen = 2048*2048;
             * cuMemAlloc(this.devicePixels, (long) pxLen * Sizeof.BYTE);
             * 
             * // Allocate device output memory
             * int resLen = 2048*2048;
             * cuMemAlloc(this.deviceOutput, (long) resLen * Sizeof.BYTE);
             * System.out.println("Mallocs done");
             */

        } catch (IOException e) {
            if (!new File(kernelName).exists())
                throw new RuntimeException("Was unable to compile " + kernelName + ":\n" + e);
            else {
                System.out.println("error\n" + e);
            }
            ;
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
     * public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c)
     * {
     * cuMemcpyHtoD(this.devicePixels, Pointer.to(pixels), (long) pixels.length *
     * Sizeof.BYTE);
     * 
     * Pointer kernelParameters = Pointer.to(
     * Pointer.to(new int[]{w}),
     * Pointer.to(new int[]{h}),
     * Pointer.to(new int[]{c}),
     * Pointer.to(this.devicePixels),
     * Pointer.to(this.deviceOutput)
     * );
     * 
     * 
     * // Set up the kernel parameters: A pointer to an array
     * // of pointers which point to the actual values.
     * 
     * // Call the kernel function.
     * int gridSizeX = (int) Math.ceil((double) w / blockSizeX);
     * int gridSizeY = (int) Math.ceil((double) h / blockSizeY);
     * cuLaunchKernel(function,
     * gridSizeX, gridSizeY, 1, // Grid dimension
     * blockSizeX, blockSizeY, 1, // Block dimension
     * 0, null, // Shared memory size and stream
     * kernelParameters, null // Kernel- and extra parameters
     * );
     * 
     * cuCtxSynchronize();
     * 
     * // Allocate host output memory and copy the device output
     * // to the host.
     * cuMemcpyDtoH(Pointer.to(result), deviceOutput, (long) result.length *
     * Sizeof.BYTE);
     * //cuMemFree(devicePixels);
     * //cuMemFree(deviceOutput);
     * 
     * return result;
     * }
     */

    public static void kernelCPU(
            int width,
            int height,
            int channels,
            byte[] img,
            byte[] result) {
        float world_radius = 30.0f;

        float cellSizeX = 2 * world_radius / ((float) width);
        float cellSizeY = 2 * world_radius / ((float) height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float world_x = -world_radius + cellSizeX * x;
                float world_y = -world_radius + cellSizeY * y;
                float dist2_to_world_centre = world_x * world_x + world_y * world_y;

                float decay = 0.0f;

                float void_p = 0.9f;
                if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
                    float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
                    decay = 0.9995f * (1.0f
                            - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
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
                        float val = (float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF);
                        final_alpha += val / 255.0f;
                    }
                }
                final_alpha = decay * final_alpha / ((float) (FILTER_SIZE * FILTER_SIZE));
                result[(y * width + x) * channels + alpha_channel] = (byte) (255 * final_alpha);

                if (final_alpha < 5.0f / 255.0f) {
                    for (int i = 0; i < channels - 1; i++) {
                        result[(y * width + x) * channels + i] = 0;
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
                            float alpha = decay * ((float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF))
                                    / 255.0f;
                            float val = ((float) (img[(y_ * width + x_) * channels + c] & 0xFF)) / 255.0f;
                            final_value += val * alpha;
                        }
                    }
                    final_value = final_value / ((float) (FILTER_SIZE * FILTER_SIZE));
                    final_value = decay * 255 * final_value / final_alpha;

                    result[(y * width + x) * channels + c] = (byte) (final_value);
                }
            }
        }
    }

    public void kernelCPUPar2(
        int width,
        int height,
        int channels,
        byte[] img,
        byte[] result) {
        float world_radius = 30.0f;

        float cellSizeX = 2 * world_radius / ((float) width);
        float cellSizeY = 2 * world_radius / ((float) height);

        List<Future<?>> futures = new ArrayList<>();

        int chunkSize = height / numProcessors;

        for (int i = 0; i < numProcessors; i++) {
            int startY = i * chunkSize;
            int endY = (i == numProcessors - 1) ? height : (i + 1) * chunkSize;

            futures.add(this.executor.submit(() -> {
                float[] world_x_vec = new float[width];
                for (int x = 0; x < width; x++) {
                    world_x_vec[x] = -world_radius + cellSizeX * x;
                }

                for (int y = startY; y < endY; y++) {
                    int offset = y * width * channels;
                    for (int x = 0; x < width; x++) {
                        float world_x = world_x_vec[x];
                        float world_y = -world_radius + cellSizeY * y;
                        float dist2_to_world_centre = world_x * world_x + world_y * world_y;

                        float decay = 0.0f;

                        float void_p = 0.9f;
                        if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
                            float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
                            decay = 0.9995f * (1.0f
                                    - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
                            if (decay < 0.0f) {
                                decay = 0.0f;
                            }
                        } else {
                            decay = 0.995f;
                        }

                        int alpha_channel = channels - 1;

                        float final_alpha = 0.0f;
                        float[] alpha_sum_vec = new float[channels - 1];
                        int radius = (FILTER_SIZE - 1) / 2;
                        for (int j = -radius; j <= radius; j++) {
                            int y_ = y + j;
                            if (y_ < 0 || y_ >= height) {
                                continue;
                            }
                            int rowOffset = y_ * width * channels;
                            for (int i_ = -radius; i_ <= radius; i_++) {
                                int x_ = x + i_;
                                if (x_ < 0 || x_ >= width) {
                                    continue;
                                }
                                float alpha = decay * ((float) (img[(rowOffset + x_ * channels + alpha_channel)] & 0xFF))
                                        / 255.0f;
                                for (int c = 0; c < channels - 1; c++) {
                                    float val = ((float) (img[(rowOffset + x_ * channels + c)] & 0xFF)) / 255.0f;
                                    alpha_sum_vec[c] += val * alpha;
                                }
                            }
                        }
                        for (int c = 0; c < channels - 1; c++) {
                            final_alpha += alpha_sum_vec[c];
                        }
                        final_alpha = decay * final_alpha / ((float) (FILTER_SIZE * FILTER_SIZE));
                        result[offset + x * channels + alpha_channel] = (byte) (255 * final_alpha);

                        if (final_alpha < 5.0f / 255.0f) {
                            for (int i_ = 0; i_ < channels - 1; i_++) {
                                result[offset + x * channels + i_] = 0;
                            }
                        } else {
                            for (int c = 0; c < channels - 1; c++) {
                                float final_value = 0.0f;
                                float channel_sum = 0.0f;
                                for (int j = -radius; j <= radius; j++) {
                                    int y_ = y + j;
                                    if (y_ < 0 || y_ >= height) {
                                        continue;
                                    }
                                    int rowOffset = y_ * width * channels;
                                    for (int i_ = -radius; i_ <= radius; i_++) {
                                        int x_ = x + i_;
                                        if (x_ < 0 || x_ >= width) {
                                            continue;
                                        }
                                        float alpha = decay * ((float) (img[(rowOffset + x_ * channels + alpha_channel)] & 0xFF))
                                                / 255.0f;
                                        float val = ((float) (img[(rowOffset + x_ * channels + c)] & 0xFF)) / 255.0f;
                                        final_value += val * alpha;
                                        channel_sum += val;
                                    }
                                }
                                final_value = final_value / ((float) (FILTER_SIZE * FILTER_SIZE));
                                final_value = decay * 255 * final_value / final_alpha;

                                if (channel_sum < 1e-6f) {
                                    final_value = 0.0f;
                                }

                                result[offset + x * channels + c] = (byte) (final_value);
                            }
                        }
                    }
                }
            }));
        }

        // Wait for all tasks to complete
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    public void kernelCPUPar(
        int width,
        int height,
        int channels,
        byte[] img,
        byte[] result)
    {
        float world_radius = 30.0f;

        float cellSizeX = 2 * world_radius / ((float) width);
        float cellSizeY = 2 * world_radius / ((float) height);

        int numProcessors = Runtime.getRuntime().availableProcessors();
        List<Future<?>> futures = new ArrayList<>();

        int chunkSize = height / numProcessors;

        for (int i = 0; i < numProcessors; i++) {
            int startY = i * chunkSize;
            int endY = (i == numProcessors - 1) ? height : (i + 1) * chunkSize;

            futures.add(this.executor.submit(() -> {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < width; x++) {
                        float world_x = -world_radius + cellSizeX * x;
                        float world_y = -world_radius + cellSizeY * y;
                        float dist2_to_world_centre = world_x * world_x + world_y * world_y;

                        float decay = 0.0f;

                        float void_p = 0.9f;
                        if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
                            float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
                            decay = 0.9995f * (1.0f
                                    - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
                            if (decay < 0.0f) {
                                decay = 0.0f;
                            }
                        } else {
                            decay = 0.995f;
                        }

                        int alpha_channel = channels - 1;

                        float final_alpha = 0.0f;
                        int radius = (FILTER_SIZE - 1) / 2;
                        for (int i_ = -radius; i_ <= radius; i_++) {
                            for (int j = -radius; j <= radius; j++) {
                                int x_ = x + i_;
                                int y_ = y + j;
                                if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                                    continue;
                                }
                                float val = (float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF);
                                final_alpha += val / 255.0f;
                            }
                        }
                        final_alpha = decay * final_alpha / ((float) (FILTER_SIZE * FILTER_SIZE));
                        result[(y * width + x) * channels + alpha_channel] = (byte) (255 * final_alpha);

                        if (final_alpha < 5.0f / 255.0f) {
                            for (int i_ = 0; i_ < channels - 1; i_++) {
                                result[(y * width + x) * channels + i_] = 0;
                            }
                        }

                        float final_value = 0.0f;
                        for (int c = 0; c < channels - 1; c++) {
                            final_value = 0.0f;
                            for (int i_ = -radius; i_ <= radius; i_++) {
                                for (int j = -radius; j <= radius; j++) {
                                    int x_ = x + i_;
                                    int y_ = y + j;
                                    if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                                        continue;
                                    }
                                    float alpha = decay * ((float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF))
                                            / 255.0f;
                                    float val = ((float) (img[(y_ * width + x_) * channels + c] & 0xFF)) / 255.0f;
                                    final_value += val * alpha;
                                }
                            }
                            final_value = final_value / ((float) (FILTER_SIZE * FILTER_SIZE));
                            final_value = decay * 255 * final_value / final_alpha;

                            result[(y * width + x) * channels + c] = (byte) (final_value);
                        }
                    }
                }
            }));
        }

        // Wait for all tasks to complete
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    public void kernelCPUPar3(
        int width,
        int height,
        int channels,
        byte[] img,
        byte[] result)
    {
    float world_radius = 30.0f;

    float cellSizeX = 2 * world_radius / ((float) width);
    float cellSizeY = 2 * world_radius / ((float) height);

    int numProcessors = Runtime.getRuntime().availableProcessors();
    List<Future<?>> futures = new ArrayList<>();

    int chunkSize = height / numProcessors;

    for (int i = 0; i < numProcessors; i++) {
        int startY = i * chunkSize;
        int endY = (i == numProcessors - 1) ? height : (i + 1) * chunkSize;

        futures.add(this.executor.submit(() -> {
            for (int x = 0; x < width; x++) {
                float world_x = -world_radius + cellSizeX * x;

                for (int y = startY; y < endY; y++) {
                    float world_y = -world_radius + cellSizeY * y;
                    float dist2_to_world_centre = world_x * world_x + world_y * world_y;

                    float decay = 0.0f;

                    float void_p = 0.9f;
                    if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
                        float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
                        decay = 0.9995f * (1.0f
                                - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
                        if (decay < 0.0f) {
                            decay = 0.0f;
                        }
                    } else {
                        decay = 0.995f;
                    }

                    int alpha_channel = channels - 1;

                    float final_alpha = 0.0f;
                    int radius = (FILTER_SIZE - 1) / 2;
                    for (int j = -radius; j <= radius; j++) {
                        int y_ = y + j;
                        if (y_ < 0 || y_ >= height) {
                            continue;
                        }
                        for (int i_ = -radius; i_ <= radius; i_++) {
                            int x_ = x + i_;
                            if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                                continue;
                            }
                            float val = (float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF);
                            final_alpha += val / 255.0f;
                        }
                    }
                    final_alpha = decay * final_alpha / ((float) (FILTER_SIZE * FILTER_SIZE));
                    result[(y * width + x) * channels + alpha_channel] = (byte) (255 * final_alpha);

                    if (final_alpha < 5.0f / 255.0f) {
                        for (int i_ = 0; i_ < channels - 1; i_++) {
                            result[(y * width + x) * channels + i_] = 0;
                        }
                    }

                    for (int c = 0; c < channels - 1; c++) {
                        float final_value = 0.0f;
                        for (int j = -radius; j <= radius; j++) {
                            int y_ = y + j;
                            if (y_ < 0 || y_ >= height) {
                                continue;
                            }
                            for (int i_ = -radius; i_ <= radius; i_++) {
                                int x_ = x + i_;
                                if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) {
                                    continue;
                                }
                                float alpha = decay * ((float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF))
                                        / 255.0f;
                                float val = ((float) (img[(y_ * width + x_) * channels + c] & 0xFF)) / 255.0f;
                                final_value += val * alpha;
                            }
                        }
                        final_value = final_value / ((float) (FILTER_SIZE * FILTER_SIZE));
                        final_value = decay * 255 * final_value / final_alpha;

                        result[(y * width + x) * channels + c] = (byte) (final_value);
                    }
                }
            }
        }));
    }

    // Wait for all tasks to complete
    for (Future<?> future : futures) {
        try {
            future.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
}

    public static void kernel(
            int width,
            int height,
            int channels,
            byte[] img,
            byte[] result) {
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
            float world_radius) {
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
            float world_radius) {
        float cellSizeX = 2 * world_radius / ((float) width);
        float cellSizeY = 2 * world_radius / ((float) height);
        float world_x = -world_radius + cellSizeX * x;
        float world_y = -world_radius + cellSizeY * y;
        float dist2_to_world_centre = world_x * world_x + world_y * world_y;

        // set alpha decay to zero as we approach the void
        float decay = 0.0f;

        float void_p = 0.9f;
        if (dist2_to_world_centre > void_p * void_p * world_radius * world_radius) {
            float dist_to_world_centre = (float) Math.sqrt(dist2_to_world_centre);
            // lerp from 1.0 to 0.0 for distance between void_p*world_radius and
            // world_radius
            decay = 0.9995f
                    * (1.0f - (dist_to_world_centre - void_p * world_radius) / ((1.0f - void_p) * world_radius));
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
                float val = (float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF);
                final_alpha += val / 255.0f;
            }
        }
        final_alpha = decay * final_alpha / ((float) (FILTER_SIZE * FILTER_SIZE));
        result[(y * width + x) * channels + alpha_channel] = (byte) (255 * final_alpha);

        if (final_alpha < 5.0f / 255.0f) {
            for (int i = 0; i < channels - 1; i++) {
                result[(y * width + x) * channels + i] = 0;
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
                    float alpha = decay * ((float) (img[(y_ * width + x_) * channels + alpha_channel] & 0xFF)) / 255.0f;
                    float val = ((float) (img[(y_ * width + x_) * channels + c] & 0xFF)) / 255.0f;
                    final_value += val * alpha;
                }
            }
            final_value = final_value / ((float) (FILTER_SIZE * FILTER_SIZE));
            final_value = decay * 255 * final_value / final_alpha;

            result[(y * width + x) * channels + c] = (byte) (final_value);
        }
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c) {
        // Benchmark the kernel
        long startTime = System.nanoTime();
        // kernel(w, h, c, pixels, result);
        //kernelCPU(w, h, c, pixels, result);
        //kernelCPUPar2(w, h, c, pixels, result);
        kernelCPUPar3(w, h, c, pixels, result);

        long endTime = System.nanoTime();
        totalTime += endTime - startTime;
        executions++;
        if (executions > 30) {
            System.out.println("Average kernel time: " + totalTime / 1000000 / executions + "ms");
            executions = 0;
            totalTime = 0;
        }

        return result;
    }

}
