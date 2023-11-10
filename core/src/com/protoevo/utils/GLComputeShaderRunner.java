package com.protoevo.utils;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static com.protoevo.test.lwjgldemo.IOUtils.ioResourceToByteBuffer;
import static com.protoevo.utils.Utils.randomLong;
import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.opengl.GLUtil.setupDebugMessageCallback;
import static org.lwjgl.stb.STBImage.*;
import static org.lwjgl.opengl.ARBDebugOutput.GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB;
import static org.lwjgl.opengl.GL43C.*;

import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;

import com.protoevo.core.ApplicationManager;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;


public class GLComputeShaderRunner {
    private class GLThread {
        private final ExecutorService executor = Executors.newSingleThreadExecutor();

        public void execute(Runnable task) {
            /*
            FutureTask<?> futureTask = new FutureTask<>(task, null);
            executor.execute(futureTask);
            try {
                futureTask.get();
            } catch (InterruptedException e) {
                System.out.println("Error executing task: " + e);
            } catch (ExecutionException e){
                System.out.println("Error executing task: " + e);
            }
            */

            task.run();
        }

        public void shutdown() {
            executor.shutdown();
        }

    }

    static final int FILTER_SIZE = 3;

    private final int blockSizeX, blockSizeY;
    private final String kernelName, functionName;
    private int executions = 0;
    private long totalTime = 0;
    private int devicePixels, deviceOutput;
    private int program, computeShader;
    private long window;
    private boolean initialized = false;
    private GLThread glThread = new GLThread();
    private ByteBuffer inputBuffer = BufferUtils.createByteBuffer(1024*1024*4);
    private ByteBuffer outputBuffer = BufferUtils.createByteBuffer(1024*1024*4);

    /* OpenGL resources */
    private int[] textures = new int[2];

    public GLComputeShaderRunner(String kernelName) {
        this(kernelName, "kernel", 8, 8);
    }

    public GLComputeShaderRunner(String kernelName, String functionName, int blockSizeX, int blockSizeY) {
        System.out.println("Creating GLComputeShaderRunner");
        this.blockSizeX = blockSizeX;
        this.blockSizeY = blockSizeY;
        this.kernelName = kernelName;
        this.functionName = functionName;

        glThread.execute(() -> {
            //initialise();
        });
    }

    private void initialise() {
        if (initialized)
            return;

        int error = 0;
        // Create the compute shader
        if (!glfwInit())
            throw new AssertionError("Failed to initialize GLFW");
        // Create opengl context
        /*
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // <- make the window visible explicitly later
        */
        window = ApplicationManager.window;
        if (window == NULL)
            throw new AssertionError("Failed to create GLFW window");
        glfwMakeContextCurrent(window);

        computeShader = glCreateShader(GL_COMPUTE_SHADER);

        // Load the shader source code
        String shaderSource = null;
        try {
            shaderSource = new String(Files.readAllBytes(Paths.get("shaders/compute/" + kernelName + ".cs.glsl")), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("Was unable to load " + kernelName + ":\n" + e);
        }

        // Compile the compute shader
        glShaderSource(computeShader, shaderSource);
        glCompileShader(computeShader);
        if (glGetShaderi(computeShader, GL_COMPILE_STATUS) != GL_TRUE) {
            throw new RuntimeException("Failed to compile compute shader:\n" + glGetShaderInfoLog(computeShader));
        }

        // Create the program and attach the compute shader
        program = glCreateProgram();
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Init Error: " + error);
        }
        System.out.println("Program: " + program);
        glAttachShader(program, computeShader);

        // Link the program
        glLinkProgram(program);
        if (glGetProgrami(program, GL_LINK_STATUS) != GL_TRUE) {
            throw new RuntimeException("Failed to link program:\n" + glGetProgramInfoLog(program));
        }

        glUseProgram(program);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Init Error: " + error);
        }

        // Delete the compute shader
        glDeleteShader(computeShader);

        // Create textures

        textures[0] = glGenTextures();
            
        glBindTexture(GL_TEXTURE_2D, textures[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        final int texSize = 1024; // Was 2048
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, texSize, texSize);

        textures[1] = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glBindImageTexture(1, textures[1], 0, false, 0, GL_READ_WRITE, GL_RGBA8UI);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        ByteBuffer inputBuffer = BufferUtils.createByteBuffer(texSize*texSize*4);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8I, texSize, texSize, 0, GL_RGBA_INTEGER, GL_BYTE, inputBuffer);

        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Init Error: " + error);
        }

        glfwMakeContextCurrent(window);

        initialized = true;
    }

    public byte[] processImage(byte[] pixels, int w, int h) {
        return processImage(pixels, pixels, w, h, 4);
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h) {
        return processImage(pixels, result, w, h, 4);
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c) {
        glThread.execute(() -> {
            processImageJob(pixels, result, w, h, c);
        });

        return result;
    }

    public byte[] processImageJob(byte[] pixels, byte[] result, int w, int h, int c) {
        if(!initialized) {
            if (ApplicationManager.window == NULL){
                System.out.println("Window is null");
                return result;
            }
            else {
                initialise();
            }
        } 
        int error = 0;
        long startTime = System.nanoTime();

        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error1a: " + error);
        }

        glBindImageTexture(0, textures[0], 0, false, 0, GL_READ_WRITE, GL_RGBA8UI);
        glBindImageTexture(1, textures[1], 0, false, 0, GL_READ_WRITE, GL_RGBA8UI);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error1z: " + error);
        }

        // Bind the program and set the input and output buffer bindings
        glUseProgram(program);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error1x: " + error);
        }

        // Set the kernel parameters
        glUniform1i(glGetUniformLocation(program, "width"), w);
        glUniform1i(glGetUniformLocation(program, "height"), h);
        glUniform1i(glGetUniformLocation(program, "channels"), c);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error1y: " + error);
        }

        inputBuffer.position(0);
        inputBuffer.put(pixels);
        inputBuffer.position(0);


        // Copy the input to the input buffer
        // Set texture parameters
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error binding texture: " + error);
        }


        glBindTexture(GL_TEXTURE_2D, textures[1]);
        //glActiveTexture(GL_TEXTURE1);

        glBindImageTexture(1, textures[1], 0, false, 0, GL_READ_WRITE, GL_RGBA8UI);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Input buffer is 32x32
        //ByteBuffer inputBuffer2 = BufferUtils.createByteBuffer(w*h*4);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8I, w, h, 0, GL_RGBA_INTEGER, GL_BYTE, inputBuffer);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA_INTEGER, GL_BYTE, inputBuffer);

        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error copy to shader test: " + error);
        }

        glBindTexture(GL_TEXTURE_2D, 0);

        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error copy to shader: " + error);
        }

        // Dispatch the compute shader
        int gridSizeX = (int) Math.ceil((double) w / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) h / blockSizeY);
        long startTimeCompute = System.nanoTime();

        // Compute runs in ~0ms
        glDispatchCompute(gridSizeX, gridSizeY, 1);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("ErrorCompute: " + error);
        }

        // Wait for the compute shader to finish
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        long endTime2 = System.nanoTime();
        long totalTimeCompute = endTime2 - startTimeCompute;
        //System.out.println("Compute time: " + totalTimeCompute / 1000000 + "ms");

        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error1: " + error);
        }

        // Read the result
        glBindTexture(GL_TEXTURE_2D, textures[0]);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error2: " + error);
        }
        glBindImageTexture(0, textures[0], 0, false, 0, GL_READ_ONLY, GL_RGBA8UI);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error3: " + error);
        }

        // Get error if fails
        outputBuffer.position(0);
        // glGetTexImage is the slowest part of the program
        long startTimeCopy = System.nanoTime();

        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, outputBuffer);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Print the error
            System.out.println("Error4: " + error);
        }
        long endTime3 = System.nanoTime();
        long totalTimeCopy = endTime3 - startTimeCopy;
        //System.out.println("GPU->CPU copy time: " + totalTimeCopy / 1000000 + "ms");

        glBindTexture(GL_TEXTURE_2D, 0);
        // Copy the output to result array

        outputBuffer.get(result);

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