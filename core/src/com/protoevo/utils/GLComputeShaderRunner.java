package com.protoevo.utils;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static com.protoevo.test.lwjgldemo.IOUtils.ioResourceToByteBuffer;
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

public class GLComputeShaderRunner {

    static final int FILTER_SIZE = 3;

    private final int blockSizeX, blockSizeY;
    private final String kernelName, functionName;
    private int executions = 0;
    private long totalTime = 0;
    private int devicePixels, deviceOutput;
    private int program, computeShader;

    public static boolean openglAvailable() {
        try {
            new GLComputeShaderRunner("diffusion");
            return true;
        } catch (RuntimeException ignored) {
            if (DebugMode.isDebugMode())
                System.out.println("OpenGL is not available.");
            return false;
        }
    }

    public GLComputeShaderRunner(String kernelName) {
        this(kernelName, "kernel", 32, 32);
    }

    public GLComputeShaderRunner(String kernelName, String functionName, int blockSizeX, int blockSizeY) {
        this.blockSizeX = blockSizeX;
        this.blockSizeY = blockSizeY;
        this.kernelName = kernelName;
        this.functionName = functionName;

        initialise();
    }

    private void initialise() {
        // Create the compute shader
        int program = glCreateProgram();
        computeShader = glCreateShader(GL_COMPUTE_SHADER);

        // Load the shader source code
        String shaderSource = null;
        try {
            shaderSource = new String(Files.readAllBytes(Paths.get("shaders/compute/" + kernelName + ".cs.glsl")), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("Was unable to load " + kernelName + ":\n" + e);
        }

        // Compile the compute shader
        GL20.glShaderSource(computeShader, shaderSource);
        GL20.glCompileShader(computeShader);
        if (GL20.glGetShaderi(computeShader, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
            throw new RuntimeException("Failed to compile compute shader:\n" + GL20.glGetShaderInfoLog(computeShader));
        }

        // Create the program and attach the compute shader
        program = GL20.glCreateProgram();
        GL20.glAttachShader(program, computeShader);

        // Link the program
        GL20.glLinkProgram(program);
        if (GL20.glGetProgrami(program, GL20.GL_LINK_STATUS) == GL11.GL_FALSE) {
            throw new RuntimeException("Failed to link program:\n" + GL20.glGetProgramInfoLog(program));
        }

        // Detach the compute shader
        GL20.glDetachShader(program, computeShader);

        // Delete the compute shader
        GL20.glDeleteShader(computeShader);

        // Get the location of the input and output buffers
        devicePixels = GL43.glGetProgramResourceIndex(program, GL43.GL_SHADER_STORAGE_BLOCK, "devicePixels");
        deviceOutput = GL43.glGetProgramResourceIndex(program, GL43.GL_SHADER_STORAGE_BLOCK, "deviceOutput");

        // Bind the input and output buffers to the shader
        GL43.glShaderStorageBlockBinding(program, devicePixels, 0);
        GL43.glShaderStorageBlockBinding(program, deviceOutput, 1);
    }

    public byte[] processImage(byte[] pixels, int w, int h) {
        return processImage(pixels, pixels, w, h, 4);
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h) {
        return processImage(pixels, result, w, h, 4);
    }

    public byte[] processImage(byte[] pixels, byte[] result, int w, int h, int c) {
        // Benchmark the kernel
        long startTime = System.nanoTime();

        // Create the input and output buffers
        int pxLen = w * h * c;
        int resLen = w * h * c;
        int bufferSize = Math.max(pxLen, resLen);
        ByteBuffer inputBuffer = BufferUtils.createByteBuffer(bufferSize);
        ByteBuffer outputBuffer = BufferUtils.createByteBuffer(bufferSize);

        // Copy the input data to the input buffer
        inputBuffer.put(pixels);
        inputBuffer.flip();

        // Create the input and output buffer objects
        int[] buffers = new int[2];
        GL15.glGenBuffers(buffers);
        int inputBufferObject = buffers[0];
        int outputBufferObject = buffers[1];

        // Bind the input buffer object and upload the input data
        GL15.glBindBuffer(GL43.GL_SHADER_STORAGE_BUFFER, inputBufferObject);
        GL15.glBufferData(GL43.GL_SHADER_STORAGE_BUFFER, inputBuffer, GL15.GL_STATIC_DRAW);

        // Bind the output buffer object and allocate the output data
        GL15.glBindBuffer(GL43.GL_SHADER_STORAGE_BUFFER, outputBufferObject);
        GL15.glBufferData(GL43.GL_SHADER_STORAGE_BUFFER, outputBuffer.capacity(), GL15.GL_STATIC_DRAW);

        // Bind the program and set the input and output buffer bindings
        GL20.glUseProgram(program);
        GL30.glBindBufferBase(GL43.GL_SHADER_STORAGE_BUFFER, 0, inputBufferObject);
        GL30.glBindBufferBase(GL43.GL_SHADER_STORAGE_BUFFER, 1, outputBufferObject);

        // Set the kernel parameters
        GL20.glUniform1i(GL20.glGetUniformLocation(program, "w"), w);
        GL20.glUniform1i(GL20.glGetUniformLocation(program, "h"), h);
        GL20.glUniform1i(GL20.glGetUniformLocation(program, "c"), c);

        // Dispatch the compute shader
        int gridSizeX = (int) Math.ceil((double) w / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) h / blockSizeY);
        GL43.glDispatchCompute(gridSizeX, gridSizeY, 1);

        // Wait for the compute shader to finish
        GL42.glMemoryBarrier(GL43.GL_SHADER_STORAGE_BARRIER_BIT);

        // Read the output data from the output buffer
        GL15.glBindBuffer(GL43.GL_SHADER_STORAGE_BUFFER, outputBufferObject);
        GL15.glGetBufferSubData(GL43.GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

        // Copy the output data to the result array
        outputBuffer.get(result);
        outputBuffer.flip();

        // Delete the input and output buffer objects
        GL15.glDeleteBuffers(buffers);

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