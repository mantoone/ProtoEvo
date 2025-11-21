package com.protoevo.test;

import com.protoevo.env.Environment;
import com.protoevo.env.FluidField;
import com.protoevo.settings.SimulationSettings;

public class FluidTest {

    public static void main(String[] args) {
        System.out.println("Starting FluidTest...");
        System.out.println("CWD: " + System.getProperty("user.dir"));
        java.io.File cwd = new java.io.File(".");
        System.out.println("Files in CWD:");
        for (java.io.File f : cwd.listFiles()) {
            System.out.println(" - " + f.getName());
        }
        
        // Enable CUDA
        Environment.settings = SimulationSettings.createDefault();
        Environment.settings.misc.useCUDA.set(true);
        Environment.settings.worldgen.chemicalFieldResolution.set(128);
        
        int size = Environment.settings.worldgen.chemicalFieldResolution.get();
        System.out.println("Creating FluidField with size: " + size);
        FluidField fluidField = new FluidField(size, size);
        
        // Inject velocity
        System.out.println("Injecting velocity at center...");
        fluidField.injectVelocity(size/2, size/2, 10.0f, 0.0f);
        
        // Step
        System.out.println("Stepping simulation...");
        try {
            for (int i = 0; i < 100; i++) {
                fluidField.step(0.1f);
                if (i % 10 == 0) {
                    System.out.println("Step " + i);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        System.out.println("FluidTest completed successfully.");
        fluidField.dispose();
    }
}
