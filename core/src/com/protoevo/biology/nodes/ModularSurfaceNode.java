package com.protoevo.biology.nodes;

import com.protoevo.biology.cells.Cell;
import com.protoevo.biology.evolution.EvolvableFloat;
import com.protoevo.core.Statistics;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Enhanced modular surface node system that allows dynamic transformation
 * between different node types while preserving learned behavioral patterns.
 * 
 * This implements the key concept from the discussion where a cell that has
 * evolved to respond to light can have its photoreceptor mutate into a 
 * binding receptor, allowing the same behavioral patterns to be co-opted
 * for cell-to-cell communication.
 */
public class ModularSurfaceNode extends SurfaceNode implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // Behavioral pattern preservation
    private final float[] behavioralMemory = new float[3]; // Preserves learned IO patterns
    private final Map<Class<? extends NodeAttachment>, Float> affinities = new HashMap<>();
    
    // Transformation parameters
    private float transformationRate = 0.001f; // How easily this node can change type
    private float patternStability = 0.8f; // How much behavioral patterns are preserved
    private float mutationPressure = 0f; // External pressure to change type
    
    // Co-option tracking
    private Class<? extends NodeAttachment> previousAttachmentType = null;
    private float[] previousBehavioralPattern = new float[3];
    private int transformationCount = 0;
    
    public ModularSurfaceNode() {
        super();
        initializeAffinities();
    }
    
    private void initializeAffinities() {
        // Initialize random affinities for each possible attachment type
        if (NodeAttachment.possibleAttachments != null) {
            for (Class<? extends NodeAttachment> attachmentType : NodeAttachment.possibleAttachments) {
                affinities.put(attachmentType, (float) Math.random());
            }
        }
    }
    
    @Override
    public void update(float delta) {
        // Store current behavioral pattern before update
        storeBehavioralPattern();
        
        // Standard node update
        super.update(delta);
        
        // Check for transformation opportunities
        evaluateTransformation(delta);
        
        // Apply behavioral pattern preservation if transformation occurred
        applyBehavioralContinuity(delta);
    }
    
    private void storeBehavioralPattern() {
        // Store current IO patterns
        float[] input = getInputActivation();
        float[] output = getOutputActivation();
        
        for (int i = 0; i < Math.min(behavioralMemory.length, input.length); i++) {
            // Exponential moving average to track behavior over time
            behavioralMemory[i] = 0.9f * behavioralMemory[i] + 0.1f * input[i];
        }
        
        // If we have output, use it to influence pattern
        if (output.length > 0) {
            for (int i = 0; i < Math.min(behavioralMemory.length, output.length); i++) {
                behavioralMemory[i] = 0.5f * behavioralMemory[i] + 0.5f * output[i];
            }
        }
    }
    
    private void evaluateTransformation(float delta) {
        if (getAttachment() == null) return;
        
        Class<? extends NodeAttachment> currentType = getAttachment().getClass();
        
        // Calculate transformation probability based on various factors
        float transformationProb = calculateTransformationProbability(delta);
        
        if (Math.random() < transformationProb) {
            attemptTransformation(currentType);
        }
    }
    
    private float calculateTransformationProbability(float delta) {
        float baseProbability = transformationRate * delta;
        
        // Increase probability based on mutation pressure
        baseProbability *= (1f + mutationPressure);
        
        // Environmental factors could influence transformation
        Cell cell = getCell();
        if (cell != null) {
            // Stress factors (low energy, damage) could increase transformation
            float stressFactor = 1f;
            if (cell.getEnergyAvailable() < cell.getRadius() * 0.1f) {
                stressFactor += 0.5f; // Energy stress
            }
            
            baseProbability *= stressFactor;
        }
        
        // Limit maximum transformation rate
        return Math.min(0.01f, baseProbability);
    }
    
    private void attemptTransformation(Class<? extends NodeAttachment> currentType) {
        // Store current behavioral pattern
        System.arraycopy(behavioralMemory, 0, previousBehavioralPattern, 0, behavioralMemory.length);
        previousAttachmentType = currentType;
        
        // Choose new attachment type based on affinities and environmental factors
        Class<? extends NodeAttachment> newType = selectNewAttachmentType(currentType);
        
        if (newType != null && newType != currentType) {
            transformToNewType(newType);
        }
    }
    
    private Class<? extends NodeAttachment> selectNewAttachmentType(Class<? extends NodeAttachment> currentType) {
        if (NodeAttachment.possibleAttachments == null) return null;
        
        float maxScore = -1f;
        Class<? extends NodeAttachment> selectedType = null;
        
        for (Class<? extends NodeAttachment> candidateType : NodeAttachment.possibleAttachments) {
            if (candidateType == currentType) continue;
            
            float score = calculateTypeScore(candidateType);
            if (score > maxScore) {
                maxScore = score;
                selectedType = candidateType;
            }
        }
        
        return selectedType;
    }
    
    private float calculateTypeScore(Class<? extends NodeAttachment> candidateType) {
        float score = affinities.getOrDefault(candidateType, 0.5f);
        
        // Bonus for types that can utilize preserved behavioral patterns
        score += calculateBehavioralCompatibility(candidateType);
        
        // Environmental pressures can influence type selection
        score += calculateEnvironmentalFitness(candidateType);
        
        return score;
    }
    
    private float calculateBehavioralCompatibility(Class<? extends NodeAttachment> candidateType) {
        // This is where the co-option magic happens!
        // Types that can meaningfully use the preserved behavioral patterns get higher scores
        
        float compatibility = 0f;
        
        // If previous type was photoreceptor and new type is chemical signaling,
        // light-seeking behavior can be co-opted for chemical gradient following
        if (previousAttachmentType == Photoreceptor.class && 
            candidateType == ChemicalSignalingReceptor.class) {
            compatibility += 0.3f; // High compatibility for sensory->communication
        }
        
        // If previous type was flagellum and new type is chemical signaling,
        // movement patterns can be co-opted for signal emission rhythms
        if (previousAttachmentType == Flagellum.class && 
            candidateType == ChemicalSignalingReceptor.class) {
            compatibility += 0.2f;
        }
        
        // Adhesion to chemical signaling allows social behaviors
        if (previousAttachmentType == AdhesionReceptor.class && 
            candidateType == ChemicalSignalingReceptor.class) {
            compatibility += 0.4f; // Very high compatibility
        }
        
        // Add more co-option pathways as needed
        
        return compatibility;
    }
    
    private float calculateEnvironmentalFitness(Class<? extends NodeAttachment> candidateType) {
        // Environmental factors that might favor certain types
        Cell cell = getCell();
        if (cell == null) return 0f;
        
        float fitness = 0f;
        
        // If there are many nearby cells, communication becomes more valuable
        int nearbyCells = countNearbyCells();
        if (candidateType == ChemicalSignalingReceptor.class && nearbyCells > 2) {
            fitness += 0.2f * Math.min(1f, nearbyCells / 5f);
        }
        
        // If cell is isolated, sensory capabilities become more valuable
        if (candidateType == Photoreceptor.class && nearbyCells < 2) {
            fitness += 0.1f;
        }
        
        return fitness;
    }
    
    private int countNearbyCells() {
        // Simple approximation - count contacts that are other cells
        Cell cell = getCell();
        if (cell == null) return 0;
        
        return (int) cell.getParticle().getContacts().stream()
            .filter(contact -> {
                Object other = contact.getOther(cell.getParticle());
                return other != null && other.getClass().getSimpleName().contains("Cell");
            })
            .count();
    }
    
    private void transformToNewType(Class<? extends NodeAttachment> newType) {
        try {
            // Create new attachment of the target type
            NodeAttachment newAttachment = newType.getConstructor(SurfaceNode.class).newInstance(this);
            
            // Replace current attachment
            setAttachment(newAttachment);
            transformationCount++;
            
            // The behavioral continuity will be applied in the next update cycle
            
        } catch (Exception e) {
            System.err.println("Failed to transform node to type " + newType.getSimpleName() + ": " + e.getMessage());
        }
    }
    
    private void applyBehavioralContinuity(float delta) {
        // Apply preserved behavioral patterns to the new attachment type
        if (previousAttachmentType != null && getAttachment() != null) {
            
            // Gradually blend previous behavioral patterns with new ones
            float[] input = getInputActivation();
            for (int i = 0; i < Math.min(input.length, previousBehavioralPattern.length); i++) {
                // Apply pattern stability - preserved patterns influence current behavior
                float preservedInfluence = patternStability * previousBehavioralPattern[i];
                float currentInfluence = (1f - patternStability) * input[i];
                input[i] = preservedInfluence + currentInfluence;
            }
            
            // Gradually reduce the influence of previous patterns
            patternStability *= 0.999f; // Slowly decay over time
        }
    }
    
    @EvolvableFloat(name = "Transformation Rate", min = 0f, max = 0.01f)
    public void setTransformationRate(float rate) {
        this.transformationRate = rate;
    }
    
    @EvolvableFloat(name = "Pattern Stability", min = 0f, max = 1f)
    public void setPatternStability(float stability) {
        this.patternStability = stability;
    }
    
    public float getTransformationRate() {
        return transformationRate;
    }
    
    public float getPatternStability() {
        return patternStability;
    }
    
    public int getTransformationCount() {
        return transformationCount;
    }
    
    public Class<? extends NodeAttachment> getPreviousAttachmentType() {
        return previousAttachmentType;
    }
    
    @Override
    public Statistics getStats() {
        Statistics stats = super.getStats();
        
        stats.put("Transformation Rate", transformationRate);
        stats.put("Pattern Stability", patternStability);
        stats.putCount("Transformations", transformationCount);
        
        if (previousAttachmentType != null) {
            stats.put("Previous Type", previousAttachmentType.getSimpleName());
        }
        
        // Show behavioral memory
        for (int i = 0; i < behavioralMemory.length; i++) {
            stats.put("Behavioral Memory " + i, behavioralMemory[i]);
        }
        
        return stats;
    }
}