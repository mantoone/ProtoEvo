package com.protoevo.biology.nodes;

import com.badlogic.gdx.math.Vector2;
import com.protoevo.biology.cells.Cell;
import com.protoevo.biology.cells.Protozoan;
import com.protoevo.core.Statistics;
import com.protoevo.env.Environment;
import com.protoevo.physics.Particle;
import com.protoevo.physics.Collision;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Enhanced communication node that can send and receive chemical signals (pheromones)
 * between cells, enabling complex multi-cellular behaviors and coordination.
 * 
 * This implements the modular surface node system discussed in the communication 
 * feature specification, allowing evolved behaviors to be co-opted for cell communication.
 */
public class ChemicalSignalingReceptor extends NodeAttachment implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // Chemical signal types - expandable system
    public enum SignalType {
        PHEROMONE_A(0), // General purpose signaling
        PHEROMONE_B(1), // Secondary signaling
        ALARM_SIGNAL(2), // Emergency/danger communication
        MATING_SIGNAL(3), // Reproductive coordination
        FOOD_SIGNAL(4),  // Resource location sharing
        TERRITORY_SIGNAL(5), // Spatial coordination
        MORPHOGEN_A(6),  // Development/differentiation signals
        MORPHOGEN_B(7);  // Additional development signals
        
        private final int id;
        SignalType(int id) { this.id = id; }
        public int getId() { return id; }
    }
    
    private final Map<SignalType, Float> chemicalConcentrations = new HashMap<>();
    private final Map<SignalType, Float> signalStrengths = new HashMap<>();
    private final Vector2 tmp = new Vector2();
    
    // Communication parameters
    private float transmissionRange = 50f; // Base transmission range
    private float signalDecayRate = 0.95f; // How quickly signals fade
    private float receptorSensitivity = 1.0f; // How sensitive this receptor is
    private SignalType activeSignalType = SignalType.PHEROMONE_A;
    
    // Signal memory for pattern recognition
    private final float[] signalHistory = new float[10]; // Last 10 signal readings
    private int historyIndex = 0;
    
    public ChemicalSignalingReceptor() {
        super(null);
        initializeSignaling();
    }
    
    public ChemicalSignalingReceptor(SurfaceNode node) {
        super(node);
        initializeSignaling();
    }
    
    private void initializeSignaling() {
        // Initialize all signal types to zero concentration
        for (SignalType type : SignalType.values()) {
            chemicalConcentrations.put(type, 0f);
            signalStrengths.put(type, 0f);
        }
    }
    
    @Override
    public void update(float delta, float[] input, float[] output) {
        // Input interpretation:
        // input[0] = signal strength to emit (-1 to 1)
        // input[1] = signal type selector (0 to 1, maps to SignalType)
        // input[2] = receptor sensitivity modifier (-1 to 1)
        
        updateActiveSignalType(input);
        updateReceptorSensitivity(input);
        
        // Emit signals based on input
        if (input.length > 0) {
            float emissionStrength = Math.max(0, input[0]); // Only emit positive signals
            emitChemicalSignal(activeSignalType, emissionStrength, delta);
        }
        
        // Receive and process signals from nearby cells
        processIncomingSignals(delta);
        
        // Update signal decay
        decaySignals(delta);
        
        // Output interpretation:
        // output[0] = strongest received signal strength
        // output[1] = signal type of strongest signal (normalized)
        // output[2] = signal pattern recognition (temporal changes)
        
        populateOutput(output);
    }
    
    private void updateActiveSignalType(float[] input) {
        if (input.length > 1) {
            // Map input[1] from [-1, 1] to signal type index
            float normalized = (input[1] + 1) / 2f; // Convert to [0, 1]
            int typeIndex = (int) (normalized * SignalType.values().length);
            typeIndex = Math.max(0, Math.min(SignalType.values().length - 1, typeIndex));
            activeSignalType = SignalType.values()[typeIndex];
        }
    }
    
    private void updateReceptorSensitivity(float[] input) {
        if (input.length > 2) {
            // Modify receptor sensitivity based on input[2]
            receptorSensitivity = 0.5f + (input[2] + 1) / 4f; // Range [0.25, 1.75]
        }
    }
    
    private void emitChemicalSignal(SignalType signalType, float strength, float delta) {
        if (strength <= 0) return;
        
        Cell cell = node.getCell();
        float energyCost = strength * 0.01f * delta; // Small energy cost for signaling
        
        if (cell.getEnergyAvailable() < energyCost) return;
        
        cell.depleteEnergy(energyCost);
        
        // Add to local signal strength
        float currentStrength = signalStrengths.getOrDefault(signalType, 0f);
        signalStrengths.put(signalType, currentStrength + strength);
        
        // Propagate signal to nearby cells
        propagateSignalToNearbyCells(signalType, strength, delta);
    }
    
    private void propagateSignalToNearbyCells(SignalType signalType, float strength, float delta) {
        Cell cell = node.getCell();
        Vector2 nodePos = node.getWorldPosition();
        
        // Find cells within transmission range
        for (Collision contact : cell.getParticle().getContacts()) {
            Object other = contact.getOther(cell.getParticle());
            if (other instanceof Particle) {
                Particle otherParticle = (Particle) other;
                if (otherParticle.getUserData() instanceof Protozoan) {
                    Protozoan otherCell = otherParticle.getUserData(Protozoan.class);
                    
                    // Check if within transmission range
                    float distance = tmp.set(otherCell.getPos()).sub(nodePos).len();
                    if (distance <= transmissionRange * getConstructionProgress()) {
                        
                        // Deliver signal to chemical receptors on other cell
                        deliverSignalToCell(otherCell, signalType, strength, distance);
                    }
                }
            }
        }
        
        // Also check environment for cells within range (broader search)
        if (cell.getEnv().isPresent()) {
            Environment env = cell.getEnv().get();
            // This would require access to environment's cell collection
            // For now, rely on contact-based propagation
        }
    }
    
    private void deliverSignalToCell(Protozoan targetCell, SignalType signalType, float strength, float distance) {
        // Find chemical signaling receptors on target cell
        for (SurfaceNode targetNode : targetCell.getSurfaceNodes()) {
            if (targetNode.exists() && targetNode.getAttachment() instanceof ChemicalSignalingReceptor) {
                ChemicalSignalingReceptor targetReceptor = (ChemicalSignalingReceptor) targetNode.getAttachment();
                
                // Apply distance-based attenuation
                float attenuatedStrength = strength * Math.max(0.1f, 1f - (distance / transmissionRange));
                targetReceptor.receiveChemicalSignal(signalType, attenuatedStrength);
            }
        }
    }
    
    private void receiveChemicalSignal(SignalType signalType, float strength) {
        float currentConcentration = chemicalConcentrations.getOrDefault(signalType, 0f);
        float adjustedStrength = strength * receptorSensitivity;
        chemicalConcentrations.put(signalType, currentConcentration + adjustedStrength);
    }
    
    private void processIncomingSignals(float delta) {
        // Process chemical concentrations and update signal history
        float strongestSignal = 0f;
        SignalType strongestType = SignalType.PHEROMONE_A;
        
        for (Map.Entry<SignalType, Float> entry : chemicalConcentrations.entrySet()) {
            if (entry.getValue() > strongestSignal) {
                strongestSignal = entry.getValue();
                strongestType = entry.getKey();
            }
        }
        
        // Update signal history for pattern recognition
        signalHistory[historyIndex] = strongestSignal;
        historyIndex = (historyIndex + 1) % signalHistory.length;
    }
    
    private void decaySignals(float delta) {
        // Decay all signal concentrations and strengths
        for (SignalType type : SignalType.values()) {
            float concentration = chemicalConcentrations.getOrDefault(type, 0f);
            float strength = signalStrengths.getOrDefault(type, 0f);
            
            chemicalConcentrations.put(type, concentration * signalDecayRate);
            signalStrengths.put(type, strength * signalDecayRate);
        }
    }
    
    private void populateOutput(float[] output) {
        if (output.length == 0) return;
        
        // Find strongest received signal
        float strongestSignal = 0f;
        SignalType strongestType = SignalType.PHEROMONE_A;
        
        for (Map.Entry<SignalType, Float> entry : chemicalConcentrations.entrySet()) {
            if (entry.getValue() > strongestSignal) {
                strongestSignal = entry.getValue();
                strongestType = entry.getKey();
            }
        }
        
        // Output[0]: Strongest signal strength (normalized)
        output[0] = Math.min(1f, strongestSignal);
        
        if (output.length > 1) {
            // Output[1]: Signal type (normalized)
            output[1] = (strongestType.getId() * 2f / SignalType.values().length) - 1f; // Map to [-1, 1]
        }
        
        if (output.length > 2) {
            // Output[2]: Pattern recognition - detect signal changes over time
            float patternSignal = calculatePatternRecognition();
            output[2] = Math.max(-1f, Math.min(1f, patternSignal));
        }
    }
    
    private float calculatePatternRecognition() {
        if (signalHistory.length < 3) return 0f;
        
        // Calculate trend in signal strength
        float recent = 0f, older = 0f;
        int halfLength = signalHistory.length / 2;
        
        // Average recent signals vs older signals
        for (int i = 0; i < halfLength; i++) {
            older += signalHistory[i];
        }
        for (int i = halfLength; i < signalHistory.length; i++) {
            recent += signalHistory[i];
        }
        
        recent /= halfLength;
        older /= halfLength;
        
        // Return normalized difference (positive = increasing, negative = decreasing)
        return (recent - older) * 2f;
    }
    
    @Override
    public float getInteractionRange() {
        return transmissionRange * getConstructionProgress();
    }
    
    @Override
    public String getName() {
        return "Chemical Signaling Receptor";
    }
    
    @Override
    public String getInputMeaning(int index) {
        switch (index) {
            case 0: return "Signal Emission Strength";
            case 1: return "Signal Type Selector";
            case 2: return "Receptor Sensitivity";
            default: return null;
        }
    }
    
    @Override
    public String getOutputMeaning(int index) {
        switch (index) {
            case 0: return "Received Signal Strength";
            case 1: return "Received Signal Type";
            case 2: return "Signal Pattern Change";
            default: return null;
        }
    }
    
    @Override
    public void addStats(Statistics stats) {
        stats.put("Active Signal Type", activeSignalType.name());
        stats.put("Receptor Sensitivity", receptorSensitivity);
        stats.putDistance("Transmission Range", transmissionRange);
        
        float totalEmitted = signalStrengths.values().stream().reduce(0f, Float::sum);
        float totalReceived = chemicalConcentrations.values().stream().reduce(0f, Float::sum);
        
        stats.put("Total Signals Emitted", totalEmitted);
        stats.put("Total Signals Received", totalReceived);
        
        // Show strongest active signal
        SignalType strongestType = SignalType.PHEROMONE_A;
        float strongestSignal = 0f;
        for (Map.Entry<SignalType, Float> entry : chemicalConcentrations.entrySet()) {
            if (entry.getValue() > strongestSignal) {
                strongestSignal = entry.getValue();
                strongestType = entry.getKey();
            }
        }
        
        if (strongestSignal > 0.01f) {
            stats.put("Strongest Received Signal", strongestType.name() + " (" + 
                     String.format("%.3f", strongestSignal) + ")");
        }
    }
}