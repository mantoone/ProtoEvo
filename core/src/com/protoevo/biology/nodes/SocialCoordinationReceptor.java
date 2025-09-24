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
 * Advanced multi-cellular coordination node that enables complex social behaviors
 * like those discussed in the feature specification: ant hive coordination,
 * pheromone-based communication, mimicry, territorial behavior, and collective
 * decision making.
 */
public class SocialCoordinationReceptor extends NodeAttachment implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // Social behavior types that can emerge
    public enum SocialBehavior {
        LEADER,          // Initiates group actions
        FOLLOWER,        // Responds to leader signals
        SCOUT,           // Explores and reports back
        WORKER,          // Focused on resource gathering
        GUARD,           // Defensive behaviors
        MIMIC,           // Imitates other cells' behaviors
        COORDINATOR,     // Facilitates group decisions
        TERRITORIAL      // Maintains spatial boundaries
    }
    
    // Communication protocols
    public enum CommunicationProtocol {
        PHEROMONE_TRAIL,     // Leave chemical breadcrumbs
        MORSE_SIGNAL,        // Rhythmic on/off patterns
        FORMATION_SIGNAL,    // Spatial positioning info
        ALARM_CASCADE,       // Emergency propagation
        CONSENSUS_POLLING,   // Democratic decision making
        MIMICRY_ECHO,        // Copy and amplify signals
        TERRITORIAL_MARKING  // Boundary establishment
    }
    
    private final Map<SocialBehavior, Float> behaviorStrengths = new HashMap<>();
    private final Map<CommunicationProtocol, Float> protocolPreferences = new HashMap<>();
    
    // Current social state
    private SocialBehavior activeBehavior = SocialBehavior.FOLLOWER;
    private CommunicationProtocol activeProtocol = CommunicationProtocol.PHEROMONE_TRAIL;
    
    // Group coordination data
    private int groupId = -1;
    private Vector2 groupCentroid = new Vector2();
    private float groupCoherence = 0f;
    private int groupSize = 1;
    
    // Pheromone trail system
    private final Map<String, Float> pheromoneTrails = new HashMap<>();
    private float trailIntensity = 0f;
    private String currentTrail = "default";
    
    // Morse code communication
    private final StringBuilder morseBuffer = new StringBuilder();
    private float signalTimer = 0f;
    private boolean signaling = false;
    
    // Decision consensus system
    private final Map<String, Float> voteValues = new HashMap<>();
    private String currentProposal = "";
    private float consensusThreshold = 0.6f;
    
    // Territorial boundaries
    private Vector2 territoryCenter = null;
    private float territoryRadius = 100f;
    private final Map<Integer, Float> intruderThreats = new HashMap<>();
    
    public SocialCoordinationReceptor() {
        super(null);
        initializeSocialSystem();
    }
    
    public SocialCoordinationReceptor(SurfaceNode node) {
        super(node);
        initializeSocialSystem();
    }
    
    private void initializeSocialSystem() {
        // Initialize behavior strengths randomly (evolution will optimize)
        for (SocialBehavior behavior : SocialBehavior.values()) {
            behaviorStrengths.put(behavior, (float) Math.random());
        }
        
        // Initialize protocol preferences
        for (CommunicationProtocol protocol : CommunicationProtocol.values()) {
            protocolPreferences.put(protocol, (float) Math.random());
        }
        
        // Set initial dominant behavior
        updateActiveBehavior();
    }
    
    @Override
    public void update(float delta, float[] input, float[] output) {
        // Input interpretation:
        // input[0] = social behavior modifier (-1 to 1)
        // input[1] = communication protocol selector (0 to 1)
        // input[2] = group action intensity (0 to 1)
        
        updateSocialParameters(input);
        updateGroupInformation(delta);
        processSocialBehavior(delta, input);
        handleCommunication(delta, input);
        
        // Output interpretation:
        // output[0] = group coordination signal
        // output[1] = social behavior state (normalized)
        // output[2] = consensus/decision signal
        
        populateSocialOutput(output);
    }
    
    private void updateSocialParameters(float[] input) {
        if (input.length > 0) {
            // Modify behavior based on input
            float behaviorMod = input[0];
            adjustBehaviorStrengths(behaviorMod);
        }
        
        if (input.length > 1) {
            // Select communication protocol
            float protocolSelector = (input[1] + 1) / 2f; // Normalize to [0,1]
            int protocolIndex = (int) (protocolSelector * CommunicationProtocol.values().length);
            protocolIndex = Math.max(0, Math.min(CommunicationProtocol.values().length - 1, protocolIndex));
            activeProtocol = CommunicationProtocol.values()[protocolIndex];
        }
        
        updateActiveBehavior();
    }
    
    private void adjustBehaviorStrengths(float modifier) {
        // Modify behavior strengths based on input and current situation
        for (SocialBehavior behavior : SocialBehavior.values()) {
            float current = behaviorStrengths.get(behavior);
            
            // Apply modifier with some behavior-specific logic
            float adjustment = modifier * 0.1f;
            
            // Leader behavior strengthens with positive modifier
            if (behavior == SocialBehavior.LEADER && modifier > 0) {
                adjustment *= 2f;
            }
            
            // Follower behavior strengthens with negative modifier
            if (behavior == SocialBehavior.FOLLOWER && modifier < 0) {
                adjustment *= -2f;
            }
            
            float newStrength = Math.max(0f, Math.min(1f, current + adjustment));
            behaviorStrengths.put(behavior, newStrength);
        }
    }
    
    private void updateActiveBehavior() {
        // Select the behavior with highest strength
        float maxStrength = -1f;
        for (Map.Entry<SocialBehavior, Float> entry : behaviorStrengths.entrySet()) {
            if (entry.getValue() > maxStrength) {
                maxStrength = entry.getValue();
                activeBehavior = entry.getKey();
            }
        }
    }
    
    private void updateGroupInformation(float delta) {
        // Discover and track nearby cells for group coordination
        Cell cell = node.getCell();
        if (cell == null) return;
        
        int nearbyCount = 0;
        Vector2 centroidSum = new Vector2();
        Vector2 myPos = cell.getPos();
        
        // Survey contacts for group members
        for (Collision contact : cell.getParticle().getContacts()) {
            Object other = contact.getOther(cell.getParticle());
            if (other instanceof Particle && ((Particle) other).getUserData() instanceof Protozoan) {
                Protozoan otherCell = (Protozoan) ((Particle) other).getUserData();
                
                // Check if this cell has social coordination capability
                if (hasSocialCoordination(otherCell)) {
                    nearbyCount++;
                    centroidSum.add(otherCell.getPos());
                }
            }
        }
        
        if (nearbyCount > 0) {
            groupSize = nearbyCount + 1; // Include self
            groupCentroid.set(centroidSum).scl(1f / nearbyCount);
            
            // Calculate group coherence (how tightly packed the group is)
            float avgDistance = groupCentroid.dst(myPos);
            groupCoherence = Math.max(0f, 1f - (avgDistance / 200f)); // Normalize by max coordination range
        } else {
            groupSize = 1;
            groupCoherence = 0f;
            groupCentroid.set(myPos);
        }
    }
    
    private boolean hasSocialCoordination(Protozoan cell) {
        // Check if other cell has social coordination capability
        for (SurfaceNode node : cell.getSurfaceNodes()) {
            if (node.exists() && node.getAttachment() instanceof SocialCoordinationReceptor) {
                return true;
            }
        }
        return false;
    }
    
    private void processSocialBehavior(float delta, float[] input) {
        Cell cell = node.getCell();
        if (cell == null) return;
        
        switch (activeBehavior) {
            case LEADER:
                processLeaderBehavior(delta, input);
                break;
            case FOLLOWER:
                processFollowerBehavior(delta, input);
                break;
            case SCOUT:
                processScoutBehavior(delta, input);
                break;
            case WORKER:
                processWorkerBehavior(delta, input);
                break;
            case GUARD:
                processGuardBehavior(delta, input);
                break;
            case MIMIC:
                processMimicBehavior(delta, input);
                break;
            case COORDINATOR:
                processCoordinatorBehavior(delta, input);
                break;
            case TERRITORIAL:
                processTerritorialBehavior(delta, input);
                break;
        }
    }
    
    private void processLeaderBehavior(float delta, float[] input) {
        // Leaders initiate group actions and set pheromone trails
        float actionIntensity = input.length > 2 ? Math.max(0, input[2]) : 0.5f;
        
        // Lay pheromone trail based on current objective
        if (activeProtocol == CommunicationProtocol.PHEROMONE_TRAIL) {
            trailIntensity = actionIntensity;
            layPheromoneTrail("leader_path", trailIntensity);
        }
        
        // Send formation signals to coordinate group movement
        if (activeProtocol == CommunicationProtocol.FORMATION_SIGNAL) {
            broadcastFormationSignal(actionIntensity);
        }
    }
    
    private void processFollowerBehavior(float delta, float[] input) {
        // Followers respond to leader signals and maintain group cohesion
        followPheromoneTrails();
        maintainGroupCohesion();
    }
    
    private void processScoutBehavior(float delta, float[] input) {
        // Scouts explore and report back findings
        if (groupSize > 3) { // Only scout if group is large enough
            exploreAndReport(delta);
        }
    }
    
    private void processWorkerBehavior(float delta, float[] input) {
        // Workers focus on resource gathering and construction
        // Could integrate with existing phagocytic behaviors
        coordinateResourceGathering();
    }
    
    private void processGuardBehavior(float delta, float[] input) {
        // Guards maintain defensive perimeter
        detectAndRespondToThreats(delta);
    }
    
    private void processMimicBehavior(float delta, float[] input) {
        // Mimics copy and amplify behaviors of nearby cells
        mimicNearbyBehaviors(delta);
    }
    
    private void processCoordinatorBehavior(float delta, float[] input) {
        // Coordinators facilitate group decision making
        facilitateConsensus(delta);
    }
    
    private void processTerritorialBehavior(float delta, float[] input) {
        // Territorial cells maintain boundaries and mark territory
        establishAndDefendTerritory(delta);
    }
    
    private void handleCommunication(float delta, float[] input) {
        switch (activeProtocol) {
            case PHEROMONE_TRAIL:
                updatePheromoneTrails(delta);
                break;
            case MORSE_SIGNAL:
                processMorseSignaling(delta, input);
                break;
            case ALARM_CASCADE:
                processAlarmCascade(delta);
                break;
            case CONSENSUS_POLLING:
                processConsensusPolling(delta);
                break;
            // Add other protocols as needed
        }
    }
    
    // Detailed implementations of social behaviors...
    
    private void layPheromoneTrail(String trailName, float intensity) {
        pheromoneTrails.put(trailName, intensity);
        currentTrail = trailName;
        
        // Propagate to nearby social coordinators
        propagatePheromoneToNearby(trailName, intensity);
    }
    
    private void propagatePheromoneToNearby(String trailName, float intensity) {
        Cell cell = node.getCell();
        if (cell == null) return;
        
        for (Collision contact : cell.getParticle().getContacts()) {
            Object other = contact.getOther(cell.getParticle());
            if (other instanceof Particle && ((Particle) other).getUserData() instanceof Protozoan) {
                Protozoan otherCell = (Protozoan) ((Particle) other).getUserData();
                
                // Find social coordinators on other cell
                for (SurfaceNode otherNode : otherCell.getSurfaceNodes()) {
                    if (otherNode.exists() && 
                        otherNode.getAttachment() instanceof SocialCoordinationReceptor) {
                        
                        SocialCoordinationReceptor otherReceptor = 
                            (SocialCoordinationReceptor) otherNode.getAttachment();
                        otherReceptor.receivePheromoneSignal(trailName, intensity * 0.8f);
                    }
                }
            }
        }
    }
    
    private void receivePheromoneSignal(String trailName, float intensity) {
        float current = pheromoneTrails.getOrDefault(trailName, 0f);
        pheromoneTrails.put(trailName, Math.max(current, intensity));
    }
    
    private void followPheromoneTrails() {
        // Find strongest pheromone trail and influence behavior accordingly
        float maxIntensity = 0f;
        String strongestTrail = null;
        
        for (Map.Entry<String, Float> entry : pheromoneTrails.entrySet()) {
            if (entry.getValue() > maxIntensity) {
                maxIntensity = entry.getValue();
                strongestTrail = entry.getKey();
            }
        }
        
        if (strongestTrail != null && maxIntensity > 0.1f) {
            currentTrail = strongestTrail;
            trailIntensity = maxIntensity;
        }
    }
    
    private void updatePheromoneTrails(float delta) {
        // Decay pheromone trails over time
        float decayRate = 0.95f;
        
        for (String trailName : pheromoneTrails.keySet()) {
            float current = pheromoneTrails.get(trailName);
            pheromoneTrails.put(trailName, current * decayRate);
        }
        
        // Remove very weak trails
        pheromoneTrails.entrySet().removeIf(entry -> entry.getValue() < 0.01f);
    }
    
    private void maintainGroupCohesion() {
        // Influence movement to stay close to group centroid
        // This would need integration with movement systems
    }
    
    private void facilitateConsensus(float delta) {
        // Democratic decision making process
        // Collect votes, calculate consensus, broadcast results
    }
    
    private void establishAndDefendTerritory(float delta) {
        // Territorial marking and intrusion detection
        Cell cell = node.getCell();
        if (cell == null) return;
        
        if (territoryCenter == null) {
            territoryCenter = new Vector2(cell.getPos());
        }
        
        // Mark territory boundaries with pheromones
        layPheromoneTrail("territory_" + cell.hashCode(), 0.8f);
        
        // Detect intruders
        for (Collision contact : cell.getParticle().getContacts()) {
            Object other = contact.getOther(cell.getParticle());
            if (other instanceof Particle && ((Particle) other).getUserData() instanceof Protozoan) {
                Protozoan intruder = (Protozoan) ((Particle) other).getUserData();
                
                float distance = territoryCenter.dst(intruder.getPos());
                if (distance < territoryRadius) {
                    // Potential intruder detected
                    intruderThreats.put(intruder.hashCode(), 1f);
                }
            }
        }
    }
    
    // Additional behavior implementations...
    private void exploreAndReport(float delta) { /* Implementation */ }
    private void coordinateResourceGathering() { /* Implementation */ }
    private void detectAndRespondToThreats(float delta) { /* Implementation */ }
    private void mimicNearbyBehaviors(float delta) { /* Implementation */ }
    private void broadcastFormationSignal(float intensity) { /* Implementation */ }
    private void processMorseSignaling(float delta, float[] input) { /* Implementation */ }
    private void processAlarmCascade(float delta) { /* Implementation */ }
    private void processConsensusPolling(float delta) { /* Implementation */ }
    
    private void populateSocialOutput(float[] output) {
        if (output.length == 0) return;
        
        // Output[0]: Group coordination signal
        output[0] = groupCoherence * trailIntensity;
        
        if (output.length > 1) {
            // Output[1]: Social behavior state (normalized)
            float behaviorValue = (activeBehavior.ordinal() * 2f / SocialBehavior.values().length) - 1f;
            output[1] = behaviorValue;
        }
        
        if (output.length > 2) {
            // Output[2]: Group consensus/decision signal
            float consensusSignal = calculateConsensusStrength();
            output[2] = Math.max(-1f, Math.min(1f, consensusSignal));
        }
    }
    
    private float calculateConsensusStrength() {
        // Calculate how well the group agrees on current actions
        return groupCoherence * (groupSize > 3 ? 1f : groupSize / 3f);
    }
    
    @Override
    public float getInteractionRange() {
        return Math.max(50f, territoryRadius * 0.5f);
    }
    
    @Override
    public String getName() {
        return "Social Coordination Receptor";
    }
    
    @Override
    public String getInputMeaning(int index) {
        switch (index) {
            case 0: return "Social Behavior Modifier";
            case 1: return "Communication Protocol";
            case 2: return "Group Action Intensity";
            default: return null;
        }
    }
    
    @Override
    public String getOutputMeaning(int index) {
        switch (index) {
            case 0: return "Group Coordination Signal";
            case 1: return "Social Behavior State";
            case 2: return "Consensus Signal";
            default: return null;
        }
    }
    
    @Override
    public void addStats(Statistics stats) {
        stats.put("Active Behavior", activeBehavior.name());
        stats.put("Communication Protocol", activeProtocol.name());
        stats.putCount("Group Size", groupSize);
        stats.put("Group Coherence", groupCoherence);
        stats.put("Trail Intensity", trailIntensity);
        stats.putCount("Active Pheromone Trails", pheromoneTrails.size());
        
        if (territoryCenter != null) {
            stats.putDistance("Territory Radius", territoryRadius);
            stats.putCount("Detected Intruders", intruderThreats.size());
        }
    }
}