package com.protoevo.settings.legacy;

public class LegacyWorldGenerationSettings {
    public static final int worldGenerationSeed = 1;
    public static final int numInitialProtozoa = 500;
    public static final int numInitialPlantPellets = 2000;
    public static final int numRingClusters = 4;
    public static final int numPopulationStartClusters = 3;
    public static final float rockClusterRadius = 1f;
    public static final float environmentRadius = 10.0f;
    public static final float populationClusterRadius = environmentRadius / 2f;
    public static final float maxRockSize = LegacySimulationSettings.maxParticleRadius * 2.5f;
    public static final float minRockSize = maxRockSize / 5f;
    public static final float attachedRockSizeChange = 0.4f;
    public static final float minRockSpikiness = (float) Math.toRadians(5);
    public static final float minRockOpeningSize = maxRockSize * 0.8f;
    public static final int rockGenerationIterations = 200;
    public static final float rockClustering = 0.95f;
    public static final float ringBreakProbability = 0.05f;
    public static final float ringBreakAngleMinSkip = (float) Math.toRadians(8);
    public static final float ringBreakAngleMaxSkip = (float) Math.toRadians(15);
}
