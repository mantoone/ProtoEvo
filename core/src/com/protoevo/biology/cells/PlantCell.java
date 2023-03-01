package com.protoevo.biology.cells;

import com.badlogic.gdx.math.MathUtils;
import com.protoevo.biology.CauseOfDeath;
import com.protoevo.core.Statistics;
import com.protoevo.physics.CollisionHandler;
import com.protoevo.env.Environment;
import com.protoevo.env.JointsManager;
import com.protoevo.settings.PlantSettings;
import com.protoevo.settings.Settings;
import com.protoevo.settings.SimulationSettings;
import com.protoevo.utils.Colour;

public class PlantCell extends Cell {
    public static final long serialVersionUID = -3975433688803760076L;

    private final float maxRadius;
    private float crowdingFactor;

    public PlantCell(float radius, Environment environment) {
        super();
        setRadius(Math.max(radius, PlantSettings.minPlantBirthRadius));
        addToEnv(environment);
        setGrowthRate(MathUtils.random(
                PlantSettings.minPlantGrowth,
                PlantSettings.maxPlantGrowth));

        maxRadius = MathUtils.random(
                1.5f * SimulationSettings.minParticleRadius,
                SimulationSettings.maxParticleRadius / 2f);

        float darken = 0.9f;
        setHealthyColour(new Colour(
                darken * (30 + MathUtils.random(105)) / 255f,
                darken * (150 + MathUtils.random(100)) / 255f,
                darken * (10 + MathUtils.random(100)) / 255f,
                1f)
        );

        setDegradedColour(degradeColour(getHealthyColour(), 0.5f));
    }

    @Override
    public float getMaxRadius() {
        return maxRadius;
    }

    @Override
    public boolean isEdible() {
        return true;
    }

    private static float randomPlantRadius() {
        float range = PlantSettings.maxPlantBirthRadius * .5f - PlantSettings.minPlantBirthRadius;
        return PlantSettings.minPlantBirthRadius + range * MathUtils.random();
    }

    public PlantCell(Environment environment) {
        this(randomPlantRadius(), environment);
    }

    private boolean shouldSplit() {
        return hasNotBurst() && getRadius() >= 0.99f * maxRadius &&
                getHealth() > Settings.minHealthToSplit;
    }

    public float getCrowdingFactor() {
        return crowdingFactor;
    }


    @Override
    public void update(float delta) {
        super.update(delta);

        if (isDead())
            return;

        addConstructionMass(delta * Settings.plantConstructionRate);
        addAvailableEnergy(delta * Settings.plantEnergyRate);

        int nProtozoaContacts = 0;
        for (CollisionHandler.Collision collision : getContacts()) {
            Object o = getOther(collision);
            if (o instanceof Protozoan)
                nProtozoaContacts++;
        }
        if (nProtozoaContacts >= 1) {
            float dps = PlantSettings.collisionDestructionRate * nProtozoaContacts;
            removeMass(delta * dps, CauseOfDeath.OVERCROWDING);
        }

        handleAttachments();

        if (shouldSplit()) {
            getEnv().requestBurst(
                this, PlantCell.class, this::createChild
            );
        }
    }

    @Override
    protected float getVoidStartDistance2() {
        return super.getVoidStartDistance2() * 0.9f * 0.9f;
    }

    public PlantCell createChild(float r) {
        return new PlantCell(r, getEnv());
    }

    public void attach(Cell otherCell) {
        JointsManager.Joining joining = new JointsManager.Joining(this, otherCell);
        JointsManager jointsManager = getEnv().getJointsManager();
        jointsManager.createJoint(joining);
        registerJoining(joining);
        otherCell.registerJoining(joining);
    }

    public void handleAttachments() {
        if (getNumAttachedCells() < 2) {
            for (CollisionHandler.Collision collision : getContacts()) {
                Object o = getOther(collision);
                if (o instanceof PlantCell
                        && ((PlantCell) o).getNumAttachedCells() < 2
                        && !isAttachedTo((PlantCell) o)) {
                    attach((PlantCell) o);
                    if (getNumAttachedCells() >= 2)
                        break;
                }
            }
        }
    }

    /**
     * <a href="https://www.desmos.com/calculator/hmhjwdk0jc">Desmos Graph</a>
     * @return The growth rate based on the crowding and current radius.
     */
//    @Override
//    public float getGrowthRate() {
//        float x = (-getCrowdingFactor() + PlantSettings.plantCriticalCrowding) / PlantSettings.plantCrowdingGrowthDecay;
//        x = (float) (Math.tanh(x));// * Math.tanh(-0.01 + 50 * getCrowdingFactor() / Settings.plantCriticalCrowding));
//        x = x < 0 ? (float) (1 - Math.exp(-PlantSettings.plantCrowdingGrowthDecay * x)) : x;
//        float growthRate = super.getGrowthRate() * x;
//        if (getRadius() > maxRadius)
//            growthRate *= Math.exp(maxRadius - getRadius());
//        growthRate = growthRate > 0 ? growthRate * getHealth() : growthRate;
//        return growthRate;
//    }

    public Statistics getStats() {
        Statistics stats = super.getStats();
        stats.put("Split Radius", Settings.statsDistanceScalar * maxRadius);
        return stats;
    }

    @Override
    public String getPrettyName() {
        return "Plant";
    }

    public int burstMultiplier() {
        return 3;
    }

//    @Override
//    public void interact(List<Object> interactions) {
//        float maxDist2 = getInteractionRange() * getInteractionRange();
//        float minDist2 = getRadius() * getRadius() * 1.1f * 1.1f;
//        for (Object interaction : interactions) {
//            if (interaction instanceof PlantCell) {
//                PlantCell other = (PlantCell) interaction;
//                Vector2 pos = other.getPos();
//                float dist2 = pos.dst2(getPos());
//                if (minDist2 < dist2 && dist2 < maxDist2) {
//                    Vector2 impulse = pos.cpy().sub(getPos()).setLength(10f / dist2);
//                    applyImpulse(impulse);
//                }
//            }
//        }
//    }
}
