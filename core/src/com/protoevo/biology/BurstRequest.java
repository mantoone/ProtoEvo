package com.protoevo.biology;

import com.badlogic.gdx.math.Vector2;
import com.protoevo.core.Simulation;
import com.protoevo.core.settings.EnvironmentSettings;
import com.protoevo.core.settings.Settings;
import com.protoevo.core.settings.SimulationSettings;
import com.protoevo.env.Environment;
import com.protoevo.utils.Geometry;

import java.util.function.Function;

public class BurstRequest<T extends Cell> {

    private final Function<Float, T> createChild;
    private final Cell parent;
    private final Class<T> cellType;

    public BurstRequest(Cell parent, Class<T> cellType, Function<Float, T> createChild) {
        this.createChild = createChild;
        this.parent = parent;
        this.cellType = cellType;
    }

    public void burst() {

        if (parent.getRadius() < parent.getMinBurstRadius())
            return;

        parent.kill();

        float angle = (float) (2 * Math.PI * Simulation.RANDOM.nextDouble());

        float volume = Geometry.getSphereVolume(parent.getRadius());
        float minVolume = Geometry.getSphereVolume(SimulationSettings.minParticleRadius);
        int maxChildren = Math.min(6, (int) (volume / minVolume));

        int nChildren = 2;
        if (maxChildren > 2) {
            int chances = parent.burstMultiplier();
            for (int i = 0; i < chances; i++)
                nChildren = Math.max(nChildren, Simulation.RANDOM.nextInt(2, maxChildren));
        }

        Environment environment = parent.getEnv();
        for (int i = 0; i < nChildren; i++) {
            Vector2 dir = new Vector2((float) Math.cos(angle), (float) Math.sin(angle));
            float p = (float) (0.3 + 0.7 * Simulation.RANDOM.nextDouble() / nChildren);

            int nEntities = environment.cellCounts.getOrDefault(cellType, 0);
            int maxEntities = environment.cellCapacities.getOrDefault(cellType, 0);
            if (nEntities > maxEntities)
                return;

            T child = createChild.apply(parent.getRadius() * p);
            child.setPos(parent.getPos().add(dir.scl(2 * child.getRadius())));
            child.applyImpulse(dir.scl(2f));

            child.setGeneration(parent.getGeneration() + 1);
            allocateChildResources(child, p);

            angle += 2 * Math.PI / nChildren;
        }
    }

    public void allocateChildResources(Cell child, float p) {
        child.setAvailableConstructionMass(parent.getConstructionMassAvailable() * p);
        child.setEnergyAvailable(parent.getEnergyAvailable() * p);
        for (Food.ComplexMolecule molecule : parent.getComplexMolecules())
            child.setComplexMoleculeAvailable(molecule, p * parent.getComplexMoleculeAvailable(molecule));

        for (CellAdhesion.CAM cam : parent.getSurfaceCAMs())
            child.setCAMAvailable(cam, p * parent.getCAMAvailable(cam));

        for (Food.Type foodType : parent.getFoodToDigest().keySet()) {
            Food oldFood = parent.getFoodToDigest().get(foodType);
            Food newFood = new Food(p * oldFood.getSimpleMass(), foodType);
            for (Food.ComplexMolecule molecule : oldFood.getComplexMolecules()) {
                float moleculeAmount = p * oldFood.getComplexMoleculeMass(molecule);
                newFood.addComplexMoleculeMass(molecule, moleculeAmount);
            }
            child.setFoodToDigest(foodType, newFood);
        }
    }

    public boolean parentEquals(Cell parent) {
        return this.parent.equals(parent);
    }
}
