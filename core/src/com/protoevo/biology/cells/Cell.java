package com.protoevo.biology.cells;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Vector2;
import com.protoevo.biology.CauseOfDeath;
import com.protoevo.biology.ComplexMolecule;
import com.protoevo.biology.ConstructionProject;
import com.protoevo.biology.Food;
import com.protoevo.biology.nodes.SurfaceNode;
import com.protoevo.biology.organelles.Organelle;
import com.protoevo.physics.Particle;
import com.protoevo.core.Statistics;
import com.protoevo.physics.CollisionHandler;
import com.protoevo.env.JointsManager;
import com.protoevo.env.Rock;
import com.protoevo.settings.Settings;
import com.protoevo.settings.SimulationSettings;
import com.protoevo.utils.Colour;
import com.protoevo.utils.Geometry;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static com.protoevo.utils.Utils.lerp;

public abstract class Cell extends Particle implements Serializable {
	private static final long serialVersionUID = 1L;

	private final Colour healthyColour = new Colour(Color.WHITE);
	private final Colour fullyDegradedColour = new Colour(Color.WHITE);
	private final Colour currentColour = new Colour();
	private int generation = 1;
	protected boolean hasHandledDeath = false;
	private float timeAlive = 0f;
	private float health = 1f;
	private float growthRate = 0.0f;
	private float energyAvailable = SimulationSettings.startingAvailableCellEnergy;
	private double constructionMassAvailable = SimulationSettings.startingAvailableConstructionMass;
	private double massChangeForGrowth = 0f;
	private final Map<ComplexMolecule, Float> availableComplexMolecules = new ConcurrentHashMap<>(0);
	private final Map<Long, Long> cellJoinings = new ConcurrentHashMap<>();  // maps cell id to joining id
	//	private final Map<CellAdhesion.CAM, Float> surfaceCAMs = new HashMap<>(0);
	private final Map<Food.Type, Float> foodDigestionRates = new HashMap<>(0);
	private final Map<Food.Type, Food> foodToDigest = new HashMap<>(0);
	private final Set<ConstructionProject> constructionProjects = new HashSet<>(0);
	//	private final Map<CellAdhesion.CAM, Float> camProductionRates = new HashMap<>(0);
	private ArrayList<Organelle> organelles = new ArrayList<>();
	private boolean hasBurst = false;
	private float repairRate = 1f;

	private Cell engulfer = null;
	private boolean fullyEngulfed = false;

	public void update(float delta) {
		if (health <= 0.05f && !super.isDead()) {
			kill(CauseOfDeath.HEALTH_TOO_LOW);
			return;
		}

		super.update(delta);
		timeAlive += delta;

		voidDamage(delta);
		digest(delta);
		repair(delta);
		grow(delta);

		for (Organelle organelle : organelles)
			organelle.update(delta);

		cellJoinings.entrySet().removeIf(this::detachCellCondition);
	}

	public void voidDamage(float delta) {
		if (getPos().len2() > getVoidStartDistance2())
			damage(delta * SimulationSettings.voidDamagePerSecond, CauseOfDeath.THE_VOID);
	}

	protected float getVoidStartDistance2() {
		return SimulationSettings.voidStartDistance2;
	}

	public void requestJointRemoval(Long joiningId) {
		JointsManager.Joining joining = getJoining(joiningId);
		if (joining != null)
			requestJointRemoval(joining);
	}

	public void requestJointRemoval(JointsManager.Joining joining) {
		getEnv().getJointsManager().requestJointRemoval(joining);
		Cell other = (Cell) joining.getOther(this);
		cellJoinings.remove(other.getId());
		other.cellJoinings.remove(getId());
	}

	public Cell getCell(Long id) {
		if (getEnv() == null)
			return null;
		return getEnv().getCell(id);
	}

	public Collection<Long> getAttachedCellIDs() {
		return cellJoinings.keySet();
	}

	public boolean isAttachedTo(Cell other) {
		return cellJoinings.containsKey(other.getId());
	}

	public void setOrganelles(ArrayList<Organelle> organelles) {
		this.organelles = organelles;
	}

	public void addOrganelle(Organelle organelle) {
		organelles.add(organelle);
	}

	public void progressConstructionProjects(float delta) {
		for (ConstructionProject project : constructionProjects) {
			progressProject(project, delta);
		}
	}

	public void progressProject(ConstructionProject project, float delta) {
		if (delta > 0 && project.notFinished() && project.canMakeProgress(
				energyAvailable,
				(float) constructionMassAvailable,
				availableComplexMolecules,
				delta)) {
			depleteEnergy(project.energyToMakeProgress(delta));
			depleteConstructionMass(project.massToMakeProgress(delta));
			if (project.requiresComplexMolecules())
				for (ComplexMolecule molecule : project.getRequiredMolecules()) {
					float amountUsed = project.complexMoleculesToMakeProgress(delta, molecule);
					depleteComplexMolecule(molecule, amountUsed);
				}
			project.progress(delta);
		}
	}

	public float getDigestionRate(Food.Type foodType) {
		return foodDigestionRates.getOrDefault(foodType, 0f);
	}

	public void setDigestionRate(Food.Type foodType, float rate) {
		foodDigestionRates.put(foodType, rate);
	}

	public float getTotalFoodMassToDigest() {
		float total = 0;
		for (Food food : foodToDigest.values()) {
			total += food.getSimpleMass();
		}
		return total;
	}

	public float getFoodToDigestMassCap() {
		return getMass(getRadius()) * 0.25f;
	}

	public void eat(Cell cell, float extraction) {

		if (!(cell instanceof PlantCell || cell instanceof MeatCell)
				|| getTotalFoodMassToDigest() >= getFoodToDigestMassCap())
			return;

		Food.Type foodType = cell instanceof PlantCell ? Food.Type.Plant : Food.Type.Meat;
		float extractedMass = cell.getMass() * extraction;
		cell.removeMass(Settings.foodExtractionWasteMultiplier * extractedMass, CauseOfDeath.EATEN);

		Food food;
		if (foodToDigest.containsKey(foodType))
			food = foodToDigest.get(foodType);
		else {
			food = new Food(extractedMass, foodType);
			foodToDigest.put(foodType, food);
		}

		food.addSimpleMass(extractedMass);

		for (ComplexMolecule molecule : cell.getComplexMolecules()) {
			if (cell.getComplexMoleculeAvailable(molecule) > 0) {
				float extractedAmount = extraction * cell.getComplexMoleculeAvailable(molecule);
				cell.depleteComplexMolecule(molecule, extractedAmount);
				if (extractedAmount <= 1e-12)
					continue;
				food.addComplexMoleculeMass(molecule, extractedMass);
			}
		}
	}

	public void addFood(Food.Type foodType, float amount) {
		Food food = foodToDigest.getOrDefault(foodType, new Food(amount, foodType));
		food.addSimpleMass(amount);
		foodToDigest.put(foodType, food);
	}

	public void digest(float delta) {
		if (foodToDigest.isEmpty())
			return;  // avoids creating iterator if there's nothing to digest

		for (Food food : foodToDigest.values()) {
			float rate = delta * SimulationSettings.digestionFactor * getDigestionRate(food.getType());
			if (food.getSimpleMass() > 0) {
				float massExtracted = food.getSimpleMass() * rate;
				addConstructionMass(massExtracted);
				food.subtractSimpleMass(massExtracted);
				energyAvailable += food.getEnergy(massExtracted);
			}
			for (ComplexMolecule molecule : food.getComplexMolecules()) {
				float amount = food.getComplexMoleculeMass(molecule);
				if (amount == 0)
					continue;
				float extracted = Math.min(amount, amount * rate);
				addAvailableComplexMolecule(molecule, extracted);
				food.subtractComplexMolecule(molecule, extracted);
			}
		}
	}

	public void repair(float delta) {
		if (!isDead() && getHealth() < 1f && getRepairRate() > 0) {
			float massRequired = getMass() * 0.01f * delta;
			float energyRequired = massRequired * 3f;
			if (massRequired < constructionMassAvailable
					&& energyRequired < energyAvailable) {
				depleteEnergy(energyRequired);
				depleteConstructionMass(massRequired);
				heal(delta * getRepairRate());
			}
		}
	}

	private float getRepairRate() {
		return Settings.cellRepairRate * repairRate;
	}

	public boolean detachCellCondition(Map.Entry<Long, Long> entry) {
		Cell other = getCell(entry.getKey());
		long joiningID = entry.getValue();
		JointsManager.Joining joining = getJoining(entry.getValue());
		if (joining == null || other == null) {
			getEnv().getJointsManager().requestJointRemoval(joiningID);
			return true;
		}

		boolean detach = other.isDead();

		if (detach) {
			requestJointRemoval(joining);
			return true;
		}

		float dist2 = joining.getAnchorA().dst2(joining.getAnchorB());
		float maxDist = joining.getMaxLength();
		detach = dist2 > maxDist * maxDist;

		if (detach)
			requestJointRemoval(joining);

		return detach;
	}

	public void addConstructionProject(ConstructionProject project) {
		constructionProjects.add(project);
	}

	public void handleInteractions(float delta) {
	}

	public void grow(float delta) {
		if (constructionMassAvailable <= 0)
			return;

		double gr = getGrowthRate();
		double dr = SimulationSettings.cellGrowthFactor * gr * ((double) delta);
		double currR = getRadiusDouble();
		double newR = currR + dr;

		if (newR > getMaxRadius())
			newR = getMaxRadius();
		else if (newR < getMinRadius())
			newR = getMinRadius();

		if (newR == currR)
			return;

		massChangeForGrowth = getMass(newR) - getMass(currR);

		if (massChangeForGrowth > constructionMassAvailable) {
			double dr2 = constructionMassAvailable / (Math.PI * getMassDensity());
			newR = Math.sqrt(currR * currR + dr2);
			massChangeForGrowth = constructionMassAvailable;
		}

		if (newR > SimulationSettings.minParticleRadius && massChangeForGrowth <= constructionMassAvailable) {
			setRadius(newR);
			if (massChangeForGrowth > 0)
				depleteConstructionMass(massChangeForGrowth);
		}
		if (newR < getMinRadius())
			kill(CauseOfDeath.GREW_TOO_SMALL);
	}

	public float getMaxRadius() {
		return SimulationSettings.maxParticleRadius;
	}

	public float getMinRadius() {
		return SimulationSettings.minParticleRadius;
	}

	public void setGrowthRate(float gr) {
		growthRate = gr;
	}

	public float getGrowthRate() {
		return growthRate;
	}

	public void registerJoining(JointsManager.Joining joining) {
		Particle other = joining.getOther(this);
		cellJoinings.put(other.getId(), joining.id);
	}

	public void deregisterJoining(JointsManager.Joining joining) {
		Particle other = joining.getOther(this);
		cellJoinings.remove(other.getId());
	}

	@Override
	public void onCollision(CollisionHandler.Collision contact, Rock rock) {
		super.onCollision(contact, rock);
		if (rock.pointInside(getPos())) {
			kill(CauseOfDeath.SUFFOCATION);
		}
	}

	@Override
	public void onCollision(CollisionHandler.Collision contact, Particle other) {
		super.onCollision(contact, other);
		if (other.isPointInside(getPos())) {
			kill(CauseOfDeath.SUFFOCATION);
		}
	}

	public boolean notBoundTo(Cell otherCell) {
		return !(
			cellJoinings.containsKey(otherCell.getId())
			|| otherCell.cellJoinings.containsKey(getId())
		);
	}

	public abstract boolean isEdible();

	public void heal(float h) {
		damage(-h, CauseOfDeath.HEALED_TO_DEATH);
	}

	public void damage(float d, CauseOfDeath cause) {
		health -= d;
		if (health > 1)
			health = 1;

		if (health < 0.05)
			kill(cause);
	}

	public String getPrettyName() {
		return "Cell";
	}

	public float getTimeAlive() {
		return timeAlive;
	}

	public float getKineticEnergyRequiredForThrust(Vector2 thrustVector) {
		float speed = getSpeed();
		float mass = getMass();
		// This is surely not correct
		return .5f * mass * (speed * speed - thrustVector.len2() / (mass * mass));
	}

	public void generateMovement(Vector2 thrustVector) {
		generateMovement(thrustVector, 0);
	}

	public float generateMovement(Vector2 thrustVector, float torque) {
		float work = getKineticEnergyRequiredForThrust(thrustVector);
		// TODO: add work to apply torque

		if (enoughEnergyAvailable(work)) {
			depleteEnergy(work);
			applyImpulse(thrustVector);
			applyTorque(torque);
			return 1f;
		} else {
			// not accurate scaling of thrust and torque
			float p = getEnergyAvailable() / work;
			thrustVector.scl(p);
			torque = torque * p;
			setEnergyAvailable(0);
			applyImpulse(thrustVector);
			applyTorque(torque);
			return p;
		}
	}

	public Statistics getStats() {
		Statistics stats = super.getStats();
		stats.putTime("Age", timeAlive);
		stats.putPercentage("Health", 100 * getHealth());
		stats.putCount("Generation", getGeneration());
		stats.putEnergy("Available Energy", energyAvailable);
		stats.putMass("Construction Mass", (float) constructionMassAvailable);
		stats.putMass("Construction Mass Limit", getConstructionMassCap());
//		if (wasteMass > 0)
//			stats.putMass("Waste Mass", wasteMass);

		stats.putSpeed("Growth Rate", getGrowthRate());
		stats.put("Repair Rate", 100 * getRepairRate(),
				Statistics.ComplexUnit.PERCENTAGE_PER_TIME);

		for (ComplexMolecule molecule : availableComplexMolecules.keySet())
			if (availableComplexMolecules.get(molecule) >= 1e-9)
				stats.putMass(String.format("Molecule %.2f Available", molecule.getSignature()),
						availableComplexMolecules.get(molecule));

		if (cellJoinings.size() > 0)
			stats.putCount("Num Cell Bindings", cellJoinings.size());

//		for (CellAdhesion.CAMJunctionType junctionType : CellAdhesion.CAMJunctionType.values()) {
//			float camMass = 0;
//			for (CellAdhesion.CAM molecule : surfaceCAMs.keySet())
//				if (molecule.getJunctionType().equals(junctionType))
//					camMass += surfaceCAMs.get(molecule);
//			if (camMass > 0)
//				stats.putMass(junctionType + " CAM Mass", camMass);
//		}

		stats.putBoolean("Being Engulfed", engulfer != null);

		Statistics.ComplexUnit massPerTime = Statistics.ComplexUnit.MASS_PER_TIME;

		for (Food.Type foodType : foodDigestionRates.keySet())
			if (foodDigestionRates.get(foodType) > 0)
				stats.put(foodType + " Digestion Rate", foodDigestionRates.get(foodType), massPerTime);

		for (Food food : foodToDigest.values())
			stats.putMass(food + " to Digest", food.getSimpleMass());

		return stats;
	}

	public Statistics getDebugStats() {
		Statistics stats = super.getDebugStats();
		stats.putCount("Num Attached Cells", cellJoinings.size());
		stats.putMass("Mass To Grow", (float) massChangeForGrowth);
		return stats;
	}

	public float getHealth() {
		return MathUtils.clamp(health, 0, 1);
	}

	public boolean isDead() {
		return super.isDead();
	}

	public void kill(CauseOfDeath causeOfDeath) {
		for (Long otherId : getAttachedCellIDs()) {
			Cell other = getCell(otherId);
			if (other != null)
				other.cellJoinings.remove(this.getId());
		}
		cellJoinings.clear();
		super.kill(causeOfDeath);
	}

	@Override
	public Colour getColour() {
		Colour healthyColour = getHealthyColour();
		Colour degradedColour = getFullyDegradedColour();
		return currentColour.set(healthyColour).lerp(degradedColour, 1 - getHealth());
	}

	public Collection<Organelle> getOrganelles() {
		return organelles;
	}

	public Colour getHealthyColour() {
		return healthyColour;
	}

	public void setHealthyColour(Colour healthyColour) {
		this.healthyColour.set(healthyColour);
	}

	public void setDegradedColour(Colour fullyDegradedColour) {
		this.fullyDegradedColour.set(fullyDegradedColour);
	}

	public Colour getFullyDegradedColour() {
		return fullyDegradedColour;
	}

	public Colour degradeColour(Colour colour, float t) {
		return lerp(colour, Colour.LIGHT_GRAY, t);
	}

	public int getGeneration() {
		return generation;
	}

	public void setGeneration(int generation) {
		this.generation = generation;
	}

	public int burstMultiplier() {
		return 1;
	}

	public float getMinBurstRadius() {
		return 2 * SimulationSettings.minParticleRadius;
	}

	public void setFoodToDigest(Food.Type foodType, Food food) {
		foodToDigest.put(foodType, food);
	}

//	public float getCAMAvailable(CellAdhesion.CAM cam) {
//		return surfaceCAMs.getOrDefault(cam, 0f);
//	}
//
//	public void setCAMAvailable(CellAdhesion.CAM cam, float amount) {
//		surfaceCAMs.put(cam, amount);
//	}

	public boolean enoughEnergyAvailable(float work) {
		return work < energyAvailable;
	}

	public float getEnergyAvailable() {
		return energyAvailable;
	}

	public void addAvailableEnergy(float energy) {
		energyAvailable = Math.min(energyAvailable + energy, getAvailableEnergyCap());
	}

	private float getAvailableEnergyCap() {
		return SimulationSettings.startingAvailableCellEnergy * getRadius() / SimulationSettings.minParticleRadius;
	}

	public void setEnergyAvailable(float energy) {
		energyAvailable = energy;
	}

	public void depleteEnergy(float energy) {
		energyAvailable = Math.max(0, energyAvailable - energy);
	}

	public Collection<ComplexMolecule> getComplexMolecules() {
		return availableComplexMolecules.keySet();
	}

	public void depleteComplexMolecule(ComplexMolecule molecule, float amount) {
		float currAmount = getComplexMoleculeAvailable(molecule);
		setComplexMoleculeAvailable(molecule, currAmount - amount);
	}

	public float getComplexMoleculeAvailable(ComplexMolecule molecule) {
		return availableComplexMolecules.getOrDefault(molecule, 0f);
	}

	public void addAvailableComplexMolecule(ComplexMolecule molecule, float amount) {
		float currentAmount = availableComplexMolecules.getOrDefault(molecule, 0f);
		float newAmount = Math.min(getComplexMoleculeMassCap(), currentAmount + amount);
		availableComplexMolecules.put(molecule, newAmount);
	}

	private float getComplexMoleculeMassCap() {
		return getMass(getRadius() * 0.1f);
	}

	public void setComplexMoleculeAvailable(ComplexMolecule molecule, float amount) {
		availableComplexMolecules.put(molecule, Math.max(0, amount));
	}

	public float getConstructionMassCap() {
		return 2 * getMassDensity() * Geometry.getCircleArea(getRadius() * 0.25f);
	}

	public void setAvailableConstructionMass(float mass) {
		constructionMassAvailable = Math.min(mass, getConstructionMassCap());
	}

	public void setAvailableConstructionMass(double mass) {
		constructionMassAvailable = Math.min(mass, getConstructionMassCap());
	}

	public float getConstructionMassAvailable() {
		return (float) constructionMassAvailable;
	}

	public double getConstructionMassAvailableDouble() {
		return constructionMassAvailable;
	}

	public void addConstructionMass(float mass) {
		setAvailableConstructionMass(constructionMassAvailable + mass);
	}

	public void addConstructionMass(double mass) {
		setAvailableConstructionMass(constructionMassAvailable + mass);
	}

	public void depleteConstructionMass(float mass) {
		constructionMassAvailable = Math.max(0, constructionMassAvailable - mass);
	}

	public void depleteConstructionMass(double mass) {
		constructionMassAvailable = Math.max(0, constructionMassAvailable - mass);
	}

//	public void setCAMProductionRate(CellAdhesion.CAM cam, float rate) {
//		camProductionRates.put(cam, rate);
//	}

	@Override
	public float getMass() {
		float extraMass = (float) constructionMassAvailable;
		for (float mass : availableComplexMolecules.values())
			extraMass += mass;
		return getMass(getRadius(), extraMass);
	}

	/**
	 * Changes the radius of the cell to remove the given amount of mass
	 *
	 * @param mass mass to remove
	 */
	public void removeMass(float mass, CauseOfDeath causeOfDeath) {
		float percentRemoved = mass / getMass();
		damage(percentRemoved, causeOfDeath);

		double newR = (1 - percentRemoved) * getRadius();
		if (newR < SimulationSettings.minParticleRadius * 0.5f)
			kill(causeOfDeath);

		setRadius(newR);
	}

	public void removeMass(float mass) {
		removeMass(mass, CauseOfDeath.LOST_TOO_MUCH_MASS);
	}

	public Map<Food.Type, Food> getFoodToDigest() {
		return foodToDigest;
	}

	public int getNumAttachedCells() {
		return cellJoinings.size();
	}

	public float getShieldFactor() {
		return 1.3f;
	}

	public void setHasBurst(boolean hasBurst) {
		this.hasBurst = hasBurst;
	}

	public boolean hasNotBurst() {
		return !hasBurst;
	}

	public void setRepairRate(float repairRate) {
		this.repairRate = repairRate;
	}

	public void setEngulfer(Cell engulfer) {
		this.engulfer = engulfer;
	}

	public boolean isEngulfed() {
		return engulfer != null;
	}

	public boolean isFullyEngulfed() {
		return fullyEngulfed;
	}

	public void setFullyEngulfed() {
		this.fullyEngulfed = true;
	}

	public List<SurfaceNode> getSurfaceNodes() {
		return null;
	}

}
