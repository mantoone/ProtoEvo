package com.protoevo.env;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.*;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonManagedReference;
import com.google.common.collect.Streams;
import com.protoevo.biology.*;
import com.protoevo.biology.cells.Cell;
import com.protoevo.biology.cells.MeatCell;
import com.protoevo.biology.cells.PlantCell;
import com.protoevo.biology.evolution.Evolvable;
import com.protoevo.biology.cells.Protozoan;
import com.protoevo.core.*;
import com.protoevo.core.Shape;
import com.protoevo.settings.WorldGenerationSettings;
import com.protoevo.settings.Settings;
import com.protoevo.settings.SimulationSettings;
import com.protoevo.utils.FileIO;
import com.protoevo.utils.Geometry;

import java.io.Serializable;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

public class Environment implements Serializable
{
	private static final long serialVersionUID = 2804817237950199223L;
	private transient World world;
	private float elapsedTime, physicsStepTime;
	@JsonIgnore
	private final Statistics stats = new Statistics();
	@JsonIgnore
	private final Statistics debugStats = new Statistics();
	public final ConcurrentHashMap<CauseOfDeath, Integer> causeOfDeathCounts =
			new ConcurrentHashMap<>(CauseOfDeath.values().length, 1);
	@JsonIgnore
	private final ConcurrentHashMap<Class<? extends Cell>, SpatialHash<Cell>> spatialHashes;
	private final transient Map<Class<? extends Particle>, Function<Float, Vector2>> spawnPositionFns
			= new HashMap<>(3, 1);
	@JsonManagedReference
	private final ChemicalSolution chemicalSolution;
	@JsonManagedReference
	private final List<Rock> rocks = new ArrayList<>();
	private final HashMap<Class<? extends Cell>, Long> bornCounts = new HashMap<>(3);
	private final HashMap<Class<? extends Cell>, Long> generationCounts = new HashMap<>(3);
	private final static HashMap<Class<? extends Cell>, String> cellClassNames = new HashMap<>(3);
	static {
		cellClassNames.put(Protozoan.class, "Protozoa");
		cellClassNames.put(PlantCell.class, "Plants");
		cellClassNames.put(MeatCell.class, "Meat");
	}
	private long crossoverEvents = 0;

	private String genomeFile = null;
	@JsonIgnore
	private final List<String> genomesToWrite = new ArrayList<>();
	@JsonIgnore
	private final Set<Cell> cellsToAdd = new HashSet<>();

	@JsonManagedReference
	private final Set<Cell> cells = new HashSet<>();
	private boolean hasInitialised;
	private Vector2[] populationStartCentres;

	@JsonManagedReference
	private final JointsManager jointsManager;
	@JsonIgnore
	private final ConcurrentLinkedQueue<BurstRequest<? extends Cell>> burstRequests = new ConcurrentLinkedQueue<>();

	public Environment()
	{
		jointsManager = new JointsManager(this);
		world = new World(new Vector2(0, 0), true);
		world.setContinuousPhysics(false);
		world.setAutoClearForces(true);
		world.setContactListener(new CollisionHandler(this));

		System.out.println("Creating chemicals solution... ");
		if (Settings.enableChemicalField) {
			chemicalSolution = new ChemicalSolution(
					this,
					SimulationSettings.chemicalFieldResolution,
					SimulationSettings.chemicalFieldRadius);
		}

		int resolution = SimulationSettings.spatialHashResolution;
//		int protozoaCap = (int) Math.ceil(SimulationSettings.maxProtozoa / (float) (resolution * resolution));
//		int plantCap = (int) Math.ceil(SimulationSettings.maxPlants / (float) (resolution * resolution));
//		int meatCap = (int) Math.ceil(SimulationSettings.maxMeat / (float) (resolution * resolution));
		int protozoaCap = 300;
		int plantCap = 75;
		int meatCap = 50;

		spatialHashes = new ConcurrentHashMap<>(3, 1);
		spatialHashes.put(Protozoan.class, new SpatialHash<>(resolution, protozoaCap, SimulationSettings.spatialHashRadius));
		spatialHashes.put(PlantCell.class, new SpatialHash<>(resolution, plantCap, SimulationSettings.spatialHashRadius));
		spatialHashes.put(MeatCell.class, new SpatialHash<>(resolution, meatCap, SimulationSettings.spatialHashRadius));

		elapsedTime = 0;
		hasInitialised = false;
	}

	public void update(float delta)
	{
		if (world == null)  // on deserialisation
			rebuildWorld();

		cells.forEach(Particle::physicsUpdate);

		elapsedTime += delta;
		long startTime = System.nanoTime();
		world.step(
				delta,
				SimulationSettings.physicsVelocityIterations,
				SimulationSettings.physicsPositionIterations);
		physicsStepTime = TimeUnit.SECONDS.convert(
				System.nanoTime() - startTime, TimeUnit.NANOSECONDS);

  		handleCellUpdates(delta);
		handleBirthsAndDeaths();
		updateSpatialHashes();

		jointsManager.flushJoints();

		if (Settings.enableChemicalField) {
			chemicalSolution.update(delta);
		}
	}

	private void handleCellUpdates(float delta) {
		cells.parallelStream().forEach(cell -> cell.update(delta));
	}

	private void handleBirthsAndDeaths() {
		for (BurstRequest<? extends Cell> burstRequest : burstRequests)
			if (burstRequest.canBurst())
				burstRequest.burst();
		burstRequests.clear();

		flushEntitiesToAdd();

		for (Cell cell : cells) {
			if (cell.isDead()) {
				dispose(cell);
				depositOnDeath(cell);
			}
		}
		cells.removeIf(Cell::isDead);
	}

	public Vector2[] createRocks() {
		System.out.println("Creating rocks structures...");
		WorldGeneration.generateClustersOfRocks(
				this, new Vector2(0, 0), 1, WorldGenerationSettings.rockClusterRadius);

		int numClusterCentres = 8;
		Vector2[] clusterCentres = new Vector2[numClusterCentres];

		for (int i = 0; i < numClusterCentres; i++) {
			float minR = WorldGenerationSettings.rockClusterRadius;
			float maxR = WorldGenerationSettings.environmentRadius / 5f - WorldGenerationSettings.rockClusterRadius;

			Vector2 centre = WorldGeneration.randomPosition(minR, maxR);
			clusterCentres[i] = centre;

			int nRings = WorldGeneration.RANDOM.nextInt(1, 3);
			float radiusRange = WorldGeneration.RANDOM.nextFloat(8.f) * WorldGenerationSettings.rockClusterRadius;
			WorldGeneration.generateClustersOfRocks(this, centre, nRings, radiusRange);
		}

		WorldGeneration.generateRocks(this, 200);
		createRockFixtures();

		return clusterCentres;
	}

	public void createRockFixtures() {
		for (Rock rock : this.getRocks()) {
			BodyDef rockBodyDef = new BodyDef();
			Body rockBody = world.createBody(rockBodyDef);
			PolygonShape rockShape = new PolygonShape();
			rockShape.set(rock.getPoints());
			rockBody.setUserData(rock);

			FixtureDef rockFixtureDef = new FixtureDef();
			rockFixtureDef.shape = rockShape;
			rockFixtureDef.density = 0.0f;
			rockFixtureDef.friction = 0.7f;
			rockFixtureDef.filter.categoryBits = ~FixtureCategories.SENSOR;

			rockBody.createFixture(rockFixtureDef);
		}
	}

	public void rebuildWorld() {
		world = new World(new Vector2(0, 0), true);
		world.setContinuousPhysics(false);
		world.setContactListener(new CollisionHandler(this));
		createRockFixtures();
		for (Cell cell : cells) {
			cell.createBody();
		}
	}

	public void initialise() {
		System.out.println("Commencing world generation... ");
		populationStartCentres = createRocks();

		// random shuffle population start centres
		List<Vector2> populationStartCentresList = Arrays.asList(populationStartCentres);
		Collections.shuffle(populationStartCentresList);
		populationStartCentres = populationStartCentresList.toArray(new Vector2[0]);

//		if (populationStartCentres.length > 0)
//			initialisePopulation(Arrays.copyOfRange(
//					populationStartCentres, 0, WorldGenerationSettings.numPopulationClusters));
//		else
		initialisePopulation();

		flushEntitiesToAdd();

		if (Settings.writeGenomes && genomeFile != null)
			writeGenomeHeaders();

		if (chemicalSolution != null)
			chemicalSolution.initialise();

		hasInitialised = true;
		System.out.println("Environment initialisation complete.");
	}

	public void writeGenomeHeaders() {
//		Protozoan protozoan = chunkManager.getAllCells()
//				.stream()
//				.filter(cell -> cell instanceof Protozoan)
//				.map(cell -> (Protozoan) cell)
//				.findAny()
//				.orElseThrow(() -> new RuntimeException("No initial population present"));
//
//		StringBuilder headerStr = new StringBuilder();
//		headerStr.append("Generation,Time Elapsed,Parent 1 ID,Parent 2 ID,ID,");
////		for (Gene<?> gene : protozoan.getGenome().getGenes())
////			headerStr.append(gene.getTraitName()).append(",");
//
//		FileIO.appendLine(genomeFile, headerStr.toString());
	}

	public boolean hasBeenInitialised() {
		return hasInitialised;
	}

	public void initialisePopulation(Vector2[] clusterCentres) {
		if (clusterCentres != null) {
			final float clusterR = WorldGenerationSettings.populationClusterRadius;
			spawnPositionFns.put(PlantCell.class, r -> randomPosition(r, clusterCentres, 1.2f*clusterR));
			spawnPositionFns.put(Protozoan.class, r -> randomPosition(r, clusterCentres, clusterR));
		}
		else {
			spawnPositionFns.put(PlantCell.class, this::randomPosition);
			spawnPositionFns.put(Protozoan.class, this::randomPosition);
		}

		System.out.println("Creating initial plant population...");
		for (int i = 0; i < WorldGenerationSettings.numInitialPlantPellets; i++) {
			PlantCell cell = new PlantCell(this);
			if (cell.isDead()) {
				System.out.println(
					"Failed to find position for plant pellet. " +
					"Try increasing the number of population clusters or reduce the number of initial plants.");
				break;
			}
		}

		System.out.println("Creating initial protozoan population...");
		for (int i = 0; i < WorldGenerationSettings.numInitialProtozoa; i++) {
			Protozoan p = Evolvable.createNew(Protozoan.class);
			p.setEnv(this);
			if (p.isDead()) {
				System.out.println(
					"Failed to find position for protozoan. " +
					"Try increasing the number of population clusters or reduce the number of initial protozoa.");
				break;
			}
		}

		for (Particle p : cellsToAdd)
			p.applyImpulse(Geometry.randomVector(.01f));
	}

	public void initialisePopulation() {
		Vector2[] clusterCentres = new Vector2[WorldGenerationSettings.numPopulationClusters];
		for (int i = 0; i < clusterCentres.length; i++)
			clusterCentres[i] = Geometry.randomPointInCircle(
					WorldGenerationSettings.environmentRadius, WorldGeneration.RANDOM);
		initialisePopulation(clusterCentres);
	}

	public Vector2 randomPosition(float entityRadius, Vector2[] clusterCentres) {
		int clusterIdx = Simulation.RANDOM.nextInt(clusterCentres.length);
		Vector2 clusterCentre = clusterCentres[clusterIdx];
		return randomPosition(entityRadius, clusterCentre, WorldGenerationSettings.populationClusterRadius);
	}

	public Vector2 randomPosition(float entityRadius, Vector2[] clusterCentres, float clusterRadius) {
		int clusterIdx = Simulation.RANDOM.nextInt(clusterCentres.length);
		Vector2 clusterCentre = clusterCentres[clusterIdx];
		return randomPosition(entityRadius, clusterCentre, clusterRadius);
	}

	public Vector2 randomPosition(float entityRadius, Vector2 centre, float clusterRadius) {
		for (int i = 0; i < 20; i++) {
			float r = Simulation.RANDOM.nextFloat(clusterRadius);
			Vector2 pos = Geometry.randomPointInCircle(r, WorldGeneration.RANDOM);
			pos.add(centre);
			Optional<? extends Shape> collision = getCollision(pos, entityRadius);
			if (collision.isPresent() && collision.get() instanceof PlantCell) {
				PlantCell plant = (PlantCell) collision.get();
				plant.kill(CauseOfDeath.ENV_CAPACITY_EXCEEDED);
				return pos;
			} else if (collision.isEmpty())
				return pos;
		}

		return null;
	}

	public Vector2 getRandomPosition(Particle particle) {
		return spawnPositionFns.getOrDefault(particle.getClass(), this::randomPosition)
				.apply(particle.getRadius());
	}

	public Vector2 randomPosition(float entityRadius) {
		return randomPosition(entityRadius, Geometry.ZERO, WorldGenerationSettings.rockClusterRadius);
	}

	public void add(Cell cell) {
		if (getLocalCount(cell) < getLocalCapacity(cell)) {
			cells.add(cell);
			// update local counts
			spatialHashes.get(cell.getClass()).add(cell);

			bornCounts.put(cell.getClass(),
					bornCounts.getOrDefault(cell.getClass(), 0L) + 1);
			generationCounts.put(cell.getClass(),
					Math.max(generationCounts.getOrDefault(cell.getClass(), 0L),
							 cell.getGeneration()));
		}
		else {
			cell.kill(CauseOfDeath.ENV_CAPACITY_EXCEEDED);
			dispose(cell);
		}
	}

	private void flushEntitiesToAdd() {
		for (Cell cell : cellsToAdd)
			add(cell);
		cellsToAdd.clear();
	}

	private void flushWrites() {
		List<String> genomeWritesHandled = new ArrayList<>();
		for (String line : genomesToWrite) {
			FileIO.appendLine(genomeFile, line);
			genomeWritesHandled.add(line);
		}
		genomesToWrite.removeAll(genomeWritesHandled);
	}

	public int getCount(Class<? extends Cell> cellClass) {
		return spatialHashes.get(cellClass).size();
	}

	private void updateSpatialHashes() {
		spatialHashes.values().forEach(SpatialHash::clear);
		cells.forEach(cell -> spatialHashes.get(cell.getClass()).add(cell));
	}

	public int getCapacity(Class<? extends Cell> cellClass) {
		return spatialHashes.get(cellClass).getTotalCapacity();
	}

	private void dispose(Particle e) {
		CauseOfDeath cod = e.getCauseOfDeath();
		if (cod != null) {
			int count = causeOfDeathCounts.getOrDefault(cod, 0);
			causeOfDeathCounts.put(cod, count + 1);
		}
		e.dispose();
	}

	public void depositOnDeath(Cell cell) {
		if (Settings.enableChemicalField) {
			if (!cell.isEngulfed() && !cell.hasChildren()) {
				if (cell instanceof Protozoan || cell instanceof MeatCell)
					chemicalSolution.depositCircle(
							cell.getPos(), cell.getRadius() * 1.5f,
							Color.FIREBRICK.cpy().mul(0.25f + 0.75f * cell.getHealth()));
				else if (cell instanceof PlantCell)
					chemicalSolution.depositCircle(cell.getPos(),
							cell.getRadius() * 1.5f,
							Color.FOREST.cpy().mul(0.25f + 0.75f * cell.getHealth()));
			}
		}
	}

	private void handleNewProtozoa(Protozoan p) {
//		if (genomeFile != null && Settings.writeGenomes) {
//			String genomeLine = p.getGeneration() + "," + elapsedTime + "," + p.getGenome().toString();
//			genomesToWrite.add(genomeLine);
//		}
	}

	public int getLocalCount(Particle cell) {
		return getLocalCount(cell.getClass(), cell.getPos());
	}

	public int getLocalCount(Class<? extends Particle> cellType, Vector2 pos) {
		return spatialHashes.get(cellType).getCount(pos);
	}

	public int getLocalCapacity(Particle cell) {
		return getLocalCapacity(cell.getClass());
	}

	public int getLocalCapacity(Class<? extends Particle> cellType) {
		return spatialHashes.get(cellType).getChunkCapacity();
	}

	public void registerToAdd(Cell e) {
		if (getLocalCount(e) >= getLocalCapacity(e)) {
			e.kill(CauseOfDeath.ENV_CAPACITY_EXCEEDED);
			dispose(e);
		}

		cellsToAdd.add(e);
	}

	public Statistics getStats(boolean includeProtozoaStats) {
		Statistics stats = new Statistics();
		stats.putTime("Time Elapsed", elapsedTime);
		stats.putCount("Protozoa", numberOfProtozoa());
		stats.putCount("Plants", getCount(PlantCell.class));
		stats.putCount("Meat Pellets", getCount(MeatCell.class));

		for (Class<? extends Cell> cellClass : generationCounts.keySet())
			stats.putCount("Max " + cellClassNames.get(cellClass) + " Generation",
							generationCounts.get(cellClass).intValue());

		for (Class<? extends Cell> cellClass : bornCounts.keySet())
			stats.putCount(cellClassNames.get(cellClass) + " Created",
					bornCounts.get(cellClass).intValue());

		stats.putCount("Crossover Events", (int) crossoverEvents);
		for (CauseOfDeath cod : CauseOfDeath.values()) {
			if (cod.isDebugDeath())
				continue;
			int count = causeOfDeathCounts.getOrDefault(cod, 0);
			if (count > 0)
				stats.putCount("Died from " + cod.getReason(), count);
		}
//		if (includeProtozoaStats)
//			stats.putAll(getProtozoaStats());
		return stats;
	}

	public Statistics getDebugStats() {
		debugStats.clear();
		for (CauseOfDeath cod : CauseOfDeath.values()) {
			if (!cod.isDebugDeath())
				continue;
			int count = causeOfDeathCounts.getOrDefault(cod, 0);
			if (count > 0)
				debugStats.put("Died from " + cod.getReason(), (float) count);
		}
		return debugStats;
	}

	public Statistics getPhysicsDebugStats() {
		debugStats.clear();

		debugStats.putCount("Bodies", world.getBodyCount());
		debugStats.putCount("Contacts", world.getContactCount());
		debugStats.putCount("Joints", world.getJointCount());
		debugStats.putCount("Fixtures", world.getFixtureCount());
		debugStats.putCount("Proxies", world.getProxyCount());
		debugStats.putTime("Physics Step Time", physicsStepTime);

		int totalCells = cells.size();
		int sleepCount = 0;
		for (Cell cell : cells)
			if (cell.getBody() != null && !cell.getBody().isAwake())
				sleepCount++;

		debugStats.putPercentage("Sleeping",  100f * sleepCount / totalCells);

		return debugStats;
	}

	public Statistics getStats() {
		return getStats(false);
	}

	public Statistics getProtozoaStats() {
		Statistics stats = new Statistics();

		// TODO: move calculating mean stats to Statistics class
//		Collection<Protozoan> protozoa = cells.stream()
//				.filter(cell -> cell instanceof Protozoan)
//				.map(cell -> (Protozoan) cell)
//				.collect(Collectors.toSet());
//
//		for (Cell e : protozoa) {
//			for (Statistics.Stat stat : e.getStats()) {
//				String key = "Sum " + stat.getName();
//				float currentValue = stats.getOrDefault(key, 0f);
//				stats.put(key, stat.getValue() + currentValue);
//			}
//		}
//
//		int numProtozoa = protozoa.size();
//		for (Cell e : protozoa) {
//			for (Map.Entry<String, Float> stat : e.getStats().entrySet()) {
//				float sumValue = stats.getOrDefault("Sum " + stat.getKey(), 0f);
//				float mean = sumValue / numProtozoa;
//				stats.put("Mean " + stat.getKey(), mean);
//				float currVar = stats.getOrDefault("Var " + stat.getKey(), 0f);
//				float deltaVar = (float) Math.pow(stat.getValue() - mean, 2) / numProtozoa;
//				stats.put("Var " + stat.getKey(), currVar + deltaVar);
//			}
//		}
		return stats;
	}
	
	public int numberOfProtozoa() {
		return getCount(Protozoan.class);
	}


	public long getGeneration() {
		return generationCounts.getOrDefault(Protozoan.class, 0L);
	}

	public Optional<? extends Shape> getCollision(Vector2 pos, float r) {
		Optional<Cell> collidingCell = Streams.concat(cells.stream(), cellsToAdd.stream())
				.filter(cell -> Geometry.doCirclesCollide(pos, r, cell.getPos(), cell.getRadius()))
				.findAny();

		if (collidingCell.isPresent())
			return collidingCell;

		return rocks.stream().filter(rock -> rock.intersectsWith(pos, r)).findAny();
	}

	public float getElapsedTime() {
		return elapsedTime;
	}

	public void setGenomeFile(String genomeFile) {
		this.genomeFile = genomeFile;
	}

	public ChemicalSolution getChemicalSolution() {
		return chemicalSolution;
	}

	public List<Rock> getRocks() {
		return rocks;
	}

	public void registerCrossoverEvent() {
		crossoverEvents++;
	}

	public World getWorld() {
		return world;
	}

	public Collection<Cell> getCells() {
		return cells;
	}

	public Collection<? extends Particle> getParticles() {
		return cells;
	}

	public void ensureAddedToEnvironment(Particle particle) {
		if (particle instanceof Cell) {
			Cell cell = (Cell) particle;
			if (!cells.contains(cell))
				registerToAdd(cell);
		}
	}

	public JointsManager getJointsManager() {
		return jointsManager;
	}

	public <T extends Cell> void requestBurst(Cell parent,
											  Class<T> cellType,
											  Function<Float, T> createChild,
											  boolean overrideMinParticleSize) {

		if (getLocalCount(cellType, parent.getPos()) >= getLocalCapacity(cellType))
			return;

		if (burstRequests.stream().anyMatch(request -> request.parentEquals(parent)))
			return;

		BurstRequest<T> request = new BurstRequest<>(parent, cellType, createChild, overrideMinParticleSize);
		burstRequests.add(request);
	}

	public <T extends Cell> void requestBurst(Cell parent, Class<T> cellType, Function<Float, T> createChild) {
		requestBurst(parent, cellType, createChild, false);
	}

	public SpatialHash<Cell> getSpatialHash(Class<? extends Cell> clazz) {
		return spatialHashes.get(clazz);
	}
}
