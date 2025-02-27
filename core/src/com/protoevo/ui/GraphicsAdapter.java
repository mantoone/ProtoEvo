package com.protoevo.ui;

import com.badlogic.gdx.Game;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Graphics;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.utils.ScreenUtils;
import com.protoevo.core.ApplicationManager;
import com.protoevo.core.Simulation;
import com.protoevo.settings.GraphicsSettings;
import com.protoevo.ui.rendering.EnvironmentRenderer;
import com.protoevo.utils.CursorUtils;

public class GraphicsAdapter extends Game {
	public static GraphicsSettings settings = GraphicsSettings.createDefault();
	private final ApplicationManager applicationManager;
	private SimulationScreen simulationScreen;
	private LoadingScreen loadingScreen;
	private TitleScreen titleScreen;
	private SpriteBatch batch;
	private Skin skin;
	private static DefaultBackgroundRenderer defaultBackgroundRenderer;

	public GraphicsAdapter(ApplicationManager applicationManager) {
		this.applicationManager = applicationManager;
	}

	@Override
	public void create() {
		Graphics.DisplayMode displayMode = Gdx.graphics.getDisplayMode();
		Gdx.graphics.setUndecorated(true);
		Gdx.graphics.setWindowedMode(displayMode.width, displayMode.height);

		CursorUtils.setDefaultCursor();
		batch = new SpriteBatch();
		ShaderProgram.pedantic = false;

		defaultBackgroundRenderer = DefaultBackgroundRenderer.getInstance();

		loadingScreen = new LoadingScreen(this, applicationManager);

		skin = UIStyle.getUISkin();

		if (applicationManager.hasSimulation()) {
			Simulation simulation = applicationManager.getSimulation();
			if (simulation.isReady())
				setSimulationScreen();
			else
				loadSimulation(applicationManager.getSimulation());
		}
		else {
			titleScreen = new TitleScreen(this);
			setScreen(titleScreen);
		}
	}

	public static void renderBackground(float delta) {
		defaultBackgroundRenderer.render(delta);
	}

	public void notifySimulationReady() {
		loadingScreen.notifySimulationReady();
	}

	public SimulationScreen getSimulationScreen() {
		return simulationScreen;
	}

	public void exitApplication() {
		applicationManager.exit();
	}

	public void switchToHeadlessMode() {
		applicationManager.switchToHeadlessMode();
	}

	@Override
	public void dispose() {
		super.dispose();
		batch.dispose();
		if (skin != null)
			skin.dispose();
		if (simulationScreen != null)
			simulationScreen.dispose();
		if (titleScreen != null)
			titleScreen.dispose();
		FrameBufferManager.dispose();
		CursorUtils.dispose();
		DefaultBackgroundRenderer.disposeInstance();
	}

	@Override
	public void render() {
		ScreenUtils.clear(EnvironmentRenderer.backgroundColor);
		super.render();
	}

	public Skin getSkin() {
		return skin;
	}

	public void moveToSandbox() {}

	public void moveToTitleScreen(Screen previousScreen) {
		applicationManager.disposeSimulationIfPresent();
		titleScreen = new TitleScreen(this);
		setScreen(titleScreen);
//		screen.dispose();
	}

	public void loadSimulation(Simulation simulation) {
		loadingScreen.setUpdatesBeforeRendering(50);
		setScreen(loadingScreen);
		applicationManager.setSimulation(simulation);
	}

	public void loadPreexistingSimulation(Simulation simulation) {
		loadingScreen.setUpdatesBeforeRendering(0);
		setScreen(loadingScreen);
		applicationManager.setSimulation(simulation);
	}

	public void moveToLoadSaveScreen(String simulationName) {
		LoadSaveScreen loadSaveScreen = new LoadSaveScreen(this, simulationName, titleScreen);
		setScreen(loadSaveScreen);
	}

	public void setSimulationScreen() {
		simulationScreen = new SimulationScreen(this, applicationManager.getSimulation());
		setScreen(simulationScreen);
	}

	public SpriteBatch getSpriteBatch() {
		return batch;
	}
}
