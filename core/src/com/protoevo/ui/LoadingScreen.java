package com.protoevo.ui;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.ScreenAdapter;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.protoevo.core.ApplicationManager;
import com.protoevo.core.Simulation;
import com.protoevo.utils.CursorUtils;

public class LoadingScreen extends ScreenAdapter {

    private final ApplicationManager applicationManager;
    private final GraphicsAdapter graphicsAdapter;
    private final SpriteBatch batch;
    private float elapsedTime = 0;
    private BitmapFont font;
    private volatile boolean simulationReady = false;
    private int updatesCompleted;
    private int updatesBeforeRendering = 50;

    public LoadingScreen(GraphicsAdapter graphicsAdapter,
                         ApplicationManager applicationManager) {
        this.graphicsAdapter = graphicsAdapter;
        this.applicationManager = applicationManager;
        batch = graphicsAdapter.getSpriteBatch();
    }

    public void setUpdatesBeforeRendering(int updatesBeforeRendering) {
        this.updatesBeforeRendering = updatesBeforeRendering;
    }

    @Override
    public void show() {
        CursorUtils.setDefaultCursor();
        simulationReady = false;
        updatesCompleted = 0;
    }

    @Override
    public void resize(int width, int height) {
        super.resize(width, height);
        font = graphicsAdapter.getSkin().getFont("default");
    }

    public void loadingString(String text) {
        elapsedTime += Gdx.graphics.getDeltaTime();

        StringBuilder textWithDots = new StringBuilder(text);
        for (int i = 0; i < Math.max(0, (int) (elapsedTime * 2) % 4); i++)
            textWithDots.append(".");

        float x = 3 * font.getLineHeight();
        batch.begin();
        font.draw(batch, textWithDots.toString(), x, x);
        batch.end();
    }

    public void renderBackground() {
        DefaultBackgroundRenderer renderer = DefaultBackgroundRenderer.getInstance();
        renderer.drawBlurredBackground();
    }

    @Override
    public void render(float delta) {
        if (simulationReady) {
            if (updatesCompleted >= updatesBeforeRendering) {
                graphicsAdapter.setSimulationScreen();
            } else {
                applicationManager.update();
                updatesCompleted++;
            }
        }
        renderBackground();
        Simulation simulation = applicationManager.getSimulation();
        if (simulation == null) {
            loadingString("Creating Simulation");
            return;
        }
        String loadingStatus = simulation.getLoadingStatus();
        if (loadingStatus != null)
            loadingString(loadingStatus);
        else
            loadingString("Loading Simulation");
    }

    public void notifySimulationReady() {
        simulationReady = true;
    }
}
