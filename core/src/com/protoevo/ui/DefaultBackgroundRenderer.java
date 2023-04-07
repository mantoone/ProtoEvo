package com.protoevo.ui;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.PixmapIO;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.utils.ScreenUtils;
import com.protoevo.env.Environment;
import com.protoevo.settings.EnvironmentSettings;
import com.protoevo.ui.rendering.EnvironmentRenderer;

public class DefaultBackgroundRenderer {

    private static DefaultBackgroundRenderer instance;

    public static DefaultBackgroundRenderer getInstance() {
        if (instance == null) {
            instance = new DefaultBackgroundRenderer();
        }
        return instance;
    }

    private final Environment environment;
    private final EnvironmentSettings settings;
    private final EnvironmentRenderer environmentRenderer;
    private final OrthographicCamera camera;
    private final FrameBuffer fbo;
    private Sprite sprite;
    private final SpriteBatch batch;
    private final ShaderProgram shader;
    private boolean simulate = true;

    public DefaultBackgroundRenderer() {

        float graphicsHeight = Gdx.graphics.getHeight();
        float graphicsWidth = Gdx.graphics.getWidth();

        camera = new OrthographicCamera();
        camera.setToOrtho(
                false, Environment.settings.world.radius.get(),
                Environment.settings.world.radius.get() * graphicsHeight / graphicsWidth);
        camera.position.set(0, 0, 0);
        camera.zoom = 1f;

        settings = DefaultBackgroundGenerator.createBgEnvSettings();
        environment = new Environment(settings);
        environment.initialise();

        environmentRenderer = new EnvironmentRenderer(camera, environment, null);

        fbo = new FrameBuffer(Pixmap.Format.RGBA8888,
                Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), false);

        batch = new SpriteBatch();
        shader = new ShaderProgram(
                Gdx.files.internal("shaders/pause/vertex.glsl"),
                Gdx.files.internal("shaders/pause/fragment.glsl"));
        if (!shader.isCompiled()) {
            throw new RuntimeException("Shader compilation failed: " + shader.getLog());
        }
    }

    public void pauseSimulation() {
        simulate = false;
    }

    public void resumeSimulation() {
        if (!simulate)
            environment.rebuildWorld();
        simulate = true;
    }

    public void simulateEnv() {
        if (simulate)
            environment.update(settings.simulationUpdateDelta.get());
    }

    public void render(Float delta) {
        ScreenUtils.clear(EnvironmentRenderer.backgroundColor);

        simulateEnv();

        camera.update();
        FrameBufferManager fboManager = FrameBufferManager.getInstance();
        fboManager.begin(fbo);
        environmentRenderer.render(delta);
        fboManager.end();

        drawBlurredBackground();
    }

    public void drawBlurredBackground() {
        sprite = new Sprite(fbo.getColorBufferTexture());
        sprite.flip(false, true);

        shader.bind();
        shader.setUniformf("u_resolution", Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        shader.setUniformf("u_blurAmount", 12f);
        shader.setUniformf("u_darkenAmount", 0.75f);
        batch.setShader(shader);

        batch.begin();
        batch.setColor(1f, 1f, 1f, 0.7f);
        batch.draw(sprite, 0, 0, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        batch.end();
    }

    public void dispose() {
        fbo.dispose();
        batch.dispose();
        shader.dispose();
    }

    public Sprite getRenderedSprite() {
        return sprite;
    }

    public static void disposeInstance() {
        if (instance != null) {
            instance.dispose();
            instance = null;
        }
    }
}
