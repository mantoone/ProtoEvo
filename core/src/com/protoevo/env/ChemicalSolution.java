package com.protoevo.env;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.utils.ScreenUtils;
import com.google.common.collect.Streams;
import com.protoevo.biology.Cell;
import com.protoevo.biology.PlantCell;
import com.protoevo.core.settings.Settings;
import com.protoevo.core.settings.SimulationSettings;
import com.protoevo.utils.JCudaKernelRunner;
import com.protoevo.utils.Java2DTexture;
import com.protoevo.utils.Utils;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

public class ChemicalSolution implements Serializable {
    public static final long serialVersionUID = 1L;

    private final Environment environment;
    private final float cellSizeX, cellSizeY;
    private final float xMin, yMin, xMax, yMax;
    private final int chemicalTextureHeight;
    private final int chemicalTextureWidth;
    private int[] chemicalField, tmpChemicalField;
    private Texture chemicalTexture;
    private final Pixmap chemicalPixmap;
    private float timeSinceUpdate = 0;
    private final JCudaKernelRunner diffusionKernel;

    public ChemicalSolution(Environment environment, int cells, float mapRadius) {
        this(environment, -mapRadius, mapRadius, -mapRadius, mapRadius, cells);
    }

    public ChemicalSolution(Environment environment,
                            float xMin, float xMax,
                            float yMin, float yMax,
                            int cells) {
        this.environment = environment;

        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;

        this.chemicalTextureWidth = cells;
        this.chemicalTextureHeight = cells;

        this.cellSizeX = cells / (xMax - xMin);
        this.cellSizeY = cells / (yMax - yMin);

        chemicalPixmap = new Pixmap(cells, cells, Pixmap.Format.RGBA8888);
        chemicalPixmap.setColor(0, 0, 0, 0);
        chemicalTexture = new Texture(chemicalPixmap);

        chemicalField = new int[cells * cells * 4];
        for (int i = 0; i < chemicalField.length; i++) {
            chemicalField[i] = i % 4 == 3 ? 0 : 255;
        }

        tmpChemicalField = new int[cells * cells * 4];

        diffusionKernel = new JCudaKernelRunner("diffusion");

//        chemicalTexture = new Java2DTexture(cells, cells);
    }

    public float getFieldWidth() {
        return xMax - xMin;
    }

    public float getFieldHeight() {
        return yMax - yMin;
    }

    public Vector2 toEnvironmentCoords(int i, int j) {
        float x = xMin + (0.5f + i) * cellSizeX;
        float y = yMin + (0.5f + j) * cellSizeY;
        return new Vector2(x, y);
    }

    public int toChemicalGridXDist(float dist) {
        return (int) (dist * cellSizeX);
    }

    public int toChemicalGridX(float x) {
        return (int) Utils.linearRemap(x, xMin, xMax, 0, chemicalTextureWidth);
    }

    private int toFloatBufferIndex(int x, int y) {
        return (y * chemicalTextureWidth+ x) * 4;
    }

    public int toChemicalGridY(float y) {
        return (int) Utils.linearRemap(y, yMin, yMax, 0, chemicalTextureHeight);
    }

    public void depositChemicals(float delta, Cell e) {
        if (e instanceof PlantCell && !e.isDead()) {
            float k = Settings.plantPheromoneDeposit;
            float deposit = .5f; // Math.min(1f, delta * k * e.getRadius() * e.getHealth());
            // float deposit = delta * 1000f;
            Color cellColour = e.getColor();

            int fieldX = toChemicalGridX(e.getPos().x);
            int fieldY = toChemicalGridY(-e.getPos().y);

            // int r = toChemicalGridXDist(e.getRadius() * 0.8f);
            // depositPixmap.fillCircle(i, j, r);
            int i = toFloatBufferIndex(fieldX, fieldY);
            if (i >= 0 && i < chemicalField.length) {
                chemicalField[i] = (int) (255 * cellColour.r);
                chemicalField[i + 1] = (int) (255 * cellColour.g);
                chemicalField[i + 2] = (int) (255 * cellColour.b);
                chemicalField[i + 3] = (int) (255 * deposit);
            }
        }
    }

    public void deposit() {
        environment.getCells().parallelStream()
                .forEach(e -> depositChemicals(timeSinceUpdate, e));
    }

    public void diffuse() {
        diffusionKernel.processImage(
                chemicalField, tmpChemicalField, chemicalTextureWidth, chemicalTextureHeight);
        int[] tmp = chemicalField;
        chemicalField = tmpChemicalField;
        tmpChemicalField = tmp;
    }

    public void updateTexture() {
        chemicalPixmap.setColor(0, 0, 0 , 0);
        chemicalPixmap.fill();
        IntStream.range(0, chemicalField.length / 4).parallel()
                .forEach(i -> {
                    float r = chemicalField[i * 4] / 255f;
                    float g = chemicalField[i * 4 + 1] / 255f;
                    float b = chemicalField[i * 4 + 2] / 255f;
                    float a = chemicalField[i * 4 + 3] / 255f;
                    chemicalPixmap.drawPixel(
                            i % chemicalTextureWidth, i / chemicalTextureWidth,
                            Color.rgba8888(r, g, b, a));
                });
        chemicalTexture = new Texture(chemicalPixmap, Pixmap.Format.RGBA8888, false);
        timeSinceUpdate = 0;
    }

    public void update(float delta) {
        timeSinceUpdate += delta;
        if (timeSinceUpdate > SimulationSettings.chemicalDiffusionInterval) {
            deposit();
            diffuse();
            updateTexture();
            timeSinceUpdate = 0;
        }
    }

    public Texture getChemicalTexture(OrthographicCamera camera) {
        return chemicalTexture;
    }

    public int getNYChunks() {
        return chemicalTextureHeight;
    }

    public int getNXChunks() {
        return chemicalTextureWidth;
    }

    public float getPlantPheromoneDensity(int i, int j) {
//        Color depositColour = new Color(depositPixmap.getPixel(i, j));
        i = toFloatBufferIndex(i, j);
        if (i >= 0 && i < chemicalField.length) {
            return chemicalField[i + 3] / 255f;
        }
        return 0f;
    }

    public float getMinX() {
        return xMin;
    }

    public float getMaxX() {
        return xMax;
    }

    public float getMinY() {
        return yMin;
    }

    public float getMaxY() {
        return yMax;
    }
}
