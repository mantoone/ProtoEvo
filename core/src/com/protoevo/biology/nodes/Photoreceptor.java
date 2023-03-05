package com.protoevo.biology.nodes;

import com.badlogic.gdx.math.Vector2;
import com.protoevo.biology.cells.Cell;
import com.protoevo.core.Statistics;
import com.protoevo.env.Environment;
import com.protoevo.physics.Shape;
import com.protoevo.physics.Particle;
import com.protoevo.utils.Colour;
import com.protoevo.utils.Utils;


import java.io.Serializable;

public class Photoreceptor extends NodeAttachment implements Serializable {

    private static final long serialVersionUID = 1L;
    private final Vector2[] ray = new Vector2[]{new Vector2(), new Vector2()};
    private final Shape.Intersection[] intersections =
            new Shape.Intersection[]{new Shape.Intersection(), new Shape.Intersection()};
    private final Vector2 tmp = new Vector2(), tmp2 = new Vector2();
    private final Vector2 attachmentRelPos = new Vector2();
    private float interactionRange = 0;
    private final Colour colour = new Colour();
    private float r, g, b;
    private int rayIdx;
    private float minSqLen;
    public static final int nRays = 8;
    public static final float fov = (float) (Math.PI / 2.);

    public Photoreceptor(SurfaceNode node) {
        super(node);
    }

    @Override
    public void update(float delta, float[] input, float[] output) {
        interactionRange = getInteractionRange();
        attachmentRelPos.set(node.getRelativePos());
        castRays();
        output[0] = colour.r;
        output[1] = colour.g;
        output[2] = colour.b;
    }

    public Vector2[] nextRay() {
        float t;
        float dt = fov / nRays;
        if (rayIdx == 0)
            t = 0;
        else if (rayIdx % 2 == 0)
            t = (float) (dt * Math.floor(rayIdx / 2f));
        else
            t = (float) (-dt * Math.floor(rayIdx / 2f));

        minSqLen = Float.MAX_VALUE;

        ray[1].set(attachmentRelPos)
                .setLength(interactionRange)
                .rotateRad(t)
                .add(node.getCell().getPos());

        rayIdx++;
        return ray;
    }

    private void castRays() {
        ray[0].set(attachmentRelPos)
                .add(node.getCell().getPos());

        for (reset(); rayIdx < nRays; nextRay()) {
            Cell cell = node.getCell();
            for (Object o : cell.getInteractionQueue())
                if (o instanceof Shape)
                    computeIntersections((Shape) o);
        }

        colour.set(r / (nRays + 1), g / (nRays + 1), b / (nRays + 1), 1);
        reset();
    }

	public boolean cullFromRayCasting(Shape o) {
		if (o instanceof Particle) {
			Vector2 otherPos = ((Particle) o).getPos();
            Vector2 myPos = node.getCell().getPos();
			Vector2 dx = tmp.set(otherPos).sub(myPos).nor();
            Vector2 dir = tmp2.set(attachmentRelPos).add(myPos).nor();
			return dx.dot(dir) < Math.cos(fov / 2f);
		}
		return false;
	}

    public Shape.Intersection[] computeIntersections(Shape o) {
        intersections[0].didCollide = false;
        intersections[1].didCollide = false;
        boolean anyCollision = o.rayCollisions(ray, intersections);
        if (!anyCollision)
            return intersections;

        float sqLen = Float.MAX_VALUE;
        for (Shape.Intersection intersection : intersections)
            if (intersection.didCollide)
                sqLen = Math.min(sqLen, intersection.point.dst2(ray[0]));

        if (sqLen < minSqLen) {
            minSqLen = sqLen;
            float w = getConstructionProgress() * computeColourFalloffWeight();
            r += o.getColor().r * w;
            g += o.getColor().g * w;
            b += o.getColor().b * w;
        }

        return intersections;
    }

    public void reset() {
        r = 1; g = 1; b = 1;
        rayIdx = 0;
    }

    public float computeColourFalloffWeight() {
        float ir2 = getInteractionRange() * getInteractionRange();
        return Utils.clampedLinearRemap(minSqLen,0.5f * ir2, ir2,1, 0);
    }

    public Vector2[] getRay() {
        return ray;
    }

    public Colour getColour() {
        return colour;
    }

    @Override
    public float getInteractionRange() {
        if (node.getCell() == null)
            return 0;
        return Utils.clampedLinearRemap(
                node.getCell().getRadius(),
                Environment.settings.protozoa.minBirthRadius.get(), Environment.settings.maxParticleRadius.get(),
                node.getCell().getRadius() * 5f, Environment.settings.protozoa.maxLightRange.get());
    }

    @Override
    public String getName() {
        return "Photoreceptor";
    }

    @Override
    public String getInputMeaning(int index) {
        return null;  // no inputs
    }

    @Override
    public String getOutputMeaning(int index) {
        if (index == 0)
            return "R";
        else if (index == 1)
            return "G";
        else if (index == 2)
            return "B";
        return null;
    }

    @Override
    public void addStats(Statistics stats) {
        stats.putPercentage("Input: Red Light", colour.r);
        stats.putPercentage("Input: Green Light", colour.g);
        stats.putPercentage("Input: Blue Light", colour.b);
        stats.putDistance("Interaction Range", getInteractionRange());
        stats.put("FoV", fov, Statistics.ComplexUnit.ANGLE);
    }
}
