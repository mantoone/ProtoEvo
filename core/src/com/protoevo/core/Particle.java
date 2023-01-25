package com.protoevo.core;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.*;
import com.badlogic.gdx.utils.Array;
import com.protoevo.biology.CauseOfDeath;
import com.protoevo.core.settings.Settings;
import com.protoevo.core.settings.SimulationSettings;
import com.protoevo.env.CollisionHandler;
import com.protoevo.env.Environment;
import com.protoevo.env.Rock;
import com.protoevo.utils.Geometry;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;


public class Particle implements Shape {

    private final Vector2[] boundingBox = new Vector2[]{new Vector2(), new Vector2()};
    private Environment environment;
    private Body body;
    private Fixture dynamicsFixture, sensorFixture;
    private boolean dead = false, disposed = false;
    private float radius = SimulationSettings.minParticleRadius;
    private final Vector2 pos = new Vector2(0, 0);
    private final ConcurrentHashMap<String, Float> stats = new ConcurrentHashMap<>();
    private final Collection<CollisionHandler.FixtureCollision> contacts = new ConcurrentLinkedQueue<>();
    private final Collection<Object> interactionObjects = new ConcurrentLinkedQueue<>();
    private CauseOfDeath causeOfDeath = null;

    public Particle() {}

    public Environment getEnv() {
        return environment;
    }

    public void setEnv(Environment environment) {
        this.environment = environment;
        Vector2 pos = environment.getRandomPosition(this);
        if (pos == null) {
            kill(CauseOfDeath.FAILED_TO_CONSTRUCT);
            return;
        }
        createBody();
        setPos(pos);
        environment.ensureAddedToEnvironment(this);
    }

    public void update(float delta) {
        if (body != null)
            pos.set(body.getPosition());

        if (getSpeed() < getRadius() / 50f) {
            body.setLinearVelocity(0, 0);
            body.setAwake(false);
        }
        dynamicsFixture.getShape().setRadius(radius);
        float interactionRange = getInteractionRange();
        if (canPossiblyInteract() && interactionRange > getRadius())
            sensorFixture.getShape().setRadius(interactionRange);

        interactionObjects.removeIf(o -> (o instanceof Particle) && ((Particle) o).isDead());
        contacts.removeIf(c -> (getOther(c) instanceof Particle) && ((Particle) getOther(c)).isDead());
    }

    public void createBody() {
        if (body != null || disposed)
            return;

        CircleShape circle = new CircleShape();
        circle.setRadius(radius);

        BodyDef bodyDef = new BodyDef();
        bodyDef.type = BodyDef.BodyType.DynamicBody;

        // Body consists of a dynamics fixture and a sensor fixture
        // The dynamics fixture is used for collisions and the sensor fixture is used for
        // detecting when the particle in interaction range with other objects
        body = environment.getWorld().createBody(bodyDef);

        // Create the dynamics fixture and attach it to the body
        FixtureDef fixtureDef = new FixtureDef();
        fixtureDef.shape = circle;
        fixtureDef.density = 1f;  // will be updated later
        fixtureDef.friction = 0.9f;
        fixtureDef.restitution = 0.2f;
        fixtureDef.filter.categoryBits = ~FixtureCategories.SENSOR;

        dynamicsFixture = body.createFixture(fixtureDef);
        dynamicsFixture.setUserData(this);

        body.setUserData(this);
        body.setLinearDamping(5f);
        body.setAngularDamping(5f);
        body.setSleepingAllowed(true);

        circle.dispose();

        if (canPossiblyInteract()) {
            // Create the sensor fixture and attach it to the body
            CircleShape interactionCircle = new CircleShape();
            interactionCircle.setRadius(radius / 2f);

            FixtureDef sensorFixtureDef = new FixtureDef();
            sensorFixtureDef.shape = interactionCircle;
            sensorFixtureDef.isSensor = true;
            sensorFixtureDef.filter.categoryBits = FixtureCategories.SENSOR;
            sensorFixtureDef.filter.maskBits = ~FixtureCategories.SENSOR;

            sensorFixture = body.createFixture(sensorFixtureDef);
            sensorFixture.setUserData(this);
            sensorFixture.setSensor(true);

            interactionCircle.dispose();
        }
    }

    /**
     * The category used for the dynamic fixture to determine
     * which sensors will interact with it.
     * @return the category bits
     */
    public short getSensorCategory() {
        return FixtureCategories.SENSOR;
    }

    /**
     * The mask used for the sensor fixture to determine
     * which dynamic fixtures will interact with it.
     * @return the mask bits
     */
    public short getSensorMask() {
        return ~FixtureCategories.SENSOR;
    }

    public Body getBody() {
        return body;
    }

    public void applyForce(Vector2 force) {
        if (body != null && !disposed)
            body.applyForceToCenter(force, true);
    }

    public void applyImpulse(Vector2 impulse) {
        if (body != null && !disposed)
            body.applyLinearImpulse(impulse, body.getWorldCenter(), true);
    }

    public float getInteractionRange() {
        return 0;
    }

    public void addInteractingObject(Object object) {
        interactionObjects.add(object);
    }

    public void removeInteractingObject(Object object) {
        interactionObjects.remove(object);
    }

    public Collection<Object> getInteractionQueue() {
        return interactionObjects;
    }

    public boolean canPossiblyInteract() {
        return false;
    }

    public void interact(List<Object> interactions) {}

    public void onCollision(CollisionHandler.FixtureCollision collision, Particle other) {
        if (getOther(collision) != null)
            contacts.add(collision);
    }

    public void onCollision(CollisionHandler.FixtureCollision collision, Rock rock) {
        if (getOther(collision) != null)
            contacts.add(collision);
    }

    public Object getOther(CollisionHandler.FixtureCollision collision) {
        return collision.objB;
    }

    public void reset() {
        contacts.clear();
    }

    public Collection<CollisionHandler.FixtureCollision> getContacts() {
        return contacts;
    }

    public float getRadius() {
        return radius;
    }

    public void setRadius(float radius) {
        this.radius = Math.max(SimulationSettings.minParticleRadius, radius);
        this.radius = Math.min(SimulationSettings.maxParticleRadius, this.radius);
        if (dynamicsFixture != null && !disposed) {
            dynamicsFixture.getShape().getRadius();
        }
    }

    public boolean isPointInside(Vector2 point) {
        float r = getRadius();
        return point.dst2(getPos()) < r*r;
    }

    public void setPos(Vector2 pos) {
        this.pos.set(pos);
        if (body != null && !disposed) {
            body.setTransform(pos, 0);
        }
    }

    public float getAngle() {
        if (body == null || disposed)
            return 0;
        return body.getAngle();
    }

    @Override
    public Vector2 getPos() {
        return pos;
    }

    public Vector2 getVel() {
        return body.getLinearVelocity();
    }

    public float getSpeed() {
        return getVel().len();
    }

    public float getMass() {
        return getMass(getRadius());
    }

    public float getMass(float r) {
        return getMass(r, 0);
    }

    public float getMass(float r, float extraMass) {
        return Geometry.getCircleArea(r) * getMassDensity() + extraMass;
    }

    public float getMassDensity() {
        return 1f;
    }

    @Override
    public boolean pointInside(Vector2 p) {
        return Geometry.isPointInsideCircle(getPos(), getRadius(), p);
    }

    @Override
    public boolean rayIntersects(Vector2 start, Vector2 end) {
        return false;
    }

    @Override
    public boolean rayCollisions(Vector2[] ray, Collision[] collision) {
        Vector2 start = ray[0], end = ray[1];
        float dirX = end.x - start.x, dirY = end.y - start.y;
        Vector2 p = getPos();

        float a = start.dst2(end);
        float b = 2 * (dirX*(start.x - p.x) + dirY*(start.y - p.y));
        float c = p.len2() + start.len2() - getRadius() * getRadius() - 2 * p.dot(start);

        float d = b*b - 4*a*c;
        if (d == 0)
            return false;

        float t1 = (float) ((-b + Math.sqrt(d)) / (2*a));
        float t2 = (float) ((-b - Math.sqrt(d)) / (2*a));

        boolean anyCollisions = false;
        if (0 <= t1 && t1 <= 1) {
            collision[0].point.set(start).lerp(end, t1);
            collision[0].didCollide = true;
            anyCollisions = true;
        }
        if (0 <= t2 && t2 <= 1) {
            collision[1].point.set(start).lerp(end, t2);
            collision[1].didCollide = true;
            anyCollisions = true;
        }
        return anyCollisions;
    }

    @Override
    public Color getColor() {
        return Color.WHITE;
    }

    @Override
    public Vector2[] getBoundingBox() {
        float x = getPos().x;
        float y = getPos().y;
        float r = getRadius();
        boundingBox[0].set(x - r, y - r);
        boundingBox[1].set(x + r, y + r);
        return boundingBox;
    }

    public boolean isCollidingWith(Rock rock) {
        Vector2[][] edges = rock.getEdges();
        float r = getRadius();
        Vector2 pos = getPos();

        if (rock.pointInside(pos))
            return true;

        for (Vector2[] edge : edges) {
            if (Geometry.doesLineIntersectCircle(edge, pos, r))
                return true;
        }
        return false;
    }

    public boolean isCollidingWith(Particle other)
    {
        if (other == this)
            return false;
        float r = getRadius() + other.getRadius();
        return other.getPos().dst2(getPos()) < r*r;
    }

    public Map<String, Float> getStats() {
        stats.clear();
        stats.put("Size", Settings.statsDistanceScalar * getRadius());
        stats.put("Speed", Settings.statsDistanceScalar * getSpeed());
        stats.put("Total Mass", Settings.statsMassScalar * getMass());
        stats.put("Mass Density", Settings.statsMassScalar * getMassDensity());
        return stats;
    }

    public boolean isDead() {
        return dead;
    }

    public void kill(CauseOfDeath causeOfDeath) {
        dead = true;
        if (this.causeOfDeath == null)
            this.causeOfDeath = causeOfDeath;
    }

    public void dispose() {
        if (disposed)
            return;
        disposed = true;
        kill(CauseOfDeath.DISPOSED);
        environment.getWorld().destroyBody(body);
    }

    public String getPrettyName() {
        return "Particle";
    }

    public Map<String, Float> getDebugStats() {
        Map<String, Float> stats = new TreeMap<>();
        stats.put("Position X", Settings.statsDistanceScalar * getPos().x);
        stats.put("Position Y", Settings.statsDistanceScalar * getPos().y);
        stats.put("Inertia", body.getInertia());
        stats.put("Num Joints", (float) body.getJointList().size);
        stats.put("Num Fixtures", (float) body.getFixtureList().size);
        stats.put("Num Contacts", (float) contacts.size());
        stats.put("Num Interactions", (float) interactionObjects.size());
        stats.put("Is Dead", dead ? 1f : 0f);
        stats.put("Is Sleeping", body.isAwake() ? 0f : 1f);
        stats.put("Local Count", (float) environment.getLocalCount(this));
        stats.put("Local Cap", (float) environment.getLocalCapacity(this));
        return stats;
    }

    public Array<JointEdge> getJoints() {
        return body.getJointList();
    }

    public CauseOfDeath getCauseOfDeath() {
        return causeOfDeath;
    }

    public float getArea() {
        return Geometry.getCircleArea(getRadius());
    }

    public void applyTorque(float v) {
        body.applyTorque(v, true);
    }

}
