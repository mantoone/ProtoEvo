package com.protoevo.env;

import com.badlogic.gdx.physics.box2d.*;
import com.protoevo.core.Particle;

import java.io.Serializable;

public class CollisionHandler implements ContactListener, Serializable {
    public static long serialVersionUID = 1L;

    private final Environment environment;

    public CollisionHandler(Environment environment) {
        this.environment = environment;
    }

    @Override
    public void beginContact(Contact contact) {
        Fixture fixtureA = contact.getFixtureA();
        Fixture fixtureB = contact.getFixtureB();
        Body bodyA = fixtureA.getBody();
        Body bodyB = fixtureB.getBody();
        bodyA.setAwake(true);
        bodyB.setAwake(true);

        if (fixtureA.isSensor() && bodyA.getUserData() instanceof Particle) {
            Particle particleA = (Particle) bodyA.getUserData();
            particleA.addInteractingObject(bodyB.getUserData());
        }
        else if (fixtureB.isSensor() && bodyB.getUserData() instanceof Particle) {
            Particle particleB = (Particle) bodyB.getUserData();
            particleB.addInteractingObject(bodyA.getUserData());
        }
        else {
            if (bodyA.getUserData() instanceof Particle)
                onContact(contact, (Particle) bodyA.getUserData(), bodyB);

            if (bodyB.getUserData() instanceof Particle)
                onContact(contact, (Particle) bodyB.getUserData(), bodyA);
        }
    }

    private void onContact(Contact contact, Particle particle, Body body) {
        if (body.getUserData() instanceof Particle) {
            Particle other = (Particle) body.getUserData();
            particle.onCollision(contact, other);
        } else if (body.getUserData() instanceof Rock) {
            Rock rock = (Rock) body.getUserData();
            particle.onCollision(contact, rock);
        }
    }

    @Override
    public void endContact(Contact contact) {
        Fixture fixtureA = contact.getFixtureA();
        Fixture fixtureB = contact.getFixtureB();
        Body bodyA = fixtureA.getBody();
        Body bodyB = fixtureB.getBody();

        if (fixtureA.isSensor() && bodyA.getUserData() instanceof Particle) {
            Particle particleA = (Particle) bodyA.getUserData();
            particleA.removeInteractingObject(bodyB.getUserData());
        }
        else if (fixtureB.isSensor() && bodyB.getUserData() instanceof Particle) {
            Particle particleB = (Particle) bodyB.getUserData();
            particleB.removeInteractingObject(bodyA.getUserData());
        }
    }

    @Override
    public void preSolve(Contact contact, Manifold oldManifold) {}

    @Override
    public void postSolve(Contact contact, ContactImpulse impulse) {}
}
