package com.protoevo.input;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.protoevo.physics.Particle;
import com.protoevo.ui.SimulationInputManager;
import com.protoevo.ui.SimulationScreen;
import com.protoevo.utils.CursorUtils;

import java.util.Optional;

public class MoveParticle extends InputAdapter {
    private final SimulationScreen simulationScreen;
    private final OrthographicCamera camera;
    private final MoveParticleButton moveParticleButton;
    private final ParticleTracker particleTracker;
    private Particle grabbedParticle;
    private final Vector2 lastMousePos = new Vector2(0, 0);
    private final Vector2 currentMousePos = new Vector2(0, 0);
    private final Vector2 mouseVel = new Vector2(0, 0);
    private boolean jediMode = true;

    public MoveParticle(SimulationScreen simulationScreen,
                        SimulationInputManager inputManager,
                        ParticleTracker particleTracker) {
        this.simulationScreen = simulationScreen;
        this.camera = simulationScreen.getCamera();
        this.moveParticleButton = inputManager.getMoveParticleButton();
        this.particleTracker = particleTracker;
    }

    public boolean pickUpEnabled() {
        if (Gdx.input.isKeyPressed(Input.Keys.CONTROL_LEFT)) {
            moveParticleButton.setHoldingControl(true);
            return true;
        } else {
            moveParticleButton.setHoldingControl(false);
        }

        return moveParticleButton.getState() == MoveParticleButton.State.CAN_HOLD;
    }

    @Override
    public boolean keyDown(int keycode) {
        if (keycode == Input.Keys.CONTROL_LEFT) {
            moveParticleButton.setHoldingControl(true);
        }
        return false;
    }

    @Override
    public boolean keyUp(int keycode) {
        if (keycode == Input.Keys.CONTROL_LEFT) {
            moveParticleButton.setHoldingControl(false);
        }
        return false;
    }

    @Override
    public boolean touchDown(int screenX, int screenY, int pointer, int button) {
        if (Gdx.input.isButtonPressed(Input.Buttons.LEFT) && pickUpEnabled()) {
            Vector3 worldSpace = camera.unproject(new Vector3(screenX, screenY, 0));
            mouseVel.set(0, 0);
            lastMousePos.set(worldSpace.x, worldSpace.y);
            Optional<Particle> particle = simulationScreen.getEnvironment().getParticles()
                    .filter(p -> p.getPos().dst(worldSpace.x, worldSpace.y) < p.getRadius())
                    .findAny();
            if (particle.isPresent()) {
                grabbedParticle = particle.get();
                moveParticleButton.setState(MoveParticleButton.State.HOLDING);
                return true;
            }
        }
        return false;
    }

    @Override

    public boolean touchDragged(int screenX, int screenY, int pointer) {
        if (simulationScreen.hasSimulationNotLoaded())
            return false;

        if (grabbedParticle != null) {
            if (grabbedParticle.equals(particleTracker.getTrackedParticle()))
                return false;

            if (!moveParticleButton.isHolding())
                moveParticleButton.setState(MoveParticleButton.State.HOLDING);

            CursorUtils.setClosedHandCursor();

            Vector3 worldSpace = camera.unproject(new Vector3(screenX, screenY, 0));
            currentMousePos.set(worldSpace.x, worldSpace.y);
            mouseVel.set(currentMousePos).sub(lastMousePos).scl(1 / Gdx.graphics.getDeltaTime());
            lastMousePos.set(worldSpace.x, worldSpace.y);

            if (jediMode) {
                Vector2 impulse = new Vector2(mouseVel).scl(.001f);
                grabbedParticle.applyImpulse(impulse);
            } else {
                grabbedParticle.setPos(lastMousePos);
                grabbedParticle.setVel(0, 0);
                grabbedParticle.setAngularVel(0);
            }
            return true;

        } else if (moveParticleButton.isHolding()) {
            moveParticleButton.revertToLastState();
        }
        return false;
    }

    @Override
    public boolean touchUp(int screenX, int screenY, int pointer, int b) {
        if (b == Input.Buttons.LEFT) {
            if (grabbedParticle != null) {
                Vector3 worldSpace = camera.unproject(new Vector3(screenX, screenY, 0));
                currentMousePos.set(worldSpace.x, worldSpace.y);

                if (mouseVel.len2() > 5000) {
                    // add a little velocity to the particle
                    Vector2 impulse = mouseVel.scl(.005f);
                    grabbedParticle.applyImpulse(impulse);
                }
            }
            grabbedParticle = null;
            if (moveParticleButton.isHolding()) {
                moveParticleButton.revertToLastState();
            }
        }
        return false;
    }

    public void toggleJediMode() {
        this.jediMode = !this.jediMode;
    }

    public boolean isJediMode() {
        return jediMode;
    }
}
