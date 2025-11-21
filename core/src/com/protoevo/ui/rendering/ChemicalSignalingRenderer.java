package com.protoevo.ui.rendering;

import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Vector2;
import com.protoevo.biology.cells.Cell;
import com.protoevo.biology.nodes.ChemicalSignalingReceptor;
import com.protoevo.biology.nodes.SurfaceNode;
import com.protoevo.utils.Colour;

/**
 * Renderer for ChemicalSignalingReceptor nodes that visualizes 
 * chemical signaling activity with dynamic colors and effects.
 */
public class ChemicalSignalingRenderer extends NodeRenderer {
    
    private final Colour signalColor = new Colour();
    private float signalIntensity = 0f;
    private float pulsation = 0f;
    private static final float PULSE_SPEED = 3f;
    
    public ChemicalSignalingRenderer(SurfaceNode node) {
        super(node);
    }
    
    @Override
    public void render(float delta, SpriteBatch batch) {
        if (!node.exists())
            return;
            
        ChemicalSignalingReceptor receptor = (ChemicalSignalingReceptor) node.getAttachment();
        if (receptor == null)
            return;
            
        Cell cell = node.getCell();
        
        // Update pulsation for visual effect
        pulsation += delta * PULSE_SPEED;
        if (pulsation > 2 * Math.PI) pulsation = 0f;
        
        // Get signal activity level from the receptor's output
        float[] output = node.getOutputActivation();
        if (output.length > 0) {
            signalIntensity = Math.abs(output[0]);
        }
        
        // Create visual representation based on signal activity
        updateSignalColor(output);
        
        // Render the receptor as a pulsating colored circle
        renderSignalingNode(batch, cell, delta);
    }
    
    private void updateSignalColor(float[] output) {
        // Base color (cyan for chemical signals)
        signalColor.set(0.2f, 0.8f, 0.9f, 1f);
        
        if (output.length > 1) {
            // Modify color based on signal type
            float signalType = output[1];
            
            // Map signal type to different hues
            if (signalType > 0.5f) {
                // Warmer colors for positive signal types (red-orange-yellow)
                signalColor.set(1f, 0.6f + 0.4f * signalType, 0.2f, 1f);
            } else if (signalType < -0.5f) {
                // Cooler colors for negative signal types (blue-purple)
                signalColor.set(0.2f + 0.3f * Math.abs(signalType), 0.3f, 1f, 1f);
            }
        }
        
        // Modulate intensity based on signal strength
        float alpha = 0.3f + 0.7f * signalIntensity;
        signalColor.a = alpha;
    }
    
    private void renderSignalingNode(SpriteBatch batch, Cell cell, float delta) {
        batch.end();
        
        // Switch to shape renderer for custom graphics
        ShapeRenderer shapeRenderer = getShapeRenderer();
        shapeRenderer.setProjectionMatrix(batch.getProjectionMatrix());
        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        
        Vector2 nodePos = node.getWorldPosition();
        float radius = cell.getRadius() * 0.25f * node.getAttachmentConstructionProgress(); // Increased from 0.15f to 0.25f
        
        // Add pulsing effect based on signal activity
        float pulseMultiplier = 1f + 0.3f * signalIntensity * (float) Math.sin(pulsation);
        radius *= pulseMultiplier;
        
        // Draw main signaling node
        shapeRenderer.setColor(signalColor.r, signalColor.g, signalColor.b, signalColor.a);
        shapeRenderer.circle(nodePos.x, nodePos.y, radius);
        
        // Draw signal transmission visualization if actively signaling
        if (signalIntensity > 0.1f) {
            renderSignalTransmission(shapeRenderer, nodePos, delta);
        }
        
        shapeRenderer.end();
        batch.begin();
    }
    
    private void renderSignalTransmission(ShapeRenderer shapeRenderer, Vector2 nodePos, float delta) {
        ChemicalSignalingReceptor receptor = (ChemicalSignalingReceptor) node.getAttachment();
        Cell cell = node.getCell();
        
        // Use a visual range that's just a bit bigger than the cell, not the actual interaction range
        float visualRange = cell.getRadius() * 2.5f; // Much smaller than the actual interaction range
        
        // Draw expanding signal waves
        float waveRadius = (float) (visualRange * 0.6f * Math.sin(pulsation * 0.7f));
        if (waveRadius > 0) {
            shapeRenderer.setColor(signalColor.r, signalColor.g, signalColor.b, 
                                 signalColor.a * 0.3f);
            shapeRenderer.circle(nodePos.x, nodePos.y, waveRadius, Math.max(8, (int)(waveRadius / 2)));
        }
        
        // Draw second wave for complex pattern
        float waveRadius2 = (float) (visualRange * 0.4f * Math.sin(pulsation * 1.2f + Math.PI/2));
        if (waveRadius2 > 0) {
            shapeRenderer.setColor(signalColor.r, signalColor.g, signalColor.b, 
                                 signalColor.a * 0.2f);
            shapeRenderer.circle(nodePos.x, nodePos.y, waveRadius2, Math.max(6, (int)(waveRadius2 / 2)));
        }
    }
    
    // Helper method to get shape renderer (assuming it exists in parent class or renderer system)
    private ShapeRenderer getShapeRenderer() {
        // This would need to be implemented based on the existing rendering system
        // For now, create a simple instance - in practice this should be managed by the renderer system
        return new ShapeRenderer();
    }
}