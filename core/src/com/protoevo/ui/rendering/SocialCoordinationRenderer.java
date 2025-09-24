package com.protoevo.ui.rendering;

import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Vector2;
import com.protoevo.biology.cells.Cell;
import com.protoevo.biology.nodes.SocialCoordinationReceptor;
import com.protoevo.biology.nodes.SurfaceNode;
import com.protoevo.utils.Colour;

/**
 * Renderer for SocialCoordinationReceptor that visualizes social behaviors
 * and group coordination with dynamic visual effects.
 */
public class SocialCoordinationRenderer extends NodeRenderer {
    
    private final Colour baseColor = new Colour();
    private final Colour groupColor = new Colour();
    private float activityLevel = 0f;
    private float groupCoordination = 0f;
    private float socialAnimation = 0f;
    
    private static final float ANIMATION_SPEED = 2f;
    
    public SocialCoordinationRenderer(SurfaceNode node) {
        super(node);
    }
    
    @Override
    public void render(float delta, SpriteBatch batch) {
        if (!node.exists())
            return;
            
        SocialCoordinationReceptor receptor = (SocialCoordinationReceptor) node.getAttachment();
        if (receptor == null)
            return;
            
        Cell cell = node.getCell();
        
        // Update animation timer
        socialAnimation += delta * ANIMATION_SPEED;
        if (socialAnimation > 2 * Math.PI) socialAnimation = 0f;
        
        // Get activity level from receptor's output
        float[] output = node.getOutputActivation();
        if (output.length > 0) {
            activityLevel = Math.abs(output[0]);
            if (output.length > 2) {
                groupCoordination = Math.abs(output[2]);
            }
        }
        
        // Update colors based on social state
        updateSocialColors(output);
        
        // Render the social coordination node
        renderSocialNode(batch, cell, delta);
    }
    
    private void updateSocialColors(float[] output) {
        // Base color indicates social role/behavior
        baseColor.set(0.9f, 0.5f, 0.2f, 1f); // Orange base for social activity
        
        if (output.length > 1) {
            // Modify color based on social behavior state
            float behaviorState = output[1];
            
            if (behaviorState > 0.5f) {
                // Leader/coordinator behaviors - warmer, brighter colors
                baseColor.set(1f, 0.7f, 0.1f, 1f); // Gold
            } else if (behaviorState > 0f) {
                // Active social behaviors - orange-red
                baseColor.set(1f, 0.5f, 0.2f, 1f);
            } else if (behaviorState > -0.5f) {
                // Follower behaviors - cooler colors
                baseColor.set(0.3f, 0.7f, 0.9f, 1f); // Blue
            } else {
                // Passive/territorial behaviors - purple
                baseColor.set(0.7f, 0.3f, 0.9f, 1f);
            }
        }
        
        // Group coordination color
        groupColor.set(0.2f, 0.9f, 0.6f, groupCoordination); // Green with alpha based on coordination
    }
    
    private void renderSocialNode(SpriteBatch batch, Cell cell, float delta) {
        batch.end();
        
        // Switch to shape renderer for custom graphics
        ShapeRenderer shapeRenderer = getShapeRenderer();
        shapeRenderer.setProjectionMatrix(batch.getProjectionMatrix());
        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        
        Vector2 nodePos = node.getWorldPosition();
        float baseRadius = cell.getRadius() * 0.2f * node.getAttachmentConstructionProgress();
        
        // Pulsing effect based on activity
        float pulseMultiplier = 1f + 0.4f * activityLevel * (float) Math.sin(socialAnimation);
        float mainRadius = baseRadius * pulseMultiplier;
        
        // Draw main social node
        shapeRenderer.setColor(baseColor.r, baseColor.g, baseColor.b, baseColor.a);
        shapeRenderer.circle(nodePos.x, nodePos.y, mainRadius);
        
        // Draw group coordination indicator
        if (groupCoordination > 0.2f) {
            renderGroupCoordinationEffects(shapeRenderer, nodePos, baseRadius, delta);
        }
        
        // Draw social network connections if highly active
        if (activityLevel > 0.5f) {
            renderSocialConnections(shapeRenderer, nodePos, cell, delta);
        }
        
        shapeRenderer.end();
        batch.begin();
    }
    
    private void renderGroupCoordinationEffects(ShapeRenderer shapeRenderer, Vector2 nodePos, 
                                              float baseRadius, float delta) {
        
        // Draw coordination rings
        float coordinationRadius = baseRadius * (2f + groupCoordination);
        float ringAnimation = (float) Math.sin(socialAnimation * 1.5f);
        
        // Outer coordination ring
        shapeRenderer.setColor(groupColor.r, groupColor.g, groupColor.b, 
                              groupColor.a * 0.6f * (0.5f + 0.5f * ringAnimation));
        shapeRenderer.circle(nodePos.x, nodePos.y, coordinationRadius, 
                           Math.max(8, (int)(coordinationRadius / 3)));
        
        // Inner coordination ring
        float innerRadius = coordinationRadius * 0.7f;
        shapeRenderer.setColor(groupColor.r, groupColor.g, groupColor.b, 
                              groupColor.a * 0.4f * (0.5f - 0.5f * ringAnimation));
        shapeRenderer.circle(nodePos.x, nodePos.y, innerRadius, 
                           Math.max(6, (int)(innerRadius / 4)));
    }
    
    private void renderSocialConnections(ShapeRenderer shapeRenderer, Vector2 nodePos, 
                                       Cell cell, float delta) {
        
        // Draw connection lines to nearby cells (simulating social network visualization)
        float connectionRange = cell.getRadius() * 3f;
        int connectionCount = 0;
        float connectionAlpha = activityLevel * 0.3f;
        
        shapeRenderer.setColor(baseColor.r, baseColor.g, baseColor.b, connectionAlpha);
        
        // This is a simplified version - in practice would need access to nearby cells
        // Draw radiating connection indicators instead
        int numConnections = Math.min(6, (int)(activityLevel * 8));
        for (int i = 0; i < numConnections; i++) {
            float angle = (float) (2 * Math.PI * i / numConnections + socialAnimation * 0.5f);
            float connectionLength = connectionRange * (0.5f + 0.5f * activityLevel);
            
            Vector2 endPoint = new Vector2(nodePos);
            endPoint.add((float) Math.cos(angle) * connectionLength, 
                        (float) Math.sin(angle) * connectionLength);
            
            // Draw connection line
            shapeRenderer.rectLine(nodePos.x, nodePos.y, endPoint.x, endPoint.y, 1f);
            
            // Draw connection endpoint
            shapeRenderer.circle(endPoint.x, endPoint.y, 2f);
        }
    }
    
    // Helper method to get shape renderer
    private ShapeRenderer getShapeRenderer() {
        // This would need to be implemented based on the existing rendering system
        return new ShapeRenderer();
    }
}