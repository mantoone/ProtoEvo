use bevy::prelude::*;
use protoevo_core::biology::Protozoan;
use protoevo_core::SimulationContext;
use rapier2d::prelude::*;

/// System to apply movement forces to protozoa
pub fn protozoan_movement(
    mut query: Query<&mut Protozoan>,
    mut simulation: ResMut<SimulationContext>,
    time: Res<Time>,
) {
    let delta = time.delta_seconds();
    
    for mut protozoan in query.iter_mut() {
        // Simple random movement for now
        // Later: implement gradient following
        
        if protozoan.energy < 20.0 {
            // Low energy - move slower or not at all
            continue;
        }
        
        // Get current position
        let pos = if let Some(rigid_body) = simulation.physics.rigid_body_set.get(protozoan.body_handle) {
            let translation = rigid_body.translation();
            Vec2::new(translation.x, translation.y)
        } else {
            continue;
        };
        
        // Simple movement pattern: move toward center if far away
        let distance_from_center = pos.length();
        let movement_force = if distance_from_center > 100.0 {
            // Move toward center
            -pos.normalize() * 50.0
        } else {
            // Random jitter
            let angle = (pos.x + pos.y + time.elapsed_seconds() * protozoan.generation as f32).sin() * std::f32::consts::PI * 2.0;
            Vec2::new(angle.cos(), angle.sin()) * 20.0
        };
        
        // Apply force to rigid body
        if let Some(rigid_body) = simulation.physics.rigid_body_set.get_mut(protozoan.body_handle) {
            rigid_body.add_force(vector![movement_force.x, movement_force.y], true);
        }
        
        // Movement costs energy
        const MOVEMENT_ENERGY_COST: f32 = 5.0;
        protozoan.energy -= delta * MOVEMENT_ENERGY_COST;
    }
}
