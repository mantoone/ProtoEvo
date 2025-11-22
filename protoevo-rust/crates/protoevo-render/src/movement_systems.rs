use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use protoevo_core::biology::Protozoan;

/// System to apply movement forces to protozoa
pub fn protozoan_movement(
    mut query: Query<(&mut ExternalForce, &Transform, &Protozoan)>,
    time: Res<Time>,
) {
    for (mut ext_force, transform, protozoan) in query.iter_mut() {
        if protozoan.energy < 20.0 {
            // Low energy - don't move
            continue;
        }
        
        let pos = transform.translation.truncate();
        
        // Simple movement pattern
        let distance_from_center = pos.length();
        let movement_force = if distance_from_center > 100.0 {
            // Move toward center
            -pos.normalize() * 50.0
        } else {
            // Random jitter
            let angle = (pos.x + pos.y + time.elapsed_secs() * protozoan.generation as f32).sin() * std::f32::consts::PI * 2.0;
            Vec2::new(angle.cos(), angle.sin()) * 20.0
        };
        
        ext_force.force = movement_force;
    }
}
