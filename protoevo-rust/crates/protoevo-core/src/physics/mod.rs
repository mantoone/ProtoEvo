// Physics module now uses bevy_rapier2d plugin
// The plugin manages the physics world internally via Bevy's ECS

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

/// Physics configuration for the simulation
#[derive(Resource, Clone)]
pub struct PhysicsConfig {
    pub gravity: Vec2,
    pub timestep_mode: TimestepMode,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0.0, 0.0), // No gravity for 2D simulation
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / 60.0,
                substeps: 1,
            },
        }
    }
}

/// Helper function to create a dynamic circular rigid body with collider
pub fn create_particle_bundle(
    position: Vec2,
    radius: f32,
    density: f32,
    linear_damping: f32,
) -> (
    RigidBody,
    Collider,
    ColliderMassProperties,
    Damping,
    Transform,
) {
    (
        RigidBody::Dynamic,
        Collider::ball(radius),
        ColliderMassProperties::Density(density),
        Damping {
            linear_damping,
            angular_damping: 1.0,
        },
        Transform::from_xyz(position.x, position.y, 0.0),
    )
}
