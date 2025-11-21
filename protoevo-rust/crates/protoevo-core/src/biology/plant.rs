use super::{Cell, CauseOfDeath};
use crate::physics::PhysicsWorld;
use rapier2d::prelude::*;
use glam::Vec2;
use bevy::prelude::*;

#[derive(Component)]
pub struct PlantCell {
    pub body_handle: RigidBodyHandle,
    pub collider_handle: ColliderHandle,
    pub radius: f32,
    pub health: f32,
    pub energy: f32,
    pub is_dead: bool,
}

impl PlantCell {
    pub fn new(
        physics: &mut PhysicsWorld,
        position: Vec2,
        radius: f32,
    ) -> Self {
        let (body_handle, collider_handle) = physics.create_particle(
            position,
            radius,
            1.0, // Density
            5.0, // Linear damping
        );

        Self {
            body_handle,
            collider_handle,
            radius,
            health: 100.0,
            energy: 100.0,
            is_dead: false,
        }
    }
}

impl Cell for PlantCell {
    fn update(&mut self, _delta: f32) {
        // Plant logic (photosynthesis, etc.)
        if self.health <= 0.0 {
            self.is_dead = true;
        }
    }

    fn physics_update(&mut self, _physics: &mut PhysicsWorld) {
        // Physics update if needed
    }

    fn get_pos(&self, physics: &PhysicsWorld) -> Vec2 {
        physics.get_position(self.body_handle).unwrap_or(Vec2::ZERO)
    }

    fn get_radius(&self) -> f32 {
        self.radius
    }

    fn is_dead(&self) -> bool {
        self.is_dead
    }

    fn kill(&mut self, _cause: CauseOfDeath) {
        self.is_dead = true;
    }

    fn get_body_handle(&self) -> Option<RigidBodyHandle> {
        Some(self.body_handle)
    }
}
