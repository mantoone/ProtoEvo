use super::{Cell, CauseOfDeath};
use crate::physics::PhysicsWorld;
use rapier2d::prelude::*;
use glam::Vec2;
use bevy::prelude::*;

/// Dead organic matter that decays over time and releases nutrients
#[derive(Component)]
pub struct MeatCell {
    pub body_handle: RigidBodyHandle,
    pub collider_handle: ColliderHandle,
    pub radius: f32,
    pub health: f32,
    pub decay_rate: f32,
    pub is_dead: bool,
}

impl MeatCell {
    pub fn new(
        physics: &mut PhysicsWorld,
        position: Vec2,
        radius: f32,
        initial_health: f32,
    ) -> Self {
        let (body_handle, collider_handle) = physics.create_particle(
            position,
            radius,
            0.5, // Lower density for dead matter
            10.0, // High damping (more viscous)
        );

        Self {
            body_handle,
            collider_handle,
            radius,
            health: initial_health,
            decay_rate: 0.05, // 5% health lost per second
            is_dead: false,
        }
    }

    pub fn decay(&mut self, delta: f32) {
        // Exponential decay based on Java implementation
        let decay_amount = self.health * self.decay_rate * delta;
        self.health -= decay_amount;
        
        if self.health <= 0.01 {
            self.is_dead = true;
        }
    }
}

impl Cell for MeatCell {
    fn update(&mut self, delta: f32) {
        self.decay(delta);
        
        if self.health <= 0.0 {
            self.is_dead = true;
        }
    }

    fn physics_update(&mut self, _physics: &mut PhysicsWorld) {
        // MeatCell has no active physics behavior
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
