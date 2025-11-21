use glam::Vec2;
use crate::physics::PhysicsWorld;
use rapier2d::prelude::RigidBodyHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CauseOfDeath {
    Starvation,
    OldAge,
    Eaten,
    Environment,
    FailedToConstruct,
}

/// Trait defining common behavior for all cells
pub trait Cell {
    fn update(&mut self, delta: f32);
    fn physics_update(&mut self, physics: &mut PhysicsWorld);
    
    fn get_pos(&self, physics: &PhysicsWorld) -> Vec2;
    fn get_radius(&self) -> f32;
    
    fn is_dead(&self) -> bool;
    fn kill(&mut self, cause: CauseOfDeath);
    
    fn get_body_handle(&self) -> Option<RigidBodyHandle>;
}
