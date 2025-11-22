use super::{Cell, CauseOfDeath};
use bevy::prelude::*;

/// Dead organic matter component
#[derive(Component)]
pub struct MeatCell {
    pub radius: f32,
    pub health: f32,
    pub is_dead: bool,
}

impl MeatCell {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            health: 100.0,
            is_dead: false,
        }
    }
    
    /// Decay over time
    pub fn decay(&mut self, delta: f32) {
        const DECAY_RATE: f32 = 0.05; // 5% per second
        self.health -= self.health * DECAY_RATE * delta;
        
        if self.health < 0.01 {
            self.is_dead = true;
        }
    }
}

impl Cell for MeatCell {
    fn update(&mut self, delta: f32) {
        self.decay(delta);
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
}
