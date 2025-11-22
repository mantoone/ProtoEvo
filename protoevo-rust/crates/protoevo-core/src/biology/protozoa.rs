use super::{Cell, CauseOfDeath};
use bevy::prelude::*;

/// Protozoan cell component
#[derive(Component)]
pub struct Protozoan {
    pub radius: f32,
    pub health: f32,
    pub energy: f32,
    pub is_dead: bool,
    pub generation: u32,
}

impl Protozoan {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            health: 100.0,
            energy: 100.0,
            is_dead: false,
            generation: 1,
        }
    }
}

impl Cell for Protozoan {
    fn update(&mut self, delta: f32, _settings: &crate::settings::Settings) {
        // Energy consumption
        const ENERGY_CONSUMPTION_RATE: f32 = 10.0;
        self.energy -= delta * ENERGY_CONSUMPTION_RATE;
        
        // Clamp energy
        self.energy = self.energy.max(0.0).min(1000.0);
        
        // Death conditions
        if self.health <= 0.0 || self.energy <= 0.0 {
            self.is_dead = true;
        }
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
