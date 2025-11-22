use super::{Cell, CauseOfDeath};
use bevy::prelude::*;

/// Plant cell component
#[derive(Component)]
pub struct PlantCell {
    pub radius: f32,
    pub health: f32,
    pub energy: f32,
    pub is_dead: bool,
}

impl PlantCell {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            health: 100.0,
            energy: 100.0,
            is_dead: false,
        }
    }
    
    /// Try to grow if there's excess energy
    pub fn try_grow(&mut self, delta: f32) -> Option<f32> {
        const GROWTH_ENERGY_THRESHOLD: f32 = 500.0;
        const GROWTH_ENERGY_COST: f32 = 100.0;
        const GROWTH_RATE: f32 = 2.0; // Units per second
        const MAX_RADIUS: f32 = 30.0;
        
        if self.energy > GROWTH_ENERGY_THRESHOLD && self.radius < MAX_RADIUS {
            let growth = GROWTH_RATE * delta;
            let new_radius = (self.radius + growth).min(MAX_RADIUS);
            let actual_growth = new_radius - self.radius;
            
            if actual_growth > 0.0 {
                // Cost energy proportional to growth
                let energy_cost = (actual_growth / growth) * GROWTH_ENERGY_COST * delta;
                self.energy -= energy_cost;
                self.radius = new_radius;
                return Some(new_radius);
            }
        }
        
        None
    }
}

impl Cell for PlantCell {
    fn update(&mut self, delta: f32) {
        // Photosynthesis: convert light to energy
        const LIGHT_LEVEL: f32 = 0.8;
        const MAX_RADIUS: f32 = 50.0;
        const PHOTOSYNTHESIS_RATE: f32 = 300.0;
        
        let area = std::f32::consts::PI * self.radius * self.radius;
        let max_area = std::f32::consts::PI * MAX_RADIUS * MAX_RADIUS;
        let photo_rate = LIGHT_LEVEL * (area / max_area);
        let energy_gain = delta * photo_rate * PHOTOSYNTHESIS_RATE;
        
        self.energy += energy_gain;
        
        // Energy decay (metabolism)
        const ENERGY_DECAY_RATE: f32 = 5.0;
        self.energy -= delta * ENERGY_DECAY_RATE;
        
        // Clamp energy
        self.energy = self.energy.max(0.0).min(1000.0);
        
        // Try to grow
        self.try_grow(delta);
        
        // Death condition
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
