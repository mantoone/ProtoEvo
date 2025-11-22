use bevy::prelude::*;

/// Trait for all cell types
pub trait Cell {
    /// Update cell state (called every frame)
    fn update(&mut self, delta: f32);
    
    /// Get cell radius
    fn get_radius(&self) -> f32;
    
    /// Check if cell is dead
    fn is_dead(&self) -> bool;
    
    /// Kill the cell
    fn kill(&mut self, cause: CauseOfDeath);
}

/// Causes of cell death
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CauseOfDeath {
    HealthTooLow,
    EnergyDepleted,
    Starvation,
    Overcrowding,
    Old,
}
