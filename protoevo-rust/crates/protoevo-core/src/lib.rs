pub mod settings;
pub mod math;
pub mod color;
pub mod physics;
pub mod biology;

use bevy::prelude::*;
use settings::Settings;

/// Main simulation context resource
#[derive(Resource)]
pub struct SimulationContext {
    pub settings: Settings,
}

impl Default for SimulationContext {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
        }
    }
}
