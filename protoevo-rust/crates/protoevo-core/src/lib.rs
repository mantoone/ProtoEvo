pub mod settings;
pub mod math;
pub mod color;
pub mod physics;
pub mod biology;

pub use settings::Settings;
pub use color::Color;
pub use rapier2d;

use bevy::prelude::*;
use crate::physics::PhysicsWorld;

#[derive(Resource)]
pub struct SimulationContext {
    pub physics: PhysicsWorld,
    pub settings: Settings,
}

impl SimulationContext {
    pub fn new() -> Self {
        let settings = Settings::default();
        Self {
            physics: PhysicsWorld::new(&settings),
            settings,
        }
    }
}
