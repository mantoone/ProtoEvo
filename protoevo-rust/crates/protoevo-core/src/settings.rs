use serde::{Deserialize, Serialize};

/// Main simulation settings structure
/// Ported from Java SimulationSettings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub simulation: SimulationSettings,
    pub worldgen: WorldGenSettings,
    pub physics: PhysicsSettings,
    pub environment: EnvironmentSettings,
    pub cell: CellSettings,
    pub protozoa: ProtozoaSettings,
    pub plant: PlantSettings,
    pub misc: MiscSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            simulation: SimulationSettings::default(),
            worldgen: WorldGenSettings::default(),
            physics: PhysicsSettings::default(),
            environment: EnvironmentSettings::default(),
            cell: CellSettings::default(),
            protozoa: ProtozoaSettings::default(),
            plant: PlantSettings::default(),
            misc: MiscSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSettings {
    pub update_delta: f32,
    pub target_fps: u32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            update_delta: 1.0 / 60.0,
            target_fps: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldGenSettings {
    pub radius: f32,
    pub void_start_distance: f32,
    pub chemical_field_resolution: usize,
    pub chemical_field_radius: f32,
    pub light_map_resolution: usize,
    pub num_initial_plant_pellets: usize,
    pub num_initial_protozoa: usize,
    pub num_population_start_clusters: usize,
    pub population_cluster_radius: f32,
    pub min_rock_cluster_radius: f32,
    pub bake_rock_lights: bool,
    pub generate_light_noise_texture: bool,
}

impl Default for WorldGenSettings {
    fn default() -> Self {
        Self {
            radius: 500.0,
            void_start_distance: 450.0,
            chemical_field_resolution: 256,
            chemical_field_radius: 500.0,
            light_map_resolution: 128,
            num_initial_plant_pellets: 100,
            num_initial_protozoa: 20,
            num_population_start_clusters: 3,
            population_cluster_radius: 100.0,
            min_rock_cluster_radius: 50.0,
            bake_rock_lights: true,
            generate_light_noise_texture: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsSettings {
    pub velocity_iterations: usize,
    pub position_iterations: usize,
    pub gravity_x: f32,
    pub gravity_y: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            velocity_iterations: 1,
            position_iterations: 1,
            gravity_x: 0.0,
            gravity_y: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    pub fluid_drag_dampening: f32,
    pub chemical_diffusion_interval: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            fluid_drag_dampening: 1.0,
            chemical_diffusion_interval: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellSettings {
    pub chemical_extraction_factor: f32,
    pub chemical_extraction_plant_conversion: f32,
    pub chemical_extraction_meat_conversion: f32,
}

impl Default for CellSettings {
    fn default() -> Self {
        Self {
            chemical_extraction_factor: 0.1,
            chemical_extraction_plant_conversion: 1.0,
            chemical_extraction_meat_conversion: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtozoaSettings {
    pub evolution_enabled: bool,
    pub engulf_eating_rate_multiplier: f32,
}

impl Default for ProtozoaSettings {
    fn default() -> Self {
        Self {
            evolution_enabled: true,
            engulf_eating_rate_multiplier: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantSettings {
    pub evolution_enabled: bool,
}

impl Default for PlantSettings {
    fn default() -> Self {
        Self {
            evolution_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiscSettings {
    pub use_gpu_compute: bool,
    pub chemical_cpu_iterations: usize,
}

impl Default for MiscSettings {
    fn default() -> Self {
        Self {
            use_gpu_compute: true,
            chemical_cpu_iterations: 1000,
        }
    }
}

impl Settings {
    /// Load settings from a TOML file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let settings = toml::from_str(&contents)?;
        Ok(settings)
    }

    /// Save settings to a TOML file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_string)?;
        Ok(())
    }
}
