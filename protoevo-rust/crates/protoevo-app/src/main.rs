use bevy::prelude::*;
use protoevo_render::{RenderPlugin, PhysicsHandle};
use protoevo_core::SimulationContext;

fn main() {
    App::new()
        .insert_resource(SimulationContext::new())
        .add_plugins(RenderPlugin)
        .add_systems(Startup, setup_simulation)
        .add_systems(Update, step_physics)
        .run();
}

fn setup_simulation(
    mut commands: Commands,
    mut simulation: ResMut<SimulationContext>,
) {
    use bevy::prelude::Vec2;
    
    // Create several plants at different positions
    let plant_positions = vec![
        Vec2::new(-80.0, -60.0),
        Vec2::new(-40.0, 40.0),
        Vec2::new(0.0, -80.0),
        Vec2::new(60.0, 60.0),
    ];
    
    for pos in plant_positions {
        let plant = protoevo_core::biology::PlantCell::new(
            &mut simulation.physics,
            pos,
            8.0 + (pos.x.abs() % 5.0), // Varied sizes
        );
        commands.spawn(plant);
    }
    
    // Create several protozoa
    let protozoa_positions = vec![
        Vec2::new(50.0, 0.0),
        Vec2::new(-60.0, 30.0),
        Vec2::new(0.0, 70.0),
    ];
    
    for pos in protozoa_positions {
        let mut protozoan = protoevo_core::biology::Protozoan::new(
            &mut simulation.physics,
            pos,
            12.0,
        );
        // Give one protozoan low health to test death conversion
        if pos.y > 60.0 {
            protozoan.health = 5.0;
        }
        commands.spawn(protozoan);
    }
    
    // Add a couple meat cells to demonstrate decay
    let meat1 = protoevo_core::biology::MeatCell::new(
        &mut simulation.physics,
        Vec2::new(-70.0, -30.0),
        10.0,
        50.0, // Half health
    );
    commands.spawn(meat1);
    
    let meat2 = protoevo_core::biology::MeatCell::new(
        &mut simulation.physics,
        Vec2::new(30.0, -50.0),
        8.0,
        10.0, // Low health - will decay soon
    );
    commands.spawn(meat2);
}

fn step_physics(mut simulation: ResMut<SimulationContext>, time: Res<Time>) {
    // Step physics with fixed timestep or delta time
    // For now just step with delta
    simulation.physics.step(time.delta_seconds());
}
