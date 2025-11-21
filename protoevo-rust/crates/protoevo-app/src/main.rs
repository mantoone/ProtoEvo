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
    // Create a test particle
    let (body_handle, _) = simulation.physics.create_particle(
        Vec2::new(0.0, 0.0),
        10.0, // Radius
        1.0,  // Density
        0.0,  // Damping
    );

    // Spawn visual representation
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.0, 1.0, 0.0),
                custom_size: Some(Vec2::new(20.0, 20.0)), // 2x radius
                ..default()
            },
            ..default()
        },
        PhysicsHandle(body_handle),
    ));
}

fn step_physics(mut simulation: ResMut<SimulationContext>, time: Res<Time>) {
    // Step physics with fixed timestep or delta time
    // For now just step with delta
    simulation.physics.step(time.delta_seconds());
}
