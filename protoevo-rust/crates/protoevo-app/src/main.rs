use bevy::prelude::*;
use protoevo_core::{biology::*, physics::create_particle_bundle, SimulationContext};
use protoevo_render::{RenderPlugin, Edible, Eater};
use bevy_rapier2d::prelude::*;
use rand::Rng;

fn main() {
    env_logger::init();
    
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RenderPlugin)
        .insert_resource(SimulationContext::default())
        .add_systems(Startup, setup_simulation)
        .run();
}

fn setup_simulation(mut commands: Commands) {
    let mut rng = rand::thread_rng();


    // Spawn plants
    for _ in 0..2000 {
        let radius = rng.gen_range(5.0..15.0);
        
        let x = rng.gen_range(-2000.0..2000.0);
        let y = rng.gen_range(-2000.0..2000.0);
        let pos = Vec2::new(x, y);
            
        let (rigid_body, collider, mass_props, damping, transform) = 
            create_particle_bundle(pos, radius, 1.0, 5.0);

        commands.spawn((
            PlantCell::new(radius),
            rigid_body,
            collider,
            mass_props,
            damping,
            transform,
            Edible, // Plants can be eaten
        ));
    
    }
    
    // Spawn protozoa
    for _ in 0..800 {
        let radius = rng.gen_range(10.0..20.0);
    
        let x = rng.gen_range(-2000.0..2000.0);
        let y = rng.gen_range(-2000.0..2000.0);
        let pos = Vec2::new(x, y);
        
        let (rigid_body, collider, mass_props, damping, transform) = 
            create_particle_bundle(pos, radius, 1.0, 2.0);

        let protozoan = Protozoan::new(radius);
        
        commands.spawn((
            protozoan,
            rigid_body,
            collider,
            mass_props,
            damping,
            transform,
            ExternalForce::default(),
            Eater, // Protozoa can eat
        ));
    }


}
