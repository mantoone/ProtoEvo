use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use protoevo_core::biology::{Cell, PlantCell, Protozoan, MeatCell};
use protoevo_core::SimulationContext;
use protoevo_core::settings::Settings;

/// System to update all cells
pub fn update_cells(
    time: Res<Time>,
    settings: Res<Settings>,
    mut plant_query: Query<&mut PlantCell>,
    mut protozoan_query: Query<&mut Protozoan>,
    mut meat_query: Query<&mut MeatCell>,
) {
    let delta = time.delta_secs();
    
    // Update plants
    for mut plant in plant_query.iter_mut() {
        plant.update(delta, &settings);
    }
    
    // Update protozoa
    for mut protozoan in protozoan_query.iter_mut() {
        protozoan.update(delta, &settings);
    }
    
    // Update meat
    for mut meat in meat_query.iter_mut() {
        meat.update(delta, &settings);
    }
}

/// System to handle plant reproduction
pub fn handle_plant_reproduction(
    mut commands: Commands,
    plant_query: Query<(Entity, &PlantCell, &Transform)>,
    settings: Res<Settings>,
) {
    for (entity, plant, transform) in plant_query.iter() {
        if plant.should_split(&settings) {
             let num_children = 3;
             let current_radius = plant.radius;
             // Area conservation: pi * r^2 = n * pi * r_child^2
             // r_child = r / sqrt(n)
             let child_radius = current_radius / (num_children as f32).sqrt();
             let pos = transform.translation.truncate();
             
             for _ in 0..num_children {
                 let offset = Vec2::new(
                     rand::random::<f32>() - 0.5,
                     rand::random::<f32>() - 0.5
                 ).normalize_or_zero() * (current_radius * 0.5);
                 
                 spawn_plant_at_position(&mut commands, pos + offset, child_radius, plant.max_radius);
             }
             
             commands.entity(entity).despawn_recursive();
        }
    }
}

fn spawn_plant_at_position(commands: &mut Commands, position: Vec2, radius: f32, max_radius: f32) {
    use protoevo_core::physics::create_particle_bundle;
    
    let (rigid_body, collider, mass_props, damping, transform) =
        create_particle_bundle(position, radius, 0.5, 8.0);
    
    commands.spawn((
        PlantCell::new(radius, max_radius),
        rigid_body,
        collider,
        mass_props,
        damping,
        transform,
    ));
}

/// System to handle cell death and conversion to meat
pub fn handle_cell_death(
    mut commands: Commands,
    plant_query: Query<(Entity, &PlantCell, &Transform)>,
    protozoan_query: Query<(Entity, &Protozoan, &Transform)>,
    meat_query: Query<(Entity, &MeatCell)>,
) {
    // Check plant deaths
    for (entity, plant, transform) in plant_query.iter() {
        if plant.is_dead() {
            let pos = transform.translation.truncate();
            commands.entity(entity).despawn_recursive();
            
            // Spawn meat cell
            spawn_meat_at_position(&mut commands, pos, plant.radius);
        }
    }
    
    // Check protozoan deaths
    for (entity, protozoan, transform) in protozoan_query.iter() {
        if protozoan.is_dead() {
            let pos = transform.translation.truncate();
            commands.entity(entity).despawn_recursive();
            
            // Spawn meat cell
            spawn_meat_at_position(&mut commands, pos, protozoan.radius);
        }
    }
    
    // Check meat decay
    for (entity, meat) in meat_query.iter() {
        if meat.is_dead() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn spawn_meat_at_position(commands: &mut Commands, position: Vec2, radius: f32) {
    use protoevo_core::physics::create_particle_bundle;
    
    let (rigid_body, collider, mass_props, damping, transform) =
        create_particle_bundle(position, radius, 0.5, 8.0);
    
    commands.spawn((
        MeatCell::new(radius),
        rigid_body,
        collider,
        mass_props,
        damping,
        transform,
    ));
}
