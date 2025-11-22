use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use protoevo_core::biology::{Cell, PlantCell, Protozoan, MeatCell};
use protoevo_core::SimulationContext;

/// System to update all cells
pub fn update_cells(
    time: Res<Time>,
    mut plant_query: Query<&mut PlantCell>,
    mut protozoan_query: Query<&mut Protozoan>,
    mut meat_query: Query<&mut MeatCell>,
) {
    let delta = time.delta_secs();
    
    // Update plants
    for mut plant in plant_query.iter_mut() {
        plant.update(delta);
    }
    
    // Update protozoa
    for mut protozoan in protozoan_query.iter_mut() {
        protozoan.update(delta);
    }
    
    // Update meat
    for mut meat in meat_query.iter_mut() {
        meat.update(delta);
    }
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
