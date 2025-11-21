use bevy::prelude::*;
use protoevo_core::biology::{Cell, PlantCell, Protozoan, MeatCell, CauseOfDeath};

/// System to update all cells
pub fn update_cells(
    mut plant_query: Query<&mut PlantCell>,
    mut protozoan_query: Query<&mut Protozoan>,
    mut meat_query: Query<&mut MeatCell>,
    time: Res<Time>,
) {
    let delta = time.delta_seconds();
    
    // Update plants
    for mut plant in plant_query.iter_mut() {
        plant.update(delta);
    }
    
    // Update protozoa
    for mut protozoan in protozoan_query.iter_mut() {
        protozoan.update(delta);
    }
    
    // Update meat (decay)
    for mut meat in meat_query.iter_mut() {
        meat.update(delta);
    }
}

/// System to handle dead cells and convert them to meat
pub fn handle_cell_death(
    mut commands: Commands,
    mut simulation: ResMut<protoevo_core::SimulationContext>,
    plant_query: Query<(Entity, &PlantCell)>,
    protozoan_query: Query<(Entity, &Protozoan)>,
    meat_query: Query<(Entity, &MeatCell)>,
) {
    // Check for dead plants
    for (entity, plant) in plant_query.iter() {
        if plant.is_dead() {
            let pos = plant.get_pos(&simulation.physics);
            let radius = plant.get_radius();
            
            // Create meat cell from dead plant
            let meat = MeatCell::new(
                &mut simulation.physics,
                pos,
                radius * 0.8, // Slightly smaller
                plant.health,
            );
            
            // Despawn plant and spawn meat
            commands.entity(entity).despawn();
            commands.spawn(meat);
        }
    }
    
    // Check for dead protozoa
    for (entity, protozoan) in protozoan_query.iter() {
        if protozoan.is_dead() {
            let pos = protozoan.get_pos(&simulation.physics);
            let radius = protozoan.get_radius();
            
            // Create meat cell from dead protozoan
            let meat = MeatCell::new(
                &mut simulation.physics,
                pos,
                radius * 0.8,
                protozoan.health,
            );
            
            commands.entity(entity).despawn();
            commands.spawn(meat);
        }
    }
    
    // Fully decayed meat should be removed
    for (entity, meat) in meat_query.iter() {
        if meat.is_dead() {
            commands.entity(entity).despawn();
        }
    }
}
