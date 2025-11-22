use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use protoevo_core::biology::{Cell, PlantCell, Protozoan, MeatCell};

/// System to update cell sizes when they grow
pub fn update_cell_sizes(
    mut plant_query: Query<(&PlantCell, &mut Collider, &mut Sprite)>,
) {
    for (plant, mut collider, mut sprite) in plant_query.iter_mut() {
        // Update collider radius
        *collider = Collider::ball(plant.radius);
        
        // Update visual size
        if let Some(ref mut size) = sprite.custom_size {
            size.x = plant.radius * 2.0;
            size.y = plant.radius * 2.0;
        }
    }
}

/// Marker component for cells that can be eaten
#[derive(Component)]
pub struct Edible;

/// Marker component for cells that can eat
#[derive(Component)]
pub struct Eater;

/// System to handle feeding via collisions
pub fn handle_feeding(
    mut collision_events: EventReader<CollisionEvent>,
    mut eater_query: Query<(&mut Protozoan, &Transform), With<Eater>>,
    mut plant_query: Query<(&mut PlantCell, &Transform), (With<Edible>, Without<Protozoan>)>,
    mut meat_query: Query<(&mut MeatCell, &Transform), (With<Edible>, Without<Protozoan>, Without<PlantCell>)>,
    mut commands: Commands,
) {
    for collision in collision_events.read() {
        if let CollisionEvent::Started(e1, e2, _flags) = collision {
            // Try eater = e1, food = e2
            attempt_feeding(*e1, *e2, &mut eater_query, &mut plant_query, &mut meat_query, &mut commands);
            // Try eater = e2, food = e1
            attempt_feeding(*e2, *e1, &mut eater_query, &mut plant_query, &mut meat_query, &mut commands);
        }
    }
}

fn attempt_feeding(
    eater_entity: Entity,
    food_entity: Entity,
    eater_query: &mut Query<(&mut Protozoan, &Transform), With<Eater>>,
    plant_query: &mut Query<(&mut PlantCell, &Transform), (With<Edible>, Without<Protozoan>)>,
    meat_query: &mut Query<(&mut MeatCell, &Transform), (With<Edible>, Without<Protozoan>, Without<PlantCell>)>,
    commands: &mut Commands,
) {
    // Get eater info
    let Ok((mut eater, eater_transform)) = eater_query.get_mut(eater_entity) else { return };
    let eater_radius = eater.radius;
    
    // Try to eat a plant
    if let Ok((mut plant, _plant_transform)) = plant_query.get_mut(food_entity) {
        // Size rule: can only eat if significantly bigger
        if eater_radius > plant.radius * 1.2 {
            // Transfer energy and health
            let energy_gain = plant.energy * 0.7; // 70% efficiency
            let health_gain = plant.health * 0.5;
            
            eater.energy = (eater.energy + energy_gain).min(1000.0);
            eater.health = (eater.health + health_gain).min(100.0);
            
            // Kill the plant
            plant.is_dead = true;
            
            info!("Protozoan ate a plant! Gained {} energy", energy_gain);
        }
    }
    
    // Try to eat meat
    if let Ok((mut meat, _meat_transform)) = meat_query.get_mut(food_entity) {
        // Can always eat meat (meat is dead)
        let energy_gain = meat.health * 2.0; // Meat gives more energy
        let health_gain = meat.health * 0.3;
        
        eater.energy = (eater.energy + energy_gain).min(1000.0);
        eater.health = (eater.health + health_gain).min(100.0);
        
        // Kill the meat
        meat.is_dead = true;
        
        info!("Protozoan ate meat! Gained {} energy", energy_gain);
    }
}
