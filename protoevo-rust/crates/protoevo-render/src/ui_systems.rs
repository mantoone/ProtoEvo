use bevy::prelude::*;
use protoevo_core::biology::{PlantCell, Protozoan, MeatCell};

/// Component to display energy/health bars above cells
#[derive(Component)]
pub struct EnergyBar {
    pub target_entity: Entity,
}

/// System to spawn energy bars for new cells
pub fn spawn_energy_bars(
    mut commands: Commands,
    plant_query: Query<Entity, Added<PlantCell>>,
    protozoan_query: Query<Entity, Added<Protozoan>>,
) {
    // Add energy bars to plants
    for entity in plant_query.iter() {
        commands.entity(entity).with_children(|parent| {
            // Energy bar background (red)
            parent.spawn(SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb(0.2, 0.0, 0.0),
                    custom_size: Some(Vec2::new(20.0, 3.0)),
                    ..default()
                },
                transform: Transform::from_xyz(0.0, 15.0, 1.0),
                ..default()
            });
            
            // Energy bar foreground (green - dynamic)
            parent.spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::srgb(0.0, 1.0, 0.0),
                        custom_size: Some(Vec2::new(20.0, 3.0)),
                        ..default()
                    },
                    transform: Transform::from_xyz(0.0, 15.0, 2.0),
                    ..default()
                },
                EnergyBar {
                    target_entity: entity,
                },
            ));
        });
    }
    
    // Add energy bars to protozoa
    for entity in protozoan_query.iter() {
        commands.entity(entity).with_children(|parent| {
            // Energy bar background
            parent.spawn(SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb(0.2, 0.0, 0.0),
                    custom_size: Some(Vec2::new(25.0, 3.0)),
                    ..default()
                },
                transform: Transform::from_xyz(0.0, 18.0, 1.0),
                ..default()
            });
            
            // Energy bar foreground (blue - dynamic)
            parent.spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::srgb(0.0, 0.5, 1.0),
                        custom_size: Some(Vec2::new(25.0, 3.0)),
                        ..default()
                    },
                    transform: Transform::from_xyz(0.0, 18.0, 2.0),
                    ..default()
                },
                EnergyBar {
                    target_entity: entity,
                },
            ));
        });
    }
}

/// System to update energy bar sizes based on cell energy
pub fn update_energy_bars(
    mut bar_query: Query<(&EnergyBar, &mut Sprite, &mut Transform)>,
    plant_query: Query<&PlantCell>,
    protozoan_query: Query<&Protozoan>,
) {
    for (energy_bar, mut sprite, mut transform) in bar_query.iter_mut() {
        let energy_percent = if let Ok(plant) = plant_query.get(energy_bar.target_entity) {
            (plant.energy / 1000.0).clamp(0.0, 1.0)
        } else if let Ok(protozoan) = protozoan_query.get(energy_bar.target_entity) {
            (protozoan.energy / 1000.0).clamp(0.0, 1.0)
        } else {
            continue;
        };
        
        // Update bar width
        if let Some(ref mut size) = sprite.custom_size {
            let max_width = if plant_query.get(energy_bar.target_entity).is_ok() {
                20.0
            } else {
                25.0
            };
            size.x = max_width * energy_percent;
            
            // Adjust position to keep left-aligned
            transform.translation.x = -max_width / 2.0 + size.x / 2.0;
        }
    }
}
