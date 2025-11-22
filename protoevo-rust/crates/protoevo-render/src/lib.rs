use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::window::PrimaryWindow;
use protoevo_compute::ChemicalFieldCompute;
use protoevo_core::biology::{PlantCell, Protozoan, MeatCell};
use protoevo_core::physics::{PhysicsConfig, create_particle_bundle};
use rand::Rng;

mod cell_systems;
use cell_systems::{update_cells, handle_cell_death, handle_plant_reproduction};

mod ui_systems;
use ui_systems::{spawn_energy_bars, update_energy_bars};

mod movement_systems;
use movement_systems::protozoan_movement;

mod feeding_systems;
pub use feeding_systems::{handle_feeding, update_cell_sizes, Edible, Eater};

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app
        // Add rapier physics plugin
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .insert_resource(PhysicsConfig::default())
        
        // Window setup
        .add_systems(Startup, setup_window)
        
        // Update systems
        .add_systems(Update, (
            camera_control,
            update_cells,
            update_cell_sizes,
            protozoan_movement,
            handle_feeding,
            handle_cell_death,
            handle_plant_reproduction,
            spawn_plant_visuals,
            spawn_protozoa_visuals,
            spawn_meat_visuals,
            spawn_energy_bars,
            update_energy_bars,
            disable_gravity,
        ));
    }
}

fn disable_gravity(mut commands: Commands, query: Query<Entity, Added<RigidBody>>) {
    for entity in query.iter() {
        commands.entity(entity).insert(GravityScale(0.0));
    }
}



fn setup_window(mut commands: Commands) {
    // Spawn 2D camera
    commands.spawn(Camera2dBundle::default()).insert(CameraController {
        pan_sensitivity: 1.0,
        last_cursor_pos: None,
    });
}

/// Camera controller component
#[derive(Component)]
struct CameraController {
    pan_sensitivity: f32,
    last_cursor_pos: Option<Vec2>,
}

/// System for camera controls
fn camera_control(
    mut query: Query<(&mut Transform, &mut OrthographicProjection, &mut CameraController)>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
) {
    let (mut transform, mut projection, mut controller) = query.single_mut();
    let Ok(window) = window_query.get_single() else { return };
    
    // Pan with middle mouse button
    if mouse_button.pressed(MouseButton::Middle) {
        if let Some(cursor_pos) = window.cursor_position() {
            if let Some(last_pos) = controller.last_cursor_pos {
                let delta = cursor_pos - last_pos;
                transform.translation.x -= delta.x * projection.scale * controller.pan_sensitivity;
                transform.translation.y += delta.y * projection.scale * controller.pan_sensitivity;
            }
            controller.last_cursor_pos = Some(cursor_pos);
        }
    } else {
        controller.last_cursor_pos = None;
    }
    
    //Zoom with scroll wheel
    for wheel in mouse_wheel.read() {
        let zoom_delta = -wheel.y * 0.1;
        projection.scale = (projection.scale + zoom_delta).max(0.1).min(10.0);
    }
}

/// Spawn visual for PlantCell
fn spawn_plant_visuals(
    mut commands: Commands,
    query: Query<Entity, Added<PlantCell>>,
) {
    let mut rng = rand::thread_rng();
    for entity in query.iter() {
        let r = rng.gen_range(0.1..0.4);
        let g = rng.gen_range(0.5..0.9);
        let b = rng.gen_range(0.0..0.3);
        commands.entity(entity).insert(
            Sprite {
                color: Color::srgb(r, g, b),
                custom_size: Some(Vec2::new(10.0, 10.0)),
                ..default()
            },
        );
    }
}

/// Spawn visual for Protozoan
fn spawn_protozoa_visuals(
    mut commands: Commands,
    query: Query<Entity, Added<Protozoan>>,
) {
    for entity in query.iter() {
        commands.entity(entity).insert(
            Sprite {
                color: Color::srgb(0.9, 0.9, 0.9),
                custom_size: Some(Vec2::new(12.0, 12.0)),
                ..default()
            },
        );
    }
}

/// Spawn visual for MeatCell
fn spawn_meat_visuals(
    mut commands: Commands,
    query: Query<(Entity, &MeatCell), Added<MeatCell>>,
) {
    for (entity, meat) in query.iter() {
        let alpha = (meat.health / 100.0).max(0.1);
        commands.entity(entity).insert(
            Sprite {
                color: Color::srgba(0.6, 0.4, 0.3, alpha),
                custom_size: Some(Vec2::new(10.0, 10.0)),
                ..default()
            },
        );
    }
}
