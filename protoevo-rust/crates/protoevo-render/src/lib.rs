use bevy::prelude::*;
use protoevo_core::biology::{PlantCell, Protozoan, MeatCell};
use protoevo_core::SimulationContext;
use rapier2d::prelude::*;

use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::TextureUsages;
use bevy::render::texture::GpuImage;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::{Render, RenderApp, RenderSet};
use protoevo_compute::ChemicalFieldCompute;

mod cell_systems;
use cell_systems::{update_cells, handle_cell_death};

mod ui_systems;
use ui_systems::{spawn_energy_bars, update_energy_bars};

mod movement_systems;
use movement_systems::protozoan_movement;

pub struct RenderPlugin;

#[derive(Resource, Clone, ExtractResource)]
pub struct ChemicalFieldImage(pub Handle<Image>);

#[derive(Resource)]
pub struct ChemicalFieldResource(pub ChemicalFieldCompute);

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ProtoEvo Rust".into(),
                resolution: (1280.0, 720.0).into(),
                present_mode: bevy::window::PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.1)))
        .add_plugins(ExtractResourcePlugin::<ChemicalFieldImage>::default())
        .add_systems(Startup, (setup_camera, setup_chemical_field_main))
        .add_systems(Update, (
            camera_control,
            update_cells,
            protozoan_movement,
            handle_cell_death,
            sync_physics_transforms,
            spawn_plant_visuals,
            spawn_protozoa_visuals,
            spawn_meat_visuals,
            spawn_energy_bars,
            update_energy_bars,
        ));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(Startup, setup_chemical_field_render)
                .add_systems(Render, update_chemical_field_render.in_set(RenderSet::Prepare));
        }
    }
}

#[derive(Component)]
pub struct CameraController {
    pub zoom_speed: f32,
    pub pan_speed: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            zoom_speed: 0.1,
            pan_speed: 1.0,
        }
    }
}

fn camera_control(
    mut query: Query<(&mut Transform, &mut OrthographicProjection, &CameraController)>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
    time: Res<Time>,
) {
    for (mut transform, mut projection, controller) in query.iter_mut() {
        // Zoom with scroll wheel
        for event in mouse_wheel.read() {
            let zoom_delta = -event.y * controller.zoom_speed;
            projection.scale *= 1.0 + zoom_delta;
            projection.scale = projection.scale.clamp(0.1, 10.0);
        }

        // Pan with middle mouse button
        if mouse_button.pressed(MouseButton::Middle) {
            for event in mouse_motion.read() {
                let pan_delta = Vec2::new(-event.delta.x, event.delta.y);
                transform.translation.x += pan_delta.x * projection.scale * controller.pan_speed;
                transform.translation.y += pan_delta.y * projection.scale * controller.pan_speed;
            }
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera2dBundle::default(),
        CameraController::default(),
    ));
}

#[derive(Component)]
pub struct PhysicsHandle(pub RigidBodyHandle);

fn sync_physics_transforms(
    simulation: Res<SimulationContext>,
    mut query: Query<(&mut Transform, &PhysicsHandle)>,
) {
    for (mut transform, handle) in query.iter_mut() {
        if let Some(body) = simulation.physics.rigid_body_set.get(handle.0) {
            let translation = body.translation();
            transform.translation.x = translation.x;
            transform.translation.y = translation.y;
            // transform.rotation = Quat::from_rotation_z(body.rotation().angle());
        }
    }
}

fn setup_chemical_field_main(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let size = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        bevy::render::render_resource::TextureDimension::D2,
        &[0; 16], // 16 bytes for Rgba32Float (4 * 4 bytes)
        bevy::render::render_resource::TextureFormat::Rgba32Float,
        bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING;

    let handle = images.add(image);

    commands.spawn(SpriteBundle {
        texture: handle.clone(),
        sprite: Sprite {
            custom_size: Some(Vec2::new(1280.0, 1280.0)),
            ..default()
        },
        ..default()
    });

    commands.insert_resource(ChemicalFieldImage(handle));
}

fn setup_chemical_field_render(
    mut commands: Commands,
    device: Res<RenderDevice>,
) {
    let compute = ChemicalFieldCompute::new(device.wgpu_device(), 512, 512);
    commands.insert_resource(ChemicalFieldResource(compute));
    bevy::log::info!("ChemicalFieldResource initialized in Render App");
}

fn update_chemical_field_render(
    chemical_field: Option<ResMut<ChemicalFieldResource>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    image_handle: Res<ChemicalFieldImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    if let Some(mut chemical_field) = chemical_field {
        if let Some(gpu_image) = gpu_images.get(&image_handle.0) {
            chemical_field.0.step(device.wgpu_device(), &queue);
            chemical_field.0.copy_to_texture(device.wgpu_device(), &queue, &gpu_image.texture);
        }
    } else {
        // bevy::log::warn!("ChemicalFieldResource missing in Render App");
    }
}

fn spawn_plant_visuals(
    mut commands: Commands,
    query: Query<(Entity, &PlantCell), Added<PlantCell>>,
) {
    for (entity, plant) in query.iter() {
        commands.entity(entity).insert((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb(0.2, 0.8, 0.2),
                    custom_size: Some(Vec2::new(plant.radius * 2.0, plant.radius * 2.0)),
                    ..default()
                },
                ..default()
            },
            PhysicsHandle(plant.body_handle),
        ));
    }
}

fn spawn_protozoa_visuals(
    mut commands: Commands,
    query: Query<(Entity, &Protozoan), Added<Protozoan>>,
) {
    for (entity, protozoan) in query.iter() {
        commands.entity(entity).insert((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb(1.0, 1.0, 1.0),
                    custom_size: Some(Vec2::new(protozoan.radius * 2.0, protozoan.radius * 2.0)),
                    ..default()
                },
                ..default()
            },
            PhysicsHandle(protozoan.body_handle),
        ));
    }
}

fn spawn_meat_visuals(
    mut commands: Commands,
    query: Query<(Entity, &MeatCell), Added<MeatCell>>,
) {
    for (entity, meat) in query.iter() {
        // Brown/gray color with health-based alpha
        let health_factor = (meat.health / 100.0).clamp(0.0, 1.0);
        let base_color = Color::srgb(0.6, 0.3, 0.2); // Brown
        commands.entity(entity).insert((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::srgba(
                        base_color.to_srgba().red,
                        base_color.to_srgba().green,
                        base_color.to_srgba().blue,
                        0.3 + 0.7 * health_factor,
                    ),
                    custom_size: Some(Vec2::new(meat.radius * 2.0, meat.radius * 2.0)),
                    ..default()
                },
                ..default()
            },
            PhysicsHandle(meat.body_handle),
        ));
    }
}
