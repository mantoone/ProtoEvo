use bevy::prelude::*;
use protoevo_core::SimulationContext;
use rapier2d::prelude::*;

pub struct RenderPlugin;

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
        .add_systems(Startup, setup_camera)
        .add_systems(Update, sync_physics_transforms);
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
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
