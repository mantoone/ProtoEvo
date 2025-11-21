use rapier2d::prelude::*;
use rapier2d::geometry::{DefaultBroadPhase, NarrowPhase};
use crate::Settings;
use glam::Vec2;

pub struct PhysicsWorld {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub gravity: Vector<Real>,
}

impl PhysicsWorld {
    pub fn new(settings: &Settings) -> Self {
        let gravity = vector![
            settings.physics.gravity_x,
            settings.physics.gravity_y
        ];

        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            gravity,
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
    }

    /// Create a circular particle (rigid body + collider)
    /// Returns the handles for the body and collider
    pub fn create_particle(
        &mut self,
        position: Vec2,
        radius: f32,
        density: f32,
        linear_damping: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y])
            .linear_damping(linear_damping)
            .build();
        
        let collider = ColliderBuilder::ball(radius)
            .density(density)
            .restitution(0.2) // Bounciness
            .build();

        let body_handle = self.rigid_body_set.insert(rigid_body);
        let collider_handle = self.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut self.rigid_body_set,
        );

        (body_handle, collider_handle)
    }

    pub fn get_body(&self, handle: RigidBodyHandle) -> Option<&RigidBody> {
        self.rigid_body_set.get(handle)
    }

    pub fn get_body_mut(&mut self, handle: RigidBodyHandle) -> Option<&mut RigidBody> {
        self.rigid_body_set.get_mut(handle)
    }

    pub fn get_position(&self, handle: RigidBodyHandle) -> Option<Vec2> {
        self.rigid_body_set.get(handle).map(|rb| {
            let t = rb.translation();
            Vec2::new(t.x, t.y)
        })
    }
    
    pub fn get_velocity(&self, handle: RigidBodyHandle) -> Option<Vec2> {
        self.rigid_body_set.get(handle).map(|rb| {
            let v = rb.linvel();
            Vec2::new(v.x, v.y)
        })
    }
}
