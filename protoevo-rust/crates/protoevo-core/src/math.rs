use glam::Vec2;
use rand::Rng;

/// Geometry utilities for 2D space
pub mod geometry {
    use super::*;

    /// Generate a random point within a circle of given radius
    pub fn random_point_in_circle(radius: f32, rng: &mut impl Rng) -> Vec2 {
        let angle = rng.gen::<f32>() * std::f32::consts::TAU;
        let r = radius * rng.gen::<f32>().sqrt();
        Vec2::new(r * angle.cos(), r * angle.sin())
    }

    /// Generate a random unit vector
    pub fn random_vector(magnitude: f32, rng: &mut impl Rng) -> Vec2 {
        let angle = rng.gen::<f32>() * std::f32::consts::TAU;
        Vec2::new(magnitude * angle.cos(), magnitude * angle.sin())
    }

    /// Calculate the overlap area between a box and a circle
    pub fn box_and_circle_intersection_overlap(
        box_x_min: f32,
        box_x_max: f32,
        box_y_min: f32,
        box_y_max: f32,
        circle_x: f32,
        circle_y: f32,
        circle_radius: f32,
    ) -> f32 {
        // Simplified approximation for performance
        // Find closest point on box to circle center
        let closest_x = circle_x.clamp(box_x_min, box_x_max);
        let closest_y = circle_y.clamp(box_y_min, box_y_max);
        
        let dx = circle_x - closest_x;
        let dy = circle_y - closest_y;
        let dist_sq = dx * dx + dy * dy;
        
        if dist_sq > circle_radius * circle_radius {
            return 0.0;
        }
        
        // Estimate overlap (simplified)
        let box_area = (box_x_max - box_x_min) * (box_y_max - box_y_min);
        let circle_area = std::f32::consts::PI * circle_radius * circle_radius;
        
        // Return approximate overlap
        box_area.min(circle_area) * (1.0 - (dist_sq / (circle_radius * circle_radius)).sqrt())
    }

    /// Clamp and linearly remap a value from one range to another
    pub fn clamped_linear_remap(
        value: f32,
        from_min: f32,
        from_max: f32,
        to_min: f32,
        to_max: f32,
    ) -> f32 {
        let t = ((value - from_min) / (from_max - from_min)).clamp(0.0, 1.0);
        to_min + t * (to_max - to_min)
    }
}

/// Common mathematical functions
pub mod functions {
    /// Clamp and linearly remap (re-export from geometry for compatibility)
    pub use super::geometry::clamped_linear_remap;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_random_point_in_circle() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let point = geometry::random_point_in_circle(10.0, &mut rng);
            assert!(point.length() <= 10.0);
        }
    }

    #[test]
    fn test_clamped_linear_remap() {
        assert_eq!(geometry::clamped_linear_remap(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert_eq!(geometry::clamped_linear_remap(-5.0, 0.0, 10.0, 0.0, 100.0), 0.0);
        assert_eq!(geometry::clamped_linear_remap(15.0, 0.0, 10.0, 0.0, 100.0), 100.0);
    }
}
