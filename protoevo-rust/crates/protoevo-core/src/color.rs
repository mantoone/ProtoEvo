use serde::{Deserialize, Serialize};

/// RGBA color representation (values 0.0 to 1.0)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    pub const fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Convert to RGBA8888 format (u32)
    pub fn to_rgba8888(&self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        let a = (self.a.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 24) | (g << 16) | (b << 8) | a
    }

    /// Create from RGBA8888 format
    pub fn from_rgba8888(rgba: u32) -> Self {
        Self {
            r: ((rgba >> 24) & 0xFF) as f32 / 255.0,
            g: ((rgba >> 16) & 0xFF) as f32 / 255.0,
            b: ((rgba >> 8) & 0xFF) as f32 / 255.0,
            a: (rgba & 0xFF) as f32 / 255.0,
        }
    }

    /// Subtract a value from all channels
    pub fn sub(&mut self, value: f32) {
        self.r = (self.r - value).max(0.0);
        self.g = (self.g - value).max(0.0);
        self.b = (self.b - value).max(0.0);
        self.a = (self.a - value).max(0.0);
    }

    /// Get a specific channel by index (0=r, 1=g, 2=b, 3=a)
    pub fn get(&self, axis: usize) -> f32 {
        match axis {
            0 => self.r,
            1 => self.g,
            2 => self.b,
            3 => self.a,
            _ => 0.0,
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::transparent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8888_conversion() {
        let color = Color::new(1.0, 0.5, 0.25, 1.0);
        let rgba = color.to_rgba8888();
        let converted = Color::from_rgba8888(rgba);
        
        assert!((converted.r - color.r).abs() < 0.01);
        assert!((converted.g - color.g).abs() < 0.01);
        assert!((converted.b - color.b).abs() < 0.01);
    }
}
