// Chemical field compute temporarily disabled
// Bevy 0.15's render_resource API has significant changes
// Will be re-enabled in future update

pub struct ChemicalFieldCompute;

impl ChemicalFieldCompute {
    pub fn new(_device: &(), _queue: &(), _width: u32, _height: u32) -> Self {
        Self
    }
    
    pub fn step(&mut self, _device: &(), _queue: &(), _dt: f32) {
        // No-op
    }
    
    pub fn copy_to_texture(&self, _device: &(), _queue: &(), _texture: &()) {
        // No-op  
    }
}
