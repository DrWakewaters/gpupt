#[derive(Clone)]
pub struct Material {
    pub color: [f32; 3],
    pub emission: [f32; 3],
    pub specular_reflection_probability: f32,
    pub maximum_specular_angle: f32,
    pub refractive_index: f32,
    pub is_opaque: bool,
    pub is_lightsource: bool,
}

impl Material {
    pub fn new(color: [f32; 3], emission: [f32; 3], specular_reflection_probability: f32, maximum_specular_angle: f32, refractive_index: f32, is_opaque: bool, is_lightsource: bool) -> Self {
        Self {
            color,
            emission,
            specular_reflection_probability,
            maximum_specular_angle,
            refractive_index,
            is_opaque,
            is_lightsource,
        }
    }
}
