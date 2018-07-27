use std::fmt::{Display, Formatter, Result};

use material::Material;

#[allow(dead_code)]
pub struct Sphere {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
    emission: [f32; 3],
    specular_reflection_probability: f32,
    maximum_specular_angle: f32,
    refractive_index: f32,
    is_opaque: bool,
    pub is_lightsource: bool,
}

impl Sphere {
    #[allow(dead_code)]
    pub fn new(position: [f32; 3], radius: f32, material: &Material) -> Self {
        Self {
            position,
            radius,
            color: material.color,
            emission: material.emission,
            specular_reflection_probability: material.specular_reflection_probability,
            maximum_specular_angle: material.maximum_specular_angle,
            refractive_index: material.refractive_index,
            is_opaque: material.is_opaque,
            is_lightsource: material.is_lightsource,
        }
    }
}

impl Display for Sphere {
    #[allow(dead_code)]
    fn fmt(&self, fmt: &mut Formatter) -> Result {
        let _ = fmt.write_str("{");
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.position[0], self.position[1], self.position[2]));
        let _ = fmt.write_str(&format!("{:.9}, ", self.radius));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.color[0], self.color[1], self.color[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.emission[0], self.emission[1], self.emission[2]));
        let _ = fmt.write_str(&format!("{:.9}, ", self.specular_reflection_probability));
        let _ = fmt.write_str(&format!("{:.9}, ", self.maximum_specular_angle));
        let _ = fmt.write_str(&format!("{:.9}, " , self.refractive_index));
        let _ = fmt.write_str(&format!("{}, ", self.is_opaque));
        let _ = fmt.write_str(&format!("{}", self.is_lightsource));
        let _ = fmt.write_str(&"}, ");
        Ok(())
    }
}
