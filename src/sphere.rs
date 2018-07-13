use std::fmt::{Display, Formatter, Result};

use material::Material;

#[allow(dead_code)]
pub struct Sphere {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
    lambertian_probability: f32,
    refractive_index: f32,
    is_opaque: bool,
    is_lightsource: bool,
}

impl Sphere {
    #[allow(dead_code)]
    pub fn new(position: [f32; 3], radius: f32, material: &Material) -> Self {
        Self {
            position,
            radius,
            color: material.color,
            lambertian_probability: material.lambertian_probability,
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
        let _ = fmt.write_str(&format!("{:.9}, ", self.lambertian_probability));
        let _ = fmt.write_str(&format!("{:.9}, " , self.refractive_index));
        let _ = fmt.write_str(&format!("{}, ", self.is_opaque));
        let _ = fmt.write_str(&format!("{}", self.is_lightsource));
        let _ = fmt.write_str(&format!("}}, "));
        Ok(())
    }
}
