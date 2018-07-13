use std::fmt::{Display, Formatter, Result};

use material::Material;
use math::{cross, dot, normalised, sub};

#[allow(dead_code)]
pub struct Triangle {
    node_0: [f32; 3],
    node_1: [f32; 3],
    node_2: [f32; 3],
    e_1: [f32; 3],
    e_2: [f32; 3],
    normal: [f32; 3],
    t_1: [f32; 3],
    t_2: [f32; 3],
    color: [f32; 3],
    lambertian_probability: f32,
    refractive_index: f32,
    is_opaque: bool,
    is_lightsource: bool,
}

impl Triangle {
    #[allow(dead_code)]
    pub fn new(node_0: [f32; 3], node_1: [f32; 3], node_2: [f32; 3], material: &Material) -> Self {
        let e_1 = sub(node_1, node_0);
        let e_2 = sub(node_2, node_0);
        let normal = normalised(cross(sub(node_2, node_0), sub(node_1, node_0)));
        let t_1: [f32; 3];
        let t_2: [f32; 3];
        if dot(normal, [1.0, 0.0, 0.0]).abs() > 0.1  {
            t_1 = cross(normal, [0.0, 1.0, 0.0]);
            t_2 = cross(normal, t_1);
        } else {
            t_1 = cross(normal, [1.0, 0.0, 0.0]);
            t_2 = cross(normal, t_1);
        }
        Self {
            node_0,
            node_1,
            node_2,
            e_1,
            e_2,
            normal,
            t_1,
            t_2,
            color: material.color,
            lambertian_probability: material.lambertian_probability,
            refractive_index: material.refractive_index,
            is_opaque: material.is_opaque,
            is_lightsource: material.is_lightsource,
        }
    }
}

impl Display for Triangle {
    #[allow(dead_code)]
    fn fmt(&self, fmt: &mut Formatter) -> Result {
        let _ = fmt.write_str("{");
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.node_0[0], self.node_0[1], self.node_0[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.node_1[0], self.node_1[1], self.node_1[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.node_2[0], self.node_2[1], self.node_2[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.e_1[0], self.e_1[1], self.e_1[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.e_2[0], self.e_2[1], self.e_2[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.normal[0], self.normal[1], self.normal[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.t_1[0], self.t_1[1], self.t_1[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.t_2[0], self.t_2[1], self.t_2[2]));
        let _ = fmt.write_str(&format!("{{{:.9}, {:.9}, {:.9}, 0.0}}, ", self.color[0], self.color[1], self.color[2]));
        let _ = fmt.write_str(&format!("{:.9}, ", self.lambertian_probability));
        let _ = fmt.write_str(&format!("{:.9}, " , self.refractive_index));
        let _ = fmt.write_str(&format!("{}, ", self.is_opaque));
        let _ = fmt.write_str(&format!("{}", self.is_lightsource));
        let _ = fmt.write_str(&format!("}}, "));
        Ok(())
    }
}
