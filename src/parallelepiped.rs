use material::Material;
use math::add;
use triangle::Triangle;

#[allow(dead_code)]
pub struct Parallelepiped {
    origin: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    material: Material,
}

impl Parallelepiped {
    #[allow(dead_code)]
    pub fn new(origin: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3], material: Material) -> Self {
        Self {
            origin,
            a,
            b,
            c,
            material: material,
        }
    }

    #[allow(dead_code)]
    pub fn to_triangles(&self) -> Vec<Triangle> {
        let mut triangles: Vec<Triangle> = Vec::new();
        let node_0 = self.origin;
        let node_1 = add(self.origin, self.a);
        let node_2 = add(self.origin, self.b);
        let node_3 = add(self.origin, self.c);
        let node_4 = add(node_1, self.b);
        let node_5 = add(node_2, self.c);
        let node_6 = add(node_3, self.a);
        let node_7 = add(node_4, self.c);

        triangles.push(Triangle::new(node_0, node_1, node_2, &self.material));
        triangles.push(Triangle::new(node_1, node_4, node_2, &self.material));

        triangles.push(Triangle::new(node_0, node_2, node_3, &self.material));
        triangles.push(Triangle::new(node_2, node_5, node_3, &self.material));

        triangles.push(Triangle::new(node_0, node_3, node_1, &self.material));
        triangles.push(Triangle::new(node_3, node_6, node_1, &self.material));

        triangles.push(Triangle::new(node_7, node_5, node_4, &self.material));
        triangles.push(Triangle::new(node_5, node_2, node_4, &self.material));

        triangles.push(Triangle::new(node_7, node_4, node_6, &self.material));
        triangles.push(Triangle::new(node_4, node_1, node_6, &self.material));

        triangles.push(Triangle::new(node_7, node_6, node_5, &self.material));
        triangles.push(Triangle::new(node_6, node_3, node_5, &self.material));

        triangles
    }
}
