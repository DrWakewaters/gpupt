use material::Material;
use parallelepiped::Parallelepiped;
use sphere::Sphere;
use triangle::Triangle;

pub fn create_scene() -> String {
    let red_material = Material::new([0.8, 0.2, 0.2], 1.0, 1.0, true, false);
    let green_material = Material::new([0.2, 0.8, 0.2], 1.0, 1.0, true, false);
    let white_material = Material::new([0.8, 0.8, 0.8], 1.0, 1.0, true, false);
    let metal_material = Material::new([0.8, 0.8, 0.8], 0.0, 1.0, true, false);
    let glossy_white_material = Material::new([0.8, 0.8, 0.8], 0.95, 1.0, true, false);
    let white_light_material = Material::new([0.8, 0.8, 0.8], 1.0, 1.0, true, true);

    let mut scene = String::new();

    // Triangles.
    let mut triangles: Vec<Triangle> = Vec::new();
    // Left wall.
    triangles.push(Triangle::new([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], &red_material));
    triangles.push(Triangle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], &red_material));
    // Right wall.
    triangles.push(Triangle::new([1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], &green_material));
    triangles.push(Triangle::new([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], &green_material));
    // Floor.
    triangles.push(Triangle::new([0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], &white_material));
    triangles.push(Triangle::new([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], &white_material));
    // Ceiling.
    triangles.push(Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], &white_material));
    triangles.push(Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], &white_material));
    // Far wall.
    triangles.push(Triangle::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], &white_material));
    triangles.push(Triangle::new([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], &white_material));
    // Ceiling light.
    triangles.push(Triangle::new([0.2, 1.0e-3, 0.2], [0.8, 1.0e-3, 0.8], [0.2, 1.0e-3, 0.8], &white_light_material));
    triangles.push(Triangle::new([0.2, 1.0e-3, 0.2], [0.8, 1.0e-3, 0.2], [0.8, 1.0e-3, 0.8], &white_light_material));
    // A cube
    let cube = Parallelepiped::new([0.45, 0.9, 0.1], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], white_material.clone());
    triangles.append(&mut cube.to_triangles());

    // Spheres.
    let mut spheres: Vec<Sphere> = Vec::new();
    spheres.push(Sphere::new([0.25, 0.8, 0.4], 0.2, &glossy_white_material));
    spheres.push(Sphere::new([0.75, 0.8, 0.4], 0.2, &metal_material));

    // Create an open-cl array with the triangles.
    let number_of_triangles = format!("{}", triangles.len());
    scene.push_str("__constant struct triangle triangles[");
    scene.push_str(&number_of_triangles);
    scene.push_str("] = {");
    for triangle in triangles {
        scene.push_str(&triangle.to_string());
    }
    scene.push_str("};");
    scene.push_str("__constant int number_of_triangles = ");
    scene.push_str(&number_of_triangles);
    scene.push_str(";");

    // Create an open-cl array with the spheres.
    let number_of_spheres = format!("{}", spheres.len());
    scene.push_str("__constant struct sphere spheres[");
    scene.push_str(&number_of_spheres);
    scene.push_str("] = {");
    for sphere in spheres {
        scene.push_str(&sphere.to_string());
    }
    scene.push_str("};");
    scene.push_str("__constant int number_of_spheres = ");
    scene.push_str(&number_of_spheres);
    scene.push_str(";");

    scene
}
