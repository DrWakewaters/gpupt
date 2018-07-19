use material::Material;
use sphere::Sphere;
use triangle::Triangle;

pub fn create_scene() -> String {
    let red_material = Material::new([0.8, 0.2, 0.2], [0.0, 0.0, 0.0], 1.0, 0.0, 1.0, true, false);
    let green_material = Material::new([0.2, 0.8, 0.2], [0.0, 0.0, 0.0], 1.0, 0.0, 1.0, true, false);
    let blue_material = Material::new([0.2, 0.2, 0.8], [0.0, 0.0, 0.0], 1.0, 0.0, 1.0, true, false);
    let white_material = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 1.0, 0.0, 1.0, true, false);
    let white_glossy_material = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.95, 0.05, 1.0, true, false);
    let metal_material = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.0, 0.05, 1.0, true, false);
    let white_light_material = Material::new([0.8, 0.8, 0.8], [1.0, 1.0, 1.0], 1.0, 0.0, 1.0, true, true);

    let mut scene = String::new();
    let mut triangles: Vec<Triangle> = Vec::new();
    let mut spheres: Vec<Sphere> = Vec::new();

    let cornell = false;
    if cornell {
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
        triangles.push(Triangle::new([0.2, 1e-3, 0.2], [0.8, 1e-3, 0.8], [0.2, 1e-3, 0.8], &white_light_material));
        triangles.push(Triangle::new([0.2, 1e-3, 0.2], [0.8, 1e-3, 0.2], [0.8, 1e-3, 0.8], &white_light_material));
    } else {
        // Left wall.
        triangles.push(Triangle::new([-0.6, 0.5, -1.5], [-0.6, 1.0, 0.5], [-0.6, 1.0, -1.5], &white_material));
        triangles.push(Triangle::new([-0.6, 0.5, -1.5], [-0.6, 0.5, 0.5], [-0.6, 1.0, 0.5], &white_material));
        // Right wall.
        triangles.push(Triangle::new([1.6, 0.5, 0.5], [1.6, 1.0, -1.5], [1.6, 1.0, 0.5], &white_material));
        triangles.push(Triangle::new([1.6, 0.5, 0.5], [1.6, 0.5, -1.5], [1.66, 1.0, -1.5], &white_material));
        // Far wall.
        triangles.push(Triangle::new([-0.6, -0.6, 1.0], [1.6, 1.0, 1.0], [-0.6, 1.0, 1.0], &white_material));
        triangles.push(Triangle::new([-0.6, -0.6, 1.0], [1.6, -0.6, 1.0], [1.6, 1.0, 1.0], &white_material));
        // Floor.
        triangles.push(Triangle::new([-0.6, 1.0, 1.0], [1.6, 1.0, -2.5], [-0.6, 1.0, -2.5], &white_material));
        triangles.push(Triangle::new([-0.6, 1.0, 1.0], [1.6, 1.0, 1.0], [1.6, 1.0, -2.5], &white_material));
        // Ceiling.
        triangles.push(Triangle::new([0.0, -1.0, 0.0], [1.0, -1.0, 1.0], [0.0, -1.0, 1.0], &white_material));
        triangles.push(Triangle::new([0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, -1.0, 1.0], &white_material));
        // Middle row.
        spheres.push(Sphere::new([0.0, 0.84, 0.6], 0.16, &white_material));
        spheres.push(Sphere::new([0.5, 0.84, 0.6], 0.16, &white_glossy_material));
        spheres.push(Sphere::new([1.0, 0.84, 0.6], 0.16, &metal_material));
        // Front row.
        spheres.push(Sphere::new([0.0, 0.93, 0.37], 0.07, &red_material));
        spheres.push(Sphere::new([0.25, 0.93, 0.37], 0.07, &white_material));
        spheres.push(Sphere::new([0.5, 0.93, 0.37], 0.07, &blue_material));
        spheres.push(Sphere::new([0.75, 0.93, 0.37], 0.07, &white_material));
        spheres.push(Sphere::new([1.0, 0.93, 0.37], 0.07, &green_material));
        // Top row.
        spheres.push(Sphere::new([-0.2, 0.2, 0.43], 0.08, &white_material));
        spheres.push(Sphere::new([0.5, 0.2, 0.43], 0.08, &white_material));
        spheres.push(Sphere::new([1.2, 0.2, 0.43], 0.08, &white_material));
        // Lightsource.
        spheres.push(Sphere::new([-1.0, -1.0, 0.5], 0.5, &white_light_material));
    }

    // Create an opencl array with the triangles and an array with the indices
    // of the triangles which are lightsources.
    let number_of_triangles = format!("{}", triangles.len());
    let mut triangle_lightsource_indices: Vec<usize> = Vec::new();
    scene.push_str("__constant Triangle triangles[");
    scene.push_str(&number_of_triangles);
    scene.push_str("] = {");
    for (i, triangle) in triangles.iter().enumerate() {
        if triangle.is_lightsource {
            triangle_lightsource_indices.push(i);
        }
        scene.push_str(&triangle.to_string());
    }
    scene.push_str("};");
    scene.push_str("__constant unsigned long number_of_triangles = ");
    scene.push_str(&number_of_triangles);
    scene.push_str(";");

    let number_of_triangle_lightsources = format!("{}", triangle_lightsource_indices.len());
    scene.push_str("__constant unsigned long triangle_lightsource_indices[");
    scene.push_str(&number_of_triangle_lightsources);
    scene.push_str("] = {");
    for triangle_lightsource_index in triangle_lightsource_indices {
        scene.push_str(&format!("{}, ", triangle_lightsource_index));
    }
    scene.push_str("};");
    scene.push_str("__constant unsigned long number_of_triangle_lightsources = ");
    scene.push_str(&number_of_triangle_lightsources);
    scene.push_str(";");

    // Create an opencl array with the spheres and an array with the indices
    // of the spheres which are lightsources.
    let number_of_spheres = format!("{}", spheres.len());
    let mut sphere_lightsource_indices: Vec<usize> = Vec::new();
    scene.push_str("__constant Sphere spheres[");
    scene.push_str(&number_of_spheres);
    scene.push_str("] = {");
    for (i, sphere) in spheres.iter().enumerate() {
        if sphere.is_lightsource {
            sphere_lightsource_indices.push(i);
        }
        scene.push_str(&sphere.to_string());
    }
    scene.push_str("};");
    scene.push_str("__constant unsigned long number_of_spheres = ");
    scene.push_str(&number_of_spheres);
    scene.push_str(";");

    let number_of_sphere_lightsources = format!("{}", sphere_lightsource_indices.len());
    scene.push_str("__constant unsigned long sphere_lightsource_indices[");
    scene.push_str(&number_of_sphere_lightsources);
    scene.push_str("] = {");
    for sphere_lightsource_index in sphere_lightsource_indices {
        scene.push_str(&format!("{}, ", sphere_lightsource_index));
    }
    scene.push_str("};");
    scene.push_str("__constant unsigned long number_of_sphere_lightsources = ");
    scene.push_str(&number_of_sphere_lightsources);
    scene.push_str(";");

    scene
}
