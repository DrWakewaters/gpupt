use material::Material;
use sphere::Sphere;
use triangle::Triangle;

pub fn create_scene() -> String {
    let red = Material::new([0.8, 0.2, 0.2], [0.0, 0.0, 0.0], 0.0, 0.0, 1.0, true, false);
    let green = Material::new([0.2, 0.8, 0.2], [0.0, 0.0, 0.0], 0.0, 0.0, 1.0, true, false);
    let blue = Material::new([0.2, 0.2, 0.8], [0.0, 0.0, 0.0], 0.0, 0.0, 1.0, true, false);
    let white = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.0, 0.0, 1.0, true, false);
    let white_glossy = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.1, 0.05, 1.0, true, false);
    let metal = Material::new([0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.999, 0.05, 1.0, true, false);
    let white_light = Material::new([0.8, 0.8, 0.8], [1.0, 1.0, 1.0], 0.0, 0.0, 1.0, true, true);

    let mut scene = String::new();
    let mut triangles: Vec<Triangle> = Vec::new();
    let mut spheres: Vec<Sphere> = Vec::new();

    let cornell = false;
    if cornell {
        // Left wall.
        triangles.push(Triangle::new([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], &red));
        triangles.push(Triangle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], &red));
        // Right wall.
        triangles.push(Triangle::new([1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], &green));
        triangles.push(Triangle::new([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], &green));
        // Floor.
        triangles.push(Triangle::new([0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], &white));
        triangles.push(Triangle::new([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], &white));
        // Ceiling.
        triangles.push(Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], &white));
        triangles.push(Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], &white));
        // Far wall.
        triangles.push(Triangle::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], &white));
        triangles.push(Triangle::new([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], &white));
        // Ceiling light.
        triangles.push(Triangle::new([0.2, 1e-3, 0.2], [0.8, 1e-3, 0.8], [0.2, 1e-3, 0.8], &white_light));
        triangles.push(Triangle::new([0.2, 1e-3, 0.2], [0.8, 1e-3, 0.2], [0.8, 1e-3, 0.8], &white_light));
    } else {
        // Left wall.
        triangles.push(Triangle::new([-0.6, 0.5, -1.5], [-0.6, 1.0, 0.5], [-0.6, 1.0, -1.5], &white));
        triangles.push(Triangle::new([-0.6, 0.5, -1.5], [-0.6, 0.5, 0.5], [-0.6, 1.0, 0.5], &white));
        // Right wall.
        triangles.push(Triangle::new([1.6, 0.5, 0.5], [1.6, 1.0, -1.5], [1.6, 1.0, 0.5], &white));
        triangles.push(Triangle::new([1.6, 0.5, 0.5], [1.6, 0.5, -1.5], [1.66, 1.0, -1.5], &white));
        // Far wall.
        triangles.push(Triangle::new([-0.6, -0.6, 1.0], [1.6, 1.0, 1.0], [-0.6, 1.0, 1.0], &white));
        triangles.push(Triangle::new([-0.6, -0.6, 1.0], [1.6, -0.6, 1.0], [1.6, 1.0, 1.0], &white));
        // Floor.
        triangles.push(Triangle::new([-0.6, 1.0, 1.0], [1.6, 1.0, -2.5], [-0.6, 1.0, -2.5], &white));
        triangles.push(Triangle::new([-0.6, 1.0, 1.0], [1.6, 1.0, 1.0], [1.6, 1.0, -2.5], &white));
        // Ceiling.
        triangles.push(Triangle::new([0.0, -1.0, 0.0], [1.0, -1.0, 1.0], [0.0, -1.0, 1.0], &white));
        triangles.push(Triangle::new([0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, -1.0, 1.0], &white));
        // Middle row.
        spheres.push(Sphere::new([0.0, 0.84, 0.6], 0.16, &white));
        spheres.push(Sphere::new([0.5, 0.84, 0.6], 0.16, &white_glossy));
        spheres.push(Sphere::new([1.0, 0.84, 0.6], 0.16, &metal));
        // Front row.
        spheres.push(Sphere::new([0.0, 0.93, 0.37], 0.07, &red));
        spheres.push(Sphere::new([0.25, 0.93, 0.37], 0.07, &white));
        spheres.push(Sphere::new([0.5, 0.93, 0.37], 0.07, &blue));
        spheres.push(Sphere::new([0.75, 0.93, 0.37], 0.07, &white));
        spheres.push(Sphere::new([1.0, 0.93, 0.37], 0.07, &green));
        // Top row.
        spheres.push(Sphere::new([-0.2, 0.2, 0.43], 0.08, &white));
        spheres.push(Sphere::new([0.5, 0.2, 0.43], 0.08, &white));
        spheres.push(Sphere::new([1.2, 0.2, 0.43], 0.08, &white));
        // Lightsource.
        spheres.push(Sphere::new([-1.0, -1.0, 0.5], 0.5, &white_light));
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
