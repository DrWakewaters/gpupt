// The distance to a triangle along a given ray.
float triangle_ray_distance(int triangle_index, Ray ray) {
    float4 h = cross(ray.direction, triangles[triangle_index].e_2);
    float a = dot(triangles[triangle_index].e_1, h);
    if(a < 1.0e-6 && a > -1.0e-6) {
        return FLT_MAX;
    }
    float f = 1.0/a;
    float4 s = ray.position - triangles[triangle_index].node_0;
    float u = dot(s, h)*f;
    if(u < 1.0e-6 || u > 1.0-1.0e-6) {
        return FLT_MAX;
    }
    float4 q = cross(s, triangles[triangle_index].e_1);
    float v = dot(ray.direction, q)*f;
    if(v < 1.0e-6 || u+v > 1.0-1.0e-6) {
        return FLT_MAX;
    }
    float d = dot(triangles[triangle_index].e_2, q)*f;
    if(d < 1.0e-6) {
        return FLT_MAX;
    }
    return d;
}

float sphere_ray_distance(int sphere_index, Ray ray) {
    float4 b = ray.position - spheres[sphere_index].position;
    float a = dot(b, ray.direction)*dot(b, ray.direction) - dot(b, b) + spheres[sphere_index].radius*spheres[sphere_index].radius;
    if(a < 1.0e-4) {
      return FLT_MAX;
    }
    float d_1 = -1.0*dot((ray.position - spheres[sphere_index].position), ray.direction) + sqrt(a);
    float d_2 = d_1 - 2.0*sqrt(a);
    if(d_2 > 1.0e-4) {
        return d_2;
    } else if(d_1 > 1.0e-4) {
        return d_1;
    }
    return FLT_MAX;
}


void compute_local_coordinate_system(float4 normal, float4 *restrict t_1, float4 *restrict t_2) {
    float4 x = {1.0, 0.0, 0.0, 0.0};
    float4 y = {0.0, 1.0, 0.0, 0.0};
    float dot_product = dot(normal, x);
    if((dot_product > 0.1) || (dot_product < -0.1)) {
        *t_1 = normalize(cross(normal, y));
    } else {
        *t_1 = normalize(cross(normal, x));
    }
    *t_2 = normalize(cross(normal, *t_1));
}

// Find a random reflection of a ray that hits a surface with a known normal.
// The direction is picked from the Lambertian distribution, which is used for perfectly
// diffuse materials.
float4 lambertian_on_hemisphere(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state) {
    float r_1 = next_float(state);
    float r_2 = next_float(state);

    float sqrt_arg = 1.0-r_1;
    float sin_theta = sqrt(sqrt_arg);
    float cos_theta = sqrt(r_1);
    float phi = 2.0*M_PI*r_2;

    float4 a = sin_theta*cos(phi)*t_1;
    float4 b = sin_theta*sin(phi)*t_2;
    float4 c = cos_theta*normal;
    return normalize(a+b+c);
}

float4 specular_on_hemisphere(float4 normal, float4 incoming_direction) {
    float4 delta_direction = 2.0f*dot(incoming_direction, normal)*normalize(normal);
    return normalize(incoming_direction-delta_direction);
}

Hitpoint compute_hitpoint(Ray ray) {
    // Find the closest intersected triangle.
    float min_distance_triangle = FLT_MAX;
    int closest_triangle_index = -1;
    for(int j = 0; j < number_of_triangles; j++) {
        float triangle_distance = triangle_ray_distance(j, ray);
        if(triangle_distance < min_distance_triangle) {
            min_distance_triangle = triangle_distance;
            closest_triangle_index = j;
        }
    }
    // Find the closest intersected sphere.
    float min_distance_sphere = FLT_MAX;
    int closest_sphere_index = -1;
    for(int j = 0; j < number_of_spheres; j++) {
        float sphere_distance = sphere_ray_distance(j, ray);
        if(sphere_distance < min_distance_sphere) {
            min_distance_sphere = sphere_distance;
            closest_sphere_index = j;
        }
    }
    // The ray escaped the scene.
    if(closest_triangle_index == -1 && closest_sphere_index == -1) {
        Hitpoint hitpoint = {{0.0, 0.0, 0.0, 0.0}, ray.direction, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, FLT_MAX, 0.0, false, false, false};
        return hitpoint;
    }
    if(min_distance_triangle < min_distance_sphere) {
        Hitpoint hitpoint = {ray.position+min_distance_triangle*ray.direction, ray.direction, triangles[closest_triangle_index].normal, triangles[closest_triangle_index].t_1, triangles[closest_triangle_index].t_2, triangles[closest_triangle_index].color, min_distance_triangle, triangles[closest_triangle_index].lambertian_probability, triangles[closest_triangle_index].is_opaque, triangles[closest_triangle_index].is_lightsource, true};
        return hitpoint;
    }
    float4 position = ray.position+min_distance_sphere*ray.direction;
    float4 normal = normalize(position-spheres[closest_sphere_index].position);
    float4 t_1 = {0.0, 0.0, 0.0, 0.0};
    float4 t_2 = {0.0, 0.0, 0.0, 0.0};
    compute_local_coordinate_system(normal, &t_1, &t_2);
    Hitpoint hitpoint = {position, ray.direction, normal, t_1, t_2, spheres[closest_sphere_index].color, min_distance_sphere, spheres[closest_sphere_index].lambertian_probability, spheres[closest_sphere_index].is_opaque, spheres[closest_sphere_index].is_lightsource, true};
    return hitpoint;
}
