// OTHER FUNCTIONS

// Given a normalised vector, compute an ON basis where this vector is a basis vector.
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

Ray create_ray(float x, float y, Camera camera, unsigned long *restrict state) {
    float dr = next_float(state)*0.0005;
    float theta = next_float(state)*2.0*M_PI;
    float dx = cos(theta)*dr;
    float dy = sin(theta)*dr;
    float4 point_on_retina = {x+dx, y+dy, -2.0, 0.0};
    float4 direction = normalize(camera.pinhole - point_on_retina);
    float distance_to_focal_plane = dot((camera.point_on_focal_plane-camera.pinhole), camera.retina_normal)/dot(direction, camera.retina_normal);
    float4 point_on_focal_plane = camera.pinhole + (distance_to_focal_plane*direction);
    float4 pinhole_translation = {camera.pinhole_radius*next_float(state), camera.pinhole_radius*next_float(state), 0.0, 0.0};
    float4 point_on_lens = camera.pinhole+pinhole_translation;
    direction = normalize(point_on_focal_plane-point_on_lens);
    return (Ray){point_on_lens, direction};
}


// RAY-SURFACE INTERSECTION DETECTION

// The distance to a triangle along a given ray.
float triangle_ray_distance(int triangle_index, Ray ray) {
    float4 h = cross(ray.direction, triangles[triangle_index].e_2);
    float a = dot(triangles[triangle_index].e_1, h);
    if(a < 1.0e-5 && a > -1.0e-5) {
        return FLT_MAX;
    }
    float f = 1.0/a;
    float4 s = ray.position - triangles[triangle_index].node_0;
    float u = dot(s, h)*f;
    if(u < 1.0e-5 || u > 1.0-1.0e-5) {
        return FLT_MAX;
    }
    float4 q = cross(s, triangles[triangle_index].e_1);
    float v = dot(ray.direction, q)*f;
    if(v < 1.0e-5 || u+v > 1.0-1.0e-5) {
        return FLT_MAX;
    }
    float d = dot(triangles[triangle_index].e_2, q)*f;
    if(d < 1.0e-5) {
        return FLT_MAX;
    }
    return d;
}

// The distance to a SPHERE along a given ray.
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


// POINT IN TRIANGLE DETECTION

// See http://blackpawn.com/texts/pointinpoly/.
bool point_in_triangle(float4 point, float4 node_0, float4 node_1, float4 node_2) {
	if(!same_side(point, node_0, node_1, node_2)) {
		return false;
	}
	if(!same_side(point, node_1, node_0, node_2)) {
		return false;
	}
	if(!same_side(point, node_2, node_0, node_1)) {
		return false;
	}
	return true;
}

bool same_side(float4 test_point, float4 point_inside, float4 first_node, float4 second_node) {
	return dot(cross(second_node-first_node, test_point-first_node), cross(second_node-first_node, point_inside-first_node)) >= 0.0;
}


// COMPUTE DIRECT LIGHT

float4 compute_direct_light(Ray ray, Hitpoint hitpoint, unsigned long *restrict state) {
    float lambertian_probability = hitpoint.lambertian_probability;
    float maximum_specular_angle = hitpoint.maximum_specular_angle;
    bool is_opaque = hitpoint.is_opaque;
    float4 brdf = {0.0, 0.0, 0.0, 0.0};
    int samples = 0;
    if(maximum_specular_angle > 0.001) {
        samples += (int)(1.0/maximum_specular_angle);
    } else {
        samples += 1;
    }
    for(int i = 0; i < samples; i++) {
        unsigned int index = next_uint_in_range(0, number_of_triangle_lightsources+number_of_sphere_lightsources, state);
        if(index < number_of_triangle_lightsources) {
            unsigned int triangle_index = triangle_lightsource_indices[index];
            float4 light_color = triangles[triangle_index].color;
            float4 light_normal = triangles[triangle_index].normal;
            float4 light_position = random_on_triangle(triangle_index, state);
            brdf += compute_direct_light_inner(hitpoint, light_color, light_normal, light_position, lambertian_probability, maximum_specular_angle, is_opaque);
        } else {
            unsigned int sphere_index = sphere_lightsource_indices[index-number_of_triangle_lightsources];
            float4 light_color = spheres[sphere_index].color;
            float4 light_normal = random_on_sphere(sphere_index, state);
            float4 light_position = spheres[sphere_index].position + spheres[sphere_index].radius*light_normal;
            brdf += compute_direct_light_inner(hitpoint, light_color, light_normal, light_position, lambertian_probability, maximum_specular_angle, is_opaque);
        }
    }
    brdf /= (float)samples;
    return brdf;
}



float4 compute_direct_light_inner(Hitpoint hitpoint, float4 light_color, float4 light_normal, float4 light_position, float lambertian_probability, float maximum_specular_angle, bool is_opaque) {
    float4 zero = {0.0, 0.0, 0.0, 0.0};
    float distance_to_light = length(light_position-hitpoint.position);
    float4 direction_to_light = normalize(light_position-hitpoint.position);
    if(dot(direction_to_light, hitpoint.normal) < 0.0f) {
        return zero;
    }
    if(dot(-1.0f*direction_to_light, light_normal) < 0.0f) {
        return zero;
    }
    Ray ray_to_light = {hitpoint.position, direction_to_light};
    Hitpoint closest_hitpoint = compute_hitpoint(ray_to_light, zero);
    // Should never happen.
    if(!closest_hitpoint.hit_surface) {
        return zero;
    }
    if(distance_to_light - closest_hitpoint.distance_from_previous > 1e-3) {
        return zero;
    }
    float4 incoming_direction = -1.0f*hitpoint.incoming_direction;
    float4 normal = hitpoint.normal;
    if(dot(incoming_direction, hitpoint.normal) < 0.0f) {
        normal *= -1.0f;
    }
    float refractive_index_1 = 1.0f;
    float refractive_index_2 = hitpoint.refractive_index;
    if(!hitpoint.hit_from_outside) {
        refractive_index_1 = hitpoint.refractive_index;
        refractive_index_2 = 1.0f;
    }
    float brdf = compute_brdf(incoming_direction, direction_to_light, normal, refractive_index_1, refractive_index_2, lambertian_probability, maximum_specular_angle, is_opaque);

    // @TODO: This assumes a lambertian lightsource. Support general lightsources.
    float brdf_lightsource = 2.0f*dot(-1.0f*direction_to_light, light_normal);
    return (brdf*brdf_lightsource/(distance_to_light*distance_to_light))*(hitpoint.accumulated_color*light_color);
}

Hitpoint compute_hitpoint(Ray ray, float4 accumulated_color) {
    // Find the closest intersected triangle.
    float min_distance_triangle = FLT_MAX;
    int closest_triangle_index = -1;
    for(unsigned long j = 0; j < number_of_triangles; j++) {
        float triangle_distance = triangle_ray_distance(j, ray);
        if(triangle_distance < min_distance_triangle) {
            min_distance_triangle = triangle_distance;
            closest_triangle_index = j;
        }
    }
    // Find the closest intersected sphere.
    float min_distance_sphere = FLT_MAX;
    int closest_sphere_index = -1;
    for(unsigned long j = 0; j < number_of_spheres; j++) {
        float sphere_distance = sphere_ray_distance(j, ray);
        if(sphere_distance < min_distance_sphere) {
            min_distance_sphere = sphere_distance;
            closest_sphere_index = j;
        }
    }
    // The ray escaped the scene.
    if(closest_triangle_index == -1 && closest_sphere_index == -1) {
        Hitpoint hitpoint = {{0.0, 0.0, 0.0, 0.0}, ray.direction, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, FLT_MAX, 0.0, 0.0, 0.0, false, false, false, true};
        return hitpoint;
    }
    if(min_distance_triangle < min_distance_sphere) {
        Hitpoint hitpoint = {ray.position+min_distance_triangle*ray.direction, ray.direction, triangles[closest_triangle_index].normal, triangles[closest_triangle_index].t_1, triangles[closest_triangle_index].t_2, triangles[closest_triangle_index].color, triangles[closest_triangle_index].emission, accumulated_color*triangles[closest_triangle_index].color, min_distance_triangle, triangles[closest_triangle_index].lambertian_probability, triangles[closest_triangle_index].maximum_specular_angle, triangles[closest_triangle_index].refractive_index, triangles[closest_triangle_index].is_opaque, triangles[closest_triangle_index].is_lightsource, true, true};
        return hitpoint;
    }
    float4 position = ray.position+min_distance_sphere*ray.direction;
    float4 normal = normalize(position-spheres[closest_sphere_index].position);
    float4 t_1 = {0.0, 0.0, 0.0, 0.0};
    float4 t_2 = {0.0, 0.0, 0.0, 0.0};
    compute_local_coordinate_system(normal, &t_1, &t_2);
    Hitpoint hitpoint = {position, ray.direction, normal, t_1, t_2, spheres[closest_sphere_index].color, spheres[closest_sphere_index].emission, accumulated_color*spheres[closest_sphere_index].color, min_distance_sphere, spheres[closest_sphere_index].lambertian_probability, spheres[closest_sphere_index].maximum_specular_angle, spheres[closest_sphere_index].refractive_index, spheres[closest_sphere_index].is_opaque, spheres[closest_sphere_index].is_lightsource, true, true};
    return hitpoint;
}

// RANDOM POINTS ON SURFACES

float4 random_on_triangle(unsigned int triangle_index, unsigned long *restrict state) {
    // @TODO: If the point is not in the triangle, just find the corresponding poin that is.
    while(true) {
        float r_1 = next_float(state);
        float r_2 = next_float(state);
        float4 point_in_parallelogram = triangles[triangle_index].node_0 + r_1*triangles[triangle_index].e_1 +  r_2*triangles[triangle_index].e_2;
        if(point_in_triangle(point_in_parallelogram, triangles[triangle_index].node_0, triangles[triangle_index].node_1,    triangles[triangle_index].node_2)) {
            return point_in_parallelogram;
        }
    }
}

// See http://mathworld.wolfram.com/SpherePointPicking.html.
float4 random_on_sphere(unsigned int sphere_index, unsigned long *restrict state) {
	float r_1 = next_float(state);
    float r_2 = next_float(state);
    float theta = 2.0f*M_PI*r_1;
	float u = 2.0f*(r_2-0.5f);
    float p = sqrt(1.0f-u*u);
    float4 random = {p*cos(theta), p*sin(theta), u, 0.0f};
	return normalize(random);
}


// SAMPLE BRDF

// @TODO: Implement transmission.
float4 sample_brdf(Hitpoint hitpoint, Ray ray, unsigned long *restrict state) {
    float random_lambertian = next_float(state);
    float4 outgoing_direction = random_lambertian < hitpoint.lambertian_probability?
        sample_brdf_lambertian(hitpoint.normal, hitpoint.t_1, hitpoint.t_2, state) :
        sample_brdf_specular(hitpoint.normal, ray.direction, hitpoint.maximum_specular_angle, state);
    if(hitpoint.is_opaque) {
        return outgoing_direction;
    }
    // @TODO: Implement transmission.
    return (float4){0.0, 0.0, 0.0, 0.0};
}

// Find a random reflection of a ray that hits a surface with a known normal.
// The direction is picked from the Lambertian distribution, which is used for perfectly
// diffuse materials.
float4 sample_brdf_lambertian(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state) {
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

float4 sample_brdf_specular(float4 normal, float4 incoming_direction, float maximum_specular_angle, unsigned long *restrict state) {
    float4 specular_direction = normalize(incoming_direction-2.0f*dot(incoming_direction, normal)*normalize(normal));
    if(maximum_specular_angle < 1.0e-6) {
        return normalize(specular_direction);
    }
    // Find phi: the angle we will rotate wo away from specular_direction, and theta: the angle we will then rotate wo around specular_direction.
    float r_x = next_float(state);
    float r_y = next_float(state);
    if(r_y > r_x) {
        r_x = r_y;
    }
    float angle = M_PI/2.0 - acos(dot(normal, specular_direction));
    float maximum_angle = fmin(angle, maximum_specular_angle);
    float phi = maximum_angle*r_x;
    // Perform the rotations of wo around specular_direction.
    // First rotate away from specular_direction by rotating around the x-axis. See https://en.wikipedia.org/wiki/Rotation_matrix. (Any axis would do though, as long as it is not parallel with specular_direction.)
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);
    float4 reflection_direction = {specular_direction.x, cos_phi*specular_direction.y-sin_phi*specular_direction.z, sin_phi*specular_direction.y+cos_phi*specular_direction.z, 0.0};
    // Then rotate wo around specular_direction. See https://math.stackexchange.com/questions/511370/how-to-rotate-one-vector-about-another.
    float theta = 2.0*M_PI*next_float(state);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    float4 reflection_parallell_specular = cos_phi*specular_direction;

    float4 reflection_perpendicular_specular = reflection_direction-reflection_parallell_specular;
    float4 w = cross(specular_direction, reflection_perpendicular_specular);
    float x_1 = cos_theta/length(reflection_perpendicular_specular);
    float x_2 = sin_theta/length(w);
    float4 reflection_perpendicular_specular_rotation = length(reflection_perpendicular_specular)*((x_1*reflection_perpendicular_specular)+(x_2*w));
    // Might not be perfectly normalised due to rounding errors. Thus, normalise!
    return normalize(reflection_perpendicular_specular_rotation+reflection_parallell_specular);
}


// COMPUTE BRDF

// @TODO: Implement transmission.
float compute_brdf(float4 incoming_direction, float4 outgoing_direction, float4 normal, float refractive_index_1, float refractive_index_2, float lambertian_probability, float maximum_specular_angle, bool is_opaque) {
    float brdf_reflection_lambertian = 0.0;
    float brdf_reflection_specular = 0.0;
    if(lambertian_probability > 0.001) {
        brdf_reflection_lambertian = compute_brdf_lambertian(outgoing_direction, normal);
    }
    if(lambertian_probability < 0.999) {
        brdf_reflection_specular = compute_brdf_specular(incoming_direction, outgoing_direction, normal, maximum_specular_angle);
    }
	float brdf_reflection = lambertian_probability*brdf_reflection_lambertian + (1.0-lambertian_probability)*brdf_reflection_specular;
	if(is_opaque) {
		return brdf_reflection;
	}
    // @TODO: Implement transmission.
    return 0.0;
}

float compute_brdf_lambertian(float4 outgoing_direction, float4 normal) {
	return 2.0*dot(outgoing_direction, normal);
}

float compute_brdf_specular(float4 incoming_direction, float4 outgoing_direction, float4 normal, float maximum_specular_angle) {
    float4 specular_direction = normalize(2.0f*dot(incoming_direction, normal)*normalize(normal)-incoming_direction);
    float phi = acos(dot(outgoing_direction, specular_direction));
    float angle = M_PI/2.0 - dot(normal, specular_direction);
    float maximum_angle = fmin(angle, maximum_specular_angle);
    if(phi < maximum_angle) {
        float a = 1.0/(2.0*M_PI*(maximum_angle-sin(maximum_angle)));
        return 2.0*M_PI*a*(maximum_angle-phi);
    }
    return 0.0;
}
