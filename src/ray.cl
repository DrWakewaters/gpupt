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
        Hitpoint hitpoint = {{0.0, 0.0, 0.0, 0.0}, ray.direction, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, FLT_MAX, 0.0, 0.0, false, false, false, true};
        return hitpoint;
    }
    if(min_distance_triangle < min_distance_sphere) {
        Hitpoint hitpoint = {ray.position+min_distance_triangle*ray.direction, ray.direction, triangles[closest_triangle_index].normal, triangles[closest_triangle_index].t_1, triangles[closest_triangle_index].t_2, triangles[closest_triangle_index].color, triangles[closest_triangle_index].emission, accumulated_color*triangles[closest_triangle_index].color, min_distance_triangle, triangles[closest_triangle_index].lambertian_probability, triangles[closest_triangle_index].refractive_index, triangles[closest_triangle_index].is_opaque, triangles[closest_triangle_index].is_lightsource, true, true};
        return hitpoint;
    }
    float4 position = ray.position+min_distance_sphere*ray.direction;
    float4 normal = normalize(position-spheres[closest_sphere_index].position);
    float4 t_1 = {0.0, 0.0, 0.0, 0.0};
    float4 t_2 = {0.0, 0.0, 0.0, 0.0};
    compute_local_coordinate_system(normal, &t_1, &t_2);
    Hitpoint hitpoint = {position, ray.direction, normal, t_1, t_2, spheres[closest_sphere_index].color, spheres[closest_sphere_index].emission, accumulated_color*spheres[closest_sphere_index].color, min_distance_sphere, spheres[closest_sphere_index].lambertian_probability, spheres[closest_sphere_index].refractive_index, spheres[closest_sphere_index].is_opaque, spheres[closest_sphere_index].is_lightsource, true, true};
    return hitpoint;
}

float4 compute_direct_light(Ray ray, Hitpoint hitpoint, unsigned long *restrict state) {
    unsigned int index = next_uint_in_range(0, number_of_triangle_lightsources+number_of_sphere_lightsources, state);
    if(index < number_of_triangle_lightsources) {
        unsigned int triangle_index = triangle_lightsource_indices[index];
        float4 light_color = triangles[triangle_index].color;
        float4 light_normal = triangles[triangle_index].normal;
        float4 light_position = random_on_triangle(triangle_index, state);
        float lambertian_probability = triangles[triangle_index].lambertian_probability;
        bool is_opaque = triangles[triangle_index].is_opaque;
        return compute_direct_light_inner(hitpoint, light_color, light_normal, light_position, lambertian_probability, is_opaque);
    } else {
        unsigned int sphere_index = sphere_lightsource_indices[index-number_of_triangle_lightsources];
        float4 light_color = spheres[sphere_index].color;
        float4 light_normal = random_on_sphere(sphere_index, state);
        float4 light_position = spheres[sphere_index].position + spheres[sphere_index].radius*light_normal;
        float lambertian_probability = spheres[sphere_index].lambertian_probability;
        bool is_opaque = spheres[sphere_index].is_opaque;
        return compute_direct_light_inner(hitpoint, light_color, light_normal, light_position, lambertian_probability, is_opaque);
    }
}

float4 compute_direct_light_inner(Hitpoint hitpoint, float4 light_color, float4 light_normal, float4 light_position, float lambertian_probability, bool is_opaque) {
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
    float brdf = compute_brdf(incoming_direction, direction_to_light, normal, refractive_index_1, refractive_index_2, lambertian_probability, is_opaque);
    // @TODO: This assumes a lambertian lightsource. Support general lightsources.
    float brdf_lightsource = 2.0f*dot(-1.0f*direction_to_light, light_normal);
    return (brdf*brdf_lightsource/(distance_to_light*distance_to_light))*(hitpoint.accumulated_color*light_color);
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

// @TODO: Compute the specular part of the brdf. Compute transmission.
float compute_brdf(float4 incoming_direction, float4 outgoing_direction, float4 normal, float refractive_index_1, float refractive_index_2, float lambertian_probability, bool is_opaque) {
	//float brdf_reflection_specular = brdf_specular(incoming_direction, outgoing_direction, normal);
    float brdf_reflection_specular = 0.0;
    float brdf_reflection_lambertian = compute_brdf_lambertian(outgoing_direction, normal);
	float brdf_reflection = (1.0-lambertian_probability)*brdf_reflection_specular + lambertian_probability*brdf_reflection_lambertian;
	if(is_opaque) {
		return brdf_reflection;
	}
    // Not implemented yet.
    return 0.0;
}

float compute_brdf_lambertian(float4 outgoing_direction, float4 normal) {
	return 2.0*dot(outgoing_direction, normal);
}
