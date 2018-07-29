// OTHER FUNCTIONS

// Given a normalised vector, compute an ON basis where this vector is a basis vector.
void compute_local_coordinate_system(float4 normal, float4 *restrict t_1, float4 *restrict t_2) {
    float4 x = {1.0f, 0.0f, 0.0f, 0.0f};
    float4 y = {0.0f, 1.0f, 0.0f, 0.0f};
    float dot_product = dot(normal, x);
    if((dot_product > 0.1f) || (dot_product < -0.1f)) {
        *t_1 = normalize(cross(normal, y));
    } else {
        *t_1 = normalize(cross(normal, x));
    }
    *t_2 = normalize(cross(normal, *t_1));
}

Ray create_ray(float x, float y, Camera camera, unsigned long *restrict state) {
    float dr = next_float(state)*0.0005f;
    float theta = next_float(state)*2.0f*M_PI;
    float dx = cos(theta)*dr;
    float dy = sin(theta)*dr;
    float4 point_on_retina = {x+dx, y+dy, -2.0f, 0.0f};
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
    float f = 1.0f/a;
    float4 s = ray.position - triangles[triangle_index].node_0;
    float u = dot(s, h)*f;
    if(u < 1.0e-5 || u > 1.0f-1.0e-5) {
        return FLT_MAX;
    }
    float4 q = cross(s, triangles[triangle_index].e_1);
    float v = dot(ray.direction, q)*f;
    if(v < 1.0e-5 || u+v > 1.0f-1.0e-5) {
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
    float d_1 = -1.0f*dot((ray.position - spheres[sphere_index].position), ray.direction) + sqrt(a);
    float d_2 = d_1 - 2.0f*sqrt(a);
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


// UPDATE

void update(Ray *ray, Hitpoint *hitpoint, unsigned long *restrict state) {
    // Find the closest intersected triangle.
    float min_distance_triangle = FLT_MAX;
    int closest_triangle_index = -1;
    Ray incoming_ray = (Ray){ray->position, ray->direction};
    for(unsigned long j = 0; j < number_of_triangles; j++) {
        float triangle_distance = triangle_ray_distance(j, incoming_ray);
        if(triangle_distance < min_distance_triangle) {
            min_distance_triangle = triangle_distance;
            closest_triangle_index = j;
        }
    }
    // Find the closest intersected sphere.
    float min_distance_sphere = FLT_MAX;
    int closest_sphere_index = -1;
    for(unsigned long j = 0; j < number_of_spheres; j++) {
        float sphere_distance = sphere_ray_distance(j, incoming_ray);
        if(sphere_distance < min_distance_sphere) {
            min_distance_sphere = sphere_distance;
            closest_sphere_index = j;
        }
    }
    if(closest_triangle_index == -1 && closest_sphere_index == -1) {
        hitpoint->hit_surface = false;
    } else if(min_distance_triangle < min_distance_sphere) {
        ray->position = ray->position+min_distance_triangle*ray->direction;
        *hitpoint = (Hitpoint){ray->position, ray->direction, triangles[closest_triangle_index].normal, triangles[closest_triangle_index].t_1, triangles[closest_triangle_index].t_2, triangles[closest_triangle_index].color, triangles[closest_triangle_index].emission, hitpoint->accumulated_color*triangles[closest_triangle_index].color, min_distance_triangle, triangles[closest_triangle_index].specular_reflection_probability, triangles[closest_triangle_index].maximum_specular_angle, triangles[closest_triangle_index].refractive_index, triangles[closest_triangle_index].is_opaque, triangles[closest_triangle_index].is_lightsource, true, true};
        if(!triangles[closest_triangle_index].is_lightsource) {
            bool may_sample_cone = false;
            ray->direction = sample(ray->position, hitpoint->incoming_direction, hitpoint->normal, hitpoint->t_1, hitpoint->t_2, hitpoint->maximum_specular_angle, hitpoint->specular_reflection_probability, &may_sample_cone, state);
            hitpoint->accumulated_color *= color_modifier(ray->position, hitpoint->incoming_direction, ray->direction, hitpoint->normal, hitpoint->specular_reflection_probability, hitpoint->maximum_specular_angle, may_sample_cone);
        }
    } else {
        ray->position = ray->position+min_distance_sphere*ray->direction;
        float4 normal = normalize(ray->position-spheres[closest_sphere_index].position);
        float4 t_1 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 t_2 = {0.0f, 0.0f, 0.0f, 0.0f};
        compute_local_coordinate_system(normal, &t_1, &t_2);
        *hitpoint = (Hitpoint){ray->position, ray->direction, normal, t_1, t_2, spheres[closest_sphere_index].color, spheres[closest_sphere_index].emission, hitpoint->accumulated_color*spheres[closest_sphere_index].color, min_distance_sphere, spheres[closest_sphere_index].specular_reflection_probability, spheres[closest_sphere_index].maximum_specular_angle, spheres[closest_sphere_index].refractive_index, spheres[closest_sphere_index].is_opaque, spheres[closest_sphere_index].is_lightsource, true, true};
        if(!spheres[closest_sphere_index].is_lightsource) {
            bool may_sample_cone = false;
            ray->direction = sample(ray->position, hitpoint->incoming_direction, hitpoint->normal, hitpoint->t_1, hitpoint->t_2, hitpoint->maximum_specular_angle, hitpoint->specular_reflection_probability, &may_sample_cone, state);
            hitpoint->accumulated_color *= color_modifier(ray->position, hitpoint->incoming_direction, ray->direction, hitpoint->normal, hitpoint->specular_reflection_probability, hitpoint->maximum_specular_angle, may_sample_cone);
        }
    }
}

// SAMPLING

float4 sample_uniform_on_triangle(unsigned int triangle_index, unsigned long *restrict state) {
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
float4 sample_uniform_on_sphere(unsigned long *restrict state) {
	float r_1 = next_float(state);
    float r_2 = next_float(state);

    float phi = 2.0f*M_PI*r_1;
	float u = 2.0f*(r_2-0.5f);
    float p = sqrt(1.0f-u*u);

    float4 random = {p*cos(phi), p*sin(phi), u, 0.0f};
	return normalize(random);
}

float4 sample_uniform_on_hemisphere(float4 normal, unsigned long *restrict state) {
    float4 uniform_on_sphere = sample_uniform_on_sphere(state);
    if(dot(uniform_on_sphere, normal) >= 0.0f) {
        return uniform_on_sphere;
    }
    return -1.0f*uniform_on_sphere;
}

float4 sample_cos_weighted_on_hemisphere(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state) {
    float r_1 = next_float(state);
    float r_2 = next_float(state);

    float phi = 2.0f*M_PI*r_2;
    float sin_theta = sqrt(1.0f-r_1);
    float cos_theta = sqrt(r_1);

    float4 a = sin_theta*cos(phi)*t_1;
    float4 b = sin_theta*sin(phi)*t_2;
    float4 c = cos_theta*normal;
    return normalize(a+b+c);
}

float4 sample_cone(float4 direction, float maximum_angle, unsigned long *restrict state) {
    while(true) {
        float4 uniform_on_hemisphere = sample_uniform_on_hemisphere(direction, state);
        float in_dot_out = dot(direction, uniform_on_hemisphere);
        if(in_dot_out > 0.0f && acos(in_dot_out) < maximum_angle) {
            return uniform_on_hemisphere;
        }
    }
}

// See http://mathworld.wolfram.com/SpherePointPicking.html.
float4 sample_cone_fast(float4 direction, float4 t_1, float4 t_2, float maximum_angle, unsigned long *restrict state) {
    float r_1 = next_float(state);
    float r_2 = next_float(state);

    float phi = 2.0f*M_PI*r_1;
    float z_min = cos(maximum_angle);
    float u = z_min + (1.0f-z_min)*r_2;
    float theta = acos(u);

    float4 a = sin(theta)*cos(phi)*t_1;
    float4 b = sin(theta)*sin(phi)*t_2;
    float4 c = cos(theta)*direction;
    return normalize(a+b+c);
}


// SAMPLE BRDF

// @TODO: Implement transmission. Implement specular reflection.
float4 sample(float4 position, float4 incoming_direction, float4 normal, float4 t_1, float4 t_2, float maximum_specular_angle, float specular_reflection_probability, bool *may_sample_cone, unsigned long *restrict state) {
    float4 direction_to_light_center = spheres[sphere_lightsource_indices[0]].position - position;
    float4 direction_to_light_center_normalized = normalize(direction_to_light_center);
    float light_radius = spheres[sphere_lightsource_indices[0]].radius;
    float distance_to_light_center = length(direction_to_light_center);
    // See https://physics.stackexchange.com/questions/331883/solid-angle-subtended-by-a-circle.
    float angle = atan(light_radius/distance_to_light_center);
    *may_sample_cone = false;
    float direction_normal = dot(direction_to_light_center_normalized, normal);
    if(direction_normal > 0.0f) {
        float angle_from_surface = M_PI/2.0f - acos(direction_normal);
        if(angle_from_surface > angle) {
            *may_sample_cone = true;
        }
    }
    float random_sample_cone = next_float(state);
    float random_sample_specular = next_float(state);
    if(random_sample_cone < light_sampling_probability && *may_sample_cone) {
        float4 t_1_cone_light = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 t_2_cone_light = {0.0f, 0.0f, 0.0f, 0.0f};
        compute_local_coordinate_system(direction_to_light_center_normalized, &t_1_cone_light, &t_2_cone_light);
        return sample_cone_fast(direction_to_light_center_normalized, t_1_cone_light, t_2_cone_light, angle, state);
    } else if(random_sample_specular < specular_reflection_probability) {
        float4 t_1_cone_specular = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 t_2_cone_specular = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 specular_direction = normalize(incoming_direction-2.0f*dot(incoming_direction, normal)*normal);
        float angle_from_surface = M_PI/2.0f - acos(dot(normal, specular_direction));
        float maximum_angle = fmin(angle_from_surface, maximum_specular_angle);
        compute_local_coordinate_system(specular_direction, &t_1_cone_specular, &t_2_cone_specular);
        return sample_cone_fast(specular_direction, t_1_cone_specular, t_2_cone_specular, maximum_angle, state);
    } else {
        return sample_cos_weighted_on_hemisphere(normal, t_1, t_2, state);
    }
}


// COLOR MODIFICATION DUE TO THE SAMPLING

// @TODO: Implement transmission. Implement specular reflection.
float color_modifier(float4 position, float4 incoming_direction, float4 outgoing_direction, float4 normal, float specular_reflection_probability, float maximum_specular_angle, bool may_sample_cone) {
    float4 direction_to_light_center = spheres[sphere_lightsource_indices[0]].position - position;
    float light_radius = spheres[sphere_lightsource_indices[0]].radius;
    float distance_to_light_center = length(direction_to_light_center);
    float maximum_angle_light = atan(light_radius/distance_to_light_center);
    float out_normal = dot(outgoing_direction, normal);
    if(out_normal > 0.0f) {
        if(may_sample_cone) {
            float brdf_lambertian = cos_weighted_probability_density(outgoing_direction, normal);
            float brdf_cone_light = cone_probability_density(normalize(direction_to_light_center), outgoing_direction, maximum_angle_light);
            float4 specular_direction = normalize(incoming_direction-2.0f*dot(incoming_direction, normal)*normal);
            float angle_from_surface = M_PI/2.0f - acos(dot(normal, specular_direction));
            float maximum_angle_specular = fmin(angle_from_surface, maximum_specular_angle);
            float brdf_cone_specular = cone_probability_density(specular_direction, outgoing_direction, maximum_angle_specular);
            float numerator = (1.0f-specular_reflection_probability)*brdf_lambertian + specular_reflection_probability*brdf_cone_specular;
            float denominator = (1.0f-light_sampling_probability)*(1.0f-specular_reflection_probability)*brdf_lambertian + (1.0f-light_sampling_probability)*specular_reflection_probability*brdf_cone_specular + light_sampling_probability*brdf_cone_light;
            float modifier = numerator/denominator;
            return modifier;
        } else {
            return 1.0;
        }
    } else {
        return 0.0f;
    }
}

float cos_weighted_probability_density(float4 outgoing_direction, float4 normal) {
	return 2.0f*dot(outgoing_direction, normal);
}

float cone_probability_density(float4 direction_to_light_center, float4 outgoing_direction, float maximum_angle) {
    float cos_angle = dot(direction_to_light_center, outgoing_direction);
    float angle = acos(cos_angle);
    if(angle > maximum_angle) {
        return 0.0f;
    }
    float cos_maximum_angle = cos(maximum_angle);
    float maximum_sold_angle = 2.0f*M_PI*(1.0f-cos_maximum_angle);
    // To avoid inf.
    if(maximum_sold_angle < 1.0e-6) {
        return FLT_MAX;
    }
    return 2.0f*M_PI/maximum_sold_angle;
}
