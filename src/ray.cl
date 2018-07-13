// The distance to a triangle along a given ray.
float triangle_ray_distance(int triangle_index, struct ray *restrict r) {
    float4 h = cross(r->direction, triangles[triangle_index].e_2);
    float a = dot(triangles[triangle_index].e_1, h);
    if(a < 1.0e-6 && a > -1.0e-6) {
        return FLT_MAX;
    }
    float f = 1.0/a;
    float4 s = r->position - triangles[triangle_index].node_0;
    float u = dot(s, h)*f;
    if(u < 1.0e-6 || u > 1.0-1.0e-6) {
        return FLT_MAX;
    }
    float4 q = cross(s, triangles[triangle_index].e_1);
    float v = dot(r->direction, q)*f;
    if(v < 1.0e-6 || u+v > 1.0-1.0e-6) {
        return FLT_MAX;
    }
    float distance = dot(triangles[triangle_index].e_2, q)*f;
    if(distance < 1.0e-6) {
        return FLT_MAX;
    }
    return distance;
}

float sphere_ray_distance(int sphere_index, struct ray *restrict r) {
    float4 b = r->position - spheres[sphere_index].position;
    float a = dot(b, r->direction)*dot(b, r->direction) - dot(b, b) + spheres[sphere_index].radius*spheres[sphere_index].radius;
    if(a < 1.0e-6) {
      return FLT_MAX;
    }
    float d_1 = -1.0*dot((r->position - spheres[sphere_index].position), r->direction) + sqrt(a);
    float d_2 = d_1 - 2.0*sqrt(a);
    if(d_2 > 1.0e-6) {
        return d_2;
    } else if(d_1 > 1.0e-6) {
        return d_1;
    }
    return FLT_MAX;
}


void compute_local_coordinate_system(float4 normal, float4 *t_1, float4 *t_2) {
    float4 t_1_computed = {0.0, 0.0, 0.0, 0.0};
    float4 x = {1.0, 0.0, 0.0, 0.0};
    float4 y = {0.0, 1.0, 0.0, 0.0};
    float dot_product = dot(normal, x);
    if((dot_product > 0.01) || (dot_product < -0.01)) {
        t_1_computed = normalize(cross(normal, y));
    } else {
        t_1_computed = normalize(cross(normal, x));
    }
    float4 t_2_computed = normalize(cross(normal, t_1_computed));
    *t_1 = t_1_computed;
    *t_2 = t_2_computed;
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
