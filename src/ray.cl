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

// Find a random reflection of a ray that hits a surface with a known normal.
// The direction is picked from the Lambertian distribution, which is used for perfectly
// diffuse materials.
float4 lambertian_on_hemisphere(int triangle_index, unsigned long *restrict state) {
    float r_1 = next_float(state);
    float r_2 = next_float(state);

    float sqrt_arg = 1.0-r_1;
    float sin_theta = sqrt(sqrt_arg);
    float cos_theta = sqrt(r_1);
    float phi = 2.0*M_PI*r_2;

    float4 a = sin_theta*cos(phi)*triangles[triangle_index].t_1;
    float4 b = sin_theta*sin(phi)*triangles[triangle_index].t_2;
    float4 c = cos_theta*triangles[triangle_index].normal;
    return normalize(a+b+c);
}
