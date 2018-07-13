__kernel void compute_pixel_color(__global float *colors, unsigned long outer_iteration) {
    unsigned long state = 0;
    // It is very important that we do not initialise the random number generator with the same
    // seed for all pixels and that a pixel gets a different seed for every outer iteration.
    pcg_init(((unsigned long)get_global_id(0))*(outer_iteration+1), &state);
    int first_index = get_global_id(0)*3;
    int spp = 20;
    float4 color_average = {0.0, 0.0, 0.0, 0.0};
    float y = ((float)(1000-get_global_id(0)/1000))/1000.0;
    float x = ((float)(1000-get_global_id(0)%1000))/1000.0;
    float4 pinhole = {0.5, 0.5, -1.0, 0.0};
    // Loop spp times and take the average color.
    float4 ambient_color = {0.0, 0.0, 0.0, 0.0};
    // For russian roulette.
    float bullet_probability = 0.05;
    float survival_boost_factor = 1.0/(1.0-bullet_probability);
    // Sample each pixel several times and take the average.
    for(int i = 0; i < spp; i++) {
        // Create a ray. Its direction is slightly random, in part to get anti-aliasing.
        float dr = next_float(&state)*0.0005;
        float theta = next_float(&state)*2.0*M_PI;
        float dx = cos(theta)*dr;
        float dy = sin(theta)*dr;
        float4 point_on_retina = {x+dx, y+dy, -2.0, 0.0};
        float4 direction = normalize(pinhole - point_on_retina);
        struct ray r = {pinhole, direction};
        float4 color = {1.0, 1.0, 1.0, 0.0};
        // Loop until we hit a light or the ray escapes the scene.
        while(true) {
            // Either kill the ray or boost it.
            float random_russian_roulette = next_float(&state);
            if(random_russian_roulette < bullet_probability) {
                break;
            } else {
                color *= survival_boost_factor;
            }
            // Find the closest intersected triangle.
            float min_distance_triangle = FLT_MAX;
            int closest_triangle_index = -1;
            for(int j = 0; j < number_of_triangles; j++) {
                float triangle_distance = triangle_ray_distance(j, &r);
                if(triangle_distance < min_distance_triangle) {
                    min_distance_triangle = triangle_distance;
                    closest_triangle_index = j;
                }
            }
            // Find the closest intersected sphere.
            float min_distance_sphere = FLT_MAX;
            int closest_sphere_index = -1;
            for(int j = 0; j < number_of_spheres; j++) {
                float sphere_distance = sphere_ray_distance(j, &r);
                if(sphere_distance < min_distance_sphere) {
                    min_distance_sphere = sphere_distance;
                    closest_sphere_index = j;
                }
            }
            // The ray escaped the scene.
            if(closest_triangle_index == -1 && closest_sphere_index == -1) {
                color_average += color*ambient_color;
                break;
            }
            // Get information about the hitpoint.
            float4 hitpoint_color = {0.0, 0.0, 0.0, 0.0};
            float hitpoint_lambertian_probability = 0.0;
            bool hitpoint_is_light = false;
            float4 hitpoint_normal = {0.0, 0.0, 0.0, 0.0};
            float4 hitpoint_t_1 = {0.0, 0.0, 0.0, 0.0};
            float4 hitpoint_t_2 = {0.0, 0.0, 0.0, 0.0};
            if(min_distance_triangle < min_distance_sphere) {
                r.position += min_distance_triangle*r.direction;
                hitpoint_color = triangles[closest_triangle_index].color;
                hitpoint_lambertian_probability = triangles[closest_triangle_index].lambertian_probability;
                hitpoint_is_light = triangles[closest_triangle_index].is_light;
                hitpoint_normal = triangles[closest_triangle_index].normal;
                hitpoint_t_1 = triangles[closest_triangle_index].t_1;
                hitpoint_t_2 = triangles[closest_triangle_index].t_2;
            } else {
                r.position += min_distance_sphere*r.direction;
                hitpoint_color = spheres[closest_sphere_index].color;
                hitpoint_lambertian_probability = spheres[closest_sphere_index].lambertian_probability;
                hitpoint_is_light = spheres[closest_sphere_index].is_light;
                hitpoint_normal = normalize(r.position-spheres[closest_sphere_index].position);
                compute_local_coordinate_system(hitpoint_normal, &hitpoint_t_1, &hitpoint_t_2);
            }
            // Multiply by the color of the surface or lightsource.
            color *= hitpoint_color;
            // The ray hit a lightsource.
            if(hitpoint_is_light) {
                color_average += color;
                break;
            }
            // Update the direction of the ray.
            float random_lambertian = next_float(&state);
            if(random_lambertian < hitpoint_lambertian_probability) {
                r.direction = lambertian_on_hemisphere(hitpoint_normal, hitpoint_t_1, hitpoint_t_2, &state);
            } else {
                r.direction = specular_on_hemisphere(hitpoint_normal, r.direction);
            }
        }
    }
    colors[first_index] = color_average.x/((float)spp);
    colors[first_index+1] = color_average.y/((float)spp);
    colors[first_index+2] = color_average.z/((float)spp);
}
