__kernel void compute_pixel_color(__global float *colors, unsigned long outer_iteration) {
    unsigned long state = 0;
    // It is very important that we do not initialise the random number generator with the same
    // seed for all pixels and that a pixel gets a different seed for every outer iteration.
    pcg_init(((unsigned long)get_global_id(0))*(outer_iteration+1), &state);
    int first_index = get_global_id(0)*3;
    float4 color_average = {0.0, 0.0, 0.0, 0.0};
    float y = ((float)(1000-get_global_id(0)/1000))/1000.0;
    float x = ((float)(1000-get_global_id(0)%1000))/1000.0;
    float4 pinhole = {0.5, 0.5, -1.0, 0.0};
    // For russian roulette.
    float bullet_probability = 0.05;
    float survival_boost_factor = 1.0/(1.0-bullet_probability);
    // Sample each pixel spp times and take the average.
    int spp = 20;
    for(int i = 0; i < spp; i++) {
        // Create a ray. Its direction is slightly random, in part to get anti-aliasing.
        float dr = next_float(&state)*0.0005;
        float theta = next_float(&state)*2.0*M_PI;
        float dx = cos(theta)*dr;
        float dy = sin(theta)*dr;
        float4 point_on_retina = {x+dx, y+dy, -2.0, 0.0};
        float4 direction = normalize(pinhole - point_on_retina);
        Ray ray = {pinhole, direction};
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
            // Find the closest hitpoint.
            struct Hitpoint hitpoint = compute_hitpoint(ray);
            if(!hitpoint.hit_surface) {
                break;
            }
            // Multiply by the color of the surface or lightsource.
            color *= hitpoint.color;
            // The ray hit a lightsource.
            if(hitpoint.is_lightsource) {
                color_average += color;
                break;
            }
            // Update the position and direction of the ray.
            ray.position = hitpoint.position;
            float random_lambertian = next_float(&state);
            if(random_lambertian < hitpoint.lambertian_probability) {
                ray.direction = lambertian_on_hemisphere(hitpoint.normal, hitpoint.t_1, hitpoint.t_2, &state);
            } else {
                ray.direction = specular_on_hemisphere(hitpoint.normal, ray.direction);
            }
        }
    }
    colors[first_index] = color_average.x/((float)spp);
    colors[first_index+1] = color_average.y/((float)spp);
    colors[first_index+2] = color_average.z/((float)spp);
}
