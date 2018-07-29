__kernel void compute_pixel_color(__global float *colors, unsigned long outer_iteration) {
    unsigned long state = 0;
    // It is very important that we do not initialise the random number generator with the same
    // seed for all pixels and that a pixel gets a different seed for every outer iteration.
    pcg_init(((unsigned long)get_global_id(0))*(outer_iteration+1), &state);
    int first_index = get_global_id(0)*3;
    float4 color_average = {0.0, 0.0, 0.0, 0.0};
    float y = ((float)(1000-get_global_id(0)/1000))/1000.0;
    float x = ((float)(1000-get_global_id(0)%1000))/1000.0;
    Camera camera = {{0.0, 0.0, 1.0, 0.0}, {0.5, 0.5, 0.4, 0.0}, {0.5, 0.5, -1.0, 0.0}, 0.02};
    // For russian roulette.
    float bullet_probability = 0.02;
    float survival_boost_factor = 1.0/(1.0-bullet_probability);
    // Sample each pixel spp times and take the average.
    int spp = 50;
    for(int i = 0; i < spp; i++) {
        // Create a ray. Its direction is slightly random, in part to get anti-aliasing.
        Ray ray = create_ray(x, y, camera, &state);
        // Just so that we can extract accumulated_color and give it to compute_hitpoint(...).
        Hitpoint hitpoint = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, 0.0, 0.0, 0.0, 0.0, false, false, false, true};
        while(true) {
            // Either delete the ray or boost it. Also, delete rays with a tiny intensity.
            float random_russian_roulette = next_float(&state);
            if((random_russian_roulette < bullet_probability) || length(hitpoint.accumulated_color) < 1.0e-3) {
                break;
            } else {
                hitpoint.accumulated_color *= survival_boost_factor;
            }
            // Find the closest hitpoint and update the hitpoint and ray.
            update(&ray, &hitpoint, &state);
            // Break if the ray didn't hit anything.
            if(!hitpoint.hit_surface) {
                break;
            }
            // The ray hit a lightsource.
            if(hitpoint.is_lightsource) {
                color_average += hitpoint.accumulated_color;
                break;
            }
        }
    }
    colors[first_index] = color_average.x/((float)spp);
    colors[first_index+1] = color_average.y/((float)spp);
    colors[first_index+2] = color_average.z/((float)spp);
}

/*
while(true) {
    // Either delete the ray or boost it. Also, delete rays with a tiny intensity.
    float random_russian_roulette = next_float(&state);
    if((random_russian_roulette < bullet_probability) || length(hitpoint.accumulated_color) < 1.0e-3) {
        break;
    } else {
        hitpoint.accumulated_color *= survival_boost_factor;
    }
    // Find the closest hitpoint and update the hitpoint and ray.
    update(&ray, &hitpoint, &state);
    // Break if the ray didn't hit anything.
    if(!hitpoint.hit_surface) {
        break;
    }
    // The ray hit a lightsource.
    if(hitpoint.is_lightsource) {
        color_average += hitpoint.accumulated_color;
        break;
    }
}
*/
