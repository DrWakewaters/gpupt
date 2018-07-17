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
    // Sample each pixel spp times and take the average.
    int spp = 20;
    bool direct_light_sampling = true;
    for(int i = 0; i < spp; i++) {
        // Create a ray. Its direction is slightly random, in part to get anti-aliasing.
        float dr = next_float(&state)*0.0005;
        float theta = next_float(&state)*2.0*M_PI;
        float dx = cos(theta)*dr;
        float dy = sin(theta)*dr;
        float4 point_on_retina = {x+dx, y+dy, -2.0, 0.0};
        float4 direction = normalize(pinhole - point_on_retina);
        Ray ray = {pinhole, direction};
        // Just so that we can extract accumulated_color and give it to compute_hitpoint(...).
        Hitpoint hitpoint = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, 0.0, 0.0, 0.0, 0.0, false, false, false, true};
        while(true) {
            // Find the closest hitpoint.
            hitpoint = compute_hitpoint(ray, hitpoint.accumulated_color);
            // Break if the ray didn't hit anything.
            if(!hitpoint.hit_surface) {
                break;
            }
            // Update the position of the ray.
            ray.position = hitpoint.position;
            // Do we sample a random point on a random light at each intersection
            // or does the ray just continue travelling until it randomly
            // hits a lightsource?
            // The latter will not work if the lightsource is small or far away.
            // The former does currently not work with light that goes directly
            // from light to eye - at least one bounce is needed.
            if(direct_light_sampling) {
                color_average += compute_direct_light(ray, hitpoint, &state);
            } else {
                // The ray hit a lightsource.
                if(hitpoint.is_lightsource) {
                    color_average += hitpoint.accumulated_color;
                    break;
                }
            }
            // Update the direction of the ray.
            ray.direction = sample_brdf(hitpoint, ray, &state);
        }
    }
    colors[first_index] = color_average.x/((float)spp);
    colors[first_index+1] = color_average.y/((float)spp);
    colors[first_index+2] = color_average.z/((float)spp);
}
