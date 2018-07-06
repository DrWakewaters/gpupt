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
    float4 ambient_color = {0.1, 0.1, 0.1, 0.0};
    for(int i = 0; i < spp; i++) {
        float dr = next_float(&state)*0.0005;
        float theta = next_float(&state)*2.0*M_PI;
        float dx = cos(theta)*dr;
        float dy = sin(theta)*dr;
        float4 point_on_retina = {x+dx, y+dy, -2.0, 0.0};
        float4 direction = pinhole - point_on_retina;
        direction = normalize(direction);
        struct ray r = {pinhole, direction};
        float4 color = {1.0, 1.0, 1.0, 0.0};
        // Loop until we hit a light or the ray hits nothing.
        while(true) {
            float min_distance = FLT_MAX;
            int closest_triangle_index = -1;
            for(int j = 0; j < 36; j++) {
                float distance = triangle_ray_distance(j, &r);
                if(distance < min_distance) {
                    min_distance = distance;
                    closest_triangle_index = j;
                }
            }
            if(closest_triangle_index == -1) {
                color_average += color*ambient_color;
                break;
            }
            color *= triangles[closest_triangle_index].color;
            if(triangles[closest_triangle_index].is_light) {
                color_average += color;
                break;
            }
            float4 displacement = min_distance*r.direction;
            float4 position = r.position + displacement;
            r.position = position;
            float4 direction = lambertian_on_hemisphere(closest_triangle_index, &state);
            r.direction = direction;
        }
    }
    colors[first_index] = color_average.x/((float)spp);
    colors[first_index+1] = color_average.y/((float)spp);
    colors[first_index+2] = color_average.z/((float)spp);
}
