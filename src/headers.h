float triangle_ray_distance(int triangle_index, struct ray *restrict r);
float4 lambertian_on_hemisphere(int triangle_index, unsigned long *restrict state);
unsigned int next_uint(unsigned long *restrict state);
float next_float(unsigned long *restrict state);
void pcg_init(unsigned long seed, unsigned long *restrict state);
