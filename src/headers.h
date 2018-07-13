float triangle_ray_distance(int triangle_index, struct ray *restrict r);
float sphere_ray_distance(int sphere_index, struct ray *restrict r);
void compute_local_coordinate_system(float4 normal, float4 *t_1, float4 *t_2);
float4 lambertian_on_hemisphere(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state);
float4 specular_on_hemisphere(float4 normal, float4 incoming_direction);
unsigned int next_uint(unsigned long *restrict state);
float next_float(unsigned long *restrict state);
void pcg_init(unsigned long seed, unsigned long *restrict state);
