// Other functions.
void compute_local_coordinate_system(float4 normal, float4 *restrict t_1, float4 *restrict t_2);
Ray create_ray(float x, float y, unsigned long *restrict state);

// Random numbers.
unsigned int next_uint(unsigned long *restrict state);
unsigned int next_uint_in_range(unsigned int inclusive_minimum, unsigned int exclusive_maximum, unsigned long *restrict state);
float next_float(unsigned long *restrict state);
void pcg_init(unsigned long seed, unsigned long *restrict state);

// Ray-surface intersection detection.
float triangle_ray_distance(int triangle_index, Ray ray);
float sphere_ray_distance(int sphere_index, Ray ray);

// Point in triangle detection.
bool point_in_triangle(float4 point, float4 node_0, float4 node_1, float4 node_2);
bool same_side(float4 test_point, float4 point_inside, float4 first_node, float4 second_node);

// Compute direct light.
float4 compute_direct_light(Ray ray, Hitpoint hitpoint, unsigned long *restrict state);
float4 compute_direct_light_inner(Hitpoint hitpoint, float4 light_color, float4 light_normal, float4 light_position, float lambertian_probability, float maximum_specular_angle, bool is_opaque);
Hitpoint compute_hitpoint(Ray ray, float4 accumulated_color);

// Random points on surfaces.
float4 random_on_triangle(unsigned int triangle_index, unsigned long *restrict state);
float4 random_on_sphere(unsigned int sphere_index, unsigned long *restrict state);

// Sample brdf.
float4 sample_brdf(Hitpoint hitpoint, Ray ray, unsigned long *restrict state);
float4 sample_brdf_lambertian(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state);
float4 sample_brdf_specular(float4 normal, float4 incoming_direction, float maximum_specular_angle, unsigned long *restrict state);

// Compute brdf.
float compute_brdf(float4 incoming_direction, float4 outgoing_direction, float4 normal, float refractive_index_1, float refractive_index_2, float lambertian_probability, float maximum_specular_angle, bool is_opaque);
float compute_brdf_lambertian(float4 outgoing_direction, float4 normal);
float compute_brdf_specular(float4 incoming_direction, float4 outgoing_direction, float4 normal, float maximum_specular_angle);
