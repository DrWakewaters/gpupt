// Other functions.
void compute_local_coordinate_system(float4 normal, float4 *restrict t_1, float4 *restrict t_2);
Ray create_ray(float x, float y, Camera camera, unsigned long *restrict state);

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

// Update.
void update(Ray *ray, Hitpoint *hitpoint, unsigned long *restrict state);

// Sampling.
float4 sample_uniform_on_triangle(unsigned int triangle_index, unsigned long *restrict state);
float4 sample_uniform_on_sphere(unsigned long *restrict state);
float4 sample_uniform_on_hemisphere(float4 normal, unsigned long *restrict state);
float4 sample_cos_weighted_on_hemisphere(float4 normal, float4 t_1, float4 t_2, unsigned long *restrict state);
float4 sample_cone(float4 direction, float maximum_angle, unsigned long *restrict state);
float4 sample_cone_fast(float4 direction, float4 t_1, float4 t_2, float maximum_angle, unsigned long *restrict state);
float4 sample(float4 position, float4 incoming_direction, float4 normal, float4 t_1, float4 t_2, float maximum_specular_angle, float specular_reflection_probability, bool *may_sample_cone, unsigned long *restrict state);

// Color modification due to the sampling.
float color_modifier(float4 position, float4 incoming_direction, float4 outgoing_direction, float4 normal, bool may_sample_cone);
float cos_weighted_probability_density(float4 outgoing_direction, float4 normal);
float cone_probability_density(float4 direction_to_light_center, float4 outgoing_direction, float maximum_angle);
