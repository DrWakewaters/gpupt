typedef struct Triangle {
    float4 node_0;
    float4 node_1;
    float4 node_2;
    float4 e_1;
    float4 e_2;
    float4 normal;
    float4 t_1;
    float4 t_2;
    float4 color;
    float4 emission;
    float lambertian_probability;
    float refractive_index;
    bool is_opaque;
    bool is_lightsource;
} Triangle;

typedef struct Sphere {
    float4 position;
    float radius;
    float4 color;
    float4 emission;
    float lambertian_probability;
    float refractive_index;
    bool is_opaque;
    bool is_lightsource;
} Sphere;

typedef struct Ray {
    float4 position;
    float4 direction;
} Ray;

typedef struct Hitpoint {
    float4 position;
    float4 incoming_direction;
    float4 normal;
    float4 t_1;
    float4 t_2;
    float4 color;
    float4 emission;
    float4 accumulated_color;
    float distance_from_previous;
    float lambertian_probability;
    float refractive_index;
    bool is_opaque;
    bool is_lightsource;
    bool hit_surface;
    bool hit_from_outside;
} Hitpoint;
