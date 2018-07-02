extern crate ocl;
extern crate png;

use std::f32::consts::PI;

use std::env::current_dir;
use std::io::{BufWriter};
use std::fs::File;

use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use ocl::enums::PlatformInfo;
use ocl::error::Result;
use ocl::flags::DeviceType;
use png::{BitDepth, ColorType, Encoder, HasParameters};

#[allow(dead_code)]
fn pixel_colors(outer_iteration: u64) -> Result<Vec<f32>> {
    let src = r#"

    // Structs.

    struct triangle {
        float4 node_0;
        float4 node_1;
        float4 node_2;
        float4 e_1;
        float4 e_2;
        float4 normal;
        float4 t_1;
        float4 t_2;
        float4 color;
        bool is_light;
    };

    struct ray {
        float4 position;
        float4 direction;
    };

    // Constants.

    #ifndef M_PI
    #define M_PI 3.14159265358979323846264338327950288
    #endif

    __constant unsigned long multiplier = 6364136223846793005u;
    __constant unsigned long increment = 1442695040888963407u;

    __constant struct triangle triangles[36] = {
        // Left wall.
        {{0.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.2, 0.2, 0.0}, false},
        {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.2, 0.2, 0.0}, false},
        // Right wall.
        {{1.0, 0.0, 1.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, {0.0, 1.0, -1.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.2, 0.8, 0.2, 0.0}, false},
        {{1.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {0.0, 1.0, -1.0, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.2, 0.8, 0.2, 0.0}, false},
        // Floor.
        {{0.0, 1.0, 1.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, -1.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.0, 1.0, 1.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, -1.0, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Ceiling.
        {{0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Far wall.
        {{0.0, 0.0, 1.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, {0.0, 1.0, 1.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Left side of cube 1.
        {{0.1, 0.8, 0.7, 0.0}, {0.1, 1.0, 0.9, 0.0}, {0.1, 1.0, 0.7, 0.0}, {0.0, 0.2, 0.2, 0.0}, {0.0, 0.2, 0.0, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.1, 0.8, 0.7, 0.0}, {0.1, 0.8, 0.9, 0.0}, {0.1, 1.0, 0.9, 0.0}, {0.0, 0.0, 0.2, 0.0}, {0.0, 0.2, 0.2, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Right side of cube 1.
        {{0.3, 0.8, 0.9, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.3, 1.0, 0.9, 0.0}, {0.0, 0.2, -0.2, 0.0}, {0.0, 0.2, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.3, 0.8, 0.9, 0.0}, {0.3, 0.8, 0.7, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.0, 0.0, -0.2, 0.0}, {0.0, 0.2, -0.2, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Bottom of cube 1.
        {{0.1, 1.0, 0.9, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.1, 1.0, 0.7, 0.0}, {0.2, 0.0, -0.2, 0.0}, {0.0, 0.0, -0.2, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.1, 1.0, 0.9, 0.0}, {0.3, 1.0, 0.9, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.0, -0.2, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Top of cube 1.
        {{0.1, 0.8, 0.7, 0.0}, {0.3, 0.8, 0.9, 0.0}, {0.1, 0.8, 0.9, 0.0}, {0.2, 0.0, 0.2, 0.0}, {0.0, 0.0, 0.2, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.1, 0.8, 0.7, 0.0}, {0.3, 0.8, 0.7, 0.0}, {0.3, 0.8, 0.9, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.0, 0.2, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Near side of cube 1.
        {{0.1, 0.8, 0.7, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.1, 1.0, 0.7, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.2, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.1, 0.8, 0.7, 0.0}, {0.3, 0.8, 0.7, 0.0}, {0.3, 1.0, 0.7, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Far side of cube 1.
        {{0.1, 0.8, 0.9, 0.0}, {0.3, 1.0, 0.9, 0.0}, {0.1, 1.0, 0.9, 0.0}, {0.0, 0.2, 0.0, 0.0}, {0.0, 0.2, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.1, 0.8, 0.9, 0.0}, {0.3, 0.8, 0.9, 0.0}, {0.3, 1.0, 0.9, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Left side of cube 2.
        {{0.7, 0.8, 0.7, 0.0}, {0.7, 1.0, 0.9, 0.0}, {0.7, 1.0, 0.7, 0.0}, {0.0, 0.2, 0.2, 0.0}, {0.0, 0.2, 0.0, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.7, 0.8, 0.7, 0.0}, {0.7, 0.8, 0.9, 0.0}, {0.7, 1.0, 0.9, 0.0}, {0.0, 0.0, 0.2, 0.0}, {0.0, 0.2, 0.2, 0.0}, {-1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Right side of cube 2.
        {{0.9, 0.8, 0.9, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.9, 1.0, 0.9, 0.0}, {0.0, 0.2, -0.2, 0.0}, {0.0, 0.2, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.9, 0.8, 0.9, 0.0}, {0.9, 0.8, 0.7, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.0, 0.0, -0.2, 0.0}, {0.0, 0.2, -0.2, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Bottom of cube 2.
        {{0.7, 1.0, 0.9, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.7, 1.0, 0.7, 0.0}, {0.2, 0.0, -0.2, 0.0}, {0.0, 0.0, -0.2, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.7, 1.0, 0.9, 0.0}, {0.9, 1.0, 0.9, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.0, -0.2, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Top of cube 2.
        {{0.7, 0.8, 0.7, 0.0}, {0.9, 0.8, 0.9, 0.0}, {0.7, 0.8, 0.9, 0.0}, {0.2, 0.0, 0.2, 0.0}, {0.0, 0.0, 0.2, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.7, 0.8, 0.7, 0.0}, {0.9, 0.8, 0.7, 0.0}, {0.9, 0.8, 0.9, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.0, 0.2, 0.0}, {0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Near side of cube 2.
        {{0.7, 0.8, 0.7, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.7, 1.0, 0.7, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.2, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.7, 0.8, 0.7, 0.0}, {0.9, 0.8, 0.7, 0.0}, {0.9, 1.0, 0.7, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Far side of cube 2.
        {{0.7, 0.8, 0.9, 0.0}, {0.9, 1.0, 0.9, 0.0}, {0.7, 1.0, 0.9, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.2, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        {{0.7, 0.8, 0.9, 0.0}, {0.9, 0.8, 0.9, 0.0}, {0.9, 1.0, 0.9, 0.0}, {0.2, 0.0, 0.0, 0.0}, {0.2, 0.2, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.8, 0.8, 0.8, 0.0}, false},
        // Ceiling light.
        {{0.3, 1.0e-3, 0.3, 0.0}, {0.7, 1.0e-3, 0.7, 0.0}, {0.3, 1.0e-3, 0.7, 0.0}, {0.4, 0.0, 0.4, 0.0}, {0.0, 0.0, 0.4, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, true},
        {{0.3, 1.0e-3, 0.3, 0.0}, {0.7, 1.0e-3, 0.3, 0.0}, {0.7, 1.0e-3, 0.7, 0.0}, {0.4, 0.0, 0.0, 0.0}, {0.4, 0.0, 0.4, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0}, true},
    };

    // Headers.
    float triangle_ray_distance(int triangle_index, struct ray *restrict r);
    float4 lambertian_on_hemisphere(int triangle_index, unsigned long *restrict state);
    unsigned int next_uint(unsigned long *restrict state);
    float next_float(unsigned long *restrict state);
    void pcg_init(unsigned long seed, unsigned long *restrict state);

    // Functions.

    // The distance to a triangle along a given ray.
    float triangle_ray_distance(int triangle_index, struct ray *restrict r) {
        float4 h = cross(r->direction, triangles[triangle_index].e_2);
        float a = dot(triangles[triangle_index].e_1, h);
        if(a < 1.0e-6 && a > -1.0e-6) {
            return FLT_MAX;
        }
        float f = 1.0/a;
        float4 s = r->position - triangles[triangle_index].node_0;
        float u = dot(s, h)*f;
        if(u < 1.0e-6 || u > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        float4 q = cross(s, triangles[triangle_index].e_1);
        float v = dot(r->direction, q)*f;
        if(v < 1.0e-6 || u+v > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        float distance = dot(triangles[triangle_index].e_2, q)*f;
        if(distance < 1.0e-6) {
            return FLT_MAX;
        }
        return distance;
    }

    // Find a random reflection of a ray that hits a surface with a known normal.
    // The direction is picked from the Lambertian distribution, which is used for perfectly
    // diffuse materials.
    float4 lambertian_on_hemisphere(int triangle_index, unsigned long *restrict state) {
        float r_1 = next_float(state);
        float r_2 = next_float(state);

        float sqrt_arg = 1.0-r_1;
        float sin_theta = sqrt(sqrt_arg);
        float cos_theta = sqrt(r_1);
        float phi = 2.0*M_PI*r_2;

        float4 a = sin_theta*cos(phi)*triangles[triangle_index].t_1;
        float4 b = sin_theta*sin(phi)*triangles[triangle_index].t_2;
        float4 c = cos_theta*triangles[triangle_index].normal;
        return normalize(a+b+c);
    }

    // A random number generator. See https://en.wikipedia.org/wiki/Permuted_congruential_generator
    unsigned int next_uint(unsigned long *restrict state) {
        unsigned long x = *state;
        unsigned int count = (unsigned int)(x >> 59);
        *state = x*multiplier + increment;
        x ^= x >> 18;
        return (((unsigned int)(x >> 27)) >> count) | (((unsigned int)(x >> 27)) << (-count & 31));
    }

    float next_float(unsigned long *restrict state) {
        unsigned int random_uint = next_uint(state);
        return ((float)random_uint)/((float)UINT_MAX);
    }

    void pcg_init(unsigned long seed, unsigned long *restrict state) {
        *state = 2*seed + 1;
        (void)next_uint(state);
    }

    // Kernels.

    __kernel void compute_pixel_color(__global float *colors, unsigned long outer_iteration) {
        unsigned long state = 0;
        // It is very important that we do not initialise the random number generator with the same
        // seed for all pixels and that a pixel gets a different seed for every outer iteration.
        pcg_init(((unsigned long)get_global_id(0))*(outer_iteration+1), &state);
        int first_index = get_global_id(0)*3;
        int spp = 50;
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
            for(int i = 0; i < 5; i++) {
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
        colors[first_index] = color_average[0]/((float)spp);
        colors[first_index+1] = color_average[1]/((float)spp);
        colors[first_index+2] = color_average[2]/((float)spp);
    }
    "#;
    // We can choose between DeviceType::GPU and DeviceType::CPU depending on wether we want
    // the kernel to run on a GPU or CPU. If there are several devices of a type, we pick by
    // choosing devices[<device number>] when making the queue and the program.
    // To get information about available devices, run get_opencl_info() in main.
    // If running on a CPU we do not risk having the OS kill the kernel.
    // If the kernel is killed, we could make the kernel run faster by setting spp to a lower value.
    let platform = Platform::first()?;
    let devices = Device::list(platform, Some(DeviceType::GPU))?;
    let context = Context::builder().build()?;
    let queue = Queue::new(&context, devices[0], None)?;
    let program = Program::builder()
        .src(src)
        .devices(devices[0])
        .build(&context)?;
    let colors = Buffer::builder()
        .queue(queue.clone())
        .len(3_000_000)
        .fill_val(0.0_f32)
        .build()?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("compute_pixel_color")
        .queue(queue.clone())
        .global_work_size(1_000_000)
        .arg(&colors)
        .arg(&outer_iteration)
        .build()?;
    unsafe {
        kernel
            .cmd()
            .enq()?;
    }
    let mut out_colors = vec![0.0_f32; colors.len()];
    colors
        .read(&mut out_colors)
        .enq()?;
    Ok(out_colors)
}

#[allow(dead_code)]
fn get_opencl_info() {
    let context = Context::builder().build().unwrap();
    let devices = context.devices();
    for device in devices {
        println!("{:?}", device);
    }
    let platforms = Platform::list();
    for platform in platforms {
        println!("PLATFORM");
        println!("Profile: {:?}", platform.info(PlatformInfo::Profile));
        println!("Version: {:?}", platform.info(PlatformInfo::Version));
        println!("Name: {:?}", platform.info(PlatformInfo::Name));
        println!("Vendor: {:?}", platform.info(PlatformInfo::Vendor));
        println!("Extensions: {:?}", platform.info(PlatformInfo::Extensions));
        println!("");
        let devices = Device::list(platform, None);
        match devices {
            Ok(devices) => {
                for device in devices {
                    println!("DEVICE");
                    println!("device: {:?}", device);
                    println!("name: {:?}", device.name());
                    println!("vendor: {:?}", device.vendor());
                    println!("all: {:?}", device.to_string());
                    println!("");
                }
            }
            Err(err) => {
                println!("{:?}", err);
            }
        }
    }
}

fn main() {
    //get_opencl_info();
    //return;
    let mut colors_average: Vec<f32> = Vec::new();
    for _ in 0..3_000_000 {
        colors_average.push(0.0);
    }
    let iterations_wanted = 10;
    let mut iterations = 0;
    // Not to risk having the OS kill the kernel (the code running on the GPU), it must not run
    // more than a few seconds. This is not enough to get an image with low variance, thus
    // we run it several times and take the average.
    for i in 0..iterations_wanted {
        let colors = pixel_colors(i);
        match colors {
            Ok(colors) => {
                iterations += 1;
                for (i, color) in colors.iter().enumerate() {
                    colors_average[i] += color;
                }
            }
            Err(err) => {
                println!("{:?}", err);
            }
        }
    }
    for i in 0..3_000_000 {
        colors_average[i] /= iterations as f32;
    }
    // When writing to file we want u8. Also, we want to avoid having dark pixels being too dark,
    // so we apply some gamma correction.
    let mut colors_for_drawing: Vec<u8> = Vec::new();
    let gamma = 10.0;
    for i in 0..1_000_000 {
        let r = colors_average[3*i];
        let g = colors_average[3*i+1];
        let b = colors_average[3*i+2];
        let mut max_intensity = r;
        if g > max_intensity {
            max_intensity = g;
        }
        if b > max_intensity {
            max_intensity = b;
        }
        let factor = if max_intensity > 1.0e-12 {
            let modified_max_intensity = (gamma*max_intensity).atan()/(PI/2.0);
            modified_max_intensity/max_intensity
        } else {
            1.0
        };
        // At this point, factor*r, factor*g and factor*b are all guaranteed to lie in the interval [0, 1].
        colors_for_drawing.push((factor*r*255.0) as u8);
        colors_for_drawing.push((factor*g*255.0) as u8);
        colors_for_drawing.push((factor*b*255.0) as u8);
    }
    // Write to file in the currect directory.
    let mut current_directory = current_dir().unwrap();
    current_directory.push("gpu.png");
    let image_filename = current_directory.to_str().unwrap();
    let file = File::create(image_filename).unwrap();
    let buf_writer = &mut BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, 1000, 1000);
    encoder.set(ColorType::RGB).set(BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&colors_for_drawing).unwrap();
}
