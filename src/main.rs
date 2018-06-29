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

    // Constants.

    #ifndef M_PI
    #define M_PI 3.14159265358979323846264338327950288
    #endif

    __constant unsigned long multiplier = 6364136223846793005u;
    __constant unsigned long increment = 1442695040888963407u;

    // Structs.

    struct triangle {
        float node_0[3];
        float node_1[3];
        float node_2[3];
        float e_1[3];
        float e_2[3];
        float normal[3];
        float t_3[3];
        float t_1[3];
        float t_2[3];
        float color[3];
        float lambertian_probability;
        bool is_light;
    };

    struct ray {
        float position[3];
        float direction[3];
    };

    // Headers.

    void cross_33(float *left, float *right, float *result);
    void dot_33(float *left, float *right, float *result);
    void sub_33(float *left, float *right, float *result);
    void add_33(float *left, float *right, float *result);
    void mul_13(float scalar, float *vector, float *result);
    void length_3(float *vector, float *result);
    void normalise_3(float *vector);
    void set_basis_vectors(float *u_1, float *u_2, float *u_3, float *normal);
    void lambertian_on_hemisphere(float *normal, float *t_3, float *t_1, float *t_2, float *direction, unsigned long *state);
    void specular_on_hemisphere(float *normal, float *incoming_direction, float *direction);
    unsigned int next_uint(unsigned long *state);
    float next_float(unsigned long *state);
    void pcg_init(unsigned long seed, unsigned long *state);
    float triangle_ray_distance(struct triangle *t, struct ray *r);

    // Functions.

    void cross_33(float *left, float *right, float *result) {
        result[0] = left[1]*right[2]-left[2]*right[1];
        result[1] = left[2]*right[0]-left[0]*right[2];
        result[2] = left[0]*right[1]-left[1]*right[0];
    }

    void dot_33(float *left, float *right, float *result) {
        *result = left[0]*right[0]+left[1]*right[1]+left[2]*right[2];
    }

    void sub_33(float *left, float *right, float *result) {
        result[0] = left[0]-right[0];
        result[1] = left[1]-right[1];
        result[2] = left[2]-right[2];
    }

    void add_33(float *left, float *right, float *result) {
        result[0] = left[0]+right[0];
        result[1] = left[1]+right[1];
        result[2] = left[2]+right[2];
    }

    void mul_13(float scalar, float *vector, float *result) {
        result[0] = scalar*vector[0];
        result[1] = scalar*vector[1];
        result[2] = scalar*vector[2];
    }

    void normalise_3(float *vector) {
        float length = sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
        vector[0] /= length;
        vector[1] /= length;
        vector[2] /= length;
    }

    void length_3(float *vector, float *result) {
        *result = sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
    }

    // The distance to a triangle along a given ray.
    float triangle_ray_distance(struct triangle *t, struct ray *r) {
        float h[3] = {0.0, 0.0, 0.0};
        cross_33(r->direction, t->e_2, h);
        float a = 0.0;
        dot_33(t->e_1, h, &a);
        if(a < 1.0e-6 && a > -1.0e-6) {
            return FLT_MAX;
        }
        float f = 1.0/a;
        float s[3] = {0.0, 0.0, 0.0};
        sub_33(r->position, t->node_0, s);
        float u = 0.0;
        dot_33(s, h, &u);
        u *= f;
        if(u < 1.0e-6 || u > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        float q[3] = {0.0, 0.0, 0.0};
        cross_33(s, t->e_1, q);
        float v = 0.0;
        dot_33(r->direction, q, &v);
        v *= f;
        if(v < 1.0e-6 || u+v > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        float distance = 0.0;
        dot_33(t->e_2, q, &distance);
        distance *= f;
        if(distance < 1.0e-6) {
            return FLT_MAX;
        }
        return distance;
    }

    // Use the Gram-Schmidt process to find an ON basis. Let normal be one of the basis vectors.
    // Might not be the quickest way to achieve this. The implementation can be improved.
    void set_basis_vectors(float *u_1, float *u_2, float *u_3, float *normal) {
        u_1[0] = normal[0];
        u_1[1] = normal[1];
        u_1[2] = normal[2];
        normalise_3(u_1);
        float v_2[3] = {0.0, 0.0, 1.0};
        float v_3[3] = {0.0, 1.0, 0.0};

        float u_1_v_2 = 0.0;
        dot_33(u_1, v_2, &u_1_v_2);
        float u_1_v_2_u_1[3] = {0.0, 0.0, 0.0};
        mul_13(u_1_v_2, u_1, u_1_v_2_u_1);
        sub_33(v_2, u_1_v_2_u_1, u_2);

        float u_2_length = 0.0;
        length_3(u_2, &u_2_length);
        if(u_2_length < 1.0e-6) {
            v_2[0] = 1.0;
            v_2[1] = 0.0;
            v_2[2] = 0.0;

            float u_1_v_2 = 0.0;
            dot_33(u_1, v_2, &u_1_v_2);
            float u_1_v_2_u_1[3] = {0.0, 0.0, 0.0};
            mul_13(u_1_v_2, u_1, u_1_v_2_u_1);
            sub_33(v_2, u_1_v_2_u_1, u_2);
        }
        normalise_3(u_2);

        float u_1_v_3 = 0.0;
        dot_33(u_1, v_3, &u_1_v_3);
        float u_1_v_3_u_1[3] = {0.0, 0.0, 0.0};
        mul_13(u_1_v_3, u_1, u_1_v_3_u_1);

        float u_2_v_3 = 0.0;
        dot_33(u_2, v_3, &u_2_v_3);
        float u_2_v_3_u_2[3] = {0.0, 0.0, 0.0};
        mul_13(u_2_v_3, u_2, u_2_v_3_u_2);

        float v_3_minus_u_1_v_3_u_1[3] = {0.0, 0.0, 0.0};
        sub_33(v_3, u_1_v_3_u_1, v_3_minus_u_1_v_3_u_1);

        sub_33(v_3_minus_u_1_v_3_u_1, u_2_v_3_u_2, u_3);

        float u_3_length = 0.0;
        length_3(u_3, &u_3_length);
        if(u_3_length < 1.0e-6) {
            v_3[0] = 1.0;
            v_3[1] = 0.0;
            v_3[2] = 0.0;

            float u_1_v_3 = 0.0;
            dot_33(u_1, v_3, &u_1_v_3);
            float u_1_v_3_u_1[3] = {0.0, 0.0, 0.0};
            mul_13(u_1_v_3, u_1, u_1_v_3_u_1);

            float u_2_v_3 = 0.0;
            dot_33(u_2, v_3, &u_2_v_3);
            float u_2_v_3_u_2[3] = {0.0, 0.0, 0.0};
            mul_13(u_2_v_3, u_2, u_2_v_3_u_2);

            float v_3_minus_u_1_v_3_u_1[3] = {0.0, 0.0, 0.0};
            sub_33(v_3, u_1_v_3_u_1, v_3_minus_u_1_v_3_u_1);

            sub_33(v_3_minus_u_1_v_3_u_1, u_2_v_3_u_2, u_3);
        }
        normalise_3(u_3);
    }

    // Find the mirror-like reflection of a ray that hits a surface with a known normal.
    void specular_on_hemisphere(float *normal, float *incoming_direction, float *direction) {
        float factor = 0.0;
        dot_33(incoming_direction, normal, &factor);
        factor *= 2.0;
        float direction_modifier[3] = {0.0, 0.0, 0.0};
        mul_13(factor, normal, direction_modifier);
        sub_33(incoming_direction, direction_modifier, direction);
        normalise_3(direction);
    }

    // Find a random reflection of a ray that hits a surface with a known normal.
    // The direction is picked from the Lambertian distribution, which is used for perfectly
    // diffuse materials.
    void lambertian_on_hemisphere(float *normal, float *t_3, float *t_1, float *t_2, float *direction, unsigned long *state) {
        float r_1 = next_float(state);
        float r_2 = next_float(state);

        float sqrt_arg = 1.0-r_1;
        float sin_theta = sqrt(sqrt_arg);
        float cos_theta = sqrt(r_1);
        float phi = 2.0*M_PI*r_2;

        float a[3] = {0.0, 0.0, 0.0};
        mul_13(sin_theta*cos(phi), t_1, a);

        float b[3] = {0.0, 0.0, 0.0};
        mul_13(sin_theta*sin(phi), t_2, b);

        float c[3] = {0.0, 0.0, 0.0};
        mul_13(cos_theta, t_3, c);

        float d[3] = {0.0, 0.0, 0.0};

        add_33(a, b, d);
        add_33(d, c, direction);

        normalise_3(direction);

        float dot_product = 0.0;
        dot_33(direction, normal, &dot_product);
    }

    // A random number generator. See https://en.wikipedia.org/wiki/Permuted_congruential_generator
    unsigned int next_uint(unsigned long *state) {
        unsigned long x = *state;
        unsigned int count = (unsigned int)(x >> 59);
        *state = x*multiplier + increment;
        x ^= x >> 18;
        return (((unsigned int)(x >> 27)) >> count) | (((unsigned int)(x >> 27)) << (-count & 31));
    }

    float next_float(unsigned long *state) {
        return ((float)next_uint(state))/((float)UINT_MAX);
    }

    void pcg_init(unsigned long seed, unsigned long *state) {
        *state = seed*0x4d595df4d0f33173 + increment;
        (void)next_uint(state);
    }

    // Kernels.

    __kernel void compute_pixel_color(__global float *colors, unsigned long outer_iteration) {
        unsigned long state = 0;
        // It is very important that we do not initialise the random number generator with the same
        // seed for all pixels and that a pixel gets a different seed for every outer iteration.
        pcg_init(((unsigned long)get_global_id(0))*(outer_iteration+1), &state);
        struct triangle triangles[34] = {
            // Left wall.
            {{0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.2, 0.2}, 1.0, false},
            {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.2, 0.2}, 1.0, false},
            // Right wall.
            {{1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.2, 0.8, 0.2}, 1.0, false},
            {{1.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.2, 0.8, 0.2}, 1.0, false},
            // Floor.
            {{0.0, 1.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Ceiling light.
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 1.0, true},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 1.0, true},
            // Far wall.
            {{0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Left side of cube 1.
            {{0.1, 0.8, 0.7}, {0.1, 1.0, 0.9}, {0.1, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.1, 0.8, 0.7}, {0.1, 0.8, 0.9}, {0.1, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Right side of cube 1.
            {{0.3, 0.8, 0.9}, {0.3, 1.0, 0.7}, {0.3, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.3, 0.8, 0.9}, {0.3, 0.8, 0.7}, {0.3, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Bottom of cube 1.
            {{0.1, 1.0, 0.9}, {0.3, 1.0, 0.7}, {0.1, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.1, 1.0, 0.9}, {0.3, 1.0, 0.9}, {0.3, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Top of cube 1.
            {{0.1, 0.8, 0.7}, {0.3, 0.8, 0.9}, {0.1, 0.8, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.1, 0.8, 0.7}, {0.3, 0.8, 0.7}, {0.3, 0.8, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Near side of cube 1.
            {{0.1, 0.8, 0.7}, {0.3, 1.0, 0.7}, {0.1, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.1, 0.8, 0.7}, {0.3, 0.8, 0.7}, {0.3, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Far side of cube 1.
            {{0.1, 0.8, 0.9}, {0.3, 1.0, 0.9}, {0.1, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            {{0.1, 0.8, 0.9}, {0.3, 0.8, 0.9}, {0.3, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 0.0, false},
            // Left side of cube 2.
            {{0.7, 0.8, 0.7}, {0.7, 1.0, 0.9}, {0.7, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.7, 0.8, 0.7}, {0.7, 0.8, 0.9}, {0.7, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Right side of cube 2.
            {{0.9, 0.8, 0.9}, {0.9, 1.0, 0.7}, {0.9, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.9, 0.8, 0.9}, {0.9, 0.8, 0.7}, {0.9, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Bottom of cube 2.
            {{0.7, 1.0, 0.9}, {0.9, 1.0, 0.7}, {0.7, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.7, 1.0, 0.9}, {0.9, 1.0, 0.9}, {0.9, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Top of cube 2.
            {{0.7, 0.8, 0.7}, {0.9, 0.8, 0.9}, {0.7, 0.8, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.7, 0.8, 0.7}, {0.9, 0.8, 0.7}, {0.9, 0.8, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Near side of cube 2.
            {{0.7, 0.8, 0.7}, {0.9, 1.0, 0.7}, {0.7, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.7, 0.8, 0.7}, {0.9, 0.8, 0.7}, {0.9, 1.0, 0.7}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            // Far side of cube 2.
            {{0.7, 0.8, 0.9}, {0.9, 1.0, 0.9}, {0.7, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false},
            {{0.7, 0.8, 0.9}, {0.9, 0.8, 0.9}, {0.9, 1.0, 0.9}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.8, 0.8, 0.8}, 1.0, false}
        };
        for(int i = 0; i < 34; i++) {
            sub_33(&triangles[i].node_1, &triangles[i].node_0, &triangles[i].e_1);
            sub_33(&triangles[i].node_2, &triangles[i].node_0, &triangles[i].e_2);
            set_basis_vectors(&triangles[i].t_3, &triangles[i].t_1, &triangles[i].t_2, &triangles[i].normal);
        }
        int first_index = get_global_id(0)*3;
        int spp = 10;
        float color_average[3] = {0.0, 0.0, 0.0};
        float y = ((float)(1000-get_global_id(0)/1000))/1000.0;
        float x = ((float)(1000-get_global_id(0)%1000))/1000.0;
        float pinhole[3] = {0.5, 0.5, -1.0};
        // Loop spp times and take the average color.
        for(int i = 0; i < spp; i++) {
            float dr = next_float(&state)*0.0005;
            float theta = next_float(&state)*2.0*M_PI;
            float dx = cos(theta)*dr;
            float dy = sin(theta)*dr;
            float point_on_retina[3] = {x+dx, y+dy, -2.0};
            float direction[3] = {0.0, 0.0, 0.0};
            sub_33(pinhole, point_on_retina, direction);
            normalise_3(direction);
            struct ray r = {{pinhole[0], pinhole[1], pinhole[2]}, {direction[0], direction[1], direction[2]}};
            float color[3] = {1.0, 1.0, 1.0};
            // Loop until we hit a light.
            while(true) {
                float min_distance = FLT_MAX;
                int closest_triangle_index = -1;
                for(int j = 0; j < 34; j++) {
                    float distance = triangle_ray_distance(&triangles[j], &r);
                    if(distance < min_distance) {
                        min_distance = distance;
                        closest_triangle_index = j;
                    }
                }
                // @TODO: Breaking is most likely bad for GPUs. Investigate this and come up with
                // a solution. 
                if(closest_triangle_index == -1) {
                    break;
                }
                color[0] *= triangles[closest_triangle_index].color[0];
                color[1] *= triangles[closest_triangle_index].color[1];
                color[2] *= triangles[closest_triangle_index].color[2];
                if(triangles[closest_triangle_index].is_light) {
                    color_average[0] += color[0];
                    color_average[1] += color[1];
                    color_average[2] += color[2];
                    break;
                }
                float displacement[3] = {0.0, 0.0, 0.0};
                mul_13(min_distance, r.direction, displacement);
                float position[3] = {0.0, 0.0, 0.0};
                add_33(r.position, displacement, position);
                r.position[0] = position[0];
                r.position[1] = position[1];
                r.position[2] = position[2];
                float direction[3] = {0.0, 0.0, 0.0};
                float random = next_float(&state);
                // @TODO: Branching is very bad for GPUs, so come up with a solution.
                //if(random < triangles[closest_triangle_index].lambertian_probability) {
                    lambertian_on_hemisphere(triangles[closest_triangle_index].normal, triangles[closest_triangle_index].t_3, triangles[closest_triangle_index].t_1, triangles[closest_triangle_index].t_2, direction, &state);
                //} else {
                //    specular_on_hemisphere(triangles[closest_triangle_index].normal, r.direction, direction);
                //}
                r.direction[0] = direction[0];
                r.direction[1] = direction[1];
                r.direction[2] = direction[2];
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
    let iterations_wanted = 100;
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
    let gamma = 5.0;
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
