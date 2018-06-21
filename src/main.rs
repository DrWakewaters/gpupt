extern crate ocl;
extern crate png;

mod pngfile;

use std::env::current_dir;
use std::io::{BufWriter};
use std::fs::File;

use ocl::ProQue;
use png::{BitDepth, ColorType, Encoder, HasParameters};

fn pixel_colors() -> Vec<f32> {
    let src = r#"

    struct triangle {
        double node_0[3];
        double node_1[3];
        double node_2[3];
        double e_1[3];
        double e_2[3];
        double normal[3];
        float color;
    };

    struct ray {
        double position[3];
        double direction[3];
    };

    void cross_33(double *left, double *right, double *result) {
        result[0] = left[1]*right[2]-left[2]*right[1];
        result[1] = left[2]*right[0]-left[0]*right[2];
        result[2] = left[0]*right[1]-left[1]*right[0];
    }

    void dot_33(double *left, double *right, double *result) {
        *result = left[0]*right[0]+left[1]*right[1]+left[2]*right[2];
    }

    void sub_33(double *left, double *right, double *result) {
        result[0] = left[0]-right[0];
        result[1] = left[1]-right[1];
        result[2] = left[2]-right[2];
    }

    void normalise_3(double *vector) {
        double length = sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
        vector[0] /= length;
        vector[1] /= length;
        vector[2] /= length;
    }

    double triangle_ray_distance(struct triangle t, struct ray r) {
        double h[3] = {0.0, 0.0, 0.0};
        cross_33(r.direction, t.e_2, h);
        double a = 0.0;
        dot_33(t.e_1, h, &a);
        if(a < 1.0e-6 && a > -1.0e-6) {
            return FLT_MAX;
        }
        double f = 1.0/a;
        double s[3] = {0.0, 0.0, 0.0};
        sub_33(r.position, t.node_0, s);
        double u = 0.0;
        dot_33(s, h, &u);
        u *= f;
        if(u < 1.0e-6 || u > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        double q[3] = {0.0, 0.0, 0.0};
        cross_33(s, t.e_1, q);
        double v = 0.0;
        dot_33(r.direction, q, &v);
        v *= f;
        if(v < 1.0e-6 || u+v > 1.0-1.0e-6) {
            return FLT_MAX;
        }
        double distance = 0.0;
        dot_33(t.e_2, q, &distance);
        distance *= f;
        if(distance < 1.0e-6) {
            return FLT_MAX;
        }
        return distance;
    }

    __kernel void compute_pixel_color(__global float* buffer) {
        double y = 1000-get_global_id(0)/1000;
        double x = 1000-get_global_id(0)%1000;
        double pinhole[3] = {500.0, 500.0, -1000.0};
        double point_on_retina[3] = {x, y, -2000.0};
        double direction[3] = {0.0, 0.0, 0.0};
        sub_33(pinhole, point_on_retina, direction);
        normalise_3(direction);

        struct ray r = {{pinhole[0], pinhole[1], pinhole[2]}, {direction[0], direction[1], direction[2]}};

        struct triangle triangles[10] = {
            {{0.0, 0.0, 0.0}, {0.0, 1000.0, 1000.0}, {0.0, 1000.0, 0.0}, {0.0, 1000.0, 1000.0}, {0.0, 1000.0, 0.0}, {1.0, 0.0, 0.0}, 255.0 + 0.0*1000.0 + 0.0*1000000.0},
            {{0.0, 0.0, 0.0}, {0.0, 0.0, 1000.0}, {0.0, 1000.0, 1000.0}, {0.0, 0.0, 1000.0}, {0.0, 1000.0, 1000.0}, {1.0, 0.0, 0.0}, 255.0 + 0.0*1000.0 + 0.0*1000000.0},
            {{1000.0, 0.0, 1000.0}, {1000.0, 1000.0, 0.0}, {1000.0, 1000.0, 1000.0}, {0.0, 1000.0, -1000.0}, {0.0, 1000.0, 0.0}, {-1.0, 0.0, 0.0}, 0.0 + 255.0*1000.0 + 0.0*1000000.0},
            {{1000.0, 0.0, 1000.0}, {1000.0, 0.0, 0.0}, {1000.0, 1000.0, 0.0}, {0.0, 0.0, -1000.0}, {0.0, 1000.0, -1000.0}, {-1.0, 0.0, 0.0}, 0.0 + 255.0*1000.0 + 0.0*1000000.0},
            {{0.0, 1000.0, 1000.0}, {1000.0, 1000.0, 0.0}, {0.0, 1000.0, 0.0}, {1000.0, 0.0, -1000.0}, {0.0, 0.0, -1000.0}, {0.0, -1.0, 0.0}, 0.0 + 0.0*1000.0 + 255.0*1000000.0},
            {{0.0, 1000.0, 1000.0}, {1000.0, 1000.0, 1000.0}, {1000.0, 1000.0, 0.0}, {1000.0, 0.0, 0.0}, {1000.0, 0.0, -1000.0}, {0.0, -1.0, 0.0}, 0.0 + 0.0*1000.0 + 255.0*1000000.0},
            {{0.0, 0.0, 0.0}, {1000.0, 0.0, 1000.0}, {0.0, 0.0, 1000.0}, {1000.0, 0.0, 1000.0}, {0.0, 0.0, 1000.0}, {0.0, 1.0, 0.0}, 255.0 + 255.0*1000.0 + 255.0*1000000.0},
            {{0.0, 0.0, 0.0}, {1000.0, 0.0, 0.0}, {1000.0, 0.0, 1000.0}, {1000.0, 0.0, 0.0}, {1000.0, 0.0, 1000.0}, {0.0, 1.0, 0.0}, 255.0 + 255.0*1000.0 + 255.0*1000000.0},
            {{0.0, 0.0, 1000.0}, {1000.0, 1000.0, 1000.0}, {0.0, 1000.0, 1000.0}, {1000.0, 1000.0, 0.0}, {0.0, 1000.0, 0.0}, {0.0, -1.0, 0.0}, 120.0 + 80.0*1000.0 + 40.0*1000000.0},
            {{0.0, 0.0, 1000.0}, {1000.0, 0.0, 1000.0}, {1000.0, 1000.0, 1000.0}, {1000.0, 0.0, 0.0}, {1000.0, 1000.0, 0.0}, {0.0, -1.0, 0.0}, 120.0 + 80.0*1000.0 + 40.0*1000000.0}
        };

        double min_distance = FLT_MAX;
        double color = 0.0;
        double distance = 0.0;
        for(int i = 0; i < 10; i++) {
            distance = triangle_ray_distance(triangles[i], r);
            if(distance < min_distance) {
                min_distance = distance;
                color = triangles[i].color;
            }
        }

        buffer[get_global_id(0)] = color;
    }

    "#;
    let pro_que = ProQue::builder()
        .src(src)
        .dims(1000000)
        .build()
        .unwrap();
    let buffer = pro_que.create_buffer::<f32>().unwrap();
    let kernel = pro_que.kernel_builder("compute_pixel_color")
        .arg(&buffer)
        .build()
        .unwrap();
    unsafe {
        kernel.enq();
    }
    let mut vec = vec![0.0_f32; buffer.len()];
    buffer.read(&mut vec).enq();
    vec
}

fn main() {
    let colors = pixel_colors();
    let mut colors_for_drawing: Vec<u8> = Vec::new();
    for color in colors {
        let color = color as u32;
        colors_for_drawing.push(((color/1_000_000)%1_000) as u8);
        colors_for_drawing.push(((color/1_000)%1_000) as u8);
        colors_for_drawing.push((color%1_000) as u8);
    }
    let current_directory = current_dir();
    match current_directory {
        Ok(mut directory) => {
            directory.push("gpu.png");
            let image_filename = directory.to_str().unwrap();
            let file = File::create(image_filename).unwrap();
            let buf_writer = &mut BufWriter::new(file);
            let mut encoder = Encoder::new(buf_writer, 1000, 1000);
            encoder.set(ColorType::RGB).set(BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&colors_for_drawing);
        }
        Err(e) => {
			println!("{:?}", e);
		}
    }
}
