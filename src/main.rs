extern crate ocl;
extern crate png;

use std::f32::consts::PI;

use std::env::current_dir;
use std::fs::File;
use std::io::{BufWriter, stdin};

use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use ocl::enums::PlatformInfo;
use png::{BitDepth, ColorType, Encoder, HasParameters};

#[allow(dead_code)]
fn get_opencl_info() {
    let context = Context::builder()
        .build()
        .unwrap();
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

#[allow(dead_code)]
fn pixel_colors() -> ocl::error::Result<Vec<f32>> {
    let mut src = String::new();
    src.push_str(&String::from_utf8_lossy(include_bytes!("structs.cl")));
    src.push_str(&String::from_utf8_lossy(include_bytes!("constants.cl")));
    src.push_str(&String::from_utf8_lossy(include_bytes!("headers.h")));
    src.push_str(&String::from_utf8_lossy(include_bytes!("ray.cl")));
    src.push_str(&String::from_utf8_lossy(include_bytes!("pcg.cl")));
    src.push_str(&String::from_utf8_lossy(include_bytes!("kernels.cl")));
    // We can choose between DeviceType::GPU and DeviceType::CPU depending on wether we want
    // the kernel to run on a GPU or CPU. If there are several devices of a type, we pick by
    // choosing devices[<device number>] when making the queue and the program.
    // To get information about available devices, run get_opencl_info() in main.
    // If running on a CPU we do not risk having the OS kill the kernel.
    // If the kernel is killed, we could make the kernel run faster by setting spp to a lower value.
    let platform = Platform::first()?; // @TODO: We should check all platforms.
    let devices = Device::list(platform, None)?;
    println!("Following opencl devices found.");
    println!("");
    for (index, device) in devices.iter().enumerate() {
        let name = match device.name() {
            Ok(device_name) => {
                device_name
            }
            Err(_) => {
                String::from("Unknown device")
            }
        };
        println!("{:?}: {:?}", index, name);
    }
    println!("");
    let device_index: usize;
    loop {
        let mut index = String::new();
        println!("Input the number of the device you want to use.");
        stdin().read_line(&mut index)
            .expect("Failed to read choice.");
        device_index = match index.trim().parse() {
            Ok(number) => {
                number
            }
            Err(err) => {
                println!("{:?}", err);
                continue;
            }
        };
        break;
    }
    println!("Rendering.");
    let mut colors_average: Vec<f32> = Vec::new();
    for _ in 0..3_000_000 {
        colors_average.push(0.0);
    }
    let iterations = 50;
    // Not to risk having the OS kill the kernel when running on a GPU, it must not run
    // more than a few seconds. This is not enough to get an image with low variance, thus
    // we run it several times and take the average.
    for i in 0_u64..iterations {
        let context = Context::builder().build()?;
        let queue = Queue::new(&context, devices[device_index], None)?;
        let program = Program::builder()
            .src(src.clone())
            .devices(devices[device_index])
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
            .arg(&i)
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
        for (i, out_color) in out_colors.iter().enumerate() {
            colors_average[i] += out_color;
        }
    }
    for i in 0..3_000_000 {
        colors_average[i] /= iterations as f32;
    }
    println!("Finished rendering.");
    Ok(colors_average)
}

#[allow(dead_code)]
fn write_to_file(colors_for_drawing: Vec<u8>) -> std::io::Result<()> {
    let mut current_directory = current_dir()?;
    current_directory.push("gpu.png");
    if let Some(image_filename) = current_directory.to_str() {
        let file = File::create(image_filename)?;
        let buf_writer = &mut BufWriter::new(file);
        let mut encoder = Encoder::new(buf_writer, 1000, 1000);
        encoder.set(ColorType::RGB).set(BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&colors_for_drawing)?;
        Ok(())
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::Other, ""))
    }
}

fn main() {
    // Not to risk having the OS kill the kernel (the code running on the GPU), it must not run
    // more than a few seconds. This is not enough to get an image with low variance, thus
    // we run it several times and take the average.
    let colors = pixel_colors();
    // When writing to file we want u8. Also, we want to avoid having dark pixels being too dark,
    // so we apply some gamma correction.
    let mut colors_for_drawing: Vec<u8> = Vec::new();
    let gamma = 10.0;
    match colors {
        Ok(colors) => {
            for i in 0..1_000_000 {
                let r = colors[3*i];
                let g = colors[3*i+1];
                let b = colors[3*i+2];
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
            println!("Writing to file.");
            let result = write_to_file(colors_for_drawing);
            match result {
                Ok(()) => {
                    println!("Finished writing to file.");
                }
                Err(err) => {
                    println!("{:?}", err);
                }
            }
        }
        Err(err) => {
            println!("{:?}", err);
        }
    }
}
