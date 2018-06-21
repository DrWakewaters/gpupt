
//use std::io::{BufReader, BufWriter, Error};
//use std::result::Result;
//use std::fs::File;

//use bincode::{serialize_into, deserialize_from};
//use png::{BitDepth, ColorType, Encoder, HasParameters};
//use time::now;

//use datafordrawing::DataForDrawing;
//use physicsscene::PhysicsScene;
//use rendereroutput::RendererOutput;
/*
pub fn make_file(mut renderer_output: &mut RendererOutput, image_filename: &str) {
	let data_for_drawing = DataForDrawing::new(&mut renderer_output);
	let draw_result = draw_frame(&image_filename, &data_for_drawing);
	if let Err(e) = draw_result {
		println!("{:?}", e);
	}
}

fn draw_frame(image_filename: &str, data_for_drawing: &DataForDrawing) -> Result<(), Error> {
	let file = File::create(image_filename)?;
	let buf_writer = &mut BufWriter::new(file);
	let mut encoder = Encoder::new(buf_writer, data_for_drawing.width, data_for_drawing.height);
	encoder.set(ColorType::RGB).set(BitDepth::Eight);
	let mut writer = encoder.write_header()?;
	writer.write_image_data(&data_for_drawing.colors)?;
	Ok(())
}

pub fn write_frame(text_filename: &str, renderer_output: &RendererOutput) -> Result<(), Error> {
	let tm = now();
	print!("Writing to text file starts at {}:{}:{}.{}. ", tm.tm_hour, tm.tm_min, tm.tm_sec, tm.tm_nsec/10_000_000);
	let file = File::create(text_filename)?;
	let buf_writer = BufWriter::new(file);
	let result = serialize_into(buf_writer, renderer_output);
	if let Err(err) = result {
		println!("{:?}", err);
	}
	let duration = now() - tm;
	println!("It took {}:{}:{}.{}.", duration.num_hours(), duration.num_minutes()%60, duration.num_seconds()%60, (duration.num_milliseconds()%1000)/10);
	Ok(())
}

pub fn read_frame(text_filename: &str) -> Result<RendererOutput, Error> {
	let file = File::open(text_filename)?;
	let buf_reader = BufReader::new(file);
	let deserialized: RendererOutput = deserialize_from(buf_reader).unwrap();
	Ok(deserialized)
}

pub fn write_scene(text_filename: &str, scene: &PhysicsScene) -> Result<(), Error> {
	let file = File::create(text_filename)?;
	let buf_writer = BufWriter::new(file);
	let result = serialize_into(buf_writer, scene);
	if let Err(err) = result {
		println!("{:?}", err);
	}
	Ok(())
}

pub fn read_scene(text_filename: &str) -> Result<PhysicsScene, Error> {
	let file = File::open(text_filename)?;
	let buf_reader = BufReader::new(file);
	let deserialized: PhysicsScene = deserialize_from(buf_reader).unwrap();
	Ok(deserialized)
}
*/
