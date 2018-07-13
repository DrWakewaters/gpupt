#[allow(dead_code)]
pub fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
	[left[0]+right[0], left[1]+right[1], left[2]+right[2]]
}

#[allow(dead_code)]
pub fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
	[left[1]*right[2]-left[2]*right[1], left[2]*right[0]-left[0]*right[2], left[0]*right[1]-left[1]*right[0]]
}

#[allow(dead_code)]
pub fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
	left[0]*right[0]+left[1]*right[1]+left[2]*right[2]
}

#[allow(dead_code)]
pub fn normalised(vector: [f32; 3]) -> [f32; 3] {
	let norm = (vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]).sqrt();
	[vector[0]/norm, vector[1]/norm, vector[2]/norm]
}

#[allow(dead_code)]
pub fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
	[left[0]-right[0], left[1]-right[1], left[2]-right[2]]
}
