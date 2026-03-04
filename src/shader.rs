use nalgebra::{Vector2, Vector3, max};

use crate::texture;

pub struct FragmentShaderPayload {
    pub barycentric: Vector3<f32>,
    pub view_pos: Vector3<f32>,
    pub color: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub tex_coords: Vector2<f32>,
    pub texture: Option<texture::Texture>,
    pub screen_heights: Vector3<f32>,
}

impl FragmentShaderPayload {
    pub fn new(barycentric: Vector3<f32>, view_pos: Vector3<f32>, color: Vector3<f32>, normal: Vector3<f32>, tex_coords: Vector2<f32>, texture: Option<texture::Texture>, screen_heights: Vector3<f32>) -> Self {
        Self {
            barycentric,
            view_pos,
            color,
            normal,
            tex_coords,
            texture,
            screen_heights
        }
    }
}

pub struct VertexShaderPayload {
    position: Vector3<f32>,
}

struct PointLight {
    position: Vector3<f32>,
    intensity: Vector3<f32>,
}

pub fn vertex_shader(payload: &VertexShaderPayload) -> Vector3<f32> {
    payload.position
}

pub fn normal_fragment_shader(payload: &FragmentShaderPayload) -> Vector3<f32> {
    let return_color = (payload.normal.fixed_rows::<3>(0).normalize() + Vector3::<f32>::new(1.0, 1.0, 1.0)) / 2.0;
    Vector3::<f32>::new(return_color.x, return_color.y, return_color.z)
}

pub fn texture_fragment_shader() {
    
}

pub fn blinnphong_fragment_shader(payload: &FragmentShaderPayload) -> Vector3<f32> {
    let ka = Vector3::<f32>::new(0.005, 0.005, 0.005);
    // let kd = payload.color;
    let kd = Vector3::<f32>::new(1.0, 1.0, 1.0);
    let ks = Vector3::<f32>::new(0.7937, 0.7937, 0.7937);
    let l1 = PointLight{
        position: Vector3::<f32>::new(20.0, 20.0, 20.0),
        intensity: Vector3::<f32>::new(500.0, 500.0, 500.0),
    };
    let l2 = PointLight{
        position: Vector3::<f32>::new(-20.0, 20.0, 0.0),
        intensity: Vector3::<f32>::new(1000.0, 1000.0, 1000.0),
    };
    let lights: Vec<PointLight> = vec![l1, l2];
    let amb_light_intensity = Vector3::<f32>::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::<f32>::new(0.0, 0.0, 4.0);
    let p = 60.0;
    let color = payload.color;
    let point = payload.view_pos;
    let normal = payload.normal;
    let mut result_color = Vector3::<f32>::new(0.0, 0.0, 0.0);

    result_color += ka.component_mul(&amb_light_intensity);

    for i in 0..lights.len() {
        let l = (lights[i].position - point).normalize();
        let v = (eye_pos - point).normalize();
        let n = normal.normalize();

        let mut r2 = (lights[i].position - point).norm().powi(2);
        if r2 < 1e-6 { r2 = 1e-6 };

        let diff = n.dot(&l).max(0.0);
        let diffuse = kd.component_mul(&lights[i].intensity) * diff / r2;

        let h = (l + v).normalize();
        let spec_angle = n.dot(&h).max(0.0);
        let spec = spec_angle.powf(p);
        let specular = ks.component_mul(&lights[i].intensity) * spec / r2;

        result_color += diffuse + specular;
    }

    result_color
}

pub fn displacement_fragment_shader() {

}

pub fn bump_fragment_shader() {

}

pub fn wireframe_fragment_shader(payload: &FragmentShaderPayload) -> Vector3<f32> {
    let bc = payload.barycentric;
    let heights = payload.screen_heights;

    // 设定线宽（像素单位）
    let line_width_pixels = 1.5; 

    // 计算当前像素到三条边的实际垂直距离（像素单位）
    // 距离 = 重心坐标值 * 对应的总高度
    let dist_to_edge_0 = bc.x * heights.x; // 到边 1-2 的距离
    let dist_to_edge_1 = bc.y * heights.y; // 到边 0-2 的距离
    let dist_to_edge_2 = bc.z * heights.z; // 到边 0-1 的距离

    // 找到最近的边
    let min_dist = dist_to_edge_0.min(dist_to_edge_1).min(dist_to_edge_2);

    if min_dist < line_width_pixels {
        return Vector3::<f32>::new(1.0, 1.0, 1.0); // 白色线
    } else {
        return Vector3::<f32>::new(-1.0, -1.0, -1.0); // 丢弃
    }
}