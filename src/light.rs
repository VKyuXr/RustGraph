use nalgebra::{Matrix4, Vector3, Vector4};

#[derive(Clone)]
pub struct ShadowCascade {
    pub near: f32,
    pub far: f32,
    pub resolution: u32,
    pub depth_buffer: Vec<f32>,
    pub light_view_proj: Matrix4<f32>,
}

pub struct DirectionalLight {
    pub direction: Vector3<f32>,
    pub color: Vector3<f32>,
    pub intensity: f32,
    pub cascades: Vec<ShadowCascade>,
}

impl DirectionalLight {
    pub fn new(
        direction: Vector3<f32>,
        color: Vector3<f32>,
        intensity: f32,
    ) -> Self {
        let cascade_specs = [
            (0.1f32, 10.0f32, 4096u32),
            (10.0, 30.0, 2048),
            (30.0, 100.0, 1024),
        ];

        let cascades = cascade_specs
            .iter()
            .map(|&(near, far, resolution)| ShadowCascade {
                near,
                far,
                resolution,
                depth_buffer: vec![f32::INFINITY; (resolution * resolution) as usize],
                light_view_proj: Matrix4::identity(),
            })
            .collect();

        Self {
            direction: direction.normalize(),
            color,
            intensity,
            cascades,
        }
    }
}

pub fn compute_cascade_matrices(
    cascades: &mut [ShadowCascade],
    light_dir: Vector3<f32>,
    cam_view: &Matrix4<f32>,
    fov: f32,
    aspect: f32,
) {
    let inv_view = cam_view.try_inverse().expect("Camera view matrix is singular");

    for cascade in cascades.iter_mut() {
        let corners = frustum_corners_world(cascade.near, cascade.far, fov, aspect, &inv_view);

        let light_view = light_view_matrix(light_dir, &corners);
        let light_view_proj = light_ortho_proj(&light_view, &corners);

        cascade.light_view_proj = light_view_proj;
    }
}

fn frustum_corners_world(
    near: f32,
    far: f32,
    fov: f32,
    aspect: f32,
    inv_view: &Matrix4<f32>,
) -> [Vector3<f32>; 8] {
    let tan_half = (fov * 0.5).to_radians().tan();

    let near_h = tan_half * near;
    let near_w = near_h * aspect;
    let far_h = tan_half * far;
    let far_w = far_h * aspect;

    let corners_view = [
        Vector3::new(-near_w, near_h, -near),
        Vector3::new(near_w, near_h, -near),
        Vector3::new(near_w, -near_h, -near),
        Vector3::new(-near_w, -near_h, -near),
        Vector3::new(-far_w, far_h, -far),
        Vector3::new(far_w, far_h, -far),
        Vector3::new(far_w, -far_h, -far),
        Vector3::new(-far_w, -far_h, -far),
    ];

    corners_view.map(|c| {
        let w = inv_view * Vector4::new(c.x, c.y, c.z, 1.0);
        Vector3::new(w.x, w.y, w.z)
    })
}

fn light_view_matrix(light_dir: Vector3<f32>, corners: &[Vector3<f32>; 8]) -> Matrix4<f32> {
    let center = corners.iter().fold(Vector3::zeros(), |a, b| a + b) / 8.0;

    let eye = center - light_dir * 50.0;
    let up = if light_dir.x.abs() < 0.9 && light_dir.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    };

    let z = -light_dir;
    let x = up.cross(&z).normalize();
    let y = z.cross(&x);

    let view = Matrix4::new(
        x.x, x.y, x.z, -x.dot(&eye),
        y.x, y.y, y.z, -y.dot(&eye),
        z.x, z.y, z.z, -z.dot(&eye),
        0.0, 0.0, 0.0, 1.0,
    );

    view
}

fn light_ortho_proj(light_view: &Matrix4<f32>, corners: &[Vector3<f32>; 8]) -> Matrix4<f32> {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for corner in corners {
        let ls = light_view * Vector4::new(corner.x, corner.y, corner.z, 1.0);
        min_x = min_x.min(ls.x);
        max_x = max_x.max(ls.x);
        min_y = min_y.min(ls.y);
        max_y = max_y.max(ls.y);
        min_z = min_z.min(ls.z);
        max_z = max_z.max(ls.z);
    }

    let margin = (max_z - min_z) * 0.1;
    let l = min_x - 1.0;
    let r = max_x + 1.0;
    let b = min_y - 1.0;
    let t = max_y + 1.0;
    let n = max_z + margin; // 近平面 z（离光源更近，z 值更大）
    let f = min_z - margin; // 远平面 z（离光源更远，z 值更小）

    // 手动构建正交投影矩阵，映射 [l,r]×[b,t]×[n,f] → NDC[-1,1]³
    // n > f（右手坐标系，摄像机看向 -z 方向）
    let mut proj = Matrix4::identity();
    proj[(0, 0)] = 2.0 / (r - l);
    proj[(1, 1)] = 2.0 / (t - b);
    proj[(2, 2)] = 2.0 / (f - n); // f-n < 0，正确
    proj[(0, 3)] = -(r + l) / (r - l);
    proj[(1, 3)] = -(t + b) / (t - b);
    proj[(2, 3)] = -(n + f) / (f - n);

    proj * light_view
}