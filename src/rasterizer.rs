use bitflags::bitflags;
use nalgebra::{ Matrix4, Perspective3, Vector2, Vector3, Vector4, max, min };
use std::{ collections::BTreeMap, f32 };

use crate::triangle;
use crate::config;
use crate::shader;
use crate::texture;

bitflags! {
    #[derive(PartialEq)]
    pub struct Buffers: u32 {
        const Color = 0b0001;
        const Depth = 0b0010;
    }
}

#[derive(PartialEq)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Copy, Clone)]
pub struct PosBufId {
    pos_id: u32,
}

#[derive(Copy, Clone)]
pub struct IndBufId {
    ind_id: u32,
}

#[derive(Copy, Clone)]
pub struct ColBufId {
    col_id: u32,
}

pub fn model_matrix(pos: Vector3<f32>, rot: Vector3<f32>) -> Matrix4<f32> {
    let r = rot.map(|angle| angle.to_radians());
    Matrix4::new_translation(&pos) * Matrix4::from_euler_angles(r.x, r.y, r.z)
}

pub fn view_matrix(eye_pos: Vector3<f32>, eye_rot: Vector3<f32>) -> Matrix4<f32> {
    let r = eye_rot.map(|angle| angle.to_radians());
    Matrix4::from_euler_angles(r.x, r.y, r.z).transpose() * Matrix4::new_translation(&-eye_pos)
}

pub fn projection_matrix(eye_fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Matrix4<f32> {
    Perspective3::new(aspect_ratio, eye_fov.to_radians(), z_near, z_far).to_homogeneous()
}

fn inside_triangle(x: f32, y: f32, v:&[Vector3<f32>; 3], culling_enabled: bool) -> bool {
    let p = Vector2::new(x + 0.5, y + 0.5);
    let v0 = Vector2::new(v[0].x, v[0].y);
    let v1 = Vector2::new(v[1].x, v[1].y);
    let v2 = Vector2::new(v[2].x, v[2].y);

    let edge_func = |a: Vector2<f32>, b: Vector2<f32>, p: Vector2<f32>| {
        (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    };

    let e0 = edge_func(v0, v1, p);
    let e1 = edge_func(v1, v2, p);
    let e2 = edge_func(v2, v0, p);

    if culling_enabled {
        e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0
    } else {
        let all_pos = e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
        let all_neg = e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;
        all_pos || all_neg
    }
}

fn compute_barycentric_2d(x: f32, y: f32, v: &[Vector3<f32>; 3]) -> (f32, f32, f32) {
    let v0 = &v[0];
    let v1 = &v[1];
    let v2 = &v[2];
    let x0 = v0.x;
    let y0 = v0.y;
    let x1 = v1.x;
    let y1 = v1.y;
    let x2 = v2.x;
    let y2 = v2.y;

    let denom = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);

    if denom == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let c1 = (x * (y1 - y2) + (x2 - x1) * y + x1 * y2 - x2 * y1) / denom;
    let c2 = (x * (y2 - y0) + (x0 - x2) * y + x2 * y0 - x0 * y2) / denom;
    let c3 = (x * (y0 - y1) + (x1 - x0) * y + x0 * y1 - x1 * y0) / denom;

    (c1, c2, c3)
}

type VertexShaderFunc = fn(&shader::VertexShaderPayload) -> Vector3<f32>;

type FragmentShaderFunc = fn(&shader::FragmentShaderPayload) -> Vector3<f32>;

pub struct Rasterizer {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,

    pos_buf: BTreeMap<u32, Vec<Vector3<f32>>>,
    ind_buf: BTreeMap<u32, Vec<Vector3<u32>>>,
    col_buf: BTreeMap<u32, Vec<Vector3<u8>>>,
    frame_buf: Vec<Vector3<u8>>,
    depth_buf: Vec<f32>,

    width: u16,
    height: u16,

    next_id: u32,
    config: config::RasterizerConfig,

    pub vertex_shader: Option<VertexShaderFunc>,
    pub fragment_shader: Option<FragmentShaderFunc>,
    texture: Option<texture::Texture>,
}

impl Rasterizer {
    pub fn new(w: u16, h: u16, configure: config::RasterizerConfig) -> Self {
        return Self {
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            projection: Matrix4::identity(),

            pos_buf: BTreeMap::new(),
            ind_buf: BTreeMap::new(),
            col_buf: BTreeMap::new(),
            frame_buf: vec![Vector3::zeros(); w as usize * h as usize],
            depth_buf: vec![f32::INFINITY; w as usize * h as usize],
            
            width: w,
            height: h,

            next_id: 0,
            config: configure,
            vertex_shader: None,
            fragment_shader: None,
            texture: None,
        }
    }

    pub fn load_positions(&mut self, positions: Vec<Vector3<f32>>) -> PosBufId {
        let id = PosBufId {
            pos_id: self.get_next_id()
        };
        self.pos_buf.insert(id.pos_id, positions);

        return id;
    }

    pub fn load_indices(&mut self, indices: Vec<Vector3<u32>>) -> IndBufId {
        let id = IndBufId {
            ind_id: self.get_next_id()
        };
        self.ind_buf.insert(id.ind_id, indices);

        return id;
    }

    pub fn load_colors(&mut self, colors: Vec<Vector3<u8>>) -> ColBufId {
        let id = ColBufId {
            col_id: self.get_next_id()
        };
        self.col_buf.insert(id.col_id, colors);

        return id;
    }

    pub fn set_model(&mut self, m: Matrix4<f32>) {
        self.model = m;
    }

    pub fn set_view(&mut self, v: Matrix4<f32>) {
        self.view = v;
    }

    pub fn set_projection(&mut self, p: Matrix4<f32>) {
        self.projection = p;
    }

    pub fn set_pixel(&mut self, point: Vector2<u16>, color: Vector3<u8>) {
        if point.x >= self.width || point.y >= self.height {
            return;
        }
        self.frame_buf[(self.height - point.y - 1) as usize * self.width as usize + point.x as usize] = color;
    }

    pub fn clear(&mut self, buff: Buffers) {
        if buff.contains(Buffers::Color) {
            self.frame_buf.fill(Vector3::zeros());
        }
        if buff.contains(Buffers::Depth) {
            self.depth_buf.fill(f32::MAX);
        }
    }

    // pub fn draw(&mut self, triangle_list: &[triangle::Triangle]) {
    //     let mvp = self.projection * self.view * self.model; 
    //     let model_view = self.view * self.model;
        
    //     let inv_trans = model_view.try_inverse().expect("Model-View matrix is singular").transpose();

    //     let f1: f32 = (100.0 - 0.1) / 2.0;
    //     let f2: f32 = (100.0 + 0.1) / 2.0;

    //     let width_f = self.width as f32;
    //     let height_f = self.height as f32;

    //     for t in triangle_list {
    //         // 变换法线
    //         let normals: [Vector3<f32>; 3] = t.normal.map(|n| {
    //             let transformed = inv_trans * Vector4::new(n.x, n.y, n.z, 0.0);
    //             Vector3::new(transformed.x, transformed.y, transformed.z).normalize()
    //         });

    //         // 计算观察空间坐标
    //         let viewspace_pos: [Vector3<f32>; 3] = t.v.map(|vertex| {
    //             let v_homo = model_view * Vector4::new(vertex.x, vertex.y, vertex.z, 1.0);
    //             Vector3::new(v_homo.x, v_homo.y, v_homo.z)
    //         });

    //         // 变换顶点到裁剪空间
    //         let clip_space: [Vector4<f32>; 3] = t.v.map(|v| {
    //             mvp * Vector4::new(v.x, v.y, v.z, 1.0)
    //         });

    //         // 视口变换及深度计算
    //         let mut screen_vertices: [Vector3<f32>; 3] = [Vector3::zeros(); 3];
    //         let mut inv_w: [f32; 3] = [0.0; 3]; // 存储 1/w 用于可能的透视校正

    //         for i in 0..3 {
    //             let v_clip = clip_space[i];
                
    //             let w = if v_clip.w.abs() > 1e-6 { v_clip.w } else { 1.0 };
    //             let w_inv = 1.0 / w;
    //             inv_w[i] = w_inv;

    //             let x_ndc = v_clip.x * w_inv;
    //             let y_ndc = v_clip.y * w_inv;
    //             let z_ndc = v_clip.z * w_inv;

    //             // 视口变换
    //             let x_screen = 0.5 * width_f * (x_ndc + 1.0);
    //             let y_screen = 0.5 * height_f * (y_ndc + 1.0);
                
    //             // 深度映射
    //             let z_depth = z_ndc * f1 + f2;

    //             screen_vertices[i] = Vector3::new(x_screen, y_screen, z_depth);
    //         }

    //         // 构建三角形对象
    //         let mut new_triangle = triangle::Triangle::new();
    //         for i in 0..3 {
    //             new_triangle.set_vertex(i, screen_vertices[i as usize]);
    //             new_triangle.set_normal(i, normals[i as usize]);
    //         }

    //         // 调用光栅化
    //         self.rasterize_triangle(&new_triangle, &viewspace_pos);
    //     }
    // }

    pub fn draw(&mut self, triangle_list: &[triangle::Triangle]) {
        let scale = self.config.ssaa_scale;
        let original_width = self.width;
        let original_height = self.height;

        if scale == 1 {
            let mut frame_buf_temp = vec![Vector3::<u8>::zeros(); original_width as usize * original_height as usize];
            let mut depth_buf_temp = vec![f32::INFINITY; original_width as usize * original_height as usize];

            self.draw_internal(triangle_list, original_width, original_height, &mut frame_buf_temp, &mut depth_buf_temp);
            
            self.frame_buf = frame_buf_temp;
            self.depth_buf = depth_buf_temp;
            return;
        }

        let ss_width = (original_width as u32 * scale as u32) as u16;
        let ss_height = (original_height as u32 * scale as u32) as u16;
        
        let total_pixels = (ss_width as usize) * (ss_height as usize);

        let mut depth_buf_ss = vec![f32::INFINITY; total_pixels];
        let mut frame_buf_ss = vec![Vector3::<u8>::new(0, 0, 0); total_pixels];

        self.draw_internal(triangle_list, ss_width, ss_height, &mut frame_buf_ss, &mut depth_buf_ss);

        for y in 0..original_height {
            for x in 0..original_width {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;

                for sy in 0..scale {
                    for sx in 0..scale {
                        let hx = (x as u32 * scale as u32 + sx as u32) as u16;
                        let hy = (y as u32 * scale as u32 + sy as u32) as u16;
                        
                        let idx = (hy as usize * ss_width as usize) + (hx as usize);
                        
                        let color = frame_buf_ss[idx];
                        r_sum += color[0] as u32;
                        g_sum += color[1] as u32;
                        b_sum += color[2] as u32;
                    }
                }

                let pixel_count = (scale * scale) as u32;
                let final_idx = (y as usize * original_width as usize) + (x as usize);
                
                self.frame_buf[final_idx] = Vector3::new(
                    (r_sum / pixel_count) as u8,
                    (g_sum / pixel_count) as u8,
                    (b_sum / pixel_count) as u8,
                );
            }
        }
    }

    fn draw_internal(
        &mut self, 
        triangle_list: &[triangle::Triangle], 
        render_width: u16, 
        render_height: u16,
        frame_buf: &mut Vec<Vector3<u8>>, 
        depth_buf: &mut Vec<f32>
    ) {
        let width_f = render_width as f32;
        let height_f = render_height as f32;

        let mvp = self.projection * self.view * self.model; 
        let model_view = self.view * self.model;
        let inv_trans = model_view.try_inverse().expect("Model-View matrix is singular").transpose();

        let f1: f32 = (100.0 - 0.1) / 2.0;
        let f2: f32 = (100.0 + 0.1) / 2.0;

        let get_idx = |x: u16, y: u16| -> usize {
            (y as usize * render_width as usize) + (x as usize)
        };

        for t in triangle_list {
            let normals: [Vector3<f32>; 3] = t.normal.map(|n| {
                let transformed = inv_trans * Vector4::new(n.x, n.y, n.z, 0.0);
                Vector3::new(transformed.x, transformed.y, transformed.z).normalize()
            });

            let viewspace_pos: [Vector3<f32>; 3] = t.v.map(|vertex| {
                let v_homo = model_view * Vector4::new(vertex.x, vertex.y, vertex.z, 1.0);
                Vector3::new(v_homo.x, v_homo.y, v_homo.z)
            });

            let clip_space: [Vector4<f32>; 3] = t.v.map(|v| {
                mvp * Vector4::new(v.x, v.y, v.z, 1.0)
            });

            let mut screen_vertices: [Vector3<f32>; 3] = [Vector3::zeros(); 3];
            
            for i in 0..3 {
                let v_clip = clip_space[i];
                let w = if v_clip.w.abs() > 1e-6 { v_clip.w } else { 1.0 };
                let w_inv = 1.0 / w;

                let x_ndc = v_clip.x * w_inv;
                let y_ndc = v_clip.y * w_inv;
                let z_ndc = v_clip.z * w_inv;

                let x_screen = 0.5 * width_f * (x_ndc + 1.0);
                let y_screen = 0.5 * height_f * (1.0 - y_ndc);
                
                let z_depth = z_ndc * f1 + f2;

                screen_vertices[i] = Vector3::new(x_screen, y_screen, z_depth);
            }

            let mut new_triangle = triangle::Triangle::new();
            for i in 0..3 {
                new_triangle.set_vertex(i, screen_vertices[i as usize]);
                new_triangle.set_normal(i, normals[i as usize]);
            }

            self.rasterize_triangle_with_buffers(
                &new_triangle, 
                &viewspace_pos, 
                render_width, 
                render_height,
                frame_buf, 
                depth_buf,
                &get_idx
            );
        }
    }

    fn rasterize_triangle_with_buffers(
        &mut self,
        t: &triangle::Triangle,
        view_pos: &[Vector3<f32>; 3],
        width: u16,
        height: u16,
        frame_buf: &mut Vec<Vector3<u8>>,
        depth_buf: &mut Vec<f32>,
        get_idx: &dyn Fn(u16, u16) -> usize
    ) {
        let vertices = t.get_vertex(); 

        let x_min_f = vertices.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let x_max_f = vertices.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        let y_min_f = vertices.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max_f = vertices.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);

        let mut x_min = x_min_f.floor() as u16;
        let mut x_max = x_max_f.ceil() as u16;
        let mut y_min = y_min_f.floor() as u16;
        let mut y_max = y_max_f.ceil() as u16;

        x_min = x_min.max(0);
        x_max = x_max.min(width.saturating_sub(1));
        y_min = y_min.max(0);
        y_max = y_max.min(height.saturating_sub(1));

        let v_clip = t.to_vector4(); 
        let w0 = v_clip[0].w;
        let w1 = v_clip[1].w;
        let w2 = v_clip[2].w;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let x_center = x as f32 + 0.5;
                let y_center = y as f32 + 0.5;

                if !inside_triangle(x_center, y_center, &vertices, self.config.culling_enabled) {
                    continue;
                }

                let (alpha, beta, gamma) = compute_barycentric_2d(x_center, y_center, &vertices);

                let w_inv_interp = alpha / w0 + beta / w1 + gamma / w2;
                
                if w_inv_interp.abs() < 1e-8 {
                    continue;
                }
                
                let w_corr = 1.0 / w_inv_interp;

                let z_interpolated = (alpha * v_clip[0].z / w0 + beta * v_clip[1].z / w1 + gamma * v_clip[2].z / w2) * w_corr;

                let idx = get_idx(x, y);
                if depth_buf[idx] <= z_interpolated {
                    continue;
                }
                // depth_buf[idx] = z_interpolated;

                let perspective_interp = |a: f32, b: f32, g: f32, v0: f32, v1: f32, v2: f32, w0: f32, w1: f32, w2: f32, w_c: f32| {
                    (a * v0 / w0 + b * v1 / w1 + g * v2 / w2) * w_c
                };

                // 插值颜色
                let r = perspective_interp(alpha, beta, gamma, t.color[0][0], t.color[1][0], t.color[2][0], w0, w1, w2, w_corr);
                let g_val = perspective_interp(alpha, beta, gamma, t.color[0][1], t.color[1][1], t.color[2][1], w0, w1, w2, w_corr);
                let b_val = perspective_interp(alpha, beta, gamma, t.color[0][2], t.color[1][2], t.color[2][2], w0, w1, w2, w_corr);
                let interpolated_color = Vector3::<f32>::new(r.clamp(0.0, 1.0), g_val.clamp(0.0, 1.0), b_val.clamp(0.0, 1.0));

                // 插值法线
                let nx = perspective_interp(alpha, beta, gamma, t.normal[0][0], t.normal[1][0], t.normal[2][0], w0, w1, w2, w_corr);
                let ny = perspective_interp(alpha, beta, gamma, t.normal[0][1], t.normal[1][1], t.normal[2][1], w0, w1, w2, w_corr);
                let nz = perspective_interp(alpha, beta, gamma, t.normal[0][2], t.normal[1][2], t.normal[2][2], w0, w1, w2, w_corr);
                let interpolated_normal = Vector3::<f32>::new(nx, ny, nz).normalize();

                // 插值纹理坐标
                let u = perspective_interp(alpha, beta, gamma, t.tex_coords[0][0], t.tex_coords[1][0], t.tex_coords[2][0], w0, w1, w2, w_corr);
                let v_tex = perspective_interp(alpha, beta, gamma, t.tex_coords[0][1], t.tex_coords[1][1], t.tex_coords[2][1], w0, w1, w2, w_corr);
                let interpolated_texcoords = Vector2::<f32>::new(u, v_tex);

                // 插值观察空间位置
                let px = perspective_interp(alpha, beta, gamma, view_pos[0][0], view_pos[1][0], view_pos[2][0], w0, w1, w2, w_corr);
                let py = perspective_interp(alpha, beta, gamma, view_pos[0][1], view_pos[1][1], view_pos[2][1], w0, w1, w2, w_corr);
                let pz = perspective_interp(alpha, beta, gamma, view_pos[0][2], view_pos[1][2], view_pos[2][2], w0, w1, w2, w_corr);
                let interpolated_shadingcoords = Vector3::<f32>::new(px, py, pz);

                // 1. 获取屏幕空间的三个顶点 (t.get_vertex() 返回的应该是 screen_vertices)
                let v0 = vertices[0];
                let v1 = vertices[1];
                let v2 = vertices[2];

                // 2. 辅助函数：计算点 P 到线段 AB 的垂直距离 (2D 平面)
                // 公式：Area = 0.5 * base * height  =>  height = 2 * Area / base
                // 向量叉积的 Z 分量等于平行四边形面积
                fn point_line_distance(p: &Vector3<f32>, a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
                    let ab = b - a;
                    let ap = p - a;
                    
                    // 2D 叉积 (只取 Z 分量，因为我们在屏幕 XY 平面)
                    // cross_z = ab.x * ap.y - ab.y * ap.x
                    let cross_z = ab.x * ap.y - ab.y * ap.x;
                    let area_x2 = cross_z.abs(); // 2 * 三角形面积
                    
                    let base_len = (ab.x * ab.x + ab.y * ab.y).sqrt();
                    
                    if base_len < 1e-6 {
                        return 0.0;
                    }
                    
                    // height = (2 * Area) / base
                    return area_x2 / base_len;
                }

                // 3. 计算三个高度
                // h0: 顶点 v0 到边 (v1-v2) 的距离
                let h0 = point_line_distance(&v0, &v1, &v2);
                // h1: 顶点 v1 到边 (v0-v2) 的距离
                let h1 = point_line_distance(&v1, &v0, &v2);
                // h2: 顶点 v2 到边 (v0-v1) 的距离
                let h2 = point_line_distance(&v2, &v0, &v1);

                let screen_heights = Vector3::new(h0, h1, h2);

                // 4. 构建 Payload 时传入
                let payload = shader::FragmentShaderPayload::new(
                    Vector3::<f32>::new(alpha, beta, gamma),
                    interpolated_shadingcoords,
                    interpolated_color,
                    interpolated_normal,
                    interpolated_texcoords,
                    self.texture.clone(),
                    screen_heights, // 传入计算好的高度
                );

                let pixel_color: Vector3<f32> = if let Some(func) = self.fragment_shader {
                    func(&payload)
                } else {
                    Vector3::new(1.0, 1.0, 1.0)
                };

                if pixel_color == Vector3::<f32>::new(-1.0, -1.0, -1.0) {
                    continue;
                }
                
                depth_buf[idx] = z_interpolated;

                let final_color = Vector3::<u8>::new(
                    (pixel_color[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[2] * 255.0).clamp(0.0, 255.0) as u8,
                );

                // 写入传入的 frame_buf
                frame_buf[idx] = final_color;
            }
        }
    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<u8>> {
        return &self.frame_buf;
    }

    fn rasterize_triangle(&mut self, t: &triangle::Triangle, view_pos: &[Vector3<f32>; 3]) {
        let vertices = t.get_vertex(); 

        // 计算包围盒
        let x_min_f = vertices.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let x_max_f = vertices.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        let y_min_f = vertices.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max_f = vertices.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);

        let mut x_min = x_min_f.floor() as u16;
        let mut x_max = x_max_f.ceil() as u16;
        let mut y_min = y_min_f.floor() as u16;
        let mut y_max = y_max_f.ceil() as u16;

        // 裁剪到屏幕范围内
        x_min = x_min.max(0);
        x_max = x_max.min(self.width.saturating_sub(1));
        y_min = y_min.max(0);
        y_max = y_max.min(self.height.saturating_sub(1));

        let v_clip = t.to_vector4(); 
        let w0 = v_clip[0].w;
        let w1 = v_clip[1].w;
        let w2 = v_clip[2].w;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let x_center = x as f32 + 0.5;
                let y_center = y as f32 + 0.5;

                if !inside_triangle(x_center, y_center, &vertices, self.config.culling_enabled) {
                    continue;
                }

                let (alpha, beta, gamma) = compute_barycentric_2d(x_center, y_center, &vertices);

                let w_inv_interp = alpha / w0 + beta / w1 + gamma / w2;
                
                // 防止除以零
                if w_inv_interp.abs() < 1e-8 {
                    continue;
                }
                
                let w_corr = 1.0 / w_inv_interp;

                // 深度插值
                let z_interpolated = (alpha * v_clip[0].z / w0 + beta * v_clip[1].z / w1 + gamma * v_clip[2].z / w2) * w_corr;

                // 深度测试
                let idx = self.get_index(x, y);
                if self.depth_buf[idx] <= z_interpolated {
                    continue;
                }
                self.depth_buf[idx] = z_interpolated;

                // 属性插值
                let perspective_interp = |a: f32, b: f32, g: f32, v0: f32, v1: f32, v2: f32, w0: f32, w1: f32, w2: f32, w_c: f32| {
                    (a * v0 / w0 + b * v1 / w1 + g * v2 / w2) * w_c
                };

                // 插值颜色
                let r = perspective_interp(alpha, beta, gamma, t.color[0][0], t.color[1][0], t.color[2][0], w0, w1, w2, w_corr);
                let g_val = perspective_interp(alpha, beta, gamma, t.color[0][1], t.color[1][1], t.color[2][1], w0, w1, w2, w_corr);
                let b_val = perspective_interp(alpha, beta, gamma, t.color[0][2], t.color[1][2], t.color[2][2], w0, w1, w2, w_corr);
                let interpolated_color = Vector3::<f32>::new(r.clamp(0.0, 1.0), g_val.clamp(0.0, 1.0), b_val.clamp(0.0, 1.0));

                // 插值法线
                let nx = perspective_interp(alpha, beta, gamma, t.normal[0][0], t.normal[1][0], t.normal[2][0], w0, w1, w2, w_corr);
                let ny = perspective_interp(alpha, beta, gamma, t.normal[0][1], t.normal[1][1], t.normal[2][1], w0, w1, w2, w_corr);
                let nz = perspective_interp(alpha, beta, gamma, t.normal[0][2], t.normal[1][2], t.normal[2][2], w0, w1, w2, w_corr);
                let interpolated_normal = Vector3::<f32>::new(nx, ny, nz).normalize();

                // 插值纹理坐标
                let u = perspective_interp(alpha, beta, gamma, t.tex_coords[0][0], t.tex_coords[1][0], t.tex_coords[2][0], w0, w1, w2, w_corr);
                let v_tex = perspective_interp(alpha, beta, gamma, t.tex_coords[0][1], t.tex_coords[1][1], t.tex_coords[2][1], w0, w1, w2, w_corr);
                let interpolated_texcoords = Vector2::<f32>::new(u, v_tex);

                // 插值观察空间位置
                let px = perspective_interp(alpha, beta, gamma, view_pos[0][0], view_pos[1][0], view_pos[2][0], w0, w1, w2, w_corr);
                let py = perspective_interp(alpha, beta, gamma, view_pos[0][1], view_pos[1][1], view_pos[2][1], w0, w1, w2, w_corr);
                let pz = perspective_interp(alpha, beta, gamma, view_pos[0][2], view_pos[1][2], view_pos[2][2], w0, w1, w2, w_corr);
                let interpolated_shadingcoords = Vector3::<f32>::new(px, py, pz);

                // 1. 获取屏幕空间的三个顶点 (t.get_vertex() 返回的应该是 screen_vertices)
                let v0 = vertices[0];
                let v1 = vertices[1];
                let v2 = vertices[2];

                // 2. 辅助函数：计算点 P 到线段 AB 的垂直距离 (2D 平面)
                // 公式：Area = 0.5 * base * height  =>  height = 2 * Area / base
                // 向量叉积的 Z 分量等于平行四边形面积
                fn point_line_distance(p: &Vector3<f32>, a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
                    let ab = b - a;
                    let ap = p - a;
                    
                    // 2D 叉积 (只取 Z 分量，因为我们在屏幕 XY 平面)
                    // cross_z = ab.x * ap.y - ab.y * ap.x
                    let cross_z = ab.x * ap.y - ab.y * ap.x;
                    let area_x2 = cross_z.abs(); // 2 * 三角形面积
                    
                    let base_len = (ab.x * ab.x + ab.y * ab.y).sqrt();
                    
                    if base_len < 1e-6 {
                        return 0.0;
                    }
                    
                    // height = (2 * Area) / base
                    return area_x2 / base_len;
                }

                // 3. 计算三个高度
                // h0: 顶点 v0 到边 (v1-v2) 的距离
                let h0 = point_line_distance(&v0, &v1, &v2);
                // h1: 顶点 v1 到边 (v0-v2) 的距离
                let h1 = point_line_distance(&v1, &v0, &v2);
                // h2: 顶点 v2 到边 (v0-v1) 的距离
                let h2 = point_line_distance(&v2, &v0, &v1);

                let screen_heights = Vector3::new(h0, h1, h2);

                // 4. 构建 Payload 时传入
                let payload = shader::FragmentShaderPayload::new(
                    Vector3::<f32>::new(alpha, beta, gamma),
                    interpolated_shadingcoords,
                    interpolated_color,
                    interpolated_normal,
                    interpolated_texcoords,
                    self.texture.clone(),
                    screen_heights, // 传入计算好的高度
                );

                let pixel_color: Vector3<f32> = if let Some(func) = self.fragment_shader {
                    func(&payload)
                } else {
                    Vector3::new(1.0, 1.0, 1.0)
                };

                if pixel_color == Vector3::<f32>::new(-1.0, -1.0, -1.0) {
                    continue;
                }

                // 写入像素
                let final_color = Vector3::<u8>::new(
                    (pixel_color[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[2] * 255.0).clamp(0.0, 255.0) as u8,
                );

                self.set_pixel(Vector2::new(x, y), final_color);
            }
        }
    }

    fn get_index(&self, x: u16, y: u16) -> usize {
        let row = (self.height - 1 - y) as usize; 
        
        let width = self.width as usize;
        let col = x as usize;

        row * width + col
    }

    fn get_next_id(&mut self) -> u32 {
        self.next_id += 1;
        return self.next_id;
    }

    fn to_vec4(v3: Vector3<f32>, w: f32) -> Vector4<f32> {
        return Vector4::<f32>::new(v3.x, v3.y, v3.z, w);
    }

    pub fn set_vertex_shader(&mut self, vertex_shader: VertexShaderFunc) {
        self.vertex_shader = Some(vertex_shader);
    }

    pub fn set_fragment_shader(&mut self, fragment_shader: FragmentShaderFunc) {
        self.fragment_shader = Some(fragment_shader);
    }
}

// impl Default for Rasterizer {
//     fn default() -> Self {
//         return Self::new(1240, 800);
//     }
// }
