use bitflags::bitflags;
use nalgebra::{ Matrix4, Perspective3, Vector2, Vector3, Vector4 };
use std::{ collections::BTreeMap, f32 };

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::triangle;
use crate::config;
use crate::shader;
use crate::light;
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

type FragmentShaderFunc = fn(&shader::FragmentShaderPayload, &shader::ShadowContext) -> Vector4<f32>;

#[derive(Clone)]
struct ScreenTriangle {
    screen_vertices: [Vector3<f32>; 3],
    viewspace_pos: [Vector3<f32>; 3],
    normals: [Vector3<f32>; 3],
    clip_w: [f32; 3],
    tex_coords: [Vector2<f32>; 3],
    color_data: [Vector3<f32>; 3],
    texture_id: Option<usize>,
    bbox_min: Vector2<f32>,
    bbox_max: Vector2<f32>,
}

pub struct Rasterizer {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,

    pos_buf: BTreeMap<u32, Vec<Vector3<f32>>>,
    ind_buf: BTreeMap<u32, Vec<Vector3<u32>>>,
    col_buf: BTreeMap<u32, Vec<Vector3<u8>>>,
    frame_buf: Vec<Vector4<u8>>,
    depth_buf: Vec<f32>,

    width: u16,
    height: u16,

    next_id: u32,
    config: config::RasterizerConfig,

    pub vertex_shader: Option<VertexShaderFunc>,
    pub fragment_shader: Option<FragmentShaderFunc>,
    textures: Vec<Arc<texture::Texture>>,
    shadow_cascades: Option<Arc<Vec<light::ShadowCascade>>>,
    debug_dir: Option<String>,
    light_color: Vector3<f32>,
    light_intensity: f32,
}

impl Rasterizer {
    pub fn new(w: u16, h: u16, configure: config::RasterizerConfig) -> Self {
        let bg = configure.background_color;
        let bg_color = Vector4::new(bg[0], bg[1], bg[2], bg[3]);
        return Self {
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            projection: Matrix4::identity(),

            pos_buf: BTreeMap::new(),
            ind_buf: BTreeMap::new(),
            col_buf: BTreeMap::new(),
            frame_buf: vec![bg_color; w as usize * h as usize],
            depth_buf: vec![f32::INFINITY; w as usize * h as usize],
            
            width: w,
            height: h,

            next_id: 0,
            config: configure,
            vertex_shader: None,
            fragment_shader: None,
            textures: Vec::new(),
            shadow_cascades: None,
            debug_dir: None,
            light_color: Vector3::new(1.0, 1.0, 1.0),
            light_intensity: 3.0,
        }
    }

    pub fn set_light(&mut self, color: Vector3<f32>, intensity: f32) {
        self.light_color = color;
        self.light_intensity = intensity;
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

    pub fn set_pixel(&mut self, point: Vector2<u16>, color: Vector4<u8>) {
        if point.x >= self.width || point.y >= self.height {
            return;
        }
        self.frame_buf[(self.height - point.y - 1) as usize * self.width as usize + point.x as usize] = color;
    }

    pub fn clear(&mut self, buff: Buffers) {
        let bg = self.config.background_color;
        let bg_color = Vector4::new(bg[0], bg[1], bg[2], bg[3]);
        if buff.contains(Buffers::Color) {
            self.frame_buf.fill(bg_color);
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

    fn precompute_triangles(
        &self,
        triangle_list: &[triangle::Triangle],
        render_width: u16,
        render_height: u16,
        textures: &[Arc<texture::Texture>],
    ) -> (Vec<ScreenTriangle>, Vec<ScreenTriangle>) {
        let width_f = render_width as f32;
        let height_f = render_height as f32;

        let mvp = self.projection * self.view * self.model;
        let model_view = self.view * self.model;
        let inv_trans = model_view.try_inverse().expect("Model-View matrix is singular").transpose();

        let f1: f32 = (100.0 - 0.1) / 2.0;
        let f2: f32 = (100.0 + 0.1) / 2.0;

        let mut opaque = Vec::with_capacity(triangle_list.len());
        let mut transparent = Vec::with_capacity(triangle_list.len());

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

            if self.config.culling_enabled {
                let edge1 = screen_vertices[1] - screen_vertices[0];
                let edge2 = screen_vertices[2] - screen_vertices[0];
                if edge1.x * edge2.y - edge1.y * edge2.x >= 0.0 {
                    continue;
                }
            }

            let color_data: [Vector3<f32>; 3] = [
                Vector3::new(t.color[0][0], t.color[0][1], t.color[0][2]),
                Vector3::new(t.color[1][0], t.color[1][1], t.color[1][2]),
                Vector3::new(t.color[2][0], t.color[2][1], t.color[2][2]),
            ];

            let clip_w: [f32; 3] = [
                clip_space[0].w,
                clip_space[1].w,
                clip_space[2].w,
            ];

            let bbox_min = Vector2::new(
                screen_vertices.iter().map(|v| v.x).fold(f32::INFINITY, f32::min),
                screen_vertices.iter().map(|v| v.y).fold(f32::INFINITY, f32::min),
            );
            let bbox_max = Vector2::new(
                screen_vertices.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max),
                screen_vertices.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max),
            );

            let is_transparent = t.texture_id
                .and_then(|id| textures.get(id))
                .map(|tex| tex.has_alpha)
                .unwrap_or(false);

            let tri = ScreenTriangle {
                screen_vertices,
                viewspace_pos,
                normals,
                clip_w,
                tex_coords: t.tex_coords,
                color_data,
                texture_id: t.texture_id,
                bbox_min,
                bbox_max,
            };

            if is_transparent {
                transparent.push(tri);
            } else {
                opaque.push(tri);
            }
        }

        (opaque, transparent)
    }

    pub fn set_shadow_cascades(&mut self, cascades: Vec<light::ShadowCascade>) {
        self.shadow_cascades = Some(Arc::new(cascades));
    }

    pub fn set_debug_dir(&mut self, dir: String) {
        std::fs::create_dir_all(&dir).expect("Failed to create debug directory");
        self.debug_dir = Some(dir);
    }

    fn render_shadow_maps(&mut self, triangle_list: &[triangle::Triangle]) {
        println!("render_shadow_maps: starting...");
        let cascades = match &mut self.shadow_cascades {
            Some(c) => {
                println!("render_shadow_maps: {} cascades found", c.len());
                Arc::make_mut(c)
            }
            None => {
                println!("render_shadow_maps: shadow_cascades is None, skipping!");
                return;
            }
        };

        let model = self.model;
        let tile_w = self.config.tile_width.max(1) as u32;
        let tile_h = self.config.tile_height.max(1) as u32;

        // 计算场景在相机空间中的最大深度，跳过不需要的级联
        let max_scene_depth = triangle_list
            .iter()
            .flat_map(|tri| tri.v.iter())
            .map(|v| {
                let world = model * Vector4::new(v.x, v.y, v.z, 1.0);
                let view = self.view * world;
                -view.z
            })
            .fold(0.0f32, f32::max);

        let active_cascades: Vec<usize> = cascades
            .iter()
            .enumerate()
            .filter(|(_, c)| c.near < max_scene_depth)
            .map(|(i, _)| i)
            .collect();

        println!(
            "Scene max depth = {:.1}, active cascades: {:?} ({} of {})",
            max_scene_depth,
            active_cascades,
            active_cascades.len(),
            cascades.len()
        );
        if active_cascades.is_empty() {
            println!("render_shadow_maps: no active cascades, skipping shadow rendering!");
            return;
        }

        let num_cascades = cascades.len();

        struct ShadowTri {
            screen: [Vector3<f32>; 3],
            clip_w: [f32; 3],
            bbox_min: Vector2<u32>,
            bbox_max: Vector2<u32>,
        }

        for &ci in &active_cascades {
            let cascade = &mut cascades[ci];
            let res = cascade.resolution;
            let res_f = res as f32;
            let lvp = cascade.light_view_proj;
            let depth = &mut cascade.depth_buffer;
            depth.fill(f32::INFINITY);

            println!(
                "  Shadow cascade {}/{}: {:.1}-{:.1} @ {}x{}",
                ci + 1,
                num_cascades,
                cascade.near,
                cascade.far,
                res,
                res
            );

            // 阶段 1：变换 + 裁剪
            let mut shadow_tris = Vec::new();
            let mut culled = 0u32;

            for tri in triangle_list {
                let clip: [Vector4<f32>; 3] = tri.v.map(|v| {
                    lvp * model * Vector4::new(v.x, v.y, v.z, 1.0)
                });

                if clip.iter().all(|v| v.x < -v.w.abs())
                    || clip.iter().all(|v| v.x > v.w.abs())
                    || clip.iter().all(|v| v.y < -v.w.abs())
                    || clip.iter().all(|v| v.y > v.w.abs())
                {
                    culled += 1;
                    continue;
                }

                let screen: [Vector3<f32>; 3] = clip.map(|v| {
                    let w = if v.w.abs() > 1e-6 { v.w } else { 1.0 };
                    let w_inv = 1.0 / w;
                    Vector3::new(
                        (v.x * w_inv * 0.5 + 0.5) * res_f,
                        (v.y * w_inv * 0.5 + 0.5) * res_f,
                        v.z * w_inv,
                    )
                });

                let x_min = (screen.iter().map(|v| v.x).fold(f32::INFINITY, f32::min).floor() as i32)
                    .max(0) as u32;
                let x_max = (screen.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max).ceil() as i32)
                    .min(res as i32 - 1) as u32;
                let y_min = (screen.iter().map(|v| v.y).fold(f32::INFINITY, f32::min).floor() as i32)
                    .max(0) as u32;
                let y_max = (screen.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max).ceil() as i32)
                    .min(res as i32 - 1) as u32;

                if x_min > x_max || y_min > y_max {
                    culled += 1;
                    continue;
                }

                shadow_tris.push(ShadowTri {
                    screen,
                    clip_w: [clip[0].w, clip[1].w, clip[2].w],
                    bbox_min: Vector2::new(x_min, y_min),
                    bbox_max: Vector2::new(x_max, y_max),
                });
            }

            println!(
                "    Transformed: {} rendered, {} culled out of {}",
                shadow_tris.len(),
                culled,
                triangle_list.len()
            );

            // 阶段 2：构建 tile → 三角形映射
            let tiles_x = (res + tile_w - 1) / tile_w;
            let tiles_y = (res + tile_h - 1) / tile_h;
            let total_tiles = (tiles_x * tiles_y) as usize;

            let mut tile_tri_lists: Vec<Vec<usize>> = vec![Vec::new(); total_tiles];
            for (tri_idx, tri) in shadow_tris.iter().enumerate() {
                let tx_min = tri.bbox_min.x / tile_w;
                let ty_min = tri.bbox_min.y / tile_h;
                let tx_max = (tri.bbox_max.x / tile_w).min(tiles_x - 1);
                let ty_max = (tri.bbox_max.y / tile_h).min(tiles_y - 1);

                for ty in ty_min..=ty_max {
                    for tx in tx_min..=tx_max {
                        tile_tri_lists[(ty * tiles_x + tx) as usize].push(tri_idx);
                    }
                }
            }

            let total_refs: usize = tile_tri_lists.iter().map(|v| v.len()).sum();
            println!(
                "    Tile grid: {}x{} ({} tiles), {} total tri refs (avg {:.1}/tile)",
                tiles_x,
                tiles_y,
                total_tiles,
                total_refs,
                total_refs as f32 / total_tiles.max(1) as f32
            );

            // 阶段 3：并行分块光栅化
            let tiles: Vec<(u32, u32)> = (0..tiles_y)
                .flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty)))
                .collect();

            let depth_ptr = depth.as_mut_ptr() as usize;
            let completed = AtomicUsize::new(0);
            let total_tiles_u = total_tiles;

            tiles.par_iter().for_each(|&(tx, ty)| {
                let depth_ptr = depth_ptr as *mut f32;
                let cell_idx = (ty * tiles_x + tx) as usize;
                let x_start = tx * tile_w;
                let y_start = ty * tile_h;
                let x_end = ((tx + 1) * tile_w).min(res);
                let y_end = ((ty + 1) * tile_h).min(res);

                for &tri_idx in &tile_tri_lists[cell_idx] {
                    let tri = &shadow_tris[tri_idx];

                    let px_min = x_start.max(tri.bbox_min.x);
                    let py_min = y_start.max(tri.bbox_min.y);
                    let px_max = x_end.min(tri.bbox_max.x);
                    let py_max = y_end.min(tri.bbox_max.y);

                    if px_min > px_max || py_min > py_max {
                        continue;
                    }

                    let w0 = tri.clip_w[0];
                    let w1 = tri.clip_w[1];
                    let w2 = tri.clip_w[2];

                    for y in py_min..=py_max {
                        for x in px_min..=px_max {
                            let (alpha, beta, gamma) =
                                compute_barycentric_2d(x as f32 + 0.5, y as f32 + 0.5, &tri.screen);

                            if alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                                continue;
                            }

                            let w_inv = alpha / w0 + beta / w1 + gamma / w2;
                            if w_inv.abs() < 1e-8 {
                                continue;
                            }
                            let w_corr = 1.0 / w_inv;

                            let z = (alpha * tri.screen[0].z / w0
                                + beta * tri.screen[1].z / w1
                                + gamma * tri.screen[2].z / w2)
                                * w_corr;

                            let idx = (y * res + x) as usize;
                            unsafe {
                                if z < *depth_ptr.add(idx) {
                                    *depth_ptr.add(idx) = z;
                                }
                            }
                        }
                    }
                }

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == total_tiles_u {
                    println!(
                        "    Shadow tiles: {}/{} ({:.0}%)",
                        done,
                        total_tiles_u,
                        done as f32 / total_tiles_u as f32 * 100.0
                    );
                }
            });

            println!("render_shadow_maps: saving cascade {} depth... debug_dir={:?}", ci, self.debug_dir);
            Self::save_depth_png(
                depth,
                res,
                res,
                &format!("{}/shadow_cascade_{}.png", self.debug_dir.as_ref().unwrap(), ci),
            );
        }
    }

    fn save_depth_png(depth: &[f32], width: u32, height: u32, path: &str) {
        let (min_d, max_d) = depth
            .iter()
            .filter(|&&d| d.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &d| (mn.min(d), mx.max(d)));

        if min_d >= max_d {
            println!("    (skip depth save: no valid depth data)");
            return;
        }

        let range = max_d - min_d;
        let mut img = image::GrayImage::new(width, height);
        for (i, &d) in depth.iter().enumerate() {
            let x = (i % width as usize) as u32;
            let y = (i / width as usize) as u32;
            let v = if d.is_finite() {
                ((d - min_d) / range * 255.0).clamp(0.0, 255.0) as u8
            } else {
                255u8
            };
            img.put_pixel(x, y, image::Luma([v]));
        }
        img.save(path).expect("Failed to save depth PNG");
        println!("    Saved depth image: {} (range [{:.3}, {:.3}])", path, min_d, max_d);
    }

    pub fn draw(&mut self, triangle_list: &[triangle::Triangle]) {
        let scale = self.config.ssaa_scale;
        let original_width = self.width;
        let original_height = self.height;

        let render_width = if scale == 1 {
            original_width
        } else {
            (original_width as u32 * scale as u32) as u16
        };
        let render_height = if scale == 1 {
            original_height
        } else {
            (original_height as u32 * scale as u32) as u16
        };

        let total_pixels = (render_width as usize) * (render_height as usize);
        let bg = self.config.background_color;
        let bg_color = Vector4::new(bg[0], bg[1], bg[2], bg[3]);
        let mut frame_buf = vec![bg_color; total_pixels];
        let mut depth_buf = vec![f32::INFINITY; total_pixels];

        let (opaque_tris, transparent_tris) =
            self.precompute_triangles(triangle_list, render_width, render_height, &self.textures);
        println!(
            "Precomputed {} opaque + {} transparent screen-space triangles.",
            opaque_tris.len(),
            transparent_tris.len()
        );

        // 渲染阴影贴图
        self.render_shadow_maps(triangle_list);

        let shadow_ctx = self.shadow_cascades.as_ref().map(|cascades| {
            shader::ShadowContext {
                cascades: Arc::clone(cascades),
                light_dir: Vector3::new(-1.0, -1.0, -1.0).normalize(),
                light_color: self.light_color,
                light_intensity: self.light_intensity,
            }
        }).unwrap_or_else(|| shader::ShadowContext {
            cascades: Arc::new(Vec::new()),
            light_dir: Vector3::new(-1.0, -1.0, -1.0).normalize(),
            light_color: self.light_color,
            light_intensity: self.light_intensity,
        });

        let inv_view = self.view.try_inverse().expect("View matrix is singular");

        let tile_w = self.config.tile_width.max(1) as u16;
        let tile_h = self.config.tile_height.max(1) as u16;

        let tiles_x = (render_width + tile_w - 1) / tile_w;
        let tiles_y = (render_height + tile_h - 1) / tile_h;
        let total_tiles = (tiles_x as usize) * (tiles_y as usize);

        println!(
            "Tile grid: {}x{} = {} tiles ({}x{} px), render: {}x{}",
            tiles_x, tiles_y, total_tiles, tile_w, tile_h, render_width, render_height
        );

        let shader = self.fragment_shader;
        let textures = &self.textures;

        let stride = render_width as usize;
        let frame_base = frame_buf.as_mut_ptr() as usize;
        let depth_base = depth_buf.as_mut_ptr() as usize;

        let completed = AtomicUsize::new(0);

        // 构建均匀网格：将每个不透明三角形分配到其包围盒覆盖的所有 tile 中
        let mut tile_tri_lists: Vec<Vec<usize>> = vec![Vec::new(); total_tiles];
        for (tri_idx, tri) in opaque_tris.iter().enumerate() {
            let cell_x_min = (tri.bbox_min.x / tile_w as f32).floor() as i32;
            let cell_x_max = (tri.bbox_max.x / tile_w as f32).floor() as i32;
            let cell_y_min = (tri.bbox_min.y / tile_h as f32).floor() as i32;
            let cell_y_max = (tri.bbox_max.y / tile_h as f32).floor() as i32;

            let cx_min = cell_x_min.max(0) as usize;
            let cx_max = (cell_x_max as usize).min(tiles_x as usize - 1);
            let cy_min = cell_y_min.max(0) as usize;
            let cy_max = (cell_y_max as usize).min(tiles_y as usize - 1);

            for cy in cy_min..=cy_max {
                for cx in cx_min..=cx_max {
                    let cell_idx = cy * tiles_x as usize + cx;
                    tile_tri_lists[cell_idx].push(tri_idx);
                }
            }
        }

        let mut total_tri_refs = 0usize;
        for list in &tile_tri_lists {
            total_tri_refs += list.len();
        }
        println!(
            "Tile-triangle mapping: {} total references (avg {:.1} tris/tile)",
            total_tri_refs,
            total_tri_refs as f32 / total_tiles as f32
        );

        let tiles: Vec<(u16, u16, u16, u16, usize)> = (0..tiles_y)
            .flat_map(|ty| {
                let y_start = ty * tile_h;
                let y_end = (y_start + tile_h).min(render_height);
                (0..tiles_x).map(move |tx| {
                    let x_start = tx * tile_w;
                    let x_end = (x_start + tile_w).min(render_width);
                    let cell_idx = ty as usize * tiles_x as usize + tx as usize;
                    (x_start, y_start, x_end, y_end, cell_idx)
                })
            })
            .collect();

        let process_tile = |&(x_start, y_start, x_end, y_end, cell_idx): &(u16, u16, u16, u16, usize)| {
            let frame_ptr = frame_base as *mut Vector4<u8>;
            let depth_ptr = depth_base as *mut f32;
            for &tri_idx in &tile_tri_lists[cell_idx] {
                let tri = &opaque_tris[tri_idx];
                let tri_min_x = tri.bbox_min.x;
                let tri_max_x = tri.bbox_max.x;
                let tri_min_y = tri.bbox_min.y;
                let tri_max_y = tri.bbox_max.y;

                if tri_max_x < x_start as f32 || tri_min_x > x_end as f32
                    || tri_max_y < y_start as f32 || tri_min_y > y_end as f32
                {
                    continue;
                }

                let x_min = (tri_min_x.floor() as u16).max(x_start);
                let x_max = (tri_max_x.ceil() as u16).min(x_end.saturating_sub(1));
                let y_min = (tri_min_y.floor() as u16).max(y_start);
                let y_max = (tri_max_y.ceil() as u16).min(y_end.saturating_sub(1));

                let tex = tri.texture_id.and_then(|id| textures.get(id).cloned());

                let w0 = tri.clip_w[0];
                let w1 = tri.clip_w[1];
                let w2 = tri.clip_w[2];

                let vertices = &tri.screen_vertices;

                for y in y_min..=y_max {
                    for x in x_min..=x_max {
                        let x_center = x as f32 + 0.5;
                        let y_center = y as f32 + 0.5;

                        let (alpha, beta, gamma) = compute_barycentric_2d(x_center, y_center, vertices);

                        if alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                            continue;
                        }

                        let w_inv_interp = alpha / w0 + beta / w1 + gamma / w2;
                        if w_inv_interp.abs() < 1e-8 {
                            continue;
                        }
                        let w_corr = 1.0 / w_inv_interp;

                        let z_interpolated = (alpha * vertices[0].z / w0
                            + beta * vertices[1].z / w1
                            + gamma * vertices[2].z / w2)
                            * w_corr;

                        let idx = y as usize * stride + x as usize;

                        // SAFETY: tiles are disjoint, each pixel is written by exactly one tile
                        unsafe {
                            if *depth_ptr.add(idx) <= z_interpolated {
                                continue;
                            }
                            *depth_ptr.add(idx) = z_interpolated;
                        }

                        let perspective_interp =
                            |a: f32, b: f32, g: f32, v0: f32, v1: f32, v2: f32, w0: f32, w1: f32, w2: f32, w_c: f32| {
                                (a * v0 / w0 + b * v1 / w1 + g * v2 / w2) * w_c
                            };

                        let r = perspective_interp(alpha, beta, gamma, tri.color_data[0][0], tri.color_data[1][0], tri.color_data[2][0], w0, w1, w2, w_corr);
                        let g_val = perspective_interp(alpha, beta, gamma, tri.color_data[0][1], tri.color_data[1][1], tri.color_data[2][1], w0, w1, w2, w_corr);
                        let b_val = perspective_interp(alpha, beta, gamma, tri.color_data[0][2], tri.color_data[1][2], tri.color_data[2][2], w0, w1, w2, w_corr);
                        let interpolated_color = Vector3::new(r.clamp(0.0, 1.0), g_val.clamp(0.0, 1.0), b_val.clamp(0.0, 1.0));

                        let nx = perspective_interp(alpha, beta, gamma, tri.normals[0][0], tri.normals[1][0], tri.normals[2][0], w0, w1, w2, w_corr);
                        let ny = perspective_interp(alpha, beta, gamma, tri.normals[0][1], tri.normals[1][1], tri.normals[2][1], w0, w1, w2, w_corr);
                        let nz = perspective_interp(alpha, beta, gamma, tri.normals[0][2], tri.normals[1][2], tri.normals[2][2], w0, w1, w2, w_corr);
                        let interpolated_normal = Vector3::new(nx, ny, nz).normalize();

                        let u = perspective_interp(alpha, beta, gamma, tri.tex_coords[0][0], tri.tex_coords[1][0], tri.tex_coords[2][0], w0, w1, w2, w_corr);
                        let v_tex = perspective_interp(alpha, beta, gamma, tri.tex_coords[0][1], tri.tex_coords[1][1], tri.tex_coords[2][1], w0, w1, w2, w_corr);
                        let interpolated_texcoords = Vector2::new(u, v_tex);

                        let px = perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][0], tri.viewspace_pos[1][0], tri.viewspace_pos[2][0], w0, w1, w2, w_corr);
                        let py = perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][1], tri.viewspace_pos[1][1], tri.viewspace_pos[2][1], w0, w1, w2, w_corr);
                        let pz = perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][2], tri.viewspace_pos[1][2], tri.viewspace_pos[2][2], w0, w1, w2, w_corr);
                        let interpolated_shadingcoords = Vector3::new(px, py, pz);

                        // Edge distances for potential AA
                        let v0 = vertices[0];
                        let v1 = vertices[1];
                        let v2 = vertices[2];

                        fn point_line_distance(p: &Vector3<f32>, a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
                            let ab = b - a;
                            let ap = p - a;
                            let cross_z = ab.x * ap.y - ab.y * ap.x;
                            let area_x2 = cross_z.abs();
                            let base_len = (ab.x * ab.x + ab.y * ab.y).sqrt();
                            if base_len < 1e-6 { return 0.0; }
                            area_x2 / base_len
                        }

                        let h0 = point_line_distance(&v0, &v1, &v2);
                        let h1 = point_line_distance(&v1, &v0, &v2);
                        let h2 = point_line_distance(&v2, &v0, &v1);
                        let screen_heights = Vector3::new(h0, h1, h2);

                        let world_h = inv_view * Vector4::new(
                            interpolated_shadingcoords.x,
                            interpolated_shadingcoords.y,
                            interpolated_shadingcoords.z,
                            1.0,
                        );
                        let world_pos = Vector3::new(world_h.x, world_h.y, world_h.z);

                        let payload = shader::FragmentShaderPayload::new(
                            Vector3::new(alpha, beta, gamma),
                            interpolated_shadingcoords,
                            world_pos,
                            interpolated_color,
                            interpolated_normal,
                            interpolated_texcoords,
                            tex.clone(),
                            screen_heights,
                        );

                        let pixel_color = if let Some(func) = shader {
                            func(&payload, &shadow_ctx)
                        } else {
                            Vector4::new(1.0, 1.0, 1.0, 1.0)
                        };

                        if pixel_color == Vector4::new(-1.0, -1.0, -1.0, -1.0) {
                            continue;
                        }

                        let src_a = pixel_color[3];
                        if src_a <= 0.001 {
                            continue;
                        }

                        let src = Vector4::<u8>::new(
                            (pixel_color[0] * 255.0).clamp(0.0, 255.0) as u8,
                            (pixel_color[1] * 255.0).clamp(0.0, 255.0) as u8,
                            (pixel_color[2] * 255.0).clamp(0.0, 255.0) as u8,
                            (src_a * 255.0).clamp(0.0, 255.0) as u8,
                        );

                        unsafe {
                            if src_a >= 1.0 {
                                *frame_ptr.add(idx) = src;
                            } else {
                                let dst = *frame_ptr.add(idx);
                                let sa = src[3] as f32 / 255.0;
                                let da = dst[3] as f32 / 255.0;
                                let out_a = sa + da * (1.0 - sa);
                                let inv_a = 1.0 / out_a;
                                *frame_ptr.add(idx) = Vector4::new(
                                    (src[0] as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) * inv_a,
                                    (src[1] as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) * inv_a,
                                    (src[2] as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) * inv_a,
                                    out_a * 255.0,
                                ).map(|c| c as u8);
                            }
                        }
                    }
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 || done == total_tiles {
                println!(
                    "  Tiles completed: {}/{} ({:.0}%)",
                    done,
                    total_tiles,
                    done as f32 / total_tiles as f32 * 100.0
                );
            }
        };

        let thread_count = self.config.thread_count;
        if thread_count > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count as usize)
                .build()
                .unwrap()
                .install(|| {
                    tiles.par_iter().for_each(process_tile);
                });
        } else {
            tiles.par_iter().for_each(process_tile);
        }

        // Pass 2: 渲染透明三角形，按深度从远到近排序，只做深度测试不写深度缓冲
        if !transparent_tris.is_empty() {
            let mut transparent_sorted: Vec<usize> = (0..transparent_tris.len()).collect();
            transparent_sorted.sort_by(|&a, &b| {
                let za = (transparent_tris[a].viewspace_pos[0].z
                    + transparent_tris[a].viewspace_pos[1].z
                    + transparent_tris[a].viewspace_pos[2].z)
                    / 3.0;
                let zb = (transparent_tris[b].viewspace_pos[0].z
                    + transparent_tris[b].viewspace_pos[1].z
                    + transparent_tris[b].viewspace_pos[2].z)
                    / 3.0;
                zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal)
            });

            println!("Rendering {} transparent triangles (sorted back-to-front)...", transparent_sorted.len());

            let stride = render_width as usize;
            let frame_ptr = frame_buf.as_mut_ptr() as *mut Vector4<u8>;
            let depth_ptr = depth_buf.as_mut_ptr() as *mut f32;

            for &tri_idx in &transparent_sorted {
                let tri = &transparent_tris[tri_idx];
                let x_min = (tri.bbox_min.x.floor() as u16).max(0);
                let x_max = (tri.bbox_max.x.ceil() as u16).min(render_width.saturating_sub(1));
                let y_min = (tri.bbox_min.y.floor() as u16).max(0);
                let y_max = (tri.bbox_max.y.ceil() as u16).min(render_height.saturating_sub(1));

                let tex = tri.texture_id.and_then(|id| textures.get(id).cloned());
                let w0 = tri.clip_w[0];
                let w1 = tri.clip_w[1];
                let w2 = tri.clip_w[2];
                let vertices = &tri.screen_vertices;

                for y in y_min..=y_max {
                    for x in x_min..=x_max {
                        let x_center = x as f32 + 0.5;
                        let y_center = y as f32 + 0.5;

                        let (alpha, beta, gamma) = compute_barycentric_2d(x_center, y_center, vertices);
                        if alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                            continue;
                        }

                        let w_inv_interp = alpha / w0 + beta / w1 + gamma / w2;
                        if w_inv_interp.abs() < 1e-8 {
                            continue;
                        }
                        let w_corr = 1.0 / w_inv_interp;

                        let z_interpolated = (alpha * vertices[0].z / w0
                            + beta * vertices[1].z / w1
                            + gamma * vertices[2].z / w2)
                            * w_corr;

                        let idx = y as usize * stride + x as usize;
                        unsafe {
                            if *depth_ptr.add(idx) <= z_interpolated {
                                continue;
                            }
                        }

                        let perspective_interp =
                            |a: f32, b: f32, g: f32, v0: f32, v1: f32, v2: f32, wa: f32, wb: f32, wg: f32, wc: f32| {
                                (a * v0 / wa + b * v1 / wb + g * v2 / wg) * wc
                            };

                        let interpolated_color = Vector3::new(
                            perspective_interp(alpha, beta, gamma, tri.color_data[0][0], tri.color_data[1][0], tri.color_data[2][0], w0, w1, w2, w_corr).clamp(0.0, 1.0),
                            perspective_interp(alpha, beta, gamma, tri.color_data[0][1], tri.color_data[1][1], tri.color_data[2][1], w0, w1, w2, w_corr).clamp(0.0, 1.0),
                            perspective_interp(alpha, beta, gamma, tri.color_data[0][2], tri.color_data[1][2], tri.color_data[2][2], w0, w1, w2, w_corr).clamp(0.0, 1.0),
                        );

                        let interpolated_normal = Vector3::new(
                            perspective_interp(alpha, beta, gamma, tri.normals[0][0], tri.normals[1][0], tri.normals[2][0], w0, w1, w2, w_corr),
                            perspective_interp(alpha, beta, gamma, tri.normals[0][1], tri.normals[1][1], tri.normals[2][1], w0, w1, w2, w_corr),
                            perspective_interp(alpha, beta, gamma, tri.normals[0][2], tri.normals[1][2], tri.normals[2][2], w0, w1, w2, w_corr),
                        ).normalize();

                        let interpolated_texcoords = Vector2::new(
                            perspective_interp(alpha, beta, gamma, tri.tex_coords[0][0], tri.tex_coords[1][0], tri.tex_coords[2][0], w0, w1, w2, w_corr),
                            perspective_interp(alpha, beta, gamma, tri.tex_coords[0][1], tri.tex_coords[1][1], tri.tex_coords[2][1], w0, w1, w2, w_corr),
                        );

                        let interpolated_shadingcoords = Vector3::new(
                            perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][0], tri.viewspace_pos[1][0], tri.viewspace_pos[2][0], w0, w1, w2, w_corr),
                            perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][1], tri.viewspace_pos[1][1], tri.viewspace_pos[2][1], w0, w1, w2, w_corr),
                            perspective_interp(alpha, beta, gamma, tri.viewspace_pos[0][2], tri.viewspace_pos[1][2], tri.viewspace_pos[2][2], w0, w1, w2, w_corr),
                        );

                        fn point_line_distance(p: &Vector3<f32>, a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
                            let ab = b - a;
                            let ap = p - a;
                            let cross_z = ab.x * ap.y - ab.y * ap.x;
                            let area_x2 = cross_z.abs();
                            let base_len = (ab.x * ab.x + ab.y * ab.y).sqrt();
                            if base_len < 1e-6 { return 0.0; }
                            area_x2 / base_len
                        }

                        let screen_heights = Vector3::new(
                            point_line_distance(&vertices[0], &vertices[1], &vertices[2]),
                            point_line_distance(&vertices[1], &vertices[0], &vertices[2]),
                            point_line_distance(&vertices[2], &vertices[0], &vertices[1]),
                        );

                        let world_h = inv_view * Vector4::new(
                            interpolated_shadingcoords.x,
                            interpolated_shadingcoords.y,
                            interpolated_shadingcoords.z,
                            1.0,
                        );
                        let world_pos = Vector3::new(world_h.x, world_h.y, world_h.z);

                        let payload = shader::FragmentShaderPayload::new(
                            Vector3::new(alpha, beta, gamma),
                            interpolated_shadingcoords,
                            world_pos,
                            interpolated_color,
                            interpolated_normal,
                            interpolated_texcoords,
                            tex.clone(),
                            screen_heights,
                        );

                        let pixel_color = if let Some(func) = shader {
                            func(&payload, &shadow_ctx)
                        } else {
                            Vector4::new(1.0, 1.0, 1.0, 1.0)
                        };

                        if pixel_color == Vector4::new(-1.0, -1.0, -1.0, -1.0) {
                            continue;
                        }

                        let src_a = pixel_color[3];
                        if src_a <= 0.001 {
                            continue;
                        }

                        let src = Vector4::<u8>::new(
                            (pixel_color[0] * 255.0).clamp(0.0, 255.0) as u8,
                            (pixel_color[1] * 255.0).clamp(0.0, 255.0) as u8,
                            (pixel_color[2] * 255.0).clamp(0.0, 255.0) as u8,
                            (src_a * 255.0).clamp(0.0, 255.0) as u8,
                        );

                        unsafe {
                            if src_a >= 1.0 {
                                *frame_ptr.add(idx) = src;
                            } else {
                                let dst = *frame_ptr.add(idx);
                                let sa = src[3] as f32 / 255.0;
                                let da = dst[3] as f32 / 255.0;
                                let out_a = sa + da * (1.0 - sa);
                                let inv_a = 1.0 / out_a;
                                *frame_ptr.add(idx) = Vector4::new(
                                    (src[0] as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) * inv_a,
                                    (src[1] as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) * inv_a,
                                    (src[2] as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) * inv_a,
                                    out_a * 255.0,
                                ).map(|c| c as u8);
                            }
                        }
                    }
                }
            }
        }

        // 调试：保存视口深度图
        Self::save_depth_png(
            &depth_buf,
            render_width as u32,
            render_height as u32,
            &format!("{}/viewport_depth.png", self.debug_dir.as_ref().unwrap()),
        );

        if scale > 1 {
            self.frame_buf = vec![Vector4::zeros(); original_width as usize * original_height as usize];
            for y in 0..original_height {
                for x in 0..original_width {
                    let mut r_sum = 0u32;
                    let mut g_sum = 0u32;
                    let mut b_sum = 0u32;
                    let mut a_sum = 0u32;
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let hx = (x as u32 * scale as u32 + sx as u32) as u16;
                            let hy = (y as u32 * scale as u32 + sy as u32) as u16;
                            let idx = hy as usize * stride + hx as usize;
                            let color = frame_buf[idx];
                            r_sum += color[0] as u32;
                            g_sum += color[1] as u32;
                            b_sum += color[2] as u32;
                            a_sum += color[3] as u32;
                        }
                    }
                    let pixel_count = (scale * scale) as u32;
                    let final_idx = y as usize * original_width as usize + x as usize;
                    self.frame_buf[final_idx] = Vector4::new(
                        (r_sum / pixel_count) as u8,
                        (g_sum / pixel_count) as u8,
                        (b_sum / pixel_count) as u8,
                        (a_sum / pixel_count) as u8,
                    );
                }
            }
        } else {
            self.frame_buf = frame_buf;
            self.depth_buf = depth_buf;
        }
    }

    pub fn frame_buffer(&self) -> &Vec<Vector4<u8>> {
        return &self.frame_buf;
    }

    pub fn save_png(&self, path: &str) {
        let w = self.width as u32;
        let h = self.height as u32;
        let mut img = image::RgbaImage::new(w, h);
        for (i, pixel) in self.frame_buf.iter().enumerate() {
            let x = (i % self.width as usize) as u32;
            let y = (i / self.width as usize) as u32;
            img.put_pixel(x, y, image::Rgba([pixel.x, pixel.y, pixel.z, pixel.w]));
        }
        img.save(path).expect("Failed to save PNG");

        // 同时保存到 debug 目录
        if let Some(ref dir) = self.debug_dir {
            let debug_path = format!("{}/final_output.png", dir);
            img.save(&debug_path).expect("Failed to save debug PNG");
            println!("Saved debug copy: {}", debug_path);
        }
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

                let world_h = self.view.try_inverse().map(|inv| inv * Vector4::new(
                    interpolated_shadingcoords.x,
                    interpolated_shadingcoords.y,
                    interpolated_shadingcoords.z,
                    1.0,
                )).unwrap_or(Vector4::new(0.0, 0.0, 0.0, 1.0));
                let world_pos = Vector3::new(world_h.x, world_h.y, world_h.z);

                let payload = shader::FragmentShaderPayload::new(
                    Vector3::<f32>::new(alpha, beta, gamma),
                    interpolated_shadingcoords,
                    world_pos,
                    interpolated_color,
                    interpolated_normal,
                    interpolated_texcoords,
                    None,
                    screen_heights,
                );

                let dummy_ctx = shader::ShadowContext {
                    cascades: Arc::new(Vec::new()),
                    light_dir: Vector3::new(-1.0, -1.0, -1.0).normalize(),
                    light_color: Vector3::new(1.0, 1.0, 1.0),
                    light_intensity: 3.0,
                };

                let pixel_color: Vector4<f32> = if let Some(func) = self.fragment_shader {
                    func(&payload, &dummy_ctx)
                } else {
                    Vector4::new(0.0, 0.0, 0.0, 1.0)
                };

                if pixel_color == Vector4::<f32>::new(-1.0, -1.0, -1.0, -1.0) {
                    continue;
                }

                let final_color = Vector4::<u8>::new(
                    (pixel_color[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[2] * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel_color[3] * 255.0).clamp(0.0, 255.0) as u8,
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

    pub fn set_textures(&mut self, textures: Vec<Arc<texture::Texture>>) {
        self.textures = textures;
    }
}

// impl Default for Rasterizer {
//     fn default() -> Self {
//         return Self::new(1240, 800);
//     }
// }
