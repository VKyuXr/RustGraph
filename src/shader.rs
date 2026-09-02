use nalgebra::{Vector2, Vector3, Vector4};
use std::sync::Arc;

use crate::texture;
use crate::light::ShadowCascade;

pub struct FragmentShaderPayload {
    pub barycentric: Vector3<f32>,
    pub view_pos: Vector3<f32>,
    pub world_pos: Vector3<f32>,
    pub color: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub tex_coords: Vector2<f32>,
    pub texture: Option<Arc<texture::Texture>>,
    pub screen_heights: Vector3<f32>,
}

impl FragmentShaderPayload {
    pub fn new(
        barycentric: Vector3<f32>,
        view_pos: Vector3<f32>,
        world_pos: Vector3<f32>,
        color: Vector3<f32>,
        normal: Vector3<f32>,
        tex_coords: Vector2<f32>,
        texture: Option<Arc<texture::Texture>>,
        screen_heights: Vector3<f32>,
    ) -> Self {
        Self {
            barycentric,
            view_pos,
            world_pos,
            color,
            normal,
            tex_coords,
            texture,
            screen_heights,
        }
    }
}

pub struct ShadowContext {
    pub cascades: Arc<Vec<ShadowCascade>>,
    pub light_dir: Vector3<f32>,
    pub light_color: Vector3<f32>,
    pub light_intensity: f32,
}

pub struct VertexShaderPayload {
    position: Vector3<f32>,
}

pub fn vertex_shader(payload: &VertexShaderPayload) -> Vector3<f32> {
    payload.position
}

pub fn normal_fragment_shader(
    payload: &FragmentShaderPayload,
    _ctx: &ShadowContext,
) -> Vector4<f32> {
    let c = (payload.normal.normalize() + Vector3::new(1.0, 1.0, 1.0)) / 2.0;
    Vector4::new(c.x, c.y, c.z, 1.0)
}

pub fn texture_fragment_shader(
    payload: &FragmentShaderPayload,
    _ctx: &ShadowContext,
) -> Vector4<f32> {
    match &payload.texture {
        Some(tex) => {
            let u = payload.tex_coords.x;
            let v = payload.tex_coords.y;
            let rgba = tex.sample_linear(u, v);
            Vector4::new(rgba.x / 255.0, rgba.y / 255.0, rgba.z / 255.0, rgba.w / 255.0)
        }
        None => Vector4::new(payload.color.x, payload.color.y, payload.color.z, 1.0),
    }
}

pub fn blinnphong_fragment_shader(
    payload: &FragmentShaderPayload,
    _ctx: &ShadowContext,
) -> Vector4<f32> {
    let ka = Vector3::new(0.005, 0.005, 0.005);
    let kd = Vector3::new(1.0, 1.0, 1.0);
    let ks = Vector3::new(0.7937, 0.7937, 0.7937);

    let light_pos = Vector3::new(20.0, 20.0, 20.0);
    let light_intensity = Vector3::new(500.0, 500.0, 500.0);
    let amb_light = Vector3::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::new(0.0, 0.0, 4.0);
    let p = 60.0;

    let point = payload.view_pos;
    let normal = payload.normal.normalize();
    let mut result = ka.component_mul(&amb_light);

    let l = (light_pos - point).normalize();
    let v = (eye_pos - point).normalize();
    let n = normal;

    let r2 = (light_pos - point).norm().powi(2).max(1e-6);
    let diff = n.dot(&l).max(0.0);
    result += kd.component_mul(&light_intensity) * diff / r2;

    let h = (l + v).normalize();
    let spec = n.dot(&h).max(0.0).powf(p);
    result += ks.component_mul(&light_intensity) * spec / r2;

    Vector4::new(result.x, result.y, result.z, 1.0)
}

fn shadow_factor(
    world_pos: Vector3<f32>,
    normal: Vector3<f32>,
    light_dir: Vector3<f32>,
    cascade: &ShadowCascade,
) -> f32 {
    let ls = cascade.light_view_proj * Vector4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    if ls.w.abs() < 1e-8 {
        return 1.0;
    }
    let w_inv = 1.0 / ls.w;
    let ndc_x = ls.x * w_inv;
    let ndc_y = ls.y * w_inv;
    let ndc_z = ls.z * w_inv;

    if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
        return 1.0;
    }

    let res = cascade.resolution as f32;
    let u = ((ndc_x * 0.5 + 0.5) * res).clamp(0.0, res - 1.0);
    let v = ((ndc_y * 0.5 + 0.5) * res).clamp(0.0, res - 1.0);

    // 斜率缩放 bias：表面越倾斜于光源，bias 越大
    let n_dot_l = normal.dot(&light_dir).max(0.0);
    let min_bias: f32 = 0.001;
    let slope_bias = 0.005 * (1.0 - n_dot_l);
    let bias = min_bias.max(slope_bias);

    // 双线性 PCF：对 4 个相邻 texel 做深度比较，再用小数部分插值结果
    let res_u = cascade.resolution as usize;
    let u0 = (u.floor() as i32).clamp(0, res_u as i32 - 1) as usize;
    let v0 = (v.floor() as i32).clamp(0, res_u as i32 - 1) as usize;
    let u1 = (u0 + 1).min(res_u - 1);
    let v1 = (v0 + 1).min(res_u - 1);
    let fu = u - u.floor();
    let fv = v - v.floor();

    let d00 = cascade.depth_buffer[v0 * res_u + u0];
    let d10 = cascade.depth_buffer[v0 * res_u + u1];
    let d01 = cascade.depth_buffer[v1 * res_u + u0];
    let d11 = cascade.depth_buffer[v1 * res_u + u1];

    let s00 = if d00 == f32::INFINITY || ndc_z <= d00 + bias { 1.0 } else { 0.0 };
    let s10 = if d10 == f32::INFINITY || ndc_z <= d10 + bias { 1.0 } else { 0.0 };
    let s01 = if d01 == f32::INFINITY || ndc_z <= d01 + bias { 1.0 } else { 0.0 };
    let s11 = if d11 == f32::INFINITY || ndc_z <= d11 + bias { 1.0 } else { 0.0 };

    (s00 * (1.0 - fu) + s10 * fu) * (1.0 - fv) + (s01 * (1.0 - fu) + s11 * fu) * fv
}

pub fn pbr_fragment_shader(
    payload: &FragmentShaderPayload,
    ctx: &ShadowContext,
) -> Vector4<f32> {
    let roughness: f32 = 0.1;
    let indirect_coeff: f32 = 0.5;

    let (base_color, alpha) = match &payload.texture {
        Some(tex) => {
            let u = payload.tex_coords.x;
            let v = payload.tex_coords.y;
            let rgba = tex.sample_linear(u, v);
            (
                Vector3::new(rgba.x / 255.0, rgba.y / 255.0, rgba.z / 255.0),
                rgba.w / 255.0,
            )
        }
        None => (payload.color, 1.0),
    };

    let normal = payload.normal.normalize();
    let point = payload.view_pos;
    let view_dir = (-point).normalize();

    let mut result = indirect_coeff * base_color * (1.0 - roughness);

    let light_dir = -ctx.light_dir;
    let half_vec = (light_dir + view_dir).normalize();

    let n_dot_l = normal.dot(&light_dir).max(0.0);
    let n_dot_v = normal.dot(&view_dir).max(0.0);
    let h_dot_v = half_vec.dot(&view_dir).max(0.0);

    // 选择级联
    let view_depth = point.norm();
    let cascade_idx = if view_depth < ctx.cascades[0].far {
        0
    } else if view_depth < ctx.cascades[1].far {
        1
    } else {
        (2).min(ctx.cascades.len() - 1)
    };

    let s = shadow_factor(payload.world_pos, normal, light_dir, &ctx.cascades[cascade_idx]);

    let sun_intensity = ctx.light_color * ctx.light_intensity * s;

    // Disney 漫反射
    let fd90 = 0.5 + 2.0 * roughness * (h_dot_v * h_dot_v);
    let fd_l = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powf(5.0);
    let fd_v = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powf(5.0);
    let fd = fd_l * fd_v;
    let diffuse = base_color.component_mul(&sun_intensity) * n_dot_l * fd / std::f32::consts::PI;

    result += diffuse;

    Vector4::new(result.x, result.y, result.z, alpha)
}