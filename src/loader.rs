use nalgebra::{ Vector3, Vector2 };
use std::path::Path;

use crate::triangle;

pub fn load_gltf_model(model_path: &str) -> Result<Vec<Vec<triangle::Triangle>>, Box<dyn std::error::Error>> {
    let path = Path::new(model_path);
    let (document, buffers, _) = gltf::import(path)?;

    let mut mesh_triangle_list: Vec<Vec<triangle::Triangle>> = Vec::new();

    for mesh in document.meshes() {
        let mut current_mesh_triangles: Vec<triangle::Triangle> = Vec::new();
        
        let mut mesh_positions: Vec<[f32; 3]> = Vec::new();
        let mut mesh_normals: Vec<[f32; 3]> = Vec::new();
        let mut mesh_tex_coords: Vec<[f32; 2]> = Vec::new();
        let mut mesh_colors: Vec<[f32; 3]> = Vec::new();

        let mut primitive_vertex_offset = 0u32;

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                eprintln!("Skipping non-triangle primitive: {:?}", primitive.mode());
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // 读取位置
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or("Missing POSITION attribute")?
                .collect();

            let vert_count = positions.len();
            if vert_count == 0 {
                continue;
            }

            // 读取法线
            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(iter) => iter.collect(),
                None => vec![[0.0, 0.0, 0.0]; vert_count],
            };

            // 读取纹理坐标
            let tex_coords: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(iter) => iter.into_f32().collect(),
                None => vec![[0.0, 0.0]; vert_count],
            };

            // 读取颜色
            let colors: Vec<[f32; 3]> = match reader.read_colors(0) {
                Some(iter) => iter
                    .into_rgba_f32()
                    .map(|c| {
                        let rgba: [f32; 4] = c.into();
                        [rgba[0], rgba[1], rgba[2]]
                    })
                    .collect(),
                None => vec![[1.0, 1.0, 1.0]; vert_count],
            };

            // 读取索引
            let indices_flat: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..vert_count as u32).collect(),
            };

            let base_index = mesh_positions.len() as u32;
            mesh_positions.extend(positions);
            mesh_normals.extend(normals);
            mesh_tex_coords.extend(tex_coords);
            mesh_colors.extend(colors);

            // 处理索引
            for chunk in indices_flat.chunks(3) {
                if chunk.len() == 3 {
                    let idx0 = (chunk[0] + primitive_vertex_offset) as usize;
                    let idx1 = (chunk[1] + primitive_vertex_offset) as usize;
                    let idx2 = (chunk[2] + primitive_vertex_offset) as usize;

                    let mut tri = triangle::Triangle::new();

                    if idx0 < mesh_positions.len() {
                        let p = mesh_positions[idx0];
                        tri.v[0] = Vector3::new(p[0], p[1], p[2]);
                    }
                    if idx0 < mesh_normals.len() {
                        let n = mesh_normals[idx0];
                        tri.normal[0] = Vector3::new(n[0], n[1], n[2]);
                    }
                    if idx0 < mesh_tex_coords.len() {
                        let uv = mesh_tex_coords[idx0];
                        tri.tex_coords[0] = Vector2::new(uv[0], uv[1]);
                    }
                    if idx0 < mesh_colors.len() {
                        let c = mesh_colors[idx0];
                        tri.color[0] = Vector3::new(c[0], c[1], c[2]);
                    }

                    // 顶点 1
                    if idx1 < mesh_positions.len() {
                        let p = mesh_positions[idx1];
                        tri.v[1] = Vector3::new(p[0], p[1], p[2]);
                    }
                    if idx1 < mesh_normals.len() {
                        let n = mesh_normals[idx1];
                        tri.normal[1] = Vector3::new(n[0], n[1], n[2]);
                    }
                    if idx1 < mesh_tex_coords.len() {
                        let uv = mesh_tex_coords[idx1];
                        tri.tex_coords[1] = Vector2::new(uv[0], uv[1]);
                    }
                    if idx1 < mesh_colors.len() {
                        let c = mesh_colors[idx1];
                        tri.color[1] = Vector3::new(c[0], c[1], c[2]);
                    }

                    // 顶点 2
                    if idx2 < mesh_positions.len() {
                        let p = mesh_positions[idx2];
                        tri.v[2] = Vector3::new(p[0], p[1], p[2]);
                    }
                    if idx2 < mesh_normals.len() {
                        let n = mesh_normals[idx2];
                        tri.normal[2] = Vector3::new(n[0], n[1], n[2]);
                    }
                    if idx2 < mesh_tex_coords.len() {
                        let uv = mesh_tex_coords[idx2];
                        tri.tex_coords[2] = Vector2::new(uv[0], uv[1]);
                    }
                    if idx2 < mesh_colors.len() {
                        let c = mesh_colors[idx2];
                        tri.color[2] = Vector3::new(c[0], c[1], c[2]);
                    }

                    current_mesh_triangles.push(tri);
                }
            }

            primitive_vertex_offset += vert_count as u32;
        }

        if !current_mesh_triangles.is_empty() {
            mesh_triangle_list.push(current_mesh_triangles);
        }
    }

    if mesh_triangle_list.is_empty() {
        return Err("No valid triangles found in the glTF file.".into());
    }

    Ok(mesh_triangle_list)
}