use nalgebra::{ Vector3, Vector2 };
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;

use crate::triangle;
use crate::texture;

pub struct LoadResult {
    pub meshes: Vec<Vec<triangle::Triangle>>,
    pub textures: Vec<Arc<texture::Texture>>,
}

pub fn load_gltf_model(model_path: &str) -> Result<LoadResult, Box<dyn std::error::Error>> {
    let path = Path::new(model_path);
    let parent = path.parent().unwrap_or(Path::new("."));

    let (document, buffers) = match gltf::import(path) {
        Ok((doc, bufs, _images)) => (doc, bufs),
        Err(e) => {
            eprintln!("gltf::import failed (likely missing images): {}", e);
            eprintln!("Falling back to buffer-only load...");

            let gltf_file = gltf::Gltf::open(path)?;
            let blob = gltf_file.blob.clone().unwrap_or_default();

            let mut buffer_data: Vec<gltf::buffer::Data> = Vec::new();
            for buffer in gltf_file.buffers() {
                let data = match buffer.source() {
                    gltf::buffer::Source::Bin => blob.clone(),
                    gltf::buffer::Source::Uri(uri) => {
                        let bin_path = parent.join(uri);
                        std::fs::read(&bin_path)?
                    }
                };
                buffer_data.push(gltf::buffer::Data(data));
            }

            (gltf_file.document, buffer_data)
        }
    };

    // 用 image crate 加载所有贴图（支持 TGA/BMP/PNG/JPEG）
    let mut textures: Vec<Arc<texture::Texture>> = Vec::new();
    let mut image_to_tex: HashMap<usize, usize> = HashMap::new();
    let mut path_to_tex: HashMap<String, usize> = HashMap::new();

    for tex in document.textures() {
        let img = tex.source();
        match img.source() {
            gltf::image::Source::Uri { uri, .. } => {
                let img_path = parent.join(uri);
                let path_str = img_path.to_string_lossy().to_string();

                if let Some(&cached_idx) = path_to_tex.get(&path_str) {
                    image_to_tex.insert(img.index(), cached_idx);
                    continue;
                }

                match texture::Texture::new(&img_path) {
                    Ok(t) => {
                        let idx = textures.len();
                        image_to_tex.insert(img.index(), idx);
                        path_to_tex.insert(path_str, idx);
                        textures.push(Arc::new(t));
                        println!("Loaded texture: {}", uri);
                    }
                    Err(e) => {
                        eprintln!("Failed to load texture {}: {}", uri, e);
                    }
                }
            }
            _ => {}
        }
    }

    println!("Loaded {} textures.", textures.len());

    // 构建 material index -> texture index 映射
    let mut material_tex: HashMap<usize, usize> = HashMap::new();
    for mat in document.materials() {
        if let Some(info) = mat.pbr_metallic_roughness().base_color_texture() {
            let tex_idx = info.texture().source().index();
            if let Some(&our_idx) = image_to_tex.get(&tex_idx) {
                material_tex.insert(mat.index().unwrap_or(0), our_idx);
            }
        }
    }

    let mut mesh_triangle_list: Vec<Vec<triangle::Triangle>> = Vec::new();

    println!("Materials: {}, material_tex mappings: {}", 
        document.materials().count(), material_tex.len());

    let mut textured_tris = 0u32;
    let mut untextured_tris = 0u32;

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

            // 获取该 primitive 的贴图索引
            let mat_idx = primitive.material().index();
            let prim_tex_id: Option<usize> = mat_idx.and_then(|mi| {
                material_tex.get(&mi).copied()
            });

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or("Missing POSITION attribute")?
                .collect();

            let vert_count = positions.len();
            if vert_count == 0 {
                continue;
            }

            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(iter) => iter.collect(),
                None => vec![[0.0, 0.0, 0.0]; vert_count],
            };

            let tex_coords: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(iter) => iter.into_f32().collect(),
                None => vec![[0.0, 0.0]; vert_count],
            };

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

            let indices_flat: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..vert_count as u32).collect(),
            };

            mesh_positions.extend(positions);
            mesh_normals.extend(normals);
            mesh_tex_coords.extend(tex_coords);
            mesh_colors.extend(colors);

            for chunk in indices_flat.chunks(3) {
                if chunk.len() == 3 {
                    let idx0 = (chunk[0] + primitive_vertex_offset) as usize;
                    let idx1 = (chunk[1] + primitive_vertex_offset) as usize;
                    let idx2 = (chunk[2] + primitive_vertex_offset) as usize;

                    let mut tri = triangle::Triangle::new();
                    tri.texture_id = prim_tex_id;
                    if prim_tex_id.is_some() { textured_tris += 1; } else { untextured_tris += 1; }

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

    println!("Triangles: {} textured, {} untextured", textured_tris, untextured_tris);

    Ok(LoadResult {
        meshes: mesh_triangle_list,
        textures,
    })
}