use nalgebra::{ Matrix4, Vector3, Vector2, UnitQuaternion, Quaternion };
use std::path::Path;

// 顶点
pub struct Vertex {
    pub position: Vector3<f32>,     // 位置
    pub normal: Vector3<f32>,       // 法线
    pub tex_coords: Vector2<f32>,   // 纹理坐标
    pub color: Vector3<f32>,        // 顶点色
}

// 材质
pub struct Material {
    pub albedo: Vector3<f32>,       // 基础色
    pub roughness: f32,             // 粗糙度
    pub metallic: f32,              // 金属度
}

// 图元
pub struct Primitive {
    pub vertices: Vec<Vertex>,      // 顶点
    pub indices: Vec<u32>,          // 索引
    pub material_index: usize,      // 材质索引
    pub bbox_min: Vector3<f32>,     // 轴对齐包围盒最小坐标
    pub bbox_max: Vector3<f32>,       // 轴对齐包围盒最大坐标
}

// 网格
pub struct Mesh {
    pub name: String,               // 名称
    pub primitives: Vec<Primitive>, // 图元
}

// 光照
pub enum LightType {
    Directional,                    // 平行光
    Point,                          // 点光源
    Spot {                          // 聚光灯
        inner_angle: f32,           // 内角
        outer_angle: f32,           // 外角
    },
}

pub struct Light {
    pub color: Vector3<f32>,        // 颜色
    pub intensity: f32,             // 强度
    pub light_type: LightType,      // 光源类型
}

// 相机
pub struct Camera {
    pub fov_y_deg: f32,                     // 垂直视角场
    pub aspect_ratio: f32,                  // 宽高比
    pub z_near: f32,                        // 近裁剪面
    pub z_far: f32,                         // 远裁剪面
    pub projection_matrix: Matrix4<f32>,    // 投影矩阵
}

// 场景
pub enum NodeType {
    Empty,                                  // 空节点
    Mesh { mesh_index: usize },             // 网格索引
    Light { light_index: usize },           // 光源索引
    Camera { camera_index: usize },         // 相机索引
}

pub struct SceneNode {
    pub name: String,                       // 名字
    pub node_type: NodeType,                // 节点类型
    pub local_transform: Matrix4<f32>,      // 局部变换矩阵
    pub world_transform: Matrix4<f32>,      // 世界变换矩阵
    pub children: Vec<usize>,               // 子节点索引
}

pub struct Scene {
    pub meshes: Vec<Mesh>,                  // 网格
    pub lights: Vec<Light>,                 // 光源
    pub cameras: Vec<Camera>,               // 相机
    pub materials: Vec<Material>,            // 材质
    pub nodes: Vec<SceneNode>,              // 节点
    pub root_indices: Vec<usize>,           // 根节点索引
}

impl Scene {
    pub fn load_from_gltf(gltf_path: &str) -> Result<Scene, Box<dyn std::error::Error>> {
        let path = Path::new(gltf_path);

        // 导入 gltf 文档和资源
        let (document, buffers, images) = gltf::import(path)?;

        let mut scene = Self {
            meshes: Vec::new(),
            lights: Vec::new(),
            cameras: Vec::new(),
            materials: Vec::new(),
            nodes: Vec::new(),
            root_indices: Vec::new(),
        };

        // 加载材质
        for mat in document.materials() {
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            
            let material = Material {
                albedo: Vector3::new(base_color[0], base_color[1], base_color[2]),
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
            };
            scene.materials.push(material);
        }

        // 加载网格
        for mesh in document.meshes() {
            let mesh_name = mesh.name().unwrap_or("UnnamedMesh").to_string();
            let mut primitives = Vec::new();

            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                
                // 读取顶点属性
                let positions: Vec<[f32; 3]> = reader.read_positions()
                    .ok_or("Missing positions")?.collect();
                
                let normals: Vec<[f32; 3]> = reader.read_normals()
                    .map(|it| it.collect()).unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
                
                let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|it| it.into_f32().collect()).unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
                
                let colors: Vec<[f32; 4]> = reader.read_colors(0)
                    .map(|it| it.into_rgba_f32().map(|c| c.into()).collect()).unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

                let count = positions.len();
                let mut vertices = Vec::with_capacity(count);
                let mut min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);

                for i in 0..count {
                    let pos = Vector3::from(positions[i]);
                    let norm = Vector3::from(normals[i]);
                    let uv = Vector2::from(tex_coords[i]);
                    let col = Vector3::new(colors[i][0], colors[i][1], colors[i][2]);

                    if pos.x < min.x { min.x = pos.x; } if pos.x > max.x { max.x = pos.x; }
                    if pos.y < min.y { min.y = pos.y; } if pos.y > max.y { max.y = pos.y; }
                    if pos.z < min.z { min.z = pos.z; } if pos.z > max.z { max.z = pos.z; }

                    vertices.push(Vertex { position: pos, normal: norm, tex_coords: uv, color: col });
                }

                // 读取索引
                let indices: Vec<u32> = reader.read_indices()
                    .map(|it| it.into_u32().collect()).unwrap_or_default();

                let material_idx = prim.material().index().unwrap_or(0);

                primitives.push(Primitive {
                    vertices,
                    indices,
                    material_index: material_idx,
                    bbox_min: min,
                    bbox_max: max,
                });
            }

            scene.meshes.push(Mesh {
                name: mesh_name,
                primitives,
            });
        }

        // 构建节点层级关系
        for node in document.nodes() {
            let name = node.name().unwrap_or("Node").to_string();
            let mat = node.transform().matrix();
            let matrix = Matrix4::new(
                mat[0][0], mat[1][0], mat[2][0], mat[3][0],
                mat[0][1], mat[1][1], mat[2][1], mat[3][1],
                mat[0][2], mat[1][2], mat[2][2], mat[3][2],
                mat[0][3], mat[1][3], mat[2][3], mat[3][3],
            );

            let node_type = if let Some(mesh_idx) = node.mesh().map(|m| m.index()) {
                NodeType::Mesh { mesh_index: mesh_idx }
            } else {
                NodeType::Empty
            };

            scene.nodes.push(SceneNode {
                name,
                node_type,
                local_transform: matrix,
                world_transform: matrix,
                children: node.children().map(|c| c.index()).collect(),
            });
        }

        // 计算根节点和世界变换
        let mut is_child = vec![false; scene.nodes.len()];
        for node in &scene.nodes {
            for &child_idx in &node.children {
                if child_idx < scene.nodes.len() {
                    is_child[child_idx] = true;
                }
            }
        }

        for (i, &is_root) in is_child.iter().enumerate() {
            if !is_root {
                scene.root_indices.push(i);
            }
        }

        // 递归更新世界矩阵
        fn update_world_transform(scene: &mut Scene, node_idx: usize, parent_world: Matrix4<f32>) {
            let local = scene.nodes[node_idx].local_transform;
            let world = parent_world * local;
            scene.nodes[node_idx].world_transform = world;

            let children_indices: Vec<usize> = scene.nodes[node_idx].children.clone();
            for child_idx in children_indices {
                update_world_transform(scene, child_idx, world);
            }
        }

        let root_indices = scene.root_indices.clone();

        for &root_idx in &root_indices {
            update_world_transform(&mut scene, root_idx, Matrix4::identity());
        }

        Ok(scene)
    }

    // 通过名称查找节点索引 
    pub fn find_node_by_name(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    // 创建一个新的相机并绑定到指定节点
    pub fn setup_camera(
        &mut self, 
        node_index: usize, 
        fov_y_deg: f32, 
        aspect_ratio: f32, 
        z_near: f32, 
        z_far: f32
    ) -> Result<usize, String> {
        if node_index >= self.nodes.len() {
            return Err("Node index out of bounds".to_string());
        }

        // 计算投影矩阵
        let fov_rad = fov_y_deg.to_radians();
        let proj_matrix = Matrix4::new_perspective(aspect_ratio, fov_rad, z_near, z_far);

        let camera = Camera {
            fov_y_deg,
            aspect_ratio,
            z_near,
            z_far,
            projection_matrix: proj_matrix,
        };

        let cam_index = self.cameras.len();
        self.cameras.push(camera);

        // 更新节点类型
        self.nodes[node_index].node_type = NodeType::Camera { camera_index: cam_index };

        Ok(cam_index)
    }

    // 创建一个新光源并添加到场景资源列表
    pub fn create_light(&mut self, color: Vector3<f32>, intensity: f32, light_type: LightType) -> usize {
        let light = Light {
            color,
            intensity,
            light_type,
        };
        let light_index = self.lights.len();
        self.lights.push(light);
        light_index
    }

    // 将已创建的光源绑定到指定节点
    pub fn bind_light_to_node(&mut self, node_index: usize, light_index: usize) -> Result<(), String> {
        if node_index >= self.nodes.len() {
            return Err("Node index out of bounds".to_string());
        }
        if light_index >= self.lights.len() {
            return Err("Light index out of bounds".to_string());
        }

        self.nodes[node_index].node_type = NodeType::Light { light_index };
        Ok(())
    }

    // 更新场景中所有节点的世界变换矩阵
    pub fn update_transforms(&mut self) {
        fn update_recursive(scene: &mut Scene, node_idx: usize, parent_world: Matrix4<f32>) {
            let local = scene.nodes[node_idx].local_transform;
            let world = parent_world * local;
            scene.nodes[node_idx].world_transform = world;

            let children = scene.nodes[node_idx].children.clone();
            for child_idx in children {
                update_recursive(scene, child_idx, world);
            }
        }

        let root_indices = self.root_indices.clone();

        for &root_idx in &root_indices {
            update_recursive(self, root_idx, Matrix4::identity());
        }
    }
}