use nalgebra::{ Vector2, Vector3, Vector4 };

pub struct Triangle {
    pub v: [Vector3<f32>; 3],           // 坐标
    pub color: [Vector3<f32>; 3],       // 颜色
    pub tex_coords: [Vector2<f32>; 3],  // 纹理坐标
    pub normal: [Vector3<f32>; 3],      // 法线向量
}

impl Triangle {
    // 创建 Triangle 实例
    pub fn new() -> Self {
        return Self {
            v: [Vector3::<f32>::zeros(); 3],
            color: [Vector3::<f32>::zeros(); 3],
            tex_coords: [Vector2::<f32>::zeros(); 3],
            normal: [Vector3::<f32>::zeros(); 3],
        }
    }

    // 获取顶点
    pub fn a(&self) -> Vector3<f32> {
        return self.v[0];
    }

    pub fn b(&self) -> Vector3<f32> {
        return self.v[1];
    }

    pub fn c(&self) -> Vector3<f32> {
        return self.v[2];
    }

    // 设置第 ind 个顶点的坐标
    pub fn set_vertex(&mut self, ind: u32, ver: Vector3::<f32>) {
        self.v[ind as usize] = ver;
    }

    pub fn get_vertex(&self) -> [Vector3<f32>; 3] {
        self.v
    }

    // 设置第 ind 个顶点的颜色
    pub fn set_color(&mut self, ind: u32, r: i32, g: i32, b:i32) {
        if (r < 0) || (r > 255) || (g < 0) || (g > 255) || (b < 0) || (b > 255) {
            panic!("Invalid color values");
        }
        
        self.color[ind as usize] = Vector3::<f32>::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    }

    // 设置第 ind 个顶点的纹理坐标
    pub fn set_tex_coord(&mut self, ind: u32, s: f32, t: f32) {
        self.tex_coords[ind as usize] = Vector2::<f32>::new(s, t);
    }

    // 设置第 ind 个顶点的法线向量
    pub fn set_normal(&mut self, ind: u32, n: Vector3::<f32>) {
        self.normal[ind as usize] = n;
    }

    // 转换为 Vector4
    pub fn to_vector4(&self) -> [Vector4::<f32>; 3] {
        return [
            Vector4::<f32>::new(self.v[0].x, self.v[0].y, self.v[0].z, 1.0),
            Vector4::<f32>::new(self.v[1].x, self.v[1].y, self.v[1].z, 1.0),
            Vector4::<f32>::new(self.v[2].x, self.v[2].y, self.v[2].z, 1.0),
        ];
    }
}

impl Default for Triangle {
    fn default() -> Self {
        return Self::new();
    }
}