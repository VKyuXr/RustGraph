use image::{ GenericImageView, ImageReader, DynamicImage, Pixel };
use nalgebra::Vector4;
use std::path::Path;
use std::error::Error;

#[derive(Clone)]
pub struct Texture {
    image_data: DynamicImage,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();

        let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
        let width = img.width();
        let height = img.height();

        Ok(Texture {
            image_data: img,
            width,
            height,
        })
    }

    fn pixel_to_vec4(&self, pixel: impl Pixel<Subpixel = u8>) -> Vector4::<f32> {
        let channels = pixel.channels();
        let r = channels[0] as f32;
        let g = channels[1] as f32;
        let b = channels[2] as f32;
        let a = if channels.len() > 3 { channels[3] as f32 } else { 255.0 };
        Vector4::<f32>::new(r, g, b, a)
    }

    fn lerp_vec4(&self, a: Vector4<f32>, b: Vector4<f32>, t: f32) -> Vector4<f32> {
        Vector4::<f32>::new(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
            a.w + (b.w - a.w) * t,
        )
    }

    // 最近邻插值
    pub fn sample_nearest(&self, u: f32, v: f32) -> Vector4<f32> {
        let x = (u * self.width as f32).floor() as i32;
        let y = ((1.0 - v) * self.height as f32).floor() as i32;
        let x_clamped = x.clamp(0, self.width as i32 - 1) as u32;
        let y_clamped = y.clamp(0, self.height as i32 - 1) as u32;
        let pixel = self.image_data.get_pixel(x_clamped, y_clamped);
        
        self.pixel_to_vec4(pixel)
    }

    // 线性插值
    pub fn sample_linear(&self, u: f32, v: f32) -> Vector4<f32> {
        let x_float = u * self.width as f32 - 0.5;
        let y_float = (1.0 - v) * self.height as f32 - 0.5;

        let x0 = x_float.floor() as i32;
        let y0 = y_float.floor() as i32;

        let dx = x_float - x0 as f32;
        let dy = y_float - y0 as f32;

        let w = self.width as i32;
        let h = self.height as i32;

        let x1 = (x0 + 1).clamp(0, w - 1);
        let y1 = (y0 + 1).clamp(0, h - 1);
        let x0_clamped = x0.clamp(0, w - 1);
        let y0_clamped = y0.clamp(0, h - 1);

        let p00 = self.image_data.get_pixel(x0_clamped as u32, y0_clamped as u32);
        let p10 = self.image_data.get_pixel(x1 as u32, y0_clamped as u32);
        let p01 = self.image_data.get_pixel(x0_clamped as u32, y1 as u32);
        let p11 = self.image_data.get_pixel(x1 as u32, y1 as u32);

        let c00 = self.pixel_to_vec4(p00);
        let c10 = self.pixel_to_vec4(p10);
        let c01 = self.pixel_to_vec4(p01);
        let c11 = self.pixel_to_vec4(p11);

        let top = self.lerp_vec4(c00, c10, dx);
        let bottom = self.lerp_vec4(c01, c11, dx);
        
        self.lerp_vec4(top, bottom, dy)
    }
}