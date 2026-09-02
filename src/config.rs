use serde::Deserialize;
use std::fs;
use toml;
use std::path::Path;

#[derive(Deserialize)]
pub struct Config {
    pub window: WindowConfig,
    pub rasterizer: RasterizerConfig,
    pub camera: CameraConfig,
    pub model: ModelConfig,
    pub output: OutputConfig,
}

#[derive(Deserialize)]
pub struct WindowConfig {
    pub title: String,
    pub width: u16,
    pub height: u16,
}

#[derive(Deserialize, Clone, Copy)]
pub struct RasterizerConfig {
    pub culling_enabled: bool,
    pub ssaa_scale: i32,
    pub tile_width: u16,
    pub tile_height: u16,
    pub thread_count: u16,
    pub background_color: [u8; 4],
}

#[derive(Deserialize)]
pub struct CameraConfig {
    pub eye_pos: [f32; 3],
    pub eye_rot: [f32; 3],
    pub model_pos: [f32; 3],
    pub model_rot: [f32; 3],
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Deserialize)]
pub struct ModelConfig {
    pub path: String,
}

#[derive(Deserialize, Clone)]
pub struct OutputConfig {
    pub png_path: String,
    pub debug_dir: String,
}

fn load_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

fn default_config() -> Config {
    Config {
        window: WindowConfig {
            title: String::from("RustGraph(CPU Rasterizer)"),
            width: 1280,
            height: 800,
        },
        rasterizer: RasterizerConfig {
            culling_enabled: false,
            ssaa_scale: 1,
            tile_width: 128,
            tile_height: 128,
            thread_count: 0,
            background_color: [0, 0, 0, 255],
        },
        camera: CameraConfig {
            eye_pos: [0.0, 0.0, 4.0],
            eye_rot: [0.0, 0.0, 0.0],
            model_pos: [0.0, -0.11, 0.0],
            model_rot: [0.0, 45.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        },
        model: ModelConfig {
            path: String::from("./model/suzanne.glb"),
        },
        output: OutputConfig {
            png_path: String::from("./output.png"),
            debug_dir: String::from("./debug_output"),
        },
    }
}

pub fn get_config() -> Config {
    let config: Config;

    if Path::new("./config.toml").exists() {
        match load_config("./config.toml") {
            Ok(cfg) => {
                config = cfg;
                println!("配置加载成功");
            }
            Err(e) => {
                eprintln!("配置加载失败：{}", e);
                return default_config();
            }
        }
    } else {
        eprintln!("配置文件不存在");
        return default_config();
    }

    config
}