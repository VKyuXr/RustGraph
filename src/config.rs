use serde::Deserialize;
use std::fs;
use toml;
use std::path::Path;

#[derive(Deserialize)]
pub struct Config {
    pub window: WindowConfig,
    pub rasterizer: RasterizerConfig,
}

#[derive(Deserialize)]
pub struct WindowConfig {
    pub title: String,
    pub width: u16,
    pub height: u16,
}

#[derive(Deserialize)]
pub struct RasterizerConfig {
    pub culling_enabled: bool,
    pub ssaa_scale: i32,
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
        }
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