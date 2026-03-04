use glfw::{ Action, Context, Key, fail_on_errors };
use nalgebra::{ Vector3 };
use softbuffer::{ Context as SoftContext, Surface };
use std::num::NonZeroU32;
use std::time::{ Instant, Duration };
use std::path::Path;

mod rasterizer;
mod triangle;
mod config;
mod loader;
mod shader;
mod texture;
// mod scene;

fn main() {
    // 加载配置文件
    let config = config::get_config();
    let window_title: String = config.window.title;
    let window_width: u16 = config.window.width;
    let window_height: u16 = config.window.height;
    let rasterizer_config = config.rasterizer;

    // 创建 GLWindow
    let mut glfw = glfw::init(glfw::fail_on_errors!()).unwrap();
    glfw.window_hint(glfw::WindowHint::Resizable(false));
    let (mut window, events) = glfw.create_window(
        window_width as u32,
        window_height as u32,
        &window_title,
        glfw::WindowMode::Windowed
    ).expect("Failed to create GLFW window.");

    // 设置为当前窗口并启用键盘轮询
    window.make_current();
    window.set_key_polling(true);

    // 创建 SoftBuffer
    let context = SoftContext::new(&window).expect("Failed to create softbuffer context.");
    let mut surface = Surface::new(&context, &window).expect("Failed to create surface.");

    let model_path = "./model/suzanne.glb";
    let mut all_triangles: Vec<triangle::Triangle> = Vec::new();

    if Path::new(model_path).exists() {
        println!("Loading model: {}", model_path);
        match loader::load_gltf_model(model_path) {
            Ok(mesh_list) => {
                println!("Successfully loaded {} meshes.", mesh_list.len());
                
                let mut total_tris = 0;
                for mesh_tris in mesh_list {
                    total_tris += mesh_tris.len();
                    all_triangles.extend(mesh_tris);
                }
                println!("Total triangles: {}", total_tris);
            },
            Err(e) => {
                eprintln!("Failed to load glTF model: {}", e);
            }
        }
    } else {
        eprintln!("Model file not found: {}. Using empty scene.", model_path);
    }

    // 创建 Rasterizer
    let mut r = rasterizer::Rasterizer::new(window_width, window_height, rasterizer_config);

    // 初始化矩阵
    let mut model_pos = Vector3::new(0.0, -0.11, 0.0);
    let mut model_rot = Vector3::new(0.0, 45.0, 0.0);
    let mut eye_pos = Vector3::<f32>::new(0.0, 0.0, 4.0);
    let eye_rot = Vector3::<f32>::new(0.0, 0.0, 0.0);

    // 设置矩阵
    r.set_model(rasterizer::model_matrix(model_pos, model_rot));
    r.set_view(rasterizer::view_matrix(eye_pos, eye_rot));
    r.set_projection(rasterizer::projection_matrix(45.0, window_width as f32 / window_height as f32, 0.1, 100.0));

    let mut should_close = false;

    // 帧率统计初始化
    let mut last_time = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    // 主循环
    while !should_close && !window.should_close() {
        // 启用事件
        glfw.poll_events();
        
        // 当尺寸为 0 时，跳过渲染
        let (win_width, win_height) = window.get_framebuffer_size();
        if win_width == 0 || win_height == 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        // 处理事件
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                should_close = true;
            }
        }

        // 关闭程序
        if should_close {
            break;
        }

        if let Err(e) = surface.resize(
            NonZeroU32::new(win_width as u32).unwrap(),
            NonZeroU32::new(win_height as u32).unwrap()) {
                eprintln!("Failed to resize surface: {}", e);
                continue;
            }

        // 渲染
        r.clear(rasterizer::Buffers::Color | rasterizer::Buffers::Depth);

        r.set_vertex_shader(shader::vertex_shader);
        // r.set_fragment_shader(shader::normal_fragment_shader);
        r.set_fragment_shader(shader::blinnphong_fragment_shader);
        // r.set_fragment_shader(shader::wireframe_fragment_shader);

        model_rot[1] += 1.0;
        r.set_model(rasterizer::model_matrix(model_pos, model_rot));

        r.set_view(rasterizer::view_matrix(eye_pos, eye_rot));
        r.set_projection(rasterizer::projection_matrix(45.0, window_width as f32 / window_height as f32, 0.1, 50.0));
        r.draw(&all_triangles);

        let fb = r.frame_buffer();
        let mut buffer = surface.buffer_mut().expect("Failed to get buffer");

        if buffer.width() != NonZeroU32::new(window_width as u32).unwrap() || buffer.height() != NonZeroU32::new(window_height as u32).unwrap() {
            continue;
        }

        for (y, row) in buffer.chunks_mut(window_width as usize).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let idx = y * (window_width as usize) + x;
                let color = fb[idx];
                let r_val = color.x.clamp(0, 255) as u32;
                let g_val = color.y.clamp(0, 255) as u32;
                let b_val = color.z.clamp(0, 255) as u32;
                
                *pixel = (0xFF << 24) | (r_val << 16) | (g_val << 8) | b_val;
            }
        }

        buffer.present().expect("Failed to present buffer");

        // 帧率统计
        frame_count += 1;

        let now = Instant::now();
        let duration = now.duration_since(last_time);

        if duration >= Duration::from_secs_f32(0.5) {
            fps = frame_count as f32 / duration.as_secs_f32();
            
            println!("CPU Rasterizer - FPS: {:.1}", fps);
            
            frame_count = 0;
            last_time = now;
        }
    }
    
    println!("Final Average FPS: {:.1}", fps);
}