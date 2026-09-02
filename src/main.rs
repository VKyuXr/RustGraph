use glfw::{ Action, Context, Key, fail_on_errors };
use nalgebra::{ Vector3 };
use softbuffer::{ Context as SoftContext, Surface };
use std::num::NonZeroU32;
use std::path::Path;

mod rasterizer;
mod triangle;
mod config;
mod loader;
mod shader;
mod texture;
mod light;

fn main() {
    // 加载配置文件
    let config = config::get_config();
    let window_width: u16 = config.window.width;
    let window_height: u16 = config.window.height;
    let rasterizer_config = config.rasterizer;

    // 加载模型
    let model_path = &config.model.path;
    let mut all_triangles: Vec<triangle::Triangle> = Vec::new();
    let mut textures = Vec::new();

    if Path::new(model_path).exists() {
        println!("Loading model: {}", model_path);
        match loader::load_gltf_model(model_path) {
            Ok(result) => {
                println!("Successfully loaded {} meshes.", result.meshes.len());
                let mut total_tris = 0;
                for mesh_tris in result.meshes {
                    total_tris += mesh_tris.len();
                    all_triangles.extend(mesh_tris);
                }
                textures = result.textures;
                println!("Total triangles: {}", total_tris);
            },
            Err(e) => {
                eprintln!("Failed to load glTF model: {}", e);
            }
        }
    } else {
        eprintln!("Model file not found: {}. Using empty scene.", model_path);
    }

    // 创建 Rasterizer（无窗口，纯离屏渲染）
    let mut r = rasterizer::Rasterizer::new(window_width, window_height, rasterizer_config);
    r.set_textures(textures);

    // 初始化矩阵
    let cam = &config.camera;
    println!("Camera config: eye_pos={:?}, eye_rot={:?}, model_pos={:?}, model_rot={:?}, fov={}, near={}, far={}",
        cam.eye_pos, cam.eye_rot, cam.model_pos, cam.model_rot, cam.fov, cam.near, cam.far);
    let model_pos = Vector3::from(cam.model_pos);
    let model_rot = Vector3::from(cam.model_rot);
    let eye_pos = Vector3::from(cam.eye_pos);
    let eye_rot = Vector3::from(cam.eye_rot);

    // 渲染
    r.clear(rasterizer::Buffers::Color | rasterizer::Buffers::Depth);
    r.set_vertex_shader(shader::vertex_shader);
    r.set_fragment_shader(shader::normal_fragment_shader);
    // r.set_fragment_shader(shader::texture_fragment_shader);
    r.set_fragment_shader(shader::pbr_fragment_shader);
    // r.set_model(rasterizer::model_matrix(model_pos, model_rot));
    r.set_view(rasterizer::view_matrix(eye_pos, eye_rot));
    r.set_projection(rasterizer::projection_matrix(cam.fov, window_width as f32 / window_height as f32, cam.near, cam.far));

    // 设置平行光 + 级联阴影贴图
    let light_dir = Vector3::new(-1.0, -1.0, -1.0);
    let light_color = Vector3::new(1.0, 1.0, 1.0);
    let light_intensity: f32 = 2.0;
    let mut light = light::DirectionalLight::new(
        light_dir,
        light_color,
        light_intensity,
    );
    r.set_light(light_color, light_intensity);
    let cam_view = rasterizer::view_matrix(eye_pos, eye_rot);
    light::compute_cascade_matrices(
        &mut light.cascades,
        light_dir.normalize(),
        &cam_view,
        cam.fov,
        window_width as f32 / window_height as f32,
    );
    r.set_debug_dir(config.output.debug_dir.clone());
    r.set_shadow_cascades(light.cascades);
    println!("Shadow cascades configured: {} levels.", 3);

    r.draw(&all_triangles);
    println!("Rendering complete.");

    // 保存 PNG
    let png_path = &config.output.png_path;
    r.save_png(png_path);
    println!("Saved PNG to: {}", png_path);

    // 创建窗口并显示
    let window_title = config.window.title;
    let mut glfw = glfw::init(glfw::fail_on_errors!()).unwrap();
    glfw.window_hint(glfw::WindowHint::Resizable(false));
    let (mut window, events) = glfw.create_window(
        window_width as u32,
        window_height as u32,
        &window_title,
        glfw::WindowMode::Windowed
    ).expect("Failed to create GLFW window.");

    window.make_current();
    window.set_key_polling(true);

    let context = SoftContext::new(&window).expect("Failed to create softbuffer context.");
    let mut surface = Surface::new(&context, &window).expect("Failed to create surface.");

    let (win_width, win_height) = window.get_framebuffer_size();
    if let Err(e) = surface.resize(
        NonZeroU32::new(win_width as u32).unwrap(),
        NonZeroU32::new(win_height as u32).unwrap()) {
            eprintln!("Failed to resize surface: {}", e);
            return;
        }

    let fb = r.frame_buffer();
    let mut buffer = surface.buffer_mut().expect("Failed to get buffer");

    for (y, row) in buffer.chunks_mut(window_width as usize).enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            let idx = y * (window_width as usize) + x;
            let color = fb[idx];
            let r_val = color.x.clamp(0, 255) as u32;
            let g_val = color.y.clamp(0, 255) as u32;
            let b_val = color.z.clamp(0, 255) as u32;
            let a_val = color.w.clamp(0, 255) as u32;
            *pixel = (a_val << 24) | (r_val << 16) | (g_val << 8) | b_val;
        }
    }

    buffer.present().expect("Failed to present buffer");

    drop(surface);
    drop(context);

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                window.set_should_close(true);
            }
        }
    }
}