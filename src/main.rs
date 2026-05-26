mod sim;
mod gui;

use std::sync::Arc;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use sim::Solver;
use gui::Gui;

struct AppState {
    window: Arc<winit::window::Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    solver: Solver,
    gui: Gui,
}

impl AppState {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();

        // 1. Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Safe static surface creation by passing Arc<Window>
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a compatible graphics adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Simulation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create logical graphics device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // VSync enabled
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // 2. Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );

        let mut egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // 3. Initialize simulation Solver
        let mut solver = Solver::new(&device, 512, 512);
        
        // Register display texture with egui
        solver.register_egui_texture(&mut egui_renderer, &device);

        // Apply double slit preset initially on startup
        sim::scene::apply_preset(&mut solver, sim::scene::Preset::DoubleSlit);

        let gui = Gui::new();

        Self {
            window,
            device,
            queue,
            surface,
            surface_config,
            egui_ctx,
            egui_winit,
            egui_renderer,
            solver,
            gui,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn render(&mut self) {
        // Step FDTD equations on GPU
        self.solver.step(&self.device, &self.queue);

        // Acquire swapchain frame
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to acquire surface texture: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Render Encoder"),
        });

        // Begin egui layout frame
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);

        // Draw GUI panels and central simulation viewport
        self.gui.draw(&self.egui_ctx, &mut self.solver);

        let full_output = self.egui_ctx.end_frame();
        
        // Tessellate UI geometry
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        // Upload new UI texture layers (like fonts) to GPU
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        // Upload vertex & index buffers for egui UI shapes
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Draw egui onto swapchain surface
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 10.0 / 255.0,
                            g: 10.0 / 255.0,
                            b: 12.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer
                .render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Clean up outdated textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        output.present();
    }
}

fn main() {
    // Set up logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let event_loop = EventLoop::new().unwrap();
    
    // Create window
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Maxwell FDTD Diffraction Simulator")
            .with_inner_size(winit::dpi::PhysicalSize::new(960, 600))
            .build(&event_loop)
            .unwrap(),
    );

    // Initialize state asynchronously and block
    let mut state = pollster::block_on(AppState::new(window.clone()));

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { window_id, event: win_event } if window_id == window.id() => {
                // Pass window events to egui-winit
                let response = state.egui_winit.on_window_event(&window, &win_event);
                if response.consumed {
                    return;
                }

                match win_event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size.width, physical_size.height);
                    }
                    WindowEvent::RedrawRequested => {
                        state.render();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
