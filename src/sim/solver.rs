use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub dt: f32,
    pub dx: f32,
    pub dy: f32,
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub source_type: u32,       // 0: CW, 1: Pulse
    pub source_freq: f32,
    pub source_amp: f32,
    pub source_phase: f32,
    pub source_angle: f32,      // in radians
    pub source_width: f32,
    pub pulse_width: f32,
    pub pulse_delay: f32,
    pub render_mode: u32,       // 0: Field, 1: Intensity
    pub display_contrast: f32,  // multiplier for display scaling
    pub clear_sim: u32,         // 1 if we should clear
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

impl Default for SimParams {
    fn default() -> Self {
        // Courant stability condition (in 2D with c = 1, dt <= 1 / sqrt(1/dx^2 + 1/dy^2))
        // dx = dy = 1.0 -> dt <= 1 / sqrt(2) = 0.707
        let dx = 1.0;
        let dy = 1.0;
        let dt = 0.5; // safely below Courant limit

        Self {
            dt,
            dx,
            dy,
            time: 0.0,
            width: 512,
            height: 512,
            source_type: 0, // CW
            source_freq: 0.08, // wavelength = 12.5 pixels
            source_amp: 2.0,
            source_phase: 0.0,
            source_angle: 0.0,
            source_width: 50.0,
            pulse_width: 20.0,
            pulse_delay: 60.0,
            render_mode: 0, // Field
            display_contrast: 0.5,
            clear_sim: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        }
    }
}

#[allow(dead_code)]
pub struct Solver {
    pub params: SimParams,
    pub param_buffer: wgpu::Buffer,

    pub texture_ez_a: wgpu::Texture,
    pub texture_ez_b: wgpu::Texture,
    pub texture_h_a: wgpu::Texture,
    pub texture_h_b: wgpu::Texture,
    pub texture_material: wgpu::Texture,
    pub texture_display: wgpu::Texture,

    pub view_ez_a: wgpu::TextureView,
    pub view_ez_b: wgpu::TextureView,
    pub view_h_a: wgpu::TextureView,
    pub view_h_b: wgpu::TextureView,
    pub view_material: wgpu::TextureView,
    pub view_display: wgpu::TextureView,

    pub bind_group_layout_uniform: wgpu::BindGroupLayout,
    pub bind_group_layout_h: wgpu::BindGroupLayout,
    pub bind_group_layout_e: wgpu::BindGroupLayout,
    pub bind_group_layout_render: wgpu::BindGroupLayout,

    pub bind_group_uniform: wgpu::BindGroup,
    
    // Ping-pong bind groups
    pub bind_group_h_step_a: wgpu::BindGroup, // Reads ez_a, h_a; writes h_b
    pub bind_group_h_step_b: wgpu::BindGroup, // Reads ez_b, h_b; writes h_a
    
    pub bind_group_e_step_a: wgpu::BindGroup, // Reads ez_a, h_b, material; writes ez_b
    pub bind_group_e_step_b: wgpu::BindGroup, // Reads ez_b, h_a, material; writes ez_a
    
    pub bind_group_render_a: wgpu::BindGroup, // Reads ez_a, material; writes display
    pub bind_group_render_b: wgpu::BindGroup, // Reads ez_b, material; writes display

    pub pipeline_h: wgpu::ComputePipeline,
    pub pipeline_e: wgpu::ComputePipeline,
    pub pipeline_render: wgpu::ComputePipeline,

    pub egui_texture_id: Option<egui::TextureId>,

    pub step_count: u32,
    pub is_running: bool,
    pub steps_per_frame: u32,

    pub material_data: Vec<f32>, // 4 floats per pixel: R = eps_r, G = sigma, B = source, A = 0
    pub material_dirty: bool,
}

impl Solver {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let params = SimParams {
            width,
            height,
            ..Default::default()
        };

        // Create uniform parameter buffer
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Helper closures for texture creation
        let create_sim_texture = |label: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };

        // Textures
        let texture_ez_a = create_sim_texture("Ez A Texture", wgpu::TextureFormat::R32Float);
        let texture_ez_b = create_sim_texture("Ez B Texture", wgpu::TextureFormat::R32Float);
        let texture_h_a = create_sim_texture("H A Texture", wgpu::TextureFormat::Rg32Float);
        let texture_h_b = create_sim_texture("H B Texture", wgpu::TextureFormat::Rg32Float);
        let texture_material = create_sim_texture("Material Texture", wgpu::TextureFormat::Rgba32Float);
        let texture_display = create_sim_texture("Display Texture", wgpu::TextureFormat::Rgba8Unorm);

        // Views
        let view_ez_a = texture_ez_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_ez_b = texture_ez_b.create_view(&wgpu::TextureViewDescriptor::default());
        let view_h_a = texture_h_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_h_b = texture_h_b.create_view(&wgpu::TextureViewDescriptor::default());
        let view_material = texture_material.create_view(&wgpu::TextureViewDescriptor::default());
        let view_display = texture_display.create_view(&wgpu::TextureViewDescriptor::default());

        // ----------------------------------------------------
        // Bind Group Layouts
        // ----------------------------------------------------
        
        // Group 0: Uniform params
        let bind_group_layout_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Params Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1: H-field update layout
        let bind_group_layout_h = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("H-field Update Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: E-field update layout
        let bind_group_layout_e = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("E-field Update Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Group 3: Render layout
        let bind_group_layout_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Bind Group: Uniform params
        let bind_group_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout_uniform,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            }],
        });

        // Ping-pong Bind Groups: H-field
        let bind_group_h_step_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("H-update Step A Bind Group"),
            layout: &bind_group_layout_h,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_ez_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_h_a) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view_h_b) },
            ],
        });

        let bind_group_h_step_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("H-update Step B Bind Group"),
            layout: &bind_group_layout_h,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view_ez_b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_h_b) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view_h_a) },
            ],
        });

        // Ping-pong Bind Groups: E-field
        let bind_group_e_step_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("E-update Step A Bind Group"),
            layout: &bind_group_layout_e,
            entries: &[
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view_ez_a) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view_ez_b) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&view_h_b) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&view_material) },
            ],
        });

        let bind_group_e_step_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("E-update Step B Bind Group"),
            layout: &bind_group_layout_e,
            entries: &[
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view_ez_b) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view_ez_a) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&view_h_a) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&view_material) },
            ],
        });

        // Render Bind Groups
        let bind_group_render_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render A Bind Group"),
            layout: &bind_group_layout_render,
            entries: &[
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&view_ez_a) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&view_material) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&view_display) },
            ],
        });

        let bind_group_render_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render B Bind Group"),
            layout: &bind_group_layout_render,
            entries: &[
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&view_ez_b) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&view_material) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&view_display) },
            ],
        });

        // ----------------------------------------------------
        // Compile Shaders & Create Compute Pipelines
        // ----------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDTD Maxwell Shader Module"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/simulation.wgsl"
            ))),
        });

        // H-field update pipeline: uses @group(0) (uniform) and @group(1) (H-field layouts)
        let pipeline_layout_h = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("H-Update Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_uniform, &bind_group_layout_h],
            push_constant_ranges: &[],
        });
        let pipeline_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("H-Update Compute Pipeline"),
            layout: Some(&pipeline_layout_h),
            module: &shader,
            entry_point: "update_h",
        });

        // E-field update pipeline: uses @group(0) (uniform) and @group(1) (E-field layouts)
        let pipeline_layout_e = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("E-Update Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_uniform, &bind_group_layout_e],
            push_constant_ranges: &[],
        });
        let pipeline_e = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("E-Update Compute Pipeline"),
            layout: Some(&pipeline_layout_e),
            module: &shader,
            entry_point: "update_e",
        });

        // Render pipeline: uses @group(0) (uniform) and @group(1) (render layouts)
        let pipeline_layout_render = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_uniform, &bind_group_layout_render],
            push_constant_ranges: &[],
        });
        let pipeline_render = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Render Compute Pipeline"),
            layout: Some(&pipeline_layout_render),
            module: &shader,
            entry_point: "render_display",
        });

        // Material CPU-side mirror buffer (initialized to vacuum: epsilon_r = 1.0, sigma = 0, source = 0, pad = 0)
        let mut material_data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            material_data.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }

        Self {
            params,
            param_buffer,
            texture_ez_a,
            texture_ez_b,
            texture_h_a,
            texture_h_b,
            texture_material,
            texture_display,
            view_ez_a,
            view_ez_b,
            view_h_a,
            view_h_b,
            view_material,
            view_display,
            bind_group_layout_uniform,
            bind_group_layout_h,
            bind_group_layout_e,
            bind_group_layout_render,
            bind_group_uniform,
            bind_group_h_step_a,
            bind_group_h_step_b,
            bind_group_e_step_a,
            bind_group_e_step_b,
            bind_group_render_a,
            bind_group_render_b,
            pipeline_h,
            pipeline_e,
            pipeline_render,
            egui_texture_id: None,
            step_count: 0,
            is_running: true,
            steps_per_frame: 4,
            material_data,
            material_dirty: true,
        }
    }

    /// Registers the display output texture with the egui_wgpu renderer
    pub fn register_egui_texture(&mut self, egui_renderer: &mut egui_wgpu::Renderer, device: &wgpu::Device) {
        let texture_id = egui_renderer.register_native_texture(
            device,
            &self.view_display,
            wgpu::FilterMode::Linear,
        );
        self.egui_texture_id = Some(texture_id);
    }

    /// Writes the CPU-side material data mirror buffer to the GPU material texture.
    pub fn update_material_texture(&mut self, queue: &wgpu::Queue) {
        if self.material_dirty {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture_material,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&self.material_data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.params.width * 16), // 4 floats * 4 bytes = 16 bytes
                    rows_per_image: Some(self.params.height),
                },
                wgpu::Extent3d {
                    width: self.params.width,
                    height: self.params.height,
                    depth_or_array_layers: 1,
                },
            );
            self.material_dirty = false;
        }
    }

    /// Steps the simulation by running H-field and E-field updates, then renders to display texture.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Run material copy if dirty
        self.update_material_texture(queue);

        // Update the params uniform buffer on GPU
        queue.write_buffer(&self.param_buffer, 0, bytemuck::cast_slice(&[self.params]));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simulation Step Encoder"),
        });

        // Run simulation microsteps if running
        let steps = if self.is_running { self.steps_per_frame } else { 0 };

        for _ in 0..steps {
            let use_step_a = self.step_count % 2 == 0;

            // 1. Update H-Field
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("H-field Update Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline_h);
                compute_pass.set_bind_group(0, &self.bind_group_uniform, &[]);
                if use_step_a {
                    // Reads ez_a, h_a; writes h_b
                    compute_pass.set_bind_group(1, &self.bind_group_h_step_a, &[]);
                } else {
                    // Reads ez_b, h_b; writes h_a
                    compute_pass.set_bind_group(1, &self.bind_group_h_step_b, &[]);
                }
                compute_pass.dispatch_workgroups(
                    (self.params.width + 15) / 16,
                    (self.params.height + 15) / 16,
                    1,
                );
            }

            // 2. Update E-Field
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("E-field Update Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline_e);
                compute_pass.set_bind_group(0, &self.bind_group_uniform, &[]);
                if use_step_a {
                    // Reads ez_a, h_b, material; writes ez_b
                    compute_pass.set_bind_group(1, &self.bind_group_e_step_a, &[]);
                } else {
                    // Reads ez_b, h_a, material; writes ez_a
                    compute_pass.set_bind_group(1, &self.bind_group_e_step_b, &[]);
                }
                compute_pass.dispatch_workgroups(
                    (self.params.width + 15) / 16,
                    (self.params.height + 15) / 16,
                    1,
                );
            }

            self.params.time += self.params.dt;
            self.step_count += 1;
        }

        // Handle a simulation clear command
        if self.params.clear_sim == 1 {
            // Run a clean pass to clear the textures by running the H and E shaders with clear_sim = 1
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Clear Pass H"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline_h);
                compute_pass.set_bind_group(0, &self.bind_group_uniform, &[]);
                compute_pass.set_bind_group(1, &self.bind_group_h_step_a, &[]);
                compute_pass.dispatch_workgroups(
                    (self.params.width + 15) / 16,
                    (self.params.height + 15) / 16,
                    1,
                );
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Clear Pass E"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline_e);
                compute_pass.set_bind_group(0, &self.bind_group_uniform, &[]);
                compute_pass.set_bind_group(1, &self.bind_group_e_step_a, &[]);
                compute_pass.dispatch_workgroups(
                    (self.params.width + 15) / 16,
                    (self.params.height + 15) / 16,
                    1,
                );
            }

            // Reset simulation clock and step count
            self.params.clear_sim = 0;
            self.params.time = 0.0;
            self.step_count = 0;

            // Send updated params to GPU immediately
            queue.write_buffer(&self.param_buffer, 0, bytemuck::cast_slice(&[self.params]));
        }

        // 3. Render Pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Render Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline_render);
            compute_pass.set_bind_group(0, &self.bind_group_uniform, &[]);

            let use_step_a = self.step_count % 2 == 0;
            if use_step_a {
                // E field is currently in A
                compute_pass.set_bind_group(1, &self.bind_group_render_a, &[]);
            } else {
                // E field is currently in B
                compute_pass.set_bind_group(1, &self.bind_group_render_b, &[]);
            }

            compute_pass.dispatch_workgroups(
                (self.params.width + 15) / 16,
                (self.params.height + 15) / 16,
                1,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Resets all field textures and sets the simulation time to 0.
    pub fn clear(&mut self) {
        self.params.clear_sim = 1;
    }
}
