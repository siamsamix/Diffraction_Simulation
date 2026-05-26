use crate::sim::solver::Solver;
use crate::sim::scene::{Preset, BrushType, apply_preset, draw_brush_stroke};

#[derive(Copy, Clone, Debug, PartialEq)]
enum GuiTab {
    Simulation,
    Waves,
    Brush,
    Presets,
}

pub struct Gui {
    active_tab: GuiTab,
    selected_preset: Preset,
    selected_brush: BrushType,
    brush_radius: i32,
    last_mouse_pos: Option<(i32, i32)>,
}

impl Gui {
    pub fn new() -> Self {
        Self {
            active_tab: GuiTab::Simulation,
            selected_preset: Preset::DoubleSlit,
            selected_brush: BrushType::PEC,
            brush_radius: 6,
            last_mouse_pos: None,
        }
    }

    /// Renders the egui layout and manages simulation configuration updates.
    pub fn draw(&mut self, ctx: &egui::Context, solver: &mut Solver) {
        // Apply beautiful dark styling
        self.apply_theme(ctx);

        // Sidebar Panel
        egui::SidePanel::left("control_panel")
            .resizable(false)
            .default_width(280.0)
            .frame(egui::Frame::none()
                .fill(ctx.style().visuals.window_fill())
                .inner_margin(12.0))
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    // Header title
                    ui.add_space(8.0);
                    ui.heading("⚡ Maxwell FDTD Solver");
                    ui.label("Diffraction & Wave Simulation");
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Tab Selector
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.active_tab, GuiTab::Simulation, "Sim");
                        ui.selectable_value(&mut self.active_tab, GuiTab::Waves, "Waves");
                        ui.selectable_value(&mut self.active_tab, GuiTab::Brush, "Brush");
                        ui.selectable_value(&mut self.active_tab, GuiTab::Presets, "Presets");
                    });
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(12.0);

                    // Render selected tab contents
                    match self.active_tab {
                        GuiTab::Simulation => self.draw_sim_tab(ui, solver),
                        GuiTab::Waves => self.draw_waves_tab(ui, solver),
                        GuiTab::Brush => self.draw_brush_tab(ui, solver),
                        GuiTab::Presets => self.draw_presets_tab(ui, solver),
                    }
                    
                    ui.add_space(20.0);
                    ui.separator();
                    ui.add_space(8.0);
                    
                    // Stats / Info
                    ui.heading("📊 Diagnostics");
                    ui.add_space(4.0);
                    egui::Grid::new("stats_grid").show(ui, |ui| {
                        ui.label("Grid Resolution:");
                        ui.colored_label(egui::Color32::WHITE, format!("{}x{}", solver.params.width, solver.params.height));
                        ui.end_row();

                        ui.label("Time Step (dt):");
                        ui.colored_label(egui::Color32::WHITE, format!("{:.3}", solver.params.dt));
                        ui.end_row();

                        ui.label("Elapsed Time:");
                        ui.colored_label(egui::Color32::WHITE, format!("{:.1}", solver.params.time));
                        ui.end_row();
                        
                        ui.label("Steps Counter:");
                        ui.colored_label(egui::Color32::WHITE, format!("{}", solver.step_count));
                        ui.end_row();
                    });
                });
            });

        // Center Viewport Panel
        egui::CentralPanel::default()
            .frame(egui::Frame::none()
                .fill(egui::Color32::from_rgb(10, 10, 12))
                .inner_margin(0.0))
            .show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    if let Some(texture_id) = solver.egui_texture_id {
                        let w = solver.params.width;
                        let h = solver.params.height;
                        
                        // Render texture matching aspect ratio
                        let response = ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            texture_id,
                            egui::vec2(w as f32, h as f32),
                        )).sense(egui::Sense::click_and_drag()));

                        // Handle drawing interactions
                        if response.dragged() || response.clicked() {
                            if let Some(hover_pos) = response.hover_pos() {
                                let rect = response.rect;
                                // Map mouse position in screen space to simulation coordinates (0..512)
                                let rx = (hover_pos.x - rect.min.x) / rect.width();
                                let ry = (hover_pos.y - rect.min.y) / rect.height();
                                
                                let gx = (rx * w as f32) as i32;
                                let gy = (ry * h as f32) as i32;
                                
                                if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                                    let prev_pos = self.last_mouse_pos.unwrap_or((gx, gy));
                                    draw_brush_stroke(solver, prev_pos.0, prev_pos.1, gx, gy, self.brush_radius, self.selected_brush);
                                    self.last_mouse_pos = Some((gx, gy));
                                }
                            }
                        } else {
                            self.last_mouse_pos = None;
                        }
                    } else {
                        ui.colored_label(egui::Color32::LIGHT_GRAY, "Initializing GPU viewport...");
                    }
                });
            });
    }

    fn draw_sim_tab(&mut self, ui: &mut egui::Ui, solver: &mut Solver) {
        ui.heading("🚀 Simulation Controls");
        ui.add_space(8.0);

        // Play / Pause
        ui.horizontal(|ui| {
            let label = if solver.is_running { "⏸ Pause" } else { "▶ Run" };
            if ui.button(label).clicked() {
                solver.is_running = !solver.is_running;
            }

            if ui.button("⏹ Clear").clicked() {
                solver.clear();
            }
            
            if !solver.is_running {
                if ui.button("⏭ Single Step").clicked() {
                    // Temporarily run 1 step
                    let prev_run = solver.is_running;
                    solver.is_running = true;
                    let prev_steps = solver.steps_per_frame;
                    solver.steps_per_frame = 1;
                    // Will run inside application update loop
                    solver.steps_per_frame = prev_steps;
                    solver.is_running = prev_run;
                }
            }
        });
        ui.add_space(12.0);

        // Simulation speed (timesteps per render frame)
        ui.label("Simulation Steps/Frame:");
        ui.add(egui::Slider::new(&mut solver.steps_per_frame, 1..=20).text("steps"));
        ui.add_space(16.0);

        ui.heading("🎨 Visualization");
        ui.add_space(8.0);

        // Render Mode (Field vs Intensity)
        ui.label("Render Mode:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut solver.params.render_mode, 0, "Field (Ez)");
            ui.radio_value(&mut solver.params.render_mode, 1, "Intensity (|Ez|²)");
        });
        ui.add_space(8.0);

        // Contrast adjustment
        ui.label("Signal Gain (Contrast):");
        ui.add(egui::Slider::new(&mut solver.params.display_contrast, 0.05..=4.0).text("gain"));
    }

    fn draw_waves_tab(&mut self, ui: &mut egui::Ui, solver: &mut Solver) {
        ui.heading("🌊 Source Properties");
        ui.add_space(8.0);

        // Source Type
        ui.label("Signal Waveform:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut solver.params.source_type, 0, "Continuous Wave (CW)");
            ui.radio_value(&mut solver.params.source_type, 1, "Gaussian Pulse");
        });
        ui.add_space(12.0);

        // Frequency (Wavelength)
        ui.label("Wave Frequency:");
        ui.add(egui::Slider::new(&mut solver.params.source_freq, 0.02..=0.25).text("f"));
        let wavelength = 1.0 / solver.params.source_freq;
        ui.label(format!("Approx Wavelength: {:.1} px", wavelength));
        ui.add_space(12.0);

        // Amplitude
        ui.label("Source Amplitude:");
        ui.add(egui::Slider::new(&mut solver.params.source_amp, 0.1..=10.0).text("A"));
        ui.add_space(12.0);

        if solver.params.source_type == 1 {
            // Pulse parameters
            ui.label("Pulse Duration:");
            ui.add(egui::Slider::new(&mut solver.params.pulse_width, 5.0..=100.0).text("σ"));
            
            ui.label("Pulse Launch Delay:");
            ui.add(egui::Slider::new(&mut solver.params.pulse_delay, 10.0..=250.0).text("delay"));
        }
    }

    fn draw_brush_tab(&mut self, ui: &mut egui::Ui, _solver: &mut Solver) {
        ui.heading("🖌 Paint Tools");
        ui.label("Click and drag inside the viewport to draw structures:");
        ui.add_space(12.0);

        // Brush Material selection
        ui.label("Active Brush Material:");
        ui.vertical(|ui| {
            ui.radio_value(&mut self.selected_brush, BrushType::PEC, "Metal Barrier (PEC)");
            ui.radio_value(&mut self.selected_brush, BrushType::Glass, "Glass Dielectric (n = 1.5)");
            ui.radio_value(&mut self.selected_brush, BrushType::Source, "Wave Source Generator");
            ui.radio_value(&mut self.selected_brush, BrushType::Vacuum, "Eraser (Vacuum)");
        });
        ui.add_space(12.0);

        // Brush Size
        ui.label("Brush Radius:");
        ui.add(egui::Slider::new(&mut self.brush_radius, 1..=32).text("px"));
    }

    fn draw_presets_tab(&mut self, ui: &mut egui::Ui, solver: &mut Solver) {
        ui.heading("📐 Optical Presets");
        ui.label("Load preset geometries:");
        ui.add_space(12.0);

        ui.vertical(|ui| {
            if ui.button("🧪 Double Slit Interference").clicked() {
                self.selected_preset = Preset::DoubleSlit;
                apply_preset(solver, Preset::DoubleSlit);
            }
            ui.add_space(6.0);
            
            if ui.button("🔬 Single Slit Diffraction").clicked() {
                self.selected_preset = Preset::SingleSlit;
                apply_preset(solver, Preset::SingleSlit);
            }
            ui.add_space(6.0);
            
            if ui.button("🔍 Convex Lens Focusing").clicked() {
                self.selected_preset = Preset::ConvexLens;
                apply_preset(solver, Preset::ConvexLens);
            }
            ui.add_space(6.0);
            
            if ui.button("💎 Prism Refraction & TIR").clicked() {
                self.selected_preset = Preset::Prism;
                apply_preset(solver, Preset::Prism);
            }
            ui.add_space(6.0);
            
            if ui.button("📏 Diffraction Grating").clicked() {
                self.selected_preset = Preset::DiffractionGrating;
                apply_preset(solver, Preset::DiffractionGrating);
            }
            ui.add_space(6.0);
            
            if ui.button("⚡ Waveguide Y-Splitter").clicked() {
                self.selected_preset = Preset::Waveguide;
                apply_preset(solver, Preset::Waveguide);
            }
            ui.add_space(12.0);
            
            ui.separator();
            ui.add_space(12.0);
            
            if ui.button("🧹 Clear Canvas (Empty Vacuum)").clicked() {
                self.selected_preset = Preset::Empty;
                apply_preset(solver, Preset::Empty);
            }
        });
    }

    /// Sets a customized, modern dark styling for the GUI panels.
    fn apply_theme(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        
        style.visuals.dark_mode = true;
        style.visuals.window_rounding = 8.0.into();
        style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(18, 18, 22);
        style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(28, 28, 33);
        style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(40, 40, 48);
        style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 60, 72);
        
        style.visuals.window_fill = egui::Color32::from_rgb(18, 18, 22);
        style.visuals.panel_fill = egui::Color32::from_rgb(14, 14, 16);
        
        ctx.set_style(style);
    }
}
