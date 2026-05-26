use crate::sim::solver::Solver;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Preset {
    Empty,
    SingleSlit,
    DoubleSlit,
    ConvexLens,
    Prism,
    DiffractionGrating,
    Waveguide,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BrushType {
    Vacuum,
    PEC,       // Perfect Electrical Conductor / Metal Barrier
    Glass,     // Dielectric Refractor (eps_r = 2.25)
    Source,    // Wave source injection
}

/// Applies a preset scene to the solver
pub fn apply_preset(solver: &mut Solver, preset: Preset) {
    let w = solver.params.width;
    let h = solver.params.height;
    
    // Reset simulation time and clear textures
    solver.clear();
    
    // Fill with vacuum default: [eps_r = 1.0, sigma = 0.0, source = 0.0, pad = 0.0]
    solver.material_data.fill(0.0);
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            solver.material_data[idx] = 1.0; // eps_r
        }
    }
    
    // Default source settings for scenes
    solver.params.source_freq = 0.08; // Wavelength ~ 12.5 pixels
    solver.params.source_amp = 2.0;
    solver.params.source_type = 0; // CW
    
    match preset {
        Preset::Empty => {
            // Nothing to do, already cleared to vacuum
        }
        
        Preset::SingleSlit => {
            let barrier_x = w / 3;
            let slit_y = h / 2;
            let slit_width = 12; // 12 pixels wide
            
            // Draw barrier
            for y in 0..h {
                if (y as i32 - slit_y as i32).abs() > (slit_width as i32 / 2) {
                    set_material_pixel(solver, barrier_x, y, 0.0, 0.0, 0.0); // PEC (eps_r = 0.0)
                }
            }
            
            // Draw vertical plane wave source on the left
            let src_x = 24;
            for y in 32..(h - 32) {
                set_source_pixel(solver, src_x, y, 1.0);
            }
        }
        
        Preset::DoubleSlit => {
            let barrier_x = w / 3;
            let slit_y = h / 2;
            let slit_width = 8;
            let slit_spacing = 24; // Distance between centers
            
            let slit1_y = slit_y - slit_spacing / 2;
            let slit2_y = slit_y + slit_spacing / 2;
            
            // Draw barrier
            for y in 0..h {
                let inside_slit1 = (y as i32 - slit1_y as i32).abs() < (slit_width as i32 / 2);
                let inside_slit2 = (y as i32 - slit2_y as i32).abs() < (slit_width as i32 / 2);
                if !inside_slit1 && !inside_slit2 {
                    set_material_pixel(solver, barrier_x, y, 0.0, 0.0, 0.0); // PEC
                }
            }
            
            // Draw vertical plane wave source
            let src_x = 24;
            for y in 32..(h - 32) {
                set_source_pixel(solver, src_x, y, 1.0);
            }
        }
        
        Preset::ConvexLens => {
            let lens_center_x = w / 2;
            let lens_center_y = h / 2;
            let r = 240.0; // Radius of lens curvature
            let offset = 216.0; // Center offset for lens shape
            
            // Equation for convex lens (intersection of two circles)
            // C1: (x - (cx - offset))^2 + (y - cy)^2 <= r^2
            // C2: (x - (cx + offset))^2 + (y - cy)^2 <= r^2
            let cx1 = lens_center_x as f32 - offset;
            let cx2 = lens_center_x as f32 + offset;
            let cy = lens_center_y as f32;
            
            for y in 0..h {
                for x in 0..w {
                    let dx1 = x as f32 - cx1;
                    let dx2 = x as f32 - cx2;
                    let dy = y as f32 - cy;
                    
                    let inside_c1 = dx1 * dx1 + dy * dy <= r * r;
                    let inside_c2 = dx2 * dx2 + dy * dy <= r * r;
                    
                    if inside_c1 && inside_c2 {
                        set_material_pixel(solver, x, y, 2.25, 0.0, 0.0); // Glass
                    }
                }
            }
            
            // Draw vertical plane wave source
            let src_x = 24;
            for y in 32..(h - 32) {
                set_source_pixel(solver, src_x, y, 1.0);
            }
        }
        
        Preset::Prism => {
            // Triangular prism vertices
            // A = (W/2 - 60, H/2 - 70)
            // B = (W/2 - 60, H/2 + 70)
            // C = (W/2 + 60, H/2)
            let ax = (w / 2) as f32 - 60.0;
            let ay = (h / 2) as f32 - 70.0;
            let bx = (w / 2) as f32 - 60.0;
            let by = (h / 2) as f32 + 70.0;
            let cx = (w / 2) as f32 + 60.0;
            let cy = (h / 2) as f32;
            
            // Helper to compute triangle area
            let area = |x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32| -> f32 {
                ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0).abs()
            };
            
            let tri_area = area(ax, ay, bx, by, cx, cy);
            
            for y in 0..h {
                for x in 0..w {
                    let px = x as f32;
                    let py = y as f32;
                    
                    let a1 = area(px, py, bx, by, cx, cy);
                    let a2 = area(ax, ay, px, py, cx, cy);
                    let a3 = area(ax, ay, bx, by, px, py);
                    
                    // If sum of sub-triangle areas equals the total triangle area (with float tolerance)
                    if (a1 + a2 + a3 - tri_area).abs() < 0.1 {
                        set_material_pixel(solver, x, y, 2.25, 0.0, 0.0); // Glass
                    }
                }
            }
            
            // Draw plane wave source tilted at 20 degrees or a straight source
            // Tilting the source: draw a diagonal line source
            let src_x_start = 24.0;
            let src_y_start = 64.0;
            let angle_rad = 20.0f32.to_radians();
            
            // Draw line source at angle
            for i in 0..380 {
                let px = src_x_start + i as f32 * angle_rad.sin();
                let py = src_y_start + i as f32 * angle_rad.cos();
                if px >= 0.0 && px < w as f32 && py >= 0.0 && py < h as f32 {
                    // Inject a directional phased source: we will just use a line source
                    // In a line source, waves naturally propagate perpendicular to the line!
                    set_source_pixel(solver, px as u32, py as u32, 1.0);
                }
            }
        }
        
        Preset::DiffractionGrating => {
            let barrier_x = w / 3;
            let period = 8;     // Total pixels per grating period
            let slit_width = 3;  // Slit size in pixels
            
            // Draw grating (alternating barrier and slits)
            for y in 0..h {
                let mod_y = y % period;
                if mod_y >= slit_width {
                    set_material_pixel(solver, barrier_x, y, 0.0, 0.0, 0.0); // PEC
                }
            }
            
            // Draw vertical plane wave source
            let src_x = 24;
            for y in 32..(h - 32) {
                set_source_pixel(solver, src_x, y, 1.0);
            }
        }
        
        Preset::Waveguide => {
            let y_center = h / 2;
            let width_wg = 14;
            let eps_wg = 4.0; // Silicon-like high index waveguide
            
            // Y-Splitter waveguide structure
            // Horizontal stem: x from 0 to W/4
            // Curves: from W/4 branching out symmetrically to W/2, then straight to W
            for x in 0..w {
                let x_f = x as f32;
                if x < w / 4 {
                    // Straight stem
                    for dy in -(width_wg as i32 / 2)..=(width_wg as i32 / 2) {
                        let py = (y_center as i32 + dy) as u32;
                        set_material_pixel(solver, x, py, eps_wg, 0.0, 0.0);
                    }
                } else if x < w / 2 {
                    // Branching curves
                    let t = (x_f - (w / 4) as f32) / ((w / 2) as f32 - (w / 4) as f32); // 0 to 1
                    
                    // We curve the two branches apart using a smooth cubic/sinusoidal interpolation
                    let offset_y = 60.0 * (t * 3.14159265 / 2.0).sin(); // curves up to 60px away
                    
                    let y_top = y_center as f32 - offset_y;
                    let y_bottom = y_center as f32 + offset_y;
                    
                    for dy in -(width_wg as i32 / 2)..=(width_wg as i32 / 2) {
                        let py_top = (y_top + dy as f32) as u32;
                        let py_bottom = (y_bottom + dy as f32) as u32;
                        set_material_pixel(solver, x, py_top, eps_wg, 0.0, 0.0);
                        set_material_pixel(solver, x, py_bottom, eps_wg, 0.0, 0.0);
                    }
                } else {
                    // Two straight branches to the end
                    let y_top = y_center - 60;
                    let y_bottom = y_center + 60;
                    
                    for dy in -(width_wg as i32 / 2)..=(width_wg as i32 / 2) {
                        let py_top = (y_top as i32 + dy) as u32;
                        let py_bottom = (y_bottom as i32 + dy) as u32;
                        set_material_pixel(solver, x, py_top, eps_wg, 0.0, 0.0);
                        set_material_pixel(solver, x, py_bottom, eps_wg, 0.0, 0.0);
                    }
                }
            }
            
            // Draw a localized source at the input of the waveguide stem
            let src_x = 12;
            for dy in -(width_wg as i32 / 2)..=(width_wg as i32 / 2) {
                let py = (y_center as i32 + dy) as u32;
                // Add a source with a Gaussian spatial profile to launch a nice fundamental mode
                let dist_norm = dy as f32 / (width_wg as f32 / 2.0);
                let amp = (-dist_norm * dist_norm * 2.0).exp();
                set_source_pixel(solver, src_x, py, amp);
            }
        }
    }
    
    solver.material_dirty = true;
}

// Helper to set material properties
fn set_material_pixel(solver: &mut Solver, x: u32, y: u32, eps_r: f32, sigma: f32, source: f32) {
    let w = solver.params.width;
    let h = solver.params.height;
    if x >= w || y >= h {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    solver.material_data[idx] = eps_r;
    solver.material_data[idx + 1] = sigma;
    solver.material_data[idx + 2] = source;
}

// Helper to update the source map (blue channel)
fn set_source_pixel(solver: &mut Solver, x: u32, y: u32, intensity: f32) {
    let w = solver.params.width;
    let h = solver.params.height;
    if x >= w || y >= h {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    solver.material_data[idx + 2] = intensity;
}

/// Draws with a brush on the material grid.
/// Handles drawing a smooth stroke between the previous mouse position and current position.
pub fn draw_brush_stroke(
    solver: &mut Solver,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    radius: i32,
    brush: BrushType,
) {
    // Determine distance between points
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let steps = std::cmp::max(dx, dy);
    
    if steps == 0 {
        draw_brush_circle(solver, x0, y0, radius, brush);
        return;
    }
    
    // Sample points along the line segment
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + t * (x1 - x0) as f32).round() as i32;
        let y = (y0 as f32 + t * (y1 - y0) as f32).round() as i32;
        draw_brush_circle(solver, x, y, radius, brush);
    }
}

/// Draws a circle with the specified brush type on the material grid
pub fn draw_brush_circle(solver: &mut Solver, cx: i32, cy: i32, radius: i32, brush: BrushType) {
    let w = solver.params.width as i32;
    let h = solver.params.height as i32;
    
    let x_start = std::cmp::max(0, cx - radius);
    let x_end = std::cmp::min(w - 1, cx + radius);
    let y_start = std::cmp::max(0, cy - radius);
    let y_end = std::cmp::min(h - 1, cy + radius);
    
    let r2 = radius * radius;
    
    for y in y_start..=y_end {
        for x in x_start..=x_end {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                let idx = ((y * w + x) * 4) as usize;
                match brush {
                    BrushType::Vacuum => {
                        // Reset to vacuum [eps_r = 1.0, sigma = 0.0, source = 0.0, pad = 0.0]
                        solver.material_data[idx] = 1.0;
                        solver.material_data[idx + 1] = 0.0;
                        solver.material_data[idx + 2] = 0.0;
                    }
                    BrushType::PEC => {
                        // Obstacle [eps_r = 0.0, sigma = 0.0, source = 0.0, pad = 0.0]
                        solver.material_data[idx] = 0.0;
                        solver.material_data[idx + 1] = 0.0;
                        solver.material_data[idx + 2] = 0.0;
                    }
                    BrushType::Glass => {
                        // Glass dielectric [eps_r = 2.25, sigma = 0.0, source = 0.0, pad = 0.0]
                        // Make sure we don't overwrite if it was a source, or maybe we do
                        solver.material_data[idx] = 2.25;
                        solver.material_data[idx + 1] = 0.0;
                        solver.material_data[idx + 2] = 0.0;
                    }
                    BrushType::Source => {
                        // Source map: sets blue channel to 1.0. Keeps eps_r = 1.0 for propagation.
                        solver.material_data[idx] = 1.0;
                        solver.material_data[idx + 1] = 0.0;
                        solver.material_data[idx + 2] = 1.0;
                    }
                }
            }
        }
    }
    
    solver.material_dirty = true;
}
