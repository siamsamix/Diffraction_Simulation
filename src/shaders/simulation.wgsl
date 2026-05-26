struct SimParams {
    dt: f32,
    dx: f32,
    dy: f32,
    time: f32,
    width: u32,
    height: u32,
    source_type: u32,       // 0: CW, 1: Pulse
    source_freq: f32,
    source_amp: f32,
    source_phase: f32,
    source_angle: f32,      // in radians
    source_width: f32,
    pulse_width: f32,
    pulse_delay: f32,
    render_mode: u32,       // 0: Field, 1: Intensity
    display_contrast: f32,  // multiplier for display scaling
    clear_sim: u32,         // 1 if we should clear
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SimParams;

// ==========================================
// GROUP 1: H-Field Update
// ==========================================
@group(1) @binding(0) var ez_read_h: texture_2d<f32>;
@group(1) @binding(1) var h_read_h: texture_2d<f32>;
@group(1) @binding(2) var h_write_h: texture_storage_2d<rg32float, write>;

// Compute Shader to update Magnetic Fields (Hx, Hy)
@compute @workgroup_size(16, 16, 1)
fn update_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (id.x >= w || id.y >= h) {
        return;
    }
    
    let pos = vec2<i32>(i32(id.x), i32(id.y));
    
    if (params.clear_sim == 1u) {
        textureStore(h_write_h, pos, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Staggered Hx(i, j + 0.5): relies on Ez(i, j+1) - Ez(i, j)
    let ez_curr = textureLoad(ez_read_h, pos, 0).r;
    
    // Clamp coordinates to prevent reading out of bounds
    let pos_y_up = vec2<i32>(pos.x, min(pos.y + 1, i32(h) - 1));
    let ez_up = textureLoad(ez_read_h, pos_y_up, 0).r;

    // Staggered Hy(i + 0.5, j): relies on Ez(i+1, j) - Ez(i, j)
    let pos_x_right = vec2<i32>(min(pos.x + 1, i32(w) - 1), pos.y);
    let ez_right = textureLoad(ez_read_h, pos_x_right, 0).r;

    let h_curr = textureLoad(h_read_h, pos, 0).rg;

    let mu = 1.0;
    let hx_new = h_curr.x - (params.dt / (mu * params.dy)) * (ez_up - ez_curr);
    let hy_new = h_curr.y + (params.dt / (mu * params.dx)) * (ez_right - ez_curr);

    textureStore(h_write_h, pos, vec4<f32>(hx_new, hy_new, 0.0, 0.0));
}

// ==========================================
// GROUP 2: E-Field Update
// ==========================================
@group(1) @binding(3) var ez_read_e: texture_2d<f32>;
@group(1) @binding(4) var ez_write_e: texture_storage_2d<r32float, write>;
@group(1) @binding(5) var h_read_e: texture_2d<f32>; // This will be the *new* H texture
@group(1) @binding(6) var material_e: texture_2d<f32>;

// Compute Shader to update Electric Field (Ez)
@compute @workgroup_size(16, 16, 1)
fn update_e(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (id.x >= w || id.y >= h) {
        return;
    }
    
    let pos = vec2<i32>(i32(id.x), i32(id.y));

    if (params.clear_sim == 1u) {
        textureStore(ez_write_e, pos, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Load material properties:
    // R = epsilon_r (dielectric permittivity relative)
    // G = conductivity (sigma)
    // B = source map (1.0 if source pixel, 0.0 otherwise)
    let mat = textureLoad(material_e, pos, 0);
    let eps_r = mat.r;
    let sigma_mat = mat.g;
    let is_source = mat.b;

    // PEC (Perfect Electric Conductor): eps_r is negative or near 0
    if (eps_r < 0.01) {
        textureStore(ez_write_e, pos, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Ez update uses:
    // Ez_new = Ez_old + dt/eps * ( (Hy(i, j) - Hy(i-1, j))/dx - (Hx(i, j) - Hx(i, j-1))/dy - sigma * Ez )
    let ez_curr = textureLoad(ez_read_e, pos, 0).r;

    let h_curr = textureLoad(h_read_e, pos, 0).rg;
    
    let pos_x_left = vec2<i32>(max(pos.x - 1, 0), pos.y);
    let h_left = textureLoad(h_read_e, pos_x_left, 0).rg;
    
    let pos_y_down = vec2<i32>(pos.x, max(pos.y - 1, 0));
    let h_down = textureLoad(h_read_e, pos_y_down, 0).rg;

    // Boundary damping layer (absorbing boundary)
    // 32-pixel thick border
    let border = 32u;
    var sigma_border = 0.0;
    
    let dist_x = min(id.x, w - 1u - id.x);
    let dist_y = min(id.y, h - 1u - id.y);
    let min_dist = min(dist_x, dist_y);
    
    if (min_dist < border) {
        let norm = 1.0 - (f32(min_dist) / f32(border));
        // High order absorption profile
        sigma_border = 0.8 * pow(norm, 3.5);
    }
    
    let total_sigma = sigma_mat + sigma_border;
    let eps = eps_r; // eps_0 = 1.0

    // Yee cell update coefficients
    let c_eze = (1.0 - (total_sigma * params.dt) / (2.0 * eps)) / (1.0 + (total_sigma * params.dt) / (2.0 * eps));
    let c_ezh = (params.dt / eps) / (1.0 + (total_sigma * params.dt) / (2.0 * eps));

    let dy_hy = (h_curr.y - h_left.y) / params.dx;
    let dx_hx = (h_curr.x - h_down.x) / params.dy;

    var ez_new = c_eze * ez_curr + c_ezh * (dy_hy - dx_hx);

    // Source injection
    if (is_source > 0.01) {
        let omega = 2.0 * 3.14159265 * params.source_freq;
        var signal = 0.0;

        if (params.source_type == 0u) {
            // Continuous Wave (Sinusoidal)
            signal = params.source_amp * sin(omega * params.time + params.source_phase);
        } else if (params.source_type == 1u) {
            // Gaussian Pulse
            let t_offset = params.pulse_delay;
            let sigma_t = params.pulse_width;
            let exponent = -pow(params.time - t_offset, 2.0) / (2.0 * pow(sigma_t, 2.0));
            signal = params.source_amp * exp(exponent) * sin(omega * params.time);
        }

        // Hard sourcing scaled by spatial amplitude weight
        ez_new = signal * is_source;
    }

    textureStore(ez_write_e, pos, vec4<f32>(ez_new, 0.0, 0.0, 0.0));
}

// ==========================================
// GROUP 3: Display Render
// ==========================================
@group(1) @binding(7) var ez_read_r: texture_2d<f32>;
@group(1) @binding(8) var material_r: texture_2d<f32>;
@group(1) @binding(9) var display_out_r: texture_storage_2d<rgba8unorm, write>;

// Compute Shader to Render the Field & Materials to Display Texture
@compute @workgroup_size(16, 16, 1)
fn render_display(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (id.x >= w || id.y >= h) {
        return;
    }

    let pos = vec2<i32>(i32(id.x), i32(id.y));
    let ez = textureLoad(ez_read_r, pos, 0).r;
    let mat = textureLoad(material_r, pos, 0);

    let eps_r = mat.r;
    let sigma = mat.g;
    let is_source = mat.b;

    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    let val = ez * params.display_contrast;

    if (params.render_mode == 0u) {
        // Mode 0: Field (Cyan - Black - Orange colormap)
        if (val >= 0.0) {
            let t = min(val, 1.0);
            rgb = mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(1.0, 0.45, 0.05), t);
        } else {
            let t = min(-val, 1.0);
            rgb = mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.05, 0.6, 1.0), t);
        }
    } else {
        // Mode 1: Intensity (Fire colormap)
        let intensity = min(val * val, 1.0);
        if (intensity < 0.3) {
            let t = intensity / 0.3;
            rgb = mix(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.6, 0.0, 0.05), t);
        } else if (intensity < 0.7) {
            let t = (intensity - 0.3) / 0.4;
            rgb = mix(vec3<f32>(0.6, 0.0, 0.05), vec3<f32>(1.0, 0.5, 0.0), t);
        } else {
            let t = (intensity - 0.7) / 0.3;
            rgb = mix(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(1.0, 0.95, 0.7), t);
        }
    }

    // Material overlays
    if (eps_r < 0.01) {
        // Metal/PEC obstacles
        rgb = mix(rgb, vec3<f32>(0.18, 0.19, 0.22), 0.85);
    } else if (eps_r > 1.01) {
        // Glass refractors
        let glass_color = vec3<f32>(0.1, 0.35, 0.5);
        rgb = mix(rgb, glass_color, 0.25);
        let is_edge = check_is_material_edge(pos, w, h);
        if (is_edge) {
            rgb = mix(rgb, vec3<f32>(0.3, 0.7, 0.9), 0.5);
        }
    }

    // Add source highlight
    if (is_source > 0.01) {
        rgb = mix(rgb, vec3<f32>(1.0, 0.85, 0.2), 0.3);
    }

    textureStore(display_out_r, pos, vec4<f32>(rgb, 1.0));
}

fn check_is_material_edge(pos: vec2<i32>, w: u32, h: u32) -> bool {
    let current_eps = textureLoad(material_r, pos, 0).r;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = pos.x + dx;
            let ny = pos.y + dy;
            if (nx >= 0 && nx < i32(w) && ny >= 0 && ny < i32(h)) {
                let neighbor_eps = textureLoad(material_r, vec2<i32>(nx, ny), 0).r;
                if (abs(neighbor_eps - current_eps) > 0.05) {
                    return true;
                }
            }
        }
    }
    return false;
}
