use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::pages;
use crate::state::AppState;

pub struct GuiApp {
    pub state: AppState,
}

impl GuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut state = AppState::new();
        if let Some(storage) = cc.storage {
            crate::state::load_persisted_settings(storage, &mut state);
        }
        let cjk_loaded = configure_egui(&cc.egui_ctx, state.theme);
        state.cjk_font_missing = !cjk_loaded;
        Self { state }
    }
}

/// Applies visuals/spacing/fonts. Returns whether a CJK fallback font loaded.
fn configure_egui(ctx: &egui::Context, theme: crate::state::ThemeChoice) -> bool {
    // Register a full custom palette for BOTH themes so every widget themes
    // correctly; egui then renders with whichever the preference resolves to.
    ctx.set_visuals_of(
        egui::Theme::Light,
        make_visuals(&pages::Palette::LIGHT, false),
    );
    ctx.set_visuals_of(egui::Theme::Dark, make_visuals(&pages::Palette::DARK, true));
    ctx.set_theme(theme.to_preference());

    // Theme-independent spacing and fonts apply to both styles.
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(12.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.interact_size = egui::vec2(72.0, 28.0);
        style.spacing.window_margin = egui::Margin::symmetric(18, 18);
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(22.0, egui::FontFamily::Proportional),
        );
    });

    let (fonts, cjk_loaded) = build_system_fonts();
    ctx.set_fonts(fonts);
    cjk_loaded
}

fn make_visuals(p: &pages::Palette, dark: bool) -> egui::Visuals {
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.panel_fill = p.app_bg;
    visuals.window_fill = p.surface;
    visuals.window_corner_radius = egui::CornerRadius::same(8);
    visuals.window_stroke = egui::Stroke::new(1.0, p.stroke);
    visuals.menu_corner_radius = egui::CornerRadius::same(4);
    visuals.faint_bg_color = p.subtle_surface;
    visuals.extreme_bg_color = p.control_fill;
    visuals.text_edit_bg_color = Some(p.control_fill);
    visuals.hyperlink_color = p.accent;
    visuals.selection.bg_fill = p.accent;
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.weak_text_color = Some(p.text_secondary);
    visuals.override_text_color = Some(p.text_primary);
    visuals.slider_trailing_fill = true;

    let corner = egui::CornerRadius::same(4);
    visuals.widgets.noninteractive.corner_radius = corner;
    visuals.widgets.noninteractive.bg_fill = p.surface;
    visuals.widgets.noninteractive.weak_bg_fill = p.subtle_surface;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, p.stroke);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, p.text_primary);

    visuals.widgets.inactive.corner_radius = corner;
    visuals.widgets.inactive.bg_fill = p.surface;
    visuals.widgets.inactive.weak_bg_fill = p.surface;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, p.control_stroke);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, p.text_primary);

    visuals.widgets.hovered.corner_radius = corner;
    visuals.widgets.hovered.bg_fill = p.hovered_bg;
    visuals.widgets.hovered.weak_bg_fill = p.hovered_bg;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, p.hovered_stroke);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, p.text_primary);

    visuals.widgets.active.corner_radius = corner;
    visuals.widgets.active.bg_fill = p.surface;
    visuals.widgets.active.weak_bg_fill = p.surface;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, p.accent_hover);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, p.text_primary);

    visuals.widgets.open = visuals.widgets.hovered;
    visuals
}

/// Builds the font set and reports whether a CJK fallback font was found.
/// When `false`, non-Latin glyphs (e.g. CJK file paths) render as tofu and the
/// UI surfaces a notice.
fn build_system_fonts() -> (egui::FontDefinitions, bool) {
    let mut fonts = egui::FontDefinitions::default();

    #[cfg(windows)]
    if let Some(message_font) = windows_message_font_data() {
        let font_name = "windows_message".to_owned();
        fonts
            .font_data
            .insert(font_name.clone(), Arc::new(message_font));
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, font_name);
    }

    let cjk_loaded = if let Some(cjk_font) = cjk_font_data() {
        let font_name = "system_cjk".to_owned();
        fonts
            .font_data
            .insert(font_name.clone(), Arc::new(cjk_font));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .entry(family)
                .or_default()
                .push(font_name.clone());
        }
        true
    } else {
        false
    };

    (fonts, cjk_loaded)
}

fn cjk_font_data() -> Option<egui::FontData> {
    let path = cjk_font_path()?;
    std::fs::read(path).ok().map(egui::FontData::from_owned)
}

fn cjk_font_path() -> Option<PathBuf> {
    cjk_font_candidates()
        .into_iter()
        .find(|path| path.is_file())
}

fn cjk_font_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if cfg!(target_os = "windows") {
        let windir = std::env::var_os("WINDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let fonts = windir.join("Fonts");
        paths.extend(
            [
                "Deng.ttf",
                "msyh.ttc",
                "simhei.ttf",
                "simsun.ttc",
                "NotoSansCJK-Regular.ttc",
            ]
            .into_iter()
            .map(|name| fonts.join(name)),
        );
    }

    paths.extend(
        [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        .into_iter()
        .map(Path::new)
        .map(Path::to_path_buf),
    );

    paths
}

#[cfg(windows)]
fn windows_message_font_data() -> Option<egui::FontData> {
    let face_name = windows_message_font_face_name()?;
    resolve_windows_font_face(&face_name)
}

#[cfg(windows)]
fn windows_message_font_face_name() -> Option<String> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        NONCLIENTMETRICSW, SPI_GETNONCLIENTMETRICS, SystemParametersInfoW,
    };

    let mut metrics = NONCLIENTMETRICSW {
        cbSize: std::mem::size_of::<NONCLIENTMETRICSW>() as u32,
        ..Default::default()
    };
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETNONCLIENTMETRICS,
            metrics.cbSize,
            std::ptr::addr_of_mut!(metrics).cast(),
            0,
        )
    };
    (ok != 0).then(|| utf16_nul_terminated(&metrics.lfMessageFont.lfFaceName))?
}

#[cfg(windows)]
fn resolve_windows_font_face(face_name: &str) -> Option<egui::FontData> {
    let fonts_dir = windows_fonts_dir();
    known_windows_font_files(face_name)
        .into_iter()
        .chain(registry_font_files(face_name))
        .map(|file_name| resolve_windows_font_path(&fonts_dir, &file_name))
        .find_map(|path| std::fs::read(path).ok())
        .map(egui::FontData::from_owned)
}

#[cfg(windows)]
fn known_windows_font_files(face_name: &str) -> Vec<String> {
    let normalized = normalize_font_name(face_name);
    match normalized.as_str() {
        "segoeui" | "segoeuihistoric" => vec!["segoeui.ttf".to_owned()],
        "microsoftyaheiui" | "microsoftyahei" => vec!["msyh.ttc".to_owned()],
        "dengxian" => vec!["Deng.ttf".to_owned()],
        "simsun" => vec!["simsun.ttc".to_owned()],
        "simhei" => vec!["simhei.ttf".to_owned()],
        _ => Vec::new(),
    }
}

#[cfg(windows)]
fn registry_font_files(face_name: &str) -> Vec<String> {
    use windows_sys::Win32::Foundation::{ERROR_NO_MORE_ITEMS, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_LOCAL_MACHINE, KEY_READ, KEY_WOW64_64KEY, REG_EXPAND_SZ, REG_SZ, RegCloseKey,
        RegEnumValueW, RegOpenKeyExW,
    };

    let mut key: HKEY = std::ptr::null_mut();
    let subkey = wide_null(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts");
    let open = unsafe {
        RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            0,
            KEY_READ | KEY_WOW64_64KEY,
            &mut key,
        )
    };
    if open != ERROR_SUCCESS {
        return Vec::new();
    }

    let face = normalize_font_name(face_name);
    let mut matches = Vec::new();
    let mut index = 0;
    loop {
        let mut value_name = vec![0u16; 256];
        let mut value_name_len = value_name.len() as u32;
        let mut value_type = 0u32;
        let mut data = vec![0u8; 1024];
        let mut data_len = data.len() as u32;
        let status = unsafe {
            RegEnumValueW(
                key,
                index,
                value_name.as_mut_ptr(),
                &mut value_name_len,
                std::ptr::null(),
                &mut value_type,
                data.as_mut_ptr(),
                &mut data_len,
            )
        };

        if status == ERROR_NO_MORE_ITEMS {
            break;
        }
        index += 1;
        if status != ERROR_SUCCESS || !matches!(value_type, REG_SZ | REG_EXPAND_SZ) {
            continue;
        }

        let value_name = String::from_utf16_lossy(&value_name[..value_name_len as usize]);
        if !normalize_registry_font_value_name(&value_name).starts_with(&face) {
            continue;
        }

        if let Some(file_name) = registry_string_data(&data[..data_len as usize]) {
            matches.push((registry_font_priority(&value_name), file_name));
        }
    }

    unsafe {
        RegCloseKey(key);
    }

    matches.sort_by_key(|(priority, _)| *priority);
    matches
        .into_iter()
        .map(|(_, file_name)| file_name)
        .collect()
}

#[cfg(windows)]
fn windows_fonts_dir() -> PathBuf {
    std::env::var_os("WINDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("Fonts")
}

#[cfg(windows)]
fn resolve_windows_font_path(fonts_dir: &Path, file_name: &str) -> PathBuf {
    let path = Path::new(file_name);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        fonts_dir.join(path)
    }
}

#[cfg(windows)]
fn registry_string_data(data: &[u8]) -> Option<String> {
    let words = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    utf16_nul_terminated(&words)
}

#[cfg(windows)]
fn registry_font_priority(value_name: &str) -> u8 {
    let lower = value_name.to_ascii_lowercase();
    let mut priority = 0;
    for marker in ["bold", "italic", "oblique", "light", "semibold", "black"] {
        if lower.contains(marker) {
            priority += 1;
        }
    }
    priority
}

#[cfg(windows)]
fn normalize_registry_font_value_name(value: &str) -> String {
    let name = value.split_once('(').map(|(name, _)| name).unwrap_or(value);
    normalize_font_name(name)
}

#[cfg(windows)]
fn normalize_font_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn utf16_nul_terminated(value: &[u16]) -> Option<String> {
    let end = value.iter().position(|ch| *ch == 0).unwrap_or(value.len());
    (end > 0).then(|| String::from_utf16_lossy(&value[..end]))
}

impl eframe::App for GuiApp {
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        visuals.panel_fill.to_normalized_gamma_f32()
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        crate::state::save_persisted_settings(storage, &self.state);
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.state.poll_background_work();
        self.state.drain_events();
        self.state.check_completion();
        if self.state.is_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        apply_dropped_files(&ctx, &mut self.state);
        let panel_fill = ui.visuals().panel_fill;
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(panel_fill)
                    .inner_margin(egui::Margin::symmetric(22, 22)),
            )
            .show_inside(ui, |ui| {
                pages::render_page(ui, &mut self.state, &ctx);
            });
        paint_drop_overlay(&ctx);
    }
}

/// Applies any files dropped onto the window this frame, regardless of which
/// page is showing. Path-typed files route to the matching config field by
/// extension (see [`AppState::apply_dropped_path`]).
fn apply_dropped_files(ctx: &egui::Context, state: &mut AppState) {
    let dropped = ctx.input(|input| input.raw.dropped_files.clone());
    for file in dropped {
        if let Some(path) = file.path {
            state.apply_dropped_path(&path);
        }
    }
}

/// Dims the window and shows guidance while files are dragged over it.
fn paint_drop_overlay(ctx: &egui::Context) {
    if ctx.input(|input| input.raw.hovered_files.is_empty()) {
        return;
    }
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("drop_overlay"),
    ));
    painter.rect_filled(screen, 0.0, egui::Color32::from_black_alpha(150));
    painter.text(
        screen.center(),
        egui::Align2::CENTER_CENTER,
        "Drop a .gguf model, .wav audio, or output file to load",
        egui::FontId::proportional(22.0),
        egui::Color32::WHITE,
    );
}
