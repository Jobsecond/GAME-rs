use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::pages;
use crate::state::AppState;

pub struct GuiApp {
    pub state: AppState,
}

impl GuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_egui(&cc.egui_ctx);
        Self {
            state: AppState::new(),
        }
    }
}

fn configure_egui(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::light());

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 10.0);
    style.spacing.button_padding = egui::vec2(12.0, 6.0);
    style.spacing.interact_size = egui::vec2(72.0, 28.0);
    style.visuals = egui::Visuals::light();
    ctx.set_global_style(style);

    ctx.set_fonts(build_system_fonts());
}

fn build_system_fonts() -> egui::FontDefinitions {
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

    if let Some(cjk_font) = cjk_font_data() {
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
    }

    fonts
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
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(18.0))
            .show_inside(ui, |ui| {
                pages::render_page(ui, &mut self.state, &ctx);
            });
    }
}
