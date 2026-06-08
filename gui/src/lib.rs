mod app;
mod notifier;
mod pages;
mod state;

pub use notifier::GuiNotifier;

pub fn run() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 700.0])
            .with_min_inner_size([760.0, 560.0])
            .with_icon(std::sync::Arc::new(app_icon())),
        ..Default::default()
    };

    if let Err(err) = eframe::run_native(
        "GAME - WAV to MIDI Extractor",
        options,
        Box::new(|cc| Ok(Box::new(app::GuiApp::new(cc)))),
    ) {
        eprintln!("failed to launch GUI: {err}");
    }
}

/// A simple procedurally-generated window/taskbar icon: a white circle (note
/// head) on the brand-blue background. Avoids shipping a binary asset.
fn app_icon() -> egui::IconData {
    const SIZE: usize = 32;
    const ACCENT: [u8; 3] = [0, 120, 212];
    let center = (SIZE as f32 - 1.0) / 2.0;
    let radius = SIZE as f32 * 0.30;
    let mut rgba = Vec::with_capacity(SIZE * SIZE * 4);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            if dx * dx + dy * dy <= radius * radius {
                rgba.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                rgba.extend_from_slice(&[ACCENT[0], ACCENT[1], ACCENT[2], 255]);
            }
        }
    }
    egui::IconData {
        rgba,
        width: SIZE as u32,
        height: SIZE as u32,
    }
}
