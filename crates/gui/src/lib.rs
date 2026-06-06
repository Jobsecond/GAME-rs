mod app;
mod notifier;
mod pages;
mod state;

pub use notifier::GuiNotifier;

pub fn run() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 700.0])
            .with_min_inner_size([760.0, 560.0]),
        ..Default::default()
    };

    if let Err(err) = eframe::run_native(
        "game-crabml - WAV to MIDI Extractor",
        options,
        Box::new(|cc| Ok(Box::new(app::GuiApp::new(cc)))),
    ) {
        eprintln!("failed to launch GUI: {err}");
    }
}
