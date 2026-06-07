pub mod config;
pub mod progress;
pub mod results;

use crate::state::AppState;

pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 120, 212);
pub const ACCENT_HOVER: egui::Color32 = egui::Color32::from_rgb(16, 110, 190);
pub const APP_BG: egui::Color32 = egui::Color32::from_rgb(243, 243, 243);
pub const SURFACE: egui::Color32 = egui::Color32::from_rgb(255, 255, 255);
pub const SUBTLE_SURFACE: egui::Color32 = egui::Color32::from_rgb(250, 250, 250);
pub const STROKE: egui::Color32 = egui::Color32::from_rgb(224, 224, 224);
pub const CONTROL_FILL: egui::Color32 = egui::Color32::from_rgb(255, 255, 255);
pub const CONTROL_STROKE: egui::Color32 = egui::Color32::from_rgb(138, 138, 138);
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(32, 32, 32);
pub const TEXT_SECONDARY: egui::Color32 = egui::Color32::from_rgb(96, 96, 96);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppPage {
    Config,
    Progress,
    Results,
}

pub fn render_page(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    match state.current_page {
        AppPage::Config => config::render(ui, state, ctx),
        AppPage::Progress => progress::render(ui, state),
        AppPage::Results => results::render(ui, state),
    }
}

pub fn page_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(24.0)
            .strong()
            .color(TEXT_PRIMARY),
    );
}

pub fn section_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(18.0)
            .strong()
            .color(TEXT_PRIMARY),
    );
}

pub fn section_frame() -> egui::Frame {
    egui::Frame::NONE
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0, STROKE))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(16, 14))
}

pub fn error_frame() -> egui::Frame {
    egui::Frame::NONE
        .fill(egui::Color32::from_rgb(253, 231, 233))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(196, 43, 28)))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(12, 10))
}

pub fn control_frame() -> egui::Frame {
    egui::Frame::NONE
        .fill(CONTROL_FILL)
        .stroke(egui::Stroke::new(1.0, CONTROL_STROKE))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(5, 3))
}

pub fn primary_button(text: &'static str) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text).color(egui::Color32::WHITE))
        .fill(ACCENT)
        .stroke(egui::Stroke::NONE)
        .corner_radius(4)
}
