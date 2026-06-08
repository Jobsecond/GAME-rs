pub mod config;
pub mod progress;
pub mod results;

use egui::Color32;

use crate::state::AppState;

/// A full set of theme-dependent colors. Two instances exist — [`Palette::LIGHT`]
/// and [`Palette::DARK`] — used to build egui's per-theme `Visuals` in
/// `configure_egui`. Most render code reads themed colors back from
/// `ui.visuals()`; the helpers in this module centralize that mapping.
#[derive(Clone, Copy)]
pub struct Palette {
    pub accent: Color32,
    pub accent_hover: Color32,
    pub app_bg: Color32,
    pub surface: Color32,
    pub subtle_surface: Color32,
    pub stroke: Color32,
    pub control_fill: Color32,
    pub control_stroke: Color32,
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub hovered_bg: Color32,
    pub hovered_stroke: Color32,
    pub track_bg: Color32,
}

impl Palette {
    pub const LIGHT: Palette = Palette {
        accent: Color32::from_rgb(0, 120, 212),
        accent_hover: Color32::from_rgb(16, 110, 190),
        app_bg: Color32::from_rgb(243, 243, 243),
        surface: Color32::from_rgb(255, 255, 255),
        subtle_surface: Color32::from_rgb(250, 250, 250),
        stroke: Color32::from_rgb(224, 224, 224),
        control_fill: Color32::from_rgb(255, 255, 255),
        control_stroke: Color32::from_rgb(138, 138, 138),
        text_primary: Color32::from_rgb(32, 32, 32),
        // Darkened from 96,96,96 toward WCAG AA contrast on the light surface.
        text_secondary: Color32::from_rgb(88, 88, 88),
        hovered_bg: Color32::from_rgb(250, 250, 250),
        hovered_stroke: Color32::from_rgb(96, 96, 96),
        track_bg: Color32::from_rgb(230, 230, 230),
    };

    pub const DARK: Palette = Palette {
        accent: Color32::from_rgb(96, 175, 255),
        accent_hover: Color32::from_rgb(140, 195, 255),
        app_bg: Color32::from_rgb(32, 32, 32),
        surface: Color32::from_rgb(43, 43, 43),
        subtle_surface: Color32::from_rgb(50, 50, 50),
        stroke: Color32::from_rgb(64, 64, 64),
        control_fill: Color32::from_rgb(56, 56, 56),
        control_stroke: Color32::from_rgb(120, 120, 120),
        text_primary: Color32::from_rgb(240, 240, 240),
        text_secondary: Color32::from_rgb(184, 184, 184),
        hovered_bg: Color32::from_rgb(58, 58, 58),
        hovered_stroke: Color32::from_rgb(150, 150, 150),
        track_bg: Color32::from_rgb(64, 64, 64),
    };
}

/// Accent color for the active theme (progress fill, primary button).
pub fn accent(ui: &egui::Ui) -> Color32 {
    ui.visuals().hyperlink_color
}

/// Background of the unfilled portion of a progress track for the active theme.
pub fn track_bg(ui: &egui::Ui) -> Color32 {
    if ui.visuals().dark_mode {
        Palette::DARK.track_bg
    } else {
        Palette::LIGHT.track_bg
    }
}

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
    ui.label(egui::RichText::new(text).size(24.0).strong());
}

pub fn section_title(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(18.0).strong());
}

pub fn section_frame(ui: &egui::Ui) -> egui::Frame {
    egui::Frame::NONE
        .fill(ui.visuals().window_fill)
        .stroke(egui::Stroke::new(1.0, ui.visuals().window_stroke.color))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(16, 14))
}

pub fn error_frame(ui: &egui::Ui) -> egui::Frame {
    let (fill, stroke) = if ui.visuals().dark_mode {
        (
            egui::Color32::from_rgb(64, 26, 28),
            egui::Color32::from_rgb(232, 86, 76),
        )
    } else {
        (
            egui::Color32::from_rgb(253, 231, 233),
            egui::Color32::from_rgb(196, 43, 28),
        )
    };
    egui::Frame::NONE
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(12, 10))
}

pub fn control_frame(ui: &egui::Ui) -> egui::Frame {
    egui::Frame::NONE
        .fill(ui.visuals().extreme_bg_color)
        .stroke(egui::Stroke::new(
            1.0,
            ui.visuals().widgets.inactive.bg_stroke.color,
        ))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(5, 3))
}

pub fn primary_button(ui: &egui::Ui, text: &'static str) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text).color(egui::Color32::WHITE))
        .fill(accent(ui))
        .stroke(egui::Stroke::NONE)
        .corner_radius(4)
}
