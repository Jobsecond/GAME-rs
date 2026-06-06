pub mod config;
pub mod progress;
pub mod results;

use crate::state::AppState;

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
