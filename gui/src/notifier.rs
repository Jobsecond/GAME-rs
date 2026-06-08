use crossbeam::channel::{Receiver, Sender, unbounded};
use egui::Context;
use game_core::{CoreEvent, NotificationLevel, Notifier};

use crate::state::format_duration;

#[derive(Clone)]
pub struct GuiNotifier {
    tx: Sender<CoreEvent>,
    ctx: Context,
}

impl GuiNotifier {
    pub fn new(tx: Sender<CoreEvent>, ctx: Context) -> Self {
        Self { tx, ctx }
    }

    pub fn channel(ctx: Context) -> (Self, Receiver<CoreEvent>) {
        let (tx, rx) = unbounded();
        (Self::new(tx, ctx), rx)
    }

    pub fn format_event(event: &CoreEvent) -> String {
        match event {
            CoreEvent::ChunkPlan { total } => format!("planning {total} chunk(s)"),
            CoreEvent::Status { stage, message, .. } => format!("[{stage}] {message}"),
            CoreEvent::Progress {
                stage,
                current,
                total,
                detail,
                ..
            } => {
                let detail = detail
                    .as_deref()
                    .map(|value| format!(" ({value})"))
                    .unwrap_or_default();
                format!("[{stage}] {current}/{total}{detail}")
            }
            CoreEvent::Timing {
                stage,
                elapsed,
                detail,
                ..
            } => {
                let detail = detail
                    .as_deref()
                    .map(|value| format!(" {value}"))
                    .unwrap_or_default();
                format!("[{stage}] {}{detail}", format_duration(*elapsed))
            }
            CoreEvent::ModelLoaded { backend, elapsed } => {
                format!("model loaded ({backend}) in {}", format_duration(*elapsed))
            }
            CoreEvent::Message { level, message } => {
                format!("[{}] {message}", level_name(*level))
            }
        }
    }
}

impl Notifier for GuiNotifier {
    fn notify(&self, event: CoreEvent) {
        if self.tx.send(event).is_ok() {
            self.ctx.request_repaint();
        }
    }
}

fn level_name(level: NotificationLevel) -> &'static str {
    match level {
        NotificationLevel::Trace => "trace",
        NotificationLevel::Debug => "debug",
        NotificationLevel::Info => "info",
        NotificationLevel::Warn => "warn",
        NotificationLevel::Error => "error",
    }
}
