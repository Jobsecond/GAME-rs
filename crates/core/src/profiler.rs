use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::notify::{CoreEvent, NotificationLevel, emit};

static CPU_PROFILING_ENABLED: OnceLock<bool> = OnceLock::new();

// Per-thread profiling state. Using thread_local! ensures concurrent chunks on different
// Rayon threads don't overwrite each other's profiling data. The session activation state
// is still coordinated via the single `session_active` cell.
thread_local! {
    static CPU_PROFILER_STATE: std::cell::RefCell<CpuProfilerState> = std::cell::RefCell::new(CpuProfilerState::default());
}

#[derive(Default)]
struct CpuProfilerState {
    active: bool,
    label: String,
    scopes: HashMap<ProfileKey, ProfileStats>,
    ops: HashMap<ProfileKey, ProfileStats>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ProfileKey {
    label: &'static str,
    details: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct ProfileStats {
    calls: u64,
    total: Duration,
    max: Duration,
}

pub(crate) struct CpuProfileSession {
    active: bool,
    started_at: Instant,
}

pub(crate) struct ProfileScope {
    active: bool,
    kind: ProfileKind,
    label: &'static str,
    details: String,
    started_at: Instant,
}

#[derive(Clone, Copy)]
enum ProfileKind {
    Scope,
    Op,
}

pub(crate) fn cpu_profiling_enabled() -> bool {
    *CPU_PROFILING_ENABLED.get_or_init(|| {
        std::env::var("GAME_CPU_PROFILE")
            .ok()
            .map(|value| {
                let value = value.trim();
                !(value.is_empty()
                    || value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off")
                    || value.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

pub(crate) fn cpu_profile_session(label: &'static str) -> CpuProfileSession {
    if !cpu_profiling_enabled() {
        return CpuProfileSession {
            active: false,
            started_at: Instant::now(),
        };
    }

    CPU_PROFILER_STATE.with(|state| {
        let mut s = state.borrow_mut();
        s.active = true;
        s.label.clear();
        s.label.push_str(label);
        s.scopes.clear();
        s.ops.clear();
    });

    CpuProfileSession {
        active: true,
        started_at: Instant::now(),
    }
}

pub(crate) fn scope(label: &'static str, details: impl Into<String>) -> ProfileScope {
    if !cpu_profiling_enabled() || !is_session_active() {
        return ProfileScope {
            active: false,
            kind: ProfileKind::Scope,
            label,
            details: String::new(),
            started_at: Instant::now(),
        };
    }

    ProfileScope {
        active: true,
        kind: ProfileKind::Scope,
        label,
        details: details.into(),
        started_at: Instant::now(),
    }
}

pub(crate) fn scope_with(label: &'static str, details: impl FnOnce() -> String) -> ProfileScope {
    if !cpu_profiling_enabled() || !is_session_active() {
        return ProfileScope {
            active: false,
            kind: ProfileKind::Scope,
            label,
            details: String::new(),
            started_at: Instant::now(),
        };
    }

    ProfileScope {
        active: true,
        kind: ProfileKind::Scope,
        label,
        details: details(),
        started_at: Instant::now(),
    }
}

pub(crate) fn op_scope_with(label: &'static str, details: impl FnOnce() -> String) -> ProfileScope {
    if !cpu_profiling_enabled() || !is_session_active() {
        return ProfileScope {
            active: false,
            kind: ProfileKind::Op,
            label,
            details: String::new(),
            started_at: Instant::now(),
        };
    }

    ProfileScope {
        active: true,
        kind: ProfileKind::Op,
        label,
        details: details(),
        started_at: Instant::now(),
    }
}

impl Drop for CpuProfileSession {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let elapsed = self.started_at.elapsed();
        CPU_PROFILER_STATE.with(|state| {
            let mut s = state.borrow_mut();
            log_profile_section("scope", &s.label, elapsed, &s.scopes);
            log_profile_section("op", &s.label, elapsed, &s.ops);
            s.active = false;
        });
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let elapsed = self.started_at.elapsed();
        CPU_PROFILER_STATE.with(|state| {
            let mut s = state.borrow_mut();
            let map = match self.kind {
                ProfileKind::Scope => &mut s.scopes,
                ProfileKind::Op => &mut s.ops,
            };
            let entry = map
                .entry(ProfileKey {
                    label: self.label,
                    details: self.details.clone(),
                })
                .or_default();
            entry.calls += 1;
            entry.total += elapsed;
            entry.max = entry.max.max(elapsed);
        });
    }
}

fn is_session_active() -> bool {
    CPU_PROFILER_STATE.with(|state| state.borrow().active)
}

fn log_profile_section(
    kind: &str,
    label: &str,
    elapsed: Duration,
    entries: &HashMap<ProfileKey, ProfileStats>,
) {
    if entries.is_empty() {
        emit(CoreEvent::Message {
            level: NotificationLevel::Info,
            message: format!(
                "cpu-profile {} `{}`: total={:.3}s entries=0",
                kind,
                label,
                elapsed.as_secs_f64()
            ),
        });
        return;
    }

    let mut rows = entries.iter().collect::<Vec<_>>();
    rows.sort_by(|(_, lhs), (_, rhs)| {
        rhs.total
            .cmp(&lhs.total)
            .then_with(|| rhs.calls.cmp(&lhs.calls))
    });

    let limit = profile_top_n();
    emit(CoreEvent::Message {
        level: NotificationLevel::Info,
        message: format!(
            "cpu-profile {} `{}`: total={:.3}s entries={} top={}",
            kind,
            label,
            elapsed.as_secs_f64(),
            rows.len(),
            rows.len().min(limit)
        ),
    });
    for (key, stats) in rows.into_iter().take(limit) {
        let avg = stats.total.as_secs_f64() / stats.calls as f64;
        emit(CoreEvent::Message {
            level: NotificationLevel::Info,
            message: format!(
                "  {} calls={} total={:.3}s avg={:.6}s max={:.6}s {}",
                key.label,
                stats.calls,
                stats.total.as_secs_f64(),
                avg,
                stats.max.as_secs_f64(),
                key.details
            ),
        });
    }
}

fn profile_top_n() -> usize {
    std::env::var("GAME_CPU_PROFILE_TOP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(20)
}
