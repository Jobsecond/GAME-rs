use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::notify::{CoreEvent, NotificationLevel, emit};

static CPU_PROFILING_ENABLED: OnceLock<bool> = OnceLock::new();
static CPU_PROFILER_STATE: OnceLock<Mutex<CpuProfilerState>> = OnceLock::new();

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

    let mut state = profiler_state().lock().unwrap();
    state.active = true;
    state.label.clear();
    state.label.push_str(label);
    state.scopes.clear();
    state.ops.clear();
    drop(state);

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
        let mut state = profiler_state().lock().unwrap_or_else(|p| p.into_inner());
        log_profile_section("scope", &state.label, elapsed, &state.scopes);
        log_profile_section("op", &state.label, elapsed, &state.ops);
        state.active = false;
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let elapsed = self.started_at.elapsed();
        let mut state = profiler_state().lock().unwrap_or_else(|p| p.into_inner());
        let map = match self.kind {
            ProfileKind::Scope => &mut state.scopes,
            ProfileKind::Op => &mut state.ops,
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
    }
}

fn profiler_state() -> &'static Mutex<CpuProfilerState> {
    CPU_PROFILER_STATE.get_or_init(|| Mutex::new(CpuProfilerState::default()))
}

fn is_session_active() -> bool {
    profiler_state().lock().unwrap().active
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
