use std::cell::RefCell;
use std::mem;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Identifies which audio chunk an event belongs to.
///
/// This is the authoritative, machine-readable chunk attribution. Frontends
/// should read it directly rather than parsing chunk identity out of the
/// human-readable `message`/`detail` strings (which remain purely decorative
/// for log display). The `game-service` layer fills this in for every event it
/// forwards from a per-chunk inference (see its `PrefixedNotifier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkContext {
    /// Zero-based index of the chunk among all chunks.
    pub index: usize,
    /// Total number of chunks after all splitting.
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoreEvent {
    /// One-time structural announcement of how many chunks will be processed,
    /// emitted once before any per-chunk work begins. Lets frontends size their
    /// progress UI without parsing the `extract_infer` status text.
    ChunkPlan { total: usize },
    Status {
        stage: &'static str,
        message: String,
        /// Set when this event belongs to a specific chunk; `None` for
        /// whole-run stages (model load, audio prep, etc.).
        chunk: Option<ChunkContext>,
    },
    Progress {
        stage: &'static str,
        /// For `d3pm_step`, the current step number (1-based). Never carries
        /// chunk identity — that lives in `chunk`.
        current: usize,
        /// For `d3pm_step`, the total step count.
        total: usize,
        detail: Option<String>,
        chunk: Option<ChunkContext>,
    },
    Timing {
        stage: &'static str,
        elapsed: Duration,
        detail: Option<String>,
        chunk: Option<ChunkContext>,
    },
    ModelLoaded {
        backend: &'static str,
        elapsed: Duration,
    },
    Message {
        level: NotificationLevel,
        message: String,
    },
}

pub trait Notifier: Send + Sync {
    fn notify(&self, event: CoreEvent);
}

#[derive(Default)]
pub struct NullNotifier;

impl Notifier for NullNotifier {
    fn notify(&self, _event: CoreEvent) {}
}

thread_local! {
    static CURRENT_NOTIFIER: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn with_notifier<T>(notifier: &dyn Notifier, f: impl FnOnce() -> T) -> T {
    let ptr = notifier as *const dyn Notifier;
    let erased = unsafe { mem::transmute::<*const dyn Notifier, (usize, usize)>(ptr) };
    CURRENT_NOTIFIER.with(|current| current.borrow_mut().push(erased));
    struct PopGuard;
    impl Drop for PopGuard {
        fn drop(&mut self) {
            CURRENT_NOTIFIER.with(|current| {
                current.borrow_mut().pop();
            });
        }
    }
    let _guard = PopGuard;
    f()
}

pub(crate) fn emit(event: CoreEvent) {
    CURRENT_NOTIFIER.with(|current| {
        if let Some(erased) = current.borrow().last().copied() {
            let ptr = unsafe { mem::transmute::<(usize, usize), *const dyn Notifier>(erased) };
            // SAFETY: `with_notifier` only stores the pointer for the dynamic extent of `f`.
            // Events are emitted synchronously on the same thread before that scope exits.
            unsafe { (&*ptr).notify(event) };
        }
    });
}
