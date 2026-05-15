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

#[derive(Debug, Clone, PartialEq)]
pub enum CoreEvent {
    Status {
        stage: &'static str,
        message: String,
    },
    Progress {
        stage: &'static str,
        current: usize,
        total: usize,
        detail: Option<String>,
    },
    Timing {
        stage: &'static str,
        elapsed: Duration,
        detail: Option<String>,
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
    static CURRENT_NOTIFIER: RefCell<Vec<(usize, usize)>> = RefCell::new(Vec::new());
}

pub(crate) fn with_notifier<T>(notifier: &dyn Notifier, f: impl FnOnce() -> T) -> T {
    let ptr = notifier as *const dyn Notifier;
    let erased = unsafe { mem::transmute::<*const dyn Notifier, (usize, usize)>(ptr) };
    CURRENT_NOTIFIER.with(|current| current.borrow_mut().push(erased));
    let result = f();
    CURRENT_NOTIFIER.with(|current| {
        current.borrow_mut().pop();
    });
    result
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
