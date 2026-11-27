//! mac: threads that make a heap sync's changed slots beside the thread
//! that syncs (`VICTORY_HEAP_SYNC_THREADS=<n>`, off by default; only with
//! `VICTORY_HEAP_LOCKS=1`, `super::slots`).
//!
//! A heap sync makes again every slot whose record the title rewrote as
//! bytes, 25,000 to 40,000 a world frame, one after the other on the thread
//! that submits. With the heap's one slot lock, more threads only stood in
//! line for it (tried on 26 November: four threads, no gain). With locks of
//! their own (`VICTORY_HEAP_LOCKS=1`) the slots of a pass can be made side
//! by side: the idea is avatarmanz's (6c17ce9, `VICTORY_VIEW_THREADS`).
//!
//! The threads are kept: a pass is one 64 KiB page of the title's heap, a
//! few hundred of them a frame, and a thread started for each would cost
//! more than the pass. [`Pool::run`] hands the workers a job (`f(i)` for
//! each `i` below `n`), works on it itself, and returns when every index
//! is done and no worker is inside the job any more. A worker that wakes
//! after the job is done finds it closed and goes back to wait, so a pass
//! never waits for a thread that was slow to wake.
//!
//! What a worker does for the syncing thread it does in that thread's
//! name: the caller's closure sets the worker's thread marks (a round ahead
//! of a submission, the submission lock held, a heap sync) for each index
//! (`HostHeap::sync_slots`).
//!
//! Plain std, outside the Windows-only heap, so that the tests run on any
//! host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// A pass with fewer changed slots than this is made by the syncing thread
/// alone: waking the others costs more than they save (a slot takes 2 to 4
/// microseconds to make, a wake under Wine some tens).
pub(crate) const WORTH: usize = 96;

/// The most threads the switch can ask for.
const MOST: usize = 8;

/// The job of a round: `f(i)` for each `i` below `n`. The reference is the
/// caller's, made to look lasting: it is read only while the round is open,
/// and [`Pool::run`] returns only once the round is closed and no worker is
/// inside it.
#[derive(Clone, Copy)]
struct Job {
    f: &'static (dyn Fn(usize) + Sync),
    n: usize,
}

#[derive(Default)]
struct State {
    /// The open round's job, and the number of the round last opened.
    job: Option<Job>,
    round: u64,
    /// Workers inside the open round's job.
    inside: usize,
    /// A worker's `f` panicked in this round.
    panicked: bool,
    /// The pool is going: workers end.
    ending: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    /// A round opened (or the pool ends).
    opened: Condvar,
    /// A worker left the job.
    left: Condvar,
    /// The next index of the open round to take.
    next: AtomicUsize,
}

impl Shared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Takes indices of `job` until none is left.
    fn work(&self, job: Job) {
        loop {
            let i = self.next.fetch_add(1, Relaxed);
            if i >= job.n {
                break;
            }
            (job.f)(i);
        }
    }
}

/// Worker threads waiting for rounds of work ([`Pool::run`]).
pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: usize,
    /// One round at a time.
    rounds: Mutex<()>,
}

impl Pool {
    /// A pool of `threads` threads in all: the caller of [`Pool::run`] and
    /// `threads - 1` workers (fewer if the system starts fewer).
    pub(crate) fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared::default());
        let mut workers = 0;
        for i in 1..threads.max(1) {
            let shared = shared.clone();
            let started = std::thread::Builder::new().name(format!("victory-heap-sync-{i}")).spawn(move || {
                let mut state = shared.state();
                // The round this thread last worked in: its indices are all
                // taken, so it is not entered again while it stays open.
                let mut worked = 0;
                loop {
                    if state.ending {
                        return;
                    }
                    let job = match state.job {
                        Some(job) if state.round != worked => job,
                        _ => {
                            state = shared.opened.wait(state).unwrap_or_else(|e| e.into_inner());
                            continue;
                        }
                    };
                    worked = state.round;
                    state.inside += 1;
                    drop(state);
                    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shared.work(job)));
                    state = shared.state();
                    state.inside -= 1;
                    state.panicked |= done.is_err();
                    shared.left.notify_all();
                }
            });
            workers += usize::from(started.is_ok());
        }
        Pool { shared, workers, rounds: Mutex::new(()) }
    }

    /// The threads a round runs on: the workers and the caller.
    pub(crate) fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Runs `f(i)` for each `i` below `n`, each once, on this thread and
    /// the workers, and returns when all are done. A panic in `f` on any
    /// thread is a panic here, after the others have finished.
    pub(crate) fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.workers == 0 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        let _round = self.rounds.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the reference is used by workers only inside the round
        // opened here, and `Closing` below (also on a panic of `f` on this
        // thread) closes the round and waits until no worker is inside it
        // before this function returns: no use of it outlives `f`.
        let lasting: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Job { f: lasting, n };
        {
            let mut state = self.shared.state();
            self.shared.next.store(0, Relaxed);
            state.panicked = false;
            state.round += 1;
            state.job = Some(job);
        }
        self.shared.opened.notify_all();
        let closing = Closing(&self.shared);
        self.shared.work(job);
        drop(closing);
        if self.shared.state().panicked {
            panic!("a heap sync thread panicked");
        }
    }
}

/// Closes the open round when dropped, and waits for the workers inside it.
struct Closing<'a>(&'a Shared);

impl Drop for Closing<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state();
        state.job = None;
        while state.inside > 0 {
            state = self.0.left.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state().ending = true;
        self.shared.opened.notify_all();
    }
}

/// The pool heap syncs use, if `VICTORY_HEAP_SYNC_THREADS` asks for one
/// (read once; said in the log when it is set to something).
pub(crate) fn get() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let value = std::env::var("VICTORY_HEAP_SYNC_THREADS").ok();
        let split = super::slots::mode().split;
        let threads = crate::mac_switches::heap_sync_threads(value.as_deref(), split, MOST);
        if threads < 2 {
            if crate::mac_switches::heap_sync_threads(value.as_deref(), true, MOST) > 1 {
                tracing::warn!(
                    "VICTORY_HEAP_SYNC_THREADS is set without VICTORY_HEAP_LOCKS=1: left off (with the heap's one \
                     slot lock the threads would only wait for each other)"
                );
            }
            return None;
        }
        let pool = Pool::new(threads);
        tracing::info!(
            threads = pool.threads(),
            worth = WORTH,
            "VICTORY_HEAP_SYNC_THREADS: a heap sync makes the changed slots of a pass on this many threads (its \
             own among them), a pass of fewer slots than `worth` alone (the frame line's \"heap sync threads\")"
        );
        Some(pool)
    })
    .as_ref()
}

/// Since the last frame line: passes made on several threads, their slots,
/// and the slots of passes the syncing thread made alone.
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A pass of `slots` changed slots was made, on several threads (`wide`) or
/// by the syncing thread alone.
pub(crate) fn note(slots: usize, wide: bool) {
    if wide {
        COUNTS[0].fetch_add(1, Relaxed);
        COUNTS[1].fetch_add(slots as u64, Relaxed);
    } else {
        COUNTS[2].fetch_add(slots as u64, Relaxed);
    }
}

/// With the frame line (`frames` since the last one). Nothing without the
/// switch.
pub(crate) fn log(frames: f64) {
    let Some(pool) = get() else { return };
    let [passes, wide, alone] = [0, 1, 2].map(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "heap sync threads (VICTORY_HEAP_SYNC_THREADS={}): {wide:.0} changed slots/frame made on several threads in \
         {passes:.1} passes/frame, {alone:.0} slots/frame in passes too small for it (under {WORTH} slots), made by \
         the syncing thread alone",
        pool.threads(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    thread_local! {
        static MARK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Every index of a round runs exactly once, whatever the number of
    /// threads and of indices, round after round, and on more than one
    /// thread when the work lasts.
    #[test]
    fn every_index_of_a_round_runs_once() {
        for threads in [1, 2, 4] {
            let pool = Pool::new(threads);
            assert_eq!(pool.threads(), threads);
            for n in [0usize, 1, 2, 3, 97, 1000] {
                for _ in 0..20 {
                    let ran: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                    pool.run(n, &|i| {
                        ran[i].fetch_add(1, Relaxed);
                    });
                    assert!(ran.iter().all(|r| r.load(Relaxed) == 1), "{threads} threads, {n} indices");
                }
            }
            // Work that lasts is shared.
            let who = Mutex::new(std::collections::HashSet::new());
            pool.run(64, &|_| {
                who.lock().unwrap().insert(std::thread::current().id());
                std::thread::sleep(std::time::Duration::from_millis(2));
            });
            let who = who.into_inner().unwrap().len();
            assert!(who <= threads && (threads == 1 || who > 1), "{who} of {threads} threads worked");
        }
    }

    /// What the caller's closure sets on a thread for an index (as
    /// `HostHeap::sync_slots` sets a worker's marks) is set while that
    /// index runs and gone after it, on the caller's thread as on a
    /// worker's; and the closure may borrow what lives only as long as the
    /// call.
    #[test]
    fn a_round_borrows_from_its_caller_and_marks_its_threads() {
        let pool = Pool::new(3);
        for round in 1..=50u32 {
            let seen: Vec<AtomicU32> = (0..40).map(|_| AtomicU32::new(0)).collect();
            let before = AtomicU32::new(0);
            MARK.with(|m| m.set(7));
            pool.run(seen.len(), &|i| {
                let old = MARK.with(|m| m.replace(round));
                before.fetch_max(if old == 7 || old == 0 { 0 } else { old }, Relaxed);
                seen[i].store(MARK.with(std::cell::Cell::get), Relaxed);
                std::thread::yield_now();
                MARK.with(|m| m.set(old));
            });
            assert!(seen.iter().all(|s| s.load(Relaxed) == round));
            assert_eq!(before.load(Relaxed), 0, "no mark of an earlier index or round was left on a thread");
            assert_eq!(MARK.with(std::cell::Cell::get), 7, "the caller's own mark is back");
        }
    }

    /// A panic in the work, on the caller's thread or a worker's, comes out
    /// of `run` after the round is over, and the pool works on.
    #[test]
    fn a_panic_in_a_round_is_the_callers_and_the_pool_goes_on() {
        let pool = Pool::new(4);
        for bad in [0usize, 13, 63] {
            let ran = AtomicU32::new(0);
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.run(64, &|i| {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    if i == bad {
                        panic!("index {i}");
                    }
                    ran.fetch_add(1, Relaxed);
                });
            }));
            assert!(out.is_err(), "the panic of index {bad} came out");
            // The thread that panicked stopped taking indices; the others
            // took the rest, unless none was in the round yet.
            assert!(ran.load(Relaxed) <= 63);
            let after = AtomicU32::new(0);
            pool.run(64, &|_| {
                after.fetch_add(1, Relaxed);
            });
            assert_eq!(after.load(Relaxed), 64, "the next round runs whole");
        }
    }
}
