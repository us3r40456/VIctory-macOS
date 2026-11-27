//! mac: the heap sync run ahead, while the host GPU runs a submission left
//! to it (`VICTORY_ASYNC_HEAP_SYNC=1`; off by default, and only with
//! `VICTORY_ASYNC_SUBMIT=1`, `crate::host::unwaited`).
//!
//! Before each submission [`super::sync_heaps`] makes again every heap slot
//! whose record the title rewrote as bytes: 80 to 160 ms a world frame on
//! the submitting thread, which then waits about as long again for the GPU.
//! With the switch on, the thread that waits for a submission left to the
//! GPU does rounds of the sync in that wait, and the unwaited thread does
//! them while nothing waits ([`super::sync_heaps_ahead`]):
//! the first at once, the next a pause after the last ended ([`pause`]:
//! 8 ms with `=1`, a larger number is the pause in ms; a round guards the
//! heap pages it took again, so each costs the title's threads a write
//! fault per page they go on writing). The sync before the next submission
//! finds what is left. The idea is avatarmanz's
//! (`VICTORY_HEAP_SYNC_ASIDE`, 53c5256), and the rule for slots left for
//! later is the one of this repository's `prep12/preparing` (d2333358).
//!
//! Why a slot may be made while a submission runs. The title writes a
//! record into its heap for work it has not submitted yet: on the console
//! the GPU reads the heap while earlier submissions run, so a record
//! changed after a submission was sent is not one that submission reads.
//! Every record changed before a submission was sent was made by the sync
//! before it, so what a round finds changed was written after the running
//! submission was sent. Title threads make slots while the GPU runs already
//! (views written by calls, `HostHeap::mirror`), and what a slot viewed
//! before is kept until the running submission has run
//! (`crate::host::retire`), so a shader reading a slot it should not reads
//! the old view, the new one or a null one, never freed memory.
//!
//! What a round leaves for the sync before the next submission:
//!
//! - a slot whose record is a view of a texture the title placed itself.
//!   Such a view is left null and made when a shader reads it, and making
//!   the slot again forgets its read mark (`crate::shadow::forget_read`),
//!   which the running submission's shaders may be setting: its read would
//!   be lost and the view made a submission late. The slot is noted
//!   ([`left`]) and made at the point it always was;
//! - the pages it had not looked at when the GPU finished: noted too, and
//!   looked at first by the next sync.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// How long after a round ended the next one starts, if the GPU still
/// runs; none if the heap sync does not run ahead (`VICTORY_ASYNC_HEAP_SYNC`,
/// `crate::mac_switches::heap_sync_pause`; nothing without
/// `VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn pause() -> Option<Duration> {
    static PAUSE: OnceLock<Option<u64>> = OnceLock::new();
    if !crate::host::unwaited::on() {
        return None;
    }
    PAUSE
        .get_or_init(|| {
            let value = std::env::var("VICTORY_ASYNC_HEAP_SYNC").ok();
            let pause = crate::mac_switches::heap_sync_pause(value.as_deref());
            if let Some(ms) = pause {
                tracing::info!(
                    ms,
                    "VICTORY_ASYNC_HEAP_SYNC: while the host GPU runs a submission left to it, the heap slots \
                     the title rewrote are made again (but views of textures it placed itself), in rounds this \
                     many ms apart"
                );
            }
            pause
        })
        .map(Duration::from_millis)
}

/// Whether the heap sync runs ahead.
pub(crate) fn on() -> bool {
    pause().is_some()
}

thread_local! {
    /// The thread is in a round ahead.
    static RUNNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A round ahead on this thread, while it lives.
pub(super) struct Round(std::time::Instant);

impl Round {
    pub(super) fn start() -> Round {
        RUNNING.with(|r| r.set(true));
        COUNTS[0].fetch_add(1, Relaxed);
        Round(std::time::Instant::now())
    }
}

impl Drop for Round {
    fn drop(&mut self) {
        RUNNING.with(|r| r.set(false));
        COUNTS[5].fetch_add(self.0.elapsed().as_micros() as u64, Relaxed);
    }
}

/// Whether this thread is in a round ahead: `HostHeap::mirror` then leaves
/// the slots named above.
pub(super) fn running() -> bool {
    RUNNING.with(std::cell::Cell::get)
}

/// Runs `f` on this thread as part of a round ahead if `round` (the answer
/// of [`running`] on the thread whose round it is), else as part of none:
/// a thread that makes slots for a syncing thread (`super::pool`) leaves
/// the slots that thread would leave, and counts as it would.
pub(super) fn as_round<R>(round: bool, f: impl FnOnce() -> R) -> R {
    let before = RUNNING.with(|r| r.replace(round));
    let out = f();
    RUNNING.with(|r| r.set(before));
    out
}

/// Since the last frame line: rounds; slots they found changed; of those,
/// slots left for the sync before the next submission; rounds cut short
/// (the GPU finished, or a round of the unwaited thread's reached its
/// time); slots that sync then made (left ones and its own); microseconds
/// in rounds.
static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A sync found `n` slots changed (in a round ahead, or before a submission).
pub(super) fn changed(n: usize) {
    COUNTS[if running() { 1 } else { 4 }].fetch_add(n as u64, Relaxed);
}

/// A round left a slot for later.
pub(super) fn left() {
    COUNTS[2].fetch_add(1, Relaxed);
}

/// A round stopped before it had looked at all it took.
pub(super) fn cut() {
    COUNTS[3].fetch_add(1, Relaxed);
}

/// The frame line of the heap sync ahead (`frames` since the last one).
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let c: [f64; 6] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "heap sync ahead: {:.1} ms/frame of heap sync done while the host GPU ran (what a wait did is in the frame \
         line's GPU time, what our thread did in neither; none in its preparing), in {:.1} rounds/frame ({:.1} cut short: the GPU finished, or our thread's \
         4 ms were up): {:.0} slots/frame \
         found changed, {:.0} of them left for the sync before the next submission (views of textures the title \
         placed itself); that sync found {:.0} slots/frame changed, the left ones among them",
        c[5] / 1000.0,
        c[0],
        c[3],
        c[1],
        c[2],
        c[4],
    );
}
