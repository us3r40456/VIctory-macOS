//! mac: flights (`VICTORY_FLIGHTS=1`; off by default, and nothing without
//! `VICTORY_ASYNC_SUBMIT=1`): the title's next submission is prepared while
//! the one left to the host GPU still runs.
//!
//! After hyphennn's flights (fork commit 578d459, `VICTORY_LATE_WAIT`): a
//! submission is sent and not waited for, and the next is prepared while
//! the GPU runs it; what it owes is settled only where something could see
//! it. There several flights are in the air and settled together; here one
//! submission stays on the GPU at a time, as with the switch off, because
//! this line keeps per host copy, not per submission, what the copy-back
//! needs (snapshots, pending ranges, the mark bitmaps, the trackers' "taken"
//! pages; `docs/unwaited.md` of 64bitMango's line §7.1 lists them), and
//! because our host copies are buffers the GPU reads in place: a title copy
//! the next submission makes on the CPU would change bytes the one left
//! still reads (on the console such a copy is ordered behind it on the
//! GPU). What moves is the finish ([`Host::finish_left`]): from before the
//! next submission's preparation to the first of
//!
//! - its GPU work found done (asked without waiting, before each list and
//!   every [`POLL`] steps: then it is finished at once, and its fences move);
//! - a step of the next one that could see or disturb what it owes
//!   ([`Why`]): the eviction pass when it is due; unordered views made
//!   again; the heap sync without the rounds ahead
//!   (`VICTORY_ASYNC_HEAP_SYNC=1` writes heap slots while the GPU runs
//!   already); the texture uploads while its own upload list is on the GPU
//!   (one list and one staging pool, reset per submission); a title copy or
//!   move; a step of ours, but one that writes only the list's own memory
//!   or our flags (`Step::Own`) from title memory it cannot write; a
//!   declaration of what shaders write (snapshots,
//!   arming); a title-memory program's tables; a fill from title memory it
//!   may write or lands in; refreshes of a copy it may write (the prewarm's
//!   rules, [`Host::prewarm`]), or of the title memory it lands in (those
//!   of the other copies are made first);
//! - just before the next one goes to the GPU.
//!
//! Refreshes made before that are made as the prewarm makes them
//! ([`crate::shadow::Shadow::update_early`]): the title's bytes, which the
//! running lists read from pages their own preparation took (the console's
//! rule: a title does not change what a submitted list reads before it
//! knows the list ran). The heap sync writes slots the running lists may
//! read, as the rounds ahead do. A view heap made meanwhile (`Host::quiet`,
//! VIctory-macOS #31) waits for the GPU first.

use super::*;

/// Turns the switch on for the tests, with [`super::tests::FORCED`].
#[cfg(test)]
pub(in crate::host) static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the title's next submission is prepared while the one left to
/// the host GPU still runs (`VICTORY_FLIGHTS=1`, with `VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Relaxed) {
        return super::on();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    super::on()
        && *ON.get_or_init(|| {
            let on = crate::switch::on("VICTORY_FLIGHTS");
            if on {
                tracing::info!(
                    "VICTORY_FLIGHTS: the title's next submission is prepared while the one left to the host GPU \
                     runs, and that one is finished at the first step that could see what it owes, at the latest \
                     before the next goes to the GPU; held waits its pending writes meet count as met, and Signals \
                     go behind it"
                );
            }
            on
        })
}

/// The GPU is asked whether it has run the submission left every this many
/// steps of the next one (and before each of its lists).
const POLL: u32 = 64;

/// Why the submission left was finished while the next was prepared.
#[derive(Clone, Copy)]
pub(in crate::host) enum Why {
    /// The GPU had run it.
    Done,
    /// The eviction pass was due (it gives up what the last submission used).
    Evict,
    /// Unordered views were made again.
    Remake,
    /// The heap sync, without the rounds ahead.
    Heaps,
    /// The texture uploads, with its upload list on the GPU.
    Uploads,
    /// A title copy or move.
    Copy,
    /// A step of ours.
    Host,
    /// A declaration of what shaders write.
    Write,
    /// A title-memory program's tables.
    Memory,
    /// A fill from title memory it may write.
    Fill,
    /// Refreshes of what it may write.
    Flush,
    /// The next one goes to the GPU.
    Give,
    /// A view heap was made (`Host::quiet`): it stood the GPU.
    Quiet,
}

const WHYS: usize = 13;

/// Since the last frame line: submissions prepared beside the one left;
/// microseconds of their preparation done while it was on the GPU.
static BESIDE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// By [`Why`]: finishes, and the microseconds each waited for the GPU.
static FINISHED_BY: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
static WAITED_BY: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
/// Refreshes the early form refused all the same (0 if the rules hold).
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// What the submission left may write or will write once it is finished,
/// read from it under the submission lock when first needed.
pub(in crate::host) struct Footprint {
    /// It runs programs that use title memory: they write copies at
    /// addresses they compute, so every copy and all title memory.
    every: bool,
    /// Copies its heaps view for unordered access (their shaders may write
    /// them with no list declaring it).
    written: std::collections::HashSet<usize>,
    /// Title memory it may write or lands in: its declared ranges, the
    /// copies above whole, its read-backs and texture copy-backs, its
    /// bottom-of-pipe writes and the fences behind it.
    spans: crate::flights::Spans,
    /// Title memory whose copies hold shader writes kept from it
    /// (`crate::memory::kept`): a refresh there drops them.
    kept: crate::flights::Spans,
}

impl Footprint {
    fn of(left: &Left) -> Footprint {
        let mut written = std::collections::HashSet::new();
        let mut spans = crate::flights::Spans::default();
        for heap in &left.ran.heaps {
            crate::bindless::written(heap, |copy| {
                if written.insert(std::sync::Arc::as_ptr(copy) as usize) {
                    let (start, len) = copy.range();
                    spans.add(start as usize, (start + len) as usize);
                }
            });
        }
        for &(at, end) in &left.declared {
            spans.add(at, end);
        }
        for &(at, n) in &left.writes {
            spans.add(at as usize, (at + n) as usize);
        }
        spans.join();
        let mut kept = crate::flights::Spans::default();
        for (at, n) in crate::memory::kept() {
            kept.add(at, at + n);
        }
        kept.join();
        Footprint { every: left.memory, written, spans, kept }
    }

    /// Whether title memory `[at, at + len)` may change when it is finished.
    fn reaches(&self, at: usize, len: usize) -> bool {
        self.every || self.spans.touches(at, len)
    }

    /// Whether `copy` may not be refreshed before it is finished (the
    /// prewarm's rules, and title memory it lands in): `ranges` are what is
    /// to be refreshed.
    fn holds(&self, copy: &std::sync::Arc<crate::shadow::Shadow>, ranges: &crate::ranges::Ranges) -> bool {
        if self.every || !copy.idle() || self.written.contains(&(std::sync::Arc::as_ptr(copy) as usize)) {
            return true;
        }
        let (start, len) = copy.range();
        self.kept.touches(start as usize, len as usize) || ranges.iter().any(|(from, to)| self.spans.touches(from, to - from))
    }
}

/// The submission left while the title's next is prepared under the
/// submission lock ([`Host::submissions_beside`]).
pub(in crate::host) struct Beside {
    /// Its fence value, until it is finished.
    value: Option<u64>,
    started: Instant,
    steps: u32,
    footprint: Option<Footprint>,
    /// Time spent finishing it (its wait and what it owes): not the next
    /// one's preparation.
    pub(in crate::host) finishing: Duration,
}

impl Beside {
    /// Whether the submission left is still not finished.
    pub(in crate::host) fn running(&self) -> bool {
        self.value.is_some()
    }

    fn footprint(&mut self) -> &Footprint {
        self.footprint.get_or_insert_with(|| {
            LEFT.lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or_else(|| Footprint { every: true, ..Footprint::empty() }, Footprint::of)
        })
    }
}

impl Footprint {
    fn empty() -> Footprint {
        Footprint {
            every: false,
            written: Default::default(),
            spans: Default::default(),
            kept: Default::default(),
        }
    }
}

impl Host {
    /// [`Host::submissions`] for the title's lists `next` with the switch
    /// on: the submission left (if any) is not finished but returned
    /// running, to be finished by the next calls here, unless `next` is the
    /// second part of a split piece (`VICTORY_COPY_ORDER=split`: it starts
    /// with a title copy) or the GPU was stood for it already.
    pub(in crate::host) fn submissions_beside(
        &self,
        next: &[&HostList],
    ) -> (std::sync::MutexGuard<'_, u64>, Option<Beside>) {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // Not a list it runs too (a list submitted twice): its steps are
        // looked at under the list's lock, which finishing it takes.
        let running = (VALUE.load(Acquire) != 0)
            .then(|| {
                LEFT.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .filter(|l| l.signalled && !l.stood)
                    .filter(|l| !l.lists.iter().any(|h| next.iter().any(|n| std::ptr::eq(&**h, *n))))
                    .map(|l| l.value)
            })
            .flatten()
            .filter(|_| !next.iter().any(|l| l.deferred().after_copy));
        let Some(value) = running else {
            self.settle_held(&mut submitted, By::Next, next);
            return (submitted, None);
        };
        BESIDE[0].fetch_add(1, Relaxed);
        let beside = Beside {
            value: Some(value),
            started: Instant::now(),
            steps: 0,
            footprint: None,
            finishing: Duration::ZERO,
        };
        (submitted, Some(beside))
    }

    /// Finishes the submission left, if `beside` still runs it ([`Why`]).
    pub(in crate::host) fn finish_beside(&self, submitted: &mut u64, beside: &mut Option<Beside>, why: Why) {
        let Some(b) = beside.as_mut() else { return };
        let Some(value) = b.value.take() else { return };
        b.footprint = None;
        let at = Instant::now();
        BESIDE[1].fetch_add((at - b.started).as_micros() as u64, Relaxed);
        // Only finished under the lock we hold: still the one left.
        if VALUE.load(Acquire) == value {
            let waited = self.finish_left(submitted, By::Next);
            FINISHED_BY[why as usize].fetch_add(1, Relaxed);
            WAITED_BY[why as usize].fetch_add(waited.as_micros() as u64, Relaxed);
        }
        b.finishing += at.elapsed();
    }

    /// Finishes the submission left if the GPU has run it.
    pub(in crate::host) fn poll_beside(&self, submitted: &mut u64, beside: &mut Option<Beside>) {
        let Some(value) = beside.as_ref().and_then(|b| b.value) else { return };
        // SAFETY: a plain query of our fence.
        if unsafe { self.fence.GetCompletedValue() } >= value {
            self.finish_beside(submitted, beside, Why::Done);
        }
    }

    /// Before `step` of the next submission's lists: finishes the
    /// submission left if the step could see or disturb what it owes.
    pub(in crate::host) fn beside_step(&self, submitted: &mut u64, beside: &mut Option<Beside>, step: &Step) {
        let Some(b) = beside.as_mut().filter(|b| b.running()) else { return };
        b.steps += 1;
        let why = match step {
            Step::Copy(..) | Step::Move(..) => Some(Why::Copy),
            Step::Host(_) => Some(Why::Host),
            &Step::Own(_, at, len) => (len > 0 && b.footprint().reaches(at, len)).then_some(Why::Host),
            Step::GpuWrite(..) => Some(Why::Write),
            Step::Memory(_) => Some(Why::Memory),
            &Step::Fill(_, src, len) => b.footprint().reaches(src, len).then_some(Why::Fill),
            Step::Refresh(..) | Step::Slot(..) | Step::Heap(_) | Step::Texture(..) => None,
        };
        match why {
            Some(why) => self.finish_beside(submitted, beside, why),
            None if b.steps.is_multiple_of(POLL) => self.poll_beside(submitted, beside),
            None => {}
        }
    }

    /// Before the next submission's refreshes are made, if the submission
    /// left still runs: takes out of `refreshes` those of copies it may
    /// write or lands in (returned, to be made once it is finished); the
    /// rest may be made now, early ([`crate::shadow::Shadow::update_early`]).
    /// `None` (nothing taken) if nothing runs beside.
    pub(in crate::host) fn beside_refreshes(
        &self,
        beside: &mut Option<Beside>,
        refreshes: &mut Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    ) -> Option<Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>> {
        let b = beside.as_mut().filter(|b| b.running())?;
        let footprint = b.footprint();
        let (held, early): (Vec<_>, Vec<_>) =
            std::mem::take(refreshes).into_iter().partition(|(copy, ranges)| footprint.holds(copy, ranges));
        *refreshes = early;
        Some(held)
    }

    /// [`Host::quiet`] on the thread that prepares the next submission: a
    /// view heap is to be made and filled, and the submission left may
    /// still run (VIctory-macOS #31): the GPU is waited for first. Under
    /// the submission lock.
    pub(in crate::host) fn stand_beside(&self) {
        if VALUE.load(Acquire) != 0 {
            FINISHED_BY[Why::Quiet as usize].fetch_add(1, Relaxed);
            self.stand();
        }
    }

    /// Whether the texture uploads would reset what the submission left
    /// runs: its upload list, and the staging pool its uploads read from
    /// (`crate::bindless::uploads`, `crate::texture::recycle`).
    pub(in crate::host) fn beside_uploads(&self, beside: &Option<Beside>) -> bool {
        beside.as_ref().is_some_and(Beside::running)
            && LEFT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_none_or(|l| l.uploads)
    }
}

/// An early refresh refused (the copy stopped being idle): made the
/// ordinary way, and counted.
pub(in crate::host) fn refused() {
    REFUSED.fetch_add(1, Relaxed);
}

/// The frame line of the flights (`frames` since the last one).
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let beside = BESIDE[0].swap(0, Relaxed) as f64 / frames;
    let beside_ms = BESIDE[1].swap(0, Relaxed) as f64 / 1000.0 / frames;
    let by: [f64; WHYS] = std::array::from_fn(|i| FINISHED_BY[i].swap(0, Relaxed) as f64 / frames);
    let ms: [f64; WHYS] = std::array::from_fn(|i| WAITED_BY[i].swap(0, Relaxed) as f64 / 1000.0 / frames);
    let refused = REFUSED.swap(0, Relaxed) as f64 / frames;
    let met = MET.swap(0, Relaxed) as f64 / frames;
    tracing::info!(
        "flights: {beside:.1}/frame prepared while the submission before ran on the host GPU, {beside_ms:.1} ms/frame \
         of preparing beside it; it was finished by (its GPU work done, eviction, unordered views made again, heap \
         sync, texture uploads, a title copy, a step of ours, a shader-write declaration, a title-memory program, a \
         fill from what it may write, refreshes of what it may write, the next going to the GPU, a view heap made): \
         {by:.1?} per frame, after waiting {ms:.1?} ms/frame for the GPU; {refused:.1} early refreshes refused (0 if \
         the rules hold); {met:.1} held-wait looks/frame answered from a pending write"
    );
}

/// Finishes so far by every [`Why`] (tests; the frame line takes them).
#[cfg(test)]
pub(super) fn finished_beside() -> u64 {
    FINISHED_BY.iter().map(|n| n.load(Relaxed)).sum()
}

/// Finishes so far for `why` (tests).
#[cfg(test)]
pub(super) fn finished_for(why: Why) -> u64 {
    FINISHED_BY[why as usize].load(Relaxed)
}

/// Since the last frame line: looks at a held wait answered from the
/// pending writes of the submission left ([`super::pending_value`]).
pub(super) static MET: AtomicU64 = AtomicU64::new(0);
