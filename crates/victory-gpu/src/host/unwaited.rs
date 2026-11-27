//! Title submissions the host GPU is not waited for
//! (`VICTORY_ASYNC_SUBMIT=1`; off by default, and then nothing here runs).
//!
//! The idea and its settle points are Belobug's (fork commits 2bfe95a,
//! a049b48, eebcf07, 9b89ec2, bdecda3, cf57ca3, a59f3e5), written again for
//! this line's submissions.
//!
//! [`Host::run`] prepares a submission of the title's, gives it to the host
//! GPU and returns ([`Host::leave`]), so the title's thread goes on while
//! the GPU runs it. What the submission still owes (the check that it ran,
//! what its shaders wrote copied back into title memory, its lists' work
//! for after, their bottom-of-pipe writes, the retired resources, its lists
//! given back) is done by [`Host::finish`], the same code as for a
//! submission that was waited for, when something waits for it:
//!
//! - the next submission, the title's or ours (the present's copy, texture
//!   uploads, captures), before anything of it is prepared
//!   ([`Host::submissions`]): one submission at a time from preparation to
//!   copy-back, as before, and ours always wait, as they read or free what
//!   they made as soon as they return;
//! - a queue's Signal, before the title's fence moves (`crate::queue`),
//!   if the submission is that queue's ([`settle_signal`]); or, with
//!   `VICTORY_ASYNC_SIGNALS=1`, the Signal returns and the fence moves when
//!   the submission is finished by something else ([`signal_behind`]);
//! - a queue's head whose wait does not hold, if the submission owes a
//!   bottom-of-pipe write where the wait reads (`crate::hold::World::owed`,
//!   [`settle_waits`]); with `VICTORY_ASYNC_SETTLE_ALL=1` every Signal and
//!   every such head, as in the fork;
//! - a present, before the picture is looked at (`crate::queue`);
//! - a queue's page mapping (`CopyPageMappingsBatchX`), which the console
//!   does after the queue's work before it: what that work owes title
//!   memory goes where the pages still are;
//! - [`Host::quiet`], which waits for the GPU alone ([`Host::stand`]): the
//!   rest is left to the next of the above, as its caller holds locks;
//! - failing all of those, a thread of ours ([`idle`]) once the GPU has
//!   run it: the title may read memory for a bottom-of-pipe write (or for
//!   what its shaders wrote) without calling us, as it may on the console,
//!   where the GPU writes it with nobody waiting.
//!
//! At most one submission is left so, and the submission lock is taken by
//! nobody without one of these, so everything that says "the last
//! submission has run" under that lock still holds. Its lists stay held
//! here until it is finished: a title list reset or released meanwhile
//! leaves its host list to us (`crate::list::restart`), as for a list a
//! queue holds.
//!
//! The fork also refreshes the next submission's buffer copies while the
//! one before runs. There `Shadow::update` goes around the ranges a running
//! submission's shaders write; here it copies under them and into their
//! snapshot, which is right only once that submission has been copied back
//! (`crate::shadow::Shadow::update_tracked`). So only the copies the
//! running submission cannot write are refreshed early, and only with
//! `VICTORY_ASYNC_PREWARM=1` ([`Host::prewarm`]).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

use super::*;

/// mac: the next submission prepared while this one runs (`VICTORY_FLIGHTS`).
pub(crate) mod flights;

/// Whether the title's submissions may return before the host GPU has run
/// them (`VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if tests::FORCED.load(Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_ASYNC_SUBMIT");
        if on {
            tracing::info!(
                "VICTORY_ASYNC_SUBMIT: the title's submissions return before the host GPU has run them, \
                 and are finished when something waits for them"
            );
        }
        on
    })
}

/// Whether every Signal and every wait that does not hold finishes the
/// submission left, whatever queue it is of and whatever it owes
/// (`VICTORY_ASYNC_SETTLE_ALL=1`): the fork's rule, to compare with.
fn all() -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    *ALL.get_or_init(|| {
        let all = crate::switch::on("VICTORY_ASYNC_SETTLE_ALL");
        if all {
            tracing::info!(
                "VICTORY_ASYNC_SETTLE_ALL: every queue Signal and every wait that does not hold \
                 finishes the submission left to the host GPU"
            );
        }
        all
    })
}

/// Whether, while a submission left to the host GPU still runs and the
/// lists that run next are known, their buffer copies are brought up to
/// date ([`Host::prewarm`]; `VICTORY_ASYNC_PREWARM=1`, off by default, and
/// nothing without `VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn prewarms() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    on() && *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_ASYNC_PREWARM");
        if on {
            tracing::info!(
                "VICTORY_ASYNC_PREWARM: while the host GPU runs a submission left to it, the buffer copies \
                 of the lists that run next are brought up to date, but for those it may write"
            );
        }
        on
    })
}

/// Whether a queue's Signal behind a submission left to the host GPU
/// returns at once, the title's fence moving when that submission is
/// finished ([`signal_behind`]; `VICTORY_ASYNC_SIGNALS=1`, off by default,
/// and nothing without `VICTORY_ASYNC_SUBMIT=1`, nor with
/// `VICTORY_ASYNC_SETTLE_ALL=1`, which asks for every Signal to wait).
/// Needs our thread: with nothing else waiting, it is what finishes the
/// submission.
fn signals() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    on() && *ON.get_or_init(|| {
        // mac: with flights, also with `VICTORY_ASYNC_SETTLE_ALL=1`.
        let asked = (crate::switch::on("VICTORY_ASYNC_SIGNALS") && !all()) || flights::on();
        let on = asked && idler().is_some();
        if on {
            tracing::info!(
                "VICTORY_ASYNC_SIGNALS: a queue's Signal does not wait for the submission left to the host GPU; \
                 its fence moves when that submission is finished"
            );
        }
        on
    })
}

/// Whether a prepared submission of the title's may be left: not one whose
/// lists run one by one (`VICTORY_GPU_TIMING`), nor in a listed frame, nor
/// while frames are captured (diagnostics builds; the dumps submit lists of
/// their own under their own locks).
pub(super) fn may(groups: usize, traced: bool) -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::capture::folder().is_some() {
        return false;
    }
    on() && groups == 1 && !traced
}

/// What finished a submission left to the host GPU.
#[derive(Clone, Copy, Debug)]
pub(crate) enum By {
    /// The title's next submission.
    Next,
    /// A submission of ours.
    Ours,
    /// A queue's Signal.
    Signal,
    /// A queue's head whose wait did not hold.
    Wait,
    /// A present.
    Present,
    /// A queue's page mapping.
    Mapping,
    /// A view heap being made (the wait for the GPU alone).
    Quiet,
    /// Nothing: our thread, once the host GPU had run it.
    Idle,
}

const WAYS: usize = 8;

/// A submission given to the host GPU and not waited for.
struct Left {
    /// Its value on the host fence.
    value: u64,
    /// Whether the fence was told to reach it (else nothing is waited for,
    /// as for a waited submission whose signal failed).
    signalled: bool,
    /// The host GPU was already waited for ([`Host::stand`]).
    stood: bool,
    /// The title's lists, held until it is finished.
    lists: Vec<std::sync::Arc<HostList>>,
    /// Where their bottom-of-pipe writes go (address, bytes), and where
    /// the fences of `signals` are.
    writes: Vec<(u64, u64)>,
    /// Signals of the title's that follow it, in order: the fence (held),
    /// its value, and since when ([`signal_behind`]).
    signals: Vec<(usize, u64, Instant)>,
    /// Whether one of them runs programs that use title memory, which
    /// write host copies at addresses they compute (`crate::memory`).
    memory: bool,
    /// mac: with [`flights`] (else empty): title memory its shaders may
    /// write as its lists declared, and that its work for after lands in
    /// (address, end); its bottom-of-pipe writes in order (address, the
    /// value if it is one, bytes); whether it runs an upload list of ours.
    declared: Vec<(usize, usize)>,
    values: Vec<(usize, Option<u64>, usize)>,
    uploads: bool,
    ran: Ran,
}

/// The submission left, under the submission lock.
static LEFT: Mutex<Option<Left>> = Mutex::new(None);

/// Its fence value while it is not finished, else 0: read without a lock.
static VALUE: AtomicU64 = AtomicU64::new(0);

/// The title queue it is of (meaningful while [`VALUE`] is not 0).
static QUEUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Since the last frame line: Signals of other queues, and waits it owed
/// no write to, that went on without finishing it.
static SPARED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Lists of finished submissions, to give back (`crate::list::recycle`) by
/// a thread that does not hold the submission lock, as before.
static DONE: Mutex<Vec<std::sync::Arc<HostList>>> = Mutex::new(Vec::new());

/// Submissions left and finished so far.
static FINISHED: AtomicU64 = AtomicU64::new(0);

/// Since the last frame line: submissions left; then, by what finished
/// them ([`By`]), how many and the microseconds it waited for the host GPU.
static LEAVES: AtomicU64 = AtomicU64::new(0);
/// And the microseconds it took to give them to the host GPU (the calls
/// that submit, which a runtime may do much of its own work in).
static GIVING_US: AtomicU64 = AtomicU64::new(0);
static FINISHES: [AtomicU64; WAYS] = [const { AtomicU64::new(0) }; WAYS];
static WAITED_US: [AtomicU64; WAYS] = [const { AtomicU64::new(0) }; WAYS];

fn take_done() -> Vec<std::sync::Arc<HostList>> {
    std::mem::take(&mut *DONE.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Submissions left and finished so far ([`settle_waits`]).
pub(crate) fn finished() -> u64 {
    FINISHED.load(Acquire)
}

/// Finishes the submission left to the host GPU, if there is one: waits
/// for it and does what it owes. Whether there was one. Nothing on a thread
/// that holds the submission lock (inside a submission none is left).
pub(crate) fn settle(by: By) -> bool {
    settle_before(by, &[])
}

/// [`settle`], when the lists that run next are known (`next`, else none).
fn settle_before(by: By, next: &[&HostList]) -> bool {
    if VALUE.load(Acquire) == 0 || HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
        return false;
    }
    let Some(host) = host() else { return false };
    drop(host.submissions(by, next));
    take_done().into_iter().for_each(crate::list::recycle);
    true
}

/// Before `queue`'s Signal moves the title's fence: [`settle`], if the
/// submission left is that queue's. A fence says the queue's own work
/// before the Signal has run, and all of it but the submission left has
/// been finished; another queue's is not what the fence is about.
pub(crate) fn settle_signal(queue: usize) -> bool {
    if VALUE.load(Acquire) == 0 {
        return false;
    }
    // Read after the value: a submission of `queue` cannot be left
    // meanwhile, as its Signal and its submissions come one at a time.
    // mac: with flights, every queue's Signal goes behind it (`signal_after`).
    if QUEUE.load(Acquire) != queue && !(all() && !flights::on()) {
        SPARED[0].fetch_add(1, Relaxed);
        return false;
    }
    settle(By::Signal)
}

/// `queue`'s Signal of `fence` to `value`, when the queue holds nothing:
/// if the submission left to the host GPU is that queue's, the Signal is
/// put behind it and the call returns (true): the fence, held meanwhile,
/// moves when the submission is finished ([`Host::finish_left`]), by the
/// next thing that waits for it or, with nothing waiting, by our thread
/// once the GPU has run it. As on the console, where a Signal does not
/// wait for the queue's work either. False (nothing done) with the switch
/// off, with nothing of the queue's left, or while it is being finished.
pub(crate) fn signal_behind(queue: usize, fence: usize, value: u64) -> bool {
    // mac: with flights, any queue's (hyphennn 578d459: a fence signalled
    // while flights are in the air is set behind them).
    if !signals() || VALUE.load(Acquire) == 0 || (QUEUE.load(Acquire) != queue && !flights::on()) {
        return false;
    }
    crate::fence::keep(fence as crate::object::This);
    if signal_after(fence, value) {
        return true;
    }
    crate::fence::let_go(fence as crate::object::This);
    false
}

/// A Signal the queues deliver (it stood behind held work, which has been
/// given to the host GPU by now), its fence held already: put behind the
/// submission left, whichever queue's, as [`signal_behind`] puts one.
pub(crate) fn signal_after(fence: usize, value: u64) -> bool {
    if !signals() || VALUE.load(Acquire) == 0 {
        return false;
    }
    let mut left = LEFT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(left) = left.as_mut() else {
        // Being finished right now.
        return false;
    };
    left.signals.push((fence, value, Instant::now()));
    // A list's wait may read the fence's value in memory.
    left.writes
        .push((crate::fence::address(fence as crate::object::This), 8));
    SIGNALS[0].fetch_add(1, Relaxed);
    true
}

/// Since the last frame line: Signals put behind a submission left, and
/// the microseconds until their fences moved, summed.
static SIGNALS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For a queue's head whose `waits` do not all hold: [`settle`], if the
/// submission left owes a bottom-of-pipe write where one of them reads
/// (`next`: the lists that run once the waits hold, if asked for).
/// Whether any submission left was finished since `seen` ([`finished`],
/// brought up to date), by this call or by another thread: the waits are
/// then to be looked at again.
///
/// A wait for memory the submission changes some other way (what its
/// shaders write, a read-back) is not seen here: its head is held until
/// the submission is finished by something else ([`idle`] at the latest),
/// and runs at the queues' next look (their timer's, within 2 ms).
pub(crate) fn settle_waits(
    waits: &[crate::hold::ValueWait],
    next: &[&std::sync::Arc<HostList>],
    seen: &mut u64,
) -> bool {
    if VALUE.load(Acquire) != 0 {
        let flights = flights::on();
        let owed = match LEFT.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            // mac: with flights, not for a value its writes are known to
            // leave there: the pump read that already ([`pending_value`]),
            // and finishing it first would not change what the wait sees.
            Some(left) => left.writes.iter().any(|&(at, n)| {
                waits.iter().any(|w| {
                    w.reads(at, n)
                        && !(flights
                            && matches!(
                                crate::flights::after_writes(&left.values, w.address as usize, usize::from(w.bytes)),
                                Some(Some(_))
                            ))
                })
            }),
            // Being finished right now: `settle` waits for that.
            None => true,
        };
        if owed || (all() && !flights) {
            let next: Vec<&HostList> = next.iter().map(|l| &***l).collect();
            settle_before(By::Wait, &next);
        } else {
            SPARED[1].fetch_add(1, Relaxed);
        }
    }
    let now = finished();
    std::mem::replace(seen, now) != now
}

/// mac: with [`flights`], what the `bytes` at `address` (a held wait's
/// value) will hold once the submission left is finished, if one of its
/// bottom-of-pipe writes stores all of them and its value is known (else
/// `None`: memory says). The wait's work is then prepared beside it, and
/// goes to the GPU behind it, which is the order the wait asks for
/// (hyphennn 578d459, `late_met`).
pub(crate) fn pending_value(address: u64, bytes: u8) -> Option<u64> {
    if VALUE.load(Acquire) == 0 || !flights::on() {
        return None;
    }
    let left = LEFT.lock().unwrap_or_else(|e| e.into_inner());
    let value = crate::flights::after_writes(&left.as_ref()?.values, address as usize, usize::from(bytes))??;
    flights::MET.fetch_add(1, Relaxed);
    Some(value)
}

/// The thread that finishes a submission left to the host GPU when nothing
/// else has by the time the GPU has run it; `None` if it could not be made
/// (then only the callers of [`settle`] and the submission lock finish it).
fn idler() -> Option<&'static std::thread::Thread> {
    static IDLER: OnceLock<Option<std::thread::Thread>> = OnceLock::new();
    IDLER
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("victory unwaited".into())
                .spawn(idle)
                .map(|made| made.thread().clone())
                .inspect_err(|e| tracing::warn!(error = %e, "no thread to finish host submissions nothing waits for"))
                .ok()
        })
        .as_ref()
}

/// An event of our thread's own on the host fence, asked for each
/// submission left while Signals stand behind them (`signals`): the thread
/// is then what moves the title's fences, and a millisecond's sleep would
/// be added to each. None if it could not be made (the thread sleeps).
fn idle_event() -> Option<HANDLE> {
    static EVENT: OnceLock<usize> = OnceLock::new();
    let event = *EVENT.get_or_init(|| {
        // SAFETY: a plain auto-reset event, kept for the run.
        let made = unsafe { CreateEventW(None, false, false, None) };
        made.map_or(0, |e| e.0 as usize)
    });
    (event != 0).then_some(HANDLE(event as *mut core::ffi::c_void))
}

/// [`idler`]'s work: sleeps while no submission is left; while one is,
/// looks at the host fence every millisecond (or when its event wakes it)
/// and finishes the submission once the GPU has run it.
///
/// With the heap sync ahead on (`crate::descriptor::ahead`), it also does
/// rounds of that sync while the GPU runs the submission and nothing else
/// holds the submission lock ([`Host::round_if_free`]): a title thread that
/// went on after its submission or its Signal does none, as it does not
/// wait here. The first a millisecond after the submission was left (the
/// title's next submission, which knows what runs next, comes first if it
/// comes at once), the next a pause after the last ended.
fn idle() {
    // The submission rounds are paced for, and when the next one is due.
    let mut round = (0, Instant::now());
    loop {
        let value = VALUE.load(Acquire);
        if value == 0 {
            // Woken by `Host::leave`, after it set the value.
            std::thread::park();
            continue;
        }
        let Some(host) = host() else { return };
        // A test that needs the submission left to stay left.
        #[cfg(test)]
        if tests::IDLE_HELD.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        // SAFETY: a plain query of our fence.
        if unsafe { host.fence.GetCompletedValue() } >= value {
            // mac: with flights the lock may be held by the next
            // submission's preparation, which finishes it itself.
            if !flights::on() {
                settle(By::Idle);
            } else if !host.settle_if_free(value) {
                std::thread::sleep(Duration::from_millis(1));
            }
            continue;
        }
        if let Some(pause) = crate::descriptor::ahead::pause() {
            if round.0 != value {
                round = (value, Instant::now() + Duration::from_millis(1));
            } else if Instant::now() >= round.1 {
                host.round_if_free(value);
                round.1 = Instant::now() + pause;
                continue;
            }
        }
        if let Some(event) = idle_event().filter(|_| signals()) {
            // SAFETY: our event; the fence decides, above.
            unsafe { WaitForSingleObject(event, 1) };
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Since the last frame line, of [`Host::prewarm`]: times it had a
/// running submission and lists to look at; of those, times it did nothing
/// as a running list uses title memory, and times the GPU finished first;
/// copies brought up to date and their bytes; copies left alone as the
/// running submission may write them, and as their memory is not tracked;
/// microseconds.
static EARLY: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// The copies that took pages early, held until the submission that ran
/// meanwhile is finished (`crate::shadow::forget_early`).
static EARLY_HELD: Mutex<Vec<std::sync::Arc<crate::shadow::Shadow>>> = Mutex::new(Vec::new());

/// What [`Host::prewarm`] gathers: the ranges of each copy to bring up to
/// date (ranges less than [`JOIN`] apart as one, as a submission's own
/// refreshes), and the copies left alone.
#[derive(Default)]
struct Early {
    wanted: Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    /// Where each copy is in `wanted`, by copy.
    index: std::collections::HashMap<usize, usize>,
    /// Copies the running lists' heaps view for unordered access.
    written: std::collections::HashSet<usize>,
    /// Title memory whose copies hold shader writes kept from it: where
    /// each stretch starts and ends, in order, none touching the next.
    kept: Vec<(usize, usize)>,
    /// Copies decided against, and how many: as the running submission may
    /// write them, as their memory is not tracked.
    left: std::collections::HashSet<usize>,
    alone: [u64; 2],
}

impl Early {
    fn add(&mut self, copy: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        // Nothing to copy, as for a submission's own refreshes.
        if len == 0 || copy.clean() {
            return;
        }
        let key = std::sync::Arc::as_ptr(copy) as usize;
        if self.left.contains(&key) {
            return;
        }
        let i = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                let (start, bytes) = copy.range();
                let (start, end) = (start as usize, (start + bytes) as usize);
                let why = if copy.dirty_bit().is_none() {
                    Some(1)
                } else if !copy.idle()
                    || self.written.contains(&key)
                    || self
                        .kept
                        .get(self.kept.partition_point(|&(_, e)| e <= start))
                        .is_some_and(|&(s, _)| s < end)
                {
                    Some(0)
                } else {
                    None
                };
                if let Some(why) = why {
                    self.left.insert(key);
                    self.alone[why] += 1;
                    return;
                }
                self.wanted.push((copy.clone(), Default::default()));
                self.index.insert(key, self.wanted.len() - 1);
                self.wanted.len() - 1
            }
        };
        self.wanted[i].1.insert(address, address + len, JOIN);
    }
}

/// The frame line of submissions left to the host GPU (`frames` since the
/// last one).
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let left = LEAVES.swap(0, Relaxed) as f64 / frames;
    let giving = GIVING_US.swap(0, Relaxed) as f64 / 1000.0 / frames;
    let by: [f64; WAYS] = std::array::from_fn(|i| FINISHES[i].swap(0, Relaxed) as f64 / frames);
    let ms: [f64; WAYS] =
        std::array::from_fn(|i| WAITED_US[i].swap(0, Relaxed) as f64 / 1000.0 / frames);
    let spared: [f64; 2] = std::array::from_fn(|i| SPARED[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "unwaited submissions: {left:.1}/frame left to the host GPU, which took {giving:.1} ms/frame to give them \
         to it; finished by (the title's next submission, \
         one of ours, a queue's signal, a held queue's wait, the present, a page mapping, a view heap being made, \
         nothing but our thread once it had run): \
         {by:.1?} per frame, after waiting {ms:.1?} ms/frame for the host GPU (the frame line's GPU time is these, \
         the giving, and our own submissions' waits); {spared:.1?} per frame went on without finishing it \
         (Signals of other queues, waits it owed no write to)"
    );
    if signals() {
        let [n, us] = [0, 1].map(|i| SIGNALS[i].swap(0, Relaxed) as f64);
        tracing::info!(
            "signals: {:.1}/frame returned without waiting for the submission left to the host GPU; their fences \
             moved {:.1} ms later on average, when it was finished",
            n / frames,
            if n > 0.0 { us / n / 1000.0 } else { 0.0 },
        );
    }
    crate::descriptor::ahead::log(frames);
    flights::log(frames);
    if !prewarms() {
        return;
    }
    let early: [f64; 8] = std::array::from_fn(|i| EARLY[i].swap(0, Relaxed) as f64 / frames);
    let marked = crate::shadow::EARLY_MARKED.swap(0, Relaxed) as f64 / frames;
    tracing::info!(
        "prewarm: {:.1} ms/frame of the next lists' preparation done while the host GPU ran, in {:.1} goes/frame \
         ({:.1} of them did nothing: a running list uses title memory; {:.1} stopped as the GPU finished first): \
         {:.0} buffer copies/frame brought up to date early ({:.2} MB/frame), {:.0} left alone as the running \
         submission may write them and {:.0} as their memory is not tracked; {marked:.1} pages/frame taken early \
         were marked as written by the running shaders all the same (the title's bytes kept; 0 if the rule holds)",
        early[7] / 1000.0,
        early[0],
        early[1],
        early[2],
        early[3],
        early[4] / 1e6,
        early[5],
        early[6],
    );
}

impl Host {
    /// The submission lock, once the submission left to the host GPU (if
    /// any) is finished: the holder prepares or submits with none running.
    /// `next`: the lists that run next, if known ([`Host::prewarm`]).
    pub(super) fn submissions(&self, by: By, next: &[&HostList]) -> std::sync::MutexGuard<'_, u64> {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        self.settle_held(&mut submitted, by, next);
        submitted
    }

    /// [`Host::submissions`]' part under the lock: the submission left, if
    /// any, is finished (its next lists' copies brought up to date first
    /// with the prewarm).
    fn settle_held(&self, submitted: &mut u64, by: By, next: &[&HostList]) {
        if VALUE.load(Acquire) != 0 {
            if !next.is_empty() && prewarms() {
                self.prewarm(next);
            }
            self.finish_left(submitted, by);
        }
    }

    /// mac: our thread's finish with [`flights`]: only if nothing holds the
    /// submission lock and the submission left is still the one with fence
    /// value `value` (which the GPU has run). Whether the lock was had.
    fn settle_if_free(&self, value: u64) -> bool {
        let mut submitted = match self.submitted.try_lock() {
            Ok(submitted) => submitted,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        if VALUE.load(Acquire) == value {
            self.finish_left(&mut submitted, By::Idle);
        }
        drop(submitted);
        take_done().into_iter().for_each(crate::list::recycle);
        true
    }

    /// Runs the title's own lists (a queue's piece, `crate::queue`), and
    /// returns the lists that have run, to give back: these when they were
    /// waited for, else those of submissions finished since.
    pub(crate) fn execute_title(
        &self,
        queue: usize,
        lists: Vec<std::sync::Arc<HostList>>,
    ) -> Vec<std::sync::Arc<HostList>> {
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        if !on() {
            self.execute(&runs);
            drop(runs);
            return lists;
        }
        let left = self.run(&runs, Some((queue, &lists)));
        drop(runs);
        let mut done = take_done();
        if !left {
            done.extend(lists);
        }
        done
    }

    /// Gives a prepared submission (`lists`, then the mark) to the host
    /// GPU and leaves it there. Under the submission lock, with none left.
    pub(super) fn leave(
        &self,
        submitted: &mut u64,
        lists: &[Option<ID3D12CommandList>],
        mark: Option<&Mark>,
        (title_queue, title): (usize, &[std::sync::Arc<HostList>]),
        ran: Ran,
        uploads: bool,
    ) {
        *submitted += 1;
        let started = Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
        // Last, the list that shows the submission ran.
        let mut group = lists.to_vec();
        if let Some(mark) = mark {
            mark.arm(*submitted);
            group.push(mark.list());
        }
        let queue = self.queue();
        // SAFETY: closed lists; the fence is ours.
        let signalled = unsafe {
            queue.ExecuteCommandLists(&group);
            queue.Signal(&self.fence, *submitted)
        };
        match &signalled {
            // The event is asked for now, as for a waited submission, and
            // only wakes the wait: the fence's value decides
            // ([`Host::wait_left`]).
            Ok(()) => {
                // SAFETY: the fence and event are ours.
                let asked = unsafe { self.fence.SetEventOnCompletion(*submitted, self.event) };
                if let Err(e) = asked {
                    tracing::warn!(error = %e, "no event for a host submission: its fence is polled");
                }
                // Our thread's wake-up too, if Signals may stand behind it.
                if signals()
                    && let Some(event) = idle_event()
                {
                    // SAFETY: the fence and event are ours.
                    let _ = unsafe { self.fence.SetEventOnCompletion(*submitted, event) };
                }
            }
            Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
        }
        // Time on the host GPU's side, as when the submission is waited
        // for: the frame line's GPU time.
        let giving = started.elapsed().as_micros() as u64;
        GIVING_US.fetch_add(giving, Relaxed);
        SUBMISSION_US[1].fetch_add(giving, Relaxed);
        LEAVES.fetch_add(1, Relaxed);
        let writes = title
            .iter()
            .flat_map(|l| {
                let writes = &l.deferred().writes;
                writes
                    .iter()
                    .map(|&(at, _, bytes)| (at as u64, bytes as u64))
                    .collect::<Vec<_>>()
            })
            .collect();
        let memory = (prewarms() || flights::on())
            && title.iter().any(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .any(|s| matches!(s, Step::Memory(_)))
            });
        // mac: what flights need to know of it (`flights::Footprint`).
        let (mut declared, mut values) = (Vec::new(), Vec::new());
        if flights::on() {
            for l in title {
                let deferred = l.deferred();
                for step in &deferred.copies {
                    if let &Step::GpuWrite(_, at, n, _, _) = step {
                        declared.push((at, at + n));
                    }
                }
                declared.extend(deferred.lands.iter().map(|&(at, n)| (at, at + n)));
                declared.extend(deferred.texture_backs.iter().map(|b| b.lands()));
                values.extend(deferred.writes.iter().map(|&(at, what, n)| {
                    let value = match what {
                        crate::queue::Written::Value(v) => Some(v),
                        _ => None,
                    };
                    (at, value, n)
                }));
            }
        }
        *LEFT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Left {
            value: *submitted,
            signalled: signalled.is_ok(),
            stood: false,
            lists: title.to_vec(),
            writes,
            signals: Vec::new(),
            memory,
            declared,
            values,
            uploads,
            ran,
        });
        QUEUE.store(title_queue, Release);
        VALUE.store(*submitted, Release);
        if let Some(idler) = idler() {
            idler.unpark();
        }
    }

    /// Waits until the host fence reaches `value`: how long. As
    /// [`Host::wait_for`], the fence decides and the event only wakes; the
    /// event was asked for when the submission was left, so it may be set
    /// already, or (under D3DMetal, #50) never: the fence is looked at
    /// first and then every 2 ms.
    ///
    /// `working`: the wait does rounds of the heap sync meanwhile
    /// (`crate::descriptor::ahead`), the first at once.
    fn wait_left(&self, value: u64, working: bool) -> Duration {
        let started = Instant::now();
        let mut seconds = 1;
        let pause = crate::descriptor::ahead::pause().filter(|_| working);
        let mut round = pause.map(|_| started);
        // SAFETY: a plain query of our fence.
        let ran = || unsafe { self.fence.GetCompletedValue() } >= value;
        // SAFETY: our fence and event.
        while unsafe { self.fence.GetCompletedValue() } < value {
            if round.is_some_and(|at| Instant::now() >= at) {
                crate::descriptor::sync_heaps_ahead(&ran);
                round = pause.map(|pause| Instant::now() + pause);
                continue;
            }
            // SAFETY: as above.
            unsafe { WaitForSingleObject(self.event, 2) };
            if started.elapsed().as_secs() >= seconds {
                seconds *= 2;
                // SAFETY: as above.
                let completed = unsafe { self.fence.GetCompletedValue() };
                tracing::warn!(
                    seconds = started.elapsed().as_secs(),
                    value,
                    completed,
                    "still waiting for the host GPU (a submission left to it)"
                );
            }
        }
        started.elapsed()
    }

    /// A round of the heap sync ahead by our thread, if nothing holds the
    /// submission lock and the submission left is still the one with fence
    /// value `value`: for 4 ms at most, so that a title thread that comes
    /// to wait for the submission (and then does the rounds itself) is not
    /// kept long.
    fn round_if_free(&self, value: u64) {
        let _submitted = match self.submitted.try_lock() {
            Ok(submitted) => submitted,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        // Finished, or another one left, meanwhile (both under the lock).
        if VALUE.load(Acquire) != value {
            return;
        }
        let _holding = Holding::enter();
        let started = Instant::now();
        let ran = || {
            // SAFETY: a plain query of our fence.
            let completed = unsafe { self.fence.GetCompletedValue() };
            completed >= value || started.elapsed() >= Duration::from_millis(4)
        };
        crate::descriptor::sync_heaps_ahead(&ran);
    }

    /// [`Host::quiet`]'s part: the host GPU has run the submission left to
    /// it, which stays to be finished. Under the submission lock.
    pub(super) fn stand(&self) {
        if VALUE.load(Acquire) == 0 {
            return;
        }
        let left = || LEFT.lock().unwrap_or_else(|e| e.into_inner());
        let value = left()
            .as_ref()
            .filter(|l| l.signalled && !l.stood)
            .map(|l| l.value);
        if let Some(value) = value {
            let waited = self.wait_left(value, false).as_micros() as u64;
            // The same one: it is only finished under the lock we hold.
            if let Some(left) = left().as_mut() {
                left.stood = true;
            }
            WAITED_US[By::Quiet as usize].fetch_add(waited, Relaxed);
            FINISHES[By::Quiet as usize].fetch_add(1, Relaxed);
            SUBMISSION_US[1].fetch_add(waited, Relaxed);
        }
    }

    /// While the submission left to the host GPU still runs: brings up to
    /// date the buffer copies `next` (the lists that run next) will ask
    /// for, as their own preparation does after it (their refresh steps,
    /// and what their heaps' views and title-memory tables reach), so that
    /// preparation finds them done. Under the submission lock, before the
    /// submission left is finished; stops when the GPU has run it.
    ///
    /// A copy is left alone, for the lists' own preparation, if the running
    /// submission may write it or its early refresh could differ from the
    /// one after it:
    ///
    /// - a list submitted declared that its shaders write it (a range
    ///   pending or a snapshot kept, [`Shadow::idle`]): title memory there
    ///   is older than the copy until the copy-back;
    /// - a heap the running lists take views from views it for unordered
    ///   access: their shaders may write it with no list declaring it;
    /// - every copy, if a running list uses title memory: its programs
    ///   write copies at addresses they compute;
    /// - it holds shader writes kept out of title memory until the frame's
    ///   end (`crate::memory::kept`): a refresh drops them from the pages
    ///   the title wrote, and the running lists may read them;
    /// - its memory is not tracked: it is copied whole at each use, and
    ///   would only be copied twice.
    ///
    /// The others take the pages the title wrote since they last took them.
    /// What the running submission reads of them it read from pages its
    /// own preparation took; a page the title writes after that, while the
    /// GPU reads it, is a race on the console too, where the GPU reads
    /// title memory itself. A page taken early that the running shaders
    /// marked as written all the same is counted and keeps the title's
    /// bytes ([`Shadow::update_early`]).
    ///
    /// Not done early: steps that write title memory or the list's upload
    /// memory (their order with the running submission's copy-back counts),
    /// the texture uploads (the running submission's upload list and
    /// staging buffers are reused by the next), the heaps' descriptor sync
    /// and the title-memory tables (the running shaders read both).
    fn prewarm(&self, next: &[&HostList]) {
        let left = {
            let left = LEFT.lock().unwrap_or_else(|e| e.into_inner());
            let Some(left) = left.as_ref().filter(|l| l.signalled && !l.stood) else {
                return;
            };
            (left.value, left.memory, left.ran.heaps.clone())
        };
        let (value, memory, heaps) = left;
        // SAFETY: a plain query of our fence.
        let ran = || unsafe { self.fence.GetCompletedValue() } >= value;
        if ran() {
            return;
        }
        let started = Instant::now();
        let _holding = Holding::enter();
        EARLY[0].fetch_add(1, Relaxed);
        if memory {
            EARLY[1].fetch_add(1, Relaxed);
            return;
        }
        let mut early = Early::default();
        for heap in &heaps {
            crate::bindless::written(heap, |copy| {
                early.written.insert(std::sync::Arc::as_ptr(copy) as usize);
            });
        }
        // In order and joined, to be searched once per copy.
        let mut kept = crate::memory::kept();
        kept.sort_unstable();
        for (at, n) in kept {
            match early.kept.last_mut() {
                Some((_, end)) if at <= *end => *end = (*end).max(at + n),
                _ => early.kept.push((at, at + n)),
            }
        }
        let mut cut = false;
        for list in next {
            if ran() {
                cut = true;
                break;
            }
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(copy, address, len) => early.add(copy, *address, *len),
                    Step::Slot(viewed, index) => {
                        crate::bindless::refresh_slot(viewed, *index, |c, a, l| early.add(c, a, l))
                    }
                    Step::Heap(viewed) => {
                        crate::bindless::refresh(viewed, |c, a, l| early.add(c, a, l))
                    }
                    &Step::Memory(heap) => crate::memory::due(heap, |c, a, l| early.add(c, a, l)),
                    _ => {}
                }
            }
        }
        let (mut copies, mut bytes) = (0, 0);
        let mut held = Vec::with_capacity(early.wanted.len());
        // The fence is asked at most every 200 microseconds.
        let mut asked = Instant::now();
        for (copy, ranges) in early.wanted {
            if cut {
                break;
            }
            if asked.elapsed() >= Duration::from_micros(200) {
                asked = Instant::now();
                if ran() {
                    cut = true;
                    break;
                }
            }
            let mut taken = Some(0);
            for (from, to) in ranges.iter() {
                taken = taken.and_then(|n| Some(n + copy.update_early(from, to - from)?));
            }
            match taken {
                Some(n) => {
                    copies += 1;
                    bytes += n as u64;
                    held.push(copy);
                }
                None => early.alone[0] += 1,
            }
        }
        // Held until the submission is finished: the pages they took are
        // noted by their addresses.
        EARLY_HELD
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(held);
        EARLY[2].fetch_add(u64::from(cut), Relaxed);
        EARLY[3].fetch_add(copies, Relaxed);
        EARLY[4].fetch_add(bytes, Relaxed);
        EARLY[5].fetch_add(early.alone[0], Relaxed);
        EARLY[6].fetch_add(early.alone[1], Relaxed);
        EARLY[7].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    }

    /// Waits for the submission left to the host GPU and finishes it, as
    /// [`Host::run`] does one it waited for. Under the submission lock.
    /// How long it waited for the GPU.
    fn finish_left(&self, submitted: &mut u64, by: By) -> Duration {
        let Some(left) = LEFT.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            // None is being finished either: that is done under our lock.
            VALUE.store(0, Release);
            return Duration::ZERO;
        };
        let _holding = Holding::enter();
        let Left {
            value,
            signalled,
            stood,
            lists,
            signals,
            mut ran,
            ..
        } = left;
        let mut waited = Duration::ZERO;
        if signalled && !stood {
            ran.gpu = self.wait_left(value, crate::descriptor::ahead::on());
            waited = ran.gpu;
            WAITED_US[by as usize].fetch_add(ran.gpu.as_micros() as u64, Relaxed);
            SUBMISSION_US[1].fetch_add(ran.gpu.as_micros() as u64, Relaxed);
        }
        if !stood {
            FINISHES[by as usize].fetch_add(1, Relaxed);
        }
        if signalled
            && let Some(mark) = self.mark()
            && !mark.saw(value)
            // SAFETY: a plain query.
            && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
        {
            self.lost(submitted, mark);
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, value, 0);
        {
            let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
            self.finish(None, &runs, ran);
        }
        // The Signals behind it: the work before them has run, and what it
        // owes title memory is there.
        for (fence, value, since) in signals {
            crate::fence::set(fence as crate::object::This, value);
            crate::fence::let_go(fence as crate::object::This);
            SIGNALS[1].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
        }
        // What copies took early while it ran is as anything else now.
        crate::shadow::forget_early();
        EARLY_HELD.lock().unwrap_or_else(|e| e.into_inner()).clear();
        DONE.lock().unwrap_or_else(|e| e.into_inner()).extend(lists);
        FINISHED.fetch_add(1, AcqRel);
        // Last: until now a thread that asks ([`settle`]) waits for our lock.
        VALUE.store(0, Release);
        waited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns the switch on for the tests here, whatever the environment.
    pub(super) static FORCED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Our thread leaves the submission left alone while set.
    pub(super) static IDLE_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// mac: both switches on (`VICTORY_ASYNC_SUBMIT`, `VICTORY_FLIGHTS`),
    /// and our thread held off, until dropped.
    struct Flights;

    impl Flights {
        fn on() -> Flights {
            FORCED.store(true, Relaxed);
            flights::FORCED.store(true, Relaxed);
            IDLE_HELD.store(true, Relaxed);
            Flights
        }
    }

    impl Drop for Flights {
        fn drop(&mut self) {
            IDLE_HELD.store(false, Relaxed);
            flights::FORCED.store(false, Relaxed);
            FORCED.store(false, Relaxed);
        }
    }

    /// Finishes left by flights so far (every [`flights::Why`]).
    fn finished_beside() -> u64 {
        flights::finished_beside()
    }

    /// mac: with flights, the title's next submission is prepared while
    /// the one left runs, and that one is finished before the next goes to
    /// the host GPU: its work for after, then its bottom-of-pipe write,
    /// come before the next one's, and its pending value is read from it
    /// until then.
    #[test]
    fn with_flights_the_one_left_is_finished_before_the_next_runs() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let flights = Flights::on();
        settle(By::Signal);
        let order = std::sync::Arc::new(Mutex::new(Vec::new()));
        let word = Box::new(AtomicU64::new(0));
        let at = word.as_ptr() as u64;
        let lists = [0, 1].map(|_| std::sync::Arc::new(host.new_list().unwrap()));
        for (i, list) in lists.iter().enumerate() {
            let log = order.clone();
            let word_at = word.as_ptr() as usize;
            list.defer_after(move || {
                // SAFETY: the test's word, alive until the end.
                let seen = unsafe { (*(word_at as *const AtomicU64)).load(Acquire) };
                log.lock().unwrap().push((i, seen));
            });
            list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(5 + i as u64), 8);
            list.close();
        }
        let before = finished_beside();
        let mut back = host.execute_title(1, vec![lists[0].clone()]);
        // Left (unless a diagnostics build captures): its value is known
        // before it is in memory.
        if VALUE.load(Acquire) != 0 {
            assert_eq!(pending_value(at, 8), Some(5));
            assert_eq!(pending_value(at, 4), Some(5));
            assert_eq!(pending_value(at + 8, 8), None);
        }
        back.extend(host.execute_title(1, vec![lists[1].clone()]));
        settle(By::Signal);
        drop(flights);
        assert_eq!(
            *order.lock().unwrap(),
            [(0, 0), (1, 5)],
            "the first's work for after ran first, and its write was there before the second's"
        );
        assert_eq!(word.load(Acquire), 6, "the second's write is the last");
        assert_eq!(pending_value(at, 8), None, "nothing is left");
        // (A diagnostics build that captures waits for every submission.)
        #[cfg(not(feature = "diagnostics"))]
        assert!(finished_beside() > before, "the first was finished beside the second");
        let _ = before;
        drop(back);
        drop(take_done());
    }

    /// mac: with flights, a step of ours that writes only the list's own
    /// memory (`Step::Own`) does not finish the one left; one that reads
    /// what the one left writes (here its bottom-of-pipe word) waits for it.
    #[test]
    fn with_flights_a_step_on_the_lists_own_memory_does_not_wait() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let flights = Flights::on();
        settle(By::Signal);
        let order = std::sync::Arc::new(Mutex::new(Vec::new()));
        let word = Box::new(AtomicU64::new(0));
        let at = word.as_ptr() as usize;
        for reads_the_word in [false, true] {
            order.lock().unwrap().clear();
            let lists = [0, 1].map(|_| std::sync::Arc::new(host.new_list().unwrap()));
            let log = order.clone();
            lists[0].defer_after(move || log.lock().unwrap().push("the first has run"));
            lists[0].defer_write(word.as_ptr().cast(), crate::queue::Written::Value(9), 8);
            let log = order.clone();
            let (from, len) = if reads_the_word { (at, 8) } else { (0, 0) };
            lists[1].defer_own(from, len, move || {
                // SAFETY: the test's word, alive until the end.
                let seen = unsafe { (*(at as *const AtomicU64)).load(Acquire) };
                log.lock().unwrap().push(if seen == 9 { "the second's step, after" } else { "the second's step, before" });
            });
            lists.iter().for_each(|l| l.close());
            let before = flights::finished_for(flights::Why::Host);
            let mut back = host.execute_title(1, vec![lists[0].clone()]);
            back.extend(host.execute_title(1, vec![lists[1].clone()]));
            settle(By::Signal);
            let order = order.lock().unwrap().clone();
            if reads_the_word {
                // Finished first (for the step, or as its GPU work was
                // done): the word is there when the step reads it.
                assert_eq!(order, ["the first has run", "the second's step, after"]);
            } else {
                assert!(order.contains(&"the first has run") && order.len() == 2, "{order:?}");
                assert_eq!(
                    flights::finished_for(flights::Why::Host),
                    before,
                    "a step on the list's own memory does not finish the one left"
                );
            }
            word.store(0, Release);
            drop(back);
            drop(take_done());
        }
        drop(flights);
    }

    /// mac: with flights, a step of the next submission that could see
    /// what the one left owes (here a step of ours) finishes it first.
    #[test]
    fn with_flights_a_step_of_ours_waits_for_the_one_left() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let flights = Flights::on();
        settle(By::Signal);
        let order = std::sync::Arc::new(Mutex::new(Vec::new()));
        let lists = [0, 1].map(|_| std::sync::Arc::new(host.new_list().unwrap()));
        let log = order.clone();
        lists[0].defer_after(move || log.lock().unwrap().push("the first has run"));
        let log = order.clone();
        lists[1].defer_host(move || log.lock().unwrap().push("the second's step"));
        lists.iter().for_each(|l| l.close());
        let mut back = host.execute_title(1, vec![lists[0].clone()]);
        back.extend(host.execute_title(1, vec![lists[1].clone()]));
        settle(By::Signal);
        drop(flights);
        assert_eq!(*order.lock().unwrap(), ["the first has run", "the second's step"]);
        drop(back);
        drop(take_done());
    }

    /// A title submission left to the host GPU owes its bottom-of-pipe
    /// write until something waits for it; then the write is there, the
    /// mark shows it ran, and its list comes back.
    #[test]
    fn a_submission_left_is_finished_by_what_waits_for_it() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        FORCED.store(true, Relaxed);
        let word = Box::new(AtomicU64::new(0));
        let list = std::sync::Arc::new(host.new_list().unwrap());
        list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(7), 8);
        list.close();
        let back = host.execute_title(1, vec![list.clone()]);
        // Left (or waited for, in a diagnostics build that captures): a
        // Signal finishes it, if another test's submission has not.
        settle(By::Signal);
        FORCED.store(false, Relaxed);
        assert_eq!(word.load(Acquire), 7, "the bottom-of-pipe write was made");
        // Its list is held by nobody else once the finished ones are given
        // up (our thread may be giving it up just now).
        drop(back);
        drop(take_done());
        let asked = Instant::now();
        while std::sync::Arc::strong_count(&list) > 1 && asked.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(std::sync::Arc::strong_count(&list), 1, "the list came back");
        // Under the lock nothing is left, and the last submission ran.
        let submitted = host.submissions(By::Ours, &[]);
        assert_eq!(VALUE.load(Acquire), 0);
        if let Some(mark) = host.mark() {
            assert!(
                mark.saw(*submitted),
                "the mark shows submission {}",
                *submitted
            );
        }
    }

    /// `VICTORY_COPY_ORDER=split` with submissions not waited for: a piece
    /// split at a list that starts after a copy (`copy_order_parts`) runs
    /// as two submissions, and the second's steps (the copy is its first)
    /// are done only when the first has run and been finished (its work
    /// for after is done). As one submission, every list's steps are done
    /// before any of it has run.
    #[test]
    fn a_split_piece_prepares_its_second_part_when_the_first_has_run() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let _split = crate::mac_switches::tests::ForcedCopyOrder::on();
        for (forced, split) in [(true, true), (false, true), (true, false)] {
            FORCED.store(forced, Relaxed);
            let order = std::sync::Arc::new(Mutex::new(Vec::new()));
            let lists = [0, 1].map(|_| std::sync::Arc::new(host.new_list().unwrap()));
            let log = order.clone();
            lists[0].defer_after(move || log.lock().unwrap().push("the first has run"));
            if split {
                lists[1].starts_after_copy();
            }
            let log = order.clone();
            lists[1].defer_host(move || log.lock().unwrap().push("the second's steps"));
            lists.iter().for_each(|l| l.close());
            let parts = copy_order_parts(lists.to_vec());
            assert_eq!(parts.len(), if split { 2 } else { 1 });
            let mut back = Vec::new();
            for part in parts {
                back.extend(host.execute_title(1, part));
            }
            settle(By::Signal);
            FORCED.store(false, Relaxed);
            let want = if split {
                ["the first has run", "the second's steps"]
            } else {
                ["the second's steps", "the first has run"]
            };
            assert_eq!(*order.lock().unwrap(), want, "not waited for: {forced}, split: {split}");
            drop(back);
            drop(take_done());
        }
    }
}
