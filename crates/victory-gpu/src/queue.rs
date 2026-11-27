//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Queues the lists' host lists behind what the queue holds, and runs what
/// may run (`crate::hold`; decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let mut held = held();
        for &l in lists {
            for segment in crate::list::segments(l) {
                let waits = segment.waits();
                held.push(this as usize, Item::Segment(segment, waits));
            }
        }
        account().submissions += 1;
        pump(&mut held, Why::Submission);
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// What ran held work: the title's submission, a queue's Signal or Wait, a
/// CPU fence call, PresentX, or our timer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Why {
    Submission,
    Signal,
    Wait,
    Fence,
    Present,
    Timer,
}

#[cfg(windows)]
use crate::hold::Item;
#[cfg(windows)]
use crate::host::unwaited;

/// Every queue's held work.
#[cfg(windows)]
type Holding = crate::hold::Held<std::sync::Arc<crate::host::HostList>>;

#[cfg(windows)]
static HELD: std::sync::Mutex<Holding> = std::sync::Mutex::new(crate::hold::Held::new());

#[cfg(windows)]
fn held() -> std::sync::MutexGuard<'static, Holding> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs held work that may run now (`why` for the frame line).
pub(crate) fn pump_held(why: Why) {
    #[cfg(windows)]
    pump(&mut held(), why);
    #[cfg(not(windows))]
    let _ = why;
}

/// The same unless another thread is at it (calls that poll: a fence's
/// value, our timer).
pub(crate) fn pump_held_if_free(why: Why) {
    #[cfg(windows)]
    if let Ok(mut held) = HELD.try_lock() {
        pump(&mut held, why);
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// How long a queue's head may wait before it runs anyway.
// shortcut: a safety valve against a wait nothing satisfies (a write we do
// not make), which would hold the title's fences until its 30 s watchdog
// ends the run; counted and logged, remove once runs show it never fires.
#[cfg(windows)]
const VALVE: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(windows)]
fn pump(held: &mut Holding, why: Why) {
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    // mac: with flights, a value the submission left to the host GPU will
    // write counts as there (`unwaited::pending_value`): the work behind
    // the wait goes to the GPU behind it.
    let read = |address: u64, bytes: u8| unwaited::pending_value(address, bytes).or_else(|| read_value(address, bytes));
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| crate::fence::value(f as This);
    let started = std::time::Instant::now();
    // In host submissions, and waiting for one the host GPU was left with
    // (`crate::host::unwaited`).
    let ran = std::cell::Cell::new(std::time::Duration::ZERO);
    let mut run = |queue: usize, segments: Vec<std::sync::Arc<crate::host::HostList>>| {
        // One host submission, or with `VICTORY_COPY_ORDER=split` one more
        // for each title copy in the piece that changes what its earlier
        // lists read: the part before it has run when the copy is made.
        for part in crate::host::copy_order_parts(segments) {
            let at = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::PumpExecutes, 1);
            // The lists that have run: these, unless the host GPU was left
            // with them (then they come back with a later piece).
            let done = host.execute_title(queue, part);
            ran.set(ran.get() + at.elapsed());
            done.into_iter().for_each(crate::list::recycle);
        }
    };
    // The title's fence moves once the work before the Signal has run.
    let mut signal = |f: usize, value: u64| {
        // Left to whoever finishes the submission the host GPU has, with
        // the hold on the fence (`VICTORY_ASYNC_SIGNALS`).
        if unwaited::signal_after(f, value) {
            return;
        }
        let at = std::time::Instant::now();
        if unwaited::settle(unwaited::By::Signal) {
            ran.set(ran.get() + at.elapsed());
        }
        crate::fence::set(f as This, value);
        crate::fence::let_go(f as This);
    };
    // A wait may be for a bottom-of-pipe write of a submission the host
    // GPU was left with: that one is finished before a head is held.
    let mut seen = unwaited::finished();
    let mut owed = |waits: &[crate::hold::ValueWait], next: &[&std::sync::Arc<crate::host::HostList>]| {
        let at = std::time::Instant::now();
        let finished = unwaited::settle_waits(waits, next, &mut seen);
        if finished {
            ran.set(ran.get() + at.elapsed());
        }
        finished
    };
    let pumped = held.pump(&mut crate::hold::World {
        read: &read,
        fence: &fence,
        run: &mut run,
        signal: &mut signal,
        owed: &mut owed,
        ahead: unwaited::prewarms(),
    });
    let ran = ran.get();
    if pumped.segments > 0 || pumped.unknown > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pumped.pieces;
        a.segments += pumped.segments;
        a.unknown += pumped.unknown;
        for (queue, waited) in pumped.waited {
            a.released.push((kind(queue), waited, why));
        }
    }
    // Once work stays held, a timer tests it every 2 ms: memory the title
    // writes from the CPU, or the valve, may let it go with no call of ours.
    if held.is_empty() {
        return;
    }
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory held work".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(2));
            pump_held_if_free(Why::Timer);
        });
    });
}

/// The value a wait tests (4 or 8 bytes), if the memory can be read.
#[cfg(windows)]
fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
}

/// A queue's type, for logs (0 graphics, 2 compute, 3 copy). Titles do not
/// release queues that hold work.
#[cfg(windows)]
fn kind(queue: usize) -> u32 {
    self::queue(queue as This).kind
}

/// What queues did with held work since the last presented frame.
#[cfg(windows)]
#[derive(Default)]
struct Account {
    submissions: u32,
    pieces: u32,
    segments: u32,
    /// Segments that waited: queue type, how long, what ran them.
    released: Vec<(u32, std::time::Duration, Why)>,
    unknown: u32,
    forced: u32,
    mappings: u32,
    /// Time in the pump outside host submissions.
    pump: std::time::Duration,
}

#[cfg(windows)]
fn account() -> std::sync::MutexGuard<'static, Account> {
    static ACCOUNT: std::sync::OnceLock<std::sync::Mutex<Account>> = std::sync::OnceLock::new();
    ACCOUNT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame line of held work: what waited, how long and what released it,
/// and what is still held at the present, with the waits that do not hold.
#[cfg(windows)]
fn account_frame(frame: u64) {
    let (cuts, early) = crate::list::take_cuts();
    let (copy_cuts, copy_splits) = (crate::list::take_copy_cuts(), crate::host::take_copy_splits());
    let (fence_waits, fence_waited) = crate::fence::take_waited();
    let a = std::mem::take(&mut *account());
    let blocked = held().blocked(&read_value);
    // `VICTORY_COPY_ORDER=split`: what it did this frame, on a line of its
    // own (the line below is only written when work was held).
    if copy_cuts > 0 || copy_splits > 0 {
        tracing::info!(
            frame,
            copy_cuts,
            copy_splits,
            pieces = a.pieces,
            "copy order this frame: title lists cut at a copy that changes what their earlier commands read; \
             host submissions added for such copies (the cuts, and copies that change what an earlier list of \
             the piece reads); the queues' pieces before the splits"
        );
    }
    if a.released.is_empty() && blocked.is_empty() && a.forced == 0 && a.unknown == 0 && a.mappings == 0 {
        return;
    }
    let longest = a.released.iter().map(|r| r.1).max().unwrap_or_default();
    let mut by = [0u32; 6];
    for r in &a.released {
        by[r.2 as usize] += 1;
    }
    tracing::info!(
        frame,
        submissions = a.submissions,
        pieces = a.pieces,
        segments = a.segments,
        cuts,
        early_resets = early,
        waited = a.released.len(),
        longest_ms = longest.as_secs_f64() * 1e3,
        pump_ms = a.pump.as_secs_f64() * 1e3,
        fence_waits,
        fence_wait_ms = fence_waited.as_secs_f64() * 1e3,
        "held work this frame (submissions; host submissions, segments; lists cut at waits; lists reset or released \
         while held, each keeping its own host list; segments that waited, released by \
         submission/signal/wait/fence/present/timer {by:?}; time in the pump outside host submissions; events the          title waited on for fence values and their summed waits); held at the present: {}",
        blocked.len()
    );
    tracing::debug!(
        frame,
        "held work released this frame (queue type, ms, by): {:?}",
        a.released.iter().map(|(k, d, w)| (k, d.as_secs_f64() * 1e3, w)).collect::<Vec<_>>()
    );
    for (queue, items, waited, unmet) in &blocked {
        let level_warn = blocked.len() > 1;
        let unmet: Vec<String> = unmet
            .iter()
            .map(|w| format!("{:#x} cmp {} {:#x} (is {:?})", w.address, w.comparison, w.reference, read_value(w.address, w.bytes)))
            .collect();
        if level_warn {
            tracing::warn!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "two or more queues hold work at the present");
        } else {
            tracing::info!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "a queue holds work at the present");
        }
    }
    if a.forced > 0 || a.unknown > 0 || a.mappings > 0 {
        tracing::warn!(
            frame,
            forced = a.forced,
            unknown = a.unknown,
            mappings = a.mappings,
            "held work: run without its waits (valve), waits with an unknown comparison (taken as holding), \
             page mappings done ahead of held work"
        );
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they run, so it signals as Signal does.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// At once, or behind the work the queue holds (its fence, and the memory
/// the title reads it at, move only once that work has run).
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let mut held = held();
        if held.holds_work(this as usize) {
            crate::fence::keep(fence);
            held.push(this as usize, Item::Signal(fence as usize, value));
        } else if unwaited::signal_behind(this as usize, fence as usize, value) {
            // The fence moves when the queue's work the host GPU was left
            // with is finished, by whoever finishes it
            // (`VICTORY_ASYNC_SIGNALS`).
        } else {
            // The queue's work the host GPU was left with has run first.
            unwaited::settle_signal(this as usize);
            crate::fence::set(fence, value);
        }
        // Another queue may wait for this fence.
        pump(&mut held, Why::Signal);
    }
    #[cfg(not(windows))]
    {
        let _ = this;
        crate::fence::set(fence, value);
    }
    S_OK
}
/// The queue's later work waits until the fence reaches `value`.
///
/// shortcut: the fence is not kept alive while the queue waits for it;
/// Grand Theft Auto VI makes no queue wait for a fence.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() && crate::fence::value(fence) < value {
        let mut held = held();
        held.push(this as usize, Item::FenceWait(fence as usize, value));
        pump(&mut held, Why::Wait);
    }
    #[cfg(not(windows))]
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Every plane's resources. Only the start of a plane description is
/// known (token, count, resources), so the planes after the first are
/// found by their stride: the first offset past the first plane where the
/// same frame token is followed by a count of 1 to 16 and a list whose
/// first entry is one of our resources. All reads there are checked.
#[cfg(windows)]
fn plane_resources(planes: u32, plane: *const Plane) -> Vec<Vec<This>> {
    let read_u64 = |at: usize| {
        let mut b = [0u8; 8];
        crate::host::read_into(at as *const u8, &mut b).then(|| u64::from_le_bytes(b))
    };
    let list = |at: usize| -> Option<Vec<This>> {
        let count = read_u64(at + 8)? as u32;
        let resources = read_u64(at + 16)? as usize;
        if !(1..=16).contains(&count) || resources == 0 {
            return None;
        }
        let all: Option<Vec<This>> = (0..count as usize)
            .map(|i| read_u64(resources + 8 * i).map(|r| r as This))
            .collect();
        all.filter(|r| r.iter().all(|&r| crate::resource::is_resource(r)))
    };
    let base = plane as usize;
    if planes == 0 || base == 0 {
        return Vec::new();
    }
    let Some(first) = list(base) else {
        return Vec::new();
    };
    let mut all = vec![first];
    if planes > 1
        && let Some(token) = read_u64(base)
        && let Some(stride) = (24..=256)
            .step_by(8)
            .find(|&o| read_u64(base + o) == Some(token) && list(base + o).is_some())
    {
        for k in 1..planes.min(4) as usize {
            match list(base + k * stride) {
                Some(r) => all.push(r),
                None => break,
            }
        }
    }
    all
}

/// Logs what the title presents whenever it changes: each plane's
/// resources as (width, height, host format) (#15: the first plane held a
/// black 1920x1080 texture while a larger render target held the picture).
#[cfg(windows)]
fn log_planes(frame: u64, planes: u32, found: &[Vec<This>], shown: usize) {
    use std::sync::Mutex;
    /// A resource's width, height and host format.
    type Shown = (u64, u32, i32);
    static LAST: Mutex<Option<(u32, Vec<Vec<Shown>>)>> = Mutex::new(None);
    let sizes: Vec<Vec<Shown>> = found
        .iter()
        .map(|plane| {
            plane
                .iter()
                .map(|&r| {
                    crate::resource::host_resource(r).map_or((0, 0, -1), |t| {
                        // SAFETY: a live resource.
                        let d = unsafe { t.GetDesc() };
                        (d.Width, d.Height, d.Format.0)
                    })
                })
                .collect()
        })
        .collect();
    let now = Some((planes, sizes));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last != now {
        let (planes, sizes) = now.as_ref().expect("just set");
        tracing::info!(
            frame,
            planes,
            found = sizes.len(),
            shown,
            resources = ?sizes,
            "presented planes: each plane's resources as (width, height, format); the first resource of plane `shown` is shown"
        );
        *last = now;
    }
}

/// The plane whose first resource is shown: `VICTORY_PRESENT_PLANE=<n>`
/// (counted from 0) when that plane was found, else the plane whose first
/// resource has the most pixels (#15: on Series S plane 0 is a 1920x1080
/// overlay and plane 1 the 3840x2160 picture). Ties keep the earlier plane.
#[cfg(windows)]
fn shown_plane(found: &[Vec<This>]) -> usize {
    static PLANE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let want = *PLANE.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_PLANE").ok().and_then(|v| v.parse().ok())
    });
    let areas: Vec<u64> = found
        .iter()
        .map(|plane| {
            plane.first().and_then(|&r| crate::resource::host_resource(r)).map_or(0, |t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                d.Width * u64::from(d.Height)
            })
        })
        .collect();
    pick_plane(want, &areas)
}

/// The plane to show given an override and each plane's first resource
/// area (0 when unknown): the override when in range, else the largest.
#[cfg(any(windows, test))]
fn pick_plane(want: Option<usize>, areas: &[u64]) -> usize {
    if let Some(w) = want
        && w < areas.len()
    {
        return w;
    }
    let mut best = 0;
    for (i, &a) in areas.iter().enumerate() {
        if a > areas.get(best).copied().unwrap_or(0) {
            best = i;
        }
    }
    best
}

/// Shows the first resource of one plane in our window (the largest
/// plane unless `VICTORY_PRESENT_PLANE` picks another).
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // What may run before the frame is shown does.
    // shortcut: PresentX does not wait behind work the queue still holds (it
    // shows the frame as far as it ran); the frame line names such work.
    pump_held(Why::Present);
    // What ran is finished, before the picture and its state are looked at.
    #[cfg(windows)]
    unwaited::settle(unwaited::By::Present);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    account_frame(frame);
    #[cfg(windows)]
    crate::shadow::trim_pool();
    #[cfg(feature = "diagnostics")]
    crate::frame::presented();
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    // The draws run so far, for the frame recorder's world frame
    // (`VICTORY_RECORD_AT=w<n>`): read where `capture::presented` reads
    // them, so that both tell the same frame as the world's first.
    #[cfg(windows)]
    let drawn = crate::draw::drawn();
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::presented(frame);
        if crate::capture::listed(frame) || crate::capture::has_targets() {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    let found = plane_resources(planes, plane_parameters);
    #[cfg(windows)]
    let shown = shown_plane(&found);
    #[cfg(windows)]
    log_planes(frame, planes, &found, shown);
    #[cfg(all(windows, feature = "diagnostics"))]
    if present_target(frame) {
        // A frame's end for the frame recorder too, without its picture.
        crate::replay::record::present(frame, None, drawn);
        return S_OK;
    }
    #[cfg(windows)]
    if let Some(&resource) = found.get(shown).and_then(|p| p.first())
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        // The other plane's first resource, blended over the picture when
        // `VICTORY_PRESENT_COMPOSITE=1` (#15: a 1080p overlay on Series S).
        let other = (found.len() == 2)
            .then(|| found[1 - shown].first().copied())
            .flatten()
            .filter(|&o| o != resource && crate::composite::wanted())
            .and_then(|o| Some((o, crate::resource::host_resource(o)?)));
        let overlay_state = other.as_ref().map(|(o, _)| {
            crate::resource::host_transition(*o, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before)
        });
        let overlay = other.as_ref().zip(overlay_state).map(|((_, t), s)| (t, s));
        let shown = host.present(&texture, state, overlay, frame);
        if let Some(((o, _), s)) = other.as_ref().zip(overlay_state) {
            crate::resource::host_transition(*o, s);
        }
        crate::resource::host_transition(resource, state);
        // The frame recorder's end of a frame (`VICTORY_RECORD`): the
        // picture is in `state` again.
        crate::replay::record::present(frame, Some((&texture, state)), drawn);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    } else {
        crate::replay::record::present(frame, None, drawn);
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics, `VICTORY_PRESENT_TARGET`: shows the render target it names
/// in place of the shown plane, if frame `frame` used one and the window
/// takes it. False (always, unless the variable is set): the plane is shown
/// as usual. No overlay is blended over a target.
#[cfg(all(windows, feature = "diagnostics"))]
fn present_target(frame: u64) -> bool {
    let Some(target) = crate::capture::present_override(frame) else {
        return false;
    };
    let Some(host) = crate::host::host() else {
        return false;
    };
    let state = crate::host::legal_state(&target.texture, target.state.load(Ordering::Relaxed));
    if host.present(&target.texture, state, None, frame).is_none() {
        return false;
    }
    // The draws traced for `blinks`, which is not asked about this frame.
    crate::draw::take_frame();
    true
}

/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once.
///
/// shortcut: done ahead of work the queue still holds (counted in the frame
/// line); queue it as an item like Signal if that count is not 0.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if held().holds_work(this as usize) {
        account().mappings += 1;
    }
    // The queue's work before it has run: what it owes title memory goes
    // where the pages still are.
    #[cfg(windows)]
    unwaited::settle(unwaited::By::Mapping);
    #[cfg(not(windows))]
    let _ = this;
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed")
        });
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Which plane gets shown: the one `VICTORY_PRESENT_PLANE` names when
    /// it names one, else the plane whose first resource has the most
    /// pixels. It only weighs the areas it is handed, so it runs on every
    /// host.
    #[test]
    fn the_largest_plane_is_shown_unless_one_is_named() {
        // Series S: a 1080p overlay, then the 4K picture.
        let series_s = [1920 * 1080, 3840 * 2160];
        assert_eq!(pick_plane(None, &series_s), 1);
        assert_eq!(pick_plane(Some(0), &series_s), 0);
        // An override past the planes found falls back to the largest.
        assert_eq!(pick_plane(Some(5), &series_s), 1);
        // Series X: one plane. Ties and unknown sizes keep the first.
        assert_eq!(pick_plane(None, &[3840 * 2160]), 0);
        assert_eq!(pick_plane(None, &[0, 0]), 0);
        assert_eq!(pick_plane(None, &[100, 100]), 0);
        assert_eq!(pick_plane(None, &[]), 0);
    }

    /// Two planes 64 bytes apart (as a longer plane description would
    /// place them), each listing one of our textures, are both found; a
    /// list naming something else is not taken for a plane. This reads
    /// title memory through the host, so it runs where a host exists.
    #[cfg(windows)]
    #[test]
    fn planes_after_the_first_are_found_by_their_stride() {
        let texture = || {
            let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
            // SAFETY: a non-zero size; leaked, as title memory outlives a test.
            let address = unsafe { std::alloc::alloc_zeroed(layout) } as u64;
            let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x105, 0, 0];
            let iid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
            let mut out = std::ptr::null();
            let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
            assert_eq!(made, S_OK);
            out
        };
        let (a, b) = (texture(), texture());
        let (list_a, list_b) = (Box::new([a]), Box::new([b]));
        let mut planes = [0u64; 16];
        planes[0] = 7;
        planes[1] = 1;
        planes[2] = list_a.as_ptr() as u64;
        planes[8] = 7;
        planes[9] = 1;
        planes[10] = list_b.as_ptr() as u64;
        let found = plane_resources(2, planes.as_ptr().cast());
        assert_eq!(found, vec![vec![a], vec![b]]);
        // The second plane names a list of something that is not ours.
        let bogus = Box::new([planes.as_ptr() as This]);
        planes[10] = bogus.as_ptr() as u64;
        assert_eq!(plane_resources(2, planes.as_ptr().cast()), vec![vec![a]]);
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
