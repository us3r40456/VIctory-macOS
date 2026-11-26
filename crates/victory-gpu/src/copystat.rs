//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time in the kernel's write-watch answers, in the view check of the
    /// guard, in protecting; bytes and time of copies of memory neither
    /// watched nor guarded (copied whole).
    KernelUs,
    ViewQueryUs,
    ProtectUs,
    WholeBytes,
    WholeUs,
    /// Snapshots of ranges shaders write (bytes, time); page faults while
    /// preparing submissions and while copying back.
    SnapshotBytes,
    SnapshotUs,
    FaultsPrep,
    FaultsBack,
    /// Pages marked changed because a write landed on another mapping of
    /// them.
    AliasMarks,
    /// Time in the guard handler; protection changes making pages read-only
    /// (calls, pages) and giving write access back (pages).
    HandlerUs,
    ProtectCalls,
    ProtectPages,
    OpenPages,
    /// Tracked pages written (distinct, per frame), and of those the ones
    /// written the frame before too.
    WrittenPages,
    WrittenAgain,
    /// In change checks: time asking the runtime for other mappings, and
    /// waiting for the guard lock.
    AliasUs,
    GuardWaitUs,
    /// Preparing submissions: refresh batches done (copies and change
    /// checks); time in each kind of step (a step that flushes the batch
    /// counts that too); texture uploads for heaps; state preludes.
    FlushUs,
    StepRefreshUs,
    StepCopyUs,
    StepWorkUs,
    StepGpuWriteUs,
    StepTextureUs,
    StepHeapUs,
    StepMemoryUs,
    UploadsUs,
    PreludeUs,
    /// Copying back: bytes of copies compared with their snapshots, and
    /// the time.
    CompareBytes,
    CompareUs,
    /// Copying back pages shaders marked as written through views: pages
    /// compared, bytes written back, the time; pages that kept the title's
    /// bytes (it wrote them, their bytes from before unknown); pages whose
    /// bytes the title's first write saved.
    MarkedPages,
    MarkedBytes,
    MarkedUs,
    MarkedKept,
    SavedPages,
    /// Title copies into memory shown at two addresses, which make every
    /// range of the submission due again.
    AliasedCopies,
    /// Preparing submissions: descriptors the title copied into its heaps
    /// as bytes, made on the host first.
    SyncHeapsUs,
    /// Of those: heap slots looked at (every slot of each page written).
    SyncSlots,
    /// Time in copies redone piece by piece after meeting an unmapped page.
    FallbackUs,
    /// Title threads writing a view into a shader-visible heap: time
    /// waiting for the heaps' lock (held by heap syncs and other writers).
    MirrorWaitUs,
    /// Our heap blocks of [`BIG`] bytes or more made, freed and resized
    /// (blocks, bytes and nanoseconds in the system allocator).
    BigMade,
    BigMadeBytes,
    BigMadeNs,
    BigFreed,
    BigFreedBytes,
    BigFreedNs,
    BigResized,
    BigResizedBytes,
    BigResizedNs,
}
const COUNTERS: usize = C::BigResizedNs as usize + 1;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// One counted copy, a note of it: its source range's start and length,
/// its microseconds, its memory kind. A refresh batch appends one per
/// copy; the dump sums the notes into one entry per source range.
type Note = (usize, usize, u64, u32);

/// Per source range (start, length): copies, bytes, microseconds, memory
/// type — what the dump sums a period of notes into.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;

/// How many note lists there are. Each new thread that copies takes the
/// next in turn; with more threads copying at once than this, two share
/// one, and wait on each other no more than they did on the one map.
const SHARDS: usize = 16;
static NOTES: [Mutex<Option<Vec<Note>>>; SHARDS] = [const { Mutex::new(None) }; SHARDS];

thread_local! {
    /// This thread's note list, as an index into `NOTES`.
    static MINE: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// This thread's note list, the next of `NOTES` for a new thread.
fn mine() -> usize {
    MINE.with(|mine| {
        let mut at = mine.get();
        if at == usize::MAX {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            at = NEXT.fetch_add(1, Relaxed) as usize % SHARDS;
            mine.set(at);
        }
        at
    })
}

/// A refresh batch's note list, held across its copies: each copy of the
/// batch appends its note without taking the lock again — and without
/// the per-copy map visit the summed map charged every copy.
pub(crate) struct Counted(std::sync::MutexGuard<'static, Option<Vec<Note>>>);

/// Takes this thread's note list for a batch of copies.
pub(crate) fn hold() -> Counted {
    Counted(NOTES[mine()].lock().unwrap_or_else(|e| e.into_inner()))
}

impl Counted {
    /// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
    pub(crate) fn range(&mut self, src: usize, len: usize, us: u64, kind: u32) {
        self.0.get_or_insert_with(Vec::new).push((src, len, us, kind));
    }
}

/// Takes every thread's note list, summed: one entry per source range.
/// The lists come out of their locks before summing, so a dump never
/// makes a copy wait while it sums.
fn merged() -> Ranges {
    let mut taken: Vec<Vec<Note>> = Vec::with_capacity(SHARDS);
    for notes in NOTES.iter() {
        if let Some(notes) = notes.lock().unwrap_or_else(|e| e.into_inner()).take() {
            taken.push(notes);
        }
    }
    let mut all: Ranges = HashMap::new();
    for notes in taken {
        for (src, len, us, kind) in notes {
            let e = all.entry((src, len)).or_insert((0, 0, 0, kind));
            e.0 += 1;
            e.1 += len as u64;
            e.2 += us;
        }
    }
    all
}

/// Snapshot bytes per host copy (its title start and length).
static SNAPSHOTS: Mutex<Option<HashMap<(u64, u64), u64>>> = Mutex::new(None);

/// A snapshot of `bytes` of the copy of `[start, start + len)`.
pub(crate) fn snapshot(start: u64, len: u64, bytes: u64) {
    *SNAPSHOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry((start, len))
        .or_default() += bytes;
}

/// Our process's working set, bytes.
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.WorkingSetSize as u64
}

/// Our process's hard page faults so far (read from disk or the page
/// file), from the system's process list: `HardFaultCount` at +0x10 of
/// each entry, the process id at +0x50.
fn hard_faults() -> u64 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    const PROCESSES: u32 = 5;
    let id = std::process::id() as usize;
    let mut buf = vec![0u8; 4 << 20];
    let mut ret = 0;
    // SAFETY: a buffer of that length.
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) } < 0 {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap_or_default()));
        }
        if next == 0 || at + next + 0x58 > buf.len() {
            return 0;
        }
        at += next;
    }
}

pub(crate) fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB ({:.0} into memory \
         shown twice); {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks ({:.1} ms); {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
         guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
         {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
         asking for other mappings, {:.1} ms waiting for the guard lock; preparing: {:.1} ms in \
         refresh batches, steps (each with the batch it flushes): refresh {:.1} ms, copy {:.1} ms, \
         work {:.1} ms, GPU writes {:.1} ms, textures {:.1} ms, heaps {:.1} ms, title memory {:.1} ms; \
         heap texture uploads {:.1} ms, state preludes {:.1} ms, heap slots the title copied {:.1} ms ({:.0} slots), title threads waiting to write views {:.1} ms; copying back compared {:.1} MB          in {:.1} ms; marked pages: {:.0} compared, {:.1} MB written back in {:.1} ms, {:.0} kept the title's, {:.0} saved; working set {:.0} MB, \
         {:.0} hard faults",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::AliasedCopies as usize],
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        ms(c[C::FallbackUs as usize]),
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        ms(c[C::KernelUs as usize]),
        ms(c[C::ViewQueryUs as usize]),
        ms(c[C::ProtectUs as usize]),
        mb(c[C::WholeBytes as usize]),
        ms(c[C::WholeUs as usize]),
        mb(c[C::SnapshotBytes as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::FaultsPrep as usize],
        c[C::FaultsBack as usize],
        c[C::AliasMarks as usize],
        ms(c[C::HandlerUs as usize]),
        c[C::ProtectCalls as usize],
        c[C::ProtectPages as usize],
        c[C::OpenPages as usize],
        c[C::WrittenPages as usize],
        c[C::WrittenAgain as usize],
        ms(c[C::AliasUs as usize]),
        ms(c[C::GuardWaitUs as usize]),
        ms(c[C::FlushUs as usize]),
        ms(c[C::StepRefreshUs as usize]),
        ms(c[C::StepCopyUs as usize]),
        ms(c[C::StepWorkUs as usize]),
        ms(c[C::StepGpuWriteUs as usize]),
        ms(c[C::StepTextureUs as usize]),
        ms(c[C::StepHeapUs as usize]),
        ms(c[C::StepMemoryUs as usize]),
        ms(c[C::UploadsUs as usize]),
        ms(c[C::PreludeUs as usize]),
        ms(c[C::SyncHeapsUs as usize]),
        c[C::SyncSlots as usize],
        ms(c[C::MirrorWaitUs as usize]),
        mb(c[C::CompareBytes as usize]),
        ms(c[C::CompareUs as usize]),
        c[C::MarkedPages as usize],
        mb(c[C::MarkedBytes as usize]),
        ms(c[C::MarkedUs as usize]),
        c[C::MarkedKept as usize],
        c[C::SavedPages as usize],
        working_set() as f64 / 1e6,
        {
            static HARD: AtomicU64 = AtomicU64::new(0);
            let hard = hard_faults();
            (hard - HARD.swap(hard, Relaxed).min(hard)) as f64 / frames
        },
    );
    let held: Vec<String> = crate::watch::held_times()
        .iter()
        .map(|(line, total, longest)| format!("watch.rs:{line} {:.1} ms (longest {longest:.1})", total / frames))
        .collect();
    tracing::info!("  guard lock held per frame: {}", held.join(", "));
    tracing::info!(
        "  heap blocks of {} KiB or more per frame: {:.0} made ({:.1} MB) in {:.2} ms, {:.0} freed ({:.1} MB) \
         in {:.2} ms, {:.0} resized ({:.1} MB) in {:.2} ms",
        BIG >> 10,
        c[C::BigMade as usize],
        mb(c[C::BigMadeBytes as usize]),
        c[C::BigMadeNs as usize] / 1e6,
        c[C::BigFreed as usize],
        mb(c[C::BigFreedBytes as usize]),
        c[C::BigFreedNs as usize] / 1e6,
        c[C::BigResized as usize],
        mb(c[C::BigResizedBytes as usize]),
        c[C::BigResizedNs as usize] / 1e6,
    );
    let mut snapshots: Vec<((u64, u64), u64)> =
        SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).take().into_iter().flatten().collect();
    snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let snapshots: Vec<String> = snapshots
        .iter()
        .take(6)
        .map(|((start, len), bytes)| format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames))
        .collect();
    tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
    let mut ranges: Vec<_> = merged().into_iter().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

/// A global allocator that counts the bytes Rust code holds (the
/// statistics' "our heap"; the binary installs it in diagnostics builds).
pub struct CountingAlloc;

/// Bytes [`CountingAlloc`] has handed out and not taken back.
pub(crate) static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocations of at least this many bytes name their caller, once per
/// power-of-two size class (who holds the big blocks of our heap).
const LARGE: usize = 32 << 20;

/// Blocks this large get virtual memory of their own from Wine's heap
/// (smaller ones come from its pages), so making or freeing one takes
/// Wine's one lock on the process's memory, which every thread's page
/// faults and protection changes wait for. Freeing a written block holds
/// it about 10 us per MB under Rosetta.
const BIG: usize = 1020 << 10;

/// Counts a call on a big block that took from `started` until now.
fn big(counters: [C; 3], bytes: usize, started: std::time::Instant) {
    let [n, b, ns] = counters;
    add(n, 1);
    add(b, bytes as u64);
    add(ns, started.elapsed().as_nanos() as u64);
}

/// Logs the stack of a large allocation of `size` bytes, the first of its
/// size class.
fn note_large(size: usize) {
    static SEEN: std::sync::Mutex<u64> = std::sync::Mutex::new(0);
    std::thread_local!(static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
    let class = 1u64 << size.ilog2().min(63);
    if INSIDE.with(|i| i.replace(true)) {
        return;
    }
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let first = *seen & class == 0;
        *seen |= class;
        first
    };
    if first {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = trace.lines().filter(|l| l.contains("victory")).take(24).collect();
        tracing::info!(mb = size >> 20, "a large heap allocation (first of its size class): {}", frames.join(" | "));
    }
    INSIDE.with(|i| i.set(false));
}

/// The tests and the bench share the note lists: one at a time.
#[cfg(test)]
static SEQUENTIAL: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch counts into its thread's list, and the dump takes every
    /// batch's counts as one entry per range. The holds end before the
    /// dump takes the lists: it takes this thread's too.
    #[test]
    fn batches_count_into_one_dump() {
        let _one = SEQUENTIAL.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut counted = hold();
            counted.range(0x5100_0000_0000, 0x100, 3, 0);
            counted.range(0x5100_0000_0000, 0x100, 4, 0);
            counted.range(0x5100_0000_1000, 0x40, 5, 7);
        }
        {
            let mut counted = hold();
            counted.range(0x5100_0000_1000, 0x40, 6, 7);
        }
        let mut all = merged();
        assert_eq!(all.remove(&(0x5100_0000_0000, 0x100)), Some((2, 0x200, 7, 0)));
        assert_eq!(all.remove(&(0x5100_0000_1000, 0x40)), Some((2, 0x80, 11, 7)));
        assert!(!merged().contains_key(&(0x5100_0000_0000, 0x100)), "the dump took the lists");
    }

    /// Threads count into their own lists, and the dump sums them.
    #[test]
    fn threads_count_into_one_dump() {
        let _one = SEQUENTIAL.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::scope(|scope| {
            for t in 0..4usize {
                scope.spawn(move || {
                    let mut counted = hold();
                    for _ in 0..250 {
                        counted.range(0x5200_0000_0000 + t * 0x1000, 0x100, 3, 0);
                    }
                });
            }
        });
        let mut all = merged();
        for t in 0..4usize {
            assert_eq!(
                all.remove(&(0x5200_0000_0000 + t * 0x1000, 0x100)),
                Some((250, 250 * 0x100, 750, 0))
            );
        }
        assert!(!merged().contains_key(&(0x5200_0000_0000, 0x100)), "the dump took the lists");
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// A refresh frame: 123,000 refreshed ranges, evict3's count, over
    /// eight submissions' batches.
    const REFRESHES: usize = 123_000;
    const BATCHES: usize = 8;
    /// Iterations per scenario; the median is reported.
    const ITERATIONS: u32 = 5;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// The `refreshes` a frame does: `repeated` ranges recopied each
    /// submission (few distinct keys), the rest per-draw scratch (nearly
    /// every call its own key). One pair per call.
    fn mix(lcg: &mut Lcg, repeated: usize, i: usize) -> (usize, usize) {
        if i % 8 < 4 {
            // A streaming buffer's range, recopied: few distinct keys.
            let src = ((lcg.next() as usize) % repeated.max(1)) << 16;
            let len = 64 << 10;
            (src, len)
        } else {
            // Per-draw scratch: nearly every call its own key.
            let src = 0x10_0000_0000 + (((lcg.next() as usize) % 0x1000_0000) << 12);
            let len = (((lcg.next() % 64) + 4) << 10) as usize;
            (src, len)
        }
    }

    /// Counts a frame's refreshes as `flush` does: one held map per
    /// submission batch.
    fn frame(repeated: usize, lcg: &mut Lcg) {
        for _ in 0..BATCHES {
            let mut counted = hold();
            for i in 0..REFRESHES / BATCHES {
                let (src, len) = mix(lcg, repeated, i);
                counted.range(src, len, 7, 0);
            }
        }
    }

    /// Empties the note lists between scenarios.
    fn reset() {
        for notes in NOTES.iter() {
            *notes.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    fn one_thread(repeated: usize, label: &str) {
        let mut times = Vec::new();
        for _ in 0..ITERATIONS {
            let mut lcg = Lcg(0x9e3779b97f4a7c15);
            let started = std::time::Instant::now();
            frame(repeated, &mut lcg);
            times.push(started.elapsed().as_micros() as f64);
            reset();
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!(
            "{label}: {median:.0} us per {REFRESHES} refreshed ranges ({:.1} ns each, median of {ITERATIONS})",
            median * 1000.0 / REFRESHES as f64
        );
    }

    fn four_threads(repeated: usize, label: &str) {
        let mut times = Vec::new();
        for _ in 0..ITERATIONS {
            let took: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
            let ready = std::sync::Barrier::new(5);
            std::thread::scope(|scope| {
                for t in 0..4 {
                    let (t, ready, took) = (t, &ready, &took);
                    scope.spawn(move || {
                        let mut lcg = Lcg(0x9e3779b97f4a7c15 ^ ((t as u64 + 1) * 0x1234));
                        ready.wait();
                        let began = std::time::Instant::now();
                        frame(repeated, &mut lcg);
                        took[t].store(began.elapsed().as_micros() as u64, Relaxed);
                    });
                }
                ready.wait();
            });
            let wall = took.iter().map(|a| a.load(Relaxed)).max().unwrap_or(0);
            times.push(wall as f64);
            reset();
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!(
            "{label}: {median:.0} us wall per {REFRESHES} refreshed ranges ({:.1} ns each, median of {ITERATIONS})",
            median * 1000.0 / REFRESHES as f64
        );
    }

    /// What a dump pays to take and sum a frame's notes, once per
    /// frame-rate line.
    fn dump(label: &str) {
        let mut lcg = Lcg(0x9e3779b97f4a7c15);
        frame(REFRESHES, &mut lcg);
        let started = std::time::Instant::now();
        let merged = merged();
        let took = started.elapsed().as_micros();
        println!("{label}: {took} us for {} ranges", merged.len());
        reset();
    }

    /// Times the counter path as the diagnostics builds pay it: run with
    /// `--ignored --nocapture`, under Wine as the port runs.
    #[test]
    #[ignore]
    fn bench_range() {
        let _one = SEQUENTIAL.lock().unwrap_or_else(|e| e.into_inner());
        one_thread(2048, "one thread, repeated ranges");
        one_thread(REFRESHES, "one thread, scattered ranges");
        four_threads(2048, "four threads, repeated ranges");
        four_threads(REFRESHES, "four threads, scattered ranges");
        dump("dump: sum of a frame's scattered notes");
    }
}

// SAFETY: every call goes to the system allocator unchanged; only counted.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let started = (layout.size() >= BIG).then(std::time::Instant::now);
        // SAFETY: the caller's layout, passed on.
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            if let Some(started) = started {
                big(
                    [C::BigMade, C::BigMadeBytes, C::BigMadeNs],
                    layout.size(),
                    started,
                );
            }
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let started = (layout.size() >= BIG).then(std::time::Instant::now);
        // SAFETY: as above.
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !p.is_null() {
            if let Some(started) = started {
                big(
                    [C::BigMade, C::BigMadeBytes, C::BigMadeNs],
                    layout.size(),
                    started,
                );
            }
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
        HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
        let started = (layout.size() >= BIG).then(std::time::Instant::now);
        // SAFETY: a block this allocator handed out with this layout.
        unsafe { std::alloc::System.dealloc(p, layout) }
        if let Some(started) = started {
            big(
                [C::BigFreed, C::BigFreedBytes, C::BigFreedNs],
                layout.size(),
                started,
            );
        }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let started = (layout.size().max(size) >= BIG).then(std::time::Instant::now);
        // SAFETY: as above.
        let q = unsafe { std::alloc::System.realloc(p, layout, size) };
        if !q.is_null() {
            if let Some(started) = started {
                big(
                    [C::BigResized, C::BigResizedBytes, C::BigResizedNs],
                    size,
                    started,
                );
            }
            HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
            HEAP_BYTES.fetch_add(size as u64, Relaxed);
            if size >= LARGE {
                note_large(size);
            }
        }
        q
    }
}

#[cfg(test)]
mod alloc_tests {
    use super::{BIG, C, COUNT, CountingAlloc, Relaxed};
    use std::alloc::{GlobalAlloc, Layout};

    /// Blocks, bytes made, freed and resized ([`BIG`] blocks only).
    fn big_counts() -> [u64; 6] {
        [
            C::BigMade,
            C::BigMadeBytes,
            C::BigFreed,
            C::BigFreedBytes,
            C::BigResized,
            C::BigResizedBytes,
        ]
        .map(|c| COUNT[c as usize].load(Relaxed))
    }

    /// A block Wine's heap maps on its own is counted when made, resized
    /// and freed; a smaller one is not.
    #[test]
    fn only_blocks_of_their_own_mapping_are_counted() {
        let layout = |size: usize| Layout::from_size_align(size, 16).unwrap();
        let before = big_counts();
        // SAFETY: each block is freed with the layout it was made with.
        unsafe {
            let small = CountingAlloc.alloc(layout(BIG - 4096));
            assert!(!small.is_null());
            CountingAlloc.dealloc(small, layout(BIG - 4096));
            let p = CountingAlloc.alloc(layout(BIG));
            assert!(!p.is_null());
            let q = CountingAlloc.realloc(p, layout(BIG), 2 * BIG);
            assert!(!q.is_null());
            CountingAlloc.dealloc(q, layout(2 * BIG));
        }
        let after = big_counts();
        let grew: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        let big = BIG as u64;
        assert_eq!(grew, [1, big, 1, 2 * big, 1, 2 * big]);
    }
}
