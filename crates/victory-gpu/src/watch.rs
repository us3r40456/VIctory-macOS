//! Which title memory changed since a host copy took it (decision 0007,
//! "Copies of title memory" and "Change tracking"). Each host copy of title
//! memory (`crate::shadow`) and each texture's staging (`crate::texture`)
//! tracks its memory ([`Tracked`]): the 64 KiB pages it has not taken since
//! they last changed. Taking pages makes their blocks read-only (guards);
//! the first write to a guarded block faults into our vectored handler,
//! which gives write access back and marks the block changed in every copy
//! holding it. Writers that do not fault say so: our own and the kernel's
//! writes into title memory ([`writing`], before they write), and the
//! runtime mapping or unmapping memory ([`title_memory_remapped`]). So
//! bringing a copy up to date costs what changed since, not how much memory
//! it covers: no system call for memory nobody wrote.
//!
//! Only the runtime's GPU memory is tracked: its write-watched pools, its
//! physical-page views and their placeholders. Other memory a copy takes
//! (the C heap, where the kernel writes system call results into buffers a
//! guard would make fail) is copied whole every time.
//!
//! Blocks that cannot be made read-only (placeholders, nothing mapped)
//! read as zeros and count as taken: only a mapping changes them, which the
//! runtime reports.
//!
//! Physical pages may be mapped at several addresses (the title fills
//! pages through a pool view and maps them into its heaps), so a write at
//! one is a change at all of them: the runtime says where else memory is
//! mapped whenever it maps it ([`title_memory_aliased`]); taking a page
//! guards every address showing it, and a write at any of them marks the
//! trackers at all. (Asking the runtime at each take instead waited for
//! its lock, which the title's page mappings hold: 4 s per world frame.)
//! DirectStorage writing pages through views of the runtime's own reports
//! the addresses showing them as written.
//!
//! shortcut: a system call writing into a guarded block fails; the known
//! ones (file reads, DirectStorage) open their blocks first. Memory freed
//! and allocated again while a tracker holds it is not seen.
//!
//! Under Rosetta (Wine on macOS) memory is tracked only when asked for, with
//! the stores Rosetta cannot resume rewritten at load time: see
//! [`fault_tracking`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

/// Whether title writes may fault into a tracker: the guards here and the
/// write watch of the runtime's pools (`victory_hle`'s `XMemVirtualAlloc`,
/// which Wine emulates with page protection). Every store that hits a
/// tracked page must be resumed after the fault. Rosetta cannot resume
/// some: it prints `unexpectedly need to EmulateForward on a synchronous
/// exception` and the process hangs (VIctory-macOS issue #1). On macOS 27
/// those are AVX masked stores that start in writable memory and end in a
/// protected page, and `vcvtps2ph` stores anywhere in one; the loader
/// rewrites both as plain stores whenever tracking runs under Rosetta
/// ([`stores_need_patching`]). Earlier Rosetta also stops on a plain 32-byte
/// `vmovups` across pages (macOS 26.6, #1), and #144 hung on an M1 with
/// both switches on, so under Rosetta tracking stays off unless
/// `VICTORY_FAULT_TRACKING=1` asks for it. Without tracking, copies take
/// their memory whole every time, as they do for memory that is not the
/// runtime's (tens of GB per world frame, #50). `0` turns tracking off
/// anywhere. Tests keep it on: their stores do not cross pages.
pub fn fault_tracking() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide(
            std::env::var("VICTORY_FAULT_TRACKING").ok().as_deref(),
            cfg!(test) || !rosetta(),
        )
    });
    *ON
}

/// Whether tracking runs under Rosetta, so the stores Rosetta cannot resume
/// after a fault must be rewritten at load time (`victory_loader::masked`).
pub fn stores_need_patching() -> bool {
    fault_tracking() && rosetta()
}

/// `fault_tracking` from the variable's value and the default.
fn decide(var: Option<&str>, default: bool) -> bool {
    match var {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

/// Whether x86-64 code here runs under Rosetta: Wine on macOS, whose
/// x86-64 Wine only runs that way on the Apple silicon Macs D3DMetal needs.
pub fn rosetta() -> bool {
    use std::ffi::{CStr, c_char};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    type HostVersion = unsafe extern "C" fn(*mut *const c_char, *mut *const c_char);
    // SAFETY: ntdll is always loaded and both names end with a NUL. Wine's
    // `wine_get_host_version` (cdecl, two out pointers) sets them to static
    // strings: the host's `uname` system name and release.
    unsafe {
        let Some(f) = GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_host_version".as_ptr().cast(),
        ) else {
            return false;
        };
        let f: HostVersion = std::mem::transmute(f);
        let (mut sysname, mut release) = (std::ptr::null(), std::ptr::null());
        f(&mut sysname, &mut release);
        !sysname.is_null() && CStr::from_ptr(sysname) == c"Darwin"
    }
}

/// The unit tracked and copied: 64 KiB, the console's page. Finer units
/// made the checks of the world's big buffers cost more than the copies
/// they saved.
pub(crate) const PAGE: usize = 64 << 10;

/// The pages a tracker has not taken since they changed, what it saves of
/// pages it armed, its bit in [`DIRTY`] (none: always taken as dirty), and
/// whether it is a host buffer copy's (the only trackers [`retake`] marks).
struct Changed(Mutex<BTreeSet<usize>>, Mutex<Saved>, Option<u32>, bool);

/// A bit per tracker, set while it holds changed pages: set and cleared
/// under the tracker's lock, read without one ([`Tracked::clean`]). Dense,
/// so that a submission asking about every copy its heaps reach (~14,000)
/// reads a few KB instead of a cache line per copy.
static DIRTY: [AtomicU64; DIRTY_WORDS] = [const { AtomicU64::new(0) }; DIRTY_WORDS];
const DIRTY_WORDS: usize = 1 << 14;

/// Bits of [`DIRTY`] given back, and the first never handed out.
static FREE_BITS: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));

fn dirty_bit() -> Option<u32> {
    let mut free = FREE_BITS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bit) = free.0.pop() {
        return Some(bit);
    }
    let bit = free.1;
    (bit < (DIRTY_WORDS * 64) as u32).then(|| {
        free.1 += 1;
        bit
    })
}

fn set_dirty(bit: Option<u32>, dirty: bool) {
    if let Some(bit) = bit {
        let (word, mask) = (&DIRTY[bit as usize / 64], 1u64 << (bit % 64));
        if dirty {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }
}

/// Whether the tracker with bit `bit` holds no changed page (see
/// [`Tracked::clean`]).
pub(crate) fn clean_bit(bit: Option<u32>) -> bool {
    bit.is_some_and(|bit| DIRTY[bit as usize / 64].load(Ordering::Acquire) & 1 << (bit % 64) == 0)
}

impl Drop for Changed {
    fn drop(&mut self) {
        if let Some(bit) = self.2 {
            set_dirty(Some(bit), false);
            FREE_BITS.lock().unwrap_or_else(|e| e.into_inner()).0.push(bit);
        }
    }
}

/// Pages (page numbers) whose bytes a tracker keeps when they are first
/// written after it took them, and those bytes: while shaders may be
/// writing its copy of them, what it held before (`crate::shadow`'s
/// copy-back of marked pages), with what the copy took of the page since
/// laid over them ([`Tracked::refreshed`]).
#[derive(Default)]
struct Saved {
    /// The armed pages as runs `[first, last)`, sorted, apart and not
    /// touching (runs armed that meet merge): a root view declares its
    /// copy from its address to the end, the world's 174 MB heap many
    /// times a submission, and a set entry per page cost 30 microseconds a
    /// declaration; descriptor tables arm many sub-ranges end to end, and
    /// the write handler checks a page under the guards' lock.
    armed: Vec<(usize, usize)>,
    /// Pages of those runs that are not armed any more.
    unarmed: BTreeSet<usize>,
    bytes: std::collections::HashMap<usize, Box<[u8]>>,
}

impl Saved {
    /// Whether page `page` is armed.
    fn armed(&self, page: usize) -> bool {
        // The first run ending after `page` is the only one that can hold it.
        let i = self.armed.partition_point(|&(_, last)| last <= page);
        self.armed.get(i).is_some_and(|&(first, _)| first <= page) && !self.unarmed.contains(&page)
    }

    /// Arms pages `[first, last)`.
    fn arm(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.unarmed.is_empty() {
            let pages: Vec<usize> = self.unarmed.range(first..last).copied().collect();
            for page in pages {
                self.unarmed.remove(&page);
            }
        }
        // Runs `[lo, hi)` overlap or touch the new one: it replaces them.
        let lo = self.armed.partition_point(|&(_, b)| b < first);
        let hi = self.armed.partition_point(|&(a, _)| a <= last);
        if lo + 1 == hi && self.armed[lo].0 <= first && last <= self.armed[lo].1 {
            return;
        }
        let run = if lo < hi {
            (self.armed[lo].0.min(first), self.armed[hi - 1].1.max(last))
        } else {
            (first, last)
        };
        self.armed.splice(lo..hi, [run]);
    }
}

impl Changed {
    /// Page `page` was taken: what it saved of it is stale.
    fn taken(&self, page: usize) {
        self.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page);
    }

    /// Page `page` is about to change for the first time since it was
    /// taken: its bytes now, if armed for it.
    fn save(&self, page: usize) {
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.armed(page) || saved.bytes.contains_key(&page) {
            return;
        }
        let mut bytes = vec![0u8; PAGE].into_boxed_slice();
        if crate::host::read_into((page * PAGE) as *const u8, &mut bytes) {
            saved.bytes.insert(page, bytes);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SavedPages, 1);
        }
    }
}

/// Byte runs `[start, end)` in order.
pub(crate) type Runs = Vec<(usize, usize)>;

/// Title memory a host copy or texture staging takes: bytes `[start, end)`.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    changed: Arc<Changed>,
    /// Parts of the range on pages that are not the runtime's memory (in
    /// order): not tracked, copied whole at each use ([`Tracked::untracked`]).
    untracked: Vec<(usize, usize)>,
}

struct Guards {
    /// Blocks (pages) we made read-only.
    protected: BTreeSet<usize>,
    /// The trackers holding each page.
    holders: BTreeMap<usize, Vec<Arc<Changed>>>,
    /// Pages `[first, last)` of writes the guards do not see that are under
    /// way ([`Writing`]): not guarded until they end, or the write fails.
    busy: Vec<(usize, usize)>,
    /// Pages showing the same physical pages elsewhere, both ways, as the
    /// runtime told when it mapped them: a write at one marks the others,
    /// with no question to the runtime (whose lock its page mappings hold
    /// for long).
    alias_of: BTreeMap<usize, Vec<usize>>,
    /// From 64bitMango 131721b7 (`VICTORY_VETTED_PAGES=1`): pages found to be
    /// the runtime's GPU memory when they were last guarded, with the page
    /// their region was known to end before: guarded again after a write
    /// opened them, they are not asked about again. Forgotten when the
    /// runtime maps or unmaps them and when no tracker holds them any more.
    vetted: BTreeMap<usize, usize>,
    /// Tracked pages written this frame and the frame before (diagnostics).
    #[cfg(feature = "diagnostics")]
    written: BTreeSet<usize>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    holders: BTreeMap::new(),
    busy: Vec::new(),
    alias_of: BTreeMap::new(),
    vetted: BTreeMap::new(),
    #[cfg(feature = "diagnostics")]
    written: BTreeSet::new(),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

/// mac: since the last frame line, microseconds threads waited for the
/// guard lock (in the title's write faults, elsewhere), microseconds in
/// protection calls and their number ([`frame_line`]). A lock taken at once
/// is not timed; a clock read costs a Wine call.
static TIMES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Whether pages guarded before are guarded again without asking what memory
/// they are (`VICTORY_VETTED_PAGES=1`, from 64bitMango 131721b7; off by default).
fn vetted_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_VETTED_PAGES"))
}

/// mac: since the last frame line, the exceptions the write-fault handler
/// (the first vectored handler) saw: write faults that opened a guarded
/// block, that found it writable already (another thread's fault opened it),
/// that were not ours, and every other exception.
static FAULTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn fault(kind: usize) {
    FAULTS[kind].fetch_add(1, Ordering::Relaxed);
}

thread_local! {
    /// mac: whether this thread is in the write-fault handler ([`TIMES`]).
    static IN_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A protection call `f`, timed and counted ([`TIMES`]).
fn timed_protect(f: impl FnOnce() -> bool) -> bool {
    let started = std::time::Instant::now();
    let done = f();
    TIMES[2].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    TIMES[3].fetch_add(1, Ordering::Relaxed);
    done
}

/// From 64bitMango e4444060 (`VICTORY_OPENED_FAULTS=1`, off by default):
/// blocks a title write fault opened that nothing has guarded since, by a
/// slot their number picks (0: none). Set under the guard lock by the fault
/// that opened the block, after every tracker holding it was marked;
/// cleared before the block is guarded again (and when it is remapped or
/// no tracker holds it); read without the lock by a fault on a block
/// another thread's fault opened meanwhile, which then has nothing to do.
/// One block's entry may push out another's: that one's next fault takes
/// the lock, as before. Fixed and never allocated: the handler writes it.
static OPENED_BLOCKS: [AtomicU64; 1 << 12] = [const { AtomicU64::new(0) }; 1 << 12];

/// Whether faults on blocks found open skip the lock (`VICTORY_OPENED_FAULTS`).
fn opened_faults() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_OPENED_FAULTS"))
}

fn opened_slot(page: usize) -> &'static AtomicU64 {
    &OPENED_BLOCKS[((page as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 52) as usize % OPENED_BLOCKS.len()]
}

/// A title write fault opened `page` (under the guard lock).
fn note_opened(page: usize) {
    if opened_faults() {
        opened_slot(page).store(page as u64 + 1, Ordering::Release);
    }
}

/// `[first, last)` is about to be guarded, or shows something new: not
/// open from a fault any more.
fn forget_opened(first: usize, last: usize) {
    if opened_faults() {
        for page in first..last {
            let _ = opened_slot(page).compare_exchange(page as u64 + 1, 0, Ordering::AcqRel, Ordering::Relaxed);
        }
    }
}

/// Whether a fault opened `page` and nothing guarded it since.
fn opened(page: usize) -> bool {
    opened_faults() && opened_slot(page).load(Ordering::Acquire) == page as u64 + 1
}

thread_local! {
    /// The last block this thread found open, and how many faults in a row
    /// it has had there: past three it takes the lock, should the block be
    /// read-only for another reason.
    static FOUND_OPEN: std::cell::Cell<(usize, u8)> = const { std::cell::Cell::new((0, 0)) };
}

/// mac: the end of the frame line, per frame over `frames` frames: the
/// guard lock's waits and the protection calls.
pub(crate) fn frame_line(frames: f64) -> String {
    let [faults, others, protect, calls, vetted] =
        TIMES.each_ref().map(|t| t.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let [opened, writable, foreign, other, found] =
        FAULTS.each_ref().map(|t| t.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    format!(
        "; guard lock waits {:.1} ms ({:.1} ms of it in write faults), protection calls {:.1} ms ({calls:.0}), {vetted:.0} pages guarded again unasked (VICTORY_VETTED_PAGES); \
         exceptions {:.0}: write faults {opened:.0} opened a guarded block, {writable:.0} found it writable, \
         {foreign:.0} not ours, {found:.0} found it open (no lock, VICTORY_OPENED_FAULTS), {other:.0} others",
        (faults + others) / 1e3,
        faults / 1e3,
        protect / 1e3,
        opened + writable + foreign + other + found
    )
}

thread_local! {
    /// Whether this thread holds the guard lock. A guarded block it faults
    /// on then is one the lock's own work touched (#50: an allocation made
    /// under the lock landed in a guarded block of the host heap, which
    /// title XMem blocks share), and taking the lock again would hang it.
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Blocks opened by faults taken under the guard lock, for its next holder
/// to mark changed (0: free slot). Fixed: the handler must not allocate.
static OPENED_UNDER_LOCK: [std::sync::atomic::AtomicUsize; 64] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 64];

/// Whether any block waits in them: the next holder of the lock scans
/// [`OPENED_UNDER_LOCK`] only when one was opened, not on every take. Only
/// touched while the lock is held, so a clear scan leaves no block behind.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A fault on the block `page` by the thread holding the guard lock: write
/// access back now, marked changed by the lock's next holder.
// shortcut: assumes a read-only block this thread writes while
// holding the lock is one of ours; with all 64 slots taken the block stays
// open without being marked (counted); a bigger table if that shows up.
fn open_under_lock(page: usize) {
    use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    // SAFETY: a block of this process; fails harmlessly if it is gone.
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    if OPENED_UNDER_LOCK.iter().any(|s| s.compare_exchange(0, page, AcqRel, Relaxed).is_ok()) {
        OPENED.store(true, Release);
    } else {
        static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        LOST.fetch_add(1, Relaxed);
    }
}

/// The guard lock, held while this lives. With diagnostics, timed for the
/// line that took it ([`held_times`]).
struct Held {
    guard: Option<std::sync::MutexGuard<'static, Guards>>,
    #[cfg(feature = "diagnostics")]
    since: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    line: u32,
}

#[track_caller]
fn guards() -> Held {
    use std::sync::atomic::Ordering::AcqRel;
    // mac: a wait for the lock is timed ([`TIMES`]).
    let guard = match GUARDS.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let started = std::time::Instant::now();
            let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
            let k = usize::from(!IN_FAULT.with(|f| f.get()));
            TIMES[k].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            guard
        }
    };
    HOLDING.with(|h| h.set(true));
    let mut held = Held {
        guard: Some(guard),
        #[cfg(feature = "diagnostics")]
        since: std::time::Instant::now(),
        #[cfg(feature = "diagnostics")]
        line: std::panic::Location::caller().line(),
    };
    if OPENED.swap(false, AcqRel) {
        for slot in &OPENED_UNDER_LOCK {
            let page = slot.swap(0, AcqRel);
            if page != 0 {
                changed(&mut held, page, page + 1, false);
            }
        }
    }
    held
}

impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.guard.as_ref().expect("held until dropped")
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.guard.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

impl Drop for Held {
    fn drop(&mut self) {
        drop(self.guard.take());
        HOLDING.with(|h| h.set(false));
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let us = self.since.elapsed().as_micros() as u64;
            let line = u64::from(self.line);
            if let Some(slot) = HELD
                .iter()
                .find(|s| s[0].load(Relaxed) == line || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok())
            {
                slot[1].fetch_add(us, Relaxed);
                slot[2].fetch_max(us, Relaxed);
            }
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| (s[0].load(Relaxed), s[1].swap(0, Relaxed) as f64 / 1e3, s[2].swap(0, Relaxed) as f64 / 1e3))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// (page, other page) for the pages (page numbers) showing the same memory
/// as `pages` elsewhere.
fn aliases(g: &Guards, pages: &[usize]) -> Vec<(usize, usize)> {
    pages
        .iter()
        .flat_map(|&p| g.alias_of.get(&p).into_iter().flatten().map(move |&a| (p, a)))
        .collect()
}

/// Guards the other mappings of taken pages, returning those that still
/// have an unguarded mapping (cannot be taken yet).
fn guard_aliases(g: &mut Guards, pairs: &[(usize, usize)]) -> BTreeSet<usize> {
    if pairs.is_empty() {
        return BTreeSet::new();
    }
    let others: Vec<usize> = pairs.iter().map(|&(_, a)| a).collect();
    let guarded: BTreeSet<usize> = protect(g, &others).into_iter().collect();
    pairs
        .iter()
        .filter(|(_, a)| !guarded.contains(a))
        .map(|&(p, _)| p)
        .collect()
}

/// Tracks `[start, start + len)` if it is the runtime's GPU memory (every
/// page changed at first); `None` for other memory, and with `views` false
/// for anything but its write-watched pools (physical-page views are
/// written through other views showing the same pages: the title fills
/// streamed textures and video planes through a pool view and maps the
/// pages into its tiled heap, where a guard sees none of it).
///
/// Only the pages of the range that are such memory are tracked; the rest is
/// never guarded and is copied whole at each use, as memory with no tracker.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, false)
}

/// [`track`] for a host buffer copy: what shaders write lands in it, so a
/// [`retake`] of its memory marks it.
pub(crate) fn track_copy(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, true)
}

fn track_as(start: usize, len: usize, views: bool, copy: bool) -> Option<Tracked> {
    if len == 0 || !fault_tracking() {
        return None;
    }
    let end = start + len;
    let pages = runtime_pages(start, end, views);
    if pages.is_empty() {
        return None;
    }
    install();
    // The rest of the range, as byte runs.
    let mut untracked: Vec<(usize, usize)> = Vec::new();
    let mut at = start;
    for &page in pages.iter().chain([&usize::MAX]) {
        let (a, b) = ((page.saturating_mul(PAGE)).max(start).min(end), (page.saturating_add(1).saturating_mul(PAGE)).min(end));
        if at < a {
            untracked.push((at, a));
        }
        at = at.max(b);
    }
    // A tracker with untracked parts is never clean: no bit.
    let bit = if untracked.is_empty() { dirty_bit() } else { None };
    set_dirty(bit, true);
    let changed = Arc::new(Changed(Mutex::new(pages.iter().copied().collect()), Mutex::default(), bit, copy));
    let mut g = guards();
    for &page in &pages {
        g.holders.entry(page).or_default().push(changed.clone());
    }
    drop(g);
    Some(Tracked {
        start,
        end,
        changed,
        untracked,
    })
}

/// Whether `protect_noting` asks about runs of pages at once
/// (`VICTORY_WATCH_RUNS=1`, off by default).
fn watch_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_WATCH_RUNS"))
}

/// Whether all of `[at, at + len)` lies in one write-watched pool (one
/// lookup: the watch answers for a range only when one watched allocation
/// holds all of it).
fn write_watched_range(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    unsafe { GetWriteWatch(0, at as *const _, len, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
}

/// Whether the page holding `at` is in a write-watched pool (one lookup).
fn write_watched(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let page = at & !(4096 - 1);
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    unsafe { GetWriteWatch(0, page as *const _, 4096, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
}

/// The pages of `[start, end)` that are the runtime's GPU memory: a
/// write-watched pool, or (with `views`) anything but plain committed memory
/// and images (views, placeholders, nothing mapped yet). Decided for every
/// region the range covers, and a page touching any other memory is left
/// out: a range can run past its allocation (a 4 MiB window of memory
/// outside buffers, `crate::shadow`), and when the first page alone decided,
/// a window starting in free memory had the process heap's pages after it
/// guarded; the heap's own code then faulted while holding its lock, and the
/// guard handler waited for the guard lock, whose holder waited for the heap
/// lock (the crate's tests hung).
///
/// shortcut: memory free or reserved when the range is tracked counts as the
/// runtime's; another allocator committing it before the first take would
/// have it guarded.
fn runtime_pages(start: usize, end: usize, views: bool) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let watched = |at: usize, len: usize| {
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; no reset, so nothing changes.
        unsafe { GetWriteWatch(0, at as *const _, len, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
    };
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // One pool holds it all (the common case; asking `VirtualQuery` costs
    // milliseconds in the runtime's big pools).
    if watched(start & !(4096 - 1), end - (start & !(4096 - 1))) {
        return (first..last).collect();
    }
    let (mut runtime, mut other) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    let mut at = start & !(4096 - 1);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize).max(at + 1).min(end);
        let ours = watched(at, next - at)
            || views && info.Type != MEM_IMAGE && !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        (if ours { &mut runtime } else { &mut other }).extend(at / PAGE..next.div_ceil(PAGE));
        at = next;
    }
    runtime.difference(&other).copied().collect()
}

impl Drop for Tracked {
    /// Pages no tracker holds any more get write access back, so that the
    /// guard set never names memory freed and allocated again; unless another
    /// mapping of the page is held: the guard here is that tracker's too (it
    /// was set when that tracker took its page), and with it opened a write
    /// here went unseen there.
    fn drop(&mut self) {
        let mut g = guards();
        let mut free = Vec::new();
        for page in self.start / PAGE..self.end.div_ceil(PAGE) {
            if let Some(holders) = g.holders.get_mut(&page) {
                holders.retain(|c| !Arc::ptr_eq(c, &self.changed));
                if holders.is_empty() {
                    g.holders.remove(&page);
                    free.push(page);
                }
            }
        }
        for page in free {
            let watched_elsewhere = g.alias_of.get(&page).is_some_and(|o| o.iter().any(|p| g.holders.contains_key(p)));
            if !watched_elsewhere {
                g.vetted.remove(&page);
                open(&mut g, page);
                // Nobody's any more: what is there next is not known open.
                forget_opened(page, page + 1);
            }
        }
    }
}

/// Gives write access back to page `page` if we guarded it.
fn open(g: &mut Guards, page: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if !g.protected.remove(&page) {
        return false;
    }
    let mut old = 0;
    // SAFETY: a block we made read-only (failing harmlessly if it is gone).
    timed_protect(|| unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) } != 0);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, 1);
    true
}

/// [`open`] of the guarded pages `[first, last)`, all in `protected`.
fn open_run(g: &mut Guards, first: usize, last: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if last - first == 1 {
        open(g, first);
        return;
    }
    let mut old = 0;
    // SAFETY: blocks we made read-only.
    if !timed_protect(|| unsafe { VirtualProtect((first * PAGE) as *const _, (last - first) * PAGE, PAGE_READWRITE, &mut old) } != 0) {
        for page in first..last {
            open(g, page);
        }
        return;
    }
    for page in first..last {
        g.protected.remove(&page);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, (last - first) as u64);
}

impl Tracked {
    /// Takes the changed pages of `[from, to)`: guards them (writes from
    /// now on mark them again) and returns the byte ranges to copy, cut to
    /// `[from, to)`, in order. A page whose part in the tracked memory lies
    /// only partly in `[from, to)` is copied in part and stays changed.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (copy, zero) = self.take_split(from, to);
        let mut runs: Vec<(usize, usize)> = copy.into_iter().chain(zero).collect();
        runs.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match merged.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// [`Tracked::take`], the runs on pages with nothing mapped apart: they
    /// read as zeros, with no copy to try (each such page cost a dozen
    /// failing kernel reads, 7-30 ms a frame in run perf4).
    pub(crate) fn take_split(&self, from: usize, to: usize) -> (Runs, Runs) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return (Vec::new(), Vec::new());
        }
        let pages: Vec<usize> = {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
            let changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(feature = "diagnostics")]
            drop(waited);
            changed.range(from / PAGE..to.div_ceil(PAGE)).copied().collect()
        };
        if pages.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut whole: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&p| (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to)
            .collect();
        // Every address showing the same pages is guarded too. A write in
        // between marks them changed again; the copy, made after, has it.
        #[cfg(feature = "diagnostics")]
        let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
        let mut g = guards();
        let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        drop(waited);
        #[cfg(feature = "diagnostics")]
        let asked = Timed(std::time::Instant::now(), crate::copystat::C::AliasUs);
        let others = aliases(&g, &whole);
        let unguarded = guard_aliases(&mut g, &others);
        #[cfg(feature = "diagnostics")]
        drop(asked);
        whole.retain(|p| !unguarded.contains(p));
        let mut unmapped = Vec::new();
        for page in protect_noting(&mut g, &whole, &mut unmapped) {
            changed.remove(&page);
            self.changed.taken(page);
        }
        set_dirty(self.changed.2, !changed.is_empty());
        drop(changed);
        drop(g);
        let (mut copy, mut zero): (Runs, Runs) = (Vec::new(), Vec::new());
        for p in pages {
            let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
            let runs = if unmapped.contains(&p) { &mut zero } else { &mut copy };
            match runs.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => runs.push((a, b)),
            }
        }
        (copy, zero)
    }

    /// Whether page `page` (a page number) changed since it was taken.
    pub(crate) fn changed(&self, page: usize) -> bool {
        self.changed
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&page)
    }

    /// The caller made pages (page numbers) of its copy equal to title
    /// memory itself (what shaders wrote went back): taken and guarded.
    pub(crate) fn taken(&self, pages: &[usize]) {
        let mut g = guards();
        let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        let others = aliases(&g, pages);
        let unguarded = guard_aliases(&mut g, &others);
        let pages: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|p| !unguarded.contains(p))
            .collect();
        for page in protect(&mut g, &pages) {
            changed.remove(&page);
            self.changed.taken(page);
        }
        set_dirty(self.changed.2, !changed.is_empty());
    }

    /// Whether no page changed since it was taken: then [`Tracked::take`]
    /// finds nothing, and asking costs no lock. A write racing this answer
    /// is found by the next asking, as one racing a take.
    pub(crate) fn clean(&self) -> bool {
        clean_bit(self.changed.2)
    }

    /// The tracker's bit for [`clean_bit`], for callers that ask about
    /// many trackers at once without touching each (none: never clean).
    pub(crate) fn bit(&self) -> Option<u32> {
        self.changed.2
    }

    /// The parts of `[from, to)` that are not tracked (see
    /// [`Tracked::untracked`] field): to be copied whole at each use.
    pub(crate) fn untracked(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.untracked
            .iter()
            .map(|&(a, b)| (a.max(from), b.min(to)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Shaders may write the copy's bytes `[from, to)` until [`Tracked::disarm`]:
    /// pages there written next after they were taken have their bytes kept
    /// first ([`Tracked::saved`]).
    pub(crate) fn arm(&self, from: usize, to: usize) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from < to {
            self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).arm(from / PAGE, to.div_ceil(PAGE));
        }
    }

    /// Page `page` (a page number) needs no saving any more: its copy-back
    /// is under way.
    pub(crate) fn unarm(&self, page: usize) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.armed(page) {
            saved.unarmed.insert(page);
        }
    }

    /// No shader writes the copy any more: nothing is armed or kept.
    pub(crate) fn disarm(&self) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        saved.armed.clear();
        saved.unarmed.clear();
        saved.bytes.clear();
    }

    /// The bytes page `page` (a page number) held when it was first written
    /// since it was taken, if it was armed then (and so changed), with what
    /// the copy took of it since: the copy's bytes before the submission.
    pub(crate) fn saved(&self, page: usize) -> Option<Box<[u8]>> {
        self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page)
    }

    /// The copy took `bytes` at title address `from`: any saved bytes of pages
    /// it covers take them too, so marked copy-back neither takes a partial
    /// refresh for the GPU's nor drops a shader write back to the pre-fault byte.
    pub(crate) fn refreshed(&self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.bytes.is_empty() {
            return;
        }
        let to = from + bytes.len();
        for page in from / PAGE..to.div_ceil(PAGE) {
            if let Some(before) = saved.bytes.get_mut(&page) {
                let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
                before[a - page * PAGE..b - page * PAGE]
                    .copy_from_slice(&bytes[a - from..b - from]);
            }
        }
    }
}

/// Makes `pages` read-only where not already (in runs), and returns those
/// now guarded or with nothing mapped (taken either way); a page that is
/// mapped but cannot be guarded is left out (it stays changed).
fn protect(g: &mut Guards, pages: &[usize]) -> Vec<usize> {
    protect_noting(g, pages, &mut Vec::new())
}

/// [`protect`], also listing in `unmapped` the pages taken because nothing
/// is mapped there.
fn protect_noting(g: &mut Guards, pages: &[usize], unmapped_pages: &mut Vec<usize>) -> Vec<usize> {
    #[cfg(feature = "diagnostics")]
    let counted = |calls: u64, pages: u64| {
        crate::copystat::add(crate::copystat::C::ProtectCalls, calls);
        crate::copystat::add(crate::copystat::C::ProtectPages, pages);
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::ProtectUs);
    let set = |from: usize, to: usize| {
        #[cfg(feature = "diagnostics")]
        counted(1, (to - from) as u64);
        #[cfg(test)]
        tests::PROTECTS.with(|n| n.set(n.get() + 1));
        let mut old = 0;
        // Before it is read-only: a fault that still finds it open writes
        // while it is open, ahead of the caller's copy.
        forget_opened(from, to);
        // SAFETY: title memory; only its protection changes.
        timed_protect(|| unsafe { VirtualProtect((from * PAGE) as *const _, (to - from) * PAGE, PAGE_READONLY, &mut old) != 0 })
    };
    let unmapped = |page: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        n == 0 || info.State != MEM_COMMIT
    };
    let mut taken = Vec::with_capacity(pages.len());
    let busy = |page: usize| g.busy.iter().any(|&(a, b)| (a..b).contains(&page));
    let pages: Vec<usize> = pages.iter().copied().filter(|&p| !busy(p)).collect();
    // Only the runtime's GPU memory is guarded. A tracked range can start
    // in free or reserved memory and run on into the host's own pages: in
    // #50 that guarded the process heap's header, and a thread holding the
    // heap lock then faulted while another held `Guards` and allocated,
    // a deadlock. Pages left out stay changed, so they are copied whole.
    // Pages guarded already passed this check when they were guarded.
    // With each page, where its region is known to end: `VirtualProtect`
    // takes one view at a time under Wine, so a run that crossed two views
    // (the runtime's page mappings make one per page) failed and was then
    // guarded page by page, one call more than needed.
    let mut allowed = RuntimeRegions::default();
    // From 64bitMango 131721b7: asking costs up to three calls into Wine per
    // page of a physical-page view (one view a page) under this lock, which
    // the title's write faults wait for; what the answer depends on changes
    // only when the runtime maps or unmaps the memory (`VICTORY_VETTED_PAGES`).
    let vetted = vetted_pages();
    // Where the run of consecutive pages each page starts ends: a pool is
    // asked about once for the run (`VICTORY_WATCH_RUNS`).
    let mut run_ends = vec![0; pages.len()];
    for i in (0..pages.len()).rev() {
        run_ends[i] = match pages.get(i + 1) {
            Some(&next) if next == pages[i] + 1 => run_ends[i + 1],
            _ => pages[i] + 1,
        };
    }
    let (pages, ends): (Vec<usize>, Vec<usize>) = pages
        .into_iter()
        .zip(run_ends)
        .filter_map(|(p, run_end)| {
            if g.protected.contains(&p) {
                return Some((p, usize::MAX));
            }
            if vetted && let Some(&end) = g.vetted.get(&p) {
                TIMES[4].fetch_add(1, Ordering::Relaxed);
                return Some((p, end));
            }
            allowed.region(p, run_end).map(|end| (p, end))
        })
        .unzip();
    let mut i = 0;
    while i < pages.len() {
        if g.protected.contains(&pages[i]) {
            taken.push(pages[i]);
            i += 1;
            continue;
        }
        // A run of consecutive pages not guarded yet, in one region.
        let first = i;
        while i + 1 < pages.len()
            && pages[i + 1] == pages[i] + 1
            && pages[i + 1] < ends[first]
            && !g.protected.contains(&pages[i + 1])
        {
            i += 1;
        }
        let run = &pages[first..=i];
        let run_ends = &ends[first..=i];
        i += 1;
        if set(run[0], run[run.len() - 1] + 1) {
            g.protected.extend(run);
            taken.extend(run);
            if vetted {
                g.vetted.extend(run.iter().copied().zip(run_ends.iter().copied()));
            }
            continue;
        }
        for (&page, &end) in run.iter().zip(run_ends) {
            if set(page, page + 1) {
                g.protected.insert(page);
                taken.push(page);
                if vetted {
                    g.vetted.insert(page, end);
                }
            } else if unmapped(page) {
                taken.push(page);
                unmapped_pages.push(page);
            }
        }
    }
    taken
}

/// Which pages are the runtime's GPU memory ([`runtime_memory`]), asked
/// once per memory region: a region shares its state, type and write watch.
#[derive(Default)]
struct RuntimeRegions {
    /// The last region asked about: its pages and whether it is runtime
    /// memory.
    last: Option<(usize, usize, bool)>,
}

impl RuntimeRegions {
    /// If `page` is the runtime's GPU memory, the page its region is known
    /// to end before (`usize::MAX` for a pool, whose end is not asked).
    /// `run_end`: where the run of consecutive pages asked about from `page`
    /// on ends.
    fn region(&mut self, page: usize, run_end: usize) -> Option<usize> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        if let Some((a, b, yes)) = self.last
            && (a..b).contains(&page)
        {
            return yes.then_some(b);
        }
        // mac: one lookup for the whole run when one pool holds all of it
        // (`VICTORY_WATCH_RUNS=1`): the profile of 27 November counted 9.7 ms
        // a frame of `NtGetWriteWatch`, 94 % of it here, page by page, under
        // the guard lock. The range question answers as the page ones do
        // (`runtime_pages` asks it so).
        if watch_runs() && run_end > page + 1 && write_watched_range(page * PAGE, (run_end - page) * PAGE) {
            self.last = Some((page, run_end, true));
            return Some(usize::MAX);
        }
        // A pool's write watch answers with one lookup. `VirtualQuery` walks
        // Wine's page table to the region's end, under the guard lock that
        // every fault waits for.
        if write_watched(page * PAGE) {
            return Some(usize::MAX);
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return None;
        }
        // Not watched: what `runtime_memory` asks of views, from this query.
        let yes = !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        let (base, size) = (info.BaseAddress as usize, info.RegionSize);
        // In our pages, rounded inward so the region never claims a page it
        // only shares.
        let (a, b) = (base.div_ceil(PAGE).min(page), ((base + size) / PAGE).max(page + 1));
        self.last = Some((a, b, yes));
        yes.then_some(b)
    }
}

/// Pages `[first, last)` changed, and so the pages showing the same memory
/// elsewhere: gives write access back where we guarded them and marks them
/// in every tracker holding them. True if one of `[first, last)` was
/// guarded.
fn changed(g: &mut Guards, first: usize, last: usize, save: bool) -> bool {
    let guarded = g.protected.range(first..last).next().is_some();
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    #[cfg(feature = "diagnostics")]
    {
        let held = others.iter().filter(|p| g.holders.contains_key(p)).count();
        crate::copystat::add(crate::copystat::C::AliasMarks, held as u64);
    }
    // Every page's bytes are saved before any is opened: once a page is
    // open, a write by another thread lands without a fault, and a save
    // after it would keep that write as the page's old bytes.
    mark_changed(g, first, last, save);
    // The pages showing the same memory elsewhere are marked, not opened:
    // nothing writes through them now, and opening them only made the next
    // take guard them again, one Wine call each (two thirds of all
    // protection changes in #50's world).
    for page in others {
        mark_changed(g, page, page + 1, save);
    }
    open_guarded(g, first, last);
    guarded
}

/// Gives write access back to the pages of `[first, last)` we guarded.
fn open_guarded(g: &mut Guards, first: usize, last: usize) {
    let guarded: Vec<usize> = g.protected.range(first..last).copied().collect();
    // In runs: one protection change for consecutive pages (a run across
    // two views fails, and goes page by page).
    let mut i = 0;
    while i < guarded.len() {
        let mut j = i + 1;
        while j < guarded.len() && guarded[j] == guarded[j - 1] + 1 {
            j += 1;
        }
        open_run(g, guarded[i], guarded[j - 1] + 1);
        i = j;
    }
}

/// Marks `[first, last)` changed in the trackers holding its pages,
/// leaving their protection as it is. With `save`, before the write lands:
/// trackers armed for a page they had taken keep its bytes.
fn mark_changed(g: &mut Guards, first: usize, last: usize, save: bool) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders {
            let first_write = {
                let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
                set_dirty(c.2, true);
                set.insert(page)
            };
            if first_write && save {
                c.save(page);
            }
        }
        #[cfg(feature = "diagnostics")]
        g.written.insert(page);
    }
}

/// At each presented frame (diagnostics): counts the tracked pages written
/// in it, and those written the frame before too.
#[cfg(feature = "diagnostics")]
pub(crate) fn frame_done() {
    let mut g = guards();
    let written = std::mem::take(&mut g.written);
    let again = written.intersection(&g.written_before).count();
    crate::copystat::add(crate::copystat::C::WrittenPages, written.len() as u64);
    crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
    g.written_before = written;
}

/// A write into title memory the guards do not see, under way: its pages
/// stay open (a tracker taking them meanwhile would guard them, and the
/// kernel's write would fail) until it is dropped, which marks them changed
/// once more.
#[must_use = "the write must happen while this lives"]
pub struct Writing {
    first: usize,
    last: usize,
}

impl Drop for Writing {
    fn drop(&mut self) {
        let mut g = guards();
        if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
            g.busy.swap_remove(i);
        }
        changed(&mut g, self.first, self.last, false);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler) or another write the
/// guards do not see: opens its guarded blocks, marks them changed and
/// keeps them open while the result lives. The runtime calls it (as
/// `kernel_will_write`) around the title's kernel reads into its own
/// memory and DirectStorage's writes to addresses.
pub fn writing(at: usize, len: usize) -> Writing {
    // The length is the title's (`fread`'s size times count, saturated).
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    changed(&mut g, first, last, true);
    Writing { first, last }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new, and read-write.
pub fn title_memory_remapped(at: usize, len: usize) {
    forget_opened(at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    // First (before the guard lock): a texture copy-back landing there now
    // is let finish, and none pending lands after (`crate::texback`).
    crate::texback::remapped(at, len);
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    // What it showed is gone already: nothing to keep.
    changed(&mut g, first, last, false);
    // Nor what it was: asked about again when next guarded.
    let gone: Vec<usize> = g.vetted.range(first..last).map(|(&p, _)| p).collect();
    for page in gone {
        g.vetted.remove(&page);
    }
    // What it showed is not what it shows now: its links go, both ways.
    let stale: Vec<usize> = g.alias_of.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.alias_of.remove(&page).unwrap_or_default() {
            if let Some(links) = g.alias_of.get_mut(&other) {
                links.retain(|&p| p != page);
                if links.is_empty() {
                    g.alias_of.remove(&other);
                }
            }
        }
    }
}

/// Shaders wrote aliased title memory `[at, at + len)` (`crate::alias`):
/// every tracker holding its pages (textures' staging, heaps, other host
/// copies), and the pages showing the same memory elsewhere, sees them
/// changed. Nothing is saved (the write happened; there is no "before" to
/// keep) and no guard is opened: a GPU write does not go through the CPU's
/// page protection, so the title's own writes stay seen. Opened, as through
/// [`changed`] in VIctory-macOS #207, each run of pages cost two protection
/// calls a submission and the title's next write there a fault. The idea is
/// Marvisak/VIctory 29decb3's `watch::gpu_wrote` (derz92's good pair on
/// VIctory-macOS #322 ran with a marking patch of their own). Returns the
/// tracked pages marked.
pub(crate) fn notify_written(at: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    if g.holders.range(first..last).next().is_none()
        && g.alias_of.range(first..last).next().is_none()
    {
        return 0;
    }
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    let marked = g.holders.range(first..last).count() + others.iter().filter(|p| g.holders.contains_key(p)).count();
    mark_changed(&mut g, first, last, false);
    for page in others {
        mark_changed(&mut g, page, page + 1, false);
    }
    marked
}

/// What shaders wrote into host copies of `[at, at + len)` is dropped:
/// every host buffer copy holding the memory takes it from title memory
/// again at its next refresh (the pages count as changed in each, aliases
/// too). Title memory itself did not change, so its guards stay (opened page
/// by page, they cost ~15 % of the submitting thread at each frame end, run
/// perf15), and other trackers (textures' staging, heaps) are left alone:
/// shaders' writes never reach them (marked, textures over withheld ranges
/// were uploaded again for nothing: 19 % of the uploads, run evict3).
pub(crate) fn retake(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let g = guards();
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    mark_copies(&g, first, last);
    for page in others {
        mark_copies(&g, page, page + 1);
    }
}

/// Pages `[first, last)` count as changed in every host buffer copy's
/// tracker holding them.
fn mark_copies(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
            set_dirty(c.2, true);
            set.insert(page);
        }
    }
}

/// The runtime mapped the page at `at` where `others` show it already: the
/// pages are linked both ways, and if we guard it there, it is guarded here
/// too (a write through the new mapping must not go unseen by the trackers
/// there).
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    let mut g = guards();
    let page = at / PAGE;
    for &other in others {
        for (x, y) in [(other / PAGE, page), (page, other / PAGE)] {
            let links = g.alias_of.entry(x).or_default();
            if !links.contains(&y) {
                links.push(y);
            }
        }
    }
    // If the new mapping cannot be guarded (a write the guards do not see is
    // under way through it), the page is changed; its bytes now may hold that
    // write already, so nothing is saved of them.
    if others.iter().any(|&o| g.protected.contains(&(o / PAGE)))
        && protect(&mut g, &[page]).is_empty()
    {
        changed(&mut g, page, page + 1, false);
    }
}

/// Whether a page of `[at, at + len)` is shown elsewhere too, as the
/// runtime told when it mapped it (tracked or not): a write here changes
/// the bytes there. Asked at submission, by a batch of refreshes that
/// keeps what a copy into title memory cannot have changed; it takes the
/// guard lock for one lookup and allocates nothing under it.
pub(crate) fn aliased(at: usize, len: usize) -> bool {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    guards().alias_of.range(first..last).next().is_some()
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // mac: the sampler leaves a thread alone while it takes exceptions
        // (`crate::profile::excepted`; one load without `VICTORY_PROFILE`).
        crate::profile::excepted();
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            fault(3);
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
        // From 64bitMango e4444060: another thread's fault opened the block
        // while this one was on its way here; every tracker is marked and
        // write access is back, so there is nothing to do and no lock to
        // wait for (`VICTORY_OPENED_FAULTS`).
        if opened(at / PAGE) {
            let again = FOUND_OPEN.with(|f| {
                let (last, n) = f.get();
                let n = if last == at { n.saturating_add(1) } else { 1 };
                f.set((at, n));
                n
            });
            if again <= 3 {
                fault(4);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        FOUND_OPEN.with(|f| f.set((0, 0)));
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        if HOLDING.with(|h| h.get()) {
            if read_only(at) {
                open_under_lock(at / PAGE);
                fault(0);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            fault(2);
            return EXCEPTION_CONTINUE_SEARCH;
        }
        IN_FAULT.with(|f| f.set(true));
        let mut g = guards();
        IN_FAULT.with(|f| f.set(false));
        if changed(&mut g, at / PAGE, at / PAGE + 1, true) {
            fault(0);
            // Every tracker holding it is marked: other threads' faults on
            // it need no lock.
            note_opened(at / PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            // mac: who wrote a block of the title's view heap last
            // (`VICTORY_HEAP_REPORTS`; one load without it).
            crate::descriptor::reports::faulted(at, r.ExceptionAddress as usize);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            fault(1);
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            fault(2);
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed memory that is read-only, as our guards make
/// it.
fn read_only(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READONLY
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) std::time::Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_fault_under_the_guard_lock_does_not_hang() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        install();
        // SAFETY: a fresh block of our own.
        let block = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!block.is_null());
        let mut old = 0;
        // SAFETY: as above.
        assert_ne!(unsafe { VirtualProtect(block.cast(), PAGE, PAGE_READONLY, &mut old) }, 0);
        {
            let _g = guards();
            // SAFETY: inside the block; the handler gives write access back.
            unsafe { block.add(100).write_volatile(7) };
        }
        // SAFETY: as above.
        assert_eq!(unsafe { block.add(100).read_volatile() }, 7);
        assert!(writable(block as usize));
        drop(guards());
        assert!(OPENED_UNDER_LOCK.iter().all(|s| s.load(std::sync::atomic::Ordering::Relaxed) != block as usize / PAGE));
        // SAFETY: allocated above.
        assert_ne!(unsafe { VirtualFree(block.cast(), 0, MEM_RELEASE) }, 0);
    }

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn fault_tracking_follows_the_variable() {
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("yes"), false));
        // Tests keep tracking on, under Rosetta too.
        assert!(fault_tracking() || std::env::var("VICTORY_FAULT_TRACKING").as_deref() == Ok("0"));
    }

    #[test]
    fn tracking_is_opt_in_under_rosetta() {
        // The variable decides when it is set; otherwise tracking is on
        // except under Rosetta (`fault_tracking`'s default is `!rosetta()`).
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("2"), false));
        // Stores are patched exactly when tracking runs under Rosetta.
        assert_eq!(stores_need_patching(), fault_tracking() && rosetta());
    }

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    /// Frees a test's allocation at `p`, unless the guards still guard or
    /// hold a page of it: then it stays allocated, as the other tests'
    /// pools do.
    ///
    /// A copy that a batch, a page table or a widened view still holds
    /// keeps its tracker after the test drops its own handle. Freed under
    /// it, the pages stayed in `protected`, and a test running at the same
    /// time that was given the same address got them unguarded:
    /// [`protect`] skips a page it has guarded already, so the new
    /// allocation stayed writable and its writes went unseen (#31: a guard
    /// test failed in 7 of 150 full runs on an M5 Pro, whichever drew the
    /// address; 2 of 300 with this). The title's own memory does not go
    /// this way: the runtime tells the guards when it unmaps it
    /// ([`title_memory_remapped`]).
    pub(crate) fn free(p: usize) {
        if still_watched(p) {
            return;
        }
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Whether the guards guard or hold a page of the allocation at `p`.
    fn still_watched(p: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // The allocation's length: its regions, while they start from `p`
        // (guarded and open pages are regions of their own).
        let mut len = 0;
        loop {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery((p + len) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 || info.AllocationBase as usize != p || info.RegionSize == 0 {
                break;
            }
            len += info.RegionSize;
        }
        let (first, last) = (p / PAGE, (p + len).div_ceil(PAGE));
        let g = guards();
        g.protected.range(first..last).next().is_some()
            || g.holders.range(first..last).next().is_some()
    }

    /// Everything is changed at first; after taking, only what is written
    /// (by faults, or reported) is, and every tracker of a page sees it.
    /// A tracked range that starts in reserved memory and runs into plain
    /// committed memory (the host's, as the process heap was in #50) never
    /// makes that memory read-only; it is left out of the tracker instead
    /// (copied whole at each use, so the tracker is never clean).
    #[test]
    fn plain_committed_memory_is_never_guarded() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        // The last two blocks committed as plain private memory.
        // SAFETY: inside our reservation.
        assert!(!unsafe { VirtualAlloc((base + 2 * PAGE) as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        let _ = t.take(base, base + 4 * PAGE);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "still writable");
        assert!(!t.changed(base / PAGE + 2) && !t.changed(base / PAGE + 3), "not tracked");
        assert_eq!(t.untracked(base, base + 4 * PAGE), [(base + 2 * PAGE, base + 4 * PAGE)], "copied whole");
        assert!(!t.clean(), "never clean");
        // SAFETY: writable, so no fault.
        unsafe { ((base + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn writes_mark_every_tracker_of_their_page() {
        let p = pool(4);
        let a = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let b = track(p + PAGE, 2 * PAGE, true).expect("tracked");
        assert!(!a.clean() && !b.clean(), "changed at first");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert!(a.clean() && !b.clean());
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 3 * PAGE)]);
        assert!(a.take(p, p + 4 * PAGE).is_empty() && b.take(p, p + 4 * PAGE).is_empty());
        assert!(a.clean() && b.clean());
        // SAFETY: committed pages of ours, now read-only: the handler opens
        // the block; the second write needs no fault.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write_volatile(1);
            ((p + PAGE + 9) as *mut u8).write_volatile(2);
        }
        assert!(!a.clean() && !b.clean(), "a write is seen by every tracker");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        assert!(a.clean());
        // `a` guarded it again; `b` still has it changed.
        assert!(b.changed(p / PAGE + 1));
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: as above: a fault again, for both.
        unsafe { ((p + PAGE + 10) as *mut u8).write_volatile(3) };
        assert!(a.changed(p / PAGE + 1) && b.changed(p / PAGE + 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        drop(writing(p + 3 * PAGE + 5, 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + 3 * PAGE, p + 4 * PAGE)]);
        title_memory_remapped(p, PAGE);
        assert!(a.changed(p / PAGE) && !a.changed(p / PAGE + 1));
        drop((a, b));
        free(p);
    }

    /// A range running from free memory over another allocation into a pool
    /// (a 4 MiB window of memory outside buffers) tracks the pool's pages
    /// and leaves the other allocation alone: taking guards the pool only.
    #[test]
    fn a_range_over_foreign_memory_guards_only_the_runtimes() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        // This test tracks a mostly free address range. In the parallel suite,
        // other tests can map their views in its holes even after both of our
        // allocations succeed. Keep those views out of this address space.
        const CHILD: &str = "VICTORY_TEST_FOREIGN_WINDOW_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "watch::tests::a_range_over_foreign_memory_guards_only_the_runtimes";
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, "1")
                .output()
                .expect("run foreign-window test in a child");
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "foreign-window child {}: {stdout}{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        const WINDOW: usize = 4 << 20;
        // In a free 4 MiB-aligned window: plain committed memory (as the
        // process heap's) 1 MiB in, a pool 2 MiB in.
        let (base, foreign, pool) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations in the window it left free, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * WINDOW, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let base = p.next_multiple_of(WINDOW);
                    let at = |offset: usize, kind: u32| {
                        VirtualAlloc((base + offset) as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (foreign, pool) = (at(1 << 20, 0), at(2 << 20, MEM_WRITE_WATCH));
                    if p != 0 && foreign == base + (1 << 20) && pool == base + (2 << 20) {
                        return Some((base, foreign, pool));
                    }
                    [foreign, pool].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("a free window");
        let t = track(base, WINDOW, true).expect("the pool is tracked");
        let taken = t.take(base, base + WINDOW);
        assert!(taken.iter().any(|&(a, b)| a <= pool && pool + PAGE <= b), "{taken:x?}");
        assert!(!taken.iter().any(|&(a, b)| a < foreign + PAGE && foreign < b), "{taken:x?}");
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(foreign), PAGE_READWRITE, "the other allocation is not");
        drop(t);
        free(foreign);
        free(pool);
    }

    /// A tracker's dirty bit goes back when it goes, and the next tracker
    /// starts dirty on it (everything changed at first).
    #[test]
    fn a_new_tracker_starts_dirty_on_a_bit_given_back() {
        let p = pool(1);
        let a = track(p, PAGE, true).expect("tracked");
        assert_eq!(a.take(p, p + PAGE).len(), 1);
        assert!(a.clean() && clean_bit(a.bit()));
        drop(a);
        let b = track(p, PAGE, true).expect("tracked");
        assert!(!b.clean() && !clean_bit(b.bit()));
        assert!(!clean_bit(None), "no bit: never clean");
        drop(b);
        free(p);
    }

    /// A page with nothing mapped is taken as zeros: `take_split` lists it
    /// apart from the pages to copy.
    #[test]
    fn unmapped_pages_are_taken_apart() {
        // SAFETY: a fresh write-watched reservation of two pages, the first
        // committed; freed below.
        let p = unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            p
        };
        let t = track(p, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take_split(p, p + 2 * PAGE), (vec![(p, p + PAGE)], vec![(p + PAGE, p + 2 * PAGE)]));
        assert!(t.take(p, p + 2 * PAGE).is_empty(), "both taken");
        drop(t);
        free(p);
    }

    /// A page only partly in the range asked is copied in part and stays
    /// changed; edge pages count whole when the tracked part is covered.
    #[test]
    fn partial_pages_stay_changed() {
        let p = pool(3);
        let t = track(p + 100, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take(p + 100, p + PAGE + 10), [(p + 100, p + PAGE + 10)]);
        assert!(!t.changed(p / PAGE) && t.changed(p / PAGE + 1));
        assert!(!t.clean(), "a page taken in part stays changed");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p + PAGE, p + 2 * PAGE + 100)]);
        assert!(t.take(p, p + 3 * PAGE).is_empty());
        assert!(t.clean());
        drop(t);
        free(p);
    }

    /// Pages of the ranges armed keep their bytes from before the first
    /// write after they were taken, whichever overlapping ranges armed
    /// them; a page unarmed keeps none until it is armed again, and none
    /// does once the tracker is disarmed.
    #[test]
    fn armed_ranges_keep_their_pages_bytes_from_before_the_first_write() {
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let all = || t.take(p, p + 4 * PAGE);
        // SAFETY: (every write below) a byte of our pool; a guarded page
        // faults into the handler, which saves it first where it is armed.
        let write = |page: usize, value: u8| unsafe { ((p + page * PAGE + 8) as *mut u8).write_volatile(value) };
        let saved = |page: usize| t.saved(p / PAGE + page).map(|bytes| bytes[8]);
        all();
        write(0, 1);
        assert_eq!(saved(0), None, "nothing armed yet");
        all();
        // Bytes 100 of page 0 to 1 of page 2: pages 0 to 2, page 3 not.
        t.arm(p + 100, p + 2 * PAGE + 1);
        t.arm(p + PAGE, p + 2 * PAGE);
        t.arm(p + PAGE + 5, p + 2 * PAGE + 1);
        t.unarm(p / PAGE + 1);
        for page in 0..4 {
            write(page, 2);
        }
        assert_eq!([saved(0), saved(1), saved(2), saved(3)], [Some(1), None, Some(0), None]);
        assert_eq!(saved(0), None, "handed out once");
        all();
        // Armed again by a range holding it.
        t.arm(p + PAGE, p + 2 * PAGE);
        write(1, 3);
        write(1, 4);
        assert_eq!(saved(1), Some(2), "from before the first write");
        t.disarm();
        all();
        write(2, 5);
        assert_eq!(saved(2), None, "disarmed");
        drop(t);
        free(p);
    }

    /// Runs armed that touch or overlap merge into one, so a page's check
    /// stays a search however many sub-ranges are armed; pages unarmed in
    /// them stay unarmed until a range holding them is armed again.
    #[test]
    fn adjacent_and_overlapping_armed_runs_coalesce() {
        let mut s = Saved::default();
        // Adjacent: [2k, 2k + 2) end to end, in order and out of it.
        for k in 0..200 {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        for k in (0..200).rev() {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        // Overlapping, nested and disjoint.
        let mut s = Saved::default();
        s.arm(10, 20);
        s.arm(15, 25);
        assert_eq!(s.armed, [(10, 25)]);
        s.arm(12, 14);
        assert_eq!(s.armed, [(10, 25)], "nested");
        s.arm(40, 50);
        s.arm(0, 5);
        assert_eq!(s.armed, [(0, 5), (10, 25), (40, 50)], "disjoint, sorted");
        s.arm(5, 10);
        assert_eq!(s.armed, [(0, 25), (40, 50)], "touching both sides");
        s.arm(30, 35);
        s.arm(20, 45);
        assert_eq!(s.armed, [(0, 50)], "spanning several");
        s.arm(60, 70);
        s.arm(55, 55);
        assert_eq!(s.armed, [(0, 50), (60, 70)], "an empty run arms nothing");
        assert!(s.armed(0) && s.armed(49) && !s.armed(50) && !s.armed(59));
        assert!(s.armed(60) && s.armed(69) && !s.armed(70) && !s.armed(1000));
        // Unarmed pages stay out of a merged run until armed again.
        s.unarmed.insert(45);
        s.unarmed.insert(65);
        s.arm(50, 60);
        assert_eq!(s.armed, [(0, 70)]);
        assert!(!s.armed(45) && !s.armed(65) && s.armed(55));
        s.arm(44, 46);
        assert!(s.armed(45) && !s.armed(65));
        s.arm(0, 70);
        assert!(s.armed(65) && s.unarmed.is_empty());
    }

    /// Plain committed memory (the C heap) is not tracked; reserved memory
    /// is, reads as taken, and changes when the runtime maps it.
    #[test]
    fn heap_memory_is_not_tracked_and_placeholders_wait_for_mappings() {
        let plain = vec![0u8; 2 * PAGE];
        assert!(track(plain.as_ptr() as usize, plain.len(), true).is_none());
        // SAFETY: a reservation of ours, freed below.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        let t = track(r, 2 * PAGE, true).expect("reserved memory is tracked");
        assert_eq!(t.take(r, r + 2 * PAGE).len(), 1, "copied once (as zeros)");
        assert!(t.take(r, r + 2 * PAGE).is_empty());
        title_memory_remapped(r, PAGE);
        assert_eq!(t.take(r, r + 2 * PAGE), [(r, r + PAGE)]);
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as changed.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        assert_ne!(view, 0);
        assert!(track(view, 2 * PAGE, false).is_none(), "views only where asked");
        let t = track(view, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + 2 * PAGE)]); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        let write = writing(view + 100, 13);
        // A tracker taking the page meanwhile leaves it open.
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        assert!(read(&mut file));
        drop(write);
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        drop(t);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// `pages` pages of a fresh pagefile section mapped twice (as the
    /// title's pool view and tiled heap show the same physical pages),
    /// reported as the runtime does when it maps them.
    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice; the test
        // keeps both views.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            )
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    thread_local! {
        /// `VirtualProtect` calls this thread made in `protect`.
        pub(super) static PROTECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// `single` views of one page each side by side, as the runtime's page
    /// mappings make them, then one view of `single` pages: the address of
    /// the first, and of the long one.
    fn views_side_by_side(single: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let len = 2 * single * PAGE;
        // SAFETY: a pagefile section and a placeholder of our own, split
        // and filled with views of the section; the test keeps them.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let map = |at: usize, offset: usize, n: usize| {
                VirtualFree(at as *mut _, n * PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    offset as u64,
                    n * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            };
            for k in 0..single {
                map(base + k * PAGE, k * PAGE, 1);
            }
            let long = base + single * PAGE;
            map(long, single * PAGE, single);
            (base, long)
        }
    }

    /// Guarding pages of separate views takes one call per view, not a
    /// call over all of them that fails first and then one per page.
    #[test]
    fn takes_guard_one_view_per_call() {
        let calls = || PROTECTS.with(std::cell::Cell::get);
        // Four one-page views and a four-page view, taken together.
        let (base, _) = views_side_by_side(4);
        let t = track(base, 8 * PAGE, true).expect("views are tracked");
        let before = calls();
        assert_eq!(t.take(base, base + 8 * PAGE), [(base, base + 8 * PAGE)]);
        assert_eq!(calls() - before, 5);
        assert!((0..8).all(|k| read_only(base + k * PAGE)));
        drop(t);
        // Each kind alone, written after: the guards catch it.
        let (base, long) = views_side_by_side(4);
        for (at, want) in [(base, 4), (long, 1)] {
            let t = track(at, 4 * PAGE, true).expect("views are tracked");
            let before = calls();
            assert_eq!(t.take(at, at + 4 * PAGE), [(at, at + 4 * PAGE)]);
            assert_eq!(calls() - before, want, "calls for the views at {at:#x}");
            for k in 0..4 {
                assert!(read_only(at + k * PAGE), "page {k} at {at:#x} guarded");
            }
            // Guarded, they mark the tracker when written.
            // SAFETY: a page of the views; the handler opens it.
            unsafe { ((at + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(t.changed(at / PAGE + 2));
            assert!(!t.changed(at / PAGE + 1));
        }
    }

    /// A write through another mapping of the same pages marks the tracker
    /// (the guard there faults, the runtime says where else it is mapped);
    /// so does a write the runtime reports at the other mapping.
    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert!(read_only(a + PAGE), "marked, still guarded: no call to guard it again");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        // A third mapping made later starts guarded, so its writes count.
        let (c, _) = twin_views(1);
        title_memory_aliased(c, &[a]);
        // SAFETY: a view of ours (another section: only the tracking
        // takes it for `a`'s pages here).
        unsafe { (c as *mut u32).write_volatile(1) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
    }

    /// Pages are reported aliased while the runtime says they are shown
    /// twice, and not after either mapping changes.
    #[test]
    fn aliased_pages_are_found_by_range() {
        let (a, b) = twin_views(3);
        assert!(
            aliased(a, 1) && aliased(b + 2 * PAGE + 5, 10),
            "either view, any byte"
        );
        assert!(aliased(a + PAGE - 1, 2), "a range that reaches a page");
        assert!(!aliased(a, 0), "no bytes");
        let plain = pool(1);
        assert!(!aliased(plain, PAGE));
        free(plain);
        // One page shown once only again: its link goes, the others stay.
        title_memory_remapped(b + PAGE, PAGE);
        assert!(!aliased(a + PAGE, PAGE) && !aliased(b + PAGE, PAGE));
        assert!(aliased(a, PAGE) && aliased(b + 2 * PAGE, PAGE));
        assert!(aliased(a, usize::MAX), "no overflow");
    }

    /// A tracker of one mapping going away leaves the guard its page shares
    /// with a tracker of another mapping, so a write there still counts (it
    /// was opened, and the pages it went unseen in stayed clean).
    #[test]
    fn a_tracker_dropped_on_one_mapping_leaves_the_other_guarded() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        drop(u);
        // SAFETY: the second view, guarded for `t`'s page too.
        unsafe { ((b + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
    }

    /// A retake has every copy of the memory, and of its other mappings,
    /// take it again, and leaves it guarded: a later write still counts.
    #[test]
    fn a_retake_marks_the_copies_and_keeps_the_guard() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        let (a, b) = twin_views(2);
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track_copy(b, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        retake(a + PAGE + 16, 32);
        // Marked in both copies, the second mapping's too, still guarded.
        assert!(!t.clean() && !u.clean());
        assert!(t.changed((a + PAGE) / PAGE) && u.changed((b + PAGE) / PAGE));
        assert!(!t.changed(a / PAGE));
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        assert_eq!(protection(b + PAGE), PAGE_READONLY);
        // Taken again (the copies re-copy that page), then guarded and clean.
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        // SAFETY: the second view, guarded all along.
        unsafe { ((b + PAGE + 8) as *mut u32).write_volatile(3) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        // A retake while marked and guarded, then a drop: nothing breaks.
        retake(a, 2 * PAGE);
        drop(u);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert!(t.clean());
    }

    /// A retake marks host buffer copies holding the memory, not other
    /// trackers of it (a texture's staging): shaders' writes reach only the
    /// copies. A title write still marks both.
    #[test]
    fn a_retake_marks_buffer_copies_alone() {
        let p = pool(2);
        let copy = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        let texture = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert_eq!(texture.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        retake(p + PAGE, 16);
        assert!(copy.changed(p / PAGE + 1), "the copy takes it again");
        assert!(!texture.changed(p / PAGE + 1) && texture.clean(), "the texture's tracker is left alone");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: a page of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(copy.changed(p / PAGE + 1) && texture.changed(p / PAGE + 1), "a title write marks both");
    }

    /// A write under way on another mapping of a page keeps the page open
    /// there, so taking the page (or marking it taken) through its tracked
    /// mapping must leave it changed and keep what was saved of it; aliasing
    /// a busy mapping to a guarded one must mark the guarded page changed.
    #[test]
    fn a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        t.arm(a, a + PAGE);
        let write = writing(b, 16);
        assert!(t.changed(a / PAGE));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.changed(a / PAGE), "take must leave the page changed");
        t.taken(&[a / PAGE]);
        assert!(t.changed(a / PAGE), "taken must leave the page changed");
        // SAFETY: the second view of the page, kept open by `write`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        drop(write);
        let saved = t.saved(a / PAGE).expect("saved bytes survive");
        assert_eq!(&saved[..4], &[0, 0, 0, 0]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty(), "taken once the write ended");
        let (c, _) = twin_views(1);
        let busy_c = writing(c, 16);
        title_memory_aliased(c, &[a]);
        assert!(
            t.changed(a / PAGE),
            "aliasing a busy page marks the tracked one"
        );
        drop(busy_c);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }
}
