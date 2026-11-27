//! mac: the title's own descriptor stores report the slots they write
//! (`VICTORY_HEAP_REPORTS`, off by default; avatarmanz 9997d255 and
//! 4cb90bee, "renderer port 2", fitted to this line).
//!
//! What the title does (Grand Theft Auto VI 1.0.2.1; `victory-loader`'s
//! `descriptor_sites` has the code): its placed shader-visible view heap is
//! the memory of its frame allocator, and the descriptor tables of draws
//! and dispatches are filled there one 32-byte record at a time by six
//! stores in two functions; other records arrive through the runtime's
//! `memcpy` and `memmove`. Views made by calls (`Create*View`,
//! `CopyDescriptors*`) are made on the host where they are written
//! (`super::mirror`) and need no report.
//!
//! What this line does without the switch: the heap's memory is guarded,
//! a write to a guarded 64 KiB page faults once and marks the page, and
//! before each submission (about twenty a frame) `sync_heaps` takes the
//! marked pages, which guards them again, and looks at every slot on them.
//! The look is cheap here; the cost is the guarding (a protection call a
//! page taken) and the write fault each taken page costs the next title
//! thread that writes it, on Wine and Rosetta far dearer than on Windows.
//!
//! With the switch the loader patches the six stores (byte-checked) to
//! jump to code that does the store and then sets a byte for the slot and
//! one for its 128 slots in maps of ours ([`Marks`]): about twenty plain
//! instructions more a record, no call, no lock and nothing that can
//! fault (the maps are committed and written when made). The title's
//! `memcpy` and `memmove` imports go through [`import`], which after the
//! copy marks each slot of the heap whose first word now is a view
//! record's.
//!
//! Two forms:
//!
//! - `VICTORY_HEAP_REPORTS=check`: nothing changes in what the sync does
//!   (the watch is asked, every slot of the written pages looked at, as
//!   before). For each slot the look finds changed, the marks say whether
//!   a store or a copy told of it. The frame line "heap reports" gives
//!   the counts: a view record that nothing told of is a writer the
//!   patches miss, logged with the place in the title's code that last
//!   faulted on its 64 KiB block. A run with this form says whether the
//!   next one is safe on this build.
//! - `VICTORY_HEAP_REPORTS=1`: the sync takes the told slots ([`take`]) in
//!   place of the written pages and looks at those alone. The watch is the
//!   fallback: every [`every`] frames (8; `VICTORY_HEAP_REPORTS_EVERY`) the
//!   sync before a submission asks it and looks at the written pages
//!   whole, as before, and what it finds that nothing told of is made
//!   then, counted and logged. Only with all six stores patched; else the
//!   sync stays as it was and the line says so.
//!
//! What `=1` changes besides the cost. A slot whose view record the title
//! overwrites with other data (its allocator's memory is reused) is not
//! told of: the host slot keeps its old view, and what it holds, until
//! the next fallback look makes it null (before: at the next sync). No
//! shader reads such a slot as a descriptor (the title wrote none there),
//! and what the slot holds stays alive as for any slot. A view record
//! written by a writer the patches miss shows up to [`every`] frames late:
//! that is what the `check` form is for.
//!
//! Plain std but for the import wrappers, so that the tests run on any
//! host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{
    AtomicPtr, AtomicU8, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use std::sync::{Mutex, OnceLock, RwLock};

use super::VIEW_TAG;

/// The switch's forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    /// The reports are compared with what the look at written pages finds.
    Check,
    /// The reports are what the sync looks at.
    Use,
}

/// `VICTORY_HEAP_REPORTS`, read once.
pub fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| mode_of(std::env::var("VICTORY_HEAP_REPORTS").ok().as_deref()))
}

/// The switch's rule: unset, empty or `0` off; `check`; anything else on.
fn mode_of(value: Option<&str>) -> Mode {
    match value {
        None | Some("" | "0") => Mode::Off,
        Some("check") => Mode::Check,
        Some(_) => Mode::Use,
    }
}

/// Whether the switch is on in either form (for the launcher).
pub fn wanted() -> bool {
    mode() != Mode::Off
}

/// Frames between the fallback's askings of the write watch
/// (`VICTORY_HEAP_REPORTS_EVERY=<frames>`, 8 if not given, 1 at least).
fn every() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| every_of(std::env::var("VICTORY_HEAP_REPORTS_EVERY").ok().as_deref()))
}

fn every_of(value: Option<&str>) -> u64 {
    value.and_then(|v| v.trim().parse().ok()).unwrap_or(8u64).max(1)
}

/// Whether the title's `memcpy` and `memmove` report too
/// (`VICTORY_HEAP_REPORTS_COPIES=0`: not, to see in the `check` form what
/// they are needed for).
fn copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_HEAP_REPORTS_COPIES").as_deref() != Ok("0"))
}

/// What the patched stores read (`victory_loader::descriptor_sites`,
/// [`MARKS_LEN`] bytes): the heap's first byte, its length in bytes (0:
/// none, nothing is marked), the map of slots (a byte each), the map of
/// chunks (a byte for [`CHUNK`] slots).
#[repr(C)]
pub struct Marks {
    start: AtomicU64,
    len: AtomicU64,
    slots: AtomicPtr<AtomicU8>,
    chunks: AtomicPtr<AtomicU8>,
}

/// The size of [`Marks`], as the loader's code assumes it.
pub const MARKS_LEN: usize = 32;
const _: () = assert!(size_of::<Marks>() == MARKS_LEN);

static MARKS: Marks = Marks {
    start: AtomicU64::new(0),
    len: AtomicU64::new(0),
    slots: AtomicPtr::new(std::ptr::null_mut()),
    chunks: AtomicPtr::new(std::ptr::null_mut()),
};

/// Slots a byte of the map of chunks stands for (4 KiB of records).
const CHUNK: usize = 128;

/// What a slot's byte says: a patched store of the title's wrote it, or
/// one of its copies did.
pub const STORE: u8 = 1;
const COPY: u8 = 2;

/// Where the stores' cells are, for `descriptor_sites::apply`.
pub fn marks_address() -> usize {
    std::ptr::from_ref(&MARKS) as usize
}

/// The title's image (base, size), how many of its stores were patched and
/// how many there are.
static IMAGE: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

/// Told by the launcher: `patched` of the title's `sites` record stores in
/// the image at `base` (`size` bytes) mark their slots.
pub fn hooked(base: usize, size: usize, patched: usize, sites: usize) {
    for (cell, value) in IMAGE.iter().zip([base, size, patched, sites]) {
        cell.store(value, Relaxed);
    }
}

/// Whether every store of the title's was patched: only then do the
/// reports stand for the written pages.
fn all_patched() -> bool {
    let [patched, sites] = [2, 3].map(|i| IMAGE[i].load(Relaxed));
    sites > 0 && patched == sites
}

/// The heap whose slots are marked: its maps, and the places in the
/// title's code that last faulted on each of its 64 KiB blocks.
struct Adopted {
    start: u64,
    count: usize,
    slots: &'static [AtomicU8],
    chunks: &'static [AtomicU8],
    writers: &'static [AtomicUsize],
    /// What the sync does with it.
    told: Told,
}

static ADOPTED: RwLock<Option<Adopted>> = RwLock::new(None);

/// Maps of heaps that went, by slot count: never freed (a store of the
/// title's may be marking while its heap goes), used again by a heap of
/// the same size.
type Maps = (&'static [AtomicU8], &'static [AtomicU8], &'static [AtomicUsize]);
static SPARE: Mutex<Vec<Maps>> = Mutex::new(Vec::new());

/// What a heap sync does with a heap's reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Told {
    /// Nothing: not the marked heap, or the switch is off.
    No,
    /// Compares them with what it finds on the written pages.
    Check,
    /// Looks at the told slots in place of the written pages.
    Use,
}

/// A placed shader-visible view heap of `count` records at `start` is
/// made: with the switch on, the first such is the one whose slots are
/// marked. Whether it is.
pub(crate) fn adopt(start: u64, count: u32) -> bool {
    match mode() {
        Mode::Off => false,
        mode => adopt_as(start, count, mode, all_patched()),
    }
}

fn adopt_as(start: u64, count: u32, mode: Mode, all_patched: bool) -> bool {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.is_some() {
        tracing::info!(
            start = format_args!("{start:#x}"),
            count,
            "heap reports: another placed view heap, looked at by its written pages as before"
        );
        return false;
    }
    let count = count as usize;
    let (slots, chunks, writers) = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        match spare.iter().position(|m| m.0.len() == count) {
            Some(at) => spare.swap_remove(at),
            None => {
                // Every byte written here: the memory is there before a
                // store of the title's marks in it.
                let bytes = |n: usize| -> &'static [AtomicU8] { Box::leak((0..n).map(|_| AtomicU8::new(0)).collect()) };
                let blocks = (count * 32).div_ceil(1 << 16);
                (bytes(count), bytes(count.div_ceil(CHUNK)), &*Box::leak((0..blocks).map(|_| AtomicUsize::new(0)).collect()))
            }
        }
    };
    for byte in slots.iter().chain(chunks) {
        byte.store(0, Relaxed);
    }
    let told = match mode {
        Mode::Use if all_patched => Told::Use,
        Mode::Off => Told::No,
        // Not every store tells: the reports are only compared.
        _ => Told::Check,
    };
    // Its first sync before a submission looks at the written pages whole.
    ASKED.store(0, Relaxed);
    MARKS.slots.store(slots.as_ptr().cast_mut(), Release);
    MARKS.chunks.store(chunks.as_ptr().cast_mut(), Release);
    MARKS.start.store(start, Release);
    // Last: a store that sees a length sees the rest.
    MARKS.len.store(count as u64 * 32, Release);
    *adopted = Some(Adopted { start, count, slots, chunks, writers, told });
    let [patched, sites] = [2, 3].map(|i| IMAGE[i].load(Relaxed));
    match (mode, told) {
        (Mode::Use, Told::Use) => tracing::info!(
            start = format_args!("{start:#x}"),
            count,
            every = every(),
            "VICTORY_HEAP_REPORTS=1: the title's view heap's slots are told by its {patched} patched record stores \
             and its memcpy/memmove; heap syncs look at the told slots, and at the written pages every `every` \
             frames (the frame line's \"heap reports\")"
        ),
        (Mode::Use, _) => tracing::warn!(
            start = format_args!("{start:#x}"),
            count,
            "VICTORY_HEAP_REPORTS=1: only {patched} of the title's {sites} record stores are patched, so the \
             reports are compared, not used: heap syncs look at the written pages as before"
        ),
        _ => tracing::info!(
            start = format_args!("{start:#x}"),
            count,
            "VICTORY_HEAP_REPORTS=check: heap syncs look at the written pages as before, and what they find is \
             compared with what the title's {patched} patched record stores (of {sites}) and its memcpy/memmove \
             told (the frame line's \"heap reports\")"
        ),
    }
    true
}

/// The heap at `start` goes: nothing more is marked for it.
pub(crate) fn let_go(start: u64) {
    // (No heap is marked without the switch.)
    if MARKS.len.load(Relaxed) == 0 {
        return;
    }
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.as_ref().is_some_and(|a| a.start == start) {
        MARKS.len.store(0, Release);
        if let Some(a) = adopted.take() {
            SPARE.lock().unwrap_or_else(|e| e.into_inner()).push((a.slots, a.chunks, a.writers));
        }
        tracing::info!(start = format_args!("{start:#x}"), "heap reports: the marked view heap went");
    }
}

/// What a heap sync does with the reports of the heap of `count` slots at
/// `start`.
pub(crate) fn told(start: u64, count: u32) -> Told {
    // (No heap is marked without the switch: one load, then.)
    if MARKS.len.load(Relaxed) == 0 {
        return Told::No;
    }
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    adopted.as_ref().filter(|a| a.start == start && a.count == count as usize).map_or(Told::No, |a| a.told)
}

/// Marks slot `slot` of the marked heap (inside it: checked by the caller).
#[inline]
fn mark_slot(slot: usize, by: u8) {
    // SAFETY: the maps are leaked and as long as the length the caller
    // checked the slot against.
    unsafe {
        (*MARKS.slots.load(Relaxed).add(slot)).store(by, Release);
        (*MARKS.chunks.load(Relaxed).add(slot / CHUNK)).store(1, Release);
    }
}

/// The told slots of the heap at `start`, each mark cleared, as runs of
/// slots `(first, count)` in order; empty if it is not the marked heap.
///
/// A store writes its record first and its mark after: a mark taken here
/// is of a record that is there to read, and a record whose mark comes
/// after this taking is found by the next.
pub(crate) fn take(start: u64) -> Vec<(u32, u32)> {
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref().filter(|a| a.start == start) else {
        return Vec::new();
    };
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut by = [0u64; 2];
    for (c, chunk) in a.chunks.iter().enumerate() {
        // Cleared before its slots are read: a mark set meanwhile sets it again.
        if chunk.load(Relaxed) == 0 || chunk.swap(0, AcqRel) == 0 {
            continue;
        }
        let first = c * CHUNK;
        for (i, slot) in a.slots[first..(first + CHUNK).min(a.count)].iter().enumerate() {
            if slot.load(Relaxed) == 0 {
                continue;
            }
            let mark = slot.swap(0, AcqRel);
            if mark == 0 {
                continue;
            }
            by[usize::from(mark != STORE)] += 1;
            let index = (first + i) as u32;
            match runs.last_mut() {
                Some((at, len)) if *at + *len == index => *len += 1,
                _ => runs.push((index, 1)),
            }
        }
    }
    COUNTS[TOLD_BY_STORE].fetch_add(by[0], Relaxed);
    COUNTS[TOLD_BY_COPY].fetch_add(by[1], Relaxed);
    COUNTS[TAKINGS].fetch_add(1, Relaxed);
    runs
}

/// Slots `first..first + count` of the heap at `start` are told again: a
/// round ahead of a submission took them and ended before it looked.
pub(crate) fn put_back(start: u64, first: u32, count: u32) {
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref().filter(|a| a.start == start) else {
        return;
    };
    for slot in (first as usize..(first + count) as usize).take_while(|&s| s < a.count) {
        // Not over a mark set since.
        let _ = a.slots[slot].compare_exchange(0, STORE, AcqRel, Relaxed);
        a.chunks[slot / CHUNK].store(1, Release);
    }
    COUNTS[PUT_BACK].fetch_add(u64::from(count), Relaxed);
}

/// A sync looked at slots told of the heap: `changed` of them had another
/// record than the host slot was made from.
pub(crate) fn changed(changed: usize) {
    COUNTS[TOLD_CHANGED].fetch_add(changed as u64, Relaxed);
}

/// A look at slots `first..last` of a written page of the heap at `start`
/// found the slots `stale` changed (`is_view` says whether a slot's record
/// now is a view record): each is counted by what told of it, and the
/// marks of the page's slots are cleared (they stood for this look). A
/// view record nothing told of is said, with the place in the title's
/// code that last faulted on its block.
pub(crate) fn looked(start: u64, first: u32, last: u32, stale: &[u32], is_view: impl Fn(u32) -> bool) {
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref().filter(|a| a.start == start) else {
        return;
    };
    COUNTS[PAGES].fetch_add(1, Relaxed);
    let mut found = [0u64; 5];
    for &index in stale {
        let Some(slot) = a.slots.get(index as usize) else {
            continue;
        };
        let mark = slot.swap(0, AcqRel);
        let kind = match (is_view(index), mark) {
            (true, STORE) => FOUND_BY_STORE,
            (true, 0) => FOUND_UNTOLD,
            (true, _) => FOUND_BY_COPY,
            (false, 0) => FOUND_OTHER,
            (false, _) => FOUND_OTHER_TOLD,
        };
        found[kind - FOUND_BY_STORE] += 1;
        if kind == FOUND_UNTOLD {
            missed(a, index);
        }
    }
    for (kind, n) in found.into_iter().enumerate() {
        COUNTS[FOUND_BY_STORE + kind].fetch_add(n, Relaxed);
    }
    // The rest of the page's marks: of slots made already (by a draw, or
    // rewritten with the record they had).
    let mut same = 0;
    for slot in a.slots.get(first as usize..(last as usize).min(a.count)).unwrap_or_default() {
        if slot.load(Relaxed) != 0 && slot.swap(0, AcqRel) != 0 {
            same += 1;
        }
    }
    COUNTS[TOLD_SAME].fetch_add(same, Relaxed);
}

/// The `check` form, after a sync before a submission has looked at the
/// written pages of the heap at `start`: marks left are of slots on pages
/// the watch did not give. Counted and cleared.
pub(crate) fn clear_rest(start: u64) {
    let rest: u64 = take_quietly(start);
    COUNTS[TOLD_ELSEWHERE].fetch_add(rest, Relaxed);
}

/// [`take`] without its counts: how many marks were cleared.
fn take_quietly(start: u64) -> u64 {
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref().filter(|a| a.start == start) else {
        return 0;
    };
    let mut n = 0;
    for (c, chunk) in a.chunks.iter().enumerate() {
        if chunk.load(Relaxed) == 0 || chunk.swap(0, AcqRel) == 0 {
            continue;
        }
        let first = c * CHUNK;
        for slot in &a.slots[first..(first + CHUNK).min(a.count)] {
            n += u64::from(slot.load(Relaxed) != 0 && slot.swap(0, AcqRel) != 0);
        }
    }
    n
}

/// A view record at slot `index` that nothing told of: counted by the
/// place in the title's code that last faulted on its block, and said
/// (the first twenty, then at every doubling).
fn missed(a: &Adopted, index: u32) {
    let n = MISSED_ALL.fetch_add(1, Relaxed);
    let rip = a.writers.get((index as usize * 32) >> 16).map_or(0, |w| w.load(Relaxed));
    let (base, size) = (IMAGE[0].load(Relaxed), IMAGE[1].load(Relaxed));
    let writer = if rip >= base && rip < base + size {
        format!("title+{:#x}", rip - base)
    } else if rip != 0 {
        format!("{rip:#x} (not the title's code)")
    } else {
        "not known (no fault on its block)".into()
    };
    if n < 20 || n.is_power_of_two() {
        tracing::warn!(
            slot = index,
            %writer,
            "heap reports: a view record in the title's heap that no store or copy told of; the last write fault \
             on its 64 KiB block came from `writer`"
        );
    }
    *WRITERS.lock().unwrap_or_else(|e| e.into_inner()).entry(writer).or_default() += 1;
}

/// A write fault of the title's at `at`, from its instruction at `rip`:
/// if in the marked heap, who wrote its block last (for [`missed`]).
#[inline]
pub(crate) fn faulted(at: usize, rip: usize) {
    if MARKS.len.load(Relaxed) == 0 {
        return;
    }
    // Not waited for: this runs in the write-fault handler.
    let Ok(adopted) = ADOPTED.try_read() else { return };
    if let Some(a) = adopted.as_ref()
        && let Some(writer) = a.writers.get(((at as u64).wrapping_sub(a.start) >> 16) as usize)
    {
        writer.store(rip, Relaxed);
    }
}

/// The frame the fallback last asked the write watch in, plus one (0: never).
static ASKED: AtomicU64 = AtomicU64::new(0);

/// Whether the sync before a submission in frame `frame` asks the write
/// watch of the marked heap and looks at its written pages whole (the
/// `=1` form's fallback): the first time, then every [`every`] frames.
pub(crate) fn ask_due(frame: u64) -> bool {
    ask_due_every(frame, every())
}

fn ask_due_every(frame: u64, every: u64) -> bool {
    let asked = ASKED.load(Relaxed);
    if asked != 0 && frame + 1 < asked + every {
        return false;
    }
    ASKED.store(frame + 1, Relaxed);
    true
}

/// The runtime's own `memcpy` and `memmove` ([`import`]).
static REAL: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// What the title's import of `name` (bound to `address`) is bound to:
/// with the switch on, its `memcpy` and `memmove` mark the view records
/// they copy into the marked heap.
pub fn import(name: &str, address: usize) -> usize {
    let (which, ours) = match name {
        "memcpy" => (0, copy as *const () as usize),
        "memmove" => (1, shift as *const () as usize),
        _ => return address,
    };
    if mode() == Mode::Off || !copies() {
        return address;
    }
    REAL[which].store(address, Relaxed);
    IMPORTS.fetch_add(1, Relaxed);
    ours
}

type Copy = unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8;

unsafe extern "system-unwind" fn copy(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: the title's copy, by the function it asked for.
    unsafe { copied(to, from, n, REAL[0].load(Relaxed)) }
}

unsafe extern "system-unwind" fn shift(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: as above.
    unsafe { copied(to, from, n, REAL[1].load(Relaxed)) }
}

/// The copy, then the marks: each slot of the marked heap the copy wrote
/// into that now holds a view record's tag.
///
/// # Safety
/// `real` copies as `memcpy` does; the rest are the title's arguments.
#[inline]
unsafe fn copied(to: *mut u8, from: *const u8, n: usize, real: usize) -> *mut u8 {
    // SAFETY: the host's function, noted by `import` before the title ran.
    let real: Copy = unsafe { std::mem::transmute(real) };
    // SAFETY: by the caller.
    let out = unsafe { real(to, from, n) };
    let len = MARKS.len.load(Acquire);
    if len != 0 && n >= 4 {
        let start = MARKS.start.load(Relaxed);
        let (lo, hi) = ((to as u64).max(start), (to as u64).saturating_add(n as u64).min(start + len));
        if lo < hi {
            COUNTS[COPIES].fetch_add(1, Relaxed);
            // Every slot the copy wrote a byte of (the fork marks only
            // those whose first word it wrote: a copy that starts inside a
            // record's first word and writes the rest of it would go
            // untold; one slot more at each end costs a look).
            for slot in (lo - start) / 32..=(hi - 1 - start) / 32 {
                // SAFETY: a slot of the heap, whose memory the copy just
                // wrote a byte of (a slot does not cross a page).
                if unsafe { ((start + slot * 32) as *const u32).read_unaligned() } == VIEW_TAG {
                    mark_slot(slot as usize, COPY);
                }
            }
        }
    }
    out
}

/// The counts of the frame line, since the last one.
const TOLD_BY_STORE: usize = 0;
const TOLD_BY_COPY: usize = 1;
const TAKINGS: usize = 2;
const TOLD_CHANGED: usize = 3;
const PUT_BACK: usize = 4;
const PAGES: usize = 5;
/// What a look at a written page found changed: view records a store told
/// of, a copy told of, nothing told of; records that are no view, untold
/// and told.
const FOUND_BY_STORE: usize = 6;
const FOUND_BY_COPY: usize = 7;
const FOUND_UNTOLD: usize = 8;
const FOUND_OTHER: usize = 9;
const FOUND_OTHER_TOLD: usize = 10;
/// Marks of a looked-at page's slots that had not changed; marks on pages
/// the watch did not give (`check`); the title's copies into the heap.
const TOLD_SAME: usize = 11;
const TOLD_ELSEWHERE: usize = 12;
const COPIES: usize = 13;
static COUNTS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// Since the start: view records nothing told of, by the place that last
/// faulted on their block; the title's imports bound to our copies.
static MISSED_ALL: AtomicU64 = AtomicU64::new(0);
static WRITERS: Mutex<std::collections::BTreeMap<String, u64>> = Mutex::new(std::collections::BTreeMap::new());
static IMPORTS: AtomicU64 = AtomicU64::new(0);

/// With the frame line (`frames` since the last one): what the reports
/// told and what the looks at written pages found. Nothing without the
/// switch.
pub(crate) fn log(frames: f64) {
    let mode = mode();
    if mode == Mode::Off {
        return;
    }
    let c: [f64; 14] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    let [patched, sites] = [2, 3].map(|i| IMAGE[i].load(Relaxed));
    let told = ADOPTED.read().unwrap_or_else(|e| e.into_inner()).as_ref().map(|a| a.told);
    let writers = {
        let by = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut most: Vec<(&String, &u64)> = by.iter().collect();
        most.sort_by(|a, b| b.1.cmp(a.1));
        let text = most.iter().take(6).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ");
        if text.is_empty() { "none".to_owned() } else { text }
    };
    let found = format!(
        "{:.1} written heap pages/frame looked at whole; slots/frame found changed there: to a view record \
         {:.0} told by a patched store, {:.0} told by a copy, {:.1} told by nothing; to no view {:.0} ({:.0} of \
         them told); {:.0} told slots/frame there had not changed",
        c[PAGES],
        c[FOUND_BY_STORE],
        c[FOUND_BY_COPY],
        c[FOUND_UNTOLD],
        c[FOUND_OTHER] + c[FOUND_OTHER_TOLD],
        c[FOUND_OTHER_TOLD],
        c[TOLD_SAME],
    );
    let since = format!(
        "since the start {} view records told by nothing, the last write faults on their blocks from: {writers}; \
         {patched} of {sites} stores patched, {} memcpy/memmove imports bound, {:.0} copies/frame into the heap",
        MISSED_ALL.load(Relaxed),
        IMPORTS.load(Relaxed),
        c[COPIES],
    );
    match told {
        None => tracing::info!("heap reports (VICTORY_HEAP_REPORTS): no placed view heap is marked yet; {since}"),
        Some(Told::Use) => tracing::info!(
            "heap reports (VICTORY_HEAP_REPORTS=1, the reports decide): {:.0} slots/frame told ({:.0} by patched \
             stores, {:.0} by copies) in {:.1} takings/frame, {:.0} of them changed and made, {:.0} put back by \
             rounds cut short; the fallback, every {} frames: {found}; {since}",
            c[TOLD_BY_STORE] + c[TOLD_BY_COPY],
            c[TOLD_BY_STORE],
            c[TOLD_BY_COPY],
            c[TAKINGS],
            c[TOLD_CHANGED],
            c[PUT_BACK],
            every(),
        ),
        Some(_) => tracing::info!(
            "heap reports (VICTORY_HEAP_REPORTS={}, the look at written pages decides, as before): {found}; \
             {:.0} slots/frame told on pages the watch did not give; {since}",
            if mode == Mode::Check { "check" } else { "1, but not every store is patched" },
            c[TOLD_ELSEWHERE],
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The marks are the process's: one test at a time adopts a heap.
    pub(crate) static ONE: Mutex<()> = Mutex::new(());

    #[test]
    fn the_switch_is_off_unless_asked_and_has_two_forms() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("")), Mode::Off);
        assert_eq!(mode_of(Some("0")), Mode::Off);
        assert_eq!(mode_of(Some("check")), Mode::Check);
        assert_eq!(mode_of(Some("1")), Mode::Use);
        assert_eq!((every_of(None), every_of(Some("x")), every_of(Some("0")), every_of(Some(" 3 "))), (8, 8, 1, 3));
    }

    #[test]
    fn marks_layout_is_what_the_stores_read() {
        assert_eq!(size_of::<Marks>(), MARKS_LEN);
        assert_eq!(std::mem::offset_of!(Marks, start), 0);
        assert_eq!(std::mem::offset_of!(Marks, len), 8);
        assert_eq!(std::mem::offset_of!(Marks, slots), 16);
        assert_eq!(std::mem::offset_of!(Marks, chunks), 24);
        assert_eq!(STORE, 1, "the byte the stores write");
    }

    /// What a patched store does after its store, as the loader's code
    /// does it: the address against the heap's start and length, then the
    /// slot's and the chunk's byte.
    fn store_marks(address: u64) {
        let at = address.wrapping_sub(MARKS.start.load(Relaxed));
        if at < MARKS.len.load(Relaxed) {
            mark_slot((at >> 5) as usize, STORE);
        }
    }

    /// Slots marked as the stores mark them come out once each, as runs in
    /// order; outside the heap nothing is marked; slots put back come out
    /// again; a heap that went is marked no more, and its maps serve the
    /// next heap of its size, cleared.
    #[test]
    fn told_slots_are_taken_once_as_runs() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x1000_0000u64;
        assert!(adopt_as(start, 1000, Mode::Use, true));
        assert_eq!((told(start, 1000), told(start + 32, 1000), told(start, 999)), (Told::Use, Told::No, Told::No));
        for slot in [5u64, 6, 300, 999] {
            store_marks(start + slot * 32 + if slot == 6 { 7 } else { 0 });
        }
        store_marks(start - 32);
        store_marks(start + 1000 * 32);
        assert_eq!(take(start), [(5, 2), (300, 1), (999, 1)]);
        assert_eq!(take(start), []);
        assert_eq!(take(start + 32), [], "not the marked heap");
        // A round cut short puts back what it took and did not look at; a
        // mark set since stays what it was.
        mark_slot(301, COPY);
        put_back(start, 300, 3);
        put_back(start, 998, 5);
        assert_eq!(take(start), [(300, 3), (998, 2)]);
        // Another placed view heap is not marked.
        assert!(!adopt_as(start + 0x10_0000, 64, Mode::Use, true));
        store_marks(start + 5 * 32);
        let maps = MARKS.slots.load(Relaxed);
        let_go_now(start);
        store_marks(start + 5 * 32);
        assert_eq!(MARKS.len.load(Relaxed), 0);
        assert_eq!(take(start), []);
        // The next heap of that size gets the same maps, cleared.
        assert!(adopt_as(start + 0x20_0000, 1000, Mode::Check, false));
        assert_eq!(MARKS.slots.load(Relaxed), maps);
        assert_eq!(take(start + 0x20_0000), [], "the mark of slot 5 went with the heap");
        let_go_now(start + 0x20_0000);
    }

    use super::let_go as let_go_now;

    #[test]
    fn the_fallback_asks_the_first_time_and_then_every_so_many_frames() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        ASKED.store(0, Relaxed);
        let asked: Vec<u64> = (0..30).filter(|&frame| ask_due_every(frame, 8)).collect();
        assert_eq!(asked, [0, 8, 16, 24]);
        // Several syncs of a frame: one asking.
        assert!(!ask_due_every(24, 8) && !ask_due_every(31, 8) && ask_due_every(32, 8));
        ASKED.store(0, Relaxed);
        assert!((5..9).all(|frame| ask_due_every(frame, 1)), "every frame");
        ASKED.store(0, Relaxed);
    }

    unsafe extern "system-unwind" fn plain_copy(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
        // SAFETY: by the caller, as `memmove`.
        unsafe { std::ptr::copy(from, to, n) };
        to
    }

    /// The model of a title heap for the test below: its records, and the
    /// record each host slot was last made from (as `HostHeap` keeps it).
    struct Model {
        records: Vec<[u32; 8]>,
        made: Vec<u64>,
    }

    impl Model {
        fn start(&self) -> u64 {
            self.records.as_ptr() as u64
        }

        fn key(&self, slot: usize) -> u64 {
            u64::from(self.records[slot][2]) << 32 | u64::from(self.records[slot][1])
        }

        fn is_view(&self, slot: usize) -> bool {
            self.records[slot][0] == VIEW_TAG
        }

        /// The slots of `slots` a heap sync finds changed
        /// (`super::super::slots::stale`, the rule of `HostHeap::sync_slots`).
        fn stale(&self, slots: impl Iterator<Item = usize>) -> Vec<usize> {
            slots
                .filter(|&s| {
                    let [_, view, generation, ..] = self.records[s];
                    crate::descriptor::slots::stale(self.is_view(s), view, generation, Some(self.made[s]))
                })
                .collect()
        }

        /// A sync makes slot `slot` from its record (none for no view).
        fn make(&mut self, slot: usize) {
            self.made[slot] = if self.is_view(slot) { self.key(slot) } else { 0 };
        }
    }

    /// The equivalence the `=1` form stands on, on generated writes: with
    /// every writer of view records telling (the stores, the copies; our
    /// own calls make their slot at once), the slots a sync finds changed
    /// among the told ones are exactly the slots a look at every slot finds
    /// changed to a view record; what the look finds besides are records
    /// that are no view (the fallback's). Two heaps made each way from the
    /// same writes show the same views after every sync, and the same
    /// everything after a fallback look.
    #[test]
    fn the_told_slots_are_the_ones_a_look_at_every_slot_finds() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        const SLOTS: usize = 5000;
        let mut scan = Model { records: vec![[0; 8]; SLOTS], made: vec![0; SLOTS] };
        let mut told_made = vec![0u64; SLOTS];
        let start = scan.start();
        assert!(adopt_as(start, SLOTS as u32, Mode::Use, true));
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let view = |n: u64| -> [u32; 8] { [VIEW_TAG, (n % 97) as u32, 1 + (n >> 8) as u32 % 3, 0, 5, 6, 7, 8] };
        let mut source: Vec<[u32; 8]> = vec![[0; 8]; 40];
        let (mut views_found, mut others_found) = (0, 0);
        for round in 0..400 {
            // The title's writes of this stretch.
            for _ in 0..next() % 60 {
                let slot = (next() % SLOTS as u64) as usize;
                match next() % 10 {
                    // A table filler's store: the record, then its mark.
                    0..=3 => {
                        scan.records[slot] = view(next());
                        store_marks(start + slot as u64 * 32);
                    }
                    // A copy of some records through the runtime's memcpy,
                    // from any byte to any byte: slots whose first word it
                    // did not write are not its to tell.
                    4..=5 => {
                        for (k, record) in source.iter_mut().enumerate() {
                            *record = if (next() + k as u64) % 5 == 0 { [0x1111_1111; 8] } else { view(next()) };
                        }
                        let records = 1 + (next() % 30) as usize;
                        let (lead, tail) = ((next() % 32) as usize, (next() % 32) as usize);
                        let slot = slot.min(SLOTS - records - 2);
                        let bytes = (records * 32 + tail).saturating_sub(lead);
                        // SAFETY: inside `source` and inside the model's records.
                        unsafe {
                            let to = (start as *mut u8).add(slot * 32 + lead);
                            copied(to, source.as_ptr().cast::<u8>().add(lead), bytes, plain_copy as *const () as usize);
                        }
                    }
                    // A view made by a call of ours: made at once on both hosts.
                    6 => {
                        scan.records[slot] = view(next());
                        scan.make(slot);
                        told_made[slot] = scan.made[slot];
                    }
                    // The allocator's other data over a slot: told by nothing.
                    7..=8 => scan.records[slot] = [next() as u32 | 1, next() as u32, 0, 0, 0, 0, 0, 0],
                    // The record the slot has, stored again.
                    _ => {
                        if scan.is_view(slot) {
                            store_marks(start + slot as u64 * 32);
                        }
                    }
                }
            }
            // The sync both ways, from the same state of the title's heap.
            let by_look = scan.stale(0..SLOTS);
            let told_model = Model { records: std::mem::take(&mut scan.records), made: std::mem::take(&mut told_made) };
            let told: Vec<usize> =
                take(start).into_iter().flat_map(|(first, n)| first as usize..(first + n) as usize).collect();
            let by_report = told_model.stale(told.iter().copied());
            let views: Vec<usize> = by_look.iter().copied().filter(|&s| told_model.is_view(s)).collect();
            let view_reports: Vec<usize> = by_report.iter().copied().filter(|&s| told_model.is_view(s)).collect();
            (scan.records, told_made) = (told_model.records, told_model.made);
            // On the second host's own state: the told slots that changed
            // are exactly the slots a look at every slot would find changed
            // to a view record.
            let pending: Vec<usize> =
                (0..SLOTS).filter(|&s| scan.is_view(s) && told_made[s] != scan.key(s)).collect();
            assert_eq!(view_reports, pending, "round {round}: found by report, found by a look at every slot");
            // The first host's look finds them all, and besides only view
            // records the second host still shows from before they were
            // overwritten (a record stored again where other data had been).
            assert!(pending.iter().all(|s| views.contains(s)), "round {round}");
            assert!(views.iter().all(|&s| pending.contains(&s) || told_made[s] == scan.key(s)), "round {round}");
            views_found += view_reports.len();
            for &slot in &by_look {
                scan.make(slot);
            }
            for &slot in &by_report {
                told_made[slot] = if scan.is_view(slot) { scan.key(slot) } else { 0 };
            }
            // Both hosts show the same view in every slot that holds a
            // view record.
            for slot in 0..SLOTS {
                if scan.is_view(slot) {
                    assert_eq!(told_made[slot], scan.made[slot], "round {round}, slot {slot}");
                }
            }
            // What differs: slots whose record is no view any more, which
            // the second host still shows the old view in until a fallback
            // look (here every 8th round) makes them null.
            let stale_others: Vec<usize> = (0..SLOTS).filter(|&s| told_made[s] != scan.made[s]).collect();
            assert!(stale_others.iter().all(|&s| !scan.is_view(s) && scan.made[s] == 0));
            if round % 8 == 7 {
                others_found += stale_others.len();
                for slot in stale_others {
                    told_made[slot] = 0;
                }
                assert_eq!(told_made, scan.made, "round {round}: after a fallback look the hosts are the same");
            }
        }
        assert!(views_found > 2000 && others_found > 100, "the test met both kinds: {views_found}, {others_found}");
        let_go_now(start);
    }

    /// The `check` form's counts: a look at a written page sorts the slots
    /// it found changed by what told of them, and clears the page's marks.
    #[test]
    fn a_look_sorts_what_it_found_by_who_told() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x7000_0000u64;
        assert!(adopt_as(start, 4096, Mode::Check, true));
        for count in &COUNTS {
            count.store(0, Relaxed);
        }
        let missed_before = MISSED_ALL.load(Relaxed);
        // On the page of slots 2048..4096: 2050 by a store, 2051 by a
        // copy, 2052 and 2053 by nothing, 2054 told and unchanged; slot 7
        // told on another page.
        mark_slot(2050, STORE);
        mark_slot(2051, COPY);
        mark_slot(2054, STORE);
        mark_slot(2055, COPY);
        mark_slot(7, STORE);
        // A write fault on that page's block, from the title's code.
        hooked(0x1_4000_0000, 0x1000_0000, 6, 6);
        faulted((start + 2052 * 32) as usize, 0x1_4000_1234);
        let views = [2050, 2051, 2052];
        looked(start, 2048, 4096, &[2050, 2051, 2052, 2053, 2055], |i| views.contains(&i));
        let c = |i: usize| COUNTS[i].load(Relaxed);
        assert_eq!((c(PAGES), c(FOUND_BY_STORE), c(FOUND_BY_COPY), c(FOUND_UNTOLD)), (1, 1, 1, 1));
        assert_eq!((c(FOUND_OTHER), c(FOUND_OTHER_TOLD), c(TOLD_SAME)), (1, 1, 1));
        assert_eq!(MISSED_ALL.load(Relaxed), missed_before + 1);
        assert_eq!(WRITERS.lock().unwrap().get("title+0x1234"), Some(&1), "the untold record's last writer");
        // The page's marks are gone; the other page's is the rest.
        clear_rest(start);
        assert_eq!(c(TOLD_ELSEWHERE), 1);
        assert_eq!(take(start), []);
        hooked(0, 0, 0, 0);
        WRITERS.lock().unwrap().clear();
        let_go_now(start);
    }

    /// The real sync with the reports deciding, on a placed view heap of
    /// the host's: a record copied in as bytes and told of is made at the
    /// next sync; one that nothing told of waits for the fallback's look
    /// at the written pages, and so does a record overwritten with other
    /// data, which keeps its view until then. In the `check` form every
    /// record is made at the next sync, as without the switch.
    #[cfg(windows)]
    #[test]
    fn a_sync_makes_the_told_slots_and_the_fallback_the_rest() {
        use crate::descriptor::{HEAP_IIDS, HeapDesc, Placement, SHADER_VISIBLE, VIEWS_KIND, create_cbv, create_heap, heap, sync_heaps};
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        const SLOTS: u32 = 2048;
        let memory = crate::watch::tests::pool(1);
        #[repr(C)]
        struct Extended(HeapDesc, Placement);
        let desc = HeapDesc { kind: VIEWS_KIND, count: SLOTS, flags: SHADER_VISIBLE, nodes: 1 };
        let extended = Extended(desc, Placement { address: memory as u64, size: u64::from(SLOTS) * 32 });
        let mut h = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut h), crate::object::S_OK);
        let start = memory as u64;
        let at = |slot: u32| start + 32 * u64::from(slot);
        let release = |h| crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(h, 0x10)(h);
        let Some(host) = heap(h).host() else {
            release(h);
            crate::watch::tests::free(memory);
            return;
        };
        // A record of ours, written where no heap sees it, copied in as
        // bytes the way the title copies records (the heap's page is
        // guarded: our handler opens it).
        let write = |slot: u32, n: u32| {
            let mut elsewhere = [0u64; 4];
            create_cbv(std::ptr::null(), &[0x1000 * (n + 1), 1, 256, 0], elsewhere.as_mut_ptr() as u64);
            // SAFETY: 32 bytes of each, ours.
            unsafe { std::ptr::copy_nonoverlapping(elsewhere.as_ptr().cast::<u8>(), at(slot) as *mut u8, 32) };
        };
        let ask_again = || ASKED.store(0, Relaxed);
        assert!(adopt_as(start, SLOTS, Mode::Use, true));
        assert_eq!(told(start, SLOTS), Told::Use);
        // The first sync of a run asks the watch: done with here.
        ask_again();
        sync_heaps();
        for count in &COUNTS {
            count.store(0, Relaxed);
        }
        let c = |i: usize| COUNTS[i].load(Relaxed);
        write(5, 0);
        write(6, 1);
        write(700, 2);
        store_marks(at(5));
        store_marks(at(700));
        sync_heaps();
        assert!(host.made(5) && host.made(700), "told and made at the sync");
        assert!(!host.made(6), "told by nothing: not looked at");
        assert_eq!((c(TOLD_BY_STORE), c(TOLD_CHANGED), c(PAGES)), (2, 2, 0));
        // The same record told again: looked at, nothing to make.
        store_marks(at(5));
        sync_heaps();
        assert_eq!((c(TOLD_BY_STORE), c(TOLD_CHANGED)), (3, 2));
        // The fallback finds what nothing told of.
        ask_again();
        sync_heaps();
        assert!(host.made(6), "made by the fallback's look at the written page");
        assert_eq!((c(PAGES), c(FOUND_UNTOLD)), (1, 1));
        // Other data over a record: the slot keeps its view until the
        // fallback looks.
        // SAFETY: slot 5 of the heap's memory.
        unsafe { std::ptr::write_bytes(at(5) as *mut u8, 0x11, 32) };
        sync_heaps();
        assert!(host.made(5), "still showing its view");
        ask_again();
        sync_heaps();
        assert!(!host.made(5), "made null by the fallback");
        assert_eq!(c(FOUND_OTHER), 1);
        // The `check` form: the look at written pages decides, as before.
        let_go_now(start);
        assert_eq!(told(start, SLOTS), Told::No);
        assert!(adopt_as(start, SLOTS, Mode::Check, true));
        write(9, 3);
        write(10, 4);
        store_marks(at(10));
        sync_heaps();
        assert!(host.made(9) && host.made(10), "made at the next sync, told or not");
        assert_eq!((c(FOUND_UNTOLD), c(FOUND_BY_STORE)), (2, 1));
        let_go_now(start);
        ask_again();
        release(h);
        crate::watch::tests::free(memory);
    }
}
