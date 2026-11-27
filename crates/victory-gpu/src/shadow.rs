//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

fn merge_writeback(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = 0;
    for ((title_byte, host_byte), old_byte) in title.iter_mut().zip(host).zip(old) {
        let mask = 0u8.wrapping_sub(u8::from(*host_byte != *old_byte));
        let delta = (*title_byte ^ *host_byte) & mask;
        *title_byte ^= delta;
        *host_byte = *title_byte;
        changed |= delta;
    }
    changed != 0
}

#[test]
fn writeback_merge_matches_bytewise_oracle() {
    for len in [0, 1, 63, 64, 65, 193, 65_536] {
        let old = vec![0x11; len];
        for stride in [1, 5, 67, len + 1] {
            let mut title = old.clone();
            let mut host = old.clone();
            for index in (0..len).step_by(3) {
                title[index] = 0x22;
            }
            for index in (0..len).step_by(stride) {
                host[index] = if index % 2 == 0 { 0x22 } else { 0x33 };
            }
            let expected: Vec<_> = host
                .iter()
                .zip(&old)
                .zip(&title)
                .map(|((&gpu, &previous), &cpu)| if gpu == previous { cpu } else { gpu })
                .collect();
            let changed = expected != title;
            assert_eq!(merge_writeback(&mut title, &mut host, &old), changed);
            assert_eq!(title, expected);
            assert_eq!(host, expected);
        }
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Widened three-channel views reading this copy, refreshed when their
    /// source bytes change. The fourth component is part of the key: float
    /// views need the bits of 1.0, integer views need the integer 1.
    widened: Mutex<WidenedViews>,
    /// Serializes source mutations with widened-cache construction and refresh.
    ///
    /// Lock order, outermost first: `widened_source`, then `before` and
    /// `pending`, then `widened`. `Shadow::widened` holds `widened_source`
    /// across `upload_buffer`, which takes no lock of ours (a plain D3D12
    /// resource creation), and takes `widened` only for short lookups
    /// inside it. Every path that writes the copy's bytes (`update`,
    /// `copy_back_or_not`, `copy_back_marked`, `copy_back_page`) takes
    /// `widened_source` first and then calls `refresh_widened`, which takes
    /// only `widened`; none of them calls another that takes it again, as
    /// the mutex is not reentrant.
    widened_source: Mutex<()>,
    has_widened: AtomicBool,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
    /// True when the title's virtual range `[start, start + len)` is
    /// `mach_vm_remap`-aliased onto `buffer`'s physical pages (zero-copy).
    aliased: bool,
    /// The last presented frame a view or a list asked for it
    /// ([`Shadow::touch`]), for [`release_idle`].
    used: AtomicU64,
}

/// A 16-byte-element host copy of a three-channel 32-bit buffer view. D3DMetal
/// reads R32G32B32 typed SRVs as R32G32; an R32G32B32A32 view over this copy
/// keeps the title's 12-byte stride and supplies `w` in the source format.
pub(crate) struct Widened {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    pub(crate) start: u64,
    pub(crate) len: u64,
    pub(crate) w: u32,
    source: Arc<Shadow>,
}

// SAFETY: the resource is free-threaded; `data` is a persistent mapping
// written only by the submitting thread.
unsafe impl Send for Widened {}
// SAFETY: as above.
unsafe impl Sync for Widened {}

/// Widened views by source range and fourth-component bits.
type WidenedViews = Vec<(u64, u64, u32, std::sync::Weak<Widened>)>;

/// Widened views by title range, including views whose exact shadow differs
/// from the parent shadow a raw UAV writes.
static WIDENED_RANGES: Mutex<Vec<(u64, u64, std::sync::Weak<Widened>)>> = Mutex::new(Vec::new());
/// Whether any widened view has been made. Writes are retained from the start
/// of a list, so the first view made after a write can still consume them.
pub(crate) static WIDENED_ANY: AtomicBool = AtomicBool::new(false);

fn live_widened() -> Vec<Arc<Widened>> {
    let mut ranges = WIDENED_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    if ranges.len().is_power_of_two() {
        ranges.retain(|(_, _, view)| view.strong_count() > 0);
    }
    ranges.iter().filter_map(|(_, _, view)| view.upgrade()).collect()
}

/// Every live widened view, for lists tracking which ones caught up with writes.
pub(crate) fn widened_all() -> Vec<Arc<Widened>> {
    live_widened()
}

/// Live widened views overlapping a title-memory range, across shadows.
pub(crate) fn widened_in(from: usize, to: usize) -> Vec<Arc<Widened>> {
    if from >= to {
        return Vec::new();
    }
    live_widened()
        .into_iter()
        .filter(|view| {
            let end = view.start.saturating_add(view.len);
            view.start < to as u64 && (from as u64) < end
        })
        .collect()
}

#[cfg(test)]
type WidenedPause = (Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);
#[cfg(test)]
static WIDENED_AFTER_REFRESH: Mutex<Vec<(u64, WidenedPause)>> = Mutex::new(Vec::new());

impl Widened {
    /// The normal shadow this cache was expanded from.
    pub(crate) fn source(&self) -> Arc<Shadow> {
        self.source.clone()
    }

    /// The whole elements of this view touched by `[from, to)`, as indices
    /// into the widened buffer. Raw UAV writes can touch part of an element.
    pub(crate) fn overlap(&self, from: usize, to: usize) -> Option<(usize, usize)> {
        let start = self.start as usize;
        let end = start.saturating_add(self.len as usize);
        let from = from.max(start);
        let to = to.min(end);
        (from < to).then(|| ((from - start) / 12, (to - start).div_ceil(12)))
    }

    fn refresh(&self, shadow: &Shadow, from: usize, to: usize) {
        let start = self.start as usize;
        let end = start + self.len as usize;
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return;
        }
        let first = (from - start) / 12;
        let last = (to - start).div_ceil(12);
        // SAFETY: the source range is inside the shadow and the destination
        // has one 16-byte element for each 12 source bytes.
        unsafe {
            let source = shadow.data.add(start - shadow.start as usize);
            for i in first..last {
                let at = source.add(i * 12);
                let to = self.data.add(i * 16);
                std::ptr::copy_nonoverlapping(at, to, 12);
                to.add(12).cast::<u32>().write_volatile(self.w);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn read_host(&self, n: usize) -> Vec<u32> {
        assert!(4 * n <= (self.len / 12 * 16) as usize);
        // SAFETY: inside the mapped widened buffer; the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.cast::<u32>(), n).to_vec() }
    }
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<crate::recent::Pool> = Mutex::new(crate::recent::Pool::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let taken = POOL.lock().unwrap_or_else(|e| e.into_inner()).take(n);
    taken.unwrap_or_else(|| Vec::with_capacity(n))
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
/// What no snapshot used for a while goes at the present ([`trim_pool`]).
fn give_back(v: Vec<u8>) {
    let now = std::time::Instant::now();
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.give(v, now, POOL_BYTES);
}

/// At each present: frees the largest pooled buffer no snapshot took lately
/// (the peak of a heavy frame, which `pooled` never picks again once lighter
/// frames fit smaller ones), outside the pool's lock.
pub(crate) fn trim_pool() {
    let now = std::time::Instant::now();
    let freed = POOL.lock().unwrap_or_else(|e| e.into_inner()).trim(now);
    drop(freed);
}

/// Snapshot memory kept for reuse at most.
const POOL_BYTES: usize = 512 << 20;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
        if self.has_widened.load(Ordering::Acquire) {
            self.refresh_widened(address as usize, address as usize + bytes.len());
        }
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Up to `len` bytes of the copy at title address `address`, as hex
    /// dwords (diagnostics: `VICTORY_TRACE_READS`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek_words(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len as usize).min((self.len as usize).saturating_sub(at)) / 4;
        let words: Vec<String> = (0..n)
            // SAFETY: inside the mapping (checked); a racing GPU write only
            // makes the logged value stale.
            .map(|i| format!("{:x}", unsafe { self.data.add(at + 4 * i).cast::<u32>().read_unaligned() }))
            .collect();
        words.join(" ")
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// A widened copy of `len` bytes at `address` for a three-channel
    /// 32-bit typed SRV. The source stays in its normal 12-byte layout;
    /// this copy has 16-byte elements with the requested `w` bits.
    pub(crate) fn widened(
        self: &Arc<Self>,
        address: u64,
        len: u64,
        w: u32,
    ) -> Option<Arc<Widened>> {
        if len == 0 || !len.is_multiple_of(12) {
            return None;
        }
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let found = || {
            self.widened
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .find(|&&(start, size, fourth, _)| {
                    start == address && size == len && fourth == w
                })
                .and_then(|(_, _, _, view)| view.upgrade())
        };
        if let Some(view) = found() {
            return Some(view);
        }
        let count = len / 12;
        let (buffer, data) = host()?.shared_buffer(count * 16)?;
        let widened = Arc::new(Widened {
            buffer,
            data,
            start: address,
            len,
            w,
            source: self.clone(),
        });
        widened.refresh(self, address as usize, (address + len) as usize);
        #[cfg(test)]
        if let Some((ready, resume)) = WIDENED_AFTER_REFRESH
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|(start, _)| *start == self.start)
            .map(|(_, barriers)| barriers.clone())
        {
            ready.wait();
            resume.wait();
        }
        let mut views = self.widened.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have made the same view while this one was
        // being allocated.
        if let Some(view) = views
            .iter()
            .find(|&&(start, size, fourth, _)| {
                start == address && size == len && fourth == w
            })
            .and_then(|(_, _, _, view)| view.upgrade())
        {
            return Some(view);
        }
        if views.len().is_power_of_two() {
            views.retain(|(_, _, _, view)| view.strong_count() > 0);
        }
        views.push((address, len, w, Arc::downgrade(&widened)));
        drop(views);
        let mut ranges = WIDENED_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        if ranges.len().is_power_of_two() {
            ranges.retain(|(_, _, view)| view.strong_count() > 0);
        }
        ranges.push((address, len, Arc::downgrade(&widened)));
        WIDENED_ANY.store(true, Ordering::Release);
        self.has_widened.store(true, Ordering::Release);
        Some(widened)
    }

    fn refresh_widened(&self, from: usize, to: usize) {
        let views: Vec<_> = self
            .widened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|(_, _, _, view)| view.upgrade())
            .collect();
        for view in views {
            view.refresh(self, from, to);
        }
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if self.aliased {
            if self.has_widened.load(Ordering::Acquire) {
                let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
                self.refresh_widened(from, to);
            }
            let marked = crate::watch::notify_written(from, to - from);
            crate::alias::noted(to - from, marked, false);
            return 0;
        }
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        // The normal shadow holds the GPU's bytes even when title memory
        // already matches or cannot be written; widened views still need them.
        if self.has_widened.load(Ordering::Acquire) {
            self.refresh_widened(from, to);
        }
        // Taken early, while the submission ran: as a page the title wrote.
        if self.kept_early(page) {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.touch();
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// A view or a list asks for the copy in this frame. The refresh of
    /// every copy a page table names does not count: it would keep them all.
    fn touch(&self) {
        let frame = crate::queue::presented();
        if self.used.load(Ordering::Relaxed) != frame {
            self.used.store(frame, Ordering::Relaxed);
        }
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            widened: Mutex::default(),
            widened_source: Mutex::default(),
            has_widened: AtomicBool::new(false),
            base: slice.map_or(0, |s| s.offset),
            slice,
            aliased: false,
            used: AtomicU64::new(crate::queue::presented()),
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// A zero-copy shadow whose title range `[start, start + len)` shares
    /// physical pages with `buffer` (`mach_vm_remap`).
    pub(crate) fn new_aliased(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
    ) -> Arc<Shadow> {
        let marks = marks().take(pages(start, len));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked: None,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            widened: Mutex::default(),
            widened_source: Mutex::default(),
            has_widened: AtomicBool::new(false),
            base: 0,
            slice: None,
            aliased: true,
            used: AtomicU64::new(crate::queue::presented()),
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// Whether the title's memory is this copy's own (`crate::alias`): it is
    /// never copied, nor copied back.
    pub(crate) fn is_aliased(&self) -> bool {
        self.aliased
    }

    /// Where title address `address` is in the copy's mapping.
    pub(crate) fn host_ptr(&self, address: u64) -> *mut u8 {
        // SAFETY: the caller asks about an address inside the copy.
        unsafe { self.data.add(self.offset(address) as usize) }
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
        list.mark_widened_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        if self.aliased {
            // Nothing to keep: shaders write title memory itself. A range
            // whose writes are not marked (root views) is noted, so that its
            // copy-back tells the other trackers over it; marked writes are
            // told page by page from the marks ([`Shadow::copy_back_marked`]).
            if !marked {
                self.pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert((address, address + len));
            }
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        if self.aliased {
            // Shaders wrote title memory itself. The other trackers over a
            // range are told here only where its writes were not marked (the
            // marks told them page by page): every declared range told whole
            // marked all of the title's big heaps changed at each submission.
            // Writes meant to stay out of title memory (`crate::memory::
            // withholding`, `back` false) are in it already; their marks were
            // dropped, so the range is told whole, and counted.
            let declared = self
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&(address, address + len));
            let (from, to) = self.inside(address, len);
            if from < to && (declared || !back) {
                let start = self.start as usize;
                if self.has_widened.load(Ordering::Acquire) {
                    let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
                    self.refresh_widened(start + from, start + to);
                }
                let marked = crate::watch::notify_written(start + from, to - from);
                crate::alias::noted(to - from, marked, !back);
            }
            return 0;
        }
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = segments.iter().find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &before[from - taken..to - taken];
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    if merge_writeback(title, host, old) {
                        let _writing = crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                if self.has_widened.load(Ordering::Acquire) {
                    self.refresh_widened(start + at, start + at + n);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Kept GPU writes stay in the normal shadow for later submissions;
        // their widened readers must follow even though nothing goes back.
        if !back && from < to && self.has_widened.load(Ordering::Acquire) {
            self.refresh_widened(start + from, start + to);
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, pages: &[usize]) -> usize {
        if self.aliased {
            let (start, end) = (self.start as usize, (self.start + self.len) as usize);
            let mut run: Option<(usize, usize)> = None;
            let flush = |from: usize, to: usize| {
                if self.has_widened.load(Ordering::Acquire) {
                    let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
                    self.refresh_widened(from, to);
                }
                let marked = crate::watch::notify_written(from, to - from);
                crate::alias::noted(to - from, marked, false);
            };
            for &page in pages {
                let (from, to) = (page.max(start), (page + PAGE).min(end));
                if from >= to {
                    continue;
                }
                match run {
                    Some((r_from, r_to)) if r_to == from => run = Some((r_from, to)),
                    Some((r_from, r_to)) => {
                        flush(r_from, r_to);
                        run = Some((from, to));
                    }
                    None => run = Some((from, to)),
                }
            }
            if let Some((from, to)) = run {
                flush(from, to);
            }
            return 0;
        }
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        // What goes back (address ranges), and the pages equal to title
        // memory once it has: written, then guarded, all at once (each
        // opening and guarding is a Wine call and a guard-lock round trip,
        // ~1,000 pages a frame in #50's world).
        // mac: with `VICTORY_COPY_BACK_SPLIT=1`, a long list of one copy's
        // pages is looked at, and then written, on several threads
        // ([`copy_back_split`]); else on this one, as before.
        let long = copy_back_split() && pages.len() >= SPLIT_MIN_PAGES && !self.has_widened.load(Ordering::Acquire);
        let split = long && !crate::watch::aliased(start, end - start);
        if long {
            split_said(split, pages.len());
        }
        #[cfg(test)]
        MARKED_SPLIT.with(|taken| taken.set(Some(split)));
        let (writes, taken) = if split {
            let parts = pages.len().div_ceil(SPLIT_MIN_PAGES / 2).min(SPLIT_PARTS);
            let each = pages.len().div_ceil(parts);
            let done: Vec<Mutex<(crate::watch::Runs, Vec<usize>)>> = (0..parts).map(|_| Mutex::default()).collect();
            crate::texture::in_parallel(parts, |k| {
                let part = &pages[(k * each).min(pages.len())..((k + 1) * each).min(pages.len())];
                *done[k].lock().unwrap_or_else(|e| e.into_inner()) = self.marked_pages(t, part, start, end);
            });
            let (mut writes, mut taken) = (Vec::new(), Vec::new());
            for part in done {
                let (w, p) = part.into_inner().unwrap_or_else(|e| e.into_inner());
                writes.extend(w);
                taken.extend(p);
            }
            (writes, taken)
        } else {
            self.marked_pages(t, pages, start, end)
        };
        let write = |(from, to): (usize, usize)| {
            let len = to - from;
            let _writing = crate::watch::writing(from, len);
            // SAFETY: this run is inside the shadow. The GPU is done and
            // widened_source keeps the merged bytes stable until the write.
            let src = unsafe { self.data.add(from - start) };
            crate::host::write_mapped(from as *mut u8, src, len);
            len
        };
        let written = if split && writes.len() >= 2 {
            // The runs do not overlap: each part writes its own.
            let parts = writes.len().min(SPLIT_PARTS);
            let each = writes.len().div_ceil(parts);
            let total = std::sync::atomic::AtomicUsize::new(0);
            crate::texture::in_parallel(parts, |k| {
                let part = &writes[(k * each).min(writes.len())..((k + 1) * each).min(writes.len())];
                let n: usize = part.iter().map(|&run| write(run)).sum();
                total.fetch_add(n, Ordering::Relaxed);
            });
            total.into_inner()
        } else {
            writes.into_iter().map(write).sum()
        };
        if !taken.is_empty() {
            t.taken(&taken);
        }
        written
    }

    /// [`Shadow::copy_back_marked`]'s look at `pages` (in order) of the
    /// copy at `[start, end)`: the title ranges that go back (written
    /// after, with the copy's bytes), and the pages that are then equal to
    /// title memory. Pages of one copy are each looked at by one call, so
    /// calls on disjoint lists may run at once (`VICTORY_COPY_BACK_SPLIT`).
    fn marked_pages(
        &self,
        t: &crate::watch::Tracked,
        pages: &[usize],
        start: usize,
        end: usize,
    ) -> (crate::watch::Runs, Vec<usize>) {
        let mut writes: crate::watch::Runs = Vec::new();
        let mut taken = Vec::new();
        let mut title = Vec::new();
        for &page in pages {
            let (from, to) = (page.max(start), (page + PAGE).min(end));
            if from >= to {
                continue;
            }
            // Taken early, while the submission ran: as a page the title wrote.
            if self.kept_early(page) {
                continue;
            }
            // Our own write below must not save the page for this copy again.
            t.unarm(page / PAGE);
            let saved = t.saved(page / PAGE);
            if saved.is_none() && t.changed(page / PAGE) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::MarkedKept, 1);
                continue;
            }
            let n = to - from;
            // SAFETY: inside the mapping; only the submitting thread writes
            // it (mac: with the split, one thread each page).
            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
            title.resize(n, 0);
            if !crate::host::read_into(from as *const u8, &mut title) {
                continue;
            }
            let differs = match &saved {
                // The title's bytes are the copy's from before: what differs
                // is the shaders'.
                None => *host != title[..],
                Some(before) => {
                    let before = &before[from - page..to - page];
                    merge_writeback(&mut title, host, before)
                }
            };
            if differs {
                match writes.last_mut() {
                    Some((_, end)) if *end == from => *end = to,
                    _ => writes.push((from, to)),
                }
            }
            if self.has_widened.load(Ordering::Acquire) {
                self.refresh_widened(from, to);
            }
            // Equal to title memory now: taken, unless it holds the page in part.
            if from == page && to == page + PAGE {
                taken.push(page / PAGE);
            }
        }
        (writes, taken)
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        if self.aliased {
            return !self.has_widened.load(Ordering::Acquire);
        }
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.aliased {
            if self.has_widened.load(Ordering::Acquire) {
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                let (from, to) = (address.max(start), address.saturating_add(len).min(end));
                if from < to {
                    let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
                    self.refresh_widened(from, to);
                }
            }
            return 0;
        }
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty(), false);
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them, the title's bytes
            // as well, as nothing says what it wrote since: a submission
            // waits for the one before it, so none of this submission's
            // shaders has run yet, and a constant the title wrote there
            // would stay stale for the whole submission. The same bytes go
            // into the snapshot `writing` took, so copy-back tells the
            // shader's write from them.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                if e > a.max(at) {
                    let (a, e) = (a.max(at), e);
                    bytes += self.update_untracked(a, e);
                    self.sync_snapshot(a, e);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    ///
    /// `early` ([`Shadow::update_early`]): the pages taken are noted, and
    /// the parts on memory that is not the runtime's are left for the
    /// submission's own refresh (copied whole each time, they would only
    /// be copied twice).
    fn update_tracked(&self, from: usize, to: usize, busy: bool, early: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        if early {
            self.note_early(runs.iter().chain(&unmapped));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        let mut sync_before = |a: usize, b: usize| {
            if let Some(before) = &mut before {
                self.sync_snapshot_in(before, a, b);
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            sync_before(a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            if self.has_widened.load(Ordering::Acquire) {
                self.refresh_widened(a, b);
            }
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to).into_iter().filter(|_| !early) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += self.copy(a, b);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// Puts the copy's bytes of `[a, b)` (title addresses), just updated,
    /// into the snapshot a submission that is writing took.
    fn sync_snapshot(&self, a: usize, b: usize) {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        self.sync_snapshot_in(&mut before, a, b);
    }

    fn sync_snapshot_in(&self, before: &mut Snapshot, a: usize, b: usize) {
        let start = self.start as usize;
        let (a, b) = (a - start, b - start);
        for (at, bytes) in before.0.iter_mut() {
            let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
            if s < e {
                // SAFETY: inside the mapping.
                let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                bytes[s - *at..e - *at].copy_from_slice(host);
            }
        }
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Whether the copy may be brought up to date while a submission still
    /// runs on the host GPU ([`Shadow::update_early`]): its memory is
    /// tracked, and no submitted list declared that its shaders write it
    /// (no range pending, no snapshot kept).
    pub(crate) fn idle(&self) -> bool {
        self.tracked.is_some()
            && self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.before.lock().unwrap_or_else(|e| e.into_inner()).0.is_empty()
    }

    /// [`Shadow::update`] made early, for a submission not yet prepared,
    /// while the one before it still runs on the host GPU
    /// (`crate::host::unwaited`, under the submission lock): the pages of
    /// `[address, address + len)` the title wrote since the copy took them
    /// are taken and copied now, so the submission's own refresh finds them
    /// done unless the title writes them again. Only for a copy that is
    /// [`Shadow::idle`] (`None` otherwise, and nothing is touched): where
    /// shaders of the running submission write, title memory is older than
    /// the copy until they are copied back, and `update` would put it over
    /// what they wrote.
    ///
    /// The pages taken are noted until the running submission is finished
    /// ([`forget_early`]): should its shaders have marked one as written
    /// all the same (through a view no list declared), the title's bytes
    /// are kept there, as for a page the title wrote meanwhile
    /// ([`Shadow::kept_early`]).
    pub(crate) fn update_early(&self, address: usize, len: usize) -> Option<usize> {
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        if !self.idle() {
            return None;
        }
        if self.clean() {
            return Some(0);
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return Some(0);
        }
        Some(self.update_tracked(from, to, false, true))
    }

    /// Notes the pages of `runs` (title addresses) as taken early.
    fn note_early<'a>(&self, runs: impl Iterator<Item = &'a (usize, usize)>) {
        let key = std::ptr::from_ref(self) as usize;
        let mut early = EARLY.lock().unwrap_or_else(|e| e.into_inner());
        for &(a, b) in runs {
            early.extend((a / PAGE..b.div_ceil(PAGE)).map(|page| (key, page)));
        }
        EARLY_ANY.store(!early.is_empty(), Ordering::Release);
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written in this copy: whether the copy took the page
    /// early, while that submission ran ([`Shadow::update_early`]). Then
    /// the copy's bytes there are title memory's with, at most, a part of
    /// what the shaders wrote: nothing goes back, and the copy takes the
    /// page from title memory again at its next refresh, which is what
    /// becomes of a page the title wrote while shaders wrote it. Counted:
    /// the copies refreshed early are chosen so that this does not happen.
    fn kept_early(&self, page: usize) -> bool {
        if !EARLY_ANY.load(Ordering::Acquire) {
            return false;
        }
        let key = (std::ptr::from_ref(self) as usize, page / PAGE);
        if !EARLY.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
            return false;
        }
        crate::watch::retake(page, PAGE);
        EARLY_MARKED.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if self.has_widened.load(Ordering::Acquire) {
            self.refresh_widened(from, to);
        }
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }
}

/// The pages (copy, page number) copies took early, while a submission ran
/// on the host GPU ([`Shadow::update_early`]), until it is finished
/// ([`forget_early`]); whoever notes them holds the copies that long, so
/// that a copy's address names no other.
static EARLY: Mutex<std::collections::BTreeSet<(usize, usize)>> = Mutex::new(std::collections::BTreeSet::new());

/// Whether [`EARLY`] holds anything (asked at every page that goes back).
static EARLY_ANY: AtomicBool = AtomicBool::new(false);

/// Pages taken early that the running submission's shaders marked as
/// written all the same ([`Shadow::kept_early`]), since last asked.
pub(crate) static EARLY_MARKED: AtomicU64 = AtomicU64::new(0);

/// The submission that ran while copies took pages early is finished: its
/// shaders' writes have gone back, and the pages are as any other.
pub(crate) fn forget_early() {
    if EARLY_ANY.swap(false, Ordering::AcqRel) {
        EARLY.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        }
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The two bitmaps of [`bitmap`], each made when first asked for.
static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] =
    [const { std::sync::OnceLock::new() }; 2];

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index, then the lock
            // words of emulated 64-bit atomics.
            let bytes = u64::from(victory_shader::emit::VIEW_WRITES_BYTES);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Frees the lock words of emulated 64-bit atomics (`victory_shader::emit`,
/// "Atomic locks") in the bitmaps made so far, after a submission the host
/// GPU ended: a lane that held a lock when its shader was cut short never
/// frees it, and every later atomic on that lock would wait out its bound
/// and go on without it (#240). Returns how many were taken. No submission
/// runs then, so no lane holds one.
pub(crate) fn free_atomic_locks() -> usize {
    let dwords = (victory_shader::emit::VIEW_WRITES_BYTES / 4) as usize;
    BITMAPS
        .iter()
        .filter_map(|b| b.get()?.as_ref())
        .map(|b| {
            // SAFETY: the mapped buffer of `dwords` dwords; the GPU is done
            // with it (one submission runs at a time, and none runs now).
            let words = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, dwords) };
            free_locks(words)
        })
        .sum()
}

/// Clears the lock words in `words`, a bitmap buffer's dwords, leaving the
/// written pages and read marks before them and the give-up count after
/// them. Returns how many were taken.
fn free_locks(words: &mut [u32]) -> usize {
    let first = (victory_shader::emit::ATOMIC_LOCKS_AT / 4) as usize;
    let locks = &mut words[first..first + victory_shader::emit::ATOMIC_LOCKS as usize];
    let taken = locks.iter().filter(|&&w| w != 0).count();
    locks.fill(0);
    taken
}

/// After a submission: how many lanes gave up waiting for a 64-bit
/// atomic's lock since the last call and went on without it, so that their
/// updates may be lost (`victory_shader::emit`, "Atomic locks"; #240). The
/// counts are cleared.
pub(crate) fn take_gave_up() -> u64 {
    let dwords = (victory_shader::emit::VIEW_WRITES_BYTES / 4) as usize;
    BITMAPS
        .iter()
        .filter_map(|b| b.get()?.as_ref())
        .map(|b| {
            // SAFETY: the mapped buffer of `dwords` dwords; the GPU is done
            // with it (one submission runs at a time, and none runs now).
            let words = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, dwords) };
            take_count(words)
        })
        .sum()
}

/// Takes the give-up count from `words`, a bitmap buffer's dwords.
fn take_count(words: &mut [u32]) -> u64 {
    u64::from(std::mem::take(
        &mut words[(victory_shader::emit::ATOMIC_LOCKS_GAVE_UP_AT / 4) as usize],
    ))
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let mut maps: Vec<&mut [u32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; the bitmaps are cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) const PARALLEL_REFRESH_MIN_BYTES: usize = 512 << 10;
pub(crate) const PARALLEL_COPY_BACK_MIN_BYTES: usize = 64 << 10;

fn parallel_copies_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

pub(crate) fn parallel_copies_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parallel_copies_enabled_value(std::env::var_os("VICTORY_PARALLEL_COPIES").as_deref()))
}

/// Whether `(start, len)` ranges are pairwise disjoint.
#[cfg(test)]
pub(crate) fn ranges_disjoint(ranges: &[(u64, u64)]) -> bool {
    spans_disjoint(
        ranges
            .iter()
            .map(|&(start, len)| (start, start.saturating_add(len)))
            .collect(),
    )
}

fn spans_disjoint(mut spans: Vec<(u64, u64)>) -> bool {
    spans.sort_unstable();
    spans.windows(2).all(|w| w[0].1 <= w[1].0)
}

/// Copy-back workers must own different watch pages. Distinct virtual pages
/// may still share backing memory, so ranges with a reported alias stay serial.
pub(crate) fn copy_back_ranges_disjoint(ranges: &[(u64, u64)]) -> bool {
    let page = PAGE as u64;
    let spans = ranges
        .iter()
        .filter(|&&(_, len)| len != 0)
        .map(|&(start, len)| (start / page, start.saturating_add(len).div_ceil(page)));
    spans_disjoint(spans.collect())
        && !ranges.iter().any(|&(start, len)| {
            len != 0 && crate::watch::aliased(start as usize, len as usize)
        })
}

#[cfg(test)]
thread_local! {
    static MARKED_COPY_BACK_PARALLEL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// mac: whether one copy's marked pages go back on several threads
/// (`VICTORY_COPY_BACK_SPLIT=1`, off by default): the frame's big compute
/// submission marks some 2,000 pages of one copy, which went back on one
/// thread in 50 to 60 ms a frame ("marked pages: 2033 compared, 97.4 MB
/// written back in 50.5 ms", diagnostics run of 26 November 23:29), with
/// the next submission waiting. Not for a copy whose memory is shown at
/// two addresses (a page written through one could be read through the
/// other by another thread), nor one with widened readers.
fn copy_back_split() -> bool {
    #[cfg(test)]
    if let Some(on) = SPLIT_FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_COPY_BACK_SPLIT");
        if on {
            tracing::info!(
                "VICTORY_COPY_BACK_SPLIT: the marked pages of one copy go back on several threads ({SPLIT_PARTS} parts at most)"
            );
        }
        on
    })
}

/// The first split, and the first copy left to one thread as its memory
/// is shown at two addresses, each said once (whether the split applies to
/// the frame's big copy shows in the log of an A/B run).
fn split_said(split: bool, pages: usize) {
    static SAID: [std::sync::Once; 2] = [const { std::sync::Once::new() }; 2];
    SAID[usize::from(split)].call_once(|| {
        if split {
            tracing::info!("VICTORY_COPY_BACK_SPLIT: first split, {pages} marked pages of one copy");
        } else {
            tracing::info!(
                "VICTORY_COPY_BACK_SPLIT: {pages} marked pages of a copy whose memory is shown at two addresses go back on one thread (said once)"
            );
        }
    });
}

/// A copy's marked pages are split from this many on, in parts of half as many at least.
const SPLIT_MIN_PAGES: usize = 64;
/// Into this many parts at most (the texture workers take them).
const SPLIT_PARTS: usize = 8;

#[cfg(test)]
thread_local! {
    /// A test's choice, whatever the environment says.
    static SPLIT_FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static MARKED_SPLIT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        // Marks of offsets past the bitmap land in its last bit, no copy's.
        bits[words - 1] = 0;
        let used = (marks().end as usize).div_ceil(32).min(words);
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            let into = if memory && memory_withheld { &mut held } else { &mut marked };
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    // Per copy, in page order, so its pages go back together.
    let mut by_copy: Vec<(Arc<Shadow>, Vec<usize>)> = Vec::new();
    for (copy, page) in pages {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n)
        {
            continue;
        }
        match by_copy.iter_mut().find(|(c, _)| Arc::ptr_eq(c, &copy)) {
            Some((_, list)) => list.push(page),
            None => by_copy.push((copy, vec![page])),
        }
    }
    for (_, list) in &mut by_copy {
        list.sort_unstable();
        list.dedup();
    }
    let total_bytes: usize = by_copy.iter().map(|(_, list)| list.len() * PAGE).sum();
    let shadow_ranges: Vec<(u64, u64)> = by_copy.iter().map(|(copy, _)| copy.range()).collect();
    let parallel = parallel_copies_enabled()
        && by_copy.len() >= 2
        && total_bytes >= PARALLEL_COPY_BACK_MIN_BYTES
        && copy_back_ranges_disjoint(&shadow_ranges);
    #[cfg(test)]
    MARKED_COPY_BACK_PARALLEL.with(|taken| taken.set(Some(parallel)));
    if parallel {
        let total_pages: usize = by_copy.iter().map(|(_, list)| list.len()).sum();
        let total_written = std::sync::atomic::AtomicUsize::new(0);
        crate::texture::in_parallel(by_copy.len(), |i| {
            let (copy, list) = &by_copy[i];
            let written = copy.copy_back_marked(list);
            total_written.fetch_add(written, Ordering::Relaxed);
        });
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, total_pages as u64);
            crate::copystat::add(
                crate::copystat::C::MarkedBytes,
                total_written.into_inner() as u64,
            );
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (total_pages, total_written);
        return;
    }
    for (copy, list) in by_copy {
        let written = copy.copy_back_marked(&list);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, list.len() as u64);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
/// Read by every draw's views (`find`) on the title's threads, at once.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

/// Epoch for [`find`]'s per-thread lookup cache; bumped on every change to
/// `BUFFERS`, `ALIASED`, or `LOOSE` (while holding that table's write lock).
static LOOKUP_EPOCH: AtomicU64 = AtomicU64::new(1);

fn lookup_changed() {
    LOOKUP_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Release);
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
    lookup_changed();
}

/// The longest title buffer registered so far (it stays after the buffer is
/// released): a buffer starting further below an address than this cannot
/// hold it. Changed under `BUFFERS`' lock.
static MAX_SPAN: AtomicU64 = AtomicU64::new(0);

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers();
    MAX_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert(
        (start, resource),
        Entry {
            end: start.saturating_add(len),
            shadow: None,
        },
    );
    lookup_changed();
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let mut map = buffers();
    if let Some(entry) = map.remove(&(start, resource)) {
        if entry.shadow.is_some() {
            changed();
        } else {
            lookup_changed();
        }
    }
}

/// The innermost entry holding `[address, end)`: the one starting closest
/// below `address` that reaches `end`, among those starting at most `span`
/// (the longest entry's length) below `end`. Big buffers hold many smaller
/// ones (a heap's placed buffers), so the one holding an address past
/// those can be any number of entries back.
///
/// The cost is the entries starting within `span` below `end`, which a
/// single very long buffer makes many: the lookup walks back until one
/// reaches `end`, where the old search stopped after 64 entries. It runs
/// under the buffers' lock on every lookup.
fn enclosing(map: &BTreeMap<(u64, usize), Entry>, span: u64, address: u64, end: u64) -> Option<(u64, usize)> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    map.range((lowest, 0)..=(address, usize::MAX))
        .rev()
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// Zero-copy title regions aliased onto host buffers (`crate::alias`), by start address.
static ALIASED: RwLock<BTreeMap<u64, Arc<Shadow>>> = RwLock::new(BTreeMap::new());
static ALIASED_SPAN: AtomicU64 = AtomicU64::new(0);
static ALIASED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn has_aliased_overlap(start: u64, end: u64) -> bool {
    if ALIASED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let map = ALIASED.read().unwrap_or_else(|e| e.into_inner());
    let span = ALIASED_SPAN.load(Ordering::Relaxed);
    let lowest = start.saturating_sub(span);
    if map
        .range(lowest..=start)
        .rev()
        .any(|(&s, sh)| s + sh.len > start)
    {
        return true;
    }
    map.range(start..end).next().is_some()
}

pub(crate) fn register_aliased(shadow: Arc<Shadow>) {
    ALIASED_SPAN.fetch_max(shadow.len, Ordering::Relaxed);
    let mut map = ALIASED.write().unwrap_or_else(|e| e.into_inner());
    if map.insert(shadow.start, shadow).is_none() {
        ALIASED_COUNT.fetch_add(1, Ordering::Release);
    }
    changed();
}

pub(crate) fn unregister_aliased(start: u64) -> bool {
    let mut map = ALIASED.write().unwrap_or_else(|e| e.into_inner());
    let removed = map.remove(&start).is_some();
    if removed {
        ALIASED_COUNT.fetch_sub(1, Ordering::Release);
        changed();
    }
    removed
}

/// The aliased regions `(start, end)` overlapping `[start, end)`, in order.
pub(crate) fn aliased_between(start: u64, end: u64) -> Vec<(u64, u64)> {
    if ALIASED_COUNT.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let map = ALIASED.read().unwrap_or_else(|e| e.into_inner());
    let lowest = start.saturating_sub(ALIASED_SPAN.load(Ordering::Relaxed));
    map.range(lowest..end)
        .map(|(&s, sh)| (s, s + sh.len))
        .filter(|&(_, e)| e > start)
        .collect()
}

/// The aliased regions now: how many, and their bytes.
pub(crate) fn aliased_totals() -> (usize, u64) {
    if ALIASED_COUNT.load(Ordering::Acquire) == 0 {
        return (0, 0);
    }
    let map = ALIASED.read().unwrap_or_else(|e| e.into_inner());
    (map.len(), map.values().map(|s| s.len).sum())
}

/// The aliased region holding all of `[address, end)`.
pub(crate) fn find_aliased(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if ALIASED_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let map = ALIASED.read().unwrap_or_else(|e| e.into_inner());
    if map.is_empty() {
        return None;
    }
    let span = ALIASED_SPAN.load(Ordering::Relaxed);
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    map.range(lowest..=address)
        .rev()
        .find(|(_, s)| s.start + s.len >= end)
        .map(|(_, s)| s.clone())
}

/// Number of sets in the per-thread 2-way lookup cache (power of two).
const LOOKUP_SETS: usize = 2048;

#[derive(Default)]
struct LookupSlot {
    address: u64,
    end: u64,
    shadow: std::sync::Weak<Shadow>,
}

struct LookupCache {
    epoch: u64,
    dirty: bool,
    slots: Box<[[LookupSlot; 2]]>,
}

impl LookupCache {
    fn new(epoch: u64) -> Self {
        let slots = std::iter::repeat_with(|| [LookupSlot::default(), LookupSlot::default()])
            .take(LOOKUP_SETS)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            epoch,
            dirty: false,
            slots,
        }
    }

    fn clear_if_stale(&mut self, epoch: u64) {
        if self.epoch != epoch {
            if self.dirty {
                for pair in &mut self.slots {
                    *pair = [LookupSlot::default(), LookupSlot::default()];
                }
                self.dirty = false;
            }
            self.epoch = epoch;
        }
    }

    fn get(&mut self, epoch: u64, address: u64, end: u64) -> Option<Arc<Shadow>> {
        if self.epoch != epoch {
            self.clear_if_stale(epoch);
            return None;
        }
        let pair = &mut self.slots[lookup_set(address, end)];
        for way in 0..2 {
            if pair[way].address == address && pair[way].end == end {
                if let Some(shadow) = pair[way].shadow.upgrade() {
                    if way == 1 {
                        pair.swap(0, 1);
                    }
                    return Some(shadow);
                }
                pair[way] = LookupSlot::default();
            }
        }
        None
    }

    fn insert(&mut self, epoch: u64, address: u64, end: u64, shadow: &Arc<Shadow>) {
        self.clear_if_stale(epoch);
        let pair = &mut self.slots[lookup_set(address, end)];
        let entry = LookupSlot {
            address,
            end,
            shadow: Arc::downgrade(shadow),
        };
        if pair[0].address == address && pair[0].end == end {
            pair[0] = entry;
        } else {
            pair[1] = std::mem::replace(&mut pair[0], entry);
        }
        self.dirty = true;
    }
}

fn lookup_set(address: u64, end: u64) -> usize {
    let mut h = address ^ end.rotate_left(29);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h as usize) & (LOOKUP_SETS - 1)
}

fn lookup_cache_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn lookup_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| lookup_cache_enabled_value(std::env::var_os("VICTORY_LOOKUP_CACHE").as_deref()))
}

std::thread_local! {
    static LOOKUP_CACHE: std::cell::RefCell<Option<LookupCache>> = const { std::cell::RefCell::new(None) };
}

/// Hits and misses of [`find`]'s per-thread lookup cache (diagnostics).
#[cfg(feature = "diagnostics")]
static LOOKUP_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Logs [`LOOKUP_STATS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_lookups(frames: f64) {
    let hits = LOOKUP_STATS[0].swap(0, Ordering::Relaxed);
    let misses = LOOKUP_STATS[1].swap(0, Ordering::Relaxed);
    let total = hits + misses;
    if total > 0 {
        tracing::info!(
            "shadow lookup cache: {:.0} hits/frame, {:.0} misses/frame ({:.1}% hit)",
            hits as f64 / frames,
            misses as f64 / frames,
            hits as f64 * 100.0 / total as f64
        );
    }
}

/// Logs the host buffers the copies of title memory hold, once per report:
/// one per title buffer and one per window of loose memory, and the
/// zero-copy aliases, whose host buffers are the title's own memory
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_held() {
    // SAFETY: live buffers.
    let size = |s: &Arc<Shadow>| unsafe { s.buffer.GetDesc() }.Width;
    let held = |sizes: Vec<u64>| (sizes.len(), sizes.iter().sum::<u64>() as f64 / 1e6);
    let buffers = held(
        buffers_read()
            .values()
            .filter_map(|e| e.shadow.as_ref().map(size))
            .collect(),
    );
    let loose = held(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(size)
            .collect(),
    );
    let aliased = held(
        ALIASED
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(size)
            .collect(),
    );
    tracing::info!(
        "host buffer copies held: {:.0} MB in all: title buffers {} ({:.0} MB), loose windows {} ({:.0} MB); \
         aliased regions {} ({:.0} MB, the title's own memory, not extra)",
        buffers.1 + loose.1,
        buffers.0,
        buffers.1,
        loose.0,
        loose.1,
        aliased.0,
        aliased.1,
    );
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    if !lookup_cache_enabled() {
        let shadow = find_uncached(address, end)?;
        shadow.touch();
        return Some(shadow);
    }
    let epoch = LOOKUP_EPOCH.load(Ordering::Acquire);
    let hit = LOOKUP_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard.as_mut().and_then(|cache| cache.get(epoch, address, end))
    });
    if let Some(shadow) = hit {
        #[cfg(feature = "diagnostics")]
        LOOKUP_STATS[0].fetch_add(1, Ordering::Relaxed);
        shadow.touch();
        return Some(shadow);
    }
    #[cfg(feature = "diagnostics")]
    LOOKUP_STATS[1].fetch_add(1, Ordering::Relaxed);
    let shadow = find_uncached(address, end)?;
    shadow.touch();
    LOOKUP_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let cache = guard.get_or_insert_with(|| LookupCache::new(epoch));
        cache.insert(epoch, address, end, &shadow);
    });
    Some(shadow)
}

fn find_uncached(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if let Some(shadow) = find_aliased(address, end) {
        return Some(shadow);
    }
    {
        let map = buffers_read();
        let Some(key) = enclosing(&map, MAX_SPAN.load(Ordering::Relaxed), address, end) else {
            drop(map);
            return loose(address, end);
        };
        if let Some(shadow) = map.get(&key).and_then(|entry| entry.shadow.clone()) {
            return Some(shadow);
        }
    }
    let mut map = buffers();
    let key = enclosing(&map, MAX_SPAN.load(Ordering::Relaxed), address, end);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len));
        released().remade(key.0);
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches. Smaller windows retain
/// fewer unused bytes and require more host buffers and page-table entries.
fn loose_window_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        if crate::switch::on("VICTORY_SMALL_LOOSE_WINDOWS") {
            256 << 10
        } else {
            4 << 20
        }
    })
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): 4 MiB of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let window = loose_window_bytes();
    let start = address & !(window - 1);
    let len = end.checked_next_multiple_of(window)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    released().remade(start);
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice));
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers, aliases, and windows).
/// Presented frames a copy goes unasked for before [`release_idle`] may
/// give it up.
const IDLE_FRAMES: u64 = 60;

/// Title addresses of copies [`release_idle`] gave up, to count those made
/// again (bounded: cleared when it grows past [`Released::CAP`]).
#[derive(Default)]
struct Released {
    starts: std::collections::HashSet<u64>,
    remade: u64,
}

impl Released {
    const CAP: usize = 1 << 16;

    fn remade(&mut self, start: u64) {
        if self.starts.remove(&start) {
            self.remade += 1;
        }
    }
}

static RELEASED: std::sync::LazyLock<Mutex<Released>> = std::sync::LazyLock::new(Mutex::default);

fn released() -> std::sync::MutexGuard<'static, Released> {
    RELEASED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether [`release_idle`] runs: `VICTORY_RELEASE_COPIES=0` turns it off.
fn release_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELEASE_COPIES").is_none_or(|v| v != "0"))
}

/// Whether `copy`, one of the copies of `[start, start + len)`, can go:
/// nothing but the table that found it and the page tables' own layouts
/// (`tables` of them) holds it, no submission's writes to it are still to
/// go back, and no view or list asked for it in [`IDLE_FRAMES`] frames.
/// A list that holds it, a heap slot made from it, a widened view of it or
/// a list still reading an older layout all hold it too, so it stays.
fn idle(copy: &Arc<Shadow>, tables: usize, frame: u64) -> bool {
    !copy.aliased
        && Arc::strong_count(copy) <= 1 + tables
        && frame.saturating_sub(copy.used.load(Ordering::Relaxed)) >= IDLE_FRAMES
        && copy.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// While the computer's memory runs low: gives up copies of title memory
/// no view or list asked for lately, the longest unasked first, until about
/// `want` bytes went (all such copies if fewer). A copy is made again, from
/// title memory, the next time a view or a list asks for it; until then
/// page tables stop naming it (the generation changes), and its memory goes
/// once the last layout naming it is rewritten. Host copies of title memory
/// grew to 7.6 GB in 20 minutes of the world on a 16 GB Mac, and none ever
/// went while the title kept its buffers. Returns the copies and bytes.
pub(crate) fn release_idle(frame: u64, want: u64) -> (usize, u64) {
    if want == 0 || !release_enabled() {
        return (0, 0);
    }
    let tables = crate::memory::tables();
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = buffers();
        let mut loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        /// Where a candidate is kept: a title buffer's entry or a loose window.
        enum Held {
            Buffer((u64, usize)),
            Loose((u64, u64)),
        }
        let mut candidates: Vec<(u64, u64, Held)> = map
            .iter()
            .filter_map(|(&key, e)| e.shadow.as_ref().map(|c| (Held::Buffer(key), c)))
            .chain(loose.iter().map(|(&key, c)| (Held::Loose(key), c)))
            .filter(|(_, c)| idle(c, tables, frame))
            .map(|(held, c)| (c.used.load(Ordering::Relaxed), c.len, held))
            .collect();
        candidates.sort_unstable_by_key(|&(used, ..)| used);
        let mut taken = 0;
        for (_, len, held) in candidates {
            if taken >= want {
                break;
            }
            let copy = match held {
                Held::Loose(key) => loose.remove(&key),
                Held::Buffer(key) => map.get_mut(&key).and_then(|e| e.shadow.take()),
            };
            if let Some(copy) = copy {
                taken += len;
                gone.push(copy);
            }
        }
        if !gone.is_empty() {
            changed();
        }
    }
    let (count, bytes) = (gone.len(), gone.iter().map(|c| c.len).sum::<u64>());
    if !gone.is_empty() {
        let mut r = released();
        if r.starts.len() > Released::CAP {
            r.starts.clear();
        }
        r.starts.extend(gone.iter().map(|c| c.start));
        tracing::info!(
            copies = gone.len(),
            mb = bytes / (1 << 20),
            made_again = r.remade,
            "host copies of title memory given up while memory is low"
        );
    }
    // The copies' buffers go with the last layout naming them, on the
    // thread that rewrites it; these references go now.
    drop(gone);
    (count, bytes)
}

pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = ALIASED
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .cloned()
        .collect();
    all.extend(buffers_read().values().filter_map(|e| e.shadow.clone()));
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = ALIASED
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .map(|s| (s.gpu, s.start, s.len))
        .collect();
    all.extend(
        buffers_read()
            .values()
            .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len))),
    );
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod parallel_copy_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// After a submission the host GPU ended, every lock word of the
    /// emulated 64-bit atomics is free again, and the written pages and read
    /// marks before them and the give-up count after them stay as they were
    /// (#240). The count is taken once.
    #[test]
    fn freeing_atomic_locks_keeps_pages_and_read_marks() {
        let dwords = (victory_shader::emit::VIEW_WRITES_BYTES / 4) as usize;
        let first = (victory_shader::emit::ATOMIC_LOCKS_AT / 4) as usize;
        let end = first + victory_shader::emit::ATOMIC_LOCKS as usize;
        let mut words = vec![0xA5A5_A5A5u32; dwords];
        assert_eq!(
            free_locks(&mut words),
            victory_shader::emit::ATOMIC_LOCKS as usize,
            "every lock word counted as taken"
        );
        assert!(
            words[..first].iter().all(|&w| w == 0xA5A5_A5A5),
            "pages and read marks kept"
        );
        assert!(words[first..end].iter().all(|&w| w == 0), "every lock free");
        assert_eq!(words[end..], [0xA5A5_A5A5], "the give-up count kept");
        words[first + 7] = 1;
        assert_eq!(free_locks(&mut words), 1, "one lock left taken");
        assert_eq!(free_locks(&mut words), 0, "none the second time");
        assert_eq!(take_count(&mut words), 0xA5A5_A5A5, "the give-up count");
        assert_eq!(take_count(&mut words), 0, "taken once");
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A copy may go only when nothing else holds it, no submission's writes
    /// to it wait to go back, and no view or list asked for it lately. (The
    /// local reference here stands in for one page table's layout.)
    #[test]
    fn only_an_unheld_settled_unasked_copy_is_idle() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let asked = copy.used.load(Ordering::Relaxed);
        let later = asked + IDLE_FRAMES;
        assert!(!idle(&copy, 1, later - 1), "asked for lately");
        assert!(idle(&copy, 1, later), "unasked, unheld and settled");
        assert!(!idle(&copy, 0, later), "held by more than the tables");
        let held = copy.clone();
        assert!(!idle(&copy, 1, later), "a list holds it");
        drop(held);
        copy.writing(p, PAGE, false);
        assert!(!idle(&copy, 1, later), "a submission's writes are still to go back");
        copy.copy_back(p, PAGE);
        assert!(idle(&copy, 1, later), "settled again");
        copy.used.store(later, Ordering::Relaxed);
        assert!(!idle(&copy, 1, later), "asked for again");
        unregister(p, p as u64);
    }

    /// The same where title memory is not tracked (what Rosetta gets for
    /// all of it): the title's constant under the busy range still reaches
    /// the copy, and copy-back still tells the shader's write from it.
    #[test]
    fn untracked_title_writes_under_a_busy_range_are_copied() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        let len = 2 * PAGE;
        // Private memory without write watching is not the runtime's.
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } as usize;
        assert!(p != 0);
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        assert!(copy.tracked.is_none(), "plain private memory is not tracked");
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our allocation.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert_eq!(copy.update(p + PAGE + 16, 16), 16, "the busy range is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        copy.write_host((p + 10) as u64, &[0xaa]);
        assert_eq!(copy.copy_back(p, len), PAGE, "the shader's page goes back");
        // SAFETY: bytes of our allocation.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x55, "the constant is not undone");
        }
        drop(copy);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// A busy range over a tracked pool page and the plain allocation after
    /// it still takes the title's changes on the plain page. The refresh is
    /// asked for both one changed dword and a view wider than the copy.
    #[test]
    fn untracked_title_writes_under_a_mixed_busy_range_are_copied() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE)
                        as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(
                            q as *const _,
                            PAGE,
                            MEM_RESERVE | MEM_COMMIT | kind,
                            PAGE_READWRITE,
                        ) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let tracked = copy.tracked.as_ref().expect("the pool page is tracked");
        assert_eq!(tracked.untracked(pool, pool + len), [(plain, plain + PAGE)]);
        assert!(!copy.clean(), "the plain page is always refreshed");
        let word = (plain + 64) as *mut u32;
        // SAFETY: (all title reads and writes below) bytes of our allocation.
        unsafe { word.write_volatile(0x1234_5678) };
        assert_eq!(
            copy.update(pool, len),
            len,
            "the initial contents reach the copy"
        );
        assert_eq!(copy.read_host(word as u64, 1), [0x1234_5678]);
        // An earlier list declares the whole mixed range as written by shaders.
        copy.writing(pool, len, false);
        // SAFETY: a dword of our plain allocation.
        unsafe { word.write_volatile(0x90ab_cdef) };
        assert_eq!(copy.update(word as usize, 4), 4, "the busy dword is copied");
        assert_eq!(
            copy.read_host(word as u64, 1),
            [0x90ab_cdef],
            "the title's dword reaches the copy"
        );
        // SAFETY: as above.
        unsafe { word.write_volatile(0xfedc_ba09) };
        assert_eq!(
            copy.update(pool - 16, len + 32),
            PAGE,
            "a widened busy view takes the plain page"
        );
        assert_eq!(
            copy.read_host(word as u64, 1),
            [0xfedc_ba09],
            "the later dword reaches the copy too"
        );
        assert_eq!(
            copy.copy_back(pool, len),
            0,
            "the refreshed copy holds no shader write"
        );
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Publishing a widened view cannot miss a source update that runs while
    /// its first copy is being made.
    #[test]
    fn widened_publication_serializes_with_source_updates() {
        if host().is_none() {
            return; // no host GPU here
        }
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            crate::watch::tests::free(p);
            return;
        };
        copy.update(p, len);

        let ready = Arc::new(std::sync::Barrier::new(3));
        let resume = Arc::new(std::sync::Barrier::new(2));
        WIDENED_AFTER_REFRESH
            .lock()
            .unwrap()
            .push((p as u64, (ready.clone(), resume.clone())));
        let started = Arc::new(AtomicBool::new(false));
        let making = copy.clone();
        let constructor = std::thread::spawn(move || making.widened(p as u64, len as u64, 1));
        let updating = copy.clone();
        let update_started = started.clone();
        let updater = {
            let ready = ready.clone();
            std::thread::spawn(move || {
                ready.wait();
                update_started.store(true, Ordering::Release);
                let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
                // SAFETY: one 12-byte element of our pool page.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        changed.as_ptr().cast::<u8>(),
                        (p + 12) as *mut u8,
                        12,
                    )
                };
                updating.update(p, len)
            })
        };
        ready.wait();
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        // Without the source lock the update finishes here; with it, the
        // updater waits until construction publishes and releases the lock.
        std::thread::sleep(std::time::Duration::from_millis(20));
        resume.wait();
        let widened = constructor.join().unwrap().expect("a widened view");
        assert!(updater.join().unwrap() > 0, "the source update runs");
        WIDENED_AFTER_REFRESH
            .lock()
            .unwrap()
            .retain(|(start, _)| *start != p as u64);
        assert_eq!(
            widened.read_host(16),
            [
                1000, 2000, 3000, 1, 0x1111_1111, 0x2222_2222, 0x3333_3333, 1, 1006, 2006, 3006,
                1, 1009, 2009, 3009, 1,
            ]
        );
        drop((widened, copy));
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A widened view follows GPU-written bytes when the computed-memory
    /// copy-back path puts the normal shadow's page into title memory.
    #[test]
    fn widened_views_follow_computed_copy_back_pages() {
        if host().is_none() {
            return; // no host GPU here
        }
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            crate::watch::tests::free(p);
            return;
        };
        copy.update(p, len);
        let widened = copy.widened(p as u64, len as u64, 1).expect("a widened view");
        let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
        // SAFETY: one 12-byte element of the mapped shadow, as a GPU write.
        unsafe {
            std::ptr::copy_nonoverlapping(changed.as_ptr().cast::<u8>(), copy.data.add(12), 12)
        };
        assert!(copy.copy_back_page(p) > 0, "the GPU-written page goes back");
        assert_eq!(
            widened.read_host(16),
            [
                1000, 2000, 3000, 1, 0x1111_1111, 0x2222_2222, 0x3333_3333, 1, 1006, 2006, 3006,
                1, 1009, 2009, 3009, 1,
            ]
        );
        let same = [0x4444_4444, 0x5555_5555, 0x6666_6666];
        // SAFETY: title memory and its mapped shadow already hold the same
        // GPU-written element, so copy-back takes its zero-write early return.
        unsafe {
            std::ptr::copy_nonoverlapping(same.as_ptr().cast::<u8>(), (p + 24) as *mut u8, 12);
            std::ptr::copy_nonoverlapping(same.as_ptr().cast::<u8>(), copy.data.add(24), 12);
        }
        assert_eq!(copy.copy_back_page(p), 0, "title memory already matches");
        assert_eq!(
            widened.read_host(16),
            [
                1000, 2000, 3000, 1, 0x1111_1111, 0x2222_2222, 0x3333_3333, 1, 0x4444_4444,
                0x5555_5555, 0x6666_6666, 1, 1009, 2009, 3009, 1,
            ]
        );
        drop((widened, copy));
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A widened view follows GPU-written bytes that stay in the normal
    /// shadow for the next submission instead of going back to title memory.
    #[test]
    fn widened_views_follow_kept_gpu_writes() {
        if host().is_none() {
            return; // no host GPU here
        }
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            crate::watch::tests::free(p);
            return;
        };
        copy.update(p, len);
        let widened = copy.widened(p as u64, len as u64, 1).expect("a widened view");
        copy.writing(p, len, false);
        let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
        // SAFETY: one 12-byte element of the mapped shadow, as a GPU write.
        unsafe {
            std::ptr::copy_nonoverlapping(changed.as_ptr().cast::<u8>(), copy.data.add(12), 12)
        };
        copy.keep_writes(p, len);
        assert_eq!(
            widened.read_host(16),
            [
                1000, 2000, 3000, 1, 0x1111_1111, 0x2222_2222, 0x3333_3333, 1, 1006, 2006, 3006,
                1, 1009, 2009, 3009, 1,
            ]
        );
        drop((widened, copy));
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A write through a distinct exact shadow patches the touched bytes of
    /// a widened element even when that shadow does not hold all 12 bytes.
    #[test]
    fn widened_views_patch_partial_elements_from_another_shadow() {
        if host().is_none() {
            return; // no host GPU here
        }
        let host = host().unwrap();
        let start = 0x1000u64;
        let (parent_buffer, parent_data) = host.shared_buffer(48).unwrap();
        // SAFETY: a live buffer.
        let parent_gpu = unsafe { parent_buffer.GetGPUVirtualAddress() };
        let parent = Shadow::new(parent_buffer, parent_data, parent_gpu, start, 48);
        // SAFETY: the parent shadow's 48 bytes.
        unsafe { std::ptr::write_bytes(parent_data, 0, 48) };

        let (writer_buffer, writer_data) = host.shared_buffer(16).unwrap();
        // SAFETY: a live buffer.
        let writer_gpu = unsafe { writer_buffer.GetGPUVirtualAddress() };
        let writer = Shadow::new(writer_buffer, writer_data, writer_gpu, start + 9, 15);
        let changed: [u8; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // SAFETY: the exact shadow's bytes, standing in for an unaligned UAV write.
        unsafe { std::ptr::copy_nonoverlapping(changed.as_ptr(), writer_data, 15) };
        let (overlap_buffer, overlap_data) = host.shared_buffer(12).unwrap();
        // SAFETY: a live buffer.
        let overlap_gpu = unsafe { overlap_buffer.GetGPUVirtualAddress() };
        let overlap = Shadow::new(overlap_buffer, overlap_data, overlap_gpu, start + 10, 10);
        let overlaid: [u8; 10] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110];
        // SAFETY: the second exact shadow's bytes.
        unsafe { std::ptr::copy_nonoverlapping(overlaid.as_ptr(), overlap_data, 10) };

        let widened = parent.widened(start, 48, 1).expect("a widened view");
        let list = host.new_list().unwrap();
        let writes = [
            (writer, start as usize + 9, 15),
            (overlap, start as usize + 10, 10),
        ];
        assert!(crate::widened::sync(
            &list,
            &writes,
            0,
            &std::collections::HashSet::new(),
        )
        .is_some());
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            widened.read_host(8),
            [0, 0, 0x6665_0100, 1, 0x6a69_6867, 0x6e6d_6c6b, 0x0f0e_0d0c, 1]
        );

    }

    /// Partial patches through different exact shadows compose across
    /// barriers instead of resetting earlier widened-only bytes.
    #[test]
    fn widened_views_compose_patches_across_barriers() {
        use windows::core::Interface;

        if host().is_none() {
            return;
        }
        let host = host().unwrap();
        let start = 0x1000u64;
        let (parent_buffer, parent_data) = host.shared_buffer(12).unwrap();
        // SAFETY: a live buffer.
        let parent_gpu = unsafe { parent_buffer.GetGPUVirtualAddress() };
        let parent = Shadow::new(parent_buffer, parent_data, parent_gpu, start, 12);
        // SAFETY: the parent shadow's 12 bytes.
        unsafe { std::ptr::write_bytes(parent_data, 0, 12) };
        let widened = parent.widened(start, 12, 1).expect("a widened view");

        let (first_buffer, first_data) = host.shared_buffer(8).unwrap();
        // SAFETY: a live buffer.
        let first_gpu = unsafe { first_buffer.GetGPUVirtualAddress() };
        let first = Shadow::new(first_buffer, first_data, first_gpu, start + 2, 4);
        let first_bytes = [1u8, 2, 3, 4];
        // SAFETY: the first exact shadow's four bytes.
        unsafe { std::ptr::copy_nonoverlapping(first_bytes.as_ptr(), first_data, 4) };
        let list = host.new_list().unwrap();
        let writes = [(first.clone(), start as usize + 2, 4)];
        assert!(crate::widened::sync(
            &list,
            &writes,
            0,
            &std::collections::HashSet::new(),
        )
        .is_some());
        list.close();
        host.execute(&[&list]);
        assert_eq!(widened.read_host(4), [0x0201_0000, 0x0000_0403, 0, 1]);

        let (second_buffer, second_data) = host.shared_buffer(8).unwrap();
        // SAFETY: a live buffer.
        let second_gpu = unsafe { second_buffer.GetGPUVirtualAddress() };
        let second = Shadow::new(second_buffer, second_data, second_gpu, start + 6, 4);
        let second_bytes = [5u8, 6, 7, 8];
        // SAFETY: the second exact shadow's four bytes.
        unsafe { std::ptr::copy_nonoverlapping(second_bytes.as_ptr(), second_data, 4) };
        let list = host.new_list().unwrap();
        let writes = [
            (first, start as usize + 2, 4),
            (second, start as usize + 6, 4),
        ];
        let ready = std::collections::HashSet::from([widened.buffer.as_raw() as usize]);
        assert!(crate::widened::sync(&list, &writes, 1, &ready).is_some());
        list.close();
        host.execute(&[&list]);
        assert_eq!(widened.read_host(4), [0x0201_0000, 0x0605_0403, 0x0000_0807, 1]);
    }

    /// An element-aligned shadow that starts after the widened view cannot
    /// back the direct source root SRV; it must use the composed patch path.
    #[test]
    fn widened_views_patch_an_aligned_later_shadow() {
        if host().is_none() {
            return;
        }
        let host = host().unwrap();
        let start = 0x1000u64;
        let (parent_buffer, parent_data) = host.shared_buffer(24).unwrap();
        // SAFETY: a live buffer.
        let parent_gpu = unsafe { parent_buffer.GetGPUVirtualAddress() };
        let parent = Shadow::new(parent_buffer, parent_data, parent_gpu, start, 24);
        // SAFETY: the parent shadow's 24 bytes.
        unsafe { std::ptr::write_bytes(parent_data, 0, 24) };
        let widened = parent.widened(start, 24, 1).expect("a widened view");

        let (later_buffer, later_data) = host.shared_buffer(12).unwrap();
        // SAFETY: a live buffer.
        let later_gpu = unsafe { later_buffer.GetGPUVirtualAddress() };
        let later = Shadow::new(later_buffer, later_data, later_gpu, start + 12, 12);
        let changed: [u8; 12] = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        // SAFETY: the later shadow's one element.
        unsafe { std::ptr::copy_nonoverlapping(changed.as_ptr(), later_data, 12) };
        let list = host.new_list().unwrap();
        let writes = [(later, start as usize + 12, 12)];
        assert!(crate::widened::sync(
            &list,
            &writes,
            0,
            &std::collections::HashSet::new(),
        )
        .is_some());
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            widened.read_host(8),
            [0, 0, 0, 1, 0x1817_1615, 0x1c1b_1a19, 0x201f_1e1d, 1]
        );
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1. The bitmap is shared: held as a
        // submission holds it, or another test's copy-back (or submission)
        // can take these bits and put the pages back after the reads below.
        let quiet = host().and_then(|h| h.quiet());
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        drop(quiet);
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    #[test]
    fn marked_writeback_copies_adjacent_pages_and_keeps_gaps_and_edges() {
        let Some(host) = host() else {
            return;
        };
        let base = crate::watch::tests::pool(4);
        let start = base + 32;
        let end = base + 4 * PAGE - 32;
        let len = end - start;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: a live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, start as u64, len as u64);
        assert!(copy.marked(), "the title allocation is tracked");
        // SAFETY: a private allocation of four pages, untouched by the GPU.
        unsafe { (base as *mut u8).write_bytes(0x11, 4 * PAGE) };
        copy.update(start, len);
        let changed = [(start, 0xaa), (base + PAGE, 0xbb), (end - 1, 0xcc)];
        for (at, byte) in changed {
            copy.write_host(at as u64, &[byte]);
        }

        let pages = [base, base + PAGE, base + 2 * PAGE, base + 3 * PAGE];
        assert_eq!(copy.copy_back_marked(&pages), len - PAGE);
        // SAFETY: all sampled addresses lie inside our allocation.
        let read = |at: usize| unsafe { (at as *const u8).read_volatile() };
        for (at, byte) in changed {
            assert_eq!(read(at), byte);
        }
        for at in [start - 1, base + 2 * PAGE, end] {
            assert_eq!(read(at), 0x11);
        }
        assert_eq!(copy.copy_back_marked(&pages), 0);
        drop(copy);
        crate::watch::tests::free(base);
    }

    /// mac: `VICTORY_COPY_BACK_SPLIT`: a long list of one copy's marked
    /// pages, looked at and written on several threads, goes back as on
    /// one: the changed byte of each page it changed, nothing of the
    /// others, and then nothing more (the pages equal title memory).
    #[test]
    fn a_long_list_of_marked_pages_goes_back_the_same_split() {
        let Some(host) = host() else {
            return;
        };
        let n = 2 * SPLIT_MIN_PAGES + 5;
        let mut written = Vec::new();
        for split in [false, true] {
            let base = crate::watch::tests::pool(n);
            let len = n * PAGE;
            let (buffer, data) = host.shared_buffer(len as u64).unwrap();
            // SAFETY: a live D3D12 buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);
            // SAFETY: a private allocation of `n` pages, untouched by the GPU.
            unsafe { (base as *mut u8).write_bytes(0x11, len) };
            copy.update(base, len);
            // The shaders change a byte in two pages of every three.
            let changed: Vec<(usize, u8)> = (0..n)
                .filter(|i| i % 3 != 2)
                .map(|i| (base + i * PAGE + (i * 37) % PAGE, 0x80 | (i % 127) as u8))
                .collect();
            for &(at, byte) in &changed {
                copy.write_host(at as u64, &[byte]);
            }
            let pages: Vec<usize> = (0..n).map(|i| base + i * PAGE).collect();
            SPLIT_FORCED.with(|f| f.set(Some(split)));
            let back = copy.copy_back_marked(&pages);
            let was = MARKED_SPLIT.with(std::cell::Cell::get);
            SPLIT_FORCED.with(|f| f.set(None));
            assert_eq!(was, Some(split), "split: {split}");
            assert_eq!(back, changed.len() * PAGE, "split: {split}");
            // SAFETY: all sampled addresses lie inside our allocation.
            let read = |at: usize| unsafe { (at as *const u8).read_volatile() };
            for &(at, byte) in &changed {
                assert_eq!(read(at), byte, "split: {split}, at {:#x}", at - base);
            }
            for i in (0..n).filter(|i| i % 3 == 2) {
                assert_eq!(read(base + i * PAGE + 5), 0x11, "split: {split}, page {i} unchanged");
            }
            assert_eq!(copy.copy_back_marked(&pages), 0, "split: {split}, nothing more");
            drop(copy);
            crate::watch::tests::free(base);
            written.push(back);
        }
        assert_eq!(written[0], written[1]);
    }

    /// While a submission runs on the host GPU, a copy a submitted list's
    /// shaders write is left alone and an idle one takes the title's
    /// writes early, so that the next refresh finds nothing to do. A page
    /// taken early that the running shaders marked all the same keeps the
    /// title's bytes, and the copy takes them again.
    #[test]
    fn idle_copies_take_the_titles_writes_early() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let title = |at: usize| (p + at) as *mut u8;
        let in_copy = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        // As under the submission lock: a submission finished meanwhile
        // (another test's) forgets what was taken early, and the bitmap
        // is shared.
        let quiet = host().and_then(|h| h.quiet());
        copy.update(p, len);
        // A submitted list declares it as written by shaders: left alone.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(20).write_volatile(0x55) };
        assert!(!copy.idle());
        assert_eq!(copy.update_early(p, len), None, "busy: not touched");
        assert_ne!(in_copy(20), 0x55);
        assert_eq!(copy.copy_back(p, len), 0, "the shaders wrote nothing");
        // Idle now: the title's write is taken early, once.
        assert!(copy.idle());
        assert!(copy.update_early(p, len).is_some_and(|n| n > 0), "the title's page is copied");
        assert_eq!(in_copy(20), 0x55);
        assert_eq!(copy.update_early(p, len), Some(0));
        assert_eq!(copy.update(p, len), 0, "the refresh proper finds it done");
        // The running shaders write the page all the same, and mark it.
        copy.write_host((p + 10) as u64, &[0xaa]);
        let bits = b.data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);
        // SAFETY: a byte of our pool.
        assert_eq!(unsafe { title(10).read_volatile() }, 0, "the title's bytes are kept");
        assert!(!copy.clean(), "the copy takes the page again");
        forget_early();
        drop(quiet);
        assert!(copy.update(p, len) > 0);
        assert_eq!(in_copy(10), 0, "the copy holds the title's bytes again");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        // Held as a submission holds the bitmaps (as in the test above).
        let quiet = host().and_then(|h| h.quiet());
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        drop(quiet);
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(&[p]);
        copy.copy_back_marked(&[p + PAGE]);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: None,
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), entry(base + i * 0x10000, 0x10000));
        }
        map
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        let span = 256 * MIB;
        // Past all 200 children (and the 64 entries looked at before).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        let first_64 = map.range(..=(at, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(enclosing(&map, span, at, at + 8), Some((base + 150 * 0x10000, 250)));
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(enclosing(&map, span, at, at + 1), None);
        assert_eq!(enclosing(&map, span, base - 16, base + 16), None);
        // Longer than any buffer: none, without looking.
        assert_eq!(enclosing(&map, span, base, base + span + 1), None);
        assert_eq!(enclosing(&map, 0, base, base + 1), None);
    }

    #[test]
    fn registered_buffers_widen_the_search() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        assert!(MAX_SPAN.load(Ordering::Relaxed) >= len);
        let at = base + len - 64;
        let found = enclosing(&buffers(), MAX_SPAN.load(Ordering::Relaxed), at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        assert!(buffers().range((base, 0)..=(base + len, usize::MAX)).next().is_none());
    }

    #[test]
    fn loose_windows_are_reused_for_ranges_inside_them() {
        let w = loose_window_bytes();
        let base = 0x20_0000_0000;
        let mut windows = BTreeMap::new();
        windows.insert((base, w), "first");
        // A range across the next two windows got one of its own.
        windows.insert((base + w, 2 * w), "spanning");
        let span = 2 * w;
        let find = |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
        assert_eq!(find(base, 1), Some("first"));
        assert_eq!(find(base + w - 16, 16), Some("first"));
        // Inside the spanning window, also in its second half.
        assert_eq!(find(base + w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 2 * w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 3 * w - 8, 8), Some("spanning"));
        // Across the first window's end: none holds it whole.
        assert_eq!(find(base + w - 8, 16), None);
        // Past every window, before them, longer than any.
        assert_eq!(find(base + 3 * w, 1), None);
        assert_eq!(find(base - 8, 4), None);
        assert_eq!(find(base, 3 * w), None);
    }

    /// A range in one window, then a range across that window's end, which
    /// gets a window of its own over both (the first inside it): what the
    /// title writes reaches both copies, and what the GPU writes through
    /// the spanning one goes back to title memory and from there into the
    /// first at its next refresh. (Within one submission the two copies are
    /// apart, as any two copies of the same bytes are.)
    #[test]
    fn overlapping_windows_agree_through_title_memory() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let Some(h) = host() else {
            return;
        };
        // Held as a submission holds it (as in the mark tests above): writing
        // and copy_back share the bitmaps and the snapshots with every other
        // test's submission, which would otherwise interleave with them.
        let _quiet = h.quiet();
        let w = loose_window_bytes() as usize;
        // SAFETY: a fresh allocation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * w, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let b = p.next_multiple_of(w);
        let title = |at: usize| (b + at) as *mut u8;
        // SAFETY: (all title reads and writes below) bytes of our allocation.
        unsafe { title(w - 40).write_volatile(0x11) };
        let first = loose((b + w - 64) as u64, (b + w - 32) as u64).expect("a window");
        let spanning = loose((b + w - 16) as u64, (b + w + 16) as u64).expect("a window across the boundary");
        assert!(!Arc::ptr_eq(&first, &spanning));
        assert_eq!((first.start, first.len), (b as u64, w as u64));
        assert_eq!((spanning.start, spanning.len), (b as u64, 2 * w as u64), "the first lies inside it");
        let byte = |s: &Shadow, at: usize| s.read_host(((b + at) & !3) as u64, 1)[0].to_le_bytes()[(b + at) % 4];
        // The title writes: both copies take it at their refresh.
        // SAFETY: a byte of our allocation.
        unsafe { title(w - 40).write_volatile(0x22) };
        first.update(b + w - 64, 64);
        spanning.update(b + w - 64, 64);
        assert_eq!((byte(&first, w - 40), byte(&spanning, w - 40)), (0x22, 0x22));
        // A shader writes through the spanning copy; it goes back to title
        // memory, and the first copy takes it at its next refresh.
        spanning.writing(b + w - 64, 64, false);
        spanning.write_host((b + w - 40) as u64, &[0x33]);
        spanning.copy_back(b + w - 64, 64);
        // SAFETY: a byte of our allocation.
        let back = unsafe { title(w - 40).read_volatile() };
        assert_eq!(back, 0x33, "back in title memory");
        first.update(b + w - 64, 64);
        assert_eq!(byte(&first, w - 40), 0x33, "and in the first copy");
        {
            let mut loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
            loose.retain(|&(s, _), _| s != b as u64);
            lookup_changed();
        }
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn loose_memory_must_be_committed() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed.
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), loose_window_bytes() as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(p, 0);
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }

    #[test]
    fn lookup_cache_switch_defaults_on_and_disables_with_zero() {
        use std::ffi::OsStr;
        assert!(lookup_cache_enabled_value(None));
        assert!(lookup_cache_enabled_value(Some(OsStr::new("1"))));
        assert!(lookup_cache_enabled_value(Some(OsStr::new(""))));
        assert!(!lookup_cache_enabled_value(Some(OsStr::new("0"))));
    }

    #[test]
    fn lookup_cache_hits_and_invalidates_on_register_and_unregister() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let len = 0x20000usize;
        // SAFETY: fresh committed range, freed below.
        let base = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_ne!(base, 0);
        let outer_id = 0x7001usize;
        let inner_id = 0x7002usize;
        register(outer_id, base, len as u64);
        let Some(outer) = find(base + 0x1000, 256) else {
            unregister(outer_id, base);
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return;
        };
        let outer_hit = find(base + 0x1000, 256).expect("outer cache hit");
        assert!(Arc::ptr_eq(&outer, &outer_hit));

        // Registering an overlapping inner buffer without a shadow must invalidate
        // the cached outer hit on this thread.
        register(inner_id, base + 0x1000, 0x10000);
        let inner = find(base + 0x1000, 256).expect("inner shadow");
        assert!(!Arc::ptr_eq(&outer, &inner));

        // Unregistering the inner buffer restores the outer shadow.
        unregister(inner_id, base + 0x1000);
        let restored = find(base + 0x1000, 256).expect("restored outer shadow");
        assert!(Arc::ptr_eq(&outer, &restored));

        unregister(outer_id, base);
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn lookup_cache_invalidates_on_aliased_register_and_unregister() {
        let Some(h) = host() else { return };
        let Some((buffer, data)) = h.shared_buffer(0x10000) else { return };
        // SAFETY: live host buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = 0x7000_1000_0000u64;
        let outer_id = 0x7003usize;
        register(outer_id, base, 0x10000);
        let outer = find(base + 0x100, 64).expect("outer shadow");
        assert!(!outer.is_aliased());

        let aliased = Shadow::new_aliased(buffer, data, gpu, base, 0x10000);
        register_aliased(aliased.clone());
        let hit = find(base + 0x100, 64).expect("aliased shadow");
        assert!(hit.is_aliased());
        assert!(Arc::ptr_eq(&aliased, &hit));

        assert!(unregister_aliased(base));
        let restored = find(base + 0x100, 64).expect("restored outer shadow");
        assert!(!restored.is_aliased());
        assert!(Arc::ptr_eq(&outer, &restored));

        unregister(outer_id, base);
    }

    #[test]
    fn lookup_cache_does_not_keep_released_shadow_alive() {
        let base = 0x7000_2000_0000u64;
        let id = 0x7004usize;
        register(id, base, 0x10000);
        let Some(shadow) = find(base + 64, 64) else {
            unregister(id, base);
            return;
        };
        let weak = Arc::downgrade(&shadow);
        drop(shadow);
        unregister(id, base);
        assert_eq!(weak.strong_count(), 0, "TLS cache must not keep released shadow alive");
    }

    #[test]
    fn lookup_cache_concurrent_register_stays_coherent() {
        let base = 0x7000_3000_0000u64;
        let outer_id = 0x7005usize;
        let inner_id = 0x7006usize;
        register(outer_id, base, 0x20000);
        let Some(outer) = find(base + 0x1000, 64) else {
            unregister(outer_id, base);
            return;
        };
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let done = done.clone();
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        let _ = find(base + 0x1000, 64);
                    }
                    find(base + 0x1000, 64)
                })
            })
            .collect();

        for _ in 0..50 {
            register(inner_id, base + 0x1000, 0x10000);
            let _ = find(base + 0x1000, 64);
            unregister(inner_id, base + 0x1000);
        }
        // Leave inner registered as the final state and verify every reader converges to it.
        register(inner_id, base + 0x1000, 0x10000);
        let expected = find(base + 0x1000, 64).expect("final inner shadow");
        assert!(!Arc::ptr_eq(&outer, &expected));
        done.store(true, Ordering::Release);

        for r in readers {
            let seen = r.join().expect("reader thread").expect("final shadow");
            assert!(Arc::ptr_eq(&expected, &seen), "reader must not hold stale TLS entry");
        }
        unregister(inner_id, base + 0x1000);
        unregister(outer_id, base);
    }

    #[test]
    fn parallel_copies_switch_defaults_on_and_disables_with_zero() {
        use std::ffi::OsStr;
        assert!(parallel_copies_enabled_value(None));
        assert!(parallel_copies_enabled_value(Some(OsStr::new("1"))));
        assert!(!parallel_copies_enabled_value(Some(OsStr::new("0"))));
    }

    #[test]
    fn ranges_disjoint_detects_overlap_and_allows_adjacent() {
        assert!(ranges_disjoint(&[(0x1000, 0x1000), (0x2000, 0x1000)]));
        assert!(ranges_disjoint(&[(0x2000, 0x1000), (0x1000, 0x1000)]));
        assert!(!ranges_disjoint(&[(0x1000, 0x1500), (0x2000, 0x1000)]));
    }
}
