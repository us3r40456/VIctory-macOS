//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod ahead;
#[cfg(windows)]
pub(crate) mod host;
// Also built for tests, which run where `host` cannot.
#[cfg(any(windows, test))]
mod paged;
// mac: plain std too.
#[cfg(any(windows, test))]
pub(crate) mod pool;
#[cfg(any(windows, test))]
pub(crate) mod reports;
#[cfg(any(windows, test))]
pub(crate) mod slots;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (Grand Theft Auto VI) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// A shader-visible heap in the title's memory, tracked: the title also
    /// copies descriptors into it itself, as bytes (`sync_heaps`).
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

#[cfg(windows)]
pub(crate) use host::fallback_samplers;

/// A title heap's host copy.
pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of Grand Theft Auto VI's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        // Submissions held off while a view heap is made and filled
        // (`Host::quiet`), from before `pending` is taken: `mirror` waits
        // for that with `SHADER_HEAPS` held, which a submission waits for.
        let _quiet = (self.desc.kind == VIEWS_KIND)
            .then(|| crate::host::host()?.quiet())
            .flatten();
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

#[cfg(any(windows, test))]
type Registration = (u64, u64, usize);

/// Registers a heap, retiring any range it overlaps. Descriptor handles carry
/// no heap identity, so two live heaps cannot unambiguously own one address.
#[cfg(any(windows, test))]
fn register_heap(heaps: &mut Vec<Registration>, (start, end): (u64, u64), at: usize) {
    heaps.retain(|&(old_start, old_end, _)| old_end <= start || end <= old_start);
    heaps.push((start, end, at));
}

/// Removes only this heap's registration.
#[cfg(any(windows, test))]
fn unregister_heap(heaps: &mut Vec<Registration>, at: usize) {
    heaps.retain(|&(_, _, heap)| heap != at);
}

#[cfg(any(windows, test))]
static AMBIGUOUS_RANGE: std::sync::Once = std::sync::Once::new();

/// The heap owning `handle`, if exactly one registered range contains it.
#[cfg(any(windows, test))]
fn heap_at(heaps: &[Registration], handle: u64) -> Option<usize> {
    let mut found = None;
    for &(start, end, at) in heaps {
        if (start..end).contains(&handle) && found.replace(at).is_some() {
            AMBIGUOUS_RANGE.call_once(|| {
                tracing::warn!(
                    handle,
                    "overlapping descriptor heaps: descriptor write refused"
                )
            });
            return None;
        }
    }
    found
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    #[cfg(feature = "diagnostics")]
    let waited = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MirrorWaitUs);
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    drop(waited);
    let Some(at) = heap_at(&heaps, handle) else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Mirrors what the title wrote into its shader-visible heaps without a
/// call: Grand Theft Auto VI copies descriptors (our records) into its heap
/// as bytes, so only a change of its memory shows them. Each slot of a page
/// written since the last sync is made again if its record changed (in the
/// world 1,300-14,400 slots a frame, against a few hundred views made by
/// calls). Called before each submission: shaders take descriptors from the
/// heap at indices in their data, where a slot never mirrored reads nothing.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    sync(None);
}

/// A round of [`sync_heaps`] while the host GPU runs a submission left to
/// it ([`ahead`]): what the title wrote so far for the lists to come, but
/// for the slots a round leaves ([`host::HostHeap::mirror`]). `ran` says
/// that the GPU has run the submission: the round stops at the next page,
/// and the pages it took and did not look at are kept for the next sync.
#[cfg(windows)]
pub(crate) fn sync_heaps_ahead(ran: &dyn Fn() -> bool) {
    let _round = ahead::Round::start();
    sync(Some(ran));
}

/// `VICTORY_HEAP_FAST_SKIP=1`: heap syncs look at a written page the way
/// VIctory-macOS #279 does ([`host::HostHeap::sync_slots_bytes`]). Off
/// unless asked for: this line's syncs have changed since (rounds ahead,
/// told slots, slot locks), nothing here has run it in the game, and a
/// tester of the Mac port reported loading screens that froze with it.
#[cfg(windows)]
fn fast_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_HEAP_FAST_SKIP"))
}

#[cfg(windows)]
thread_local! {
    /// A heap page's bytes, for [`fast_skip`]'s look.
    static PAGE_BUF: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

#[cfg(windows)]
fn sync(ran: Option<&dyn Fn() -> bool>) {
    // mac: lock waits and made slots are counted as a sync's (`slots`).
    let _syncing = slots::Syncing::enter();
    // Held throughout: a heap is not dropped while its slots are made.
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // mac: the heap whose slots the title's own stores report
        // (`VICTORY_HEAP_REPORTS`, `reports`; `No` without the switch).
        let told = reports::told(start, heap.desc.count);
        if told == reports::Told::Use {
            // The told slots in place of the written pages; what rounds
            // ahead left first, as below.
            if ahead::on() && ran.is_none() {
                host.sync_left(start, step);
            }
            if !sync_told(heap, host, start, step, ran) {
                return;
            }
            // The fallback, in the sync before a submission every so many
            // frames: the watch asked and its pages looked at whole, below.
            if ran.is_some() || !reports::ask_due(crate::frame::current()) {
                continue;
            }
        }
        let mut runs = written.take(start as usize, usize::MAX);
        // mac: what rounds ahead left comes first, the slots only before a
        // submission.
        if ahead::on() {
            runs = host.with_left(runs);
            if ran.is_none() && told != reports::Told::Use {
                host.sync_left(start, step);
            }
        }
        let mut runs = runs.into_iter();
        while let Some((from, to)) = runs.next() {
            // By page: one with nothing mapped (the title unmaps heap pages
            // too) holds no records.
            let page_size = crate::watch::PAGE;
            for page in (from & !(page_size - 1)..to).step_by(page_size) {
                let (page, end) = (page.max(from), (page + page_size).min(to));
                // mac: a round ahead ends when the GPU has run the
                // submission; the rest of what it took is kept.
                if ran.is_some_and(|ran| ran()) {
                    host.leave(std::iter::once((page, to)).chain(runs));
                    ahead::cut();
                    return;
                }
                if !crate::host::read_into(page as *const u8, &mut [0u8; 4]) {
                    continue;
                }
                let (first, last) = ((page as u64 - start) / step, (end as u64 - start).div_ceil(step));
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(
                    crate::copystat::C::SyncSlots,
                    last.min(u64::from(heap.desc.count)).saturating_sub(first),
                );
                let (first, last) = (first as u32, last.min(u64::from(heap.desc.count)) as u32);
                if told == reports::Told::No {
                    // VIctory-macOS #279's look, on this line a switch
                    // (`fast_skip`): the page read once, empty and verified
                    // slots passed by without the view lock.
                    if fast_skip() {
                        let len = end - page;
                        let mut buf = PAGE_BUF.take();
                        buf.resize(page_size, 0);
                        if crate::host::read_into(page as *const u8, &mut buf[..len]) {
                            host.sync_slots_bytes(first, last, start, step, page as u64, &buf[..len]);
                        }
                        PAGE_BUF.set(buf);
                        continue;
                    }
                    host.sync_slots(first, last, start, step);
                    continue;
                }
                // mac: what the look finds changed is sorted by what told
                // of it (`reports`), then made as ever.
                let stale = host.stale_slots(first, last, start, step);
                // SAFETY: records on a page of the heap found mapped above.
                let is_view = |i: u32| unsafe { ((start + u64::from(i) * step) as *const u32).read_unaligned() } == VIEW_TAG;
                reports::looked(start, first, last, &stale, is_view);
                host.make_stale(stale, start, step);
            }
        }
        if told == reports::Told::Check && ran.is_none() {
            reports::clear_rest(start);
        }
    }
}

/// mac: a heap sync of the slots the title's own stores told of
/// (`VICTORY_HEAP_REPORTS=1`, [`reports`]) in place of the written pages:
/// the same look and the same making ([`host::HostHeap::sync_slots`]), of
/// those slots alone. By 64 KiB page of the title's heap, as the look at
/// written pages goes: a page with nothing mapped holds no records, and a
/// round ahead of a submission (`ran`) ends at the next page once the GPU
/// has run it. False for a round that ended so: the slots it took and did
/// not look at are told again.
#[cfg(windows)]
fn sync_told(heap: &Heap, host: &host::HostHeap, start: u64, step: u64, ran: Option<&dyn Fn() -> bool>) -> bool {
    let runs = reports::take(start);
    let page = crate::watch::PAGE as u64;
    for (k, &(first, n)) in runs.iter().enumerate() {
        let end = (first + n).min(heap.desc.count);
        let mut at = first;
        while at < end {
            let address = start + u64::from(at) * step;
            // The slots that start on this page.
            let next_page = (address / page + 1) * page;
            let to = end.min((next_page - start).div_ceil(step) as u32);
            if ran.is_some_and(|ran| ran()) {
                reports::put_back(start, at, first + n - at);
                for &(first, n) in &runs[k + 1..] {
                    reports::put_back(start, first, n);
                }
                ahead::cut();
                return false;
            }
            if crate::host::read_into(address as *const u8, &mut [0u8; 4]) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::SyncSlots, u64::from(to - at));
                let stale = host.stale_slots(at, to, start, step);
                reports::changed(stale.len());
                host.make_stale(stale, start, step);
            }
            at = to;
        }
    }
    true
}

/// Makes the views of slots `slots` that were left null (textures the title
/// placed itself, textures given up) in every shader-visible view heap, now
/// that shaders read them (`crate::shadow::take_reads`), and stamps what
/// they view as used (`crate::resource::evict`).
#[cfg(windows)]
pub(crate) fn make_read(slots: &[u32]) {
    // The placed textures those slots view are made first, without the
    // heaps' lock, which every title thread writing a view waits on (making
    // a texture takes up to milliseconds; title threads waited 1-6 s a frame,
    // runs evict2, noevict1). They are never released, so they outlive it;
    // the lock is held only to look the slots up and to make the views.
    let placed: Vec<View> = {
        let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
        heaps
            .iter()
            // SAFETY: registered heaps are live (removed when dropped).
            .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
            .filter(|heap| heap.desc.kind == VIEWS_KIND)
            .filter_map(|heap| heap.host.get()?.as_ref().map(|host| (heap, host)))
            .flat_map(|(heap, host)| slots.iter().filter(|&&i| i < heap.desc.count).filter_map(|&i| host.unmade_placed(i)))
            .collect()
    };
    #[cfg(test)]
    tests::while_making(slots);
    // Several at a time (each a committed texture; hundreds in a new shot,
    // 7 % of the submitting thread when made one by one, run evict5).
    crate::texture::in_parallel(placed.len(), |i| {
        let _ = host::placed_texture(&placed[i]);
    });
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(step)) = (heap.host.get(), increment(heap.desc.kind)) else {
            continue;
        };
        if heap.desc.kind != VIEWS_KIND {
            continue;
        }
        for &i in slots.iter().filter(|&&i| i < heap.desc.count) {
            host.mirror(i, start + u64::from(i) * u64::from(step), true);
            host.stamp_read(i);
        }
    }
}

/// Makes again, in every shader-visible view heap, the unordered texture
/// views left null when their textures were given up
/// (`VICTORY_EVICT_HEAP_UAV`): from now on shaders write through heaps.
#[cfg(windows)]
pub(crate) fn remake_unordered() {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut made = 0;
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), Some(step), VIEWS_KIND) = (heap.host.get(), increment(heap.desc.kind), heap.desc.kind) {
            made += host.remake_unordered(start, u64::from(step));
        }
    }
    tracing::info!(made, "shaders write through heaps: unordered texture views made again");
}

/// The host textures (raw pointers) that slots `slots` of the
/// shader-visible view heaps view.
#[cfg(windows)]
pub(crate) fn read_textures(slots: &[u32]) -> std::collections::HashSet<usize> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut textures = std::collections::HashSet::new();
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            textures.extend(host.textures_at(slots));
        }
    }
    textures
}

/// Every shader-visible view heap's title slots viewing one of the host
/// textures `textures` (raw pointers) are left null, made again when read
/// (`crate::resource::evict` gives those textures up). Dynamic slots stay
/// made for the lists that recorded them.
#[cfg(windows)]
pub(crate) fn evict(textures: &std::collections::HashSet<usize>) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            host.evict(textures);
        }
    }
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

/// Runs `f` on the host heap behind the title heap at `heap` while that
/// heap cannot be released; None if it is gone or has no host heap.
#[cfg(windows)]
pub(crate) fn with_host<R>(heap: usize, f: impl FnOnce(&host::HostHeap) -> R) -> Option<R> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if !heaps.iter().any(|&(_, _, h)| h == heap) {
        return None;
    }
    // SAFETY: registered heaps are live (removed when dropped, which waits
    // for the lock we hold).
    let host = unsafe { &*(heap as *const Heap) }.host.get()?.as_ref()?;
    Some(f(host))
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        unregister_heap(&mut heaps, at);
        // mac: its slots are told no more (`VICTORY_HEAP_REPORTS`).
        reports::let_go(self.start);
        // Nothing can find the heap by now. The table's lock is taken
        // inside the heaps' lock (`memory::bind`), never the other way
        // round, so it is taken here after the heaps' lock is let go.
        drop(heaps);
        crate::memory::release(at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(windows)]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // Render target and depth views: a packed 16-bit target may be
        // R8G8B8A8_UNORM on this GPU (`resource::target_format`).
        words[0] = crate::resource::target_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

/// Read by every view written and every slot a draw binds, on the
/// title's threads at once; written when a view is first seen or its
/// resource released.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    // A view seen before, as most are: its entry is found under the shared
    // lock, which keeps `forget` from making it stale while it is written.
    let known = {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        let found = views.index.get(&view).map(|&i| (i, views.list[i as usize].1));
        if let Some((i, generation)) = found {
            // SAFETY: the handle is a descriptor slot, at least 16 bytes.
            unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
        }
        found.is_some()
    };
    if known {
        #[cfg(windows)]
        mirror(handle);
        return;
    }
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    let i = match views.index.get(&view) {
        Some(&i) => i,
        None => {
            let i = match views.free.pop() {
                Some(i) => {
                    views.list[i as usize].0 = view.clone();
                    i
                }
                None => {
                    views.list.push((view.clone(), 1));
                    views.list.len() as u32 - 1
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                views.by_resource.entry(r).or_default().push(i);
            }
            views.index.insert(view, i);
            i
        }
    };
    let generation = views.list[i as usize].1;
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    drop(views);
    #[cfg(windows)]
    mirror(handle);
}

/// Bumped whenever [`forget`] invalidates a recorded view while holding `VIEWS`.
static VIEWS_FORGOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(any(windows, test))]
pub(crate) fn views_forgot() -> u64 {
    VIEWS_FORGOT.load(std::sync::atomic::Ordering::Acquire)
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    let mut removed = false;
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
        removed = true;
    }
    if removed {
        VIEWS_FORGOT.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// A current view whose resources cannot be freed while it is used: it holds
/// a reference on each resource object the view names (and none is made for
/// a resource the title's last release had already reached). The title's
/// release never waits for it. A release that would have been the last runs
/// at the pin's drop instead, on whichever thread drops it, so a pin is
/// dropped outside the locks the destructor takes (`VIEWS`, `bindless`,
/// `shadow`).
#[cfg(any(windows, test))]
pub(crate) struct ViewPin {
    view: View,
    held: [usize; 2],
}

#[cfg(any(windows, test))]
impl std::ops::Deref for ViewPin {
    type Target = View;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

#[cfg(any(windows, test))]
impl std::fmt::Debug for ViewPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.view.fmt(f)
    }
}

#[cfg(any(windows, test))]
impl Drop for ViewPin {
    fn drop(&mut self) {
        for r in self.held.into_iter().filter(|&r| r != 0) {
            crate::object::release_ref(r as This);
        }
    }
}

/// The view of entry `i` if `generation` is still its own, pinned until the
/// returned value is dropped.
///
/// The references are taken under the shared lock. `Resource::drop` takes
/// the exclusive one (`forget`) before its memory goes, so the address of a
/// current entry's resource is still there to count on. A resource whose
/// count already reached zero is on its way out, and the view is not current
/// for it.
#[cfg(any(windows, test))]
fn view_at(i: u32, generation: u32) -> Option<ViewPin> {
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    if *current != generation {
        return None;
    }
    let view = view.clone();
    let mut held = [0usize; 2];
    let mut all = true;
    for (slot, r) in view.resources().into_iter().enumerate() {
        if r == 0 {
            continue;
        }
        if crate::object::try_add_ref(r as This) {
            held[slot] = r;
        } else {
            all = false;
        }
    }
    drop(views);
    // Dropped here, with the lock released, if a resource was already gone:
    // the references taken on the others may be their last.
    let pin = ViewPin { view, held };
    all.then_some(pin)
}

/// For each record (tag, index, generation): whether it names a view that
/// is current, as [`view_at`] would find it (one lock for them all).
fn views_current_into(records: &[[u32; 3]], out: &mut Vec<bool>) {
    out.clear();
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    out.extend(records.iter().map(|&[tag, i, generation]| {
        tag == VIEW_TAG && views.list.get(i as usize).is_some_and(|(_, current)| *current == generation)
    }));
}

/// The view recorded at `handle`, if one is (and its resource is alive),
/// pinned until the returned value is dropped.
#[cfg(any(windows, test))]
pub(crate) fn view(handle: u64) -> Option<ViewPin> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
///
/// A null `desc` writes a null CBV (`Constant(0, 0)`), matching D3D12's
/// `CreateConstantBufferView` with a null description. Leaving the slot
/// untouched would keep a stale view active.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    let view = match words(desc) {
        Some([lo, hi, size, _]) => {
            let address = u64::from(lo) | u64::from(hi) << 32;
            View::Constant(address, size)
        }
        None => View::Constant(0, 0),
    };
    record(handle, view);
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    // mac: the title's own stores tell this heap's slots
    // (`VICTORY_HEAP_REPORTS`; nothing without it).
    #[cfg(windows)]
    if placed && visible && desc.kind == VIEWS_KIND {
        reports::adopt(start, desc.count);
    }
    #[cfg(windows)]
    let at = std::ptr::from_ref::<Heap>(&heap) as usize;
    #[cfg(not(windows))]
    let _ = visible;
    let result = object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out);
    #[cfg(windows)]
    if visible && result == crate::object::S_OK {
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        register_heap(&mut heaps, (start, start + size), at);
    }
    result
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
pub(crate) mod tests {
    /// Work to run while `make_read` makes the textures of a read slot (the
    /// slot), outside the heaps' lock.
    pub(crate) static WHILE_MAKING: std::sync::Mutex<Option<(u32, Work)>> = std::sync::Mutex::new(None);

    /// The work [`WHILE_MAKING`] runs.
    pub(crate) type Work = Box<dyn FnOnce() + Send>;

    pub(crate) fn while_making(slots: &[u32]) {
        let mut hook = WHILE_MAKING.lock().unwrap_or_else(|e| e.into_inner());
        if hook.as_ref().is_some_and(|(slot, _)| slots.contains(slot))
            && let Some((_, work)) = hook.take()
        {
            drop(hook);
            work();
        }
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    /// A released view heap takes its title memory page table with it, also
    /// when no heap is made at its address afterwards (else the table, its
    /// 16.8 MiB mapping and the copies it names live on).
    #[cfg(windows)]
    #[test]
    fn a_released_view_heap_drops_its_page_table() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1000,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let key = h as usize;
        assert!(heap(h).host().is_some(), "a host heap");
        let list = host.new_list().unwrap();
        assert!(crate::memory::bind(&list, key).is_some(), "a page table");
        assert!(crate::memory::has_table(key));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(h, 0x10)(h), 0);
        assert!(!crate::memory::has_table(key), "released with its heap");
    }

    #[test]
    fn registering_a_heap_retires_overlapping_ranges_but_keeps_adjacent_ones() {
        let mut heaps = vec![(10, 20, 1), (20, 30, 2), (30, 40, 3), (40, 50, 4)];
        register_heap(&mut heaps, (15, 35), 5);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5)]);
        register_heap(&mut heaps, (35, 40), 6);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5), (35, 40, 6)]);
    }

    #[test]
    fn lookup_is_half_open_and_refuses_ambiguous_ranges() {
        let heaps = [(10, 20, 1), (20, 30, 2)];
        assert_eq!(heap_at(&heaps, 10), Some(1));
        assert_eq!(heap_at(&heaps, 19), Some(1));
        assert_eq!(heap_at(&heaps, 20), Some(2));
        assert_eq!(heap_at(&heaps, 30), None);
        assert_eq!(heap_at(&[(10, 30, 1), (20, 40, 2)], 25), None);
    }

    #[test]
    fn unregistering_a_retired_heap_keeps_its_replacement() {
        let mut heaps = vec![(10, 20, 1)];
        register_heap(&mut heaps, (10, 20), 2);
        unregister_heap(&mut heaps, 1);
        assert_eq!(heaps, [(10, 20, 2)]);
    }

    #[cfg(windows)]
    struct HeapOwner(This);

    #[cfg(windows)]
    impl Drop for HeapOwner {
        fn drop(&mut self) {
            if !self.0.is_null() {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }
    }

    #[cfg(windows)]
    fn owner(desc: HeapDesc, placement: Placement) -> HeapOwner {
        let mut this = std::ptr::null();
        assert_eq!(create(desc, placement, &mut this), S_OK);
        HeapOwner(this)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0).as_deref(), Some(&View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1).as_deref(), Some(&View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2).as_deref(), view(s0).as_deref());
        assert_eq!(view(s3).as_deref(), view(s1).as_deref());
        assert!(view(at(0) + 8).is_none());
        heap[0] = 0;
    }

    /// A null CBV description overwrites the slot (D3D12 null CBV), so a
    /// later read does not see the previous view.
    #[test]
    fn null_cbv_clears_a_stale_view() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let a = slots.as_ptr() as u64;
        create_cbv(null, &[0x1000, 1, 256, 0], a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0x1_0000_1000, 256)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0, 0)));
        create_srv(null, null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Shader(0, None)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0, 0)));
    }

    /// An object of ours that frees itself at its last release, as a
    /// resource does: its views go stale first (`forget`), and `FREED` notes
    /// its address.
    static FREED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    fn freed(address: usize) -> bool {
        FREED.lock().unwrap().contains(&address)
    }

    extern "system" fn release_resource_like(this: This) -> u32 {
        if crate::object::refs(this) == 1 {
            forget(this as usize);
            FREED.lock().unwrap().push(this as usize);
        }
        crate::object::release_boxed::<crate::object::Object>(this)
    }

    /// A released resource's views go stale; the same view of a new
    /// resource is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = crate::object::test_object(release_resource_like);
        let address = resource as usize;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Shader(address, None)));
        // The last release frees the object, and its views go stale.
        assert_eq!(crate::object::release_ref(resource), 0);
        assert!(view(a).is_none(), "stale after release");
        let again = crate::object::test_object(release_resource_like);
        create_srv(null, again, std::ptr::null(), b);
        assert_eq!(
            view(b).as_deref(),
            Some(&View::Shader(again as usize, None))
        );
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old[1..], new[1..], "a new entry or generation");
        assert_eq!(crate::object::release_ref(again), 0);
    }

    /// A final release with the view pinned does not wait: the title's call
    /// returns at once, the resource is kept alive, and the last pin's drop
    /// does the freeing (which makes the views stale then).
    #[test]
    fn a_pinned_resource_is_freed_by_the_last_pin() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let at = slots.as_ptr() as u64;
        let resource = crate::object::test_object(release_resource_like);
        let address = resource as usize;
        FREED.lock().unwrap().retain(|&a| a != address);
        create_srv(null, resource, std::ptr::null(), at);
        let first = view(at).expect("a current view");
        let second = view(at).expect("the same current view again");

        // The title releases its last reference with both pins held; the
        // call returns while the object is held by them.
        assert_eq!(crate::object::release_ref(resource), 2);
        assert!(!freed(address));
        assert!(view(at).is_some(), "the entry stays current while pinned");
        drop(first);
        assert!(!freed(address));
        drop(second);
        assert!(freed(address), "the last pin freed it");
        assert!(view(at).is_none(), "stale after the last pin");
    }

    /// A record the title copies into its shader-visible heap as bytes (no
    /// call) is made on the host at the next sync, and only then.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        let memory = crate::watch::tests::pool(1);
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            // Our record, written where no heap sees it, then copied in.
            let mut elsewhere = [0u64; 2];
            let at = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], at);
            let slot = memory as u64 + 5 * 16;
            // SAFETY: 16 bytes of each, ours; the heap's page is guarded
            // (our handler opens it).
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot as *mut u8, 16) };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5), "made at the sync");
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }

    /// A placed heap reused at the same address replaces the stale range;
    /// descriptor writes go only to the new heap.
    #[cfg(windows)]
    #[test]
    fn overlapping_placed_heaps_mirror_into_the_replacement() {
        let mut memory = vec![0u8; 2 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 2,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement {
            address,
            size: 2 * 32,
        };
        let old = owner(desc, placement);
        let new = owner(desc, placement);
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                heaps
                    .iter()
                    .filter(|&&(start, end, _)| (start..end).contains(&address))
                    .count(),
                1
            );
            assert_eq!(heap_at(&heaps, address), Some(new.0 as usize));
        }

        create_cbv(std::ptr::null(), &[0x1000, 1, 256, 0], address);
        assert!(heap(old.0).pending.lock().unwrap().is_empty());
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0]);

        drop(old);
        create_cbv(std::ptr::null(), &[0x2000, 1, 256, 0], address + 32);
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0, 1]);
    }

    /// A creation that does not hand out a heap must not retire an existing
    /// registration for the same memory.
    #[cfg(windows)]
    #[test]
    fn failed_overlapping_creation_keeps_the_existing_heap() {
        let mut memory = vec![0u8; 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement { address, size: 32 };
        let existing = owner(desc, placement);
        let extended = Extended(desc, placement);
        let bad_iid = [0xff; 16];
        let mut out = std::ptr::null();
        assert_ne!(
            create_heap(std::ptr::null(), &extended.0, &bad_iid, &mut out),
            S_OK
        );
        assert!(out.is_null());
        assert_eq!(
            create_heap(
                std::ptr::null(),
                &extended.0,
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            crate::object::S_FALSE
        );
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(heap_at(&heaps, address), Some(existing.0 as usize));
        }
        create_cbv(std::ptr::null(), &[0x3000, 1, 256, 0], address);
        assert_eq!(heap(existing.0).pending.lock().unwrap().as_slice(), [0]);
    }

    /// Grand Theft Auto VI places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement { address, size: 64 * 32 };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // Remove the placed heap from the registry before its memory can
        // be reused by another heap.
        release(h);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        release(h);

        #[cfg(windows)]
        {
            // Reuse the address without relying on the allocator to give it
            // to another test. The descriptor write must reach the new heap.
            assert_eq!(create(desc, placement, &mut h), S_OK);
            create_srv(std::ptr::null(), std::ptr::null(), std::ptr::null(), address);
            let reused = heap(h);
            let queued = reused.pending.lock().unwrap().as_slice() == [0];
            release(h);
            assert!(queued, "the descriptor write reached the heap at the reused address");
        }
    }
}
