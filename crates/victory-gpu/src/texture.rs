//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, the whole memory
//! compared with the bytes last detiled at every submission that samples it.
//!
//! Both refresh paths detile only the subresources whose bytes changed: the
//! tracked one those whose bytes hash differently from the bytes last
//! detiled, the compared one those that a differing run touches. A change
//! inside one mip does not detile the others.
//!
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: how a staging buffer lays out the host
/// texture's subresources, and what was last detiled. The buffer itself is
/// taken for each upload from a small pool shared by all textures
/// ([`staging_buffer`]; kept for each texture's life, the buffers took its
/// size again in system memory: 2.6-2.8 GB with placed textures made on
/// read, run integ2).
pub(crate) struct Staging {
    /// A staging buffer's bytes.
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, compared instead).
    tracked: Option<crate::watch::Tracked>,
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// Uploads [`record_upload`] recorded, and the subresources they
    /// copied.
    uploads: std::sync::atomic::AtomicU64,
    copied: std::sync::atomic::AtomicU64,
}

/// What was last detiled. Memory not tracked keeps the bytes (compared to
/// tell a change: hashing them ran at 0.45 GB/s under Wine on Rosetta, where
/// no memory is tracked); tracked memory tells from its pages that it
/// changed, keeps nothing (`read` stays empty), and hashes each
/// subresource's bytes.
#[derive(Default)]
struct Last {
    /// Untracked memory: the bytes last detiled; `valid` while they are the
    /// memory's (not before the first read, nor after a read that failed
    /// part way).
    read: Vec<u8>,
    valid: bool,
    /// Something was detiled (bytes or zeros), so the staging buffer holds
    /// the memory as it was then.
    uploaded: bool,
    /// Tracked memory: the hash of each subresource's bytes last detiled.
    hashes: Vec<Option<blake3::Hash>>,
}

/// Per subresource of `layout` (`mips` per array slice; a 3D texture's
/// are its mips), the bytes of the texture's memory it lies in, rounded out
/// to the layout's blocks; empty if the layout cannot say.
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32) -> Vec<(usize, usize)> {
    use victory_formats::texture::Dimension;
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips, i / mips);
            let m = layout.mips.get(mip as usize)?;
            let (z0, z1) = if d.dimension == Dimension::Tex3d { (0, m.depth.max(1) - 1) } else { (slice, slice) };
            let first = layout.offset(mip, 0, 0, z0).ok()?;
            let last = layout.offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1).ok()? + u64::from(d.element_bytes);
            let (from, to) = (first / block * block, last.div_ceil(block) * block);
            Some((from as usize, (to.min(layout.size)) as usize))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Staging buffers for detiles to reuse, by size: the menu movie rewrites
/// the same textures every frame, and a fresh 12.7 MB of upload buffers a
/// frame cost ~5 ms (run textures8). Uploads run only inside
/// `Host::execute`, which waits for each submission, so the buffers of one
/// submission are free from the next on.
struct Buffers {
    /// (bytes, buffer, mapping), oldest first.
    free: std::collections::VecDeque<(u64, ID3D12Resource, usize)>,
    free_bytes: u64,
    /// Taken by the submission being prepared.
    used: Vec<(u64, ID3D12Resource, usize)>,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Buffers {}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    free: std::collections::VecDeque::new(),
    free_bytes: 0,
    used: Vec::new(),
});

/// Free staging buffers kept at most.
const KEPT_BYTES: u64 = 64 << 20;

/// Called before a submission's uploads: the buffers of the last one are
/// free again (the oldest dropped past `KEPT_BYTES`).
pub(crate) fn recycle() {
    {
        let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
        (c.current, c.used) = (0, 0);
        c.all.truncate(CHUNKS_KEPT);
    }
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let b = &mut *b;
    for e in b.used.drain(..) {
        b.free_bytes += e.0;
        b.free.push_back(e);
    }
    while b.free_bytes > KEPT_BYTES
        && let Some((n, ..)) = b.free.pop_front()
    {
        b.free_bytes -= n;
    }
}

/// A free staging buffer of `size` bytes, or a new one.
/// Sizes are taken in powers of two (at least 64 KiB), so a buffer serves
/// textures of other sizes too: a new shot's textures came in new sizes and
/// made buffers 10 % of the submitting thread (run evict5).
fn staging_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let class = size.max(1 << 16).next_power_of_two();
    let taken = {
        let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.free.iter().position(|e| e.0 == class);
        i.and_then(|i| b.free.remove(i)).inspect(|_| b.free_bytes -= class)
    };
    // Made without the pool's lock (refreshes run on several threads).
    let e = match taken {
        Some(e) => e,
        None => {
            let (buffer, data) = host()?.upload_buffer(class)?;
            crate::host::name(&buffer, || "texture staging".into());
            (class, buffer, data as usize)
        }
    };
    let made = (e.1.clone(), e.2 as *mut u8);
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).used.push(e);
    Some(made)
}

/// Staging space carved from large upload buffers, front to back: most
/// textures take a part of one instead of a buffer of their own (making
/// upload buffers was 10 % of the submitting thread in a new shot, run
/// evict5). Rewound at each submission ([`recycle`]): the last one has run.
struct Chunks {
    /// (buffer, mapping), [`CHUNK`] bytes each.
    all: Vec<(ID3D12Resource, usize)>,
    /// The chunk being carved and the bytes taken from it.
    current: usize,
    used: u64,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Chunks {}

static CHUNKS: Mutex<Chunks> = Mutex::new(Chunks { all: Vec::new(), current: 0, used: 0 });

const CHUNK: u64 = 32 << 20;
/// Chunks kept between submissions (more are made while one needs them).
const CHUNKS_KEPT: usize = 16;

/// `size` bytes of staging space: a buffer, the offset of the space in it
/// (a multiple of 512, as texture copies need) and its mapping there.
/// Space larger than a chunk is a buffer of its own ([`staging_buffer`]).
fn staging_space(size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    if size > CHUNK {
        return staging_buffer(size).map(|(b, data)| (b, 0, data));
    }
    let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some((buffer, data)) = c.all.get(c.current) {
            let at = c.used.next_multiple_of(512);
            if at + size <= CHUNK {
                let made = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                c.used = at + size;
                return Some(made);
            }
            (c.current, c.used) = (c.current + 1, 0);
            continue;
        }
        let (buffer, data) = host()?.upload_buffer(CHUNK)?;
        crate::host::name(&buffer, || "texture staging chunk".into());
        c.all.push((buffer, data as usize));
    }
}

// SAFETY: the title memory is only read, under `last`'s lock.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve that `list` records writes `resource`: from the
/// submission that runs `list` on, its host texture's contents are the
/// host's, not title memory's, which stays as it was. Marked when that
/// submission is made, not when `list` is recorded: a list recorded before
/// it but run after it must not upload title memory over the host's texels
/// (submissions run in the order they are made, which need not be the
/// order their lists were recorded in), and one run before it must still
/// get its upload.
pub(crate) fn host_writes(list: &HostList, resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        list.defer_host(move || copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_writes(list, to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(&a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(&b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Has `resource`'s host copy brought up to date at the submission that
/// runs `list`: its staging buffer detiled again if the memory changed, and
/// then copied into the host texture by the submission's own upload list,
/// once for all its lists (`crate::bindless::uploads`).
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(resource as usize) {
                let memory = crate::resource::texture_memory(resource).map(|(m, _)| m);
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                tracing::debug!(?resource, ?memory, ?desc, "a sampled texture keeps what the host copied into it: not uploaded from title memory");
            }
        }
        return;
    }
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (staging(resource, texture), crate::resource::host_copy(resource)) else {
        return;
    };
    // Each list recorded the copy before (run perf7: 149 textures uploaded
    // 3,195 times in a frame, 17 ms of GPU time), changed or not.
    list.defer_texture(staging, copy);
}

/// Records into `list` the copy of `buffer` (laid out as `staging` says)
/// into host texture `copy`. Returns the barrier that leaves it readable by
/// shaders, for the caller to record with the others after every copy: one
/// barrier per texture between the copies kept the GPU from overlapping
/// them (new shots copy hundreds of textures, run evict6).
#[must_use]
pub(crate) fn record_upload(
    list: &HostList,
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    (buffer, at, subresources): &Upload,
) -> Option<D3D12_RESOURCE_BARRIER> {
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    if let Some(b) = list.move_texture(copy, COPY_DEST) {
        // SAFETY: an open list; the barrier names a live texture.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
    copies(list, &copy.texture, staging, (buffer, *at), subresources, String::new);
    list.keep(buffer);
    staging.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    staging.copied.fetch_add(subresources.len() as u64, std::sync::atomic::Ordering::Relaxed);
    list.move_texture(copy, SHADER_RESOURCE)
}

/// The uploads [`record_upload`] recorded into `resource`'s host texture
/// (tests: counts of their own texture, whatever other tests upload
/// meanwhile).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// The subresources those uploads copied (tests, as [`uploads`]).
#[cfg(test)]
pub(crate) fn copied(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.copied.load(std::sync::atomic::Ordering::Relaxed))
}


/// Records the copy of `subresources` of `buffer` (laid out as `staging`
/// says) into `texture` (in the copy destination state); `name` names it
/// in diagnostics notes.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
fn copies(
    list: &HostList,
    texture: &ID3D12Resource,
    staging: &Staging,
    (buffer, at): (&ID3D12Resource, u64),
    subresources: &[u32],
    name: impl Fn() -> String,
) {
    for &i in subresources {
        let i = i as usize;
        let mut footprint = staging.footprints[i];
        footprint.Offset += at;
        let footprint = &footprint;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        name(),
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// What a refresh gives: a staging buffer, where its space starts in it,
/// and the subresources detiled into it.
pub(crate) type Upload = (ID3D12Resource, u64, Vec<u32>);

/// Runs `f(i)` for each `i` below `n` on up to a quarter of the CPUs (four
/// at most: the title's own threads need the rest), here when that is
/// fewer than two.
pub(crate) fn in_parallel(n: usize, f: impl Fn(usize) + Sync) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(1, 4).min(n);
    if threads < 2 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                }
            });
        }
    });
}

/// [`Staging::refresh`] of each of `work`, those whose memory may have
/// changed on several threads at once (reading title memory and detiling
/// cost ~6 ms of frame a MB on one thread in a new shot, run evict5).
pub(crate) fn refresh_all(work: &[(Arc<Staging>, Arc<crate::resource::HostTexture>)]) -> Vec<Option<Upload>> {
    // Most find nothing changed (their trackers' clean bits say so): those
    // are not handed to threads.
    let changed: Vec<usize> = (0..work.len()).filter(|&i| work[i].0.maybe_changed()).collect();
    let results: Vec<Mutex<Option<Upload>>> = changed.iter().map(|_| Mutex::new(None)).collect();
    in_parallel(changed.len(), |k| {
        *results[k].lock().unwrap_or_else(|e| e.into_inner()) = work[changed[k]].0.refresh();
    });
    let mut out: Vec<Option<Upload>> = vec![None; work.len()];
    for (k, r) in results.into_iter().enumerate() {
        out[changed[k]] = r.into_inner().unwrap_or_else(|e| e.into_inner());
    }
    out
}

impl Staging {
    /// Whether its memory may have changed since it was last detiled (an
    /// untracked texture always may).
    pub(crate) fn maybe_changed(&self) -> bool {
        self.tracked.as_ref().is_none_or(|t| !t.clean())
    }

    /// A staging buffer with the texture memory detiled into it if it
    /// changed since the last time (and is mapped at all), and the
    /// subresources that changed (only those are detiled and to be copied).
    pub(crate) fn refresh(&self) -> Option<Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, buffer) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        buffer
    }

    /// Returns the bytes of title memory read, and the buffer it detiled
    /// into with the subresources detiled.
    fn refresh_now(&self) -> (usize, Option<Upload>) {
        let all = || (0..self.footprints.len() as u32).collect::<Vec<_>>();
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            // Most refreshes find nothing: the tracker's dense bit says so
            // without its lock (123,000 refreshes a frame, run evict3).
            if t.clean() {
                return (0, None);
            }
            let ranges = t.take(start, start + size);
            if ranges.is_empty() {
                return (0, None);
            }
            // Only the subresources in the pages that changed (streamed
            // textures change page by page; a whole re-read and detile of
            // each cost up to 430 MB a frame, run integ2).
            let n = self.footprints.len();
            let span = |i: usize| self.spans.get(i).copied().unwrap_or((0, size));
            let touched: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (a, b) = span(i);
                    ranges.iter().any(|&(f, e)| f - start < b && a < e - start)
                })
                .collect();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page. Not kept: it would
            // double each texture's system memory. Only the touched spans
            // are written, and only they are hashed and detiled.
            let mut scratch = Scratch::take(size);
            let read = &mut scratch.0[..size];
            let mut bytes = 0;
            let mut done: Vec<(usize, usize)> = Vec::new();
            for &i in &touched {
                let (a, b) = span(i);
                if !done.contains(&(a, b)) {
                    // SAFETY: inside the buffer of the texture's size.
                    crate::host::copy_mapped(unsafe { read.as_mut_ptr().add(a) }, self.memory.wrapping_add(a), b - a, false);
                    (bytes, _) = (bytes + b - a, done.push((a, b)));
                }
            }
            // A changed page that holds only the subresource's bytes says it
            // changed. Pages it shares with other data the title writes
            // (small textures): its own bytes decide. (Hashing a movie frame
            // cost 8 ms a frame.)
            let page = crate::watch::PAGE;
            last.hashes.resize(n, None);
            let fresh: Vec<u32> = touched
                .into_iter()
                .filter(|&i| {
                    let (a, b) = span(i);
                    let (a0, b0) = (start + a, start + b);
                    if ranges.iter().any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0)) {
                        last.hashes[i] = None;
                        return true;
                    }
                    let hash = blake3::hash(&read[a..b]);
                    last.hashes[i].replace(hash) != Some(hash)
                })
                .map(|i| i as u32)
                .collect();
            if fresh.is_empty() {
                return (bytes, None);
            }
            let buffer = self.detile(read, &fresh);
            if buffer.is_none() {
                for &i in &fresh {
                    last.hashes[i as usize] = None;
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, bytes as u64);
            return (bytes, buffer.map(|(b, at)| (b, at, fresh)));
        }
        // Compared with the bytes last detiled, which tracked memory does
        // not keep. Hashing them instead ran at 0.45 GB/s under Wine on
        // Rosetta (blake3's AVX2 code, emulated), where no memory is
        // tracked: 30-250 ms per frame; the comparison runs at memory speed.
        // Read through the kernel, which refuses an unmapped page (streamed
        // textures may be unmapped until their pages arrive) instead of
        // faulting: asking `VirtualQuery` first cost about 2 ms in the
        // title's big pools (see `crate::host::copy_mapped`).
        let memory = self.memory;
        let Some(runs) = COMPARE_SCRATCH.with_borrow_mut(|scratch| {
            scratch.resize(COMPARE_CHUNK, 0);
            read_changed(last, size, scratch, |at, to| {
                crate::host::read_into(memory.wrapping_add(at), to)
            })
        }) else {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not readable"
            );
            // Never uploaded: zeros, as a committed texture starts (one in
            // a heap of textures starts with whatever the range held).
            if last.uploaded {
                return (0, None);
            }
            let zeros = vec![0u8; size];
            let every = all();
            let buffer = self.detile(&zeros, &every);
            last.uploaded = buffer.is_some();
            return (0, buffer.map(|(b, at)| (b, at, every)));
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if runs.is_empty() {
            return (size, None);
        }
        // Only the subresources the changed runs touch, as the tracked path
        // above does from the tracker's pages: the others would detile and
        // upload the bytes their host texture already holds. (Runs in bytes
        // no subresource lies in touch nothing, and detile nothing.)
        let n = self.footprints.len();
        let span = |i: usize| self.spans.get(i).copied().unwrap_or((0, size));
        let fresh: Vec<u32> = (0..n)
            .filter(|&i| {
                let (a, b) = span(i);
                touches(&runs, a, b)
            })
            .map(|i| i as u32)
            .collect();
        let buffer = if fresh.is_empty() { None } else { self.detile(&last.read, &fresh) };
        last.uploaded = last.uploaded || buffer.is_some();
        if buffer.is_none() {
            // Read again (and detiled) at the next refresh.
            last.valid = false;
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, buffer.map(|(b, at)| (b, at, fresh)))
    }

    /// Detiles `subresources` of `tiled` (the texture's memory) into a
    /// staging buffer (`None` if none could be had or a subresource could
    /// not be read).
    fn detile(&self, tiled: &[u8], subresources: &[u32]) -> Option<(ID3D12Resource, u64)> {
        let (buffer, at, data) = staging_space(self.size)?;
        #[cfg(feature = "diagnostics")]
        // Scanned only when its message is wanted: a pass over every
        // texture read, in timing runs too.
        if tracing::enabled!(tracing::Level::DEBUG) {
            if tiled.iter().all(|&b| b == 0) {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            } else {
                tracing::debug!(memory = ?self.memory, desc = ?self.layout.desc, "a sampled texture detiled");
            }
        }
        for &i in subresources {
            let (footprint, rows) = (&self.footprints[i as usize], self.rows[i as usize]);
            let (mip, slice) = (i % self.mips, i / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return None;
            }
        }
        Some((buffer, at))
    }
}

/// A buffer for the title memory a refresh reads, kept by the thread for
/// its next refresh: a fresh one for each texture cost a zero-fill and a
/// page fault per 4 KiB (129,000 faults in a frame, run placed1).
struct Scratch(Vec<u8>);

thread_local! {
    static SCRATCH: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

impl Scratch {
    /// The thread's buffer, at least `size` bytes.
    fn take(size: usize) -> Self {
        let mut v = SCRATCH.take();
        if v.len() < size {
            v.resize(size, 0);
        }
        Scratch(v)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The largest textures' buffers are not kept.
        const KEEP: usize = 32 << 20;
        if self.0.len() <= KEEP {
            SCRATCH.set(std::mem::take(&mut self.0));
        }
    }
}

/// Bytes of title memory read at a time to compare it: small enough to stay
/// in the cache between the read and the comparison, large enough that a
/// texture takes few kernel calls.
const COMPARE_CHUNK: usize = 1 << 20;

thread_local! {
    /// Where [`read_changed`] reads a chunk (`COMPARE_CHUNK` bytes).
    static COMPARE_SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Reads the `size` bytes of title memory behind `read` (`read(at, to)` fills
/// `to` with the bytes from `at` on, false if a page of them is not
/// readable) into `last.read`, the bytes last detiled, if they differ from
/// them: whether they did, or none if a page was not readable (the texture
/// then keeps what it was last given).
///
/// The memory is read a `scratch` at a time and compared with `last.read`
/// there, so memory that did not change is read once, as comparing it alone
/// reads it, and not copied. The answer is the runs of bytes that differ
/// (they need not be maximal across the chunks), kept up to date in
/// `last.read`; a memory not read whole yet is one run of everything. A
/// read that fails part way leaves `last.read` half new: it stays invalid
/// until all of the memory has been read, which then counts as a change
/// however the bytes compare.
fn read_changed(
    last: &mut Last,
    size: usize,
    scratch: &mut [u8],
    read: impl Fn(usize, &mut [u8]) -> bool,
) -> Option<Vec<(usize, usize)>> {
    last.read.resize(size, 0);
    if !last.valid {
        // Not the memory's yet (never read, or a read failed part way):
        // read it all, which counts as a change however the bytes compare.
        if !read(0, &mut last.read) {
            return None;
        }
        last.valid = true;
        return Some(vec![(0, size)]);
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < size {
        let n = scratch.len().min(size - at);
        let chunk = &mut scratch[..n];
        if !read(at, chunk) {
            // Where no differing byte was kept, `last.read` is still the
            // memory's; where one was, it is half new, and not trusted.
            last.valid = runs.is_empty();
            return None;
        }
        let kept = &mut last.read[at..at + n];
        if *kept == *chunk {
            // The usual case: one slice comparison, no byte walk.
            at += n;
            continue;
        }
        let mut i = 0;
        while i < n {
            if kept[i] != chunk[i] {
                let first = i;
                while i < n && kept[i] != chunk[i] {
                    i += 1;
                }
                runs.push((at + first, i - first));
                kept[first..i].copy_from_slice(&chunk[first..i]);
            } else {
                i += 1;
            }
        }
        at += n;
    }
    Some(runs)
}

/// Whether any changed run of a texture's memory (offset, length) touches
/// the subresource's bytes `[a, b)`.
fn touches(runs: &[(usize, usize)], a: usize, b: usize) -> bool {
    runs.iter().any(|&(at, len)| at < b && a < at + len)
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
pub(crate) fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let spans = spans(&layout, count, u32::from(d.MipLevels));
    let s = Arc::new(Staging {
        size: total,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        spans,
        last: Mutex::default(),
        uploads: Default::default(),
        copied: Default::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    /// `read_changed` over `memory` in chunks of 1000 bytes, the bytes from
    /// `hole` on unreadable (a read of them copies what comes before it and
    /// fails, as the kernel's does): whether the memory changed (any run),
    /// and the bytes read.
    fn changed(last: &mut Last, memory: &[u8], hole: usize) -> (Option<bool>, usize) {
        let read_bytes = std::cell::Cell::new(0);
        let answer = read_changed(last, memory.len(), &mut [0u8; 1000], |at, to| {
            let n = hole.saturating_sub(at).min(to.len());
            to[..n].copy_from_slice(&memory[at..at + n]);
            read_bytes.set(read_bytes.get() + n);
            n == to.len()
        });
        (answer.map(|runs| !runs.is_empty()), read_bytes.get())
    }

    #[test]
    fn untracked_memory_reads_as_changed_once_per_change() {
        let mut last = Last::default();
        let mut memory: Vec<u8> = (0..4500).map(|i| i as u8).collect();
        let all = usize::MAX;
        assert_eq!(
            changed(&mut last, &memory, all),
            (Some(true), 4500),
            "never detiled"
        );
        assert_eq!(last.read, memory);
        // Read once either way, whether or not it changed and wherever: first
        // chunk, last byte of one, first of the next, middle, tail.
        assert_eq!(changed(&mut last, &memory, all), (Some(false), 4500));
        for at in [0, 999, 1000, 2500, 4499] {
            memory[at] ^= 0xff;
            assert_eq!(
                changed(&mut last, &memory, all),
                (Some(true), 4500),
                "at {at}"
            );
            assert_eq!(last.read, memory);
            assert_eq!(changed(&mut last, &memory, all), (Some(false), 4500));
        }
    }

    #[test]
    fn untracked_memory_reports_the_runs_that_changed() {
        fn read(last: &mut Last, memory: &[u8]) -> Vec<(usize, usize)> {
            let mut scratch = [0u8; 1000];
            read_changed(last, memory.len(), &mut scratch, |at, to| {
                to.copy_from_slice(&memory[at..at + to.len()]);
                true
            })
            .unwrap()
        }
        let mut last = Last::default();
        let mut memory = vec![0u8; 4500];
        // Memory not read whole yet: everything counts as changed, once.
        assert_eq!(read(&mut last, &memory), vec![(0, 4500)]);
        assert_eq!(read(&mut last, &memory), Vec::<(usize, usize)>::new());
        // A run of two bytes, and a lone one at the end of the memory.
        memory[1] = 1;
        memory[2] = 2;
        memory[4499] = 3;
        assert_eq!(read(&mut last, &memory), vec![(1, 2), (4499, 1)]);
        assert_eq!(last.read, memory);
        // A run stops at the scratch's edge; the intersection does not
        // need runs maximal.
        memory[999] = 9;
        memory[1000] = 10;
        assert_eq!(read(&mut last, &memory), vec![(999, 1), (1000, 1)]);
        assert_eq!(last.read, memory);
        // Only the runs' bytes are kept from the read; the rest keep theirs.
        assert_eq!(read(&mut last, &memory), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn changed_runs_touch_only_the_subresources_they_overlap() {
        let runs = [(10usize, 5usize), (100, 1)];
        assert!(touches(&runs, 0, 16));
        assert!(touches(&runs, 14, 20));
        assert!(!touches(&runs, 15, 20));
        assert!(touches(&runs, 0, 200));
        assert!(!touches(&runs, 101, 500));
        assert!(!touches(&runs, 99, 100));
        assert!(!touches(&[], 0, 500));
    }

    #[test]
    fn a_page_that_cannot_be_read_keeps_the_last_bytes_unless_it_overwrote_them() {
        let mut last = Last::default();
        let mut memory = vec![7u8; 4500];
        // Never read: nothing to keep, and nothing detiled.
        assert_eq!(changed(&mut last, &memory, 3000).0, None);
        assert!(!last.valid);
        assert_eq!(changed(&mut last, &memory, 4500).0, Some(true));
        // Unchanged up to the page: what was detiled is still there.
        assert_eq!(changed(&mut last, &memory, 3000).0, None);
        assert!(last.valid);
        assert_eq!(last.read, memory);
        // Changed before the page: `last.read` is half new, and not trusted.
        memory[10] = 8;
        assert_eq!(changed(&mut last, &memory, 3000).0, None);
        assert!(!last.valid);
        // Readable again, the memory as it was then: detiled whatever the
        // bytes of `last.read` say, and once.
        assert_eq!(changed(&mut last, &memory, 4500), (Some(true), 4500));
        assert_eq!(last.read, memory);
        assert_eq!(changed(&mut last, &memory, 4500).0, Some(false));
    }

    #[test]
    fn kernel_reads_refuse_the_pages_committed_refuses() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };
        const PAGE: usize = 4096;
        // SAFETY: reserving and committing three private pages for the test,
        // freed below.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                3 * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!base.is_null());
        // SAFETY: the three pages are ours.
        unsafe { base.write_bytes(7, 3 * PAGE) };
        let protect = |protection| {
            let mut old = 0;
            // SAFETY: the middle page of the allocation above.
            let done = unsafe { VirtualProtect(base.add(PAGE).cast(), PAGE, protection, &mut old) };
            assert_ne!(done, 0);
        };
        let read = |at, to: &mut [u8]| crate::host::read_into(base.wrapping_add(at), to);
        let mut last = Last::default();
        let mut scratch = [0u8; PAGE];
        protect(PAGE_NOACCESS);
        assert!(!committed(base, 3 * PAGE));
        assert_eq!(read_changed(&mut last, 3 * PAGE, &mut scratch, read), None);
        protect(PAGE_READWRITE);
        assert!(committed(base, 3 * PAGE));
        assert_eq!(
            read_changed(&mut last, 3 * PAGE, &mut scratch, read),
            Some(vec![(0, 3 * PAGE)])
        );
        assert_eq!(read_changed(&mut last, 3 * PAGE, &mut scratch, read), Some(Vec::new()));
        // Out of reach again: unchanged as far as can be told.
        protect(PAGE_NOACCESS);
        assert_eq!(read_changed(&mut last, 3 * PAGE, &mut scratch, read), None);
        assert!(last.valid);
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
