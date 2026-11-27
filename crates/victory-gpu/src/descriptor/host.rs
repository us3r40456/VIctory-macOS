//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::slots::{Made, Pieces};
use super::{VIEW_TAG, View, ViewPin, view_at, views_current_into, views_forgot};
use crate::host::host;

mod quick;
#[cfg(test)]
mod split_tests;

const SYNC_BLOCK_SLOTS: u32 = 128;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from,
    /// and the lock a slot is made under: one for the heap, or (mac:
    /// `VICTORY_HEAP_LOCKS=1`, `super::slots`) one of many by the slot.
    made: Made,
    /// mac: what heap syncs ahead of a submission left for a later sync
    /// (`super::ahead`): slots found rewritten whose views are left for a
    /// read to make ([`HostHeap::mirror`], for the sync before the
    /// submission), and runs of title heap memory a round took as written
    /// and did not look at (for the next sync of any kind).
    left: Mutex<Left>,
    /// Per `SYNC_BLOCK_SLOTS` block, the `views_forgot()` epoch last verified
    /// (VIctory-macOS #279's quick look, `VICTORY_HEAP_FAST_SKIP=1`; the
    /// records themselves are atomics here, so its bit per made slot is
    /// [`Made::peek`]).
    block_forgot: Box<[AtomicU64]>,
    /// Per title slot, its record once the slot is made from it and holds
    /// what a draw reads (a texture's view not left null), else 0: what
    /// [`HostHeap::mirror`] finds without the `made` lock, which every
    /// title thread binding or writing views would otherwise wait on. Set
    /// last, under that lock, once the slot is made; cleared first.
    ready: Box<[AtomicU64]>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one. Paged:
    /// the title views through a fraction of the slots, and a dense table
    /// of them all was 37.7 MB, written when the heap was made. Behind one
    /// lock, or (mac: `VICTORY_HEAP_LOCKS=1`) in pieces with a lock each.
    kept: Pieces<Held>,
    /// Unordered views of textures left null by [`HostHeap::evict`]
    /// (`VICTORY_EVICT_HEAP_UAV`), made again by
    /// [`HostHeap::remake_unordered`] once shaders write through heaps.
    nulled_unordered: Mutex<Vec<u32>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// Null raw SRV for explicitly bound root descriptors. Its UAV is one slot below.
    null_root_srv: Option<u32>,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per title slot and per dynamic slot (tables made afresh lie there,
    /// and shaders index the view addresses at the slots their tables are
    /// in), the console descriptor's dwords 0-3 as far as the host knows
    /// them (shaders reading a descriptor as data, `crate::memory`;
    /// [`write_address`]): the buffer, its mapping and host address, made on
    /// first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
    /// Live widened SRV buffers, by host address, with the number of slots
    /// showing each. Heap-indexing shaders can read any of them.
    widened: Mutex<HashMap<usize, (usize, std::sync::Weak<crate::shadow::Widened>)>>,
}

/// mac: slots, and runs of title heap memory (`HostHeap::left`).
type Left = (Vec<u32>, Vec<(usize, usize)>);

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        Self::new_as(sampler, count, super::slots::mode().split)
    }

    /// [`HostHeap::new`], its slots under locks of their own or (not
    /// `split`) under the heap's one lock (mac: `super::slots`).
    pub(crate) fn new_as(sampler: bool, count: u32, split: bool) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // Other threads' submissions held off from here until the heap is
        // filled (`Host::quiet`; a title heap's `host` holds them already).
        let _quiet = (!sampler).then(|| host.quiet()).flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below `HEAP_VIEWS` or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let (views, samplers) = (victory_shader::emit::HEAP_VIEWS, victory_shader::emit::HEAP_SAMPLERS);
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            // shortcut: a host that caps its heaps below `HEAP_VIEWS` (NVIDIA)
            // faults on an index past them, as before the clamp.
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let mut h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Made::new(count as usize, split),
            block_forgot: (0..(count as usize).div_ceil(SYNC_BLOCK_SLOTS as usize))
                .map(|_| AtomicU64::new(0))
                .collect(),
            left: Mutex::default(),
            ready: (0..count).map(|_| AtomicU64::new(0)).collect(),
            kept: Pieces::new((count + extra) as usize, split),
            nulled_unordered: Mutex::new(Vec::new()),
            next: Mutex::new((0, vec![false; extra as usize])),
            pairs: Mutex::new((count + extra, Vec::new())),
            null_root_srv: None,
            pairs_end: match victory_shader::emit::NULL_BUFFER_VIEW {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
            widened: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        if !sampler {
            let fixed = victory_shader::emit::NULL_BUFFER_VIEW;
            let null = if fixed > count + extra && fixed < total {
                Some(fixed)
            } else {
                tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
                // Root descriptors receive a heap index at bind time and can
                // use a pair inside a capped heap. Keep it for the heap's life.
                h.pair().map(|first| first + 1)
            };
            if let Some(srv) = null {
                h.raw_view_at(srv, None, 0, 0, false);
                h.raw_view_at(srv - 1, None, 0, 0, true);
                h.null_root_srv = Some(srv);
            }
        }
        Some(h)
    }

    pub(crate) fn null_root_view(&self, uav: bool) -> Option<u32> {
        self.null_root_srv.map(|srv| srv - u32::from(uav))
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        // A table made afresh: shaders reading a view's address there (e.g.
        // the BVH builder's base) index the view addresses at this slot.
        self.note_address(index, Some(view));
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None);
        self.set_mark(index, None);
        self.note_address(index, None);
        if index < self.count {
            self.viewed.set(index, None, false);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the integer-format
    /// bits of a typed buffer view ([`view_kind`]).
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, |v| view_stride(v) | view_kind(v)));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let split = self.kept.split();
        let old = self.kept.with(index as usize, |kept, at| {
            // A slot that held nothing and gets nothing is left alone (no page).
            let slot = kept.slot(at, resource.is_some())?;
            let old_widened = slot.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let new_widened = resource.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let old = std::mem::replace(slot, resource);
            self.note_widened(old_widened.as_ref(), false);
            self.note_widened(new_widened.as_ref(), true);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                use windows::core::Interface;
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
            }
            // mac: in pieces, the old one is retired once the piece's lock
            // is let go (below); behind the one lock, under it, as before.
            if split {
                return old;
            }
            // Title threads rewrite slots while the GPU runs a submission;
            // one of its shaders may hold the old descriptor (or read the
            // slot at an index from wrong data): what it views is freed
            // only once that work has run.
            if let Some((old, _)) = old {
                crate::host::retire(old);
            }
            None
        });
        if let Some((old, _)) = old {
            crate::host::retire(old);
        }
    }

    fn note_widened(&self, view: Option<&std::sync::Arc<crate::shadow::Widened>>, add: bool) {
        let Some(view) = view else { return };
        let key = std::sync::Arc::as_ptr(view) as usize;
        let mut widened = self.widened.lock().unwrap_or_else(|e| e.into_inner());
        if add {
            let entry = widened.entry(key).or_insert_with(|| (0, std::sync::Arc::downgrade(view)));
            entry.0 += 1;
        } else if let Some((count, _)) = widened.get_mut(&key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                widened.remove(&key);
            }
        }
    }

    /// Widened SRV buffers a heap-indexing shader may read.
    pub(crate) fn widened_buffers(&self) -> Vec<ID3D12Resource> {
        self.widened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|(_, view)| view.upgrade().map(|view| view.buffer.clone()))
            .collect()
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            // The view is a root view, which has no bounds: an entry for
            // every slot a table can lie in, the dynamic ones too (a table
            // made afresh on a host whose heap is smaller than the title's
            // read past the title slots' entries: whatever followed the
            // buffer, or a GPU fault).
            let (buffer, data) = host()?.upload_buffer(u64::from(self.address_slots()) * 16)?;
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            // (mac: every slot's lock, as the one lock was: no slot is made
            // between its record being read here and the buffer being there
            // for `note_address`.)
            let made = self.made.lock_all();
            for i in 0..self.count {
                let key = made.get(i);
                let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
                write_address(data, i as usize, view.as_deref());
            }
            // Dynamic slots are entered as they are made ([`Self::make_at`],
            // [`Self::make_null`], raw views).
            for i in self.count..self.address_slots() {
                write_address(data, i as usize, None);
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// The slots the view addresses have an entry for: the title's and the
    /// dynamic ones.
    fn address_slots(&self) -> u32 {
        self.count + self.dynamic
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if index < self.address_slots()
            && let Some(Some((_, data, _))) = self.addresses.get()
        {
            write_address(*data, index as usize, view);
        }
    }

    /// The view address entered for slot `index`; none for a slot past the
    /// view addresses' buffer (tests).
    #[cfg(test)]
    fn address_at(&self, index: u32) -> Option<u64> {
        let (buffer, data, _) = self.addresses.get()?.as_ref()?;
        // SAFETY: a live buffer.
        if 16 * u64::from(index) + 16 > unsafe { buffer.GetDesc() }.Width {
            return None;
        }
        // SAFETY: an entry of the mapped view addresses (checked).
        let [lo, hi] = unsafe { ((data + 16 * index as usize) as *const [u32; 2]).read_unaligned() };
        Some(u64::from(hi) << 32 | u64::from(lo))
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
        self.note_address(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Whether slot `index` is made from a record (VIctory-macOS #279's bit
    /// per slot; here the records are atomics, read without a lock).
    fn made_is_nonzero(&self, index: u32) -> bool {
        self.made.peek(index).is_some_and(|m| m != 0)
    }

    /// A heap sync of slots `first..last`, whose records are at `start` +
    /// index × `step`: [`HostHeap::mirror`] of each slot whose record
    /// differs from what the slot holds, the rest passed by as `mirror`
    /// would, with one lock of each kind for them all instead of two per slot
    /// (a world frame looks at ~700,000 slots, run perf3: ~75 ms).
    pub(crate) fn sync_slots(&self, first: u32, last: u32, start: u64, step: u64) {
        let stale = self.stale_slots(first, last, start, step);
        self.make_stale(stale, start, step);
    }

    /// Like [`HostHeap::sync_slots`], with VIctory-macOS #279's look
    /// ([`HostHeap::stale_slots_bytes`]): records that lie inside `buf`
    /// (`[buf_start, buf_start + buf.len())`) are read from it, and empty
    /// and already-verified slots are passed by without the view lock when
    /// no views were forgotten. On this line only with
    /// `VICTORY_HEAP_FAST_SKIP=1` (`super::fast_skip`).
    pub(crate) fn sync_slots_bytes(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        buf_start: u64,
        buf: &[u8],
    ) {
        let last = last.min(self.count);
        if first >= last {
            return;
        }
        let forgot = views_forgot();
        let stale = self.stale_slots_bytes(first, last, start, step, buf_start, buf, forgot);
        self.make_stale(stale, start, step);
        self.verified(first, last, forgot);
    }

    /// The look of [`HostHeap::sync_slots_bytes`] (`first < last <= count`;
    /// `forgot`: [`views_forgot`] read before the look).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stale_slots_bytes(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        buf_start: u64,
        buf: &[u8],
        forgot: u64,
    ) -> Vec<u32> {
        let at = |i: u32| start + u64::from(i) * step;
        let b_first = first / SYNC_BLOCK_SLOTS;
        let b_last = (last - 1) / SYNC_BLOCK_SLOTS;
        let blocks_verified = (b_first..=b_last)
            .all(|b| self.block_forgot[b as usize].load(Ordering::Acquire) == forgot);

        let buf_end = buf_start.saturating_add(buf.len() as u64);
        let mut stale: Vec<u32> = Vec::new();
        let mut candidates: Vec<(u32, [u32; 3])> = Vec::new();

        for i in first..last {
            let addr = at(i);
            let record = if addr >= buf_start && addr.saturating_add(12) <= buf_end {
                let off = (addr - buf_start) as usize;
                // SAFETY: `off + 12 <= buf.len()`, unaligned copy of 12 bytes.
                unsafe { buf.as_ptr().add(off).cast::<[u32; 3]>().read_unaligned() }
            } else {
                // SAFETY: record in title heap memory the caller found mapped.
                unsafe { (addr as *const [u32; 3]).read_unaligned() }
            };
            let [tag, view, generation] = record;
            if tag != VIEW_TAG {
                if self.made_is_nonzero(i) {
                    stale.push(i);
                }
                continue;
            }
            let key = u64::from(generation) << 32 | u64::from(view);
            if blocks_verified && self.ready[i as usize].load(Ordering::Acquire) == key {
                continue;
            }
            candidates.push((i, record));
        }

        if !candidates.is_empty() {
            let records: Vec<[u32; 3]> = candidates.iter().map(|&(_, r)| r).collect();
            let mut current = Vec::with_capacity(records.len());
            views_current_into(&records, &mut current);
            // mac: behind the one lock the look is under it; with locks of
            // their own the records are read without one (`stale_slots`).
            let _held = (!self.made.split()).then(|| self.made.lock(first));
            for (&(i, [_, view, generation]), is_current) in candidates.iter().zip(current) {
                if super::slots::stale(is_current, view, generation, self.made.peek(i)) {
                    stale.push(i);
                }
            }
        }
        // In slot order, as `stale_slots` gives them.
        stale.sort_unstable();
        stale
    }

    /// After a look of `first..last` and the making of what it found: the
    /// blocks wholly inside are verified as of `forgot`.
    pub(crate) fn verified(&self, first: u32, last: u32, forgot: u64) {
        if first >= last {
            return;
        }
        let b_first = first / SYNC_BLOCK_SLOTS;
        let b_last = (last - 1) / SYNC_BLOCK_SLOTS;
        for b in b_first..=b_last {
            let b_start = b * SYNC_BLOCK_SLOTS;
            let b_end = ((b + 1) * SYNC_BLOCK_SLOTS).min(self.count);
            if first <= b_start && last >= b_end {
                self.block_forgot[b as usize].store(forgot, Ordering::Release);
            }
        }
    }

    /// The look of [`HostHeap::sync_slots`]: the slots of `first..last`
    /// whose record differs from what the slot holds (mac: apart, for the
    /// syncs that compare what they find with what was told,
    /// `super::reports`).
    pub(crate) fn stale_slots(&self, first: u32, last: u32, start: u64, step: u64) -> Vec<u32> {
        let at = |i: u32| start + u64::from(i) * step;
        // SAFETY: records in title heap memory the caller found mapped.
        let records: Vec<[u32; 3]> =
            (first..last).map(|i| unsafe { (at(i) as *const [u32; 3]).read_unaligned() }).collect();
        let mut current = Vec::with_capacity(records.len());
        views_current_into(&records, &mut current);
        let stale: Vec<u32> = {
            // mac: behind the one lock the look is under it, as before.
            // With locks of their own the slots' records are read without
            // one: a slot found stale is looked at again under its lock
            // (`mirror`), and one found made from its record is made from
            // it or being made from it by a thread that holds its lock
            // until it is (`mirror` notes the record and makes the view in
            // one hold).
            let _held = (!self.made.split()).then(|| self.made.lock(first));
            (first..last)
                .zip(records.iter().zip(current))
                .filter(|&(i, (&[_, view, generation], current))| {
                    super::slots::stale(current, view, generation, self.made.peek(i))
                })
                .map(|(i, _)| i)
                .collect()
        };
        stale
    }

    /// The making of [`HostHeap::sync_slots`]: [`HostHeap::mirror`] of each
    /// slot of `stale`, in order.
    pub(crate) fn make_stale(&self, stale: Vec<u32>, start: u64, step: u64) {
        let at = |i: u32| start + u64::from(i) * step;
        if super::ahead::on() {
            super::ahead::changed(stale.len());
        }
        // mac: the changed slots of a pass made on several threads, where
        // slots have locks of their own (`super::pool`).
        if self.made.split()
            && let Some(pool) = super::pool::get()
        {
            let wide = stale.len() >= super::pool::WORTH;
            super::pool::note(stale.len(), wide);
            if wide {
                return self.sync_wide(pool, &stale, &at);
            }
        }
        // mac: a view heap's buffer shader views from one lookup per record.
        if !self.sampler && quick::on() {
            return self.sync_quick(&stale, at);
        }
        for i in stale {
            self.mirror(i, at(i), false);
        }
    }

    /// Whether the view made in slot `index` must be made again for a draw
    /// that reads it: a placed texture's view left null is made when a draw
    /// reads the slot through a table (heap syncs revisit it at every
    /// submission), a texture's unordered view left null when the texture was
    /// given up (`VICTORY_EVICT_HEAP_UAV`) is made again like a shader view (a
    /// table copying the null slot loses writes), and a later unordered or
    /// renderable view that upgraded the memory's `HostTexture` leaves the
    /// made view on the old one.
    fn needs_remake(&self, index: u32, view: &View, for_draw: bool) -> bool {
        if !for_draw || !(is_texture(view) || is_unordered_texture(view)) {
            return false;
        }
        let held = self.kept.get(index as usize, |held| held.map(|(res, _)| windows::core::Interface::as_raw(res)));
        let Some(held) = held else {
            return true;
        };
        is_placed_texture(view)
            && placed_texture(view).is_some_and(|p| {
                let (View::Shader(r, _) | View::Unordered(r, ..)) = p else {
                    return false;
                };
                !crate::resource::host_copy(r as crate::object::This).is_some_and(|copy| {
                    windows::core::Interface::as_raw(&copy.texture) == held
                        || copy.twin.get().and_then(Option::as_ref).is_some_and(|t| {
                            t.planes
                                .iter()
                                .any(|p| windows::core::Interface::as_raw(&p.texture) == held)
                        })
                })
            })
    }

    /// mac: what [`HostHeap::sync_slots`] does with the slots of `stale`,
    /// on the threads of `pool` and this one: the slots in parts, each part
    /// made in order by one thread the way the pass alone makes them (the
    /// quick way or `mirror`), the parts side by side. A slot is made under
    /// its own lock whichever thread makes it, so every slot ends as the
    /// pass alone leaves it; only the order between parts is free, as it is
    /// between this pass and the title's threads making slots meanwhile.
    ///
    /// A thread of the pool works in this thread's name: in a round ahead
    /// of a submission if this is one (it leaves the slots a round leaves),
    /// as a holder of the submission lock if this thread holds it, and
    /// counted as a heap sync.
    fn sync_wide(&self, pool: &super::pool::Pool, stale: &[u32], at: &(impl Fn(u32) -> u64 + Sync)) {
        // A few parts a thread: slots differ in what they cost to make.
        let part = stale.len().div_ceil(pool.threads() * 4).max(16);
        let parts: Vec<&[u32]> = stale.chunks(part).collect();
        let (round, holds) = (super::ahead::running(), crate::host::holds_submissions());
        let quick = !self.sampler && quick::on();
        pool.run(parts.len(), &|k| {
            let _syncing = super::slots::Syncing::enter();
            super::ahead::as_round(round, || {
                crate::host::as_holder(holds, || {
                    if quick {
                        self.sync_quick(parts[k], at);
                    } else {
                        for &i in parts[k] {
                            self.mirror(i, at(i), false);
                        }
                    }
                })
            });
        });
    }

    /// mac: `runs` (title heap memory a sync took as written), after the
    /// runs a round ahead took and did not look at.
    pub(crate) fn with_left(&self, runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        let mut left = std::mem::take(&mut self.left.lock().unwrap_or_else(|e| e.into_inner()).1);
        if left.is_empty() {
            return runs;
        }
        left.extend(runs);
        left
    }

    /// mac: a round ahead ends with `runs` taken and not looked at.
    pub(crate) fn leave(&self, runs: impl Iterator<Item = (usize, usize)>) {
        self.left.lock().unwrap_or_else(|e| e.into_inner()).1.extend(runs);
    }

    /// mac: the sync before a submission, of the slots rounds ahead of it
    /// left ([`HostHeap::mirror`]): each made from the record it holds now,
    /// as a sync that had found it rewritten would. A slot whose heap page
    /// the title has unmapped since holds no record (the sync asks the same
    /// of every page it looks at), and one left by several rounds is made
    /// once.
    pub(crate) fn sync_left(&self, start: u64, step: u64) {
        let mut left = std::mem::take(&mut self.left.lock().unwrap_or_else(|e| e.into_inner()).0);
        left.sort_unstable();
        left.dedup();
        super::ahead::changed(left.len());
        for i in left {
            let at = start + u64::from(i) * step;
            if crate::host::read_into(at as *const u8, &mut [0u8; 4]) {
                self.mirror(i, at, false);
            }
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there, pinned for
    /// as long as the caller uses it. Buffer views are made only for draws
    /// (`for_draw`), which also copy their data: a host copy of a buffer is
    /// made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<ViewPin> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        // Made from this record already and readable as it is (what the
        // checks below find under the lock): nothing to make. Not for a
        // draw's read of a placed texture: its host texture may have been
        // replaced since (`needs_remake` below), which only the slow path
        // looks at.
        if tag == VIEW_TAG
            && self.ready[index as usize].load(Ordering::Acquire) == key
            && let Some(view) = view_at(i, generation)
            && !(for_draw && is_placed_texture(&view))
        {
            return Some(view);
        }
        let mut made = self.made.lock(index);
        // A second recorder may have replaced the slot while this thread waited.
        // SAFETY: the caller holds the heap alive until the update completes.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made.get(index) != 0 {
                self.ready[index as usize].store(0, Ordering::Relaxed);
                made.set(index, 0);
                super::slots::made();
                self.make_null(index);
                self.note_address(index, None);
            }
            return None;
        };
        // A placed texture's view left null below is made when a draw reads
        // the slot through a table; until then the slot is not made again
        // (heap syncs revisit it at every submission). Also re-mirror if a
        // later unordered or renderable view upgraded the memory's HostTexture.
        let needs_remake = || self.needs_remake(index, &view, for_draw);
        if made.get(index) == key && !needs_remake() {
            return Some(view);
        }
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        // shortcut: views of textures the title placed itself are made
        // when a draw reads them through a table: a host texture (and a
        // staging copy) for each of the ~20,000 the title has views of
        // ran the host out of memory (run b2).
        let later = !for_draw && is_placed_texture(&view);
        // mac: a heap sync ahead of the submission leaves the slot as it is,
        // for the sync before it: made now, its read mark would be forgotten
        // under the running submission's shaders (`super::ahead`).
        if later && super::ahead::running() {
            self.left.lock().unwrap_or_else(|e| e.into_inner()).0.push(index);
            super::ahead::left();
            return Some(view);
        }
        if !later {
            // What the view needs (the texture it views, the copy of the
            // memory) is made without the slot lock, which title threads
            // binding and writing views wait on (run evict5).
            drop(made);
            prepare(&view);
            made = self.made.lock(index);
            // The title may have written another record there meanwhile (and
            // another thread made it): that one is the slot's.
            // SAFETY: as above.
            if unsafe { (handle as *const [u32; 3]).read_unaligned() } != [tag, i, generation] {
                drop(made);
                drop(view);
                return self.mirror(index, handle, for_draw);
            }
            if made.get(index) == key && !needs_remake() {
                return Some(view);
            }
        }
        {
            self.ready[index as usize].store(0, Ordering::Relaxed);
            self.set_stride(index, Some(&view));
            self.note_address(index, Some(&view));
            if later {
                // Made once a shader reads it (`crate::shadow::take_reads`).
                crate::shadow::forget_read(index);
            }
            made.set(index, key);
            super::slots::made();
            let (kept, mark) = if later {
                (None, None)
            } else {
                make_marked(&view, self.cpu(index))
            };
            self.set_mark(index, mark);
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            if !self.sampler {
                self.viewed.set(index, Some(&view), kept.is_some());
            }
            let kept = kept.flatten();
            // A texture's view left null is made again when a draw reads it.
            let ready = if is_texture(&view) && kept.is_none() { 0 } else { key };
            self.keep(index, kept);
            self.ready[index as usize].store(ready, Ordering::Release);
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made.lock(index).find(index)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.get(index as usize, |held| held.is_some());
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Title slots viewing one of the host textures `textures` (raw pointers),
    /// which are given up: each gets a null view of its kind and holds the
    /// texture no more (retired until the running submission has run), its
    /// record stays noted so heap syncs leave it, and the next draw reading
    /// it through a table or shader reading it through the heap (its read
    /// mark, cleared) makes it again. Slots past `count` stay made: recorded
    /// lists and host copies already name them, and there is no title record
    /// from which to make them again.
    /// The slots are found by a walk of the heap, once per pass that gives
    /// textures up: a map of each texture's slots, kept at every slot the
    /// title rewrote, cost ~25 us a slot (a texture viewed in thousands of
    /// slots had its list searched each time; 62,000 slots a frame, run
    /// evict3).
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        // Keep a title slot from being rebound between testing its held texture
        // and clearing the descriptor. Mirroring takes made before kept as well.
        // mac: behind the one lock it is held throughout, as before. With
        // locks of their own each slot found is tested again under its
        // lock, which is then held while its descriptor is cleared.
        let split = self.made.split();
        let _all = (!split).then(|| self.made.lock(0));
        let given_up = |k: &Held| k.1.is_none() && textures.contains(&(windows::core::Interface::as_raw(&k.0) as usize));
        let mut slots: Vec<u32> = Vec::new();
        self.kept.each(|i, k| {
            if i < self.count as usize && given_up(k) {
                slots.push(i as u32);
            }
        });
        for index in slots {
            let _one = split.then(|| self.made.lock(index));
            if split && !self.kept.get(index as usize, |held| held.is_some_and(given_up)) {
                continue;
            }
            if let Some(ready) = self.ready.get(index as usize) {
                ready.store(0, Ordering::Relaxed);
            }
            let key = self.made.peek(index).unwrap_or(0);
            let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
            match &view {
                Some(v) => null_for(v, self.cpu(index)),
                None => null_view(self.cpu(index)),
            }
            self.keep(index, None);
            if index < self.count {
                self.viewed.set(index, view.as_deref(), false);
                crate::shadow::forget_read(index);
                if view.as_deref().is_some_and(is_unordered_texture) {
                    self.nulled_unordered.lock().unwrap_or_else(|e| e.into_inner()).push(index);
                }
            }
        }
    }

    /// Makes again the unordered texture views [`HostHeap::evict`] left
    /// null, the title's heap starting at `start` with `step` bytes a slot:
    /// shaders indexing the heap may write through them now.
    pub(crate) fn remake_unordered(&self, start: u64, step: u64) -> usize {
        let slots = std::mem::take(&mut *self.nulled_unordered.lock().unwrap_or_else(|e| e.into_inner()));
        for &index in &slots {
            self.made.lock(index).set(index, 0);
            self.mirror(index, start + u64::from(index) * step, true);
        }
        slots.len()
    }

    /// The host textures (raw pointers) slots `slots` view.
    pub(crate) fn textures_at(&self, slots: &[u32]) -> Vec<usize> {
        slots
            .iter()
            .filter_map(|&i| {
                self.kept.get(i as usize, |held| {
                    held.filter(|k| k.1.is_none()).map(|k| windows::core::Interface::as_raw(&k.0) as usize)
                })
            })
            .collect()
    }

    /// The view of slot `index` if it is a placed texture's left without a
    /// host view (made when read).
    pub(crate) fn unmade_placed(&self, index: u32) -> Option<View> {
        let key = self.made.lock(index).find(index)?;
        let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten()?;
        let kept = self.kept.get(index as usize, |held| held.is_some());
        (is_placed_texture(&view) && !kept).then(|| (*view).clone())
    }

    /// A shader read slot `index` through the heap: the texture it views
    /// was used now.
    pub(crate) fn stamp_read(&self, index: u32) {
        let key = self.made.lock(index).find(index).unwrap_or(0);
        let Some(view) = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten() else {
            return;
        };
        let placed = placed_texture(&view);
        let current = placed.as_ref().unwrap_or(&view);
        if let View::Shader(resource, _) = current
            && let Some(t) = crate::resource::made_copy(*resource as crate::object::This)
        {
            t.stamp();
        }
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.lock(index).get(index) != 0
    }

    /// Whether slot `index` holds a host view of a resource (tests).
    #[cfg(test)]
    pub(crate) fn kept(&self, index: u32) -> bool {
        self.kept.get(index as usize, |held| held.is_some())
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0)
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// The view-stride table's format-kind bits of a buffer view: a typed view
/// (no structure stride, not raw) of an integer format gets
/// `VIEW_INTEGER`, a signed one also `VIEW_SIGNED`; anything else none.
/// Translated shaders select integer typed-buffer overloads for such views,
/// and pass their 16-bit typed data as bits (`victory_shader::emit`, "d16").
fn view_kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let (unordered, words): (bool, &[u32; 10]) = match view {
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => (*unordered, words),
            Err(_) => return 0,
        },
        _ => return 0,
    };
    // Buffer views only (dimension 1); a structured or raw one is no typed view.
    let (format, typed) = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        (desc.Format.0 as u32, typed_buffer(b.StructureByteStride, raw))
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        (desc.Format.0 as u32, typed_buffer(b.StructureByteStride, raw))
    };
    match (words[1] == 1 && typed).then(|| integer_format(format)).flatten() {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    }
}

/// The four-channel host format for a three-channel 32-bit typed SRV.
fn widened_format(format: i32) -> Option<DXGI_FORMAT> {
    match format {
        6 => Some(DXGI_FORMAT_R32G32B32A32_FLOAT),
        7 => Some(DXGI_FORMAT_R32G32B32A32_UINT),
        8 => Some(DXGI_FORMAT_R32G32B32A32_SINT),
        _ => None,
    }
}

/// The fourth component's bits for a widened three-channel SRV. Integer
/// formats must not receive the bit pattern of floating-point 1.0.
fn widened_w(format: i32) -> Option<u32> {
    match format {
        6 => Some(1.0f32.to_bits()),
        7 | 8 => Some(1),
        _ => None,
    }
}

/// Whether `view` is a typed three-channel 32-bit SRV backed by a widened
/// host copy under Wine.
pub(crate) fn three_channel_srv(view: &View) -> bool {
    let words: &[u32; 10] = match view {
        View::Shader(_, Some(desc)) => desc,
        View::Placed(false, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => words,
            Err(_) => return false,
        },
        _ => return false,
    };
    let desc = srv_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    desc.ViewDimension == D3D12_SRV_DIMENSION_BUFFER
        && matches!(desc.Format.0, 6..=8)
        && typed_buffer(buffer.StructureByteStride, raw)
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

/// Bytes per element of a buffer view: its structure stride if it has one,
/// 4 for a raw view, else its format's element size (`None` if the format
/// has none).
fn buffer_element(stride: u32, raw: bool, format: i32) -> Option<u64> {
    if stride > 0 {
        Some(u64::from(stride))
    } else if raw {
        Some(4)
    } else {
        crate::resource::element(format as u32).map(|(bytes, _)| bytes)
    }
}

/// Whether a buffer view is typed: no structure stride, not raw.
fn typed_buffer(stride: u32, raw: bool) -> bool {
    stride == 0 && !raw
}

/// Bytes per element of a shader resource buffer view, 0 for a format with
/// none.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0)
}

/// Writes slot `index`'s words into the view addresses at `data` (the
/// caller keeps `index` below [`HostHeap::address_slots`]), 16 bytes a
/// slot: a buffer view's title address (dword 0 low, dword 1 bits 0-15
/// high), a texture view's size fields ([`texture_words`]), zeros for
/// anything else.
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let address = view.and_then(buffer_range).map_or(0, |(a, _)| a);
    let words = match view.and_then(texture_words) {
        Some(words) => words,
        None => [address as u32, (address >> 32) as u32 & 0xFFFF, 0, 0],
    };
    // SAFETY: the mapped view addresses hold 16 bytes per title and dynamic
    // slot (below `address_slots`).
    unsafe { ((data + 16 * index) as *mut [u32; 4]).write_unaligned(words) };
}

/// A texture view's console image descriptor dwords 0-3 as far as its size
/// goes: width - 1 and height - 1 of the resource's level 0 (dword 1 bits
/// 30-31, dword 2 bits 0-13 and 14-29), dword 3 an identity swizzle, the
/// view's first and last level (bits 12-15, 16-19) and the image type (bits
/// 28-31). `None` for buffers and samplers.
// shortcut: the address, format and tiling fields read 0 (no translated
// shader reads them: the shader refuses such reads where it can tell),
// and the swizzle is the identity whatever the view's component mapping.
fn texture_words(view: &View) -> Option<[u32; 4]> {
    let (resource, desc, unordered) = match view {
        View::Shader(r, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, false),
        View::Unordered(r, _, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, true),
        View::Placed(unordered, resource, desc) if resource[0] != 1 => {
            (*resource, desc[..10].try_into().ok(), *unordered)
        }
        _ => return None,
    };
    Some(image_words(&resource, desc.as_ref(), unordered))
}

/// [`texture_words`] of a texture of resource description `resource` (as
/// words) through a view of description `desc` (none: the resource's own).
fn image_words(resource: &[u32; 14], desc: Option<&[u32; 10]>, unordered: bool) -> [u32; 4] {
    let width = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let height = resource[6];
    let (depth, mips) = (resource[7] & 0xFFFF, resource[7] >> 16);
    let (samples, dimension) = (resource[9], resource[0]);
    // Mip count 0: the whole chain.
    let deepest = if dimension == 4 { depth } else { 1 };
    let chain = 64 - (width.max(u64::from(height)).max(u64::from(deepest)) | 1).leading_zeros();
    let mips = if mips == 0 { chain } else { mips };
    // Image types: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array,
    // 14 2D MSAA, 15 2D MSAA array.
    let (ty, first, count) = match desc {
        // Shader and unordered view dimensions number alike up to 3D (8).
        Some(d) => {
            let ty = [0, 0, 8, 12, 9, 13, 14, 15, 10, 11, 11].get(d[1] as usize).copied().unwrap_or(0);
            match (unordered, d[1]) {
                (_, 6 | 7) => (ty, 0, 1),
                // An unordered view's one level is its third word (its
                // union starts at byte 8: `MipSlice` of every texture
                // member, as `uav_desc` hands it to the host); a shader
                // view's first level and count are words 4 and 5 (its union
                // starts at byte 16). Word 4 of an unordered view is an
                // array's size or a 3D view's depth.
                (true, _) => (ty, d[2], 1),
                (false, _) => (ty, d[4], if d[5] == u32::MAX { mips.saturating_sub(d[4]) } else { d[5] }),
            }
        }
        None => {
            let array = depth > 1;
            let ty = match dimension {
                2 if array => 12,
                2 => 8,
                3 if samples > 1 => 14 + u32::from(array),
                3 => 9 + 4 * u32::from(array),
                _ => 10,
            };
            (ty, 0, if unordered { 1 } else { mips })
        }
    };
    let (w, h) = (width.saturating_sub(1) as u32, height.saturating_sub(1));
    let last = first + count.max(1) - 1;
    [
        0,
        (w & 3) << 30,
        (w >> 2 & 0x3FFF) | (h & 0xFFFF) << 14,
        0xFAC | (first & 0xF) << 12 | (last & 0xF) << 16 | ty << 28,
    ]
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, u64::from(view_stride(view)))?;
    Some((shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0)?;
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, and the view's own copy
/// of title memory ([`crate::shadow::for_view`]): the slot holds it, so that
/// draws reading the view find and refresh that copy.
type Held = (ID3D12Resource, Option<HeldSource>);

/// The fields keep the normal and widened copies alive for as long as the
/// slot shows them.
#[allow(dead_code)]
enum HeldSource {
    Shadow(std::sync::Arc<crate::shadow::Shadow>),
    /// A widened three-channel view and the normal shadow it reads.
    Widened(
        std::sync::Arc<crate::shadow::Widened>,
        std::sync::Arc<crate::shadow::Shadow>,
    ),
}

/// What a slot made by [`make`] holds, if anything.
type Kept = Option<Held>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let start = address + buffer.FirstElement * element;
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    buffer.FirstElement = shadow.offset(start) / element;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    if crate::under_wine()
        && matches!(desc.Format.0, 6..=8)
        && let Some(w) = widened_w(desc.Format.0)
        && let Some(widened) = shadow.widened(start, u64::from(count) * 12, w)
    {
        desc.Format = widened_format(desc.Format.0).expect("a three-channel format");
        buffer.FirstElement = 0;
        buffer.NumElements = count;
        // SAFETY: a live widened buffer and a slot of our heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&widened.buffer, Some(&desc), at)
        };
        return Some(Some((
            widened.buffer.clone(),
            Some(HeldSource::Widened(widened, shadow.clone())),
        )));
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

/// [`crate::shader_cache::under_wine`], asked once. That function asks the
/// loader for ntdll and for an export of it by name at every call, under
/// Wine's loader lock, and [`buffer_srv`] asked it for every buffer view made:
/// each slot a heap sync makes again (25,000 to 40,000 a frame in the
/// world) and each view of a draw's fresh table, on the recording threads at
/// the same time. The answer cannot change while the process runs.
fn under_wine() -> bool {
    static UNDER_WINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *UNDER_WINE.get_or_init(crate::shader_cache::under_wine)
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0);
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let format = desc.Format.0;
    let typed_three_channel =
        typed_buffer(buffer.StructureByteStride, raw) && matches!(format, 6..=8);
    // D3DMetal 3.0 reports typed-UAV support for these formats, then tries
    // to make an unsupported three-channel Metal format and aborts inside the
    // void CreateUnorderedAccessView call. D3D12 exposes no D3DMetal version,
    // so keep the Wine-wide guard until 4.0 is characterized. The caller
    // installs a null UAV and keeps no resource or mark.
    if typed_three_channel && crate::under_wine() {
        once!(format, "three-channel typed unordered buffer view under Wine");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    if typed_three_channel {
        // The COM method has no HRESULT. On native hosts, reject a format
        // the device does not advertise and a device already removed.
        // SAFETY: plain device queries.
        if unsafe { host.device.GetDeviceRemovedReason() }.is_err() || !typed_store(format as u32) {
            once!(format, "three-channel typed unordered buffer view the host cannot make");
            return None;
        }
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    // CreateUnorderedAccessView is void and cannot report failure directly.
    // If it removed the native device, do not keep its descriptor or mark.
    // SAFETY: a plain device query.
    if typed_three_channel && unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
        once!(format, "device removed making a three-channel typed unordered buffer view");
        return None;
    }
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let is_typeless = |f: u32| family(f) == f && family(f + 1) == f;
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // A typed unordered view in a format the host cannot store to
    // (B5G5R5A1_UNORM on NVIDIA) removes the device when made, in the
    // texture's own format too: the slot stays null, its writes dropped.
    // The format checked is the one the view is made in: a typed
    // texture's own unless the requested one is castable (below).
    let typed = if view == 0 || (!typeless && !castable.contains(&view)) { own } else { view };
    if unordered && !(view == 0 && typeless) && !typed_store(typed) {
        once!(view = typed, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    // A packed 16-bit target made R8G8B8A8_UNORM on this GPU
    // (`resource::target_format`): views name the format it has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) || is_typeless(view) {
        once!(unordered, view, own, "view the host does not allow");
        return false;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let effective = if typeless || cast { view } else { own };
    if unordered && !typed_store(effective) {
        once!(
            view = effective,
            own,
            "an unordered view in a format the host cannot store"
        );
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if let Some(desc) = desc.as_mut()
                    && desc.Format.0 == 0
                    && let Some(format) = crate::resource::view_format(resource)
                {
                    desc.Format = DXGI_FORMAT(format as i32);
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
            // Shaders indexing the heap write through this view without
            // moving the texture to a write state: kept from now on, unless
            // `VICTORY_EVICT_HEAP_UAV` lets it go until shaders write
            // through heaps (`crate::resource::evictable`).
            if crate::resource::evict_heap_uav() {
                copy.note_heap_writable();
            } else {
                copy.note_state(0x8);
            }
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if let Some(desc) = desc.as_mut()
                && desc.Format.0 == 0
                && let Some(format) = crate::resource::view_format(resource)
            {
                desc.Format = DXGI_FORMAT(format as i32);
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is an unordered view of a resource object's texture.
fn is_unordered_texture(view: &View) -> bool {
    matches!(view, View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// Makes what a host view of `view` needs, if it is not made yet: the
/// texture it views (made on first use) or the host copy of the buffer
/// memory it views. The slot's view is then quick to make.
fn prepare(view: &View) {
    #[cfg(test)]
    tests::while_preparing();
    match view {
        View::Placed(..) if is_placed_texture(view) => {
            let _ = placed_texture(view);
        }
        View::Shader(r, _) | View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some() => {
            let _ = crate::resource::host_resource(*r as crate::object::This);
        }
        _ => {
            let _ = view_copy(view);
        }
    }
}

/// Whether `view` is a shader view of a texture (one placed in memory, or a
/// resource object's): such a slot left null is made when read.
fn is_texture(view: &View) -> bool {
    is_placed_texture(view)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// A view of a texture the title placed in memory itself, as a view of the
/// resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    let view_desc = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture as usize, 0, view_desc)
    } else {
        View::Shader(texture as usize, view_desc)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Typed buffer views of integer formats carry their kind in the
    /// view-stride table; float, normalized, raw and structured ones none.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(7, 0, 0)), VIEW_INTEGER, "R32G32B32_UINT");
        assert_eq!(view_kind(&uav(8, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R32G32B32_SINT");
        assert_eq!(view_kind(&uav(57, 0, 0)), VIEW_INTEGER, "R16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(64, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R8_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_kind(&srv), VIEW_INTEGER, "R8_UINT");
        assert_eq!(view_stride(&uav(12, 0, 0)) | view_kind(&uav(12, 0, 0)), 8 | VIEW_INTEGER);
        assert_eq!(widened_format(6), Some(DXGI_FORMAT_R32G32B32A32_FLOAT));
        assert_eq!(widened_format(7), Some(DXGI_FORMAT_R32G32B32A32_UINT));
        assert_eq!(widened_format(8), Some(DXGI_FORMAT_R32G32B32A32_SINT));
        assert_eq!(widened_format(2), None);
        assert_eq!(widened_w(6), Some(0x3f80_0000));
        assert_eq!(widened_w(7), Some(1));
        assert_eq!(widened_w(8), Some(1));
        assert_eq!(widened_w(2), None);
    }

    /// Three-channel 32-bit SRVs are backed by a widened copy on hosts that
    /// read R32G32B32 as R32G32. The copy keeps 12-byte source elements,
    /// supplies w in the source format and follows later title writes.
    #[test]
    fn three_channel_srvs_use_a_widened_refreshed_copy() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = host() else { return };
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first: [u32; 12] = [
            1000, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009,
        ];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        crate::shadow::register(p, p as u64, len as u64);
        let shadow = crate::shadow::find(p as u64, len as u64).expect("a host copy");
        shadow.update(p, len);

        // SAFETY: plain descriptor-heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                ..Default::default()
            })
        }
        .expect("a descriptor heap");
        // SAFETY: a live heap.
        let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 4,
                    ..Default::default()
                },
            },
        };
        let (resource, source) = buffer_srv(p as u64, len as u64, desc, at)
            .expect("a view")
            .expect("a kept view");
        let Some(HeldSource::Widened(widened, _)) = source else {
            panic!("a widened source");
        };
        // SAFETY: a live resource.
        let made = unsafe { resource.GetDesc() };
        assert_eq!(made.Width, 4 * 16);
        let float_one = 0x3f80_0000;
        let want = |w| {
            [
                1000, 2000, 3000, w, 1003, 2003, 3003, w, 1006, 2006, 3006, w, 1009, 2009, 3009, w,
            ]
        };
        assert_eq!(widened.read_host(16), want(float_one));

        // SAFETY: a live heap; each later slot is one descriptor stride on.
        let step = unsafe {
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let at_slot = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + slot * step as usize,
        };
        let (second_resource, second_source) =
            buffer_srv(p as u64, len as u64, desc, at_slot(1))
                .expect("a second view")
                .expect("a second kept view");
        let Some(HeldSource::Widened(second, _)) = second_source else {
            panic!("a second widened source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&widened, &second),
            "one widened copy per range and w"
        );
        drop((second_resource, second));

        let uint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_UINT,
            ..desc
        };
        let (uint_resource, uint_source) = buffer_srv(p as u64, len as u64, uint_desc, at_slot(2))
            .expect("a uint view")
            .expect("a kept uint view");
        let Some(HeldSource::Widened(uint, _)) = uint_source else {
            panic!("a widened uint source");
        };
        let sint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_SINT,
            ..desc
        };
        let (sint_resource, sint_source) = buffer_srv(p as u64, len as u64, sint_desc, at_slot(3))
            .expect("a sint view")
            .expect("a kept sint view");
        let Some(HeldSource::Widened(sint, _)) = sint_source else {
            panic!("a widened sint source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&uint, &sint),
            "integer formats share their widened copy"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&widened, &uint),
            "float and integer w bits need distinct copies"
        );
        assert_eq!(uint.read_host(16), want(1));
        assert_eq!(sint.read_host(16), want(1));

        let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
        // SAFETY: one 12-byte element of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(changed.as_ptr().cast::<u8>(), (p + 12) as *mut u8, 12) };
        assert!(shadow.update(p, len) > 0, "the title write is copied");
        let changed_want = |w| {
            [
                1000, 2000, 3000, w, 0x1111_1111, 0x2222_2222, 0x3333_3333, w, 1006, 2006, 3006, w,
                1009, 2009, 3009, w,
            ]
        };
        assert_eq!(widened.read_host(16), changed_want(float_one));
        assert_eq!(uint.read_host(16), changed_want(1));
        assert_eq!(sint.read_host(16), changed_want(1));

        drop((
            resource,
            widened,
            uint_resource,
            uint,
            sint_resource,
            sint,
            heap,
            shadow,
        ));
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A placed texture's view the title writes into a heap is left null
    /// and noted once: heap syncs (every submission) pass it by, a draw
    /// reading it through a table makes it.
    #[test]
    fn a_placed_texture_slot_left_null_is_made_once_a_draw_reads_it() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        for _ in 0..2 {
            assert!(host.mirror(0, title_heap.start, false).is_some());
            assert!(host.made(0), "the record is noted");
            assert!(!host.kept(0), "no host texture for it yet");
        }
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
    }

    /// A slot viewing a texture given up (`crate::resource::evict`) holds it
    /// no more: heap syncs pass it by, a draw reading it through a table
    /// makes it again.
    #[test]
    fn a_slot_viewing_a_texture_given_up_is_made_again_when_read() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!host.kept(0), "the slot holds it no more");
        assert!(host.made(0), "its record stays noted");
        assert!(host.mirror(0, title_heap.start, false).is_some());
        assert!(!host.kept(0), "heap syncs pass it by");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made again for the draw");
        assert!(crate::resource::made_copy(texture).is_some(), "with a host texture again");
    }

    /// Work to run while a view's texture is prepared, once (tests).
    pub(crate) static WHILE_PREPARING: std::sync::Mutex<Option<crate::descriptor::tests::Work>> = std::sync::Mutex::new(None);

    pub(crate) fn while_preparing() {
        let work = WHILE_PREPARING.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(work) = work {
            work();
        }
    }

    /// A resource's final release during preparation returns at once, and the
    /// resource stays alive for the make: the view's pin frees it when the
    /// caller drops the view.
    #[test]
    fn a_resource_released_while_its_view_is_prepared_outlives_the_make() {
        if host().is_none() {
            return; // no host GPU here
        }
        let resource = crate::resource::test_buffer(256);
        let mut record = [0u32; 12];
        let at = record.as_mut_ptr() as u64;
        // R8_UINT buffer SRV, all 256 elements.
        let desc = [62, 1, 0x1688, 0, 0, 256, 0, 0, 0, 0];
        crate::descriptor::create_srv(std::ptr::null(), resource, &desc, at);
        let Some(heap) = HostHeap::new(false, 1) else {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10)(
                resource,
            );
            return;
        };

        let address = resource as usize;
        let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
            resource,
            0x10,
        );
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            // The title's last reference goes while the view is being made.
            // The call must not wait for the make: this very thread is the
            // one making the view, so a wait would never end.
            let released = std::thread::spawn(move || release(address as crate::object::This));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !released.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the title's release waited for the view's make"
                );
                std::thread::yield_now();
            }
            assert_eq!(released.join().unwrap(), 1, "the pin still holds it");
        }));

        let view = heap.mirror(0, at, false).expect("the current view");
        assert!(WHILE_PREPARING.lock().unwrap().is_none(), "the prepare ran");
        assert!(heap.made(0), "the view was made from the live resource");
        assert!(
            crate::descriptor::view(at).is_some(),
            "still current while the caller holds the view"
        );
        // The last pin's drop frees the resource and makes its views stale.
        drop(view);
        assert!(crate::descriptor::view(at).is_none(), "stale once it is freed");
    }

    /// A record the title writes into a slot while another thread prepares
    /// the slot's old record is the one the slot ends up viewing.
    #[test]
    fn a_slot_rewritten_while_its_view_is_prepared_views_the_new_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let s = 960u32;
        let at = title_heap.start + u64::from(s) * 32;
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let view_of = |address: u64| {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            view
        };
        let (first, second) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + (1 << 16));
        crate::descriptor::create_placed_srv(null, &words, &view_of(first), at);
        let host = title_heap.host().expect("a host view heap");
        let (done, waited) = std::sync::mpsc::channel();
        let second_view = view_of(second);
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            // The title writes the other record into the slot meanwhile.
            std::thread::spawn(move || {
                crate::descriptor::create_placed_srv(std::ptr::null(), &words, &second_view, at);
                done.send(()).unwrap();
            });
            assert!(waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(), "the write waits");
        }));
        // A draw reads the slot: its first record is prepared.
        assert!(host.mirror(s, at, true).is_some());
        let second_texture = crate::resource::texture_at(second, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(second_texture).expect("made").texture) as usize;
        assert_eq!(host.textures_at(&[s]), [raw], "the slot views the record written last");
    }

    /// What a view needs is made without the slot lock: another thread
    /// binding a view of the same heap meanwhile is not held up.
    #[test]
    fn views_bound_while_another_slot_gets_its_texture_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (first, second) = (950u32, 951u32);
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        for (i, at) in [(first, 0u64), (second, 1 << 16)] {
            let address = memory.as_ptr() as u64 + at;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, slot(i));
        }
        let host = title_heap.host().expect("a host view heap");
        let heap_at = title_heap.start;
        let heap = heap as usize;
        let (done, waited) = std::sync::mpsc::channel();
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            std::thread::spawn(move || {
                let h = crate::descriptor::heap(heap as crate::object::This);
                h.host().unwrap().mirror(second, heap_at + u64::from(second) * 32, true);
                done.send(()).unwrap();
            });
            assert!(
                waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                "a view bound while another slot's texture is made waits"
            );
        }));
        assert!(host.mirror(first, slot(first), true).is_some());
        assert!(WHILE_PREPARING.lock().unwrap().is_none(), "the prepare ran");
        assert!(host.kept(first) && host.kept(second), "both made");
    }

    /// The textures of slots shaders read are made without the heaps' lock:
    /// a title thread writing a view meanwhile is not held up by the make.
    #[test]
    fn views_written_while_read_slots_get_their_textures_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (read, written) = (900u32, 901u32);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, slot(read));
        let host = title_heap.host().expect("a host view heap");
        assert!(!host.kept(read), "left null until read");
        let (done, waited) = std::sync::mpsc::channel();
        let at = slot(written);
        *crate::descriptor::tests::WHILE_MAKING.lock().unwrap() = Some((
            read,
            Box::new(move || {
                std::thread::spawn(move || {
                    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, at);
                    done.send(()).unwrap();
                });
                assert!(
                    waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                    "a view written while a read slot's texture is made waits"
                );
            }),
        ));
        crate::descriptor::make_read(&[read]);
        assert!(crate::descriptor::tests::WHILE_MAKING.lock().unwrap().is_none(), "the make ran");
        assert!(host.kept(read), "made for the read");
    }

    /// A heap sync makes the slots whose records the title changed as bytes
    /// (a copy of another slot's record, a record cleared) and passes the
    /// others by.
    #[test]
    fn heap_syncs_make_the_slots_whose_records_changed() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // A view of no memory: a view of the test's own heap memory would get
        // a host copy whose 4 MiB window the guards may cover (the process
        // heap guarded read-only hung the suite).
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start);
        let host = title_heap.host().expect("a host view heap");
        // SAFETY: slots 0 and 1 of the heap's memory, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32) };
        assert!(!host.made(1), "copied as bytes: not seen yet");
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && host.made(1) && !host.made(2));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes((start + 32) as *mut u8, 0, 32) };
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && !host.made(1), "a cleared record leaves its slot");
    }

    /// mac: a heap sync ahead of the submission makes a slot whose record
    /// changed, but leaves one whose record is a view of a texture the
    /// title placed itself as it was; the sync before the submission then
    /// notes that record, as it always did.
    #[test]
    fn a_sync_ahead_leaves_placed_texture_views_for_the_sync_before_the_submission() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // Slot 0: a 4x1 RGBA8 texture in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, start);
        // Slot 2: a constant buffer view (of no memory, as above).
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start + 64);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.made(0) && host.made(2));
        // The title copies both records as bytes: slot 0's to 1, 2's to 3.
        // SAFETY: slots of the heap's memory, 32 bytes each.
        unsafe {
            std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32);
            std::ptr::copy_nonoverlapping((start + 64) as *const u8, (start + 96) as *mut u8, 32);
        }
        assert!(!host.made(1) && !host.made(3), "copied as bytes: not seen yet");
        {
            let _round = super::super::ahead::Round::start();
            host.sync_slots(0, 4, start, 32);
        }
        assert!(host.made(3), "the buffer view is made ahead");
        assert!(!host.made(1), "the placed texture's view is left as it was");
        // Before the submission: the slots left, then what was written since.
        host.sync_left(start, 32);
        assert!(host.made(1), "its record is noted");
        assert!(!host.kept(1), "and its view left for a read to make");
        host.sync_left(start, 32);
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && host.made(1) && host.made(2) && host.made(3));
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A table made afresh lies in dynamic slots, and a shader reading a
    /// view's address (e.g. the BVH builder's base) indexes the view
    /// addresses there: they have an entry for each dynamic slot, entered
    /// whenever the slot is made (a view, null, or a raw view of ours).
    /// Giving a slot back leaves its entry: every table made afresh makes
    /// all its slots. (Hosts whose view heaps are smaller than the title's,
    /// NVIDIA, make such tables; the shader read past the buffer.)
    ///
    /// Any heap size the host takes will do: a title heap of 4 slots leaves
    /// dynamic slots past it either way ([`DYNAMIC`] or [`MIN_DYNAMIC`]).
    #[test]
    fn view_addresses_cover_tables_made_afresh() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let end = heap.count() + heap.dynamic;
        assert!(heap.view_addresses().is_some());
        let first = heap.dynamic(2).unwrap();
        assert!(first >= heap.count() && first + 1 < end);
        // A buffer the title placed at 0xd_6013_e000: 16 elements of 4 bytes.
        let address = 0xd_6013_e000u64;
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 64);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 4);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let view = View::Placed(false, resource, desc);
        assert_eq!(buffer_range(&view), Some((address, 64)));
        assert_eq!(heap.address_at(first), Some(0), "not made yet");
        heap.make_at(first, &view);
        assert_eq!(heap.address_at(first), Some(address));
        assert_eq!(heap.address_at(first + 1), Some(0), "not made yet");
        // The last dynamic slot has its entry too.
        heap.make_at(end - 1, &view);
        assert_eq!(heap.address_at(end - 1), Some(address));
        // A slot given back keeps its entry until it is made again; made
        // null, it has its entry cleared.
        heap.release(first, 2);
        assert_eq!(
            heap.address_at(first),
            Some(address),
            "given back, not made"
        );
        heap.make_null(first);
        assert_eq!(heap.address_at(first), Some(0));
        assert_eq!(heap.address_at(end - 1), Some(address), "still made");
        // Raw views of ours (root views, host copies' view pairs) in dynamic
        // slots clear what an earlier table entered there.
        heap.make_at(first, &view);
        heap.raw_view_at(first, None, 0, 0, false);
        assert_eq!(heap.address_at(first), Some(0), "a root view");
        heap.make_at(first, &view);
        heap.make_at(first + 1, &view);
        heap.raw_pair_at(first, None, 0);
        assert_eq!(heap.address_at(first), Some(0), "a view pair");
        assert_eq!(heap.address_at(first + 1), Some(0), "a view pair");
        // Slots past the dynamic ones (host copies' view pairs) have no
        // entry, and nothing is written past the last: a mark there (the
        // buffer's allocation reaches past it) stays.
        assert_eq!(heap.address_at(end), None);
        let host = host().unwrap();
        let (buffer, data, _) = heap.addresses.get().unwrap().as_ref().unwrap();
        // SAFETY: plain queries of a live buffer.
        let (desc, size) = unsafe {
            let desc = buffer.GetDesc();
            let info = host.device.GetResourceAllocationInfo(0, &[desc]);
            (desc, info.SizeInBytes)
        };
        assert_eq!(desc.Width, 16 * u64::from(end));
        assert!(size >= desc.Width + 8, "room for the mark");
        let past = (data + 16 * end as usize) as *mut u64;
        let mark = 0x5a5a_5a5a_5a5a_5a5a;
        // SAFETY: mapped memory of the buffer's allocation (checked).
        unsafe { past.write_unaligned(mark) };
        heap.note_address(end, Some(&view));
        // SAFETY: as above.
        assert_eq!(unsafe { past.read_unaligned() }, mark);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(heap.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_VIEWS);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            // Released on a thread of its own.
            let since = std::time::Instant::now();
            while references(&buffer) > 1 && since.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// A texture view's size words are the console image descriptor's
    /// fields: level 0's size of the resource, the view's levels, its type.
    #[test]
    fn a_texture_view_gives_its_image_descriptor_size_fields() {
        // A 1000x600 2D texture of 10 levels, then an array of 6 with the
        // whole chain (mip count 0, 1024 wide: 11 levels).
        let mut texture = [0u32; 14];
        (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 1000, 600, 1 | 10 << 16, 1);
        let mut array = texture;
        (array[4], array[7]) = (1024, 6);
        let mut view = [0u32; 10];
        // Levels 2-4 of the texture: width - 1 = 999 (0b11_1110_0111).
        (view[1], view[4], view[5]) = (4, 2, 3);
        let words = image_words(&texture, Some(&view), false);
        assert_eq!(words, [0, 3 << 30, 249 | 599 << 14, 0xFAC | 2 << 12 | 4 << 16 | 9 << 28]);
        // Its whole chain from level 1.
        (view[4], view[5]) = (1, u32::MAX);
        assert_eq!(image_words(&texture, Some(&view), false)[3], 0xFAC | 1 << 12 | 9 << 16 | 9 << 28);
        // An unordered view of level 3: that level alone.
        let uav = [0, 4, 3, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&texture, Some(&uav), true)[3], 0xFAC | 3 << 12 | 3 << 16 | 9 << 28);
        // An unordered view of level 2 of six array slices: the level is
        // not the slice count (word 4).
        let uav = [0, 5, 2, 0, 6, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&array, Some(&uav), true)[3], 0xFAC | 2 << 12 | 2 << 16 | 13 << 28);
        // The array's own view: a 2D array, levels 0-10.
        let words = image_words(&array, None, false);
        assert_eq!(words[1..], [3 << 30, 255 | 599 << 14, 0xFAC | 10 << 16 | 13 << 28]);
    }

    /// The words `image_words` reads are where the host's view descriptions
    /// have their levels: an unordered view's `MipSlice` in word 2, a shader
    /// view's `MostDetailedMip` and `MipLevels` in words 4 and 5.
    #[test]
    fn view_description_words_hold_the_levels_where_image_words_reads_them() {
        let words = |uav: &D3D12_UNORDERED_ACCESS_VIEW_DESC| -> [u32; 10] {
            // SAFETY: 40 plain bytes.
            unsafe { std::ptr::from_ref(uav).cast::<[u32; 10]>().read_unaligned() }
        };
        assert_eq!(std::mem::size_of::<D3D12_UNORDERED_ACCESS_VIEW_DESC>(), 40);
        assert_eq!(std::mem::size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>(), 40);
        let mut uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        uav.Anonymous.Texture2D = D3D12_TEX2D_UAV { MipSlice: 3, PlaneSlice: 0 };
        assert_eq!(words(&uav)[1..5], [4, 3, 0, 0]);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Anonymous.Texture2DArray =
            D3D12_TEX2D_ARRAY_UAV { MipSlice: 2, FirstArraySlice: 1, ArraySize: 6, PlaneSlice: 0 };
        assert_eq!(words(&uav)[1..5], [5, 2, 1, 6]);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uav.Anonymous.Texture3D = D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 0, WSize: 32 };
        assert_eq!(words(&uav)[1..5], [8, 1, 0, 32]);
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 2, MipLevels: 3, ..Default::default() };
        // SAFETY: 40 plain bytes.
        let w = unsafe { std::ptr::from_ref(&srv).cast::<[u32; 10]>().read_unaligned() };
        assert_eq!((w[1], w[4], w[5]), (4, 2, 3));
    }

    fn placed_memory() -> u64 {
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: non-zero size and power-of-two alignment; leaked for the test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    fn placed_tex_desc(width: u64, height: u32, format: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width as u32,
            (width >> 32) as u32,
            height,
            1 | (1 << 16),
            format,
            1,
            0,
            0,
            flags,
            0,
        ]
    }

    fn placed_tex_uav_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn placed_tex_srv_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_SRV_DIMENSION_TEXTURE2D
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1; // MipLevels
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    #[test]
    fn placed_uav_textures_create_real_host_uavs() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let _ = host.log_messages();
        let bgrx_uav = typed_store(88);
        let cases = [
            (28u32, 28u32, 4u32, true),     // R8G8B8A8_UNORM
            (27u32, 28u32, 4u32, true),     // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
            (29u32, 29u32, 4u32, true),     // R8G8B8A8_UNORM_SRGB -> linear R8G8B8A8_UNORM
            (29u32, 0u32, 4u32, true),      // R8G8B8A8_UNORM_SRGB + UNKNOWN -> linear 28
            (91u32, 91u32, 4u32, true),     // B8G8R8A8_UNORM_SRGB -> linear B8G8R8A8_UNORM
            (91u32, 0u32, 4u32, true),      // B8G8R8A8_UNORM_SRGB + UNKNOWN -> linear 87
            (28u32, 28u32, 0u32, false),    // missing ALLOW_UNORDERED_ACCESS -> null fallback
            (93u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_UNORM_SRGB -> linear 88 if supported
            (93u32, 0u32, 4u32, bgrx_uav),  // B8G8R8X8_UNORM_SRGB + UNKNOWN -> linear 88
            (92u32, 88u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM
            (92u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM_SRGB
            (40u32, 40u32, 4u32, false),    // D32_FLOAT + UAV flag (no depth flag) -> null fallback
            (55u32, 55u32, 4u32, false),    // D16_UNORM + UAV flag (no depth flag) -> null fallback
            (39u32, 40u32, 4u32, false),    // R32_TYPELESS + D32_FLOAT view -> null fallback
            (53u32, 55u32, 4u32, false),    // R16_TYPELESS + D16_UNORM view -> null fallback
            (27u32, 27u32, 4u32, false),    // R8G8B8A8_TYPELESS + typeless view -> null fallback
        ];
        let mut slots = [[0u32; 8]; 24];
        for (idx, (res_fmt, view_fmt, flags, expect_real)) in cases.into_iter().enumerate() {
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, flags);
            let desc = placed_tex_uav_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, slot_addr);
            let view = crate::descriptor::view(slot_addr).unwrap();
            assert!(matches!(
                placed_texture(&view),
                Some(View::Unordered(_, 0, Some(_)))
            ));
            assert!(heap.mirror(idx as u32, slot_addr, false).is_some());
            assert!(
                !heap.kept(idx as u32),
                "case {idx} mirror(false) should defer host creation"
            );
            let mirrored = heap.mirror(idx as u32, slot_addr, true).unwrap();
            let dyn_slot = heap.count + idx as u32;
            heap.make_at(dyn_slot, &mirrored);
            assert_eq!(heap.kept(idx as u32), expect_real, "case {idx} mirror");
            assert_eq!(
                heap.kept(dyn_slot),
                expect_real,
                "case {idx} make_at"
            );
        }
        for (offset, (res_fmt, view_fmt, expect_real)) in [
            (39u32, 40u32, false), // R32_TYPELESS + D32_FLOAT SRV -> null fallback
            (53u32, 55u32, false), // R16_TYPELESS + D16_UNORM SRV -> null fallback
            (29u32, 0u32, true),   // R8G8B8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 29
            (91u32, 0u32, true),   // B8G8R8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 91
        ]
        .into_iter()
        .enumerate()
        {
            let idx = cases.len() + offset;
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, 4);
            let desc = placed_tex_srv_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot_addr);
            heap.mirror(idx as u32, slot_addr, true).unwrap();
            assert_eq!(
                heap.kept(idx as u32),
                expect_real,
                "srv case {offset}"
            );
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None);
    }

    /// Wine's D3DMetal aborts while creating these typed UAVs. The production
    /// path must leave a null descriptor, no kept resource and no write mark.
    fn three_channel_uav_falls_back_under_wine(format: DXGI_FORMAT) {
        let Some(host) = host() else { return };
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, 64);
        let mut resource = [0u32; 14];
        resource[0] = 1; // buffer
        resource[4] = 64;
        let mut words = [0u32; 16];
        words[0] = format.0 as u32;
        words[1] = 1; // buffer
        words[4] = 1; // one 12-byte element
        (words[10], words[11]) = (p as u32, (p >> 32) as u32);
        let slot = heap.dynamic(1).expect("a dynamic slot");
        heap.make_at(slot, &View::Placed(true, resource, words));
        assert!(!heap.kept(slot), "a refused UAV keeps no resource");
        let marks = heap.marks.as_ref().expect("the write-mark table").1;
        // SAFETY: two dwords of the mapped table for this live slot.
        let mark = unsafe { std::slice::from_raw_parts((marks as *const u32).add(slot as usize * 2), 2) };
        assert_eq!(mark, [victory_shader::emit::VIEW_MARK_NONE, 0], "a refused UAV has no write mark");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok(), "the null fallback leaves the device alive");
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// Synthetic hardware characterization: create a typed R32G32B32 UAV,
    /// store one three-component element, and copy the raw bytes back. Wine
    /// exercises the safe fallback instead, because raw creation aborts.
    fn characterize_r32g32b32(format: DXGI_FORMAT, scalar: &str, value: &str, expected: [u32; 3]) {
        eprintln!("typed UAV characterization: {format:?}");
        if crate::under_wine() {
            three_channel_uav_falls_back_under_wine(format);
            eprintln!("typed UAV characterization: Wine fallback passed");
            return;
        }
        let Some(host) = host() else {
            eprintln!("typed UAV characterization: no host device");
            return;
        };
        let device = &host.device;
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        // SAFETY: a plain query into a struct of its size.
        let support_result = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        };
        eprintln!("typed UAV characterization: format support {support_result:?}, Support1={:#x}", support.Support1.0);
        if support_result.is_err()
            || support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 == 0
        {
            eprintln!("typed UAV characterization: host does not advertise typed UAV support; skipped");
            return;
        }
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a plain committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_UNKNOWN,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                )
                .unwrap();
        }
        let buffer = made.unwrap();
        // SAFETY: a plain shader-visible descriptor heap.
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .unwrap();
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: 1, ..Default::default() },
            },
        };
        // SAFETY: a live buffer and one slot of our heap.
        unsafe {
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateUnorderedAccessView(&buffer, None::<&ID3D12Resource>, Some(&desc), cpu);
        }
        // SAFETY: a plain device query.
        let removed = unsafe { device.GetDeviceRemovedReason() };
        eprintln!("typed UAV characterization: after creation {removed:?}");
        removed.expect("creating the typed UAV leaves the device alive");
        let source = format!(
            "// {format:?}\nRWBuffer<{scalar}3> output : register(u0);\n[numthreads(1, 1, 1)]\nvoid main() {{ output[0] = {scalar}3({value}); }}\n"
        );
        let (root, pso) = crate::compute::pipeline(
            &source,
            windows::core::s!("typed uav characterization"),
            0,
            &[crate::compute::Param::Table(&[crate::compute::uav(0, 1)])],
        )
        .expect("the typed-store shader compiles");
        let list = host.new_list().unwrap();
        let (readback, data) = list.readback(host, 256).unwrap();
        // SAFETY: an open list, live objects, and a descriptor in our heap.
        unsafe {
            let gpu = heap.GetGPUDescriptorHandleForHeapStart();
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, gpu);
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyResource(&readback, &buffer);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: execute waited for the copy; the mapped readback is 256 bytes.
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), 36) };
        eprintln!("typed UAV characterization: raw readback {bytes:?}");
        let want: Vec<u8> = expected.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(bytes, want, "typed UAV store/raw readback");
    }

    #[test]
    fn characterize_r32g32b32_float_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_FLOAT,
            "float",
            "1.25, -2.5, 3.75",
            [1.25f32.to_bits(), (-2.5f32).to_bits(), 3.75f32.to_bits()],
        );
    }

    #[test]
    fn characterize_r32g32b32_uint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_UINT,
            "uint",
            "0x11223344u, 0x55667788u, 0x99aabbccu",
            [0x1122_3344, 0x5566_7788, 0x99aa_bbcc],
        );
    }

    #[test]
    fn characterize_r32g32b32_sint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_SINT,
            "int",
            "-1, -2, -3",
            [(-1i32) as u32, (-2i32) as u32, (-3i32) as u32],
        );
    }

    #[test]
    fn sync_slots_skips_empty_and_clears_after_forget() {
        use crate::object::{HRESULT, S_OK};
        let Some(host) = HostHeap::new(false, 256) else {
            return;
        };
        let desc: [u32; 14] = [1, 0, 0, 0, 64, 0, 1, 1 | 1 << 16, 0, 1, 0, 0, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut resource = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(made, S_OK);

        let mut title_heap = vec![0u8; 256 * 32];
        let start = title_heap.as_ptr() as u64;
        crate::descriptor::create_srv(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            start,
        );
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(host.made(0));
        assert!(host.made_is_nonzero(0));
        assert!(!host.made(1));
        assert!(!host.made_is_nonzero(1));

        // Second sync over the verified block keeps slot 0 live and skips empty slots.
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(host.made(0));

        // Forgetting the resource bumps VIEWS_FORGOT so the verified block re-checks
        // and clears slot 0 even though the title memory bytes didn't change.
        crate::descriptor::forget(resource as usize);
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(!host.made(0));
        assert!(!host.made_is_nonzero(0));

        crate::object::release(resource);
        let _ = title_heap.as_mut_ptr();
    }
}
