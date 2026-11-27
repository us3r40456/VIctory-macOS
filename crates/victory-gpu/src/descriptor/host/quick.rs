//! mac: a heap sync's changed slots made the quick way
//! (`VICTORY_QUICK_SYNC`, `crate::mac_switches::quick_sync`; on under Wine).
//!
//! [`HostHeap::sync_slots`] hands every slot whose record changed to
//! [`HostHeap::mirror`], 25,000 to 40,000 times a world frame on the
//! submitting thread, and most of those records are shader views of buffers
//! (five of six made slots of the world's heap). For such a record `mirror`
//! looks the host copy of the view's memory up three times
//! (`prepare`, then `buffer_srv` and the census's `seen` under the slot
//! lock; each a walk of the title's buffers under their lock, and of the
//! views' own copies for a view off its copy's elements), and the title
//! writes the same record into many slots of a page (a table's shared
//! buffers).
//!
//! Here the records of a pass (one written 64 KiB page of the title's heap,
//! up to 2,048 slots) are resolved once each: what `make` and `seen` would
//! work out for the record (the host buffer, the view's description on it,
//! the copy to hold, the census entry) is found with one lookup, without
//! the slot lock, and every slot of the pass that holds the record is then
//! made from it under the lock, slot by slot, by the steps `mirror` takes
//! there. A record stays resolved for the pass (a few milliseconds), its
//! view pinned: a later slot gets what an earlier slot of the same pass got,
//! where `mirror` would have looked again.
//!
//! Only records whose view `mirror` would make as a buffer shader view take
//! this way: constant buffers, shader views of buffer resources, shader
//! views of buffers the title placed itself. Everything else (textures,
//! unordered access, samplers, records that are no view, views the host
//! refuses) goes to `mirror` as before, slot by slot in the same order.
//! [`resolve`] follows `make` and `buffer_srv`: a change there belongs here
//! too (the test below compares the two ways).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{
    HeldSource, HostHeap, element_size, in_buffer, srv_desc, under_wine, view_kind, view_stride, widened_format,
    widened_w,
};
use crate::bindless::Seen;
use crate::census_maps::Hashing;
use crate::descriptor::{VIEW_TAG, View, ViewPin, view_at};
use crate::host::{Host, host};
use crate::shadow::{Shadow, Widened};

/// Whether heap syncs make their buffer shader views this way.
pub(super) fn on() -> bool {
    crate::mac_switches::quick_sync(crate::shader_cache::under_wine)[1]
}

/// What a slot made from a buffer shader view gets, as [`super::make`] and
/// `crate::bindless` work it out.
struct Made {
    /// The host buffer the view is made on and its description there.
    buffer: ID3D12Resource,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    /// The copy the view reads, which the slot holds if it is the view's
    /// own (`crate::shadow::for_view`), and the widened copy a
    /// three-channel view is made on instead (with the copy it follows).
    shadow: Arc<Shadow>,
    own: bool,
    widened: Option<Arc<Widened>>,
    /// The view-stride table's entry.
    stride: u32,
    /// The census entry.
    seen: Seen,
}

impl Made {
    /// What the slot holds beside the buffer, as `buffer_srv` returns it.
    fn held(&self) -> Option<HeldSource> {
        match &self.widened {
            Some(widened) => Some(HeldSource::Widened(widened.clone(), self.shadow.clone())),
            None => self.own.then(|| HeldSource::Shadow(self.shadow.clone())),
        }
    }
}

/// A record resolved in a pass: the record, its view (pinned: its resource
/// stays for the pass) and what slots holding it get.
struct Resolved {
    record: [u32; 3],
    view: ViewPin,
    made: Made,
}

/// What `make(view, ..)` would make and `seen(view)` enter, if `view` is one
/// that `make` gives `buffer_srv` and that `buffer_srv` makes; none for any
/// other view, and for one the host refuses (`mirror` then does what it
/// did, warnings included). Called without the slot lock, as `prepare` is:
/// the lookup makes the host copy on first use.
fn resolve(view: &View) -> Option<Made> {
    // The buffer's title address and the view's host description, as `make`
    // hands them to `buffer_srv`; for a constant buffer also the copy `make`
    // and `seen` find for it.
    let (address, mut desc, constant) = match view {
        &View::Constant(address, size) => {
            let len = u64::from(size);
            // Whole dwords inside the copy: then `make`'s view is of
            // `size` bytes and `seen`'s range is the same.
            if size == 0 || size % 4 != 0 {
                return None;
            }
            let shadow = crate::shadow::find(address, len)?;
            if shadow.left(address) < len {
                return None;
            }
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: size / 4,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            (address, desc, Some(shadow))
        }
        View::Shader(resource, Some(words)) => {
            let resource = *resource as crate::object::This;
            let (data, _) = crate::resource::buffer_bytes(resource)?;
            // `make` asks for a host texture first.
            if crate::resource::host_copy(resource).is_some() {
                return None;
            }
            (data as u64, srv_desc(words), None)
        }
        View::Placed(false, resource, words) if resource[0] == 1 => {
            let address = u64::from(words[10]) | u64::from(words[11]) << 32;
            (address, srv_desc(words[..10].try_into().ok()?), None)
        }
        _ => return None,
    };
    // A buffer view by its description too: for any other dimension
    // `prepare` and `seen` look the copy up by another element size than
    // `buffer_srv` does.
    let element = element_size(&desc);
    if element == 0 || desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: a buffer view's union member.
    let (first, count) = unsafe { (desc.Anonymous.Buffer.FirstElement, desc.Anonymous.Buffer.NumElements) };
    // The view's title range: `buffer_srv`'s `start`, `buffer_range`'s result.
    let start = address + first * element;
    let len = u64::from(count) * element;
    let (shadow, own) = match &constant {
        // `for_view` finds this copy again (the same address and length)
        // and takes it for a view that starts on one of its elements.
        Some(found) if found.offset(start).is_multiple_of(element) => (found.clone(), false),
        _ => crate::shadow::for_view(start, len, element)?,
    };
    let first = shadow.offset(start) / element;
    if !in_buffer(&shadow.buffer, first, count, element) {
        return None;
    }
    let widened = match (under_wine() && matches!(desc.Format.0, 6..=8)).then(|| widened_w(desc.Format.0)).flatten() {
        Some(w) => shadow.widened(start, u64::from(count) * 12, w),
        None => None,
    };
    let buffer = match &widened {
        Some(widened) => {
            desc.Format = widened_format(desc.Format.0)?;
            desc.Anonymous.Buffer.FirstElement = 0;
            widened.buffer.clone()
        }
        None => {
            desc.Anonymous.Buffer.FirstElement = first;
            shadow.buffer.clone()
        }
    };
    // What `seen` enters: a constant buffer's range on the copy `find`
    // gave, any other view's on the copy it reads.
    let seen = match constant {
        Some(found) => Seen::shader_buffer(found, address, len),
        None => Seen::shader_buffer(shadow.clone(), start, len),
    };
    Some(Made {
        buffer,
        desc,
        shadow,
        own,
        widened,
        stride: view_stride(view) | view_kind(view),
        seen,
    })
}

impl HostHeap {
    /// [`HostHeap::mirror`] of each slot of `stale` (a view heap's, whose
    /// records are at `at`), in order: buffer shader views from one lookup
    /// per record ([`resolve`]), the rest by `mirror` itself.
    pub(super) fn sync_quick(&self, stale: &[u32], at: impl Fn(u32) -> u64) {
        let host = host();
        // The records resolved in this pass, by view index and generation:
        // their place in `resolved`, or none for one that `mirror` makes.
        let mut by_key: HashMap<u64, Option<usize>, Hashing> =
            HashMap::with_capacity_and_hasher(stale.len(), Hashing::new(true));
        let mut resolved: Vec<Resolved> = Vec::new();
        // mac: the census updates of the slots made here, entered a batch
        // at a time with `VICTORY_CENSUS_FAST=1` (at once without it).
        let mut census = self.viewed.batch();
        for &index in stale {
            let handle = at(index);
            // SAFETY: a record in title heap memory the caller found mapped.
            let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let [tag, i, generation] = record;
            let found = match host {
                Some(host) if index < self.count && tag == VIEW_TAG => {
                    let key = u64::from(generation) << 32 | u64::from(i);
                    by_key
                        .entry(key)
                        .or_insert_with(|| {
                            let view = view_at(i, generation)?;
                            let made = resolve(&view)?;
                            resolved.push(Resolved { record, view, made });
                            Some(resolved.len() - 1)
                        })
                        .map(|r| (host, r))
                }
                _ => None,
            };
            if !found.is_some_and(|(host, r)| self.show(host, index, handle, &resolved[r], &mut census)) {
                self.mirror(index, handle, false);
            }
            census.enter_if_full();
        }
        // Before the pass returns (its thread reads the census next).
        drop(census);
        // The pins go here, with no lock of the heap held.
    }

    /// Makes slot `index` from `resolved`, the record the title had at
    /// `handle`, unless the slot is made from it already: `mirror`'s steps
    /// under the slot lock for a buffer shader view the host makes. False,
    /// with nothing done, if the title has written another record there
    /// meanwhile (`mirror` then makes the slot from that one). The census
    /// update goes to `census` (mac: entered later with
    /// `VICTORY_CENSUS_FAST=1`).
    fn show(
        &self,
        host: &Host,
        index: u32,
        handle: u64,
        resolved: &Resolved,
        census: &mut crate::bindless::Batch<'_>,
    ) -> bool {
        let Resolved { record, view, made: from } = resolved;
        let key = u64::from(record[2]) << 32 | u64::from(record[1]);
        let mut made = self.made.lock(index);
        // SAFETY: as the caller's read.
        if unsafe { (handle as *const [u32; 3]).read_unaligned() } != *record {
            return false;
        }
        if made.get(index) == key {
            return true;
        }
        self.ready[index as usize].store(0, Ordering::Relaxed);
        self.write_stride(index, from.stride);
        self.note_address(index, Some(view));
        made.set(index, key);
        crate::descriptor::slots::made();
        // SAFETY: a live buffer (held by `from`) and a slot of our heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&from.buffer, Some(&from.desc), self.cpu(index))
        };
        // A shader view has no write mark.
        self.set_mark(index, None);
        census.set_seen(index, Some(from.seen.clone()));
        self.keep(index, Some((from.buffer.clone(), from.held())));
        // A buffer view is ready as soon as it is made (only a texture's
        // view left null waits for a draw).
        self.ready[index as usize].store(key, Ordering::Release);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a slot shows of itself: whether it is made and holds a host
    /// resource, the record `mirror` finds without the lock, its
    /// view-stride entry, and the range the census has for it.
    fn shown(heap: &HostHeap, index: u32) -> (bool, bool, u64, u32, Option<crate::bindless::Range>) {
        let stride = heap.strides.as_ref().map_or(0, |(_, data)| {
            // SAFETY: a dword of the mapped table, one per slot.
            unsafe { (*data as *const u32).add(index as usize).read_volatile() }
        });
        let ready = heap.ready[index as usize].load(Ordering::Acquire);
        (heap.made(index), heap.kept(index), ready, stride, heap.viewed.range_at(index))
    }

    /// Slots a heap sync makes the quick way show what `mirror` makes of
    /// the same records: a buffer's shader views (one record in two slots,
    /// another range of the buffer, structured elements, a view the title
    /// placed off its copy's elements, a constant buffer), and records that
    /// are not the quick way's (a view the host refuses, no record), which
    /// `mirror` makes as before.
    #[test]
    fn quick_slots_show_what_mirror_makes() {
        if host().is_none() {
            return; // no host GPU here
        }
        let resource = crate::resource::test_buffer(256);
        let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10);
        let memory = crate::resource::buffer_bytes(resource).expect("a buffer").0 as u64;
        const SLOTS: u32 = 8;
        // The title's records, 32 bytes a slot.
        let mut records = [0u32; 8 * SLOTS as usize];
        let start = records.as_mut_ptr() as u64;
        let at = |i: u32| start + 32 * u64::from(i);
        let null = std::ptr::null();
        // Format, dimension (buffer), mapping, padding, first element (two
        // words), count, stride, flags, padding.
        // R8_UINT: all 256 elements, then elements 16 to 47.
        crate::descriptor::create_srv(null, resource, &[62, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0], at(0));
        // SAFETY: slots 0 and 1 of `records`.
        unsafe { std::ptr::copy_nonoverlapping(at(0) as *const u8, at(1) as *mut u8, 32) };
        crate::descriptor::create_srv(null, resource, &[62, 1, 0x1688, 0, 16, 0, 32, 0, 0, 0], at(2));
        // Structured, 12-byte elements 1 to 8.
        crate::descriptor::create_srv(null, resource, &[0, 1, 0x1688, 0, 1, 0, 8, 12, 0, 0], at(3));
        // No elements: refused. (Not a view past the buffer's end: its
        // host copy would be a window of the test's own memory.)
        crate::descriptor::create_srv(null, resource, &[62, 1, 0x1688, 0, 0, 0, 0, 0, 0, 0], at(4));
        // Slot 5: no record.
        // A buffer view the title places itself, 12-byte elements from byte
        // 4 of the buffer's memory: off its copy's elements.
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 8, 12, 0, 0]);
        (placed[10], placed[11]) = ((memory + 4) as u32, ((memory + 4) >> 32) as u32);
        let mut buffer = [0u32; 14];
        (buffer[0], buffer[4]) = (1, 256);
        crate::descriptor::create_placed_srv(null, &buffer, &placed, at(6));
        // A constant buffer in the buffer's memory.
        crate::descriptor::create_cbv(null, &[memory as u32, (memory >> 32) as u32, 64, 0], at(7));
        for (i, quick) in [(0, true), (2, true), (3, true), (4, false), (6, true), (7, true)] {
            let view = crate::descriptor::view(at(i)).expect("a view");
            assert_eq!(resolve(&view).is_some(), quick, "slot {i} goes the quick way");
        }
        let (Some(slow), Some(quick)) = (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS)) else {
            release(resource);
            return;
        };
        let all: Vec<u32> = (0..SLOTS).collect();
        for &i in &all {
            slow.mirror(i, at(i), false);
        }
        quick.sync_quick(&all, at);
        for &i in &all {
            assert_eq!(shown(&quick, i), shown(&slow, i), "slot {i}");
        }
        let whole = shown(&quick, 0);
        assert!(whole.0 && whole.1 && whole.2 != 0, "a buffer view is made, held and ready");
        assert_eq!(whole.4.map(|r| (r.1, r.2, r.3)), Some((memory, 256, false)), "its range in the census");
        assert_eq!(shown(&quick, 1), whole, "the same record, the same slot contents");
        assert_eq!(shown(&quick, 2).4.map(|r| (r.1, r.2)), Some((memory + 16, 32)));
        assert_eq!(shown(&quick, 3).4.map(|r| (r.1, r.2)), Some((memory + 12, 96)));
        assert_eq!(shown(&quick, 3).3 & 0xFFFF, 12, "the structured view's stride");
        assert!(shown(&quick, 4).0 && !shown(&quick, 4).1, "the refused view's record is noted, nothing held");
        assert!(!shown(&quick, 5).0, "no record, no slot");
        let own = shown(&quick, 6);
        assert!(own.0 && own.1, "the view off its copy's elements is made and holds its own copy");
        assert_eq!(own.4.map(|r| (r.1, r.2)), Some((memory + 4, 96)));
        assert_ne!(own.4.map(|r| r.0), whole.4.map(|r| r.0), "a copy of its own");
        assert_eq!(shown(&quick, 7).4.map(|r| (r.1, r.2)), Some((memory, 64)), "the constant buffer's range");

        // Slots made from their records already are passed by.
        let before: Vec<_> = all.iter().map(|&i| shown(&quick, i)).collect();
        quick.sync_quick(&all, at);
        assert_eq!(all.iter().map(|&i| shown(&quick, i)).collect::<Vec<_>>(), before);
        // A slot whose record is no longer the one resolved is left as it
        // is, for `mirror` to make.
        let view = crate::descriptor::view(at(2)).expect("a view");
        let made = resolve(&view).expect("the quick way");
        // SAFETY: slot 2 of `records`.
        let record = unsafe { (at(2) as *const [u32; 3]).read_unaligned() };
        let part = Resolved { record, view, made };
        let host = host().expect("a host");
        assert!(!quick.show(host, 3, at(3), &part, &mut quick.viewed.batch()), "slot 3 holds another record");
        assert_eq!(shown(&quick, 3), before[3]);
        assert!(quick.show(host, 2, at(2), &part, &mut quick.viewed.batch()), "slot 2 holds it, and is made from it already");
        assert_eq!(shown(&quick, 2), before[2]);
        drop(part);

        // A record the title clears, and one it replaces, at the next sync.
        // SAFETY: slots 1 and 2 of `records`.
        unsafe {
            std::ptr::write_bytes(at(1) as *mut u8, 0, 32);
            std::ptr::copy_nonoverlapping(at(0) as *const u8, at(2) as *mut u8, 32);
        }
        for i in [1, 2] {
            slow.mirror(i, at(i), false);
        }
        quick.sync_quick(&[1, 2], at);
        for &i in &all {
            assert_eq!(shown(&quick, i), shown(&slow, i), "slot {i} after the rewrite");
        }
        assert!(!shown(&quick, 1).0, "a cleared record leaves its slot");
        assert_eq!(shown(&quick, 2), whole, "the replaced slot shows the whole view");
        drop((slow, quick));
        release(resource);
    }
}
