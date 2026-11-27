//! Canonical zero-copy backing registry for Apple Silicon Metal buffers.
//!
//! Enforces the single-canonical-`MTLBuffer`-per-physical-backing invariant
//! (issue #221): when two Windows virtual address views map the same physical
//! section (`MapViewOfFile3` / `MapViewOfFile`), only ONE `MTLBuffer` wraps
//! the physical pages inside the global `MTLResidencySet`. Any secondary view
//! resolves to the canonical buffer's handle and GPU address (CPU-only view)
//! so the GPU never observes stale caches across aliasing `MTLBuffer`s in one
//! command buffer.

use std::collections::{BTreeMap, HashMap};

/// Identifies the physical backing of a title memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackingKey {
    /// Standalone `VirtualAlloc` reservation/commit at `base`.
    VirtualAlloc(u64),
    /// Pagefile-backed section (`CreateFileMapping` / `XMemMapPhysicalPages`)
    /// at `(section_id, section_offset)`.
    PhysicalSection { section_id: u64, offset: u64 },
}

impl BackingKey {
    /// Packs the backing key into a non-zero 64-bit token for the unixlib ABI.
    pub fn abi_token(self) -> u64 {
        match self {
            // Bit 63 clear: VirtualAlloc addresses on x86-64 user space (< 2^47).
            Self::VirtualAlloc(base) => base & !(1u64 << 63),
            // Bit 63 set: deterministic FNV-1a mix of (section_id, offset).
            Self::PhysicalSection { section_id, offset } => {
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for b in section_id.to_le_bytes().into_iter().chain(offset.to_le_bytes()) {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                (h | (1u64 << 63)).max(1u64 << 63 | 1)
            }
        }
    }
}

/// One canonical `MTLBuffer` backing one or more CPU virtual address views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalBuffer {
    pub handle: u64,
    pub canonical_cpu_addr: u64,
    pub length: u64,
    pub gpu_address: u64,
    pub backing: BackingKey,
    pub refcount: u32,
}

/// Result of resolving a title CPU address into a registered zero-copy buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBuffer {
    pub handle: u64,
    pub gpu_address: u64,
    pub offset: u64,
    pub remaining: u64,
}

#[derive(Debug, Clone, Copy)]
struct ViewSpan {
    end: u64,
    handle: u64,
}

/// Action the caller should take against the Metal unixlib after consulting
/// [`CanonicalRegistry::plan_register`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterAction {
    /// Physical backing is already represented by a canonical `MTLBuffer`.
    /// Attach this CPU view to it without creating a duplicate `MTLBuffer`.
    ReuseCanonical(CanonicalBuffer),
    /// Create a new canonical `MTLBuffer` over `(cpu_addr, length)` and insert
    /// it into the global `MTLResidencySet`.
    CreateNew { abi_backing_key: u64 },
}

/// Pure-Rust registry tracking canonical zero-copy `MTLBuffer`s and all CPU
/// virtual address views pointing to them.
#[derive(Default)]
pub struct CanonicalRegistry {
    by_backing: HashMap<BackingKey, u64>,
    by_handle: HashMap<u64, CanonicalBuffer>,
    /// Start CPU address -> `(end, canonical handle)` for O(log N) address resolution.
    views: BTreeMap<u64, ViewSpan>,
    active_bytes: u64,
}

impl CanonicalRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decides whether registering `(cpu_addr, length, backing)` should create
    /// a new canonical `MTLBuffer` or reuse an existing one for `backing`.
    pub fn plan_register(&self, backing: BackingKey) -> RegisterAction {
        if let Some(&handle) = self.by_backing.get(&backing)
            && let Some(&buf) = self.by_handle.get(&handle)
        {
            return RegisterAction::ReuseCanonical(buf);
        }
        RegisterAction::CreateNew {
            abi_backing_key: backing.abi_token(),
        }
    }

    /// Records a newly created canonical `MTLBuffer` returned by the unixlib.
    pub fn insert_canonical(
        &mut self,
        cpu_addr: u64,
        length: u64,
        backing: BackingKey,
        handle: u64,
        gpu_address: u64,
    ) -> CanonicalBuffer {
        let entry = CanonicalBuffer {
            handle,
            canonical_cpu_addr: cpu_addr,
            length,
            gpu_address,
            backing,
            refcount: 1,
        };
        self.by_backing.insert(backing, handle);
        self.by_handle.insert(handle, entry);
        self.views.insert(
            cpu_addr,
            ViewSpan {
                end: cpu_addr.saturating_add(length),
                handle,
            },
        );
        self.active_bytes = self.active_bytes.saturating_add(length);
        entry
    }

    /// Attaches a secondary CPU view (`cpu_addr..cpu_addr+length`) to an
    /// already-registered canonical `MTLBuffer` (no second `MTLBuffer` created).
    pub fn attach_view(&mut self, cpu_addr: u64, length: u64, backing: BackingKey) -> Option<CanonicalBuffer> {
        let &handle = self.by_backing.get(&backing)?;
        let entry = self.by_handle.get_mut(&handle)?;
        entry.refcount += 1;
        let snapshot = *entry;
        self.views.insert(
            cpu_addr,
            ViewSpan {
                end: cpu_addr.saturating_add(length),
                handle,
            },
        );
        Some(snapshot)
    }

    /// Resolves any title CPU address `[addr, addr + len)` to its canonical
    /// `MTLBuffer` handle and exact GPU virtual address.
    pub fn resolve(&self, addr: u64, len: u64) -> Option<ResolvedBuffer> {
        let (&start, &span) = self.views.range(..=addr).next_back()?;
        let end = addr.checked_add(len)?;
        if end > span.end {
            return None;
        }
        let buf = self.by_handle.get(&span.handle)?;
        let offset = addr - start;
        if offset.checked_add(len)? > buf.length {
            return None;
        }
        Some(ResolvedBuffer {
            handle: buf.handle,
            gpu_address: buf.gpu_address.checked_add(offset)?,
            offset,
            remaining: buf.length - offset,
        })
    }

    /// Unregisters the CPU view starting at `cpu_addr`. Returns `Some(handle)`
    /// when the last reference drops so the caller removes the canonical
    /// `MTLBuffer` from `MTLResidencySet` and frees it in the unixlib.
    pub fn remove_view(&mut self, cpu_addr: u64) -> Option<u64> {
        let span = self.views.remove(&cpu_addr)?;
        let entry = self.by_handle.get_mut(&span.handle)?;
        if entry.refcount > 1 {
            entry.refcount -= 1;
            return None;
        }
        let removed = self.by_handle.remove(&span.handle)?;
        self.by_backing.remove(&removed.backing);
        self.active_bytes = self.active_bytes.saturating_sub(removed.length);
        Some(removed.handle)
    }

    pub fn active_buffers(&self) -> usize {
        self.by_handle.len()
    }

    pub fn active_bytes(&self) -> u64 {
        self.active_bytes
    }

    pub fn active_views(&self) -> usize {
        self.views.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_physical_section_views_share_one_canonical_buffer() {
        let mut reg = CanonicalRegistry::new();
        let backing = BackingKey::PhysicalSection {
            section_id: 42,
            offset: 0x10_0000,
        };
        let len = 174 << 20; // 174 MiB constants heap

        // First view creates the canonical MTLBuffer.
        let token = match reg.plan_register(backing) {
            RegisterAction::CreateNew { abi_backing_key } => abi_backing_key,
            RegisterAction::ReuseCanonical(_) => panic!("expected fresh canonical creation"),
        };
        assert_ne!(token, 0);
        reg.insert_canonical(0x0f85_0000, len, backing, 1, 0x0001_0000_0000);
        assert_eq!(reg.active_buffers(), 1);
        assert_eq!(reg.active_bytes(), len);

        // Second view of the exact same physical section MUST reuse canonical buffer #1.
        match reg.plan_register(backing) {
            RegisterAction::ReuseCanonical(buf) => assert_eq!(buf.handle, 1),
            RegisterAction::CreateNew { .. } => panic!("duplicate MTLBuffer would violate #221 #4"),
        }
        let attached = reg.attach_view(0x4000_0000, len, backing).unwrap();
        assert_eq!(attached.handle, 1);
        assert_eq!(attached.refcount, 2);
        assert_eq!(reg.active_buffers(), 1, "still exactly one canonical MTLBuffer");
        assert_eq!(reg.active_views(), 2);

        // Addresses in EITHER CPU view resolve to the exact same canonical handle & GPU address.
        let r1 = reg.resolve(0x0f85_0000 + 0x1234, 256).unwrap();
        let r2 = reg.resolve(0x4000_0000 + 0x1234, 256).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1.handle, 1);
        assert_eq!(r1.gpu_address, 0x0001_0000_1234);
        assert_eq!(r1.offset, 0x1234);

        // Removing the first view keeps the canonical buffer alive for view 2.
        assert_eq!(reg.remove_view(0x0f85_0000), None);
        assert_eq!(reg.active_buffers(), 1);
        assert!(reg.resolve(0x0f85_0000 + 0x1234, 256).is_none());
        assert_eq!(reg.resolve(0x4000_0000 + 0x1234, 256), Some(r2));

        // Removing the last view returns handle 1 for unixlib teardown.
        assert_eq!(reg.remove_view(0x4000_0000), Some(1));
        assert_eq!(reg.active_buffers(), 0);
        assert_eq!(reg.active_bytes(), 0);
    }

    #[test]
    fn resolve_rejects_out_of_bounds_ranges() {
        let mut reg = CanonicalRegistry::new();
        let backing = BackingKey::VirtualAlloc(0x2_0000_0000);
        reg.insert_canonical(0x2_0000_0000, 65536, backing, 7, 0x9000_0000);
        assert!(reg.resolve(0x2_0000_0000, 65536).is_some());
        assert!(reg.resolve(0x2_0000_0000, 65537).is_none());
        assert!(reg.resolve(0x2_0000_ff00, 512).is_none());
        assert!(reg.resolve(0x1_ffff_0000, 64).is_none());
    }
}
