//! Route A.5 (issue #221) native Apple Silicon Metal bridge over a Wine
//! unixlib (`victory_metal.dll` + `victory_metal.so`).
//!
//! Keeps 100% of `victory-hle` on stock CrossOver Wine (no Wine or `ntdll`
//! rebuild) while wrapping title allocations (`XMemVirtualAlloc` and
//! `MapViewOfFile3` physical sections) directly into zero-copy `MTLBuffer`s
//! inside a global `MTLResidencySet`.
//!
//! Controlled by `VICTORY_METAL_BRIDGE=1` (off by default until the full
//! Metal command encoder is wired in Step 2B-3), with optional size/budget
//! filters `VICTORY_METAL_MIN_MB` and `VICTORY_METAL_BUDGET_MB`.

pub mod registry;

#[cfg(windows)]
pub mod bridge;

use std::sync::OnceLock;

/// ABI version shared with `tools/macos/victory-metal-unixlib.sh`.
pub const ABI_VERSION: u32 = 1;

pub(crate) const OP_PING: u32 = 0;
pub(crate) const OP_REGISTER_BUFFER: u32 = 1;
pub(crate) const OP_UNREGISTER_BUFFER: u32 = 2;
pub(crate) const OP_FLUSH_RESIDENCY: u32 = 3;
pub(crate) const OP_BENCH_ZERO_COPY: u32 = 4;
pub(crate) const OP_CONVERT_DXIL: u32 = 5;
pub(crate) const OP_STATS: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PingArgs {
    pub abi_version: u32,
    pub flags: u32,
    pub recommended_vram: u64,
    pub max_buffer_len: u64,
    pub device_name: [u8; 128],
}

impl Default for PingArgs {
    fn default() -> Self {
        Self {
            abi_version: 0,
            flags: 0,
            recommended_vram: 0,
            max_buffer_len: 0,
            device_name: [0; 128],
        }
    }
}

/// Decoded capabilities reported by the Apple Silicon Metal unixlib.
#[derive(Debug, Clone)]
pub struct PingInfo {
    pub abi_version: u32,
    pub unified_memory: bool,
    pub residency_set: bool,
    pub metalfx: bool,
    pub irconverter: bool,
    pub recommended_vram: u64,
    pub max_buffer_len: u64,
    pub device_name: String,
}

impl PingInfo {
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) fn from_raw(raw: &PingArgs) -> Self {
        let nul = raw
            .device_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw.device_name.len());
        let device_name = String::from_utf8_lossy(&raw.device_name[..nul]).into_owned();
        Self {
            abi_version: raw.abi_version,
            unified_memory: raw.flags & (1 << 0) != 0,
            residency_set: raw.flags & (1 << 1) != 0,
            metalfx: raw.flags & (1 << 2) != 0,
            irconverter: raw.flags & (1 << 3) != 0,
            recommended_vram: raw.recommended_vram,
            max_buffer_len: raw.max_buffer_len,
            device_name,
        }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct RegisterBufferArgs {
    pub cpu_addr: u64,
    pub length: u64,
    pub backing_key: u64,
    pub flags: u32,
    pub reserved: u32,
    pub handle: u64,
    pub gpu_address: u64,
    pub reused: u32,
    pub status: i32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct UnregisterBufferArgs {
    pub handle: u64,
    pub flags: u32,
    pub remaining_refs: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct FlushResidencyArgs {
    pub pending_before: u32,
    pub total_allocations: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct BenchArgs {
    pub handle: u64,
    pub offset: u64,
    pub length: u64,
    pub mode: u32,
    pub seed: u32,
    pub checksum_out: u64,
    pub gpu_ns: u64,
    pub wall_ns: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct ConvertDxilArgs {
    pub dxil_ptr: u64,
    pub dxil_len: u64,
    pub entry_ptr: u64,
    pub out_ptr: u64,
    pub out_cap: u64,
    pub out_len: u64,
    pub stage: u32,
    pub status: i32,
}

/// Live counters returned by `OP_STATS`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub active_buffers: u64,
    pub active_bytes: u64,
    pub total_registered: u64,
    pub total_reused: u64,
    pub residency_commits: u64,
    pub dxil_converted: u64,
}

/// Whether `VICTORY_METAL_BRIDGE=1` asks to register title allocations into
/// the zero-copy Metal `MTLResidencySet`.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_METAL_BRIDGE"))
}

#[cfg(windows)]
fn min_bytes() -> Option<usize> {
    static MIN: OnceLock<Option<usize>> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_METAL_MIN_MB")
            .ok()
            .and_then(|s| crate::alias_plan::parse_mib(&s))
    })
}

#[cfg(windows)]
fn budget_bytes() -> Option<usize> {
    static BUDGET: OnceLock<Option<usize>> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("VICTORY_METAL_BUDGET_MB")
            .ok()
            .and_then(|s| crate::alias_plan::parse_mib(&s))
    })
}

/// Called after a read-write, 64 KiB-aligned title memory commit (e.g.
/// `XMemVirtualAlloc` or a `MapViewOfFile3` physical-section view) when
/// `VICTORY_METAL_BRIDGE=1`. Wraps the range zero-copy into the global
/// `MTLResidencySet` while enforcing optional size/budget limits and the
/// `#221` #4 single-canonical-`MTLBuffer` invariant.
#[cfg(windows)]
pub fn note_committed(base: usize, len: usize, backing: registry::BackingKey) {
    if !enabled() || base == 0 || len < crate::watch::PAGE {
        return;
    }
    let Some(b) = bridge::global() else {
        return;
    };
    let current = b.stats().map_or(0, |s| s.active_bytes as usize);
    if !crate::alias_plan::region_within_limits(len, current, min_bytes(), budget_bytes()) {
        return;
    }
    if let Some((buf, reused)) = b.register_buffer(base as u64, len as u64, backing, false)
        && len >= 16 << 20
    {
        tracing::info!(
            base = format_args!("{base:#x}"),
            len_mb = len >> 20,
            handle = buf.handle,
            gpu = format_args!("{:#x}", buf.gpu_address),
            reused,
            "victory_metal: large title heap registered zero-copy in MTLResidencySet"
        );
    }
}

#[cfg(not(windows))]
pub fn note_committed(_base: usize, _len: usize, _backing: registry::BackingKey) {}

/// Called when a previously committed title view is released.
#[cfg(windows)]
pub fn note_released(base: usize) {
    if !enabled() || base == 0 {
        return;
    }
    if let Some(b) = bridge::global() {
        let _ = b.unregister_view(base as u64, false);
    }
}

#[cfg(not(windows))]
pub fn note_released(_base: usize) {}
