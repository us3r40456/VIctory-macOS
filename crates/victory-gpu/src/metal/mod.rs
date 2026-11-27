//! Route A.5 (issue #221) native Apple Silicon Metal bridge over a Wine
//! unixlib (`victory_metal.dll` + `victory_metal.so`).
//!
//! Keeps 100% of `victory-hle` on stock CrossOver Wine (no Wine or `ntdll`
//! rebuild) while wrapping title allocations (`XMemVirtualAlloc` and
//! `MapViewOfFile3` physical sections) directly into zero-copy `MTLBuffer`s
//! inside a global `MTLResidencySet`, and providing native Apple Silicon
//! `MTLFXTemporalScaler`, `MTLFXFrameInterpolator` (with separate Plane 0
//! scene + Plane 1 UI), and `CAMetalLayer` presentation.

#[cfg(windows)]
pub mod bridge;
#[cfg(windows)]
pub mod fx;
#[cfg(windows)]
pub mod present;
pub mod registry;

use std::sync::OnceLock;

/// ABI version shared with `tools/macos/victory-metal-unixlib.sh`.
pub const ABI_VERSION: u32 = 2;

pub(crate) const OP_PING: u32 = 0;
pub(crate) const OP_REGISTER_BUFFER: u32 = 1;
pub(crate) const OP_UNREGISTER_BUFFER: u32 = 2;
pub(crate) const OP_FLUSH_RESIDENCY: u32 = 3;
pub(crate) const OP_BENCH_ZERO_COPY: u32 = 4;
pub(crate) const OP_CONVERT_DXIL: u32 = 5;
pub(crate) const OP_STATS: u32 = 6;
pub(crate) const OP_CREATE_TEXTURE: u32 = 7;
pub(crate) const OP_DESTROY_TEXTURE: u32 = 8;
pub(crate) const OP_UPLOAD_TEXTURE: u32 = 9;
pub(crate) const OP_READBACK_TEXTURE: u32 = 10;
pub(crate) const OP_SCALER_CREATE: u32 = 11;
pub(crate) const OP_SCALER_ENCODE: u32 = 12;
pub(crate) const OP_SCALER_DESTROY: u32 = 13;
pub(crate) const OP_INTERP_CREATE: u32 = 14;
pub(crate) const OP_INTERP_ENCODE: u32 = 15;
pub(crate) const OP_INTERP_DESTROY: u32 = 16;
pub(crate) const OP_SURFACE_ATTACH: u32 = 17;
pub(crate) const OP_SURFACE_PRESENT: u32 = 18;
pub(crate) const OP_SURFACE_DESTROY: u32 = 19;

/// Common Apple Metal `MTLPixelFormat` numeric values.
pub mod pixel_format {
    pub const INVALID: u32 = 0;
    pub const R8_UNORM: u32 = 10;
    pub const R16_FLOAT: u32 = 25;
    pub const R32_FLOAT: u32 = 55;
    pub const RG16_FLOAT: u32 = 65;
    pub const RGBA8_UNORM: u32 = 70;
    pub const BGRA8_UNORM: u32 = 80;
    pub const RGB10A2_UNORM: u32 = 90;
    pub const RG11B10_FLOAT: u32 = 92;
    pub const RGBA16_FLOAT: u32 = 115;

    /// Maps a DXGI_FORMAT integer value to the matching `MTLPixelFormat`.
    pub const fn from_dxgi(dxgi: u32) -> Option<u32> {
        match dxgi {
            // DXGI_FORMAT_R16G16B16A16_FLOAT (10), _TYPELESS (9)
            9 | 10 => Some(RGBA16_FLOAT),
            // DXGI_FORMAT_R10G10B10A2_TYPELESS (23), _UNORM (24)
            23 | 24 => Some(RGB10A2_UNORM),
            // DXGI_FORMAT_R11G11B10_FLOAT (26)
            26 => Some(RG11B10_FLOAT),
            // DXGI_FORMAT_R8G8B8A8_TYPELESS (27), _UNORM (28), _UNORM_SRGB (29)
            27..=29 => Some(RGBA8_UNORM),
            // DXGI_FORMAT_R16G16_TYPELESS (33), _FLOAT (34)
            33 | 34 => Some(RG16_FLOAT),
            // DXGI_FORMAT_R32_TYPELESS (39), D32_FLOAT (40), R32_FLOAT (41)
            39..=41 => Some(R32_FLOAT),
            // DXGI_FORMAT_R16_TYPELESS (53), _FLOAT (54)
            53 | 54 => Some(R16_FLOAT),
            // DXGI_FORMAT_R8_TYPELESS (60), _UNORM (61)
            60 | 61 => Some(R8_UNORM),
            // DXGI_FORMAT_B8G8R8A8_UNORM (87), _TYPELESS (90), _UNORM_SRGB (91)
            87 | 90 | 91 => Some(BGRA8_UNORM),
            _ => None,
        }
    }

    /// Bytes per pixel for supported uncompressed MetalFX/present formats.
    pub const fn bytes_per_pixel(mtl: u32) -> usize {
        match mtl {
            R8_UNORM => 1,
            R16_FLOAT => 2,
            R32_FLOAT | RG16_FLOAT | RGBA8_UNORM | BGRA8_UNORM | RGB10A2_UNORM | RG11B10_FLOAT => 4,
            RGBA16_FLOAT => 8,
            _ => 0,
        }
    }
}

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
    pub frame_interpolator: bool,
    pub metal4_fx: bool,
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
            frame_interpolator: raw.flags & (1 << 4) != 0,
            metal4_fx: raw.flags & (1 << 5) != 0,
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

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct CreateTextureArgs {
    pub width: u32,
    pub height: u32,
    pub pixel_format: u32,
    pub usage: u32,
    pub storage_mode: u32,
    pub status: i32,
    pub handle: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct DestroyHandleArgs {
    pub handle: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct TextureIoArgs {
    pub texture_handle: u64,
    pub buffer_handle: u64,
    pub cpu_ptr: u64,
    pub offset: u64,
    pub bytes_per_row: u32,
    pub width: u32,
    pub height: u32,
    pub status: i32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct ScalerCreateArgs {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_format: u32,
    pub depth_format: u32,
    pub motion_format: u32,
    pub output_format: u32,
    pub reactive_format: u32,
    pub flags: u32,
    pub handle: u64,
    pub status: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct ScalerEncodeArgs {
    pub scaler_handle: u64,
    pub color_tex: u64,
    pub depth_tex: u64,
    pub motion_tex: u64,
    pub reactive_tex: u64,
    pub output_tex: u64,
    pub input_content_width: u32,
    pub input_content_height: u32,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub mv_scale_x: f32,
    pub mv_scale_y: f32,
    pub pre_exposure: f32,
    pub flags: u32,
    pub gpu_ns: u64,
    pub wall_ns: u64,
    pub status: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct InterpCreateArgs {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_format: u32,
    pub output_format: u32,
    pub depth_format: u32,
    pub motion_format: u32,
    pub ui_format: u32,
    pub flags: u32,
    pub handle: u64,
    pub status: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct InterpEncodeArgs {
    pub interp_handle: u64,
    pub color_tex: u64,
    pub prev_color_tex: u64,
    pub depth_tex: u64,
    pub motion_tex: u64,
    pub ui_tex: u64,
    pub output_tex: u64,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub mv_scale_x: f32,
    pub mv_scale_y: f32,
    pub delta_time: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub fov_degrees: f32,
    pub flags: u32,
    pub interpolated: u32,
    pub gpu_ns: u64,
    pub wall_ns: u64,
    pub status: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct SurfaceAttachArgs {
    pub width: u32,
    pub height: u32,
    pub pixel_format: u32,
    pub flags: u32,
    pub handle: u64,
    pub status: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct SurfacePresentArgs {
    pub surface_handle: u64,
    pub color_tex: u64,
    pub ui_tex: u64,
    pub after_min_duration_ns: u64,
    pub flags: u32,
    pub status: i32,
    pub gpu_ns: u64,
    pub wall_ns: u64,
}

/// Whether `VICTORY_METAL_BRIDGE=1` asks to register title allocations into
/// the zero-copy Metal `MTLResidencySet`.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_METAL_BRIDGE"))
}

/// Whether `VICTORY_METAL_FRAMEGEN=1` activates live Apple `MTLFXFrameInterpolator`
/// frame generation at presentation (with separate Plane 0 scene + Plane 1 UI).
pub fn framegen_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_METAL_FRAMEGEN"))
}

/// Whether `VICTORY_METAL_PRESENT=1` attaches a native `CAMetalLayer` overlay
/// view directly onto the game's macOS window for presentation.
pub fn present_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_METAL_PRESENT"))
}

/// Whether `VICTORY_UPSCALER=metalfx` or `VICTORY_METAL_FX=1` requests serving
/// the title's FidelityFX upscaler directly via Apple `MTLFXTemporalScaler`.
pub fn metalfx_upscaler_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::switch::on("VICTORY_METAL_FX")
            || std::env::var("VICTORY_UPSCALER")
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("metalfx"))
    })
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

#[cfg(test)]
mod tests {
    use super::pixel_format::*;

    #[test]
    fn dxgi_to_metal_pixel_format_mapping() {
        assert_eq!(from_dxgi(28), Some(RGBA8_UNORM));
        assert_eq!(from_dxgi(87), Some(BGRA8_UNORM));
        assert_eq!(from_dxgi(10), Some(RGBA16_FLOAT));
        assert_eq!(from_dxgi(40), Some(R32_FLOAT));
        assert_eq!(from_dxgi(34), Some(RG16_FLOAT));
        assert_eq!(from_dxgi(61), Some(R8_UNORM));
        assert_eq!(bytes_per_pixel(RGBA8_UNORM), 4);
        assert_eq!(bytes_per_pixel(RGBA16_FLOAT), 8);
    }
}
