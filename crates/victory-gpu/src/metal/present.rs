//! Live MetalFX Frame Generation (`MTLFXFrameInterpolator`) and native
//! `CAMetalLayer` presentation wired into `Host::present`.
//!
//! Controlled by:
//! - `VICTORY_METAL_FRAMEGEN=1`: generates a hardware-interpolated mid-frame
//!   between consecutive presented frames using Apple's `MTLFXFrameInterpolator`
//!   with separate Plane 0 (`colorTexture`) and Plane 1 (`uiTexture`) so UI and
//!   subtitles stay sharp with zero ghosting.
//! - `VICTORY_METAL_PRESENT=1`: attaches a native `CAMetalLayer` directly onto
//!   the macOS window (`NSWindow`), bypassing `IDXGISwapChain3::Present` while
//!   leaving Win32/Cocoa input untouched.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{DXGI_PRESENT, IDXGISwapChain3};

use super::bridge::{
    self, FrameInterpolatorDesc, FrameInterpolatorDispatch, MetalBridge,
};
use super::fx;
use super::pixel_format as pf;
use crate::host::{Host, HostList, transition};

/// Interpolated frames presented and cumulative GPU time (µs) since the last
/// 5-second frame-rate report.
static INTERP_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_GPU_US: AtomicU64 = AtomicU64::new(0);

/// Formats a short suffix for the 5-second frame-rate log when MetalFX Frame
/// Generation has produced interpolated frames since the last report.
pub(crate) fn frame_line(real_frames: f64) -> String {
    let generated = INTERP_FRAMES.swap(0, Ordering::Relaxed);
    let gpu_us = INTERP_GPU_US.swap(0, Ordering::Relaxed);
    if generated == 0 {
        return String::new();
    }
    let ms = (gpu_us as f64 / 1000.0) / (generated as f64);
    let ratio = (real_frames + generated as f64) / real_frames.max(1.0);
    format!("; MetalFX FG +{generated} interp frames ({ratio:.2}x displayed, {ms:.2} ms GPU/interp)")
}

/// Persistently mapped D3D12 linear staging buffer (READBACK or UPLOAD).
struct Staging {
    resource: ID3D12Resource,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: accessed only under the presenter mutex after GPU completion.
unsafe impl Send for Staging {}

impl Staging {
    fn new(device: &ID3D12Device, heap: D3D12_HEAP_TYPE, len: usize) -> Option<Self> {
        let state = if heap == D3D12_HEAP_TYPE_READBACK {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else {
            D3D12_RESOURCE_STATE_GENERIC_READ
        };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain committed linear buffer + persistent map.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: len as u64,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut res,
                )
                .ok()?;
            let resource = res?;
            let mut ptr = std::ptr::null_mut();
            resource.Map(0, None, Some(&mut ptr)).ok()?;
            Some(Self {
                resource,
                ptr: ptr.cast(),
                len,
            })
        }
    }
}

/// Owned Metal texture + shape metadata.
#[derive(Default)]
struct MetalTex {
    handle: u64,
    width: u32,
    height: u32,
    format: u32,
}

impl MetalTex {
    fn ensure(&mut self, b: &MetalBridge, w: u32, h: u32, fmt: u32) -> Option<u64> {
        if self.handle != 0 && (self.width, self.height, self.format) == (w, h, fmt) {
            return Some(self.handle);
        }
        if self.handle != 0 {
            b.destroy_texture(self.handle);
            self.handle = 0;
        }
        let h_tex = b.create_texture(w, h, fmt, false)?;
        self.handle = h_tex;
        self.width = w;
        self.height = h;
        self.format = fmt;
        Some(h_tex)
    }

    fn destroy(&mut self, b: &MetalBridge) {
        if self.handle != 0 {
            b.destroy_texture(self.handle);
            self.handle = 0;
        }
    }
}

/// Outcome of `MetalPresenter::present_with_metal`.
pub(crate) struct PresentOutcome {
    /// True if `CAMetalLayer` already presented the frame(s) directly so DXGI
    /// `Present` should be skipped.
    pub handled_present: bool,
    /// Number of extra interpolated frames displayed (0 or 1).
    pub extra_frames: u64,
}

/// Live MetalFX Frame Interpolator + optional native `CAMetalLayer` presenter.
pub(crate) struct MetalPresenter {
    want_framegen: bool,
    want_surface: bool,
    rb_color: Option<Staging>,
    rb_ui: Option<Staging>,
    up_interp: Option<Staging>,
    color_tex: MetalTex,
    ui_tex: MetalTex,
    interp_out_tex: MetalTex,
    interp_handle: u64,
    interp_shape: (u32, u32, u32, u32), // (w, h, color_fmt, ui_fmt)
    surface_handle: u64,
    surface_shape: (u32, u32, u32), // (w, h, fmt)
    surface_failed: bool,
    last_frame_at: Option<Instant>,
    logged_active: bool,
}

impl MetalPresenter {
    pub(crate) fn from_env() -> Option<Self> {
        let want_framegen = super::framegen_enabled();
        let want_surface = super::present_enabled();
        if !want_framegen && !want_surface {
            return None;
        }
        let b = bridge::global()?;
        let can_fg = want_framegen && b.info().frame_interpolator;
        if want_framegen && !can_fg {
            tracing::warn!(
                "VICTORY_METAL_FRAMEGEN=1 requested, but MTLFXFrameInterpolator is unsupported on this OS/GPU"
            );
        }
        if !can_fg && !want_surface {
            return None;
        }
        Some(Self {
            want_framegen: can_fg,
            want_surface,
            rb_color: None,
            rb_ui: None,
            up_interp: None,
            color_tex: MetalTex::default(),
            ui_tex: MetalTex::default(),
            interp_out_tex: MetalTex::default(),
            interp_handle: 0,
            interp_shape: (0, 0, 0, 0),
            surface_handle: 0,
            surface_shape: (0, 0, 0),
            surface_failed: false,
            last_frame_at: None,
            logged_active: false,
        })
    }

    /// Runs native `MTLFXFrameInterpolator` (when `VICTORY_METAL_FRAMEGEN=1`)
    /// and/or direct `CAMetalLayer` presentation (`VICTORY_METAL_PRESENT=1`)
    /// for Plane 0 (`color`) + optional Plane 1 (`ui`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        host: &Host,
        copy: &HostList,
        swap_chain: &IDXGISwapChain3,
        color: (&ID3D12Resource, D3D12_RESOURCE_STATES, DXGI_FORMAT),
        ui: Option<(&ID3D12Resource, D3D12_RESOURCE_STATES, DXGI_FORMAT)>,
        out_width: u32,
        out_height: u32,
        swap_format: DXGI_FORMAT,
    ) -> PresentOutcome {
        let none = PresentOutcome {
            handled_present: false,
            extra_frames: 0,
        };
        let Some(b) = bridge::global() else {
            return none;
        };
        // SAFETY: live D3D12 resource.
        let c_desc = unsafe { color.0.GetDesc() };
        let (cw, ch) = (c_desc.Width as u32, c_desc.Height);
        let Some(c_mtl) = pf::from_dxgi(color.2.0 as u32) else {
            return none;
        };
        let c_bpp = pf::bytes_per_pixel(c_mtl) as u32;
        if c_bpp == 0 || cw == 0 || ch == 0 {
            return none;
        }
        let c_pitch = (cw * c_bpp).next_multiple_of(256);
        let c_bytes = (c_pitch * ch) as usize;
        if self.rb_color.as_ref().is_none_or(|s| s.len < c_bytes) {
            self.rb_color = Staging::new(&host.device, D3D12_HEAP_TYPE_READBACK, c_bytes);
        }
        let Some(rb_c) = &self.rb_color else {
            return none;
        };

        // Optional Plane 1 UI metadata + staging buffer.
        let ui_info = ui.and_then(|(u_tex, u_st, u_fmt)| {
            // SAFETY: live D3D12 resource.
            let ud = unsafe { u_tex.GetDesc() };
            let (uw, uh) = (ud.Width as u32, ud.Height);
            let u_mtl = pf::from_dxgi(u_fmt.0 as u32)?;
            let u_bpp = pf::bytes_per_pixel(u_mtl) as u32;
            if u_bpp == 0 || uw == 0 || uh == 0 {
                return None;
            }
            let u_pitch = (uw * u_bpp).next_multiple_of(256);
            let u_bytes = (u_pitch * uh) as usize;
            if self.rb_ui.as_ref().is_none_or(|s| s.len < u_bytes) {
                self.rb_ui = Staging::new(&host.device, D3D12_HEAP_TYPE_READBACK, u_bytes);
            }
            self.rb_ui.as_ref()?;
            Some((u_tex, u_st, u_fmt, uw, uh, u_mtl, u_pitch))
        });

        // 1) Copy Plane 0 (+ Plane 1) from D3D12 into persistent UMA readback staging.
        copy.reset();
        let list = &copy.list;
        // SAFETY: live D3D12 textures and persistently mapped readback buffers.
        unsafe {
            if color.1 != D3D12_RESOURCE_STATE_COPY_SOURCE {
                list.ResourceBarrier(&[transition(
                    color.0,
                    color.1,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            let src_c = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(color.0),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let dst_c = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rb_c.resource),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: color.2,
                            Width: cw,
                            Height: ch,
                            Depth: 1,
                            RowPitch: c_pitch,
                        },
                    },
                },
            };
            list.CopyTextureRegion(&dst_c, 0, 0, 0, &src_c, None);
            if color.1 != D3D12_RESOURCE_STATE_COPY_SOURCE {
                list.ResourceBarrier(&[transition(
                    color.0,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    color.1,
                )]);
            }

            if let Some((u_tex, u_st, u_fmt, uw, uh, _, u_pitch)) = ui_info
                && let Some(rb_u) = &self.rb_ui
            {
                if u_st != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[transition(
                        u_tex,
                        u_st,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                }
                let src_u = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(u_tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let dst_u = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rb_u.resource),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: u_fmt,
                                Width: uw,
                                Height: uh,
                                Depth: 1,
                                RowPitch: u_pitch,
                            },
                        },
                    },
                };
                list.CopyTextureRegion(&dst_u, 0, 0, 0, &src_u, None);
                if u_st != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[transition(
                        u_tex,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        u_st,
                    )]);
                }
            }
        }
        copy.close();
        host.execute(&[copy]);

        // 2) Upload Plane 0 (and optional Plane 1) into Metal private textures.
        let Some(c_handle) = self.color_tex.ensure(b, cw, ch, c_mtl) else {
            return none;
        };
        let Some(rb_c) = &self.rb_color else {
            return none;
        };
        if !b.upload_texture_from_ptr(c_handle, rb_c.ptr, c_pitch, cw, ch) {
            return none;
        }
        let mut u_handle = 0u64;
        let mut u_mtl_fmt = 0u32;
        if let Some((_, _, _, uw, uh, u_mtl, u_pitch)) = ui_info
            && let Some(rb_u) = &self.rb_ui
            && let Some(h) = self.ui_tex.ensure(b, uw, uh, u_mtl)
            && b.upload_texture_from_ptr(h, rb_u.ptr, u_pitch, uw, uh)
        {
            u_handle = h;
            u_mtl_fmt = u_mtl;
        }

        // Measure real inter-frame delta for smooth interpolation + pacing.
        let now = Instant::now();
        let dt_s = self
            .last_frame_at
            .replace(now)
            .map_or(1.0 / 30.0, |prev| now.duration_since(prev).as_secs_f32())
            .clamp(1.0 / 240.0, 0.5);

        // 3) Run Apple MTLFXFrameInterpolator when VICTORY_METAL_FRAMEGEN=1.
        let mut interp_ready = false;
        if self.want_framegen {
            let wanted_shape = (cw, ch, c_mtl, u_mtl_fmt);
            if self.interp_handle == 0 || self.interp_shape != wanted_shape {
                if self.interp_handle != 0 {
                    b.destroy_frame_interpolator(self.interp_handle);
                    self.interp_handle = 0;
                }
                match b.create_frame_interpolator(&FrameInterpolatorDesc {
                    input_width: (cw / 2).max(1),
                    input_height: (ch / 2).max(1),
                    output_width: cw,
                    output_height: ch,
                    color_format: c_mtl,
                    output_format: c_mtl,
                    depth_format: pf::R32_FLOAT,
                    motion_format: pf::RG16_FLOAT,
                    ui_format: u_mtl_fmt,
                }) {
                    Ok(h) => {
                        self.interp_handle = h;
                        self.interp_shape = wanted_shape;
                    }
                    Err(e) => {
                        tracing::warn!(err = e, cw, ch, "MTLFXFrameInterpolator creation failed");
                        self.want_framegen = false;
                    }
                }
            }
            if self.interp_handle != 0
                && let Some(out_h) = self.interp_out_tex.ensure(b, cw, ch, c_mtl)
            {
                let hint = fx::latest_motion_hint().unwrap_or_default();
                match b.encode_frame_interpolator(
                    self.interp_handle,
                    &FrameInterpolatorDispatch {
                        color_tex: c_handle,
                        prev_color_tex: 0,
                        depth_tex: 0,
                        motion_tex: 0, // built-in Metal optical flow
                        ui_tex: u_handle,
                        output_tex: out_h,
                        jitter_x: hint.jitter[0],
                        jitter_y: hint.jitter[1],
                        mv_scale_x: hint.motion_scale[0],
                        mv_scale_y: hint.motion_scale[1],
                        delta_time: dt_s,
                        near_plane: hint.near_plane,
                        far_plane: hint.far_plane,
                        fov_degrees: hint.fov_degrees,
                        reset: hint.reset,
                        depth_reversed: hint.depth_reversed,
                        ui_composited: false,
                        skip_prev_update: false,
                    },
                ) {
                    Ok(res) if res.interpolated => {
                        interp_ready = true;
                        INTERP_FRAMES.fetch_add(1, Ordering::Relaxed);
                        INTERP_GPU_US.fetch_add(res.gpu_ns / 1000, Ordering::Relaxed);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(err = e, "MTLFXFrameInterpolator encode failed");
                    }
                }
            }
        }

        // 4) Native CAMetalLayer presentation (VICTORY_METAL_PRESENT=1).
        if self.want_surface && !self.surface_failed {
            let surf_shape = (out_width, out_height, pf::BGRA8_UNORM);
            if self.surface_handle == 0 || self.surface_shape != surf_shape {
                if self.surface_handle != 0 {
                    b.destroy_surface(self.surface_handle);
                    self.surface_handle = 0;
                }
                match b.attach_surface(out_width, out_height, pf::BGRA8_UNORM, false) {
                    Ok(h) => {
                        self.surface_handle = h;
                        self.surface_shape = surf_shape;
                        tracing::info!(
                            width = out_width,
                            height = out_height,
                            framegen = self.want_framegen,
                            "victory_metal: attached native CAMetalLayer overlay to window"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            err = e,
                            "victory_metal: CAMetalLayer attach unavailable; falling back to DXGI"
                        );
                        self.surface_failed = true;
                    }
                }
            }
            if self.surface_handle != 0 {
                if !self.logged_active {
                    self.logged_active = true;
                    tracing::info!(
                        framegen = self.want_framegen,
                        has_ui = u_handle != 0,
                        "victory_metal: live native CAMetalLayer presentation active"
                    );
                }
                let mut extra = 0u64;
                if interp_ready {
                    // Present interpolated mid-frame immediately (UI already composited by MetalFX).
                    if b.present_surface(self.surface_handle, self.interp_out_tex.handle, 0, 0, false)
                        .is_ok()
                    {
                        extra = 1;
                    }
                }
                // Present real frame N (Plane 0 + Plane 1) smoothly paced at half the interval.
                let half_ns = if extra > 0 {
                    ((dt_s * 0.5).clamp(0.002, 0.08) * 1e9) as u64
                } else {
                    0
                };
                if b.present_surface(self.surface_handle, c_handle, u_handle, half_ns, false).is_ok()
                {
                    return PresentOutcome {
                        handled_present: true,
                        extra_frames: extra,
                    };
                }
            }
        }

        // 5) Hybrid DXGI Frame Generation: present the interpolated mid-frame
        //    to the DXGI swap chain right before Host::present presents real frame N.
        if interp_ready && cw == out_width && ch == out_height && color.2 == swap_format {
            if !self.logged_active {
                self.logged_active = true;
                tracing::info!(
                    width = cw,
                    height = ch,
                    has_ui = u_handle != 0,
                    "victory_metal: live MTLFXFrameInterpolator active (hybrid DXGI 2x present)"
                );
            }
            if self.up_interp.as_ref().is_none_or(|s| s.len < c_bytes) {
                self.up_interp = Staging::new(&host.device, D3D12_HEAP_TYPE_UPLOAD, c_bytes);
            }
            if let Some(up) = &self.up_interp
                && b.readback_texture_to_ptr(self.interp_out_tex.handle, up.ptr, c_pitch, cw, ch)
            {
                // SAFETY: valid swap chain backbuffer and mapped upload staging buffer.
                unsafe {
                    let idx = swap_chain.GetCurrentBackBufferIndex();
                    if let Ok(back) = swap_chain.GetBuffer::<ID3D12Resource>(idx) {
                        copy.reset();
                        let l = &copy.list;
                        l.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_PRESENT,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )]);
                        let src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&up.resource),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                    Offset: 0,
                                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                        Format: swap_format,
                                        Width: cw,
                                        Height: ch,
                                        Depth: 1,
                                        RowPitch: c_pitch,
                                    },
                                },
                            },
                        };
                        let dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&back),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                        };
                        l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                        l.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_PRESENT,
                        )]);
                        copy.close();
                        host.execute(&[copy]);
                        let _ = swap_chain.Present(0, DXGI_PRESENT(0));
                        return PresentOutcome {
                            handled_present: false,
                            extra_frames: 1,
                        };
                    }
                }
            }
        }

        none
    }
}

impl Drop for MetalPresenter {
    fn drop(&mut self) {
        if let Some(b) = bridge::global() {
            if self.surface_handle != 0 {
                b.destroy_surface(self.surface_handle);
            }
            if self.interp_handle != 0 {
                b.destroy_frame_interpolator(self.interp_handle);
            }
            self.color_tex.destroy(b);
            self.ui_tex.destroy(b);
            self.interp_out_tex.destroy(b);
        }
    }
}
