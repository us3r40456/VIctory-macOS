//! Native Apple `MTLFXTemporalScaler` backend for `amd_fidelityfx_loader_xbox.dll`
//! (`VICTORY_UPSCALER=metalfx` or `VICTORY_METAL_FX=1`) plus shared motion/camera
//! hints exported to `MetalPresenter`'s `MTLFXFrameInterpolator`.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::bridge::{self, MetalBridge, TemporalScalerDesc, TemporalScalerDispatch};
use super::pixel_format as pf;
use crate::host::transition;

/// Camera & motion hints captured from the title's latest upscaler dispatch,
/// consumed by `MTLFXFrameInterpolator` when active.
#[derive(Debug, Clone, Copy)]
pub struct MotionHint {
    pub jitter: [f32; 2],
    pub motion_scale: [f32; 2],
    pub delta_time: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub fov_degrees: f32,
    pub depth_reversed: bool,
    pub reset: bool,
}

impl Default for MotionHint {
    fn default() -> Self {
        Self {
            jitter: [0.0, 0.0],
            motion_scale: [1.0, 1.0],
            delta_time: 1.0 / 30.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            fov_degrees: 60.0,
            depth_reversed: false,
            reset: false,
        }
    }
}

static LAST_HINT: Mutex<Option<MotionHint>> = Mutex::new(None);

/// Records the latest engine camera/motion metadata from an upscaler dispatch.
pub fn record_motion_hint(hint: MotionHint) {
    *LAST_HINT.lock().unwrap_or_else(|e| e.into_inner()) = Some(hint);
}

/// Takes the latest engine camera/motion metadata (if any).
pub fn latest_motion_hint() -> Option<MotionHint> {
    *LAST_HINT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Halton radical inverse in base `base` for subpixel jitter generation.
pub fn halton(mut index: i32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let b = base as f32;
    while index > 0 {
        f /= b;
        r += f * (index % base as i32) as f32;
        index /= base as i32;
    }
    r
}

/// Standard temporal upscaler jitter phase count for `(render_width, output_width)`.
pub fn jitter_phase_count(render_width: u32, output_width: u32) -> i32 {
    let ratio = (output_width.max(1) as f32) / (render_width.max(1) as f32);
    (8.0 * ratio * ratio).ceil().max(8.0) as i32
}

/// Standard Halton(2, 3) subpixel jitter offset in `[-0.5, 0.5]`.
pub fn jitter_offset(index: i32, phases: i32) -> (f32, f32) {
    let p = phases.max(1);
    let i = index.rem_euclid(p) + 1;
    (halton(i, 2) - 0.5, halton(i, 3) - 0.5)
}

/// Staging D3D12 buffer (`READBACK` or `UPLOAD`) kept persistently mapped.
struct StagingBuf {
    resource: ID3D12Resource,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: only accessed under `MetalFxUpscaler` mutex after GPU completion.
unsafe impl Send for StagingBuf {}

impl StagingBuf {
    fn new(device: &ID3D12Device, heap: D3D12_HEAP_TYPE, len: usize) -> Option<Self> {
        let state = if heap == D3D12_HEAP_TYPE_READBACK {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else {
            D3D12_RESOURCE_STATE_GENERIC_READ
        };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain committed linear buffer creation + persistent map.
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

/// Parameters for one `MetalFxUpscaler::dispatch` call.
pub struct UpscaleInputs<'a> {
    pub color: (&'a ID3D12Resource, u32),
    pub depth: (&'a ID3D12Resource, u32),
    pub motion: (&'a ID3D12Resource, u32),
    pub reactive: Option<(&'a ID3D12Resource, u32)>,
    pub output: (&'a ID3D12Resource, u32),
    pub render_size: [u32; 2],
    pub output_size: [u32; 2],
    pub jitter: [f32; 2],
    pub motion_scale: [f32; 2],
    pub frame_ms: f32,
    pub pre_exposure: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub fov_radians: f32,
    pub reset: bool,
}

struct ScalerInstance {
    handle: u64,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    color_fmt: u32,
    depth_fmt: u32,
    motion_fmt: u32,
    output_fmt: u32,
    reactive_fmt: u32,
    color_tex: u64,
    depth_tex: u64,
    motion_tex: u64,
    reactive_tex: u64,
    output_tex: u64,
}

impl ScalerInstance {
    fn destroy(&self, b: &MetalBridge) {
        b.destroy_texture(self.color_tex);
        b.destroy_texture(self.depth_tex);
        b.destroy_texture(self.motion_tex);
        b.destroy_texture(self.reactive_tex);
        b.destroy_texture(self.output_tex);
        b.destroy_temporal_scaler(self.handle);
    }
}

/// Live `MTLFXTemporalScaler` context backing `fidelityfx.rs`.
pub struct MetalFxUpscaler {
    pub max_render: [u32; 2],
    pub max_output: [u32; 2],
    pub auto_exposure: bool,
    pub depth_reversed: bool,
    instance: Mutex<Option<ScalerInstance>>,
    rb_color: Mutex<Option<StagingBuf>>,
    rb_depth: Mutex<Option<StagingBuf>>,
    rb_motion: Mutex<Option<StagingBuf>>,
    rb_reactive: Mutex<Option<StagingBuf>>,
    up_output: Mutex<Option<StagingBuf>>,
}

impl MetalFxUpscaler {
    pub fn new(
        max_render: [u32; 2],
        max_output: [u32; 2],
        auto_exposure: bool,
        depth_reversed: bool,
    ) -> Option<Self> {
        let b = bridge::global()?;
        if !b.info().metalfx {
            return None;
        }
        Some(Self {
            max_render,
            max_output,
            auto_exposure,
            depth_reversed,
            instance: Mutex::new(None),
            rb_color: Mutex::new(None),
            rb_depth: Mutex::new(None),
            rb_motion: Mutex::new(None),
            rb_reactive: Mutex::new(None),
            up_output: Mutex::new(None),
        })
    }

    pub fn estimated_gpu_bytes(&self) -> [u64; 2] {
        let px = u64::from(self.max_output[0]) * u64::from(self.max_output[1]);
        let total = px * 32; // conservative ~32 B/output pixel for temporal history
        [total, total / 2]
    }

    /// Executes native Apple `MTLFXTemporalScaler` on the supplied D3D12
    /// textures by staging through unified-memory readback/upload buffers on
    /// `host`, returning `(gpu_ns, wall_ns)`.
    pub fn dispatch(&self, inputs: &UpscaleInputs<'_>) -> Option<(u64, u64)> {
        let b = bridge::global()?;
        let host = crate::host::host()?;

        // SAFETY: live D3D12 textures.
        let (cd, dd, md, od) = unsafe {
            (
                inputs.color.0.GetDesc(),
                inputs.depth.0.GetDesc(),
                inputs.motion.0.GetDesc(),
                inputs.output.0.GetDesc(),
            )
        };
        let color_fmt = pf::from_dxgi(cd.Format.0 as u32)?;
        let depth_fmt = pf::from_dxgi(dd.Format.0 as u32)?;
        let motion_fmt = pf::from_dxgi(md.Format.0 as u32)?;
        let output_fmt = pf::from_dxgi(od.Format.0 as u32)?;
        let reactive_fmt = inputs
            .reactive
            .and_then(|(r, _)| {
                // SAFETY: live D3D12 texture.
                pf::from_dxgi(unsafe { r.GetDesc() }.Format.0 as u32)
            })
            .unwrap_or(0);

        let in_w = (cd.Width as u32).max(1);
        let in_h = cd.Height.max(1);
        let out_w = (od.Width as u32).max(1);
        let out_h = od.Height.max(1);

        let mut inst_guard = self.instance.lock().unwrap_or_else(|e| e.into_inner());
        let matches = inst_guard.as_ref().is_some_and(|i| {
            (
                i.in_w,
                i.in_h,
                i.out_w,
                i.out_h,
                i.color_fmt,
                i.depth_fmt,
                i.motion_fmt,
                i.output_fmt,
                i.reactive_fmt,
            ) == (
                in_w,
                in_h,
                out_w,
                out_h,
                color_fmt,
                depth_fmt,
                motion_fmt,
                output_fmt,
                reactive_fmt,
            )
        });
        if !matches {
            if let Some(old) = inst_guard.take() {
                old.destroy(b);
            }
            let handle = b
                .create_temporal_scaler(&TemporalScalerDesc {
                    input_width: in_w,
                    input_height: in_h,
                    output_width: out_w,
                    output_height: out_h,
                    color_format: color_fmt,
                    depth_format: depth_fmt,
                    motion_format: motion_fmt,
                    output_format: output_fmt,
                    reactive_format: reactive_fmt,
                    auto_exposure: self.auto_exposure,
                    sync_init: true,
                })
                .ok()?;
            let color_tex = b.create_texture(in_w, in_h, color_fmt, false)?;
            let depth_tex = b.create_texture(in_w, in_h, depth_fmt, false)?;
            let motion_tex = b.create_texture(in_w, in_h, motion_fmt, false)?;
            let output_tex = b.create_texture(out_w, out_h, output_fmt, false)?;
            let reactive_tex = if reactive_fmt != 0 {
                b.create_texture(in_w, in_h, reactive_fmt, false).unwrap_or(0)
            } else {
                0
            };
            *inst_guard = Some(ScalerInstance {
                handle,
                in_w,
                in_h,
                out_w,
                out_h,
                color_fmt,
                depth_fmt,
                motion_fmt,
                output_fmt,
                reactive_fmt,
                color_tex,
                depth_tex,
                motion_tex,
                reactive_tex,
                output_tex,
            });
        }
        let inst = inst_guard.as_ref()?;

        let ensure = |slot: &Mutex<Option<StagingBuf>>, heap, need: usize| -> Option<()> {
            let mut g = slot.lock().unwrap_or_else(|e| e.into_inner());
            if g.as_ref().is_none_or(|s| s.len < need) {
                *g = StagingBuf::new(&host.device, heap, need);
            }
            g.as_ref().map(|_| ())
        };

        let pitch = |w: u32, fmt: u32| -> u32 {
            (w * pf::bytes_per_pixel(fmt) as u32).next_multiple_of(256)
        };
        let c_pitch = pitch(in_w, color_fmt);
        let d_pitch = pitch(in_w, depth_fmt);
        let m_pitch = pitch(in_w, motion_fmt);
        let o_pitch = pitch(out_w, output_fmt);
        ensure(&self.rb_color, D3D12_HEAP_TYPE_READBACK, (c_pitch * in_h) as usize)?;
        ensure(&self.rb_depth, D3D12_HEAP_TYPE_READBACK, (d_pitch * in_h) as usize)?;
        ensure(&self.rb_motion, D3D12_HEAP_TYPE_READBACK, (m_pitch * in_h) as usize)?;
        ensure(&self.up_output, D3D12_HEAP_TYPE_UPLOAD, (o_pitch * out_h) as usize)?;
        if reactive_fmt != 0 {
            let r_pitch = pitch(in_w, reactive_fmt);
            ensure(&self.rb_reactive, D3D12_HEAP_TYPE_READBACK, (r_pitch * in_h) as usize)?;
        }

        // Copy inputs to UMA readback staging in one D3D12 submission.
        let copy_in = host.new_list().ok()?;
        let record_rb = |list: &ID3D12GraphicsCommandList,
                         tex: &ID3D12Resource,
                         cur_state: u32,
                         buf: &ID3D12Resource,
                         fmt: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
                         w: u32,
                         h: u32,
                         row_pitch: u32| {
            let st = D3D12_RESOURCE_STATES(cur_state as i32);
            // SAFETY: valid live textures and mapped readback buffers.
            unsafe {
                if st != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[transition(tex, st, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                }
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(buf),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: fmt,
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: row_pitch,
                            },
                        },
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                if st != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[transition(tex, D3D12_RESOURCE_STATE_COPY_SOURCE, st)]);
                }
            }
        };

        {
            let c_buf = self.rb_color.lock().unwrap_or_else(|e| e.into_inner());
            let d_buf = self.rb_depth.lock().unwrap_or_else(|e| e.into_inner());
            let m_buf = self.rb_motion.lock().unwrap_or_else(|e| e.into_inner());
            record_rb(
                &copy_in.list,
                inputs.color.0,
                inputs.color.1,
                &c_buf.as_ref()?.resource,
                cd.Format,
                in_w,
                in_h,
                c_pitch,
            );
            record_rb(
                &copy_in.list,
                inputs.depth.0,
                inputs.depth.1,
                &d_buf.as_ref()?.resource,
                dd.Format,
                in_w,
                in_h,
                d_pitch,
            );
            record_rb(
                &copy_in.list,
                inputs.motion.0,
                inputs.motion.1,
                &m_buf.as_ref()?.resource,
                md.Format,
                in_w,
                in_h,
                m_pitch,
            );
            if let Some((r_tex, r_state)) = inputs.reactive
                && reactive_fmt != 0
            {
                let r_buf = self.rb_reactive.lock().unwrap_or_else(|e| e.into_inner());
                let r_pitch = pitch(in_w, reactive_fmt);
                // SAFETY: live texture.
                let rd = unsafe { r_tex.GetDesc() };
                record_rb(
                    &copy_in.list,
                    r_tex,
                    r_state,
                    &r_buf.as_ref()?.resource,
                    rd.Format,
                    in_w,
                    in_h,
                    r_pitch,
                );
            }
        }
        copy_in.close();
        host.execute(&[&copy_in]);

        // Upload to Metal private textures and run native MTLFXTemporalScaler.
        {
            let c_buf = self.rb_color.lock().unwrap_or_else(|e| e.into_inner());
            let d_buf = self.rb_depth.lock().unwrap_or_else(|e| e.into_inner());
            let m_buf = self.rb_motion.lock().unwrap_or_else(|e| e.into_inner());
            if !b.upload_texture_from_ptr(inst.color_tex, c_buf.as_ref()?.ptr, c_pitch, in_w, in_h)
                || !b.upload_texture_from_ptr(inst.depth_tex, d_buf.as_ref()?.ptr, d_pitch, in_w, in_h)
                || !b.upload_texture_from_ptr(inst.motion_tex, m_buf.as_ref()?.ptr, m_pitch, in_w, in_h)
            {
                return None;
            }
            if inst.reactive_tex != 0 {
                let r_buf = self.rb_reactive.lock().unwrap_or_else(|e| e.into_inner());
                let r_pitch = pitch(in_w, reactive_fmt);
                let _ = b.upload_texture_from_ptr(
                    inst.reactive_tex,
                    r_buf.as_ref()?.ptr,
                    r_pitch,
                    in_w,
                    in_h,
                );
            }
        }

        let content_w = inputs.render_size[0].clamp(1, in_w);
        let content_h = inputs.render_size[1].clamp(1, in_h);
        let (gpu_ns, wall_ns) = b
            .encode_temporal_scaler(
                inst.handle,
                &TemporalScalerDispatch {
                    color_tex: inst.color_tex,
                    depth_tex: inst.depth_tex,
                    motion_tex: inst.motion_tex,
                    reactive_tex: inst.reactive_tex,
                    output_tex: inst.output_tex,
                    input_content_width: content_w,
                    input_content_height: content_h,
                    jitter_x: inputs.jitter[0],
                    jitter_y: inputs.jitter[1],
                    mv_scale_x: inputs.motion_scale[0],
                    mv_scale_y: inputs.motion_scale[1],
                    pre_exposure: inputs.pre_exposure,
                    reset: inputs.reset,
                    depth_reversed: self.depth_reversed,
                },
            )
            .ok()?;

        // Export engine camera/motion metadata to FrameInterpolator.
        let fov_deg = if inputs.fov_radians > 0.05 && inputs.fov_radians < std::f32::consts::PI {
            inputs.fov_radians.to_degrees()
        } else {
            60.0
        };
        record_motion_hint(MotionHint {
            jitter: inputs.jitter,
            motion_scale: inputs.motion_scale,
            delta_time: (inputs.frame_ms * 1e-3).clamp(1.0 / 240.0, 0.25),
            near_plane: inputs.near_plane.min(inputs.far_plane).max(0.001),
            far_plane: inputs.near_plane.max(inputs.far_plane).max(1.0),
            fov_degrees: fov_deg,
            depth_reversed: self.depth_reversed,
            reset: inputs.reset,
        });

        // Read upscaled output back into D3D12 upload buffer and copy into title's output texture.
        {
            let up = self.up_output.lock().unwrap_or_else(|e| e.into_inner());
            if !b.readback_texture_to_ptr(inst.output_tex, up.as_ref()?.ptr, o_pitch, out_w, out_h) {
                return None;
            }
            let copy_out = host.new_list().ok()?;
            let st = D3D12_RESOURCE_STATES(inputs.output.1 as i32);
            // SAFETY: valid upload staging buffer and destination texture.
            unsafe {
                if st != D3D12_RESOURCE_STATE_COPY_DEST {
                    copy_out
                        .list
                        .ResourceBarrier(&[transition(inputs.output.0, st, D3D12_RESOURCE_STATE_COPY_DEST)]);
                }
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&up.as_ref()?.resource),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: od.Format,
                                Width: out_w,
                                Height: out_h,
                                Depth: 1,
                                RowPitch: o_pitch,
                            },
                        },
                    },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(inputs.output.0),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                copy_out.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                if st != D3D12_RESOURCE_STATE_COPY_DEST {
                    copy_out
                        .list
                        .ResourceBarrier(&[transition(inputs.output.0, D3D12_RESOURCE_STATE_COPY_DEST, st)]);
                }
            }
            copy_out.close();
            host.execute(&[&copy_out]);
        }

        Some((gpu_ns, wall_ns))
    }
}

impl Drop for MetalFxUpscaler {
    fn drop(&mut self) {
        if let Some(b) = bridge::global()
            && let Ok(mut g) = self.instance.lock()
            && let Some(inst) = g.take()
        {
            inst.destroy(b);
        }
    }
}
