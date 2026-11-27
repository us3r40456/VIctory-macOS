//! Live Windows PE -> CrossOver Wine unixlib bridge (`victory_metal.dll` ->
//! `victory_metal.so`).
//!
//! Loads `victory_metal.dll` dynamically from `WINEDLLPATH` so stock CrossOver
//! Wine requires zero rebuilds. Every call goes directly through Wine's
//! `__wine_unix_call_dispatcher`.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

use super::registry::{
    BackingKey, CanonicalBuffer, CanonicalRegistry, RegisterAction, ResolvedBuffer,
};
use super::{
    BenchArgs, ConvertDxilArgs, CreateTextureArgs, DestroyHandleArgs, FlushResidencyArgs,
    InterpCreateArgs, InterpEncodeArgs, OP_BENCH_ZERO_COPY, OP_CONVERT_DXIL, OP_CREATE_TEXTURE,
    OP_DESTROY_TEXTURE, OP_FLUSH_RESIDENCY, OP_INTERP_CREATE, OP_INTERP_DESTROY, OP_INTERP_ENCODE,
    OP_PING, OP_READBACK_TEXTURE, OP_REGISTER_BUFFER, OP_SCALER_CREATE, OP_SCALER_DESTROY,
    OP_SCALER_ENCODE, OP_STATS, OP_SURFACE_ATTACH, OP_SURFACE_DESTROY, OP_SURFACE_PRESENT,
    OP_UNREGISTER_BUFFER, OP_UPLOAD_TEXTURE, PingArgs, PingInfo, RegisterBufferArgs,
    ScalerCreateArgs, ScalerEncodeArgs, Stats, SurfaceAttachArgs, SurfacePresentArgs,
    TextureIoArgs, UnregisterBufferArgs,
};

type FnAvailable = unsafe extern "system" fn() -> i32;
type FnCall = unsafe extern "system" fn(u32, *mut c_void) -> i32;

/// Descriptor for creating a native Apple `MTLFXTemporalScaler`.
#[derive(Debug, Clone, Copy)]
pub struct TemporalScalerDesc {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_format: u32,
    pub depth_format: u32,
    pub motion_format: u32,
    pub output_format: u32,
    pub reactive_format: u32,
    pub auto_exposure: bool,
    pub sync_init: bool,
}

/// Parameters for one `MTLFXTemporalScaler` dispatch.
#[derive(Debug, Clone, Copy)]
pub struct TemporalScalerDispatch {
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
    pub reset: bool,
    pub depth_reversed: bool,
}

/// Descriptor for creating a native macOS 26/27 `MTLFXFrameInterpolator`.
#[derive(Debug, Clone, Copy)]
pub struct FrameInterpolatorDesc {
    /// Input width for depth/motion (0 => `output_width / 2` for optical flow).
    pub input_width: u32,
    /// Input height for depth/motion (0 => `output_height / 2` for optical flow).
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_format: u32,
    pub output_format: u32,
    pub depth_format: u32,
    pub motion_format: u32,
    /// Separate Plane 1 UI pixel format (`0` if no UI plane).
    pub ui_format: u32,
}

/// Parameters for one `MTLFXFrameInterpolator` dispatch.
#[derive(Debug, Clone, Copy)]
pub struct FrameInterpolatorDispatch {
    pub color_tex: u64,
    /// Optional explicit previous color texture (`0` => internal ping-pong).
    pub prev_color_tex: u64,
    /// Optional engine depth texture (`0` => built-in flat depth).
    pub depth_tex: u64,
    /// Optional engine motion vectors (`0` => built-in Metal optical flow).
    pub motion_tex: u64,
    /// Optional Plane 1 UI texture (any size; auto-resampled to output size).
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
    pub reset: bool,
    pub depth_reversed: bool,
    pub ui_composited: bool,
    pub skip_prev_update: bool,
}

/// Timing and outcome of one `MTLFXFrameInterpolator` pass.
#[derive(Debug, Clone, Copy)]
pub struct FrameInterpolatorResult {
    pub interpolated: bool,
    pub gpu_ns: u64,
    pub wall_ns: u64,
}

/// Loaded `victory_metal.dll` entry points + global canonical buffer registry.
pub struct MetalBridge {
    call: FnCall,
    ping: PingInfo,
    registry: Mutex<CanonicalRegistry>,
}

// SAFETY: `FnCall` is a process-lifetime function pointer into the loaded DLL
// and `victory_metal.so` serializes all Metal state internally.
unsafe impl Send for MetalBridge {}
// SAFETY: as above.
unsafe impl Sync for MetalBridge {}

impl MetalBridge {
    /// Attempts to load `victory_metal.dll` and ping the Apple Silicon Metal
    /// unixlib. Returns `None` outside macOS/Wine or if `victory_metal.dll`
    /// is not on `WINEDLLPATH`.
    fn load() -> Option<Self> {
        let dll_path = std::env::var("VICTORY_METAL_DLL")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| std::ffi::CString::new(s).ok());
        // SAFETY: NUL-terminated path passed to Win32 loader APIs.
        let (available, call) = unsafe {
            let mut module = std::ptr::null_mut();
            if let Some(ref path) = dll_path {
                module = LoadLibraryA(path.as_ptr().cast());
            }
            if module.is_null() {
                module = LoadLibraryA(c"victory_metal.dll".as_ptr().cast());
            }
            if module.is_null() {
                tracing::debug!(
                    "victory_metal.dll not found (checked VICTORY_METAL_DLL and WINEDLLPATH); Metal unixlib bridge inactive"
                );
                return None;
            }
            let p_avail = GetProcAddress(module, c"victory_metal_available".as_ptr().cast());
            let p_call = GetProcAddress(module, c"victory_metal_call".as_ptr().cast());
            let (Some(a), Some(c)) = (p_avail, p_call) else {
                tracing::warn!("victory_metal.dll missing expected exports");
                return None;
            };
            (
                std::mem::transmute::<unsafe extern "system" fn() -> isize, FnAvailable>(a),
                std::mem::transmute::<unsafe extern "system" fn() -> isize, FnCall>(c),
            )
        };
        // SAFETY: validated export from `victory_metal.dll`.
        if unsafe { available() } == 0 {
            tracing::warn!("victory_metal.dll loaded, but Wine unixlib failed to initialize");
            return None;
        }
        let mut raw = PingArgs::default();
        // SAFETY: `raw` is a properly aligned C-layout argument struct.
        let status = unsafe { call(OP_PING, (&raw mut raw).cast()) };
        if status != 0 {
            tracing::warn!(status = format_args!("{status:#x}"), "victory_metal OP_PING failed");
            return None;
        }
        let ping = PingInfo::from_raw(&raw);
        tracing::info!(
            device = %ping.device_name,
            abi = ping.abi_version,
            unified = ping.unified_memory,
            residency_set = ping.residency_set,
            metalfx = ping.metalfx,
            frame_interpolator = ping.frame_interpolator,
            metal4_fx = ping.metal4_fx,
            irconverter = ping.irconverter,
            recommended_mb = ping.recommended_vram >> 20,
            "victory_metal unixlib bridge active (Route A.5)"
        );
        Some(Self {
            call,
            ping,
            registry: Mutex::new(CanonicalRegistry::new()),
        })
    }

    pub fn info(&self) -> &PingInfo {
        &self.ping
    }

    /// Raw ping call (useful for measuring PE -> unixlib dispatch latency).
    pub fn ping_raw(&self) -> Option<PingInfo> {
        let mut raw = PingArgs::default();
        // SAFETY: valid C-layout struct for `OP_PING`.
        let status = unsafe { (self.call)(OP_PING, (&raw mut raw).cast()) };
        (status == 0).then(|| PingInfo::from_raw(&raw))
    }

    /// Registers a CPU virtual address range `[cpu_addr, cpu_addr + length)`
    /// backed by `backing` into the global `MTLResidencySet` zero-copy.
    pub fn register_buffer(
        &self,
        cpu_addr: u64,
        length: u64,
        backing: BackingKey,
        defer_commit: bool,
    ) -> Option<(CanonicalBuffer, bool)> {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        match reg.plan_register(backing) {
            RegisterAction::ReuseCanonical(_) => {
                let buf = reg.attach_view(cpu_addr, length, backing)?;
                Some((buf, true))
            }
            RegisterAction::CreateNew { abi_backing_key } => {
                let mut args = RegisterBufferArgs {
                    cpu_addr,
                    length,
                    backing_key: abi_backing_key,
                    flags: u32::from(defer_commit),
                    ..Default::default()
                };
                // SAFETY: valid C-layout struct for `OP_REGISTER_BUFFER`.
                let status = unsafe { (self.call)(OP_REGISTER_BUFFER, (&raw mut args).cast()) };
                if status != 0 || args.status != 0 || args.handle == 0 {
                    tracing::warn!(
                        cpu_addr = format_args!("{cpu_addr:#x}"),
                        length,
                        ntstatus = format_args!("{status:#x}"),
                        err = args.status,
                        "victory_metal newBufferWithBytesNoCopy failed"
                    );
                    return None;
                }
                let buf = reg.insert_canonical(
                    cpu_addr,
                    length,
                    backing,
                    args.handle,
                    args.gpu_address,
                );
                Some((buf, args.reused != 0))
            }
        }
    }

    /// Resolves any title CPU address range `[addr, addr + len)` to its
    /// canonical zero-copy `MTLBuffer` handle and GPU address.
    pub fn resolve(&self, addr: u64, len: u64) -> Option<ResolvedBuffer> {
        let reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.resolve(addr, len)
    }

    /// Unregisters a previously registered CPU view; releases the canonical
    /// `MTLBuffer` from `MTLResidencySet` once its last view is removed.
    pub fn unregister_view(&self, cpu_addr: u64, defer_commit: bool) -> bool {
        let drop_handle = {
            let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
            reg.remove_view(cpu_addr)
        };
        if let Some(handle) = drop_handle {
            let mut args = UnregisterBufferArgs {
                handle,
                flags: u32::from(defer_commit),
                remaining_refs: 0,
            };
            // SAFETY: valid C-layout struct for `OP_UNREGISTER_BUFFER`.
            let status = unsafe { (self.call)(OP_UNREGISTER_BUFFER, (&raw mut args).cast()) };
            return status == 0;
        }
        true
    }

    /// Flushes any deferred `MTLResidencySet` additions/removals in one commit.
    pub fn flush_residency(&self) -> Option<(u32, u32)> {
        let mut args = FlushResidencyArgs::default();
        // SAFETY: valid C-layout struct for `OP_FLUSH_RESIDENCY`.
        let status = unsafe { (self.call)(OP_FLUSH_RESIDENCY, (&raw mut args).cast()) };
        (status == 0).then_some((args.pending_before, args.total_allocations))
    }

    /// Runs a real Metal compute checksum read over `[offset, offset + length)`
    /// of zero-copy buffer `handle`. Returns `(checksum, gpu_ns, wall_ns)`.
    pub fn bench_read(&self, handle: u64, offset: u64, length: u64) -> Option<(u64, u64, u64)> {
        let mut args = BenchArgs {
            handle,
            offset,
            length,
            mode: 0,
            ..Default::default()
        };
        // SAFETY: valid C-layout struct for `OP_BENCH_ZERO_COPY`.
        let status = unsafe { (self.call)(OP_BENCH_ZERO_COPY, (&raw mut args).cast()) };
        (status == 0).then_some((args.checksum_out, args.gpu_ns, args.wall_ns))
    }

    /// Runs a real Metal compute write (`word[i] = seed ^ i`) directly into
    /// zero-copy buffer `handle`. Returns `(gpu_ns, wall_ns)`.
    pub fn bench_write(&self, handle: u64, offset: u64, length: u64, seed: u32) -> Option<(u64, u64)> {
        let mut args = BenchArgs {
            handle,
            offset,
            length,
            mode: 1,
            seed,
            ..Default::default()
        };
        // SAFETY: valid C-layout struct for `OP_BENCH_ZERO_COPY`.
        let status = unsafe { (self.call)(OP_BENCH_ZERO_COPY, (&raw mut args).cast()) };
        (status == 0).then_some((args.gpu_ns, args.wall_ns))
    }

    /// Converts an SM 6.x DXIL container to Apple `.metallib` bytecode via
    /// `libmetalirconverter.dylib` inside the unixlib.
    pub fn convert_dxil(&self, dxil: &[u8], entry: &str) -> Result<(u32, Vec<u8>), i32> {
        let entry_c = std::ffi::CString::new(entry).map_err(|_| -2)?;
        let mut query = ConvertDxilArgs {
            dxil_ptr: dxil.as_ptr() as u64,
            dxil_len: dxil.len() as u64,
            entry_ptr: entry_c.as_ptr() as u64,
            out_ptr: 0,
            out_cap: 0,
            out_len: 0,
            stage: 0,
            status: 0,
        };
        // SAFETY: pointers remain valid through the synchronous unixlib call.
        let nt = unsafe { (self.call)(OP_CONVERT_DXIL, (&raw mut query).cast()) };
        if nt != 0 || query.status != 0 || query.out_len == 0 {
            return Err(if query.status != 0 { query.status } else { -2 });
        }
        let mut out = vec![0u8; query.out_len as usize];
        let mut fill = ConvertDxilArgs {
            dxil_ptr: dxil.as_ptr() as u64,
            dxil_len: dxil.len() as u64,
            entry_ptr: entry_c.as_ptr() as u64,
            out_ptr: out.as_mut_ptr() as u64,
            out_cap: out.len() as u64,
            out_len: 0,
            stage: 0,
            status: 0,
        };
        // SAFETY: `out` buffer has exact capacity `query.out_len`.
        let nt = unsafe { (self.call)(OP_CONVERT_DXIL, (&raw mut fill).cast()) };
        if nt != 0 || fill.status != 0 {
            return Err(if fill.status != 0 { fill.status } else { -2 });
        }
        out.truncate(fill.out_len as usize);
        Ok((fill.stage, out))
    }

    pub fn stats(&self) -> Option<Stats> {
        let mut raw = Stats::default();
        // SAFETY: valid C-layout struct for `OP_STATS`.
        let status = unsafe { (self.call)(OP_STATS, (&raw mut raw).cast()) };
        (status == 0).then_some(raw)
    }

    // -----------------------------------------------------------------------
    // Step 2B-2: Textures, MetalFX Scaler/Interpolator, and CAMetalLayer
    // -----------------------------------------------------------------------

    /// Allocates a private (or shared) 2D `MTLTexture`.
    pub fn create_texture(
        &self,
        width: u32,
        height: u32,
        pixel_format: u32,
        shared: bool,
    ) -> Option<u64> {
        let mut args = CreateTextureArgs {
            width,
            height,
            pixel_format,
            usage: 0,
            storage_mode: u32::from(shared),
            status: 0,
            handle: 0,
        };
        // SAFETY: valid C-layout struct for `OP_CREATE_TEXTURE`.
        let nt = unsafe { (self.call)(OP_CREATE_TEXTURE, (&raw mut args).cast()) };
        (nt == 0 && args.status == 0 && args.handle != 0).then_some(args.handle)
    }

    pub fn destroy_texture(&self, handle: u64) {
        if handle == 0 {
            return;
        }
        let mut args = DestroyHandleArgs { handle };
        // SAFETY: valid C-layout struct for `OP_DESTROY_TEXTURE`.
        unsafe { (self.call)(OP_DESTROY_TEXTURE, (&raw mut args).cast()) };
    }

    /// Uploads linear pixels from CPU memory (or a registered zero-copy buffer)
    /// into a private `MTLTexture`.
    pub fn upload_texture_from_ptr(
        &self,
        texture_handle: u64,
        src: *const u8,
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> bool {
        let mut args = TextureIoArgs {
            texture_handle,
            buffer_handle: 0,
            cpu_ptr: src as u64,
            offset: 0,
            bytes_per_row,
            width,
            height,
            status: 0,
        };
        // SAFETY: caller ensures `src` stays valid for `bytes_per_row * height`.
        let nt = unsafe { (self.call)(OP_UPLOAD_TEXTURE, (&raw mut args).cast()) };
        nt == 0 && args.status == 0
    }

    /// Uploads linear pixels directly from a registered zero-copy `MTLBuffer`.
    pub fn upload_texture_from_buffer(
        &self,
        texture_handle: u64,
        buffer_handle: u64,
        offset: u64,
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> bool {
        let mut args = TextureIoArgs {
            texture_handle,
            buffer_handle,
            cpu_ptr: 0,
            offset,
            bytes_per_row,
            width,
            height,
            status: 0,
        };
        // SAFETY: valid C-layout struct for `OP_UPLOAD_TEXTURE`.
        let nt = unsafe { (self.call)(OP_UPLOAD_TEXTURE, (&raw mut args).cast()) };
        nt == 0 && args.status == 0
    }

    /// Reads a private `MTLTexture` back to CPU memory.
    pub fn readback_texture_to_ptr(
        &self,
        texture_handle: u64,
        dst: *mut u8,
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> bool {
        let mut args = TextureIoArgs {
            texture_handle,
            buffer_handle: 0,
            cpu_ptr: dst as u64,
            offset: 0,
            bytes_per_row,
            width,
            height,
            status: 0,
        };
        // SAFETY: caller ensures `dst` has room for `bytes_per_row * height`.
        let nt = unsafe { (self.call)(OP_READBACK_TEXTURE, (&raw mut args).cast()) };
        nt == 0 && args.status == 0
    }

    /// Creates a native Apple `MTLFXTemporalScaler`.
    pub fn create_temporal_scaler(&self, desc: &TemporalScalerDesc) -> Result<u64, i32> {
        let mut flags = 0u32;
        if desc.auto_exposure {
            flags |= 1 << 0;
        }
        if desc.sync_init {
            flags |= 1 << 1;
        }
        let mut args = ScalerCreateArgs {
            input_width: desc.input_width,
            input_height: desc.input_height,
            output_width: desc.output_width,
            output_height: desc.output_height,
            color_format: desc.color_format,
            depth_format: desc.depth_format,
            motion_format: desc.motion_format,
            output_format: desc.output_format,
            reactive_format: desc.reactive_format,
            flags,
            handle: 0,
            status: 0,
            reserved: 0,
        };
        // SAFETY: valid C-layout struct for `OP_SCALER_CREATE`.
        let nt = unsafe { (self.call)(OP_SCALER_CREATE, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 && args.handle != 0 {
            Ok(args.handle)
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    /// Encodes and executes one `MTLFXTemporalScaler` pass. Returns `(gpu_ns, wall_ns)`.
    pub fn encode_temporal_scaler(
        &self,
        handle: u64,
        dispatch: &TemporalScalerDispatch,
    ) -> Result<(u64, u64), i32> {
        let mut flags = 0u32;
        if dispatch.reset {
            flags |= 1 << 0;
        }
        if dispatch.depth_reversed {
            flags |= 1 << 1;
        }
        let mut args = ScalerEncodeArgs {
            scaler_handle: handle,
            color_tex: dispatch.color_tex,
            depth_tex: dispatch.depth_tex,
            motion_tex: dispatch.motion_tex,
            reactive_tex: dispatch.reactive_tex,
            output_tex: dispatch.output_tex,
            input_content_width: dispatch.input_content_width,
            input_content_height: dispatch.input_content_height,
            jitter_x: dispatch.jitter_x,
            jitter_y: dispatch.jitter_y,
            mv_scale_x: dispatch.mv_scale_x,
            mv_scale_y: dispatch.mv_scale_y,
            pre_exposure: dispatch.pre_exposure,
            flags,
            gpu_ns: 0,
            wall_ns: 0,
            status: 0,
            reserved: 0,
        };
        // SAFETY: valid C-layout struct for `OP_SCALER_ENCODE`.
        let nt = unsafe { (self.call)(OP_SCALER_ENCODE, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 {
            Ok((args.gpu_ns, args.wall_ns))
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    pub fn destroy_temporal_scaler(&self, handle: u64) {
        if handle == 0 {
            return;
        }
        let mut args = DestroyHandleArgs { handle };
        // SAFETY: valid C-layout struct for `OP_SCALER_DESTROY`.
        unsafe { (self.call)(OP_SCALER_DESTROY, (&raw mut args).cast()) };
    }

    /// Creates a native macOS 26/27 `MTLFXFrameInterpolator`.
    pub fn create_frame_interpolator(&self, desc: &FrameInterpolatorDesc) -> Result<u64, i32> {
        let mut args = InterpCreateArgs {
            input_width: desc.input_width,
            input_height: desc.input_height,
            output_width: desc.output_width,
            output_height: desc.output_height,
            color_format: desc.color_format,
            output_format: desc.output_format,
            depth_format: desc.depth_format,
            motion_format: desc.motion_format,
            ui_format: desc.ui_format,
            flags: 0,
            handle: 0,
            status: 0,
            reserved: 0,
        };
        // SAFETY: valid C-layout struct for `OP_INTERP_CREATE`.
        let nt = unsafe { (self.call)(OP_INTERP_CREATE, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 && args.handle != 0 {
            Ok(args.handle)
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    /// Encodes and executes one `MTLFXFrameInterpolator` pass (including built-in
    /// Metal compute optical flow whenever `motion_tex == 0` and automatic Plane 1
    /// UI resampling).
    pub fn encode_frame_interpolator(
        &self,
        handle: u64,
        dispatch: &FrameInterpolatorDispatch,
    ) -> Result<FrameInterpolatorResult, i32> {
        let mut flags = 0u32;
        if dispatch.reset {
            flags |= 1 << 0;
        }
        if dispatch.depth_reversed {
            flags |= 1 << 1;
        }
        if dispatch.ui_composited {
            flags |= 1 << 2;
        }
        if dispatch.skip_prev_update {
            flags |= 1 << 3;
        }
        let mut args = InterpEncodeArgs {
            interp_handle: handle,
            color_tex: dispatch.color_tex,
            prev_color_tex: dispatch.prev_color_tex,
            depth_tex: dispatch.depth_tex,
            motion_tex: dispatch.motion_tex,
            ui_tex: dispatch.ui_tex,
            output_tex: dispatch.output_tex,
            jitter_x: dispatch.jitter_x,
            jitter_y: dispatch.jitter_y,
            mv_scale_x: dispatch.mv_scale_x,
            mv_scale_y: dispatch.mv_scale_y,
            delta_time: dispatch.delta_time,
            near_plane: dispatch.near_plane,
            far_plane: dispatch.far_plane,
            fov_degrees: dispatch.fov_degrees,
            flags,
            interpolated: 0,
            gpu_ns: 0,
            wall_ns: 0,
            status: 0,
            reserved: 0,
        };
        // SAFETY: valid C-layout struct for `OP_INTERP_ENCODE`.
        let nt = unsafe { (self.call)(OP_INTERP_ENCODE, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 {
            Ok(FrameInterpolatorResult {
                interpolated: args.interpolated != 0,
                gpu_ns: args.gpu_ns,
                wall_ns: args.wall_ns,
            })
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    pub fn destroy_frame_interpolator(&self, handle: u64) {
        if handle == 0 {
            return;
        }
        let mut args = DestroyHandleArgs { handle };
        // SAFETY: valid C-layout struct for `OP_INTERP_DESTROY`.
        unsafe { (self.call)(OP_INTERP_DESTROY, (&raw mut args).cast()) };
    }

    /// Attaches a native `CAMetalLayer` overlay view to the game's macOS `NSWindow`.
    pub fn attach_surface(
        &self,
        width: u32,
        height: u32,
        pixel_format: u32,
        vsync: bool,
    ) -> Result<u64, i32> {
        let mut args = SurfaceAttachArgs {
            width,
            height,
            pixel_format,
            flags: u32::from(vsync),
            handle: 0,
            status: 0,
            reserved: 0,
        };
        // SAFETY: valid C-layout struct for `OP_SURFACE_ATTACH`.
        let nt = unsafe { (self.call)(OP_SURFACE_ATTACH, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 && args.handle != 0 {
            Ok(args.handle)
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    /// Presents `color_tex` (with optional `ui_tex` composited in one compute
    /// pass) directly onto an attached `CAMetalLayer` drawable.
    pub fn present_surface(
        &self,
        surface_handle: u64,
        color_tex: u64,
        ui_tex: u64,
        after_min_duration_ns: u64,
        wait_completed: bool,
    ) -> Result<(u64, u64), i32> {
        let mut args = SurfacePresentArgs {
            surface_handle,
            color_tex,
            ui_tex,
            after_min_duration_ns,
            flags: u32::from(wait_completed),
            status: 0,
            gpu_ns: 0,
            wall_ns: 0,
        };
        // SAFETY: valid C-layout struct for `OP_SURFACE_PRESENT`.
        let nt = unsafe { (self.call)(OP_SURFACE_PRESENT, (&raw mut args).cast()) };
        if nt == 0 && args.status == 0 {
            Ok((args.gpu_ns, args.wall_ns))
        } else {
            Err(if args.status != 0 { args.status } else { -1 })
        }
    }

    pub fn destroy_surface(&self, handle: u64) {
        if handle == 0 {
            return;
        }
        let mut args = DestroyHandleArgs { handle };
        // SAFETY: valid C-layout struct for `OP_SURFACE_DESTROY`.
        unsafe { (self.call)(OP_SURFACE_DESTROY, (&raw mut args).cast()) };
    }
}

/// Process-wide singleton Metal unixlib bridge (`None` if unavailable).
pub fn global() -> Option<&'static MetalBridge> {
    static BRIDGE: OnceLock<Option<MetalBridge>> = OnceLock::new();
    BRIDGE.get_or_init(MetalBridge::load).as_ref()
}

#[cfg(test)]
mod tests {
    use super::super::pixel_format as pf;
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
    };

    fn f32_to_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let frac = bits & 0x7f_ffff;
        if exp <= 0 {
            return sign;
        }
        if exp >= 31 {
            return sign | 0x7c00;
        }
        sign | ((exp as u16) << 10) | ((frac >> 13) as u16)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = (h >> 15) as u32;
        let exp = ((h >> 10) & 0x1f) as i32;
        let frac = (h & 0x3ff) as f32;
        let v = match exp {
            0 => frac * 2f32.powi(-24),
            31 => f32::NAN,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
        };
        if sign != 0 { -v } else { v }
    }

    #[test]
    fn metal_unixlib_ping_and_call_latency() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        let info = bridge.info();
        assert_eq!(info.abi_version, 2);
        assert!(!info.device_name.is_empty());
        assert!(info.unified_memory, "Apple Silicon must report unified memory");
        assert!(info.residency_set, "macOS 15+ must support MTLResidencySet");
        assert!(info.metalfx, "Apple Silicon must support MetalFX");

        const N: u32 = 10_000;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            assert!(bridge.ping_raw().is_some());
        }
        let ns_per_call = t0.elapsed().as_nanos() as f64 / f64::from(N);
        eprintln!(
            "victory_metal ping: device='{}' abi={} unified={} residency={} metalfx={} frame_interp={} metal4_fx={} irconverter={} latency={:.1} ns/call",
            info.device_name,
            info.abi_version,
            info.unified_memory,
            info.residency_set,
            info.metalfx,
            info.frame_interpolator,
            info.metal4_fx,
            info.irconverter,
            ns_per_call,
        );
    }

    #[test]
    fn metal_zero_copy_64mb_virtual_alloc_cpu_gpu_cpu() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        const LEN: usize = 64 << 20;
        // SAFETY: standalone 64 MiB VirtualAlloc freed before test exit.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u32>();
        assert!(!ptr.is_null(), "VirtualAlloc 64 MiB failed");
        let words = LEN / 4;
        let mut expected: u64 = 0;
        // SAFETY: `ptr` owns `words` u32 elements.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, words);
            for (i, slot) in slice.iter_mut().enumerate() {
                let v = (i as u32).wrapping_mul(2654435761) ^ 0x5a5a_1234;
                *slot = v;
                expected = expected.wrapping_add(u64::from(v));
            }
        }

        let base = ptr as u64;
        let (buf, reused) = bridge
            .register_buffer(base, LEN as u64, BackingKey::VirtualAlloc(base), false)
            .expect("zero-copy register 64 MiB VirtualAlloc");
        assert!(!reused);
        assert_ne!(buf.gpu_address, 0);

        let (sum0, _, _) = bridge.bench_read(buf.handle, 0, LEN as u64).expect("warmup read");
        assert_eq!(sum0, expected, "GPU zero-copy read checksum mismatch");
        let (sum, gpu_ns, wall_ns) = bridge.bench_read(buf.handle, 0, LEN as u64).expect("bench read");
        assert_eq!(sum, expected);
        let read_gbps = (LEN as f64) / (gpu_ns.max(1) as f64);

        let seed = 0xdead_beefu32;
        let (w_gpu_ns, w_wall_ns) = bridge
            .bench_write(buf.handle, 0, LEN as u64, seed)
            .expect("bench write");
        let write_gbps = (LEN as f64) / (w_gpu_ns.max(1) as f64);

        // SAFETY: GPU command buffer already completed; read back directly on CPU.
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, words);
            for (idx, &val) in [0usize, 1, 4095, 65536, words / 2, words - 1]
                .into_iter()
                .map(|i| (i, &slice[i]))
            {
                assert_eq!(val, seed ^ (idx as u32), "GPU->CPU zero-copy mismatch at word {idx}");
            }
        }

        eprintln!(
            "victory_metal 64 MiB VirtualAlloc zero-copy: read {:.1} GB/s (gpu {:.2} ms, wall {:.2} ms) | write {:.1} GB/s (gpu {:.2} ms, wall {:.2} ms)",
            read_gbps,
            gpu_ns as f64 / 1e6,
            wall_ns as f64 / 1e6,
            write_gbps,
            w_gpu_ns as f64 / 1e6,
            w_wall_ns as f64 / 1e6,
        );

        assert!(bridge.unregister_view(base, false));
        // SAFETY: `ptr` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn metal_zero_copy_section_canonical_invariant() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        const LEN: usize = 4 << 20;
        // SAFETY: valid anonymous pagefile-backed section with two views.
        let (mapping, v1, v2) = unsafe {
            let m = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!m.is_null(), "CreateFileMappingW failed");
            let a = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, LEN);
            let b = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, LEN);
            assert!(!a.Value.is_null() && !b.Value.is_null() && a.Value != b.Value);
            (m, a, b)
        };

        let backing = BackingKey::PhysicalSection {
            section_id: 0x221_0004,
            offset: 0,
        };
        let (b1, reused1) = bridge
            .register_buffer(v1.Value as u64, LEN as u64, backing, false)
            .expect("register view 1");
        assert!(!reused1);

        let (b2, reused2) = bridge
            .register_buffer(v2.Value as u64, LEN as u64, backing, false)
            .expect("attach view 2");
        assert!(reused2, "second view must share canonical MTLBuffer (#221 #4)");
        assert_eq!(b1.handle, b2.handle);
        assert_eq!(b1.gpu_address, b2.gpu_address);

        let words = LEN / 4;
        let mut expected: u64 = 0;
        // SAFETY: `v2` maps `words` u32 elements of the shared section.
        unsafe {
            let s2 = std::slice::from_raw_parts_mut(v2.Value.cast::<u32>(), words);
            for (i, slot) in s2.iter_mut().enumerate() {
                let v = (i as u32).wrapping_mul(17) ^ 0x1357_2468;
                *slot = v;
                expected = expected.wrapping_add(u64::from(v));
            }
        }
        let resolved = bridge.resolve(v2.Value as u64 + 1024, 4096).expect("resolve view 2");
        assert_eq!(resolved.handle, b1.handle);
        assert_eq!(resolved.offset, 1024);

        let (sum, _, _) = bridge
            .bench_read(resolved.handle, 0, LEN as u64)
            .expect("GPU read canonical buffer after view 2 write");
        assert_eq!(sum, expected, "0 stale words across dual section views");

        assert!(bridge.unregister_view(v1.Value as u64, false));
        assert!(bridge.unregister_view(v2.Value as u64, false));
        // SAFETY: unmapping views and closing section handle.
        unsafe {
            UnmapViewOfFile(v2);
            UnmapViewOfFile(v1);
            CloseHandle(mapping);
        }
    }

    #[test]
    fn metal_irconverter_converts_dxil_to_metallib() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        if !bridge.info().irconverter {
            eprintln!("skipping: libmetalirconverter.dylib not loaded");
            return;
        }
        #[cfg(victory_dxc_detile)]
        {
            let dxil: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpudetile.dxil"));
            let (stage, metallib) = bridge
                .convert_dxil(dxil, "main")
                .expect("libmetalirconverter failed to convert gpudetile.dxil");
            assert!(metallib.len() >= 16, "metallib too small: {}", metallib.len());
            assert_eq!(&metallib[0..4], b"MTLB", "expected Apple MetalLib magic 'MTLB'");
            eprintln!(
                "victory_metal IRConverter: gpudetile.dxil ({} B) -> metallib ({} B, stage={})",
                dxil.len(),
                metallib.len(),
                stage,
            );
        }
    }

    #[test]
    fn metal_fx_temporal_scaler_upscales_960x540_to_1080p() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        if !bridge.info().metalfx {
            eprintln!("skipping: MetalFX temporal scaler unsupported on this device");
            return;
        }
        let (in_w, in_h) = (960u32, 540u32);
        let (out_w, out_h) = (1920u32, 1080u32);
        let scaler = bridge
            .create_temporal_scaler(&TemporalScalerDesc {
                input_width: in_w,
                input_height: in_h,
                output_width: out_w,
                output_height: out_h,
                color_format: pf::RGBA16_FLOAT,
                depth_format: pf::R32_FLOAT,
                motion_format: pf::RG16_FLOAT,
                output_format: pf::RGBA16_FLOAT,
                reactive_format: 0,
                auto_exposure: true,
                sync_init: true,
            })
            .expect("create MTLFXTemporalScaler");

        let color_tex = bridge.create_texture(in_w, in_h, pf::RGBA16_FLOAT, false).unwrap();
        let depth_tex = bridge.create_texture(in_w, in_h, pf::R32_FLOAT, false).unwrap();
        let motion_tex = bridge.create_texture(in_w, in_h, pf::RG16_FLOAT, false).unwrap();
        let out_tex = bridge.create_texture(out_w, out_h, pf::RGBA16_FLOAT, false).unwrap();

        // Flat warm HDR color (0.25, 0.50, 0.75, 1.0) + flat depth 0.5 + zero motion.
        let px = [
            f32_to_f16(0.25),
            f32_to_f16(0.50),
            f32_to_f16(0.75),
            f32_to_f16(1.0),
        ];
        let color_buf: Vec<[u16; 4]> = vec![px; (in_w * in_h) as usize];
        let depth_buf: Vec<f32> = vec![0.5f32; (in_w * in_h) as usize];
        let motion_buf: Vec<[u16; 2]> = vec![[0u16, 0u16]; (in_w * in_h) as usize];

        assert!(bridge.upload_texture_from_ptr(
            color_tex,
            color_buf.as_ptr().cast(),
            in_w * 8,
            in_w,
            in_h
        ));
        assert!(bridge.upload_texture_from_ptr(
            depth_tex,
            depth_buf.as_ptr().cast(),
            in_w * 4,
            in_w,
            in_h
        ));
        assert!(bridge.upload_texture_from_ptr(
            motion_tex,
            motion_buf.as_ptr().cast(),
            in_w * 4,
            in_w,
            in_h
        ));

        let mut last_gpu_ns = 0;
        let mut last_wall_ns = 0;
        for frame in 0..3 {
            let (gpu_ns, wall_ns) = bridge
                .encode_temporal_scaler(
                    scaler,
                    &TemporalScalerDispatch {
                        color_tex,
                        depth_tex,
                        motion_tex,
                        reactive_tex: 0,
                        output_tex: out_tex,
                        input_content_width: in_w,
                        input_content_height: in_h,
                        jitter_x: 0.0,
                        jitter_y: 0.0,
                        mv_scale_x: 1.0,
                        mv_scale_y: 1.0,
                        pre_exposure: 1.0,
                        reset: frame == 0,
                        depth_reversed: true,
                    },
                )
                .expect("encode MTLFXTemporalScaler");
            last_gpu_ns = gpu_ns;
            last_wall_ns = wall_ns;
        }

        let mut out_buf = vec![[0u16; 4]; (out_w * out_h) as usize];
        assert!(bridge.readback_texture_to_ptr(
            out_tex,
            out_buf.as_mut_ptr().cast(),
            out_w * 8,
            out_w,
            out_h
        ));
        let mid = out_buf[(out_h / 2 * out_w + out_w / 2) as usize];
        let rgba = [
            f16_to_f32(mid[0]),
            f16_to_f32(mid[1]),
            f16_to_f32(mid[2]),
            f16_to_f32(mid[3]),
        ];
        assert!(
            (rgba[0] - 0.25).abs() < 0.08
                && (rgba[1] - 0.50).abs() < 0.08
                && (rgba[2] - 0.75).abs() < 0.08,
            "unexpected MetalFX temporal scaler output: {rgba:?}"
        );
        eprintln!(
            "victory_metal MTLFXTemporalScaler 960x540 -> 1920x1080: gpu {:.2} ms, wall {:.2} ms, center={rgba:?}",
            last_gpu_ns as f64 / 1e6,
            last_wall_ns as f64 / 1e6,
        );

        bridge.destroy_texture(color_tex);
        bridge.destroy_texture(depth_tex);
        bridge.destroy_texture(motion_tex);
        bridge.destroy_texture(out_tex);
        bridge.destroy_temporal_scaler(scaler);
    }

    #[test]
    fn metal_fx_frame_interpolator_1080p_with_4k_ui_and_optical_flow() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        if !bridge.info().frame_interpolator {
            eprintln!("skipping: MTLFXFrameInterpolator requires macOS 26+");
            return;
        }
        // Series S geometry: 1920x1080 Plane 0 scene + 3840x2160 Plane 1 UI!
        let (w, h) = (1920u32, 1080u32);
        let (ui_w, ui_h) = (3840u32, 2160u32);
        let interp = bridge
            .create_frame_interpolator(&FrameInterpolatorDesc {
                input_width: w / 2,
                input_height: h / 2,
                output_width: w,
                output_height: h,
                color_format: pf::RGBA8_UNORM,
                output_format: pf::RGBA8_UNORM,
                depth_format: pf::R32_FLOAT,
                motion_format: pf::RG16_FLOAT,
                ui_format: pf::RGBA8_UNORM,
            })
            .expect("create MTLFXFrameInterpolator");

        let color_tex = bridge.create_texture(w, h, pf::RGBA8_UNORM, false).unwrap();
        let ui_tex = bridge.create_texture(ui_w, ui_h, pf::RGBA8_UNORM, false).unwrap();
        let out_tex = bridge.create_texture(w, h, pf::RGBA8_UNORM, false).unwrap();

        // Build two scene frames with a bright block moving right by +40 px.
        let make_scene = |box_x: u32| -> Vec<[u8; 4]> {
            let mut buf = vec![[20u8, 30, 50, 255]; (w * h) as usize];
            for y in 400..600 {
                for x in box_x..(box_x + 160).min(w) {
                    buf[(y * w + x) as usize] = [240, 180, 60, 255];
                }
            }
            buf
        };
        let frame0 = make_scene(600);
        let frame1 = make_scene(640);

        // Separate 4K Plane 1 UI: transparent everywhere except an opaque red
        // HUD banner in the top-left corner (x in 0..400, y in 0..160 -> 0..200, 0..80 at 1080p).
        let mut ui_buf = vec![[0u8, 0, 0, 0]; (ui_w * ui_h) as usize];
        for y in 0..160 {
            for x in 0..400 {
                ui_buf[(y * ui_w + x) as usize] = [255, 40, 40, 255];
            }
        }
        assert!(bridge.upload_texture_from_ptr(ui_tex, ui_buf.as_ptr().cast(), ui_w * 4, ui_w, ui_h));

        // Frame 0: seed history (interpolated == false).
        assert!(bridge.upload_texture_from_ptr(color_tex, frame0.as_ptr().cast(), w * 4, w, h));
        let r0 = bridge
            .encode_frame_interpolator(
                interp,
                &FrameInterpolatorDispatch {
                    color_tex,
                    prev_color_tex: 0,
                    depth_tex: 0,
                    motion_tex: 0, // triggers built-in Metal optical flow
                    ui_tex,
                    output_tex: out_tex,
                    jitter_x: 0.0,
                    jitter_y: 0.0,
                    mv_scale_x: 1.0,
                    mv_scale_y: 1.0,
                    delta_time: 1.0 / 30.0,
                    near_plane: 0.1,
                    far_plane: 1000.0,
                    fov_degrees: 60.0,
                    reset: true,
                    depth_reversed: false,
                    ui_composited: false,
                    skip_prev_update: false,
                },
            )
            .expect("seed frame 0");
        assert!(!r0.interpolated, "first frame should seed history");

        // Frame 1: generate interpolated mid-frame with optical flow + 4K UI Plane 1!
        assert!(bridge.upload_texture_from_ptr(color_tex, frame1.as_ptr().cast(), w * 4, w, h));
        let r1 = bridge
            .encode_frame_interpolator(
                interp,
                &FrameInterpolatorDispatch {
                    color_tex,
                    prev_color_tex: 0,
                    depth_tex: 0,
                    motion_tex: 0,
                    ui_tex,
                    output_tex: out_tex,
                    jitter_x: 0.0,
                    jitter_y: 0.0,
                    mv_scale_x: 1.0,
                    mv_scale_y: 1.0,
                    delta_time: 1.0 / 30.0,
                    near_plane: 0.1,
                    far_plane: 1000.0,
                    fov_degrees: 60.0,
                    reset: false,
                    depth_reversed: false,
                    ui_composited: false,
                    skip_prev_update: false,
                },
            )
            .expect("interpolate frame 1");
        assert!(r1.interpolated, "second frame must produce an interpolated frame");

        let mut out_buf = vec![[0u8; 4]; (w * h) as usize];
        assert!(bridge.readback_texture_to_ptr(out_tex, out_buf.as_mut_ptr().cast(), w * 4, w, h));

        // 1) Crisp HUD banner preserved on top-left (zero ghosting!)
        let hud_px = out_buf[(40 * w + 100) as usize];
        assert!(
            hud_px[0] > 200 && hud_px[1] < 90 && hud_px[2] < 90,
            "Plane 1 UI overlay not preserved cleanly: {hud_px:?}"
        );
        // 2) Background pixel outside moving box stays dark blue.
        let bg_px = out_buf[(900 * w + 100) as usize];
        assert!(
            bg_px[0] < 60 && bg_px[2] < 90,
            "unexpected background pixel: {bg_px:?}"
        );
        // 3) Center of moving box at midpoint (x ~ 700, y = 500) is bright gold.
        let mid_px = out_buf[(500 * w + 700) as usize];
        assert!(
            mid_px[0] > 180 && mid_px[1] > 130,
            "interpolated moving block missing at midpoint: {mid_px:?}"
        );

        // Warm frames 1..=3: measure steady-state GPU time after first-frame compile.
        let mut steady_gpu_ns = r1.gpu_ns;
        let mut steady_wall_ns = r1.wall_ns;
        for step in 2..=4u32 {
            let next_scene = make_scene(600 + step * 40);
            assert!(bridge.upload_texture_from_ptr(color_tex, next_scene.as_ptr().cast(), w * 4, w, h));
            let r = bridge
                .encode_frame_interpolator(
                    interp,
                    &FrameInterpolatorDispatch {
                        color_tex,
                        prev_color_tex: 0,
                        depth_tex: 0,
                        motion_tex: 0,
                        ui_tex,
                        output_tex: out_tex,
                        jitter_x: 0.0,
                        jitter_y: 0.0,
                        mv_scale_x: 1.0,
                        mv_scale_y: 1.0,
                        delta_time: 1.0 / 30.0,
                        near_plane: 0.1,
                        far_plane: 1000.0,
                        fov_degrees: 60.0,
                        reset: false,
                        depth_reversed: false,
                        ui_composited: false,
                        skip_prev_update: false,
                    },
                )
                .expect("steady-state interpolate");
            assert!(r.interpolated);
            steady_gpu_ns = r.gpu_ns;
            steady_wall_ns = r.wall_ns;
        }

        eprintln!(
            "victory_metal MTLFXFrameInterpolator 1080p + 4K UI (optical flow): first gpu {:.2} ms, steady gpu {:.2} ms (wall {:.2} ms), hud={hud_px:?}, mid={mid_px:?}",
            r1.gpu_ns as f64 / 1e6,
            steady_gpu_ns as f64 / 1e6,
            steady_wall_ns as f64 / 1e6,
        );

        bridge.destroy_texture(color_tex);
        bridge.destroy_texture(ui_tex);
        bridge.destroy_texture(out_tex);
        bridge.destroy_frame_interpolator(interp);
    }
}
