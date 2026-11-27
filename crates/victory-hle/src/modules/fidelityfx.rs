//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`). Without it, or if it offers no
//! upscaler for the host GPU, this module stays out of the registry and the
//! title's own console DLLs run as before.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Only the upscaler: other
//! effects (frame generation) are refused.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::OnceLock;

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
const DLL: &str = "amd_fidelityfx_upscaler_dx12.dll";

// Return codes.
const OK: u32 = 0;
const UNKNOWN_TYPE: u32 = 2;
const NO_PROVIDER: u32 = 4;
const PARAMETER: u32 = 6;

// Description types.
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
const QUERY_PHASE_COUNT: u64 = 0x1_0004;
const QUERY_JITTER_OFFSET: u64 = 0x1_0005;
const QUERY_MEMORY_USAGE: u64 = 0x1_0008;
const CONSOLE_BACKEND: u64 = 0x100_0002;
const HOST_BACKEND: u64 = 2;
const GET_VERSIONS: u64 = 4;
const GET_PROVIDER_VERSION: u64 = 6;
const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;

// CreateUpscale flags (FidelityFX SDK).
const FLAG_DEPTH_INVERTED: u32 = 1 << 3;
const FLAG_AUTO_EXPOSURE: u32 = 1 << 5;

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// The backend description: a device.
#[repr(C)]
struct Backend {
    header: Header,
    device: *mut c_void,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    rest: [u32; 18],
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);

/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
struct GetVersions {
    header: Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}

#[repr(C)]
struct ProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

#[repr(C)]
struct PhaseCount {
    header: Header,
    render_width: u32,
    output_width: u32,
    count: *mut i32,
}

#[repr(C)]
struct JitterOffset {
    header: Header,
    index: i32,
    phases: i32,
    x: *mut f32,
    y: *mut f32,
}

#[repr(C)]
struct MemoryUsage {
    header: Header,
    usage: *mut [u64; 2],
}

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

struct Host {
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

static HOST: OnceLock<Host> = OnceLock::new();

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

/// Serves the title's FidelityFX calls with the PC upscaler in `dir`
/// (`VICTORY_FIDELITYFX_DIR`) if it offers one for the host GPU; else the
/// title's own DLLs run. Call once, before the import registry is first
/// used: it holds this module only when serving.
pub fn use_host_dir(dir: Option<&Path>) {
    let Some(dir) = dir else {
        if metalfx_active() {
            tracing::info!(
                "FidelityFX: Apple MetalFX Temporal Scaler serves the title's upscaler (VICTORY_UPSCALER=metalfx)"
            );
        } else {
            tracing::info!(
                "no FidelityFX folder (VICTORY_FIDELITYFX_DIR): a title's own FidelityFX DLLs run"
            );
        }
        return;
    };
    match load(dir) {
        Ok((host, versions)) => {
            tracing::info!(folder = %dir.display(), versions, "FidelityFX: the host's upscaler serves the title's");
            let _ = HOST.set(host);
        }
        Err(why) => tracing::warn!(
            folder = %dir.display(),
            "FidelityFX: {why}; the title's own DLLs run"
        ),
    }
}

/// Whether native Apple `MTLFXTemporalScaler` serves the title's FidelityFX
/// upscaler (`VICTORY_UPSCALER=metalfx` or `VICTORY_METAL_FX=1`).
fn metalfx_active() -> bool {
    HOST.get().is_none()
        && victory_gpu::metal::metalfx_upscaler_enabled()
        && victory_gpu::metal::bridge::global().is_some_and(|b| b.info().metalfx)
}

/// Whether the host's FidelityFX (or native Apple MetalFX) serves the title's.
pub(crate) fn serving() -> bool {
    HOST.get().is_some() || metalfx_active()
}

fn load(dir: &Path) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(dir.join(DLL)).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!("no {DLL} there"));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{DLL} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{DLL} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query: std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?),
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            device,
        }
    };
    let versions = versions(&host)?;
    Ok((host, versions))
}

/// The upscaler versions the host DLL offers for the host GPU, by name; an
/// error if none.
fn versions(host: &Host) -> Result<String, String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind: CREATE_UPSCALE,
        device: host.device.as_raw(),
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { (host.query)(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no upscaler for the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    query.names = names.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    unsafe { (host.query)(null_mut(), &mut query.header) };
    Ok(names
        .iter()
        .filter(|n| !n.is_null())
        // SAFETY: NUL-terminated names the DLL keeps.
        .map(|&n| unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", "))
}

/// Puts the host backend after the title's upscaler description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    let after = upscale.header.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    upscale.header.next = &raw mut backend.header;
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    // SAFETY: an upscaler description, by its type.
    let mut upscale = unsafe { desc.cast::<CreateUpscale>().read() };
    if metalfx_active() {
        let auto_exposure = upscale.flags & FLAG_AUTO_EXPOSURE != 0;
        let depth_reversed = upscale.flags & FLAG_DEPTH_INVERTED != 0;
        let Some(fx) = victory_gpu::metal::fx::MetalFxUpscaler::new(
            upscale.max_render,
            upscale.max_output,
            auto_exposure,
            depth_reversed,
        ) else {
            return NO_PROVIDER;
        };
        tracing::info!(
            provider = "MetalFX Temporal Scaler",
            render = ?upscale.max_render,
            output = ?upscale.max_output,
            flags = format_args!("{:#x}", upscale.flags),
            "FidelityFX: upscaler context on Apple MetalFX (VICTORY_UPSCALER=metalfx)"
        );
        // SAFETY: title provided writable `context` pointer; released in `ffxDestroyContext`.
        unsafe { context.write(Box::into_raw(Box::new(fx)).cast()) };
        return OK;
    }
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    let mut backend = Backend {
        header: Header {
            kind: HOST_BACKEND,
            next: null_mut(),
        },
        device: host.device.as_raw(),
    };
    link(&mut upscale, &mut backend);
    // SAFETY: a PC chain; the title's later descriptions live for the call.
    let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
    if code != OK {
        tracing::error!(code, render = ?upscale.max_render, output = ?upscale.max_output, flags = format_args!("{:#x}", upscale.flags), "FidelityFX: the host refused the title's upscaler");
        return code;
    }
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    unsafe { (host.query)(context, &mut version.header) };
    // SAFETY: a NUL-terminated name the DLL keeps, if any.
    let name = (!version.name.is_null())
        .then(|| unsafe { CStr::from_ptr(version.name) }.to_string_lossy());
    tracing::info!(
        provider = name.as_deref().unwrap_or("?"),
        id = format_args!("{:#x}", version.id),
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        "FidelityFX: upscaler context on the host"
    );
    OK
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if metalfx_active() {
        if context.is_null() {
            return PARAMETER;
        }
        // SAFETY: context was created by `ffxCreateContext` with `Box::into_raw`.
        let ptr = unsafe { *context };
        if !ptr.is_null() {
            // SAFETY: reclaiming the boxed `MetalFxUpscaler`.
            unsafe {
                drop(Box::from_raw(ptr.cast::<victory_gpu::metal::fx::MetalFxUpscaler>()));
                *context = null_mut();
            }
        }
        return OK;
    }
    // SAFETY: the title passes its context.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.destroy)(context, memory) })
}

/// Settings (key-value pairs, debug messages) name nothing console-specific.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if metalfx_active() {
        return OK;
    }
    // SAFETY: the title passes its context and descriptions.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.configure)(context, desc) })
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if metalfx_active() {
        return match kind {
            GET_PROVIDER_VERSION => {
                // SAFETY: caller supplied a `ProviderVersion` struct.
                let out = unsafe { &mut *desc.cast::<ProviderVersion>() };
                out.id = 0x4D46_5831; // "MFX1"
                out.name = c"MetalFX Temporal Scaler".as_ptr();
                OK
            }
            QUERY_PHASE_COUNT => {
                // SAFETY: caller supplied a `PhaseCount` struct.
                let q = unsafe { &*desc.cast::<PhaseCount>() };
                if q.count.is_null() {
                    return PARAMETER;
                }
                // SAFETY: valid writable count pointer.
                unsafe {
                    q.count.write(victory_gpu::metal::fx::jitter_phase_count(
                        q.render_width,
                        q.output_width,
                    ));
                }
                OK
            }
            QUERY_JITTER_OFFSET => {
                // SAFETY: caller supplied a `JitterOffset` struct.
                let q = unsafe { &*desc.cast::<JitterOffset>() };
                if q.x.is_null() || q.y.is_null() {
                    return PARAMETER;
                }
                let (jx, jy) = victory_gpu::metal::fx::jitter_offset(q.index, q.phases);
                // SAFETY: valid writable float pointers.
                unsafe {
                    q.x.write(jx);
                    q.y.write(jy);
                }
                OK
            }
            QUERY_MEMORY_USAGE => {
                // SAFETY: caller supplied a `MemoryUsage` struct.
                let q = unsafe { &*desc.cast::<MemoryUsage>() };
                if q.usage.is_null() {
                    return PARAMETER;
                }
                let bytes = if !context.is_null() {
                    // SAFETY: valid `MetalFxUpscaler` context pointer.
                    let ptr = unsafe { *context };
                    if !ptr.is_null() {
                        // SAFETY: valid live `MetalFxUpscaler`.
                        unsafe { &*ptr.cast::<victory_gpu::metal::fx::MetalFxUpscaler>() }
                            .estimated_gpu_bytes()
                    } else {
                        [64 << 20, 32 << 20]
                    }
                } else {
                    [64 << 20, 32 << 20]
                };
                // SAFETY: valid writable `[u64; 2]` pointer.
                unsafe { q.usage.write(bytes) };
                OK
            }
            GET_VERSIONS | MEMORY_USAGE_WITH_DEVICE => PARAMETER,
            _ => UNKNOWN_TYPE,
        };
    }
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    // These carry the console device.
    if kind == GET_VERSIONS || kind == MEMORY_USAGE_WITH_DEVICE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: a query naming a device was refused"
        );
        return PARAMETER;
    }
    // SAFETY: as above.
    unsafe { (host.query)(context, desc) }
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    // SAFETY: an upscaling, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    let list = dispatch.list.cast::<Object>().cast_const();
    if metalfx_active() {
        if context.is_null() {
            return PARAMETER;
        }
        // SAFETY: `context` holds a `MetalFxUpscaler` created by `ffxCreateContext`.
        let ptr = unsafe { *context };
        if ptr.is_null() {
            return PARAMETER;
        }
        // SAFETY: live `MetalFxUpscaler`.
        let fx = unsafe { &*ptr.cast::<victory_gpu::metal::fx::MetalFxUpscaler>() };
        // Execute prior commands recorded into `list` so MetalFX reads rendered color/depth/MV.
        if !victory_gpu::external::flush_and_reopen(list) {
            return PARAMETER;
        }
        let get = |idx: usize| -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)> {
            let r = dispatch.resources[idx];
            if r.resource.is_null() {
                return None;
            }
            victory_gpu::external::current_texture(r.resource.cast::<Object>())
        };
        let (Some(color), Some(depth), Some(motion), Some(output)) = (get(0), get(1), get(2), get(6))
        else {
            return PARAMETER;
        };
        let reactive = get(4);
        let rest = &dispatch.rest;
        let inputs = victory_gpu::metal::fx::UpscaleInputs {
            color: (&color.0, color.1),
            depth: (&depth.0, depth.1),
            motion: (&motion.0, motion.1),
            reactive: reactive.as_ref().map(|(t, s)| (t, *s)),
            output: (&output.0, output.1),
            render_size: [rest[4], rest[5]],
            output_size: [rest[6], rest[7]],
            jitter: [f32::from_bits(rest[0]), f32::from_bits(rest[1])],
            motion_scale: [f32::from_bits(rest[2]), f32::from_bits(rest[3])],
            frame_ms: f32::from_bits(rest[10]),
            pre_exposure: f32::from_bits(rest[11]),
            reset: rest[12] != 0,
            near_plane: f32::from_bits(rest[13]),
            far_plane: f32::from_bits(rest[14]),
            fov_radians: f32::from_bits(rest[15]),
        };
        let Some((gpu_ns, wall_ns)) = fx.dispatch(&inputs) else {
            return PARAMETER;
        };
        once!(
            info,
            gpu_ms = gpu_ns as f64 / 1e6,
            wall_ms = wall_ns as f64 / 1e6,
            render = ?inputs.render_size,
            output = ?inputs.output_size,
            "FidelityFX: first MetalFX Temporal Scaler upscale completed"
        );
        // Leave all resources in the title's requested post-dispatch states on `list`.
        for r in dispatch.resources.iter().filter(|r| !r.resource.is_null()) {
            if let Some(w) = host_state(r.state) {
                let _ = victory_gpu::external::host_texture(list, r.resource.cast::<Object>(), w);
            }
        }
        return OK;
    }
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    // Alive for the call (the list holds them too).
    let mut textures = Vec::new();
    for r in dispatch
        .resources
        .iter_mut()
        .filter(|r| !r.resource.is_null())
    {
        let wanted = host_state(r.state);
        let texture = wanted.and_then(|w| {
            victory_gpu::external::host_texture(list, r.resource.cast::<Object>(), w)
                .filter(|&(_, state)| state == w)
        });
        let Some((texture, _)) = texture else {
            once!(
                warn,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return PARAMETER;
        };
        r.resource = texture.as_raw();
        textures.push(texture);
    }
    once!(info, formats = ?textures.iter().map(|t| {
        // SAFETY: a live texture.
        unsafe { t.GetDesc() }.Format.0
    }).collect::<Vec<_>>(), "FidelityFX: first upscaling, host formats of its textures");
    victory_gpu::external::record(list, |host_list| {
        dispatch.list = host_list.as_raw();
        // SAFETY: the PC dispatch on host objects; the context is the host's.
        unsafe { (host.dispatch)(context, &raw mut dispatch.header) }
    })
    .unwrap_or(PARAMETER)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).err().unwrap().contains(DLL));
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }
}
