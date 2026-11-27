//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// A console state as a host texture state: PC bits only (not the console's
/// own, compression hints from bit 25 up, which the host rejects), and none
/// of the buffer-only ones the console allows on textures (vertex, constant
/// and index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    state & 0x01ff_ffff & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

thread_local! {
    /// The thread holds the submission lock ([`Host::execute`] or
    /// [`Host::quiet`]).
    static HOLDS_SUBMISSIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The thread holds the submission lock while this lives.
struct Holding(bool);

impl Holding {
    fn enter() -> Self {
        Holding(HOLDS_SUBMISSIONS.with(|h| h.replace(true)))
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        HOLDS_SUBMISSIONS.with(|h| h.set(self.0));
    }
}

/// mac: whether this thread holds the submission lock.
pub(crate) fn holds_submissions() -> bool {
    HOLDS_SUBMISSIONS.with(std::cell::Cell::get)
}

/// mac: runs `f` on this thread as a holder of the submission lock if
/// `holds`: for a thread that works for one that holds it and waits for
/// that work (`crate::descriptor::pool`), so that [`Host::quiet`] does not
/// wait there for the lock its own caller holds.
pub(crate) fn as_holder<R>(holds: bool, f: impl FnOnce() -> R) -> R {
    let _holding = holds.then(Holding::enter);
    f()
}

pub(crate) mod unwaited;

/// What the end of a submission ([`Host::finish`]) needs from its start:
/// how long the host GPU was waited for, the list the device was removed
/// in (when lists run one by one), what its refreshes and heaps hold until
/// it has run, and whether its frame is listed (diagnostics).
struct Ran {
    gpu: Duration,
    removed_in: Option<usize>,
    batch: Batch,
    heaps: Vec<std::sync::Arc<crate::bindless::Viewed>>,
    traced: bool,
}

/// Submissions held off ([`Host::quiet`]).
pub(crate) struct Quiet<'a> {
    // Dropped in this order: the mark, then the lock.
    _holding: Holding,
    _held: std::sync::MutexGuard<'a, u64>,
}

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    /// The direct queue; a new one replaces it when it stops running work
    /// ([`Host::lost`]).
    queue: Mutex<ID3D12CommandQueue>,
    pub(crate) depth_bounds_supported: bool,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// The list every submission ends with, made on first use ([`Mark`]).
    mark: OnceLock<Option<Mark>>,
    /// Extra marks for native calls before the final call of a split submission.
    split_marks: Mutex<Option<Vec<Mark>>>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on, and the mark's mapped
// buffers only used, under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// What shows that a submission ran: a list run last in every submission,
/// which copies the submission's number from upload memory into read-back
/// memory. Under D3DMetal, macOS ends a command buffer when the GPU locks up
/// on it (`kIOGPUCommandBufferCallbackErrorHang`: about 50 ms into a draw
/// whose every pixel runs long) or takes a page fault in it ("MMU
/// interrupt" in the gpuEvent report, VIctory-macOS #178), and after some
/// of those the queue ignores everything submitted to it, presents included
/// (`kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`), while its fence
/// still completes and the device is not removed (VIctory-macOS #50). A
/// list after the ended one in the same submission does not run either.
/// `WriteBufferImmediate` writes nothing under D3DMetal 4.0 beta 3, so this
/// is a copy.
struct Mark {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
    _buffers: [ID3D12Resource; 2],
    /// The number the list copies (upload memory).
    number: *mut u64,
    /// Where it lands (read-back memory).
    seen: *const u64,
}

impl Mark {
    fn new(host: &Host) -> windows::core::Result<Mark> {
        let (number, numbers) = host
            .mapped_buffer(8, false)
            .ok_or_else(windows::core::Error::empty)?;
        let (seen, seen_at) = host
            .mapped_buffer(8, true)
            .ok_or_else(windows::core::Error::empty)?;
        // SAFETY: plain object creation, and a copy between our buffers,
        // which stay in their states (copy source, copy destination).
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            list.CopyBufferRegion(&seen, 0, &number, 0, 8);
            list.Close()?;
            name(&list, || "submission mark".into());
            Ok(Mark {
                list,
                _allocator: allocator,
                _buffers: [number, seen],
                number: numbers.cast(),
                seen: seen_at.cast_const().cast(),
            })
        }
    }

    /// Sets the number the next run of the list copies. Under the
    /// submission lock, once the last submission has finished.
    fn arm(&self, value: u64) {
        // SAFETY: our mapped upload buffer, which no submission reads now.
        unsafe { self.number.write_volatile(value) }
    }

    /// Whether the last run of the list copied `value`.
    fn saw(&self, value: u64) -> bool {
        // SAFETY: our mapped read-back buffer.
        unsafe { self.seen.read_volatile() == value }
    }

    fn list(&self) -> Option<ID3D12CommandList> {
        self.list.cast().ok()
    }
}

/// Submissions that did not run to their mark, and queues made since to
/// replace one that stopped running work.
static LOST: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Lanes that gave up waiting for a 64-bit atomic's lock, over the run
/// (`crate::shadow::take_gave_up`).
static GAVE_UP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => {
                text.push_str("; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command")
            }
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // The frame recorder (`VICTORY_RECORD`, docs/replay.md), before
        // anything is made with the device: it sees what the device makes.
        crate::replay::record::from_env(&device);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool()
            // D3DMetal reports the test and does not apply it (Metal has
            // none): bounds that should reject a draw leave it whole, as when
            // the host has no test, and the plain pipeline states are safer.
            && !crate::under_wine();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        crate::pipeline::note_skips();
        crate::ambient::announce();
        crate::ray_skip::announce();
        crate::texback::announce();
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match victory_shader::emit::host::Host::from_device(device.clone()).fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            queue: Mutex::new(queue),
            depth_bounds_supported,
            fence,
            submitted: Mutex::new(0),
            mark: OnceLock::new(),
            split_marks: Mutex::new(Some(Vec::new())),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// Diagnostics: our process's private commit and working set, bytes.
#[cfg(feature = "diagnostics")]
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.PagefileUsage as u64, c.WorkingSetSize as u64)
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: GPU time per kind of command and
/// pipeline (ms, commands) and lists' wall time against the sum of their
/// commands' (ms), since the last frame-rate line.
#[cfg(feature = "diagnostics")]
type GpuTimes = (std::collections::HashMap<String, (f64, u64)>, f64, f64);
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// Adds a timed list's commands (from [`HostList::slowest`]) to [`GPU_TIMES`],
/// by the command's kind and pipeline: its note before the first ` | `,
/// without counts and addresses.
#[cfg(feature = "diagnostics")]
fn gpu_times(wall: Duration, commands: &[(f64, String)]) {
    let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, wall_ms, sum) = times.get_or_insert_with(Default::default);
    *wall_ms += wall.as_secs_f64() * 1e3;
    for (ms, note) in commands {
        // "draw 3x1 [mesh 1f.., pixel 2e..] | ...": the kind and the pipeline.
        let head = note.split(" | ").next().unwrap_or_default();
        let kind = head.split(' ').next().unwrap_or_default();
        let key = match head.find('[') {
            Some(at) => format!("{kind} {}", &head[at..]),
            None => kind.to_owned(),
        };
        let e = by.entry(key).or_default();
        (e.0, e.1) = (e.0 + ms, e.1 + 1);
        *sum += ms;
    }
}

/// Logs [`GPU_TIMES`] per frame over `frames` frames: the 25 costliest.
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let Some((by, wall, sum)) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!(
        "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame",
        wall / frames,
        sum / frames
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (key, (ms, n)) in by.iter().take(25) {
        tracing::info!("  GPU {:.2} ms/frame in {:.1} commands/frame: {key}", ms / frames, *n as f64 / frames);
    }
}

/// Diagnostics: the host device's video memory use and budget, in MB, local
/// (on the card) and non-local (system memory the GPU reads), as the
/// system's memory manager sees them: use over budget makes it page
/// allocations at submissions.
pub(crate) fn video_memory() -> Option<[(f64, f64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let device = &host()?.device;
            // SAFETY: plain device and DXGI queries.
            unsafe {
                let luid = device.GetAdapterLuid();
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid(luid).ok()
            }
        })
        .as_ref()?;
    let info = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a valid out pointer.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage as f64 / 1e6, info.Budget as f64 / 1e6))
    };
    Some([info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?])
}

#[cfg(test)]
pub(crate) mod tests {
    #[test]
    fn mesh_support_requires_both_feature_queries_to_confirm_support() {
        use super::MeshSupport;

        for (mesh_shaders, shader_model_6_6, supported) in [
            (Some(true), Some(true), true),
            (Some(false), Some(true), false),
            (Some(true), Some(false), false),
            (None, Some(true), false),
            (Some(true), None, false),
        ] {
            let features = MeshSupport {
                mesh_shaders,
                shader_model_6_6,
            };
            assert_eq!(features.supported(), supported);
        }
    }

    /// Run in isolation with VICTORY_PRESENT_MAX=1280x720. Exercise the
    /// real swap chain and its resize path without opening a visible window.
    #[test]
    #[ignore = "requires DXGI and an explicit 720p presentation cap"]
    fn capped_presentation_resizes_a_hidden_swap_chain() {
        use super::*;
        let host = host().expect("a host GPU");
        assert_eq!(crate::composite::output_size(1920, 1080), (1280, 720));
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: a hidden Win32 control owned by this test thread.
        let hwnd = unsafe {
            CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("VIctory display test"),
                WS_OVERLAPPEDWINDOW, 0, 0, 320, 180, None, None, None, None)
        }.unwrap();
        let presenter = host.presenter_for_window(&host.queue(), hwnd, 320, 180, format).unwrap();
        host.presenter.lock().unwrap().presenter = Some(presenter);
        let state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
        let picture = crate::composite::result_texture(&host.device, 1920, 1080, format).unwrap();
        let ui = crate::composite::result_texture(&host.device, 3840, 2160, format).unwrap();
        let small = crate::composite::result_texture(&host.device, 640, 360, format).unwrap();
        for (frame, (source, overlay, expected)) in [
            (&picture, None, (1280, 720)),
            (&picture, Some((&ui, state)), (1280, 720)),
            (&small, None, (640, 360)),
        ].into_iter().enumerate() {
            assert!(host.present(source, state, overlay, frame as u64 + 1).is_some());
            let presentation = host.presenter.lock().unwrap();
            let chain = &presentation.presenter.as_ref().unwrap().swap_chain;
            // SAFETY: a live swap chain after presenting and waiting for the copy.
            let desc = unsafe { chain.GetDesc1() }.unwrap();
            assert_eq!((desc.Width, desc.Height), expected);
            // SAFETY: plain query after executing the presentation command list.
            unsafe { host.device.GetDeviceRemovedReason() }.unwrap();
            eprintln!("presented frame {} on a {}x{} DXGI swap chain", frame + 1, desc.Width, desc.Height);
        }
        host.presenter.lock().unwrap().presenter.take();
        // SAFETY: the swap chain was released and this thread owns the window.
        unsafe { DestroyWindow(hwnd) }.unwrap();
    }

    #[derive(Default)]
    struct NativeCallTrace {
        groups: Vec<(Vec<usize>, bool)>,
        calls: Vec<usize>,
    }

    thread_local! {
        static NATIVE_CALLS: std::cell::RefCell<Option<NativeCallTrace>> =
            const { std::cell::RefCell::new(None) };

        /// Every transition barrier built on this thread (resource, before,
        /// after), in order.
        pub(crate) static ISSUED: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The barriers built on this thread since the last call.
    pub(crate) fn issued() -> Vec<(usize, u32, u32)> {
        ISSUED.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    pub(crate) fn start_native_call_trace() {
        NATIVE_CALLS.with(|trace| {
            assert!(trace.borrow_mut().replace(NativeCallTrace::default()).is_none());
        });
    }

    pub(crate) fn take_native_call_trace() -> Vec<(Vec<usize>, bool)> {
        NATIVE_CALLS.with(|trace| trace.borrow_mut().take().unwrap().groups)
    }

    pub(super) fn native_call(lists: usize) {
        NATIVE_CALLS.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.calls.push(lists);
            }
        });
    }

    pub(super) fn native_call_group(fence_waited: bool) {
        NATIVE_CALLS.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.groups.push((std::mem::take(&mut trace.calls), fence_waited));
            }
        });
    }

    /// Replays transitions in the order the GPU runs them from the states
    /// in `states` (resources not in it are not checked), by the debug
    /// layer's rule: each barrier's before-state is the resource's current
    /// one. The first that is not, else the states after.
    pub(crate) fn replay(issued: &[(usize, u32, u32)], states: &mut std::collections::HashMap<usize, u32>) -> Result<(), String> {
        for (k, &(resource, before, after)) in issued.iter().enumerate() {
            if let Some(state) = states.get_mut(&resource) {
                if *state != before {
                    return Err(format!("barrier {k} on {resource:#x}: before {before:#x}, but the resource is in {state:#x}"));
                }
                *state = after;
            }
        }
        Ok(())
    }

    #[test]
    fn frame_report_includes_the_frames_before_the_last_caption() {
        use super::FrameWindow;
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let sample = |seconds, frames, drawn| FrameWindow {
            since: start + Duration::from_secs(seconds),
            frames,
            drawn,
        };
        let mut caption = sample(0, 0, 10);
        let mut report = caption;
        let second = Duration::from_secs(1);
        let period = Duration::from_secs(5);
        let early = sample(4, 120, 370);
        assert_eq!(caption.sample(early, second).unwrap().fps, 30.0);
        assert!(report.sample(early, period).is_none());

        let end = sample(5, 125, 510);
        let short = caption.sample(end, second).unwrap();
        let full = report.sample(end, period).unwrap();
        assert_eq!((short.fps, short.ms, short.draws), (5.0, 200.0, 28.0));
        assert_eq!((full.frames, full.fps, full.ms, full.draws), (125.0, 25.0, 40.0, 4.0));
        assert!(report.sample(end, period).is_none());
        let following = report.sample(sample(10, 135, 530), period).unwrap();
        assert_eq!((following.frames, following.fps, following.draws), (10.0, 2.0, 2.0));
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    /// Shaders that use title memory mark their reads in their own bitmap:
    /// read marks of both bitmaps are taken, each slot once until it is
    /// forgotten, and a forgotten slot's next read is taken again.
    #[test]
    fn read_marks_of_both_bitmaps_are_taken() {
        let Some(host) = host() else { return };
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let (plain, memory) = (victory_shader::emit::HEAP_VIEWS - 2, victory_shader::emit::HEAP_VIEWS - 1);
        crate::shadow::tests::mark_read(false, plain);
        crate::shadow::tests::mark_read(true, memory);
        let taken = crate::shadow::take_reads();
        assert!(taken.contains(&plain), "a plain shader's read");
        assert!(taken.contains(&memory), "a read by a shader that uses title memory");
        let again = crate::shadow::take_reads();
        assert!(!again.contains(&plain) && !again.contains(&memory), "each once");
        crate::shadow::forget_read(memory);
        assert!(!crate::shadow::take_reads().contains(&memory), "forgotten: its mark cleared");
        crate::shadow::tests::mark_read(true, memory);
        assert!(crate::shadow::take_reads().contains(&memory), "read again after it was forgotten");
    }

    /// The textures shaders read through the heap since the last eviction
    /// pass are named again at each pass (a slot's read is reported as new
    /// only once), and the marks are cleared for the next.
    #[test]
    fn heap_reads_since_the_last_pass_name_their_textures() {
        let Some(host) = host() else { return };
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
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(texture).unwrap().texture) as usize;
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..2 {
            crate::shadow::tests::mark_read(false, 0);
            let slots = crate::shadow::reads_since();
            assert!(slots.contains(&0), "the slot read");
            assert!(crate::descriptor::read_textures(&slots).contains(&raw), "its texture named");
            assert!(!crate::shadow::reads_since().contains(&0), "the marks cleared");
        }
    }

    /// The idle rule against the heap reads that thrashed (run evict1): a
    /// texture shaders read through the heap in every frame stays over more
    /// frames and time than the idle limits; one read once is given up. A
    /// list of the title's naming a texture (a table read, a copy) stamps it
    /// as used; a list of ours (uploads) does not.
    #[test]
    fn textures_read_every_frame_stay_and_title_lists_stamp_theirs() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let step = 32;
        // Two 4x1 RGBA8 textures, linear, in memory of their own, viewed at
        // slots 5 and 6.
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let heap_host = title_heap.host().expect("a host view heap");
        let textures: Vec<crate::object::This> = [5u32, 6]
            .iter()
            .map(|&slot| {
                let address = memory.as_ptr() as u64 + u64::from(slot) * (1 << 13);
                let mut view = [0u32; 16];
                view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                let at = title_heap.start + u64::from(slot) * step;
                crate::descriptor::create_placed_srv(null, &words, &view, at);
                assert!(heap_host.mirror(slot, at, true).is_some());
                crate::resource::texture_at(address, &words).expect("its resource")
            })
            .collect();
        let owned: Vec<_> = textures.iter().map(|&t| crate::resource::owned(t).expect("may be given up")).collect();
        let (idle_frames, idle_ms) = (crate::resource::IDLE_FRAMES, crate::resource::IDLE_MS);
        {
            // No submission takes the marks meanwhile.
            let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Passes a third of the idle limits apart, over three times them.
            for k in 0..10 {
                crate::shadow::tests::mark_read(false, 5);
                if k == 0 {
                    crate::shadow::tests::mark_read(false, 6);
                }
                let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
                crate::resource::evict_among(&owned, &read, 1000 + k * idle_frames / 3, 1_000_000 + k * idle_ms / 3, 0);
                assert!(crate::resource::made_copy(textures[0]).is_some(), "read every frame: stays (frame {k})");
            }
        }
        assert!(crate::resource::made_copy(textures[1]).is_none(), "read once: given up");
        drop(owned);
        // A title list naming the texture stamps it, an upload list does not.
        let copy = crate::resource::host_copy(textures[0]).unwrap();
        for (ours, stamped) in [(true, false), (false, true)] {
            copy.unstamp();
            let list = host.new_list().unwrap();
            if ours {
                list.set_ours();
            }
            let _ = list.transition(textures[0], 0x40);
            assert_eq!(copy.last_used().0 != u64::MAX, stamped, "ours: {ours}");
        }
    }

    /// Textures the heaps view are brought up to date only if the title used
    /// them lately: one whose pages changed but that no shader read is not
    /// uploaded; read in the last pass, it is.
    #[test]
    fn heap_textures_unread_lately_are_not_brought_up() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(null, address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        viewed.set(0, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        copy.unstamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 0, "not read lately: not brought up");
        copy.stamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 1, "read in the last pass: brought up");
    }

    /// Heap textures brought up in one submission are refreshed side by
    /// side, each into its own staging and its own upload.
    #[test]
    fn heap_textures_brought_up_together_each_get_their_upload() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let pool = crate::watch::tests::pool(6);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let objects: Vec<crate::object::This> = (0..6u32)
            .map(|k| {
                // RGBA8 128x128 (one page each), linear, each its own bytes.
                let address = (pool + k as usize * crate::watch::PAGE) as u64;
                // SAFETY: the texture's own page of the pool.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x10 * k as u8, crate::watch::PAGE) };
                let words: [u32; 14] = [3, 0, 0, 0, 128, 0, 128, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                let mut object = std::ptr::null();
                assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
                crate::resource::host_copy(object).expect("a host texture").stamp();
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                viewed.set(k, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
                object
            })
            .collect();
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "each brought up once");
        }
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "unchanged: not again");
        }
    }

    #[test]
    fn list_scratch_grows_to_fit_and_keeps_what_the_last_cycle_reached() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let chunks = || list.scratch.lock().unwrap().chunks.len();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        let wide = list.scratch(200 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[2].capacity, 256 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        // The cycle reached the third chunk: all kept, at their addresses.
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch(200 << 10).unwrap(), wide);
        assert_eq!(chunks(), 3);
        list.close();
        host.execute(&[&list]);
        list.reset();
        // One that reached the second: the third stays (one chunk of slack).
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(chunks(), 3);
        // One that stayed in the first: the third goes (the old rule kept
        // every chunk for good), the second stays.
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(chunks(), 2);
        assert_eq!(list.scratch(512).unwrap(), first);
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        list.close();
    }

    #[test]
    fn list_uploads_share_scratch_chunks_at_texture_alignment() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let (first, at, _) = list.upload(1).unwrap();
        assert_eq!(at, 0);
        // A 256-byte scratch use in between leaves the next upload at 512.
        let (gpu, _) = list.scratch(1).unwrap();
        let (second, next, _) = list.upload(700).unwrap();
        assert_eq!(next, 512);
        assert_eq!(first, second, "one chunk, not a buffer per copy");
        // SAFETY: a live buffer.
        assert_eq!(gpu, unsafe { first.GetGPUVirtualAddress() } + 256);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        // A copy larger than a chunk has a buffer of its own, at 0.
        let (big, at, _) = list.upload(SCRATCH_CHUNK + 1).unwrap();
        assert_eq!(at, 0);
        assert_ne!(big.as_raw(), first.as_raw());
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        list.close();
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }

    #[test]
    fn parallel_batch_flush_updates_multiple_shadows_identically() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if super::host().is_none() {
            return;
        }
        let len = 0x40000usize;
        let mut ptrs = Vec::new();
        let mut shadows = Vec::new();
        for i in 0..3usize {
            // SAFETY: fresh committed allocation, freed below.
            let p = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
            assert_ne!(p, 0);
            let pattern = (0x21 + i as u8).wrapping_mul(3);
            // SAFETY: `len` committed bytes at `p`.
            unsafe { std::ptr::write_bytes(p as *mut u8, pattern, len) };
            let id = 0x9100 + i;
            crate::shadow::register(id, p as u64, len as u64);
            let shadow = crate::shadow::find(p as u64, len as u64).expect("shadow");
            ptrs.push((id, p, pattern));
            shadows.push(shadow);
        }
        let mut batch = super::Batch::default();
        for (i, &(_, p, _)) in ptrs.iter().enumerate() {
            batch.add(&shadows[i], p, len);
        }
        let copied = batch.flush();
        assert!(copied >= 3 * len);
        for (i, &(_, _, pattern)) in ptrs.iter().enumerate() {
            assert_eq!(shadows[i].read_host(ptrs[i].1 as u64, 4), vec![u32::from_le_bytes([pattern; 4]); 4]);
            crate::shadow::unregister(ptrs[i].0, ptrs[i].1 as u64);
            // SAFETY: our allocation.
            unsafe { VirtualFree(ptrs[i].1 as *mut _, 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn bind_pipeline_deduplicates_kept_objects_until_reset() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let Ok((root, pso)) = crate::compute::pipeline(
            "[numthreads(1, 1, 1)] void main() {}",
            windows::core::s!("bind_pipeline_test"),
            0,
            &[],
        ) else {
            list.close();
            return;
        };

        list.bind_pipeline("test_pso", &pso, &root);
        list.bind_pipeline("test_pso", &pso, &root);
        assert_eq!(list.deferred().kept.len(), 2);
        assert_eq!(list.deferred().bound_pipelines.len(), 1);

        list.close();
        list.reset();
        assert!(list.deferred().kept.is_empty());
        assert!(list.deferred().bound_pipelines.is_empty());
        list.bind_pipeline("test_pso", &pso, &root);
        assert_eq!(list.deferred().kept.len(), 2);
        list.close();
    }

    #[test]
    fn fill_scratch_reuses_upload_until_title_copy_and_reset() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let data = [0x11u8; 64];
        let a = list.fill_scratch(data.as_ptr(), 64).expect("first fill");
        let b = list.fill_scratch(data.as_ptr(), 64).expect("cached fill");
        assert_eq!(a, b);
        assert_eq!(list.deferred().copies.len(), 1);

        let mut dst = [0u8; 16];
        list.defer_copy(dst.as_mut_ptr(), data.as_ptr(), 16);
        let c = list.fill_scratch(data.as_ptr(), 64).expect("after title copy");
        assert_ne!(a, c);
        assert_eq!(list.deferred().copies.len(), 3);

        list.close();
        list.reset();
        let d = list.fill_scratch(data.as_ptr(), 64).expect("after reset");
        assert_eq!(d, a, "scratch restarts from chunk 0 after reset");
        list.close();
    }
}

type WidenedWrite = (std::sync::Arc<crate::shadow::Shadow>, usize, usize);

/// UAV writes and heap-writing draws in command-recording order.
enum WidenedEvent {
    Write(WidenedWrite),
    Heap(std::sync::Arc<crate::bindless::Viewed>),
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset (`trim_scratch` gives up what a peak grew).
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// Shadow ranges written through UAVs: the history late views may need,
    /// and ordered events written since the last barrier.
    widened_writes: Mutex<Vec<WidenedWrite>>,
    widened_pending: Mutex<Vec<WidenedEvent>>,
    /// Exact source shadows whose buffers widening commands read. Their COM
    /// objects alone do not keep an arena slice out of the reuse pool.
    widened_sources: Mutex<Vec<std::sync::Arc<crate::shadow::Shadow>>>,
    /// Widened buffers that have consumed every write recorded so far.
    widened_ready: Mutex<std::collections::HashSet<usize>>,
    /// Widened buffers this list has transitioned, by host address.
    widened_states: Mutex<
        std::collections::HashMap<usize, (ID3D12Resource, D3D12_RESOURCE_STATES)>,
    >,
    /// A list of ours (uploads), whose uses are no use of the title's
    /// (`crate::resource::evict`).
    ours: std::sync::atomic::AtomicBool,
    /// The draw or dispatch being recorded runs a program using title
    /// memory ([`HostList::writer_uses_memory`]).
    memory_writer: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The budget this list's indirect executions charge instead of its
    /// own: that of the host list before it, where the title list was cut
    /// only for a run of indirect draws ([`HostList::continue_budget`]).
    /// The buffer, its host address and where it is mapped.
    budget_from: Mutex<Option<(ID3D12Resource, u64, usize)>>,
    /// Clamped indirect arguments (`crate::indirect`), reused by each
    /// execution in turn; replaced by a bigger one when too small.
    arguments: Mutex<Option<ID3D12Resource>>,
    /// Dispatches, buffer copies, clears and immediate writes recorded since
    /// the reset: what may have written indirect arguments (hoisted clamps,
    /// `crate::indirect`).
    writes: std::sync::atomic::AtomicU64,
    /// Title barriers, pipeline syncs, buffer copies, clears and immediate
    /// writes recorded since the reset: what may make an earlier write
    /// readable as indirect arguments (hoisted clamps of indirect
    /// dispatches, `crate::indirect`). Not the title's dispatches: by
    /// D3D12's rules what one writes is not read as arguments before a
    /// barrier.
    orders: std::sync::atomic::AtomicU64,
    /// Clamped arguments of the list's indirect draws that the list before
    /// it writes ([`HostList::clamp_room`]).
    hoisted: Mutex<Hoisted>,
    /// Clamps of the list after it recorded here but not yet run
    /// (`crate::indirect`, batches): recorded where the run of clamps ends.
    pub(crate) clamp_batch: Mutex<Option<crate::indirect::Pending>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's buffers for hoisted clamped arguments, how many are in use
/// since the reset, the bytes taken of the last of those, and whether the
/// list before has left them readable as arguments.
#[derive(Default)]
struct Hoisted {
    all: Vec<ID3D12Resource>,
    used: usize,
    at: u64,
    readable: bool,
}

// SAFETY: D3D12 objects are free-threaded; a host list is recorded by the
// title list's one recording thread, then run by one submission at a time
// (the queues hold it between, `crate::hold`).
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States =
    std::collections::HashMap<usize, (u32, u32, std::sync::Arc<crate::resource::HostTexture>)>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// On D3DMetal, a separate native queue call before a deferred UAV-to-SRV
/// prelude can make the preceding producer's writes visible.
fn split_uav_srv_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on), and for
/// the frame recorder (`VICTORY_RECORD`), whose reports name pipelines and
/// resources by it.
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        return name_it(object, || text);
    }
    name_it(object, text);
}

fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer and memory copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy between overlapping title ranges (destination, source, bytes).
    Move(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a copy into it.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings the current buffer view of a table slot up to date: the title
    /// may have rewritten the slot since the draw was recorded.
    Slot(std::sync::Arc<crate::bindless::Viewed>, u32),
    /// Work that writes host memory only.
    // mac: made only by tests and diagnostics since the rest is `Own`.
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    Host(Box<dyn Fn() + Send>),
    /// mac: work that writes only the list's own memory or our own flags,
    /// and reads no title memory but `[address, address + bytes)` (none if
    /// 0 bytes): run as [`Step::Host`]; a submission prepared while the one
    /// before still runs need not wait for that one first
    /// (`unwaited::flights`), unless that one may write what it reads.
    Own(Box<dyn Fn() + Send>, usize, usize),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then), and
    /// whether their program uses title memory (then the range is withheld
    /// when such a list misses, `crate::memory::withholding`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a title copy intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Pipelines [`HostList::bind_pipeline`] already kept and noted since reset.
    bound_pipelines: std::collections::HashSet<usize>,
    /// Unaligned root-CBV scratch copies (`(source, bytes)` → host GPU address)
    /// valid until the next title copy or list reset.
    filled_scratch: std::collections::HashMap<(usize, usize), u64>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Table slots asked for since the last title write (heap, index).
    slots: std::collections::HashSet<(usize, u32)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The memory values the list waits for before it starts (a list the
    /// title records with a wait after commands is cut there,
    /// `crate::list`).
    waits: Vec<crate::hold::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<Box<dyn Fn() + Send>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
    /// With `VICTORY_COPY_ORDER=split` (else all empty): the buffer ranges
    /// (title addresses) the list's commands read so far, as their
    /// refreshes name them ([`HostList::defer_refresh`]: table views,
    /// vertex and index buffers, root views, indirect arguments).
    reads: crate::ranges::Ranges,
    /// The ranges the title's copies recorded in the list write (address,
    /// bytes).
    copied: Vec<(usize, usize)>,
    /// The list is the part of a title list after a copy that changes what
    /// the part before it reads (`crate::list`): it starts a host
    /// submission ([`copy_order_parts`]).
    after_copy: bool,
    /// Title textures of the kinds copied back (`crate::texback`) that the
    /// list's shaders write, copied when it closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, which land in title memory once the list has run.
    texture_backs: Vec<std::sync::Arc<crate::texback::Back>>,
    /// mac: title memory the list's work for after writes (address, bytes):
    /// textures read back into buffers ([`HostList::note_landing`]).
    lands: Vec<(usize, usize)>,
}

/// Features reported by the D3D12 device used for rendering. A failed
/// feature query leaves its result unknown.
#[derive(Clone, Copy)]
pub struct MeshSupport {
    pub mesh_shaders: Option<bool>,
    pub shader_model_6_6: Option<bool>,
}

impl MeshSupport {
    /// Translated mesh shaders need both features.
    pub fn supported(self) -> bool {
        self.mesh_shaders == Some(true) && self.shader_model_6_6 == Some(true)
    }
}

/// Checks the active host without loading a title. Returns `None` if
/// VIctory could not initialise its D3D12 host.
pub fn host_mesh_support() -> Option<MeshSupport> {
    Some(host()?.mesh_support())
}

impl Host {
    pub(crate) fn mesh_support(&self) -> MeshSupport {
        static MESH: OnceLock<MeshSupport> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let support = unsafe {
                let mesh_shaders = self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                );
                let shader_model = self.device.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of_val(&model) as u32,
                );
                MeshSupport {
                    mesh_shaders: mesh_shaders
                        .ok()
                        .map(|()| options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED),
                    shader_model_6_6: shader_model
                        .ok()
                        .map(|()| model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0),
                }
            };
            tracing::info!(mesh = support.supported(), "host mesh shader support");
            support
        })
    }

    /// Whether the host's waves have at most 32 lanes, 32 among them
    /// (NVIDIA, Intel): `WaveSize(64)` is refused there, so wave64 compute
    /// programs are translated for waves of 32
    /// (`victory_shader::emit::Options::host_wave32`); logged once.
    pub(crate) fn wave32(&self) -> bool {
        static WAVE32: OnceLock<bool> = OnceLock::new();
        *WAVE32.get_or_init(|| {
            let host = victory_shader::emit::host::Host::from_device(self.device.clone());
            match host.wave32() {
                Ok(wave32) => {
                    tracing::info!(wave32, "host waves of 32 lanes at most");
                    wave32
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host wave lanes not known");
                    false
                }
            }
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                widened_writes: Mutex::default(),
                widened_pending: Mutex::default(),
                widened_sources: Mutex::default(),
                widened_ready: Mutex::default(),
                widened_states: Mutex::default(),
                ours: false.into(),
                memory_writer: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                budget_from: Mutex::new(None),
                arguments: Mutex::new(None),
                writes: Default::default(),
                orders: Default::default(),
                hoisted: Mutex::default(),
                clamp_batch: Mutex::new(None),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Waits until the host fence reaches `value`. The event is only a
    /// wake-up: the fence's completed value decides. Under D3DMetal a
    /// submission's work can finish without its event ever being set
    /// (#50: the title's main thread slept for good in this wait while the
    /// queue was idle), so the wait wakes every second to check the fence
    /// itself. A late event left set by an earlier wait cannot end a later
    /// wait early, as the fence is checked after every wake-up too.
    fn wait_for(&self, value: u64) {
        use std::sync::atomic::AtomicU64;
        static MISSED: AtomicU64 = AtomicU64::new(0);
        let mut seconds = 0u32;
        loop {
            // SAFETY: our event and fence.
            let woke = unsafe { WaitForSingleObject(self.event, 1000) };
            // SAFETY: as above.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed >= value {
                if woke != windows::Win32::Foundation::WAIT_OBJECT_0 {
                    let n = MISSED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        tracing::warn!(n, value, completed, "a host submission finished without setting its event; went on after checking the fence");
                    }
                }
                return;
            }
            if woke != windows::Win32::Foundation::WAIT_OBJECT_0 {
                seconds += 1;
                if seconds.is_power_of_two() {
                    tracing::warn!(seconds, value, completed, "still waiting for the host GPU");
                }
                if seconds == 8 {
                    let what = WAITING_ON.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    tracing::error!(%what, value, "host GPU hang: the submission waited on");
                }
            }
        }
    }

    /// The queue submissions go to now.
    fn queue(&self) -> ID3D12CommandQueue {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The list submissions end with ([`Mark`]); `None` if it could not be
    /// made, and then submissions are not checked.
    fn mark(&self) -> Option<&Mark> {
        self.mark
            .get_or_init(|| {
                Mark::new(self)
                    .inspect_err(|e| tracing::warn!(error = %e, "no submission mark: submissions the host GPU drops go unseen"))
                    .ok()
            })
            .as_ref()
    }

    /// After a submission that did not run to its mark: what it drew is
    /// missing. If the queue no longer runs work, a new queue replaces it,
    /// and the next present makes its swap chain on that one. Under the
    /// submission lock.
    fn lost(&self, submitted: &mut u64, mark: &Mark) {
        let lost = LOST[0].fetch_add(1, Ordering::Relaxed) + 1;
        // A lane that held a lock of an emulated 64-bit atomic when the GPU
        // ended its shader never frees it (`victory_shader::emit`, "Atomic
        // locks"): free them all before anything else runs.
        let freed = crate::shadow::free_atomic_locks();
        if freed > 0 {
            tracing::warn!(
                freed,
                "atomic locks left taken by the ended submission freed"
            );
        }
        if self.runs(submitted, mark) {
            if lost.is_power_of_two() {
                tracing::warn!(
                    lost,
                    "the host GPU ended a submission before it finished, and what it drew is missing \
                     (under D3DMetal, macOS ends a command buffer when the GPU locks up, as on a draw \
                     that holds it too long, or takes a page fault)"
                );
            }
            return;
        }
        if let Err(e) = self.replace_queue() {
            fail(format_args!(
                "the host queue stopped running work, and no new queue could be made: {e}"
            ));
        }
        if !self.runs(submitted, mark) {
            fail(format_args!(
                "the host queue stopped running work, and a new queue does not run it either"
            ));
        }
        let queues = LOST[1].fetch_add(1, Ordering::Relaxed) + 1;
        if queues.is_power_of_two() {
            tracing::warn!(
                lost,
                queues,
                "the host queue stopped running work after the GPU ended a submission \
                 (macOS ignores what is submitted to it): moved to a new queue"
            );
        }
    }

    /// Whether the queue runs work: the mark alone, as a submission of its
    /// own. Under the submission lock.
    fn runs(&self, submitted: &mut u64, mark: &Mark) -> bool {
        *submitted += 1;
        mark.arm(*submitted);
        let queue = self.queue();
        // SAFETY: a closed list; the fence and event are ours.
        unsafe {
            queue.ExecuteCommandLists(&[mark.list()]);
            let waited = queue
                .Signal(&self.fence, *submitted)
                .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
            match waited {
                Ok(()) => self.wait_for(*submitted),
                Err(e) => tracing::warn!(error = %e, "host check not waited for"),
            }
        }
        mark.saw(*submitted)
    }

    /// A new direct queue in place of the current one. Under the submission
    /// lock, so no submission is on the old queue.
    fn replace_queue(&self) -> windows::core::Result<()> {
        // SAFETY: plain object creation.
        let queue: ID3D12CommandQueue = unsafe {
            self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }?;
        *self.queue.lock().unwrap_or_else(|e| e.into_inner()) = queue;
        Ok(())
    }

    /// Holds other threads' submissions off until dropped, around the making
    /// of a shader-visible host view heap and its fill with null views
    /// (millions of descriptor copies). Under Wine on D3DMetal, lists other
    /// threads submitted during such a fill lost their views: their
    /// shaders' writes were dropped and their reads gave zeros, whatever
    /// heap the lists bound (VIctory-macOS issue #31). Nothing on a thread
    /// that holds the submission lock already, nor off Wine.
    pub(crate) fn quiet(&self) -> Option<Quiet<'_>> {
        if !crate::under_wine() {
            return None;
        }
        if HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            // mac: with flights the submission left may still run while
            // the next one is prepared under the lock: the GPU first.
            if unwaited::flights::on() {
                self.stand_beside();
            }
            return None;
        }
        let held = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // A submission the host GPU was not waited for has run too.
        self.stand();
        Some(Quiet { _holding: Holding::enter(), _held: held })
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.run(runs, None);
    }

    /// [`Host::execute`]. `title`: the lists are the title's own, held so
    /// (with the queue they are of), and may be left to the host GPU
    /// without waiting for it ([`unwaited`]): whether they were.
    fn run(&self, runs: &[&HostList], title: Option<(usize, &[std::sync::Arc<HostList>])>) -> bool {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        // The one before, if the host GPU was not waited for, is finished
        // first.
        let by = if title.is_some() { unwaited::By::Next } else { unwaited::By::Ours };
        // (The title's lists are next: their buffer copies may be brought
        // up to date while it still runs, `unwaited::Host::prewarm`.)
        // mac: with flights, the title's lists are prepared while it runs,
        // and it is finished at the first step that could see what it owes
        // (`beside`, `unwaited::flights`), at the latest before they go to
        // the GPU.
        let (mut submitted, mut beside) = if title.is_some() && unwaited::flights::on() {
            self.submissions_beside(runs)
        } else {
            (self.submissions(by, if title.is_some() { runs } else { &[] }), None)
        };
        let _holding = Holding::enter();
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ExecuteLists, runs.len() as u64);
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first: the last submission has run.
        if beside.is_some() && crate::resource::evict_due() {
            self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Evict);
        }
        crate::resource::evict_unused();
        // The first shader writing through a heap may run in this
        // submission: unordered views left null are made again first.
        if crate::bindless::take_remake() {
            self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Remake);
            crate::descriptor::remake_unordered();
        }
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        // mac: beside a submission still running only as the rounds ahead
        // sync (`VICTORY_ASYNC_HEAP_SYNC=1`).
        if !crate::descriptor::ahead::on() {
            self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Heaps);
        }
        crate::descriptor::sync_heaps();
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // And textures read through tables, uploaded by the same list.
        let textures: Vec<_> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Texture(staging, copy) => Some((staging.clone(), copy.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // mac: one upload list and one staging pool, reset per submission.
        if (!heaps.is_empty() || !textures.is_empty()) && self.beside_uploads(&beside) {
            self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Uploads);
        }
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty() || !textures.is_empty())
            .then(|| crate::bindless::uploads(&heaps, &textures))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        let runs: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &runs[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(not(feature = "diagnostics"))]
        let traced = false;
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        *submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        let gpu_timing = timing();
        #[cfg(not(feature = "diagnostics"))]
        let gpu_timing = false;
        let split_preludes = !gpu_timing && split_uav_srv_prelude();
        let mut native_boundaries = Vec::new();
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        for list in runs {
            // Its first title copy opens a list the copy-order split cut.
            batch.cut = list.after_copy();
            self.poll_beside(&mut submitted, &mut beside);
            if beside.as_ref().is_some_and(unwaited::flights::Beside::running) {
                let mut gate = |step: &Step| self.beside_step(&mut submitted, &mut beside, step);
                list.run_deferred(&mut batch, Some(&mut gate));
            } else {
                list.run_deferred(&mut batch, None);
            }
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some((prelude, uav_to_srv)) = list.prelude(split_preludes) {
                if uav_to_srv && !lists.is_empty() {
                    native_boundaries.push(lists.len());
                }
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        // mac: with flights, while the submission left still runs, the
        // refreshes of copies it can neither write nor land in are made
        // early; then it is finished, and the others are made.
        if let Some(held) = self.beside_refreshes(&mut beside, &mut batch.refreshes) {
            batch.early = true;
            batch.flush();
            batch.early = false;
            if !held.is_empty() {
                self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Flush);
                batch.refreshes = held;
            }
        }
        batch.flush();
        batch.count_zero_copy();
        #[cfg(feature = "diagnostics")]
        batch.log_refreshes(*submitted + 1, runs.len());
        // mac: one submission on the host GPU at a time.
        self.finish_beside(&mut submitted, &mut beside, unwaited::flights::Why::Give);
        // Time finishing it is not this one's preparing.
        let aside = beside.as_ref().map_or(Duration::ZERO, |b| b.finishing);
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if gpu_timing {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        let mut removed_in = None;
        let mark = self.mark();
        // A failed earlier call may leave the queue able to run the final mark.
        // Give each native call its own mark so failure recovery can see it.
        if mark.is_none() {
            native_boundaries.clear();
        }
        let mut split_pool = (!native_boundaries.is_empty())
            .then(|| self.split_marks.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(pool) = split_pool.as_deref_mut() {
            if let Some(marks) = pool.as_mut()
                && let Err(error) = (marks.len()..native_boundaries.len()).try_for_each(|_| {
                    marks.push(Mark::new(self)?);
                    Ok::<_, windows::core::Error>(())
                })
            {
                tracing::warn!(
                    %error,
                    "could not make split submission marks, keeping the native batch together"
                );
                *pool = None;
            }
            if pool.is_none() {
                native_boundaries.clear();
            }
        }
        let split_marks = split_pool
            .as_deref()
            .and_then(Option::as_deref)
            .unwrap_or(&[]);
        if !native_boundaries.is_empty() {
            static FIRST_SPLIT: std::sync::Once = std::sync::Once::new();
            FIRST_SPLIT.call_once(|| {
                tracing::info!(
                    native_calls = native_boundaries.len() + 1,
                    "split host submission at deferred UAV-to-SRV preludes"
                );
            });
        }
        // What a hang would be waiting on (`Host::wait_for` names it after 8 s).
        *WAITING_ON.lock().unwrap_or_else(|e| e.into_inner()) = {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| {
                    l.pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            format!("{} lists, pipelines {pipelines:?}", runs.len())
        };
        // The title's own lists, as one group: given to the host GPU and
        // left there, to be finished when something waits for them.
        // (Not a submission cut at a deferred UAV-to-SRV prelude,
        // `VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE`: its native calls and their
        // marks are the waited path's below.)
        if let Some(title) = title
            && native_boundaries.is_empty()
            && unwaited::may(groups.len(), traced)
        {
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(aside).as_micros() as u64, Ordering::Relaxed);
            let ran = Ran { gpu: Duration::ZERO, removed_in, batch, heaps, traced };
            self.leave(&mut submitted, groups[0], mark, title, ran, uploads.is_some());
            return true;
        }
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
            // Last, the list that shows the submission ran.
            let mut group = group.to_vec();
            if let Some(mark) = mark {
                mark.arm(*submitted);
                group.push(mark.list());
            }
            let queue = self.queue();
            // SAFETY: closed lists; the fence and event are ours.
            let waited = unsafe {
                // All native calls share the existing fence signal and wait.
                let mut first = 0;
                for (&end, mark) in native_boundaries.iter().zip(split_marks) {
                    mark.arm(*submitted);
                    let mut marked = group[first..end].to_vec();
                    marked.push(mark.list());
                    #[cfg(test)]
                    tests::native_call(marked.len());
                    queue.ExecuteCommandLists(&marked);
                    first = end;
                }
                #[cfg(test)]
                tests::native_call(group.len() - first);
                queue.ExecuteCommandLists(&group[first..]);
                let waited = queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        self.wait_for(*submitted);
                        true
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "host submission not waited for");
                        false
                    }
                }
            };
            #[cfg(test)]
            tests::native_call_group(waited);
            if waited
                && let Some(mark) = mark
                && (!mark.saw(*submitted)
                    || split_marks.iter().take(native_boundaries.len()).any(|m| !m.saw(*submitted)))
                // SAFETY: a plain query.
                && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
            {
                self.lost(&mut submitted, mark);
            }
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 {
                // SAFETY: a plain query.
                let frequency = unsafe { queue.GetTimestampFrequency() }.unwrap_or(1);
                gpu_times(started.elapsed(), &runs[i].slowest(frequency, usize::MAX).unwrap_or_default());
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i]
                    .notes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .len();
                // macOS port: GPU timestamps read 0 under D3DMetal, so the list's own
                // pipelines are the only way to tell which work a slow list did.
                let pipelines: std::collections::BTreeSet<String> = runs[i]
                    .pipelines
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                    .into_iter()
                    .collect();
                tracing::warn!(
                    ms = started.elapsed().as_millis(),
                    commands,
                    ?pipelines,
                    "slow host list"
                );
                match runs[i].slowest(frequency, 6) {
                    Some(slowest) => {
                        for (ms, op) in slowest {
                            tracing::warn!("slow list command, {ms:.1} ms: {op}");
                        }
                    }
                    None => {
                        // mac: without timestamps, at least what the slow
                        // list holds: a list that binds no title pipeline
                        // (copies, clears, passes of ours) is blank above.
                        tracing::warn!("slow list commands by kind: {}", runs[i].kinds(12));
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            tracing::warn!(
                                frequency,
                                "GPU timestamps do not advance on this adapter: \
                                 slow lists are timed whole, their commands are not"
                            )
                        });
                    }
                }
            }
            // SAFETY: a plain query.
            if groups.len() > 1 && removed_in.is_none() && unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                removed_in = Some(i);
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(aside).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        self.finish(Some(submitted), runs, Ran { gpu, removed_in, batch, heaps, traced });
        false
    }

    /// After a submission has run: the device is checked, what its shaders
    /// wrote goes back into title memory, then its lists' work for after
    /// and their bottom-of-pipe writes. `submitted`: the submission lock,
    /// given up before a listed frame's dumps (diagnostics); none for a
    /// submission finished later ([`unwaited`]), whose finisher keeps it.
    fn finish(&self, submitted: Option<std::sync::MutexGuard<'_, u64>>, runs: &[&HostList], ran: Ran) {
        let Ran { gpu, removed_in, batch, heaps, traced } = ran;
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| {
                    l.pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            tracing::warn!(
                ms = gpu.as_millis(),
                lists = runs.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, removed_in)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        // Shaders that read memory with no copy computed something else than
        // the console's: it stays out of title memory, where it broke the
        // title's own data (runs c1, c4), and so does what every program
        // using title memory wrote while any misses
        // (`crate::memory::withholding`). Only lists with such programs have
        // checks; all are asked before anything goes back. Other programs in
        // those lists write through bounded views from their own inputs:
        // their ranges go back (the imposter transforms beside the BVH
        // builder in one list, run light13).
        let missed: Vec<bool> = runs
            .iter()
            .map(|list| list.deferred().missed.iter().any(|check| check()))
            .collect();
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        for (list, &missed) in runs.iter().zip(&missed) {
            let deferred = list.deferred();
            let missed = missed || withholding && !deferred.missed.is_empty();
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _, memory) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    // A range both kinds declare is withheld whole.
                    let goes_back = !(missed && *memory);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= goes_back,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, goes_back));
                        }
                    }
                }
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held.iter().any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        crate::shadow::copy_back_marks(&withheld, withholding);
        // Texture views left null that shaders read: made for what follows.
        let read = crate::shadow::take_reads();
        if !read.is_empty() {
            crate::descriptor::make_read(&read);
        }
        // Lanes that gave up waiting for a 64-bit atomic's lock: their
        // updates may be lost (`victory_shader::emit`, "Atomic locks").
        let gave_up = crate::shadow::take_gave_up();
        if gave_up > 0 {
            let before = GAVE_UP.fetch_add(gave_up, Ordering::Relaxed);
            let total = before + gave_up;
            if before == 0 || before.ilog2() != total.ilog2() {
                tracing::warn!(
                    gave_up,
                    total,
                    "lanes gave up waiting for a 64-bit atomic's lock and went on without it, so their updates may be lost (#240)"
                );
            }
        }
        let total_back_bytes: usize = back
            .iter()
            .filter(|b| b.3)
            .map(|b| b.2)
            .sum();
        if crate::shadow::parallel_copies_enabled()
            && total_back_bytes >= crate::shadow::PARALLEL_COPY_BACK_MIN_BYTES
        {
            type BackGroup = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize, bool)>);
            let mut groups: Vec<BackGroup> = Vec::new();
            for (shadow, address, len, goes_back) in &back {
                match groups
                    .iter_mut()
                    .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
                {
                    Some((_, items)) => items.push((*address, *len, *goes_back)),
                    None => groups.push((shadow.clone(), vec![(*address, *len, *goes_back)])),
                }
            }
            let shadow_ranges: Vec<(u64, u64)> = groups.iter().map(|(s, _)| s.range()).collect();
            if groups.len() >= 2 && crate::shadow::copy_back_ranges_disjoint(&shadow_ranges) {
                crate::texture::in_parallel(groups.len(), |i| {
                    let (shadow, items) = &groups[i];
                    for &(address, len, goes_back) in items {
                        if goes_back {
                            let written = shadow.copy_back(address, len);
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        } else {
                            shadow.keep_writes(address, len);
                            crate::memory::note_withheld(len as u64);
                            crate::memory::drop_withheld(address, len);
                        }
                    }
                });
            } else {
                for (shadow, address, len, goes_back) in &back {
                    let (address, len) = (*address, *len);
                    if *goes_back {
                        let written = shadow.copy_back(address, len);
                        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                    } else {
                        shadow.keep_writes(address, len);
                        crate::memory::note_withheld(len as u64);
                        crate::memory::drop_withheld(address, len);
                    }
                }
            }
        } else {
            for (shadow, address, len, goes_back) in &back {
                let (address, len) = (*address, *len);
                if *goes_back {
                    let written = shadow.copy_back(address, len);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                } else {
                    shadow.keep_writes(address, len);
                    crate::memory::note_withheld(len as u64);
                    crate::memory::drop_withheld(address, len);
                }
            }
        }
        // Textures shaders wrote that the title reads on the CPU
        // (`crate::texback`): the lists that copied them have run.
        if crate::texback::on() {
            let backs: Vec<_> = runs.iter().flat_map(|l| l.deferred().texture_backs.clone()).collect();
            crate::texback::land_all(&backs);
        }
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        // The submission has run and the next waits for our lock. Host
        // copies the title gave up meanwhile may have their last holder here.
        release_retired();
        release_later(Box::new((batch, back, withheld, heaps)));
        self.log_messages();
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced && let Some(submitted) = submitted {
            let sub = *submitted;
            drop(submitted);
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(sub, &targets);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (submitted, traced);
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC {
                                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                Count: STAMPS,
                                NodeMask: 0,
                            },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The list's noted commands by kind (and pipeline, for draws,
    /// dispatches and passes of ours), the `n` most frequent first, as
    /// "count kind" (diagnostics; the commands are noted with
    /// `VICTORY_D3D12_DRED` on).
    #[cfg(feature = "diagnostics")]
    fn kinds(&self, n: usize) -> String {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut by: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for (note, _) in notes.iter() {
            let head = note.split(" | ").next().unwrap_or_default();
            let kind = head.split(' ').next().unwrap_or_default();
            let pipeline = ["[compute ", "[vertex ", "[mesh ", "[ambient "].iter().find_map(|p| head.find(p));
            let key = match pipeline {
                Some(at) => format!("{kind} {}", head[at..].chars().take(96).collect::<String>()),
                None => kind.to_owned(),
            };
            *by.entry(key).or_default() += 1;
        }
        if by.is_empty() {
            return "none noted (VICTORY_D3D12_DRED=1 notes them)".to_owned();
        }
        let mut by: Vec<(String, u32)> = by.into_iter().collect();
        by.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut text = by.iter().take(n).map(|(kind, count)| format!("{count} {kind}")).collect::<Vec<_>>().join(", ");
        if by.len() > n {
            text.push_str(&format!(", and {} other kinds", by.len() - n));
        }
        text
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran);
    /// `None` when its timestamps did not advance (D3DMetal reads them all
    /// as 0, at a frequency of 60).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Option<Vec<(f64, String)>> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Some(Vec::new());
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (count + 1) * 8,
        };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Some(Vec::new());
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        if count > 0 && stamps.windows(2).all(|w| w[0] == w[1]) {
            return None;
        }
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (
                    ticks as f64 * 1e3 / frequency.max(1) as f64,
                    notes[i].0.chars().take(300).collect(),
                )
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        Some(times)
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        crate::indirect::flush(self);
        self.clamp_done();
        self.textures_back();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count =
                (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list
                    .ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        self.widen_close();
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_writes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_sources.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.trim_scratch();
        self.writes.store(0, Ordering::Relaxed);
        self.orders.store(0, Ordering::Relaxed);
        self.own_budget();
        {
            let mut hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
            (hoisted.used, hoisted.at, hoisted.readable) = (0, 0, false);
        }
        *self.clamp_batch.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(ID3D12Resource, u32)> {
        // One host texture for the whole move (`crate::resource::evict` may
        // give it up meanwhile; this list then holds it).
        let copy = crate::resource::host_copy(resource)?;
        // Unordered access through the twin skips `track` below.
        copy.note_state(wanted);
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = copy.legal_state(wanted);
        // Cloned (two calls into the host) only for a barrier.
        self.track(&copy, wanted)
            .map(|before| (copy.texture.clone(), before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        copy.note_state(wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                if !self.ours.load(Ordering::Relaxed) {
                    copy.stamp();
                }
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = copy.legal_state(wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// (`crate::texback`), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource)
            || self.deferred().back_wanted.iter().any(|t| t.owner() == resource as usize)
        {
            return;
        }
        if let Some(target) = crate::texback::Target::new(resource) {
            self.deferred().back_wanted.push(target);
        }
    }

    /// As the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers (`crate::texback::record`).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            if target.valid()
                && let Some(barrier) = self.move_texture(&target.copy, COPY_SOURCE)
            {
                // SAFETY: an open list; the barrier names a live texture.
                unsafe { self.list.ResourceBarrier(&[barrier]) };
            }
            if let Some(back) = crate::texback::record(self, target) {
                self.deferred().texture_backs.push(std::sync::Arc::new(back));
            }
        }
    }

    /// The list is one of ours (uploads): what it uses is no use of the
    /// title's.
    pub(crate) fn set_ours(&self) {
        self.ours.store(true, Ordering::Relaxed);
    }

    /// What the list's shaders write from now until the guard goes is
    /// written by a program using title memory (`uses`) or not.
    pub(crate) fn writer_uses_memory(&self, uses: bool) -> impl Drop + '_ {
        struct Reset<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        self.memory_writer.store(uses, Ordering::Relaxed);
        Reset(&self.memory_writer)
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// A widened SRV is about to be read by this list. A view first made after
    /// an earlier barrier also needs the writes retained since this list began.
    pub(crate) fn mark_widened_read(&self, buffer: &ID3D12Resource) {
        let key = buffer.as_raw() as usize;
        let ready = self
            .widened_ready
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&key);
        if !ready {
            self.sync_widened();
        }
        // Reads need no transition: a cached view is either still common or
        // was left shader-readable by its last widening copy.
    }

    /// A UAV wrote a shadow range that may feed a widened SRV after a later
    /// buffer barrier. Keep writes even before the first widened view exists.
    pub(crate) fn mark_widened_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        if !crate::under_wine() || len == 0 {
            return;
        }
        self.widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(WidenedEvent::Write((shadow.clone(), address, len)));
    }

    /// A heap-indexing shader may write through any of this heap's current
    /// UAV views. Its ranges are read at the barrier, not while binding.
    pub(crate) fn mark_widened_heap(&self, heap: &std::sync::Arc<crate::bindless::Viewed>) {
        if !crate::under_wine() {
            return;
        }
        self.widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(WidenedEvent::Heap(heap.clone()));
    }

    /// Keeps an exact source shadow's arena slice alive until the list resets.
    pub(crate) fn keep_shadow(&self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let mut sources = self.widened_sources.lock().unwrap_or_else(|e| e.into_inner());
        if !sources.iter().any(|other| std::sync::Arc::ptr_eq(shadow, other)) {
            sources.push(shadow.clone());
        }
    }

    /// After the title's buffer/UAV barrier, widen the shadow ranges written
    /// before it so later SRVs in this submission see the UAV output.
    pub(crate) fn sync_widened(&self) {
        let mut pending = self.widened_pending.lock().unwrap_or_else(|e| e.into_inner());
        let events = std::mem::take(&mut *pending);
        let mut expanded = Vec::new();
        // mac: with `VICTORY_CENSUS_FAST=1` a heap's ranges once a barrier,
        // where its last draw stood (`crate::census_fast::last_of_each`):
        // every draw of a heap before the barrier stands for the same ranges,
        // and of equal writes only the last decides what a widened view gets.
        let last = crate::mac_switches::census_fast().then(|| {
            crate::census_fast::last_of_each(events.iter().map(|event| match event {
                WidenedEvent::Heap(heap) => Some(std::sync::Arc::as_ptr(heap) as usize),
                WidenedEvent::Write(_) => None,
            }))
        });
        for (k, event) in events.into_iter().enumerate() {
            match event {
                WidenedEvent::Write(write) => expanded.push(WidenedEvent::Write(write)),
                WidenedEvent::Heap(_) if last.as_ref().is_some_and(|last| !last[k]) => {}
                WidenedEvent::Heap(heap) => {
                    // mac: the kept answer with `VICTORY_CENSUS_FAST=1`.
                    if let Some(writes) = crate::bindless::unordered_writes(&heap) {
                        expanded.extend(writes.iter().cloned().map(WidenedEvent::Write));
                        continue;
                    }
                    expanded.extend(
                        crate::bindless::widened_writes(&heap)
                            .into_iter()
                            .map(WidenedEvent::Write),
                    );
                }
            }
        }
        *pending = expanded;
        let sampled: Vec<_> = pending
            .iter()
            .map(|event| match event {
                WidenedEvent::Write(write) => write.clone(),
                WidenedEvent::Heap(_) => unreachable!("heap events expanded above"),
            })
            .collect();
        if !crate::shadow::WIDENED_ANY.load(Ordering::Acquire) {
            self.widened_writes.lock().unwrap_or_else(|e| e.into_inner()).extend(sampled);
            pending.clear();
            return;
        }
        let ready = self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if sampled.is_empty()
            && crate::shadow::widened_all().iter().all(|view| {
                ready.contains(&(view.buffer.as_raw() as usize))
            })
        {
            return;
        }
        let history = self.widened_writes.lock().unwrap_or_else(|e| e.into_inner());
        let synced = history.len();
        let mut writes = history.clone();
        writes.extend(sampled.iter().cloned());
        drop(history);
        if writes.is_empty() {
            return;
        }
        let Some(views) = crate::widened::sync(self, &writes, synced, &ready) else {
            return;
        };
        self.widened_writes.lock().unwrap_or_else(|e| e.into_inner()).extend(sampled);
        pending.clear();
        self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).extend(views);
    }

    /// Moves widened buffers into the unordered state for widening copies.
    pub(crate) fn widen_begin_all(&self, buffers: &[ID3D12Resource]) {
        for buffer in buffers {
            self.widen_state(buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    /// Widening copies are complete; later shaders read these buffers.
    pub(crate) fn widen_end_all(&self, buffers: &[ID3D12Resource]) {
        for buffer in buffers {
            self.widen_state(buffer, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }
    }

    fn widen_state(&self, buffer: &ID3D12Resource, wanted: D3D12_RESOURCE_STATES) {
        let key = buffer.as_raw() as usize;
        let mut states = self.widened_states.lock().unwrap_or_else(|e| e.into_inner());
        let before = states.get(&key).map_or(D3D12_RESOURCE_STATE_COMMON, |(_, state)| *state);
        if before == wanted {
            return;
        }
        states.insert(key, (buffer.clone(), wanted));
        drop(states);
        self.keep(buffer);
        // SAFETY: an open list and a live widened buffer.
        unsafe {
            self.list.ResourceBarrier(&[transition(buffer, before, wanted)]);
        }
    }

    /// Returns widened buffers to common so the CPU can refresh them before
    /// the next submission.
    fn widen_close(&self) {
        self.widened_writes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let states = std::mem::take(&mut *self.widened_states.lock().unwrap_or_else(|e| e.into_inner()));
        for (_, (buffer, before)) in states {
            if before != D3D12_RESOURCE_STATE_COMMON {
                // SAFETY: an open list and a live widened buffer.
                unsafe {
                    self.list.ResourceBarrier(&[transition(&buffer, before, D3D12_RESOURCE_STATE_COMMON)]);
                }
            }
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self, detect_uav_to_srv: bool) -> Option<(ID3D12CommandList, bool)> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies start from the real states, then the list's moves from
        // them; only after both do they become the states the list leaves.
        let mut uav_to_srv = false;
        let mut moves = || -> Vec<D3D12_RESOURCE_BARRIER> {
            states
                .values()
                .filter_map(|(first, _, copy)| {
                    let real = copy.state.load(Ordering::Relaxed);
                    (real != *first).then(|| {
                        uav_to_srv |= detect_uav_to_srv
                            && real == D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32
                            && *first & D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32 != 0;
                        transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(*first as i32))
                    })
                })
                .collect()
        };
        let left = || {
            for (_, last, copy) in states.values() {
                copy.state.store(*last, Ordering::Relaxed);
            }
        };
        if copies.is_empty() && states.values().all(|(first, _, copy)| copy.state.load(Ordering::Relaxed) == *first) {
            left();
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = t.legal_state(wanted);
                        (real != wanted).then(|| {
                            let there = transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32));
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            there
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                let barriers = moves();
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        left();
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok().map(|list| (list, uav_to_srv))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps()
                && at < STAMPS - 1
            {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list
                    .SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// Keeps `pso` and `root` alive until the list is reset and records `name`,
    /// deduplicated per pipeline pointer until reset.
    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound_pipelines.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.defer_title_copy(Step::Copy(dst as usize, src as usize, len));
    }

    /// Moves bytes between overlapping title ranges at submission.
    pub(crate) fn defer_move(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.defer_title_copy(Step::Move(dst as usize, src as usize, len));
    }

    fn defer_title_copy(&self, step: Step) {
        self.count_write();
        let mut deferred = self.deferred();
        deferred.copies.push(step);
        deferred.refreshed.clear();
        deferred.slots.clear();
        deferred.filled_scratch.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Host GPU address of scratch memory holding `len` bytes from `src` as of
    /// submission, reused across draws until a title copy or list reset.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        if let Some(&gpu) = self.deferred().filled_scratch.get(&key) {
            return Some(gpu);
        }
        let (gpu, to) = self.scratch(len)?;
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Fill(to as usize, src as usize, len as usize));
        deferred.filled_scratch.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            // What the commands read, for the order of title copies.
            if crate::mac_switches::copy_order_split().is_some_and(|longest| len <= longest) {
                deferred.reads.insert(address, address.saturating_add(len), 0);
            }
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Whether a title copy into `[dst, dst + len)` recorded now changes
    /// what a command recorded in this list reads (always false without
    /// `VICTORY_COPY_ORDER=split`): the title list is then cut before it
    /// (`crate::list`).
    pub(crate) fn read_before(&self, dst: usize, len: usize) -> bool {
        self.deferred().reads.overlaps(dst, dst.saturating_add(len))
    }

    /// The title records a copy into `[dst, dst + len)` in this list
    /// (noted with `VICTORY_COPY_ORDER=split`, for [`copy_order_parts`]).
    pub(crate) fn title_copies(&self, dst: usize, len: usize) {
        if len > 0 && crate::mac_switches::copy_order_split().is_some() {
            self.deferred().copied.push((dst, len));
        }
    }

    /// The list goes on a title list after a copy that changes what the
    /// part before it reads.
    pub(crate) fn starts_after_copy(&self) {
        self.deferred().after_copy = true;
    }

    /// Whether [`HostList::starts_after_copy`] was called since the reset
    /// (also counted for `VICTORY_ALIAS_BUFFERS`'s frame line).
    pub(crate) fn after_copy(&self) -> bool {
        self.deferred().after_copy
    }

    /// Brings the buffer range of the view currently in `index` up to date
    /// when the list is submitted, once until a title write intervenes.
    ///
    /// Measuring: one step per slot a table draw reads (the set above keeps
    /// repeats out), timed with the refreshes in `StepRefreshUs`, and a slot
    /// whose buffer is clean adds nothing to the batch.
    pub(crate) fn defer_slot(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>, index: u32) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, index);
        if deferred.slots.insert(key) {
            deferred.copies.push(Step::Slot(viewed.clone(), index));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let memory = self.memory_writer.load(Ordering::Relaxed);
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked, memory));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(Box::new(work));
    }

    /// mac: work for after the list (a texture read back into a buffer)
    /// writes title memory `[at, at + len)`: what a submission prepared
    /// while the list still runs must not read before it is there
    /// (`unwaited::flights`).
    pub(crate) fn note_landing(&self, at: usize, len: usize) {
        if len > 0 {
            self.deferred().lands.push((at, len));
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.count_write();
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory value the list waits for before it starts.
    pub(crate) fn defer_wait(&self, wait: crate::hold::ValueWait) {
        self.deferred().waits.push(wait);
    }

    /// The memory values the list waits for before it starts.
    pub(crate) fn waits(&self) -> Vec<crate::hold::ValueWait> {
        self.deferred().waits.clone()
    }

    /// Whether nothing has been recorded since the list was made or reset
    /// that a wait recorded now would have to follow: no work, texture
    /// use, bottom-of-pipe write or pipeline (state settings alone carry
    /// over).
    pub(crate) fn is_empty(&self) -> bool {
        let d = self.deferred();
        let deferred_empty = d.copies.is_empty() && d.uploads.is_empty() && d.writes.is_empty() && d.after.is_empty() && d.missed.is_empty();
        drop(d);
        deferred_empty
            && self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// `size` bytes of mapped upload memory for a texture to copy from,
    /// alive until the list is reset: the buffer, the offset in it (a
    /// multiple of 512, as texture copies need) and the CPU pointer. It is
    /// the list's scratch, not a buffer made for each copy (fresh upload
    /// buffers cost ~5 ms per 12.7 MB, run textures8 in `crate::texture`),
    /// except for a copy larger than a chunk.
    pub(crate) fn upload(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let counts = &crate::list::FOOTPRINT_UPLOADS;
            counts[0].fetch_add(1, Relaxed);
            counts[1].fetch_add(size, Relaxed);
            let own = size.max(1).next_multiple_of(256) > SCRATCH_CHUNK;
            counts[2].fetch_add(u64::from(own), Relaxed);
        }
        let (buffer, at, _, data) = self.scratch_in(size, 512)?;
        Some((buffer, at, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.mapped_buffer(size, true)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (_, _, gpu, data) = self.scratch_in(len, 256)?;
        Some((gpu, data))
    }

    /// `len` bytes of upload memory at a multiple of `align` (a power of
    /// two) until the list is reset: the buffer, the offset in it, the host
    /// GPU address and the CPU pointer.
    fn scratch_in(&self, len: u64, align: u64) -> Option<(ID3D12Resource, u64, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer.clone());
            return Some((buffer, 0, gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used.next_multiple_of(align).min(chunk.capacity) {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used.next_multiple_of(align);
        let data = (chunk.mapped + at as usize) as *mut u8;
        let result = (chunk.buffer.clone(), at, chunk.gpu + at, data);
        s.used = at + len;
        Some(result)
    }

    /// Starts the next cycle of scratch at the first chunk and gives up
    /// the chunks past the one after the last cycle's last (only after the
    /// list's last submission finished, when the GPU no longer reads them).
    /// A list so keeps what its last cycle needed, and a cycle a little
    /// larger does not allocate again, but a rare peak is not kept for good.
    fn trim_scratch(&self) {
        let spent = {
            let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            let kept = (s.current + 2).min(s.chunks.len());
            let spent = s.chunks.split_off(kept);
            (s.current, s.used) = (0, 0);
            spent
        };
        // Each release is a kernel call, as in [`release_later`].
        if !spent.is_empty() {
            release_later(Box::new(spent));
        }
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        if let Some((_, gpu, _)) = self.budget_from.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            return Some(*gpu);
        }
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Own(
                Box::new(move || unsafe { (at as *mut u32).write_volatile(0) }),
                0,
                0,
            ));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// From here on this list's indirect executions charge the budget
    /// `before` charges, and do not zero it: `before` is the host list just
    /// before it in a title list that was cut between them only for a run
    /// of indirect draws (`crate::list`, hoisted clamps). The executions of
    /// both then share one budget, charged in the title's order, as they do
    /// in the one host list they are recorded in without that cut; so the
    /// records past the budget are the same ones with and without it.
    /// (A host submission's zeroing is done before its first list runs, and
    /// the two run in order: in one submission, or `before` in an earlier
    /// one.) The buffer lives as long as either list names it.
    pub(crate) fn continue_budget(&self, before: &HostList) {
        let Some(gpu) = before.budget() else { return };
        let theirs = before.budget_from.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let from = theirs.or_else(|| {
            let (buffer, data) = before.budget.get()?.as_ref()?;
            Some((buffer.clone(), gpu, *data))
        });
        *self.budget_from.lock().unwrap_or_else(|e| e.into_inner()) = from;
    }

    /// The list's indirect executions charge its own budget (again).
    pub(crate) fn own_budget(&self) {
        *self.budget_from.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The work charged so far to the budget the list's indirect executions
    /// charge, as its word is now (read when the list is not running).
    #[cfg(test)]
    pub(crate) fn budget_spent(&self) -> u32 {
        let from = self.budget_from.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|b| b.2);
        let at = from.or_else(|| self.budget.get().and_then(|b| b.as_ref()).map(|b| b.1)).unwrap_or(0);
        // SAFETY: a mapped word of a live buffer.
        if at == 0 { 0 } else { unsafe { (at as *const u32).read_volatile() } }
    }

    /// A command of this list reads `[address, address + len)` of title
    /// buffers through the host list before it (the hoisted clamp of an
    /// indirect draw, whose refresh that list holds): noted like this
    /// list's own reads ([`HostList::defer_refresh`]), so that a title copy
    /// into them recorded here keeps its order with the draw
    /// (`VICTORY_COPY_ORDER=split`).
    pub(crate) fn note_hoisted_read(&self, address: usize, len: usize) {
        if crate::mac_switches::copy_order_split().is_some_and(|longest| len <= longest) {
            self.deferred().reads.insert(address, address.saturating_add(len), 0);
        }
    }

    /// The list's argument buffer, at least `bytes` long, common between
    /// its uses (`crate::indirect`).
    pub(crate) fn arguments(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live buffer.
        if let Some(b) = arguments.as_ref().filter(|b| unsafe { b.GetDesc() }.Width >= bytes) {
            return Some(b.clone());
        }
        let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
        name(&made, || "indirect arguments".into());
        // Commands recorded earlier may still name the smaller one.
        if let Some(old) = arguments.replace(made.clone()) {
            self.keep(&old);
        }
        Some(made)
    }

    /// Counts a buffer copy, clear, immediate write or clamp in the list:
    /// what may write indirect arguments, and make them readable as such
    /// at once.
    pub(crate) fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.orders.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a dispatch of the title's: it may write indirect arguments,
    /// which are read as such only after a barrier ([`HostList::orders`]).
    pub(crate) fn count_dispatch(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a barrier or pipeline sync of the title's.
    pub(crate) fn count_order(&self) {
        self.orders.fetch_add(1, Ordering::Relaxed);
    }

    /// Dispatches, buffer copies, clears and immediate writes since the
    /// reset.
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Barriers, pipeline syncs, buffer copies, clears and immediate writes
    /// since the reset: [`HostList::count_write`], [`HostList::count_order`].
    pub(crate) fn orders(&self) -> u64 {
        self.orders.load(Ordering::Relaxed)
    }

    /// Room for `bytes` of clamped arguments of one of this list's indirect
    /// draws, which `clamps`, the list before it, writes: the buffer and the
    /// offset. The buffers are this list's own: common between submissions,
    /// moved to unordered access in `clamps` at their first use, to indirect
    /// argument where it ends ([`HostList::clamp_end`]) and back to common
    /// where this list ends, so their states are known whatever runs
    /// between. None for more than a buffer may hold, or past the buffers a
    /// list may have: that draw is clamped in place.
    pub(crate) fn clamp_room(&self, clamps: &HostList, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        const MOST: u64 = 1 << 20;
        const BUFFERS: usize = 32;
        let bytes = bytes.next_multiple_of(16);
        let mut hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        if bytes > MOST || hoisted.readable {
            return None;
        }
        // SAFETY: a plain query on a live buffer.
        let width = |b: &ID3D12Resource| unsafe { b.GetDesc() }.Width;
        if let Some(last) = hoisted.used.checked_sub(1).map(|i| hoisted.all[i].clone())
            && hoisted.at + bytes <= width(&last)
        {
            let at = hoisted.at;
            hoisted.at += bytes;
            return Some((last, at));
        }
        let next = hoisted.used;
        if next == BUFFERS {
            return None;
        }
        if hoisted.all.get(next).is_none_or(|b| width(b) < bytes) {
            let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
            name(&made, || "hoisted indirect arguments".into());
            // Nothing recorded since the reset names the smaller one.
            hoisted.all.truncate(next);
            hoisted.all.push(made);
        }
        let buffer = hoisted.all[next].clone();
        hoisted.used += 1;
        hoisted.at = bytes;
        clamps.keep(&buffer);
        // SAFETY: an open list and a live buffer.
        unsafe {
            clamps.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )])
        };
        Some((buffer, 0))
    }

    /// Where `clamps` ends: the clamped arguments it wrote for this list
    /// become readable as indirect arguments.
    pub(crate) fn clamp_end(&self, clamps: &HostList) {
        // The batch of clamps first: then its writes are made readable.
        crate::indirect::flush(clamps);
        let mut hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        if hoisted.used == 0 || hoisted.readable {
            return;
        }
        hoisted.readable = true;
        let barriers: Vec<_> = hoisted.all[..hoisted.used]
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT))
            .collect();
        // SAFETY: an open list and live buffers.
        unsafe { clamps.list.ResourceBarrier(&barriers) };
    }

    /// Where this list ends: its hoisted arguments back to common.
    fn clamp_done(&self) {
        let hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        if !hoisted.readable {
            return;
        }
        let barriers: Vec<_> = hoisted.all[..hoisted.used]
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON))
            .collect();
        // SAFETY: an open list and live buffers.
        unsafe { self.list.ResourceBarrier(&barriers) };
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` on host memory at submission, in order with the copies.
    // mac: used only by tests and diagnostics since the rest is `defer_own`.
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// mac: [`HostList::defer_host`] for work that writes only the list's
    /// own memory or our own flags and reads no title memory but
    /// `[at, at + len)` (none if `len` is 0) ([`Step::Own`]).
    pub(crate) fn defer_own(&self, at: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Own(Box::new(work), at, len));
    }

    /// Brings a texture's staging buffer and host texture `copy` up to date
    /// when the list is submitted (`crate::bindless::uploads`), keeping
    /// both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred().copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    /// `gate`: called before each step (mac: `unwaited::flights`, which
    /// may finish the submission before first).
    fn run_deferred(&self, batch: &mut Batch, mut gate: Option<&mut dyn FnMut(&Step)>) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            if let Some(gate) = gate.as_mut() {
                gate(step);
            }
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) | Step::Slot(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Move(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Host(_) | Step::Own(..) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) | &Step::Move(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    // An experiment may leave the copy out (diagnostics).
                    #[cfg(feature = "diagnostics")]
                    if batch.rewritten(dst, src, len) {
                        continue;
                    }
                    batch.title_copied(dst, len);
                    crate::alias::title_copy(dst, len, std::mem::take(&mut batch.cut));
                    if matches!(step, Step::Move(..)) {
                        move_mapped(dst as *mut u8, src as *const u8, len);
                    } else {
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    }
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    #[cfg(feature = "diagnostics")]
                    batch.reads.read(*address, *len);
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Slot(viewed, index) => {
                    #[cfg(feature = "diagnostics")]
                    if let Some((address, len)) = crate::bindless::slot_range(viewed, *index) {
                        batch.reads.read(address, len);
                    }
                    crate::bindless::refresh_slot(viewed, *index, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                Step::Host(work) | Step::Own(work, ..) => {
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked {
                        batch.flush();
                    }
                    shadow.writing(*address, *len, *marked);
                    continue;
                }
                // Brought up to date before the submission's lists were made.
                Step::Texture(..) => continue,
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                &Step::Memory(heap) => {
                    crate::memory::prepare(heap, |s, a, l| batch.add(s, a, l));
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// Pieces split so that a title copy keeps its order with the commands
/// before it, since the last frame line ([`take_copy_splits`]).
static COPY_SPLITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The host submissions added by [`copy_order_parts`] since the last call.
pub(crate) fn take_copy_splits() -> u64 {
    COPY_SPLITS.swap(0, Ordering::Relaxed)
}

/// A queue's piece (`lists`, in the order they run) as the host
/// submissions it is run in: one, unless `VICTORY_COPY_ORDER=split` and a
/// title copy recorded in one of the lists changes a buffer range that a
/// command of an earlier list reads, or a list is the part of a title list
/// cut at such a copy (`crate::ranges::copy_order_splits`). The parts are
/// run one after the other: each is prepared (its copies made, its buffer
/// copies brought up to date) only when the one before has run on the host
/// GPU and been finished, also when the title's submissions are not waited
/// for ([`unwaited`], [`Host::submissions`]).
///
/// Seen: the ranges [`HostList::defer_refresh`] is given (constant buffers
/// and buffer views in tables, vertex and index buffers, root views,
/// indirect arguments), whole 64 KiB pages for unordered views that mark
/// their writes. Not seen, and able to have the same fault: what shaders
/// read through the view heap itself or through title memory's page table
/// (`Step::Heap`, `Step::Memory`: any buffer the heap reaches), memory the
/// runtime shows at two addresses, title memory of textures (uploaded
/// before a submission's lists, so a copy into it is a submission late for
/// every reader), and a copy whose source or destination an earlier
/// command of the submission writes on the GPU.
pub(crate) fn copy_order_parts(
    lists: Vec<std::sync::Arc<HostList>>,
) -> Vec<Vec<std::sync::Arc<HostList>>> {
    if lists.len() < 2 || crate::mac_switches::copy_order_split().is_none() {
        return vec![lists];
    }
    // One list's lock at a time (a title may submit a list twice). Most
    // pieces have no copy after their first list: nothing more is read.
    let copies: Vec<(Vec<(usize, usize)>, bool)> = lists
        .iter()
        .map(|l| {
            let d = l.deferred();
            (d.copied.clone(), d.after_copy)
        })
        .collect();
    if copies[1..].iter().all(|(copied, after_copy)| copied.is_empty() && !after_copy) {
        return vec![lists];
    }
    let reads: Vec<crate::ranges::Ranges> = lists.iter().map(|l| l.deferred().reads.clone()).collect();
    let splits = crate::ranges::copy_order_splits(
        reads.iter().zip(&copies).map(|(r, (copied, after_copy))| (r, copied.as_slice(), *after_copy)),
    );
    if splits.is_empty() {
        return vec![lists];
    }
    COPY_SPLITS.fetch_add(splits.len() as u64, Ordering::Relaxed);
    let mut parts = Vec::with_capacity(splits.len() + 1);
    let mut rest = lists;
    for &at in splits.iter().rev() {
        parts.push(rest.split_off(at));
    }
    parts.push(rest);
    parts.reverse();
    parts
}

/// A submission's refreshes not done yet, as the ranges of each buffer copy
/// asked for (ranges less than [`JOIN`] apart are done as one): done before
/// a step that writes title memory or gives a range to shaders, and before
/// the lists run.
///
/// A copy's ranges brought up to date are not copied again in the
/// submission until a step writes them: every list of a world submission
/// using title memory asks for every copy whole, and the title copies into
/// buffers ~80 times in between. A step that may write anywhere
/// ([`Batch::title_written`]) makes all of them due again, a title copy
/// ([`Batch::title_copied`]) only what it writes. What the title's threads
/// write meanwhile is for later submissions (on the console it would race
/// the GPU).
#[derive(Default)]
struct Batch {
    /// The ranges still to bring up to date, by copy.
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    /// mac: the refreshes are made while the submission before still runs
    /// (`unwaited::flights`; `Shadow::update_early`).
    early: bool,
    /// Where each copy's refreshes are in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The ranges of each copy brought up to date since the last write that
    /// can have changed them, by copy (held, so that its address names no
    /// other).
    current: std::collections::HashMap<
        usize,
        (std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges),
    >,
    /// Totals and the largest ranges refreshed by this submission, for a
    /// bounded diagnostics line. No range contents or view attribution.
    #[cfg(feature = "diagnostics")]
    summary: crate::ranges::RefreshSummary,
    /// The small ranges the submission's commands read through tables so
    /// far ([`Batch::rewritten`]).
    #[cfg(feature = "diagnostics")]
    reads: crate::ranges::SmallReads,
    /// Where each copy in `current` sits in memory, by copy: a title copy
    /// reaches by its bytes only the copies that can hold them, not
    /// every copy flushed.
    extents: victory_formats::extents::Extents,
    /// Aliased memory asked for (`crate::alias`), by copy: never copied,
    /// counted once per submission for the frame line.
    zero_copy: std::collections::HashMap<usize, crate::ranges::Ranges>,
    /// Host copies refreshed with bytes copied, and those bytes (counted with
    /// `zero_copy`).
    copied: (usize, usize),
    /// The list being prepared starts after a copy-order cut and has not
    /// made a title copy yet (`crate::alias::title_copy`).
    cut: bool,
}

/// How far apart a copy's pending ranges may be and still be done as one,
/// the bytes between copied too: a call that copies a range costs about
/// what copying this much does at the 6-7 GB/s testers saw (a guess, not
/// measured).
const JOIN: usize = crate::watch::PAGE;

/// [`crate::shadow::Shadow::update`], or with `early` its form for a
/// submission prepared while the one before still runs (mac:
/// `unwaited::flights`; a copy it refuses, which the flights' rules leave
/// out, is counted and brought up to date the ordinary way).
fn refresh(shadow: &crate::shadow::Shadow, address: usize, len: usize, early: bool) -> usize {
    if !early {
        return shadow.update(address, len);
    }
    shadow.update_early(address, len).unwrap_or_else(|| {
        unwaited::flights::refused();
        shadow.update(address, len)
    })
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        // Title memory that is the copy's own: nothing to copy, ever.
        if shadow.is_aliased() && len > 0 {
            self.zero_copy.entry(key).or_default().insert(address, end, JOIN);
        }
        // Nothing to copy: a world submission asks for every copy its heaps'
        // views and page tables reach (~9,000), a few of them written.
        if shadow.clean() {
            return;
        }
        // Only what is not current: most lists ask for a copy whole, after
        // a title copy took a few bytes of it.
        let empty = crate::ranges::Ranges::default();
        let have = self.current.get(&key).map_or(&empty, |(_, have)| have);
        let mut todo = have.missing(address, end).peekable();
        if todo.peek().is_none() {
            // Already current: what a later list asked again, and what
            // #259 keeps current across its moves.
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, (end - address) as u64);
            }
            return;
        }
        let i = *self.pending.entry(key).or_insert_with(|| {
            self.refreshes.push((shadow.clone(), Default::default()));
            self.refreshes.len() - 1
        });
        for (from, to) in todo {
            self.refreshes[i].1.insert(from, to, JOIN);
        }
    }

    /// Before a step that may write any title memory: pending refreshes are
    /// done first, and copies are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.current.clear();
        self.extents.clear();
    }

    /// As [`Batch::title_written`], before a copy into the title memory
    /// `[at, at + len)`, which `copy_mapped` writes and nothing more: only
    /// what the copy can have changed is checked again after it.
    // shortcut: memory shown at two addresses is what the runtime said it
    // maps twice (`crate::watch::aliased`, as the trackers know it); where
    // it says so, the copy makes everything due again, as it may change
    // bytes any copy holds.
    fn title_copied(&mut self, at: usize, len: usize) {
        if crate::watch::aliased(at, len) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::AliasedCopies, 1);
            return self.title_written();
        }
        self.flush();
        let (from, to) = touched(at, at.saturating_add(len));
        // Only the copies whose memory the bytes can be in: the rest
        // hold nothing of the window, and what a no-op remove would
        // leave whole it leaves whole.
        for key in self.extents.overlapping(from, to) {
            if let Some((_, have)) = self.current.get_mut(&key) {
                have.remove(from, to);
            }
        }
    }

    /// Does the refreshes; returns the bytes copied.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let work = std::mem::take(&mut self.refreshes);
        let total_requested: usize = work
            .iter()
            .flat_map(|(_, ranges)| ranges.iter())
            .map(|(from, to)| to - from)
            .sum();
        let mut total = 0;
        #[cfg(feature = "diagnostics")]
        let mut counted = crate::copystat::hold();
        if crate::shadow::parallel_copies_enabled()
            && work.len() >= 2
            && total_requested >= crate::shadow::PARALLEL_REFRESH_MIN_BYTES
        {
            type RefreshOut = Vec<(usize, usize, usize, u64)>;
            let results: Vec<Mutex<RefreshOut>> =
                (0..work.len()).map(|_| Mutex::default()).collect();
            let early = self.early;
            crate::texture::in_parallel(work.len(), |i| {
                let (shadow, ranges) = &work[i];
                let mut out = Vec::new();
                for (from, to) in ranges.iter() {
                    let started = std::time::Instant::now();
                    let copied = refresh(shadow, from, to - from, early);
                    let us = started.elapsed().as_micros() as u64;
                    out.push((from, to, copied, us));
                }
                *results[i].lock().unwrap_or_else(|e| e.into_inner()) = out;
            });
            for ((shadow, _), done) in work.into_iter().zip(results) {
                let key = std::sync::Arc::as_ptr(&shadow) as usize;
                let (start, len) = shadow.range();
                let (before, mut spent) = (total, 0);
                for (from, to, copied, us) in done.into_inner().unwrap_or_else(|e| e.into_inner()) {
                    spent += us;
                    #[cfg(feature = "diagnostics")]
                    {
                        counted.range(from, to - from, us, 0);
                        self.summary
                            .record(from as u64, (to - from) as u64, shadow.gpu(from as u64));
                    }
                    PREPARING[0].fetch_add(us, Ordering::Relaxed);
                    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                    total += copied;
                    let (from, to) = taken(from, to, (start as usize, (start + len) as usize));
                    let (_, have) = match self.current.entry(key) {
                        std::collections::hash_map::Entry::Vacant(vacant) => {
                            self.extents
                                .flushed(key, start as usize, (start + len) as usize);
                            vacant.insert((shadow.clone(), Default::default()))
                        }
                        std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                    };
                    have.insert(from, to, 0);
                }
                if total > before {
                    self.copied.0 += 1;
                    self.copied.1 += total - before;
                    if crate::physical_runs::on() {
                        crate::physical_runs::refreshed(start as usize, len as usize, total - before, spent);
                    }
                }
            }
            return total;
        }
        for (shadow, ranges) in work {
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            let (start, len) = shadow.range();
            let (before, mut spent) = (total, 0);
            for (from, to) in ranges.iter() {
                let started = std::time::Instant::now();
                let copied = refresh(&shadow, from, to - from, self.early);
                let us = started.elapsed().as_micros() as u64;
                spent += us;
                #[cfg(feature = "diagnostics")]
                {
                    counted.range(from, to - from, us, 0);
                    self.summary
                        .record(from as u64, (to - from) as u64, shadow.gpu(from as u64));
                }
                PREPARING[0].fetch_add(us, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                total += copied;
                let (from, to) = taken(from, to, (start as usize, (start + len) as usize));
                // The copy's memory joins the index as its first range
                // joins `current`: a later title copy reaches it by its
                // bytes.
                let (_, have) = match self.current.entry(key) {
                    std::collections::hash_map::Entry::Vacant(vacant) => {
                        self.extents
                            .flushed(key, start as usize, (start + len) as usize);
                        vacant.insert((shadow.clone(), Default::default()))
                    }
                    std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                };
                have.insert(from, to, 0);
            }
            if total > before {
                self.copied.0 += 1;
                self.copied.1 += total - before;
                // mac: where the copied memory lies (`VICTORY_PHYSICAL_RUNS`).
                if crate::physical_runs::on() {
                    crate::physical_runs::refreshed(start as usize, len as usize, total - before, spent);
                }
            }
        }
        total
    }

    /// At the end of the submission's preparing, with `VICTORY_ALIAS_BUFFERS`:
    /// what it asked of aliased memory and copied into host copies, for the
    /// frame line.
    fn count_zero_copy(&mut self) {
        if !crate::alias::enabled() {
            return;
        }
        let (mut ranges, mut bytes) = (0, 0);
        for asked in self.zero_copy.values() {
            for (from, to) in asked.iter() {
                ranges += 1;
                bytes += to - from;
            }
        }
        crate::alias::asked(self.zero_copy.len(), ranges, bytes, self.copied.0, self.copied.1);
        self.zero_copy.clear();
        self.copied = (0, 0);
    }

    /// Diagnostics, before a title copy of `len` bytes from `src` to `dst`
    /// (a list's `CopyBufferRegion`, buffer `CopyResource` or
    /// `CopyMemoryToMemoryX`): logs it when it changes a small range that a
    /// command earlier in the same submission reads through a table. On
    /// the console a list's copies and draws run in their order, so that
    /// command reads the old bytes; here every step of a submission is
    /// done before its first GPU command, so it runs with the new ones
    /// (the title's TAA passes then work on a quarter of the screen,
    /// `notes/uav-size.md`). The first 32 such copies of a run and those
    /// of listed capture frames; `VICTORY_COPY_ORDER=0` turns it off. With
    /// `VICTORY_COPY_ORDER=split` such a copy starts a submission of its
    /// own, so a line here then names one the split did not see.
    ///
    /// Returns whether the copy is to be left out: with
    /// `VICTORY_COPY_ORDER=keep`, an experiment, when all it writes is one
    /// such range. Then the commands before it run with the bytes the
    /// title gave them, and those after it too (wrongly: they wanted the
    /// new ones): it shows what the earlier ones do with their own bytes.
    #[cfg(feature = "diagnostics")]
    fn rewritten(&self, dst: usize, src: usize, len: usize) -> bool {
        static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        // Off, or on and whether such a copy is left out.
        static MODE: OnceLock<Option<bool>> = OnceLock::new();
        let Some(keep) = *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_ORDER").as_deref() {
            Ok("0") => None,
            Ok("keep") => {
                tracing::info!("VICTORY_COPY_ORDER=keep: a title copy that only changes a small range read earlier in its submission is not made (experiment)");
                Some(true)
            }
            _ => Some(false),
        }) else {
            return false;
        };
        let mut left_out = false;
        for (from, to, views) in self.reads.written(dst, len) {
            // Both ranges read as the copy reads them (an unmapped page
            // reads as zeros); the same bytes change nothing.
            let n = (to - from).min(64);
            let (mut old, mut new) = ([0u8; 64], [0u8; 64]);
            copy_mapped(old.as_mut_ptr(), from as *const u8, n, false);
            copy_mapped(new.as_mut_ptr(), (src + (from - dst)) as *const u8, n, false);
            if old[..n] == new[..n] {
                continue;
            }
            left_out |= keep && (from, to) == (dst, dst.saturating_add(len));
            let k = LOGGED.fetch_add(1, Ordering::Relaxed);
            if k >= 32 && !crate::capture::listed(crate::queue::presented() + 1) {
                continue;
            }
            let floats = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            tracing::info!(
                target: "victory_gpu::trace",
                "copy order: a title copy changes {:#x}+{}, which commands earlier in this submission read through a table ({views} view ranges): {:?} becomes {:?}; {}",
                from,
                to - from,
                floats(&old[..n]),
                floats(&new[..n]),
                if left_out { "left out (VICTORY_COPY_ORDER=keep)" } else { "they run with the new bytes" }
            );
        }
        left_out
    }

    /// Logs one compact line for the submission, if it refreshed title memory.
    #[cfg(feature = "diagnostics")]
    fn log_refreshes(&self, submission: u64, lists: usize) {
        if !self.summary.is_empty() {
            tracing::info!(
                submission,
                lists,
                "title-memory refreshes: {}",
                self.summary
            );
        }
    }
}

/// What a write to title memory `[from, to)` changes in the trackers: the
/// 64 KiB pages it touches, whole (`crate::watch::writing`); just the bytes
/// where nothing is tracked.
fn touched(from: usize, to: usize) -> (usize, usize) {
    let page = crate::watch::PAGE;
    if crate::watch::fault_tracking() {
        (from / page * page, to.div_ceil(page).saturating_mul(page))
    } else {
        (from, to)
    }
}

/// What a refresh of `[from, to)` of the copy of title memory `memory`
/// (where it starts and ends) leaves current: all of it, but with tracking
/// only the pages it took, those it holds whole (the ends of the copy's
/// memory count whole, as `Tracked::take` counts them). A page it holds in
/// part stays changed, and the next request that holds it whole must take
/// it, or copy-back reads it as written by the title and keeps the
/// shaders' writes there out of title memory.
fn taken(from: usize, to: usize, memory: (usize, usize)) -> (usize, usize) {
    let page = crate::watch::PAGE;
    if crate::watch::fault_tracking() {
        (
            if from <= memory.0 {
                from
            } else {
                from.next_multiple_of(page)
            },
            if to >= memory.1 { to } else { to / page * page },
        )
    } else {
        (from, to)
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::MoveUs,
        moved.elapsed().as_micros() as u64,
    );
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FallbackUs);
    // The whole copy failed. Copy each readable region in one call and
    // zero the holes. In particular, a hole that starts inside a 64 KiB
    // block does not require a faulting read of each remaining 4 KiB page.
    let (start, end) = (src as usize, src as usize + len);
    let mut at = start;
    while at < end {
        let (region_end, readable) = source_region(at);
        let next = region_end.min(end);
        let to = dst.wrapping_add(at - start);
        if !readable {
            // Wine's view of memory can omit host mappings. Keep trying
            // the initial 64 KiB piece before treating the region as a hole.
            let piece = ((at >> 16) + 1).saturating_mul(1 << 16).min(end);
            if read(at as *const u8, to, piece - at) {
                at = piece;
                continue;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (next - at) as u64);
            // SAFETY: inside the destination, which is mapped.
            unsafe { to.write_bytes(0, next - at) };
        } else if !read(at as *const u8, to, next - at) {
            // Protection can change after the query. Keep trying smaller
            // pieces and zero only the pages the kernel cannot read.
            for_pieces(at, next - at, 64 << 10, |block, len| {
                let to = dst.wrapping_add(block - start);
                if !read(block as *const u8, to, len) {
                    for_pieces(block, len, 4 << 10, |page, len| {
                        let to = dst.wrapping_add(page - start);
                        if !read(page as *const u8, to, len) {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Unmapped, len as u64);
                            // SAFETY: inside the mapped destination.
                            unsafe { to.write_bytes(0, len) };
                        }
                    });
                }
            });
        }
        at = next;
    }
}

/// Moves bytes between overlapping title ranges when both are committed.
fn move_mapped(dst: *mut u8, src: *const u8, len: usize) {
    if crate::texture::committed(src, len) && crate::texture::committed(dst, len) {
        let _writing = crate::watch::writing(dst as usize, len);
        // SAFETY: both ranges name committed title memory. The copy
        // preserves the source bytes when the ranges overlap.
        unsafe { std::ptr::copy(src, dst, len) };
    }
}

/// The end and readability of the region containing `at`.
fn source_region(at: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let page_end = (at | 0xfff).saturating_add(1);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return (page_end, false);
    }
    let readable = info.State == MEM_COMMIT
        && info.Protect != 0
        && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
    let end = (info.BaseAddress as usize)
        .saturating_add(info.RegionSize)
        .max(page_end);
    (end, readable)
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn long_unmapped_stretches_read_as_zeros() {
        // 64 reserved blocks: the first and 8 KiB of the second committed,
        // and the last: a hole starting inside a block and spanning many.
        const N: usize = 64;
        let commit = |at: *mut u8, len: usize, fill: u8| {
            // SAFETY: inside the reservation below, ours to fill.
            unsafe {
                assert!(!VirtualAlloc(at.cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, len);
            }
        };
        // SAFETY: a fresh reservation, released below.
        let base = unsafe { VirtualAlloc(std::ptr::null(), N * BLOCK, MEM_RESERVE, PAGE_NOACCESS) };
        let base = base.cast::<u8>();
        assert!(!base.is_null());
        commit(base, BLOCK + (8 << 10), 7);
        // SAFETY: the last block of the reservation.
        commit(unsafe { base.add((N - 1) * BLOCK) }, BLOCK, 9);
        // Neither end aligned.
        let mut to = vec![0xeeu8; N * BLOCK - 200];
        // SAFETY: inside the reservation.
        copy_mapped(
            to.as_mut_ptr(),
            unsafe { base.add(100) },
            N * BLOCK - 200,
            false,
        );
        let (hole, last) = (BLOCK + (8 << 10) - 100, (N - 1) * BLOCK - 100);
        assert!(to[..hole].iter().all(|&b| b == 7));
        assert!(to[hole..last].iter().all(|&b| b == 0));
        assert!(to[last..].iter().all(|&b| b == 9));
        release(base);
    }

    #[test]
    fn protected_holes_inside_a_copy_preserve_the_readable_bytes() {
        const PAGE: usize = 4 << 10;
        // SAFETY: a fresh allocation, filled and protected by this test.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
                .cast::<u8>()
        };
        assert!(!base.is_null());
        let mut old = 0;
        // SAFETY: these are pages of the allocation above.
        unsafe {
            base.write_bytes(7, 4 * BLOCK);
            for offset in [3 * PAGE, BLOCK + PAGE] {
                assert_ne!(
                    VirtualProtect(base.add(offset).cast(), PAGE, PAGE_NOACCESS, &mut old),
                    0
                );
            }
        }
        let mut to = vec![0xee; 4 * BLOCK - 200];
        copy_mapped(to.as_mut_ptr(), base.wrapping_add(100), to.len(), false);
        for (i, byte) in to.into_iter().enumerate() {
            let page = (i + 100) / PAGE;
            let expected = if [3, BLOCK / PAGE + 1].contains(&page) { 0 } else { 7 };
            assert_eq!(byte, expected, "byte {i:#x}");
        }
        release(base);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A buffer in the readback heap, mapped, in the copy-destination state.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, true)
    }

    /// A mapped buffer of `size` bytes: upload memory, or (`readback`)
    /// readback memory in the copy-destination state.
    fn mapped_buffer(&self, size: u64, readback: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback { D3D12_HEAP_TYPE_READBACK } else { D3D12_HEAP_TYPE_UPLOAD },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    if readback { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, readback, "no mapped buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
pub(crate) fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    /// The queue the swap chain presents on.
    queue: ID3D12CommandQueue,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    /// The overlay blend pass, once tried (#15).
    compositor: Option<std::result::Result<crate::composite::Compositor, ()>>,
    /// The texture the blend pass writes.
    blended: Option<ID3D12Resource>,
}

#[derive(Clone, Copy)]
struct FrameWindow {
    since: Instant,
    frames: u64,
    drawn: u64,
}

struct FrameRates {
    frames: f64,
    fps: f64,
    ms: f64,
    draws: f64,
}

impl FrameWindow {
    /// Advances this window once its own interval has elapsed.
    fn sample(&mut self, next: Self, interval: Duration) -> Option<FrameRates> {
        let elapsed = next.since.duration_since(self.since);
        if elapsed < interval {
            return None;
        }
        let seconds = elapsed.as_secs_f64();
        let frames = (next.frames - self.frames) as f64;
        let draws = (next.drawn - self.drawn) as f64 / frames;
        *self = next;
        Some(FrameRates {
            frames,
            fps: frames / seconds,
            ms: seconds * 1000.0 / frames,
            draws,
        })
    }
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    frames: u64,
    caption_window: FrameWindow,
    report_window: FrameWindow,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let start = FrameWindow {
            since: Instant::now(),
            frames: 0,
            drawn: crate::draw::drawn(),
        };
        Meter {
            frames: 0,
            caption_window: start,
            report_window: start,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        let next = FrameWindow {
            since: Instant::now(),
            frames: self.frames,
            drawn: crate::draw::drawn(),
        };
        let Some(caption_rates) = self.caption_window.sample(next, Duration::from_secs(1)) else {
            return;
        };
        if let Some(report) = self.report_window.sample(next, Duration::from_secs(5)) {
            let FrameRates { frames: report_frames, fps, ms, draws } = report;
            let [cpu, gpu] = SUBMISSION_US.each_ref().map(|counter| {
                counter.swap(0, Ordering::Relaxed) as f64 / 1000.0 / report_frames
            });
            let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
                let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
                PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / report_frames
            });
            let [back_ms, back_mb] = [0, 1].map(|i| {
                COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / report_frames
            });
            // With `VICTORY_ALIAS_BUFFERS` only (else empty): the same frames'
            // zero-copy and copy paths.
            let zero_copy = crate::alias::frame_line(report_frames);
            // mac: the guard lock's waits and the protection calls, always.
            let guards = crate::watch::frame_line(report_frames);
            let dropped = crate::draw::take_dropped() as f64 / report_frames;
            let indirect = crate::list::take_indirect() as f64 / report_frames;
            let [records, counted] =
                crate::indirect::take_records().map(|n| n as f64 / report_frames);
            let root_views = crate::draw::take_root_view_slots() as f64 / report_frames;
            let clamps = crate::indirect::take_dispatches() as f64 / report_frames;
            let hoisted = crate::indirect::take_hoisted() as f64 / report_frames;
            let hoists = crate::list::take_hoists() as f64 / report_frames;
            let (batched, batches) = crate::indirect::take_batched();
            let (batched, batches) = (batched as f64 / report_frames, batches as f64 / report_frames);
            let [for_dispatches, dispatches_hoisted, dispatches_batched] =
                crate::indirect::take_for_dispatches().map(|n| n as f64 / report_frames);
            let dispatch_hoists = crate::list::take_dispatch_hoists() as f64 / report_frames;
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame; indirect \
                 max records {records:.0}/frame ({counted:.0} under a count), {clamps:.0} clamp \
                 dispatches/frame ({hoisted:.0} clamps in the list before their draw's, \
                 {hoists:.0} lists cut for them; {batched:.0} of them in {batches:.0} batches; \
                 {for_dispatches:.0} clamps of indirect dispatches, {dispatches_hoisted:.0} of them \
                 in the list before, {dispatch_hoists:.0} lists cut at one, {dispatches_batched:.0} \
                 batched){zero_copy}{guards}",
                making = crate::pipeline::making()
            );
            unwaited::log(report_frames);
            crate::ray_skip::log(report_frames);
            crate::texback::log(report_frames);
            crate::physical_runs::log(report_frames);
            // mac: the heap locks' waits (`VICTORY_HEAP_LOCKS`).
            crate::descriptor::slots::log(report_frames);
            crate::descriptor::pool::log(report_frames);
            crate::descriptor::reports::log(report_frames);
            crate::ambient_estimate::log(report_frames);
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let (heaps, heap_used, heap_free) = crate::placement::stats();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 heaps of textures {heaps} ({:.0} MB used, {:.0} MB free); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                heap_used as f64 / 1e6,
                heap_free as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                if let Some([(local, budget), (system, system_budget)]) = video_memory() {
                    let (commit, working_set) = process_memory();
                    tracing::info!(
                        "video memory: {local:.0} MB of a {budget:.0} MB budget on the card, \
                         {system:.0} MB of {system_budget:.0} MB in system memory; process commit {:.0} MB, \
                         working set {:.0} MB; Rust heap {:.0} MB; placed textures made {:.0} MB; list upload memory {} blocks {:.0} MB",
                        commit as f64 / 1e6,
                        working_set as f64 / 1e6,
                        crate::copystat::HEAP_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::resource::PLACED_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::list::UPLOADS[0].load(Ordering::Relaxed),
                        crate::list::UPLOADS[1].load(Ordering::Relaxed) as f64 / 1e6,
                    );
                }
                crate::shadow::log_held();
                let [gpu, title] = crate::resource::texture_census().map(|r| r.map(|b| b as f64 / 1e6));
                tracing::info!(
                    "host texture MB (used by lists, viewed by heap slots only, neither): GPU-written {:.0} {:.0} {:.0}, \
                     from title memory {:.0} {:.0} {:.0}",
                    gpu[0], gpu[1], gpu[2], title[0], title[1], title[2]
                );
                let [given, given_bytes, remade, soon] =
                    [0, 1, 2, 3].map(|i| crate::resource::EVICTIONS[i].swap(0, Ordering::Relaxed));
                tracing::info!(
                    "host textures given up: {given} ({:.0} MB); made again after being given up: {remade}, {soon} of them within {} frames",
                    given_bytes as f64 / 1e6,
                    crate::resource::SOON_FRAMES,
                );
                // Window caption updates leave the report counters intact.
                let drops = crate::pipeline::take_drops();
                crate::frame::log();
                log_gpu_times(report_frames);
                crate::copystat::log(report_frames);
                crate::bindless::log(report_frames);
                crate::shadow::log_views(report_frames);
                crate::shadow::log_lookups(report_frames);
                let [n, bytes] = [0, 1].map(|i| crate::list::READ_BACKS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "texture read backs: {:.1}/frame, {:.1} KB/frame",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3
                    );
                }
                let [n, bytes, own] =
                    [0, 1, 2].map(|i| crate::list::FOOTPRINT_UPLOADS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "buffer to texture copies: {:.1}/frame, {:.1} KB/frame, {:.1}/frame in a buffer of their own",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3,
                        own as f64 / report_frames
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / report_frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let FrameRates { fps, ms, draws, .. } = caption_rates;
        let text = format!("{fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = crate::caption::with_stats(caption, &text);
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on), with `overlay` (a
    /// texture and its host state, which it keeps) blended over it when
    /// `VICTORY_PRESENT_COMPOSITE=1`. `None` if nothing was shown; else, in
    /// diagnostics builds, whether the frame was black.
    pub(crate) fn present(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(picture_format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        // The overlay with its view format, when it is to be blended in.
        let overlay = overlay.filter(|_| crate::composite::wanted()).and_then(|(o, s)| {
            // SAFETY: a live resource.
            let f = swap_format(unsafe { o.GetDesc() }.Format)?;
            Some((o, s, f))
        });
        let native = (desc.Width as u32, desc.Height);
        let size = crate::composite::output_size(native.0, native.1);
        let can_sample = desc.SampleDesc.Count == 1
            && desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE == D3D12_RESOURCE_FLAG_NONE
            && (desc.Format == picture_format || crate::resource::family(desc.Format.0 as u32) == desc.Format.0 as u32);
        let use_pass = overlay.is_some() || (size != native && can_sample);
        let (width, height) = if use_pass { size } else { native };
        let format = if use_pass { crate::composite::result_format(picture_format) } else { picture_format };
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        // A swap chain presents on the queue it was made with. Once that
        // queue stopped running work and was replaced ([`Host::lost`]), the
        // chain goes first (a window takes one at a time), and a new one is
        // made for the same window on the new queue.
        let queue = self.queue();
        let mut shown = None;
        if presenter.as_ref().is_some_and(|p| p.queue != queue) {
            shown = presenter.take().map(|p| p.hwnd);
        }
        if presenter.is_none() {
            *presenter = match retry.attempt(Instant::now(), || {
                self.presenter(&queue, shown, width, height, format)
            })? {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        // The blend pass and the texture it writes, made on first use.
        let mut blended = None;
        if use_pass {
            if p.compositor.is_none() {
                p.compositor = Some(crate::composite::Compositor::new(&self.device).map_err(|e| {
                    tracing::warn!(error = %e, "no overlay blending: the pass did not compile");
                }));
            }
            if let Some(Ok(compositor)) = &p.compositor {
                let current = p.blended.as_ref().is_some_and(|t| {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    (d.Width, d.Height, d.Format) == (u64::from(width), height, format)
                });
                if !current {
                    p.blended =
                        crate::composite::result_texture(&self.device, u64::from(width), height, format).ok();
                }
                if let Some(result) = &p.blended {
                    blended = Some((compositor, result.clone()));
                }
            }
        }
        // If the pass or its output is unavailable, the back buffer must
        // match the original texture before the direct CopyResource below.
        let (width, height, format) = if blended.is_some() {
            (width, height, format)
        } else {
            (native.0, native.1, picture_format)
        };
        // What is presented, whenever it changes: the title's texture in its
        // host format, and the swap chain format it is copied into.
        /// Width, height and format.
        type Shape = (u32, u32, i32);
        static SHOWN: Mutex<Option<(Shape, Shape)>> = Mutex::new(None);
        let shown = Some(((native.0, native.1, desc.Format.0), (width, height, format.0)));
        let mut last = SHOWN.lock().unwrap_or_else(|e| e.into_inner());
        if *last != shown {
            *last = shown;
            tracing::info!(
                frame,
                width = desc.Width,
                height = desc.Height,
                texture_format = desc.Format.0,
                flags = desc.Flags.0,
                swap_format = format.0,
                output_width = width,
                output_height = height,
                "presenting a title texture"
            );
        }
        drop(last);
        // This also covers a swap chain created at the capped size when the
        // blend pass later cannot be built: width, height and format were reset
        // to the title texture's own above, so the buffers are resized to match
        // before the direct copy.
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // What is copied to the back buffer: the blend or the picture,
            // and the state it returns to afterwards.
            let (source, state) = match &blended {
                Some((compositor, result)) => {
                    list.ResourceBarrier(&[transition(texture, state, READ)]);
                    if let Some((o, ostate, _)) = overlay {
                        list.ResourceBarrier(&[transition(o, D3D12_RESOURCE_STATES(ostate as i32), READ)]);
                    }
                    compositor.record(&self.device, list, (texture, picture_format),
                        overlay.map(|(o, _, f)| (o, f)), result, format);
                    if let Some((o, ostate, _)) = overlay {
                        list.ResourceBarrier(&[transition(o, READ, D3D12_RESOURCE_STATES(ostate as i32))]);
                    }
                    list.ResourceBarrier(&[
                        transition(texture, READ, state),
                        transition(result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    ]);
                    (result.clone(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                }
                None => {
                    list.ResourceBarrier(&[transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    (texture.clone(), state)
                }
            };
            let texture = &source;
            list.ResourceBarrier(&[transition(
                &back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    /// A swap chain on `queue` for `shown`, else for the title's window,
    /// else for a window of our own.
    fn presenter(
        &self,
        queue: &ID3D12CommandQueue,
        shown: Option<HWND>,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match shown.or_else(title_window) {
            Some(hwnd) => hwnd,
            None if std::env::var("VICTORY_WINDOW_SECONDARY").is_ok_and(|v| {
                v == "1"
                    || v.split_once(',').is_some_and(|(x, y)| {
                        x.trim().parse::<i32>().is_ok() && y.trim().parse::<i32>().is_ok()
                    })
            }) =>
            {
                return Err(("secondary title window unavailable", windows::core::Error::empty()));
            }
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        self.presenter_for_window(queue, hwnd, width, height, format)
    }

    fn presenter_for_window(
        &self,
        queue: &ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                queue: queue.clone(),
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                compositor: None,
                blended: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// Diagnostics, `VICTORY_COPIES_IN_VIDEO_MEMORY`: [`Host::shared_buffer`]s
    /// (host copies of title memory among them) in video memory the CPU maps
    /// (a GPU upload heap, where the host has one), to measure how much of
    /// the GPU's time is shaders reading the copies over the bus. CPU reads
    /// of them (snapshots, copy-backs) become slow: for that measurement only.
    #[cfg(feature = "diagnostics")]
    fn copies_in_video_memory(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_COPIES_IN_VIDEO_MEMORY") {
                return false;
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            }
            .is_ok()
                && options.GPUUploadHeapSupported.as_bool();
            tracing::warn!(on = ok, "host copies of title memory in video memory (diagnostics)");
            ok
        })
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        #[cfg(feature = "diagnostics")]
        let heap = if self.copies_in_video_memory() {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
        } else {
            heap
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    // mac: taken as before; counted, and tried again first, with
    // `VICTORY_HEAP_LOCKS` (`crate::descriptor::slots`).
    crate::descriptor::slots::locked(&RETIRED, crate::descriptor::slots::RETIRED).push(resource);
}

/// After a submission has run, with none running: the retired resources go.
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    if !retired.is_empty() {
        release_later(Box::new(retired));
    }
}

/// Drops `what` on a thread of its own: releasing a D3D12 resource is a
/// kernel call (~5 % of a world frame on the title's submitting thread,
/// run perf1). Only for objects no GPU work in flight uses.
fn release_later(what: Box<dyn std::any::Any + Send>) {
    type Release = std::sync::mpsc::Sender<Box<dyn std::any::Any + Send>>;
    static RELEASE: OnceLock<Release> = OnceLock::new();
    let release = RELEASE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn std::any::Any + Send>>();
        // Without the thread the receiver is gone: sends fail and drop here.
        let _ = std::thread::Builder::new()
            .name("victory-release".into())
            .spawn(move || rx.into_iter().for_each(drop));
        tx
    });
    let _ = release.send(what);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds: all of
/// them in listed capture frames, the first 500 elsewhere.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if is_traced(what)
        && (crate::capture::listed(crate::queue::presented() + 1)
            || LOGGED.fetch_add(1, Ordering::Relaxed) < 500)
    {
        tracing::info!("traced command: {what}");
    }
}

/// The shader hashes `VICTORY_TRACE_SHADERS` lists (comma-separated;
/// diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_shaders() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// The shader hashes `VICTORY_TRACE_READS` lists (comma-separated;
/// diagnostics): in listed capture frames, what a pipeline with one of
/// them reads is logged (`crate::draw`).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_reads() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_READS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// Whether `what` names a shader [`traced_shaders`] lists.
#[cfg(feature = "diagnostics")]
pub(crate) fn is_traced(what: &str) -> bool {
    traced_shaders().iter().any(|h| what.contains(h.as_str()))
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!(
                    "; list {i} of {}, of its {}: {n}",
                    runs.len(),
                    notes.len()
                ));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(
    history: &[D3D12_AUTO_BREADCRUMB_OP],
    markers: &[(u32, String)],
    done: usize,
) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done]
        .iter()
        .rposition(barrier)
        .map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::ISSUED.with(|l| l.borrow_mut().push((resource.as_raw() as usize, before.0 as u32, after.0 as u32)));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A barrier that waits for every unordered-access write so far.
pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {
    // `mod tests` is already taken at the top of this file
    use super::Batch;
    use crate::shadow::Shadow;
    use crate::watch::PAGE;
    use std::sync::Arc;


    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A list asking again for what the submission already brought up to
    /// date copies nothing, and the copystat line says so: the skipped
    /// bytes are what staying current saved the later lists.
    #[test]
    #[cfg(feature = "diagnostics")]
    fn a_still_current_ask_is_counted_skipped() {
        let Some((p, copy)) = plain(2) else {
            return;
        };
        let len = 2 * PAGE;
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert!(batch.flush() > 0, "all of it at first");
        let (skips, bytes) = (
            crate::copystat::count(crate::copystat::C::Skipped),
            crate::copystat::count(crate::copystat::C::SkippedBytes),
        );
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        assert!(
            crate::copystat::count(crate::copystat::C::Skipped) > skips,
            "the ask counted"
        );
        assert!(
            crate::copystat::count(crate::copystat::C::SkippedBytes) >= bytes + len as u64,
            "its bytes"
        );
        done(p, Some(copy));
    }

    /// A copy of `pages` pages of plain private memory, which is not the
    /// runtime's: untracked, copied whole whatever was asked for. None
    /// without a host GPU.
    fn plain(pages: usize) -> Option<(usize, Arc<Shadow>)> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        // SAFETY: a fresh allocation, freed by `done`.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0);
        crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
        let copy = crate::shadow::find(p as u64, (pages * PAGE) as u64);
        if copy.is_none() {
            done(p, None);
        }
        copy.map(|c| (p, c))
    }

    fn done(p: usize, copy: Option<Arc<Shadow>>) {
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// Fills `len` bytes of title memory at `at`.
    fn fill(at: usize, len: usize, byte: u8) {
        // SAFETY: memory of the test's own allocations.
        unsafe { std::ptr::write_bytes(at as *mut u8, byte, len) };
    }

    /// A title copy of `len` bytes from `src` to `dst`, as a list's step
    /// runs it.
    fn title_copy(batch: &mut Batch, dst: usize, src: usize, len: usize) {
        batch.title_copied(dst, len);
        super::copy_mapped(dst as *mut u8, src as *const u8, len, true);
    }

    #[test]
    fn overlapping_title_copy_refreshes_its_destination() {
        let Some((pa, a)) = plain(4) else {
            return; // no host GPU here
        };
        let (pb, b) = plain(4).expect("second host buffer");
        fill(pa, PAGE, 0x12);
        fill(pa + PAGE, PAGE, 0x34);
        fill(pb, 4 * PAGE, 0x56);
        let mut batch = Batch::default();
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        let list = super::host().unwrap().new_list().unwrap();
        list.defer_move((pa + PAGE) as *mut u8, pa as *const u8, 2 * PAGE);
        list.run_deferred(&mut batch, None);
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        assert_eq!(
            batch.flush(),
            2 * PAGE,
            "only the overlapping copy's destination"
        );
        assert_eq!(a.read_host((pa + PAGE) as u64, 1)[0], 0x1212_1212);
        assert_eq!(a.read_host((pa + 2 * PAGE) as u64, 1)[0], 0x3434_3434);
        assert_eq!(b.read_host(pb as u64, 1)[0], 0x5656_5656);
        done(pa, Some(a));
        done(pb, Some(b));
    }

    /// Two copies of title memory, every list of a submission asking for
    /// both whole between title copies: a title copy makes due again only
    /// the bytes it wrote, and copies far apart in one copy are two
    /// ranges, not what is between them.
    #[test]
    fn a_title_copy_makes_due_again_only_the_bytes_it_wrote() {
        let (Some((pa, a)), Some((pb, b))) = (plain(64), plain(4)) else {
            return; // no host GPU here
        };
        let (la, lb) = (64 * PAGE, 4 * PAGE);
        fill(pb, lb, 0x5a);
        let mut batch = Batch::default();
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), la + lb, "all of both at first");
        // Two copies into the big one, a list asking for both after them.
        title_copy(&mut batch, pa + 5 * PAGE, pb, PAGE);
        title_copy(&mut batch, pa + 50 * PAGE, pb, 2 * PAGE);
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(
            batch.flush(),
            3 * PAGE,
            "the bytes written, not the 45 pages between"
        );
        assert_eq!(a.read_host((pa + 5 * PAGE) as u64, 1)[0], 0x5a5a_5a5a);
        assert_eq!(a.read_host((pa + 52 * PAGE - 4) as u64, 1)[0], 0x5a5a_5a5a);
        assert_eq!(a.read_host((pa + 52 * PAGE) as u64, 1)[0], 0, "not past it");
        assert_eq!(
            a.read_host((pa + 5 * PAGE - 4) as u64, 1)[0],
            0,
            "nor before it"
        );
        // And nothing more until the next write.
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), 0);
        // A step that may write anywhere makes everything due again.
        batch.title_written();
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), la + lb);
        done(pa, Some(a));
        done(pb, Some(b));
    }

    /// What a title copy wrote is checked again where it is, and only
    /// there, down to the bytes where nothing is tracked.
    #[test]
    fn a_title_copy_within_a_page_is_refreshed_exactly_where_nothing_is_tracked() {
        let Some((p, copy)) = plain(2) else {
            return; // no host GPU here
        };
        fill(p + PAGE, 8, 0x77);
        let mut batch = Batch::default();
        batch.add(&copy, p, 2 * PAGE);
        batch.flush();
        title_copy(&mut batch, p + 100, p + PAGE, 8);
        batch.add(&copy, p, 2 * PAGE);
        let tracking = crate::watch::fault_tracking();
        // With tracking the pages the write marks changed, whole.
        assert_eq!(batch.flush(), if tracking { PAGE } else { 8 });
        assert_eq!(copy.read_host(p as u64 + 100, 1)[0], 0x7777_7777);
        done(p, Some(copy));
    }

    /// A range asked for in part first is current only where it covers
    /// whole pages with tracking (a page taken in part stays changed, and
    /// copy-back reads that as the title's write): the request for all of
    /// the copy takes the ends whole.
    #[test]
    fn a_partial_request_leaves_the_ends_to_the_next() {
        let p = crate::watch::tests::pool(10);
        let len = 10 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            done(p, None);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p + 100, 6 * PAGE);
        assert_eq!(batch.flush(), 6 * PAGE, "just the bytes asked for");
        batch.add(&copy, p, len);
        let tracking = crate::watch::fault_tracking();
        let rest = if tracking { 5 * PAGE } else { 4 * PAGE };
        assert_eq!(
            batch.flush(),
            rest,
            "the pages the first took in part, and the rest"
        );
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0);
        // A fresh submission finds nothing changed.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), if tracking { 0 } else { len });
        done(p, Some(copy));
    }

    /// With tracking, a title copy inside one page leaves the page taken
    /// whole by the request after it, not changed for the next submission.
    #[test]
    fn a_title_copy_inside_a_page_takes_the_page_whole_with_tracking() {
        if !crate::watch::fault_tracking() {
            return;
        }
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            done(p, None);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len);
        let from = [0x33u8; 50];
        title_copy(&mut batch, p + PAGE + 100, from.as_ptr() as usize, 50);
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "the page the copy touched, once");
        assert_eq!(copy.read_host((p + PAGE + 100) as u64, 1)[0], 0x3333_3333);
        let mut next = Batch::default();
        next.add(&copy, p, len);
        assert_eq!(next.flush(), 0, "not left changed for the next submission");
        done(p, Some(copy));
    }

    /// Pages the runtime maps at two addresses: a title copy at one
    /// changes what a copy of the other holds, so it makes everything due
    /// again, as any step that may write anywhere does.
    #[test]
    fn a_title_copy_into_pages_shown_elsewhere_makes_everything_due_again() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let len = 2 * PAGE;
        crate::shadow::register(a, a as u64, len as u64);
        crate::shadow::register(b, b as u64, len as u64);
        let (Some(ca), Some(cb)) = (
            crate::shadow::find(a as u64, len as u64),
            crate::shadow::find(b as u64, len as u64),
        ) else {
            crate::shadow::unregister(a, a as u64);
            crate::shadow::unregister(b, b as u64);
            return; // no host GPU here
        };
        let src = crate::watch::tests::pool(1);
        fill(src, PAGE, 0x42);
        let mut batch = Batch::default();
        batch.add(&ca, a, len);
        batch.add(&cb, b, len);
        batch.flush();
        // Through the first view; the second shows it too.
        title_copy(&mut batch, a, src, PAGE);
        batch.add(&ca, a, len);
        batch.add(&cb, b, len);
        batch.flush();
        assert_eq!(
            cb.read_host(b as u64, 1)[0],
            0x4242_4242,
            "the other view's copy has it"
        );
        assert_eq!(ca.read_host(a as u64, 1)[0], 0x4242_4242);
        crate::shadow::unregister(a, a as u64);
        crate::shadow::unregister(b, b as u64);
        done(src, None);
    }
}

#[cfg(test)]
mod withholding {
    /// A list in which a program using title memory missed memory: what
    /// that program wrote is withheld from title memory and leaves the
    /// copies at once; what an ordinary program wrote in the same list goes
    /// back and stays in its copy (run light13: the imposter transforms in
    /// the BVH builder's list); an ordinary range overlapping a withheld one
    /// is withheld with it, since its copy-back would take the other's bytes
    /// along.
    #[test]
    fn only_programs_using_title_memory_are_withheld() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let pools = [crate::watch::tests::pool(1), crate::watch::tests::pool(1)];
        for &p in &pools {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(p, p as u64, PAGE as u64);
        }
        let copies: Vec<_> = pools.iter().map(|&p| crate::shadow::find(p as u64, PAGE as u64).unwrap()).collect();
        let list = host.new_list().unwrap();
        // (copy, start, bytes, the writer uses title memory, byte written at +0 and +16)
        let writes = [(0, 0, 32, false, 0x22), (1, 16, 16, true, 0x33), (1, 0, 32, false, 0x44)];
        for &(k, at, len, memory, value) in &writes {
            let _writer = list.writer_uses_memory(memory);
            let (copy, address) = (copies[k].clone(), (pools[k] + at) as u64);
            copy.refresh(&list, address, len);
            copy.gpu_writes(&list, address, len, false);
            // The shader's write, after the step that keeps the range's bytes.
            list.defer_host(move || copy.write_host(address, &[value]));
        }
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let title = |k: usize, at: usize| unsafe { ((pools[k] + at) as *const u8).read_volatile() };
        let copy = |k: usize, at: usize| {
            copies[k].update(pools[k], PAGE);
            copies[k].read_host((pools[k] + at) as u64, 1)[0] & 0xff
        };
        assert_eq!((title(0, 0), copy(0, 0)), (0x22, 0x22), "an ordinary program's write goes back and stays");
        assert_eq!((title(1, 16), copy(1, 16)), (0x11, 0x11), "withheld, then dropped from the copy");
        assert_eq!((title(1, 0), copy(1, 0)), (0, 0), "overlapping a withheld range: withheld with it");
        for (&p, copy) in pools.iter().zip(copies) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }
}

#[cfg(test)]
mod marks {
    use super::LOST;
    use std::sync::atomic::Ordering;
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::s;

    /// A submission that ran shows its number where the CPU reads it, so
    /// that one the host GPU drops can be told apart (VIctory-macOS #50).
    #[test]
    fn a_submission_that_ran_shows_its_number() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let mark = host.mark().expect("the mark list");
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        // Under the lock the last submission is the last one marked,
        // whatever other tests submitted meanwhile.
        let submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            mark.saw(*submitted),
            "the mark shows submission {}",
            *submitted
        );
    }

    /// Work goes on, on the same fence, once a new queue replaces the
    /// first, and a queue that runs work passes the check.
    #[test]
    fn work_goes_on_on_a_new_queue() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let mark = host.mark().expect("the mark list");
        let old = host.queue();
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            assert!(host.runs(&mut submitted, mark), "the queue runs work");
            host.replace_queue().unwrap();
            assert!(host.runs(&mut submitted, mark), "the new queue runs work");
        }
        assert!(host.queue() != old, "a new queue");
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        let submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        assert!(mark.saw(*submitted), "a submission after the change ran");
    }

    /// A draw over a `width` x `height` target whose pixels each loop `n`
    /// times, then write `tag` into their own word of a buffer the CPU reads.
    struct Spin {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        _target: ID3D12Resource,
        _rtvs: ID3D12DescriptorHeap,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        buffer: ID3D12Resource,
        words: *const u32,
        width: u32,
        height: u32,
    }

    const SPIN: &str = r#"
RWByteAddressBuffer o : register(u1);
cbuffer c : register(b0) { uint n; uint tag; uint width; }
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps(float4 pos : SV_Position) : SV_Target {
    uint id = (uint)pos.y * width + (uint)pos.x;
    uint a = id;
    for (uint i = 0; i < n; i++) { a = a * 1664525u + 1013904223u; }
    o.Store(id * 8 + 4, a);
    o.Store(id * 8, tag);
    return float4(a & 1, 0, 0, 1);
}
"#;

    impl Spin {
        fn new(host: &super::Host, width: u32, height: u32) -> Spin {
            use crate::compute::Param;
            use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
            let blob = |entry: windows::core::PCSTR, target: windows::core::PCSTR| {
                let mut code: Option<ID3DBlob> = None;
                // SAFETY: the source and names outlive the call.
                unsafe {
                    D3DCompile(
                        SPIN.as_ptr().cast(),
                        SPIN.len(),
                        s!("spin"),
                        None,
                        None,
                        entry,
                        target,
                        0,
                        0,
                        &mut code,
                        None,
                    )
                }
                .unwrap();
                code.unwrap()
            };
            let (vs, ps) = (blob(s!("vs"), s!("vs_5_1")), blob(s!("ps"), s!("ps_5_1")));
            // SAFETY: the blobs' own bytes.
            let bytes = |b: &ID3DBlob| unsafe {
                D3D12_SHADER_BYTECODE {
                    pShaderBytecode: b.GetBufferPointer(),
                    BytecodeLength: b.GetBufferSize(),
                }
            };
            let device = &host.device;
            let root = crate::compute::root_signature(
                device,
                &[
                    Param::Uav { register: 1 },
                    Param::Constants {
                        register: 0,
                        count: 3,
                    },
                ],
            )
            .unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                // SAFETY: borrowed for the call.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                VS: bytes(&vs),
                PS: bytes(&ps),
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            // SAFETY: plain object creation with valid descriptions.
            unsafe {
                let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
                let texture = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(width),
                    Height: height,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let mut target: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &texture,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut target,
                    )
                    .unwrap();
                let target = target.unwrap();
                let rtvs: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
                device.CreateRenderTargetView(&target, None, rtv);
                let words = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(width * height) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let mut buffer: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_CUSTOM,
                            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &words,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                let mut at = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut at)).unwrap();
                Spin {
                    root,
                    pso,
                    _target: target,
                    _rtvs: rtvs,
                    rtv,
                    buffer,
                    words: at.cast_const().cast(),
                    width,
                    height,
                }
            }
        }

        /// Draws; how long the submission took (ms) and how many pixels
        /// wrote `tag`.
        fn draw(&self, host: &super::Host, n: u32, tag: u32) -> (f64, usize) {
            let list = host.new_list().unwrap();
            let l = &list.list;
            // SAFETY: an open list; our pipeline, buffer and target.
            unsafe {
                l.SetPipelineState(&self.pso);
                l.SetGraphicsRootSignature(&self.root);
                l.SetGraphicsRootUnorderedAccessView(0, self.buffer.GetGPUVirtualAddress());
                l.SetGraphicsRoot32BitConstants(1, 3, [n, tag, self.width].as_ptr().cast(), 0);
                l.OMSetRenderTargets(1, Some(&self.rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: self.width as f32,
                    Height: self.height as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: self.width as i32,
                    bottom: self.height as i32,
                }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                l.DrawInstanced(3, 1, 0, 0);
            }
            list.close();
            let started = std::time::Instant::now();
            host.execute(&[&list]);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let pixels = (self.width * self.height) as usize;
            // SAFETY: our mapped buffer of 2 words per pixel.
            let ran = (0..pixels)
                .filter(|&i| unsafe { self.words.add(2 * i).read_volatile() } == tag)
                .count();
            (ms, ran)
        }
    }

    /// Under D3DMetal, macOS ends a draw whose every pixel runs long about
    /// 50 ms in, and after some of those the queue ignores all later work
    /// while its fence still completes (VIctory-macOS #50). The ended
    /// submissions are seen, a new queue replaces the one that stopped, and
    /// a draw after that runs whole. Wine on macOS only, and ignored:
    /// elsewhere these draws would run for seconds or reset the GPU.
    #[test]
    #[ignore]
    fn work_goes_on_after_macos_stops_the_queue() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (width, height) = (512, 256);
        let pixels = (width * height) as usize;
        let spin = Spin::new(host, width, height);
        let mut tag = 1;
        // Loop passes per millisecond, from a draw of 20 to 40 ms.
        let mut n = 1024u32;
        let per_ms = loop {
            tag += 1;
            let (ms, ran) = spin.draw(host, n, tag);
            assert_eq!(ran, pixels, "a short draw runs whole");
            if ms >= 20.0 {
                break f64::from(n) / ms;
            }
            n *= 2;
        };
        let long = (per_ms * 4000.0).min(f64::from(u32::MAX)) as u32;
        let [lost, queues] = [0, 1].map(|i| LOST[i].load(Ordering::Relaxed));
        for _ in 0..8 {
            tag += 1;
            let (_, ran) = spin.draw(host, long, tag);
            assert_ne!(ran, pixels, "macOS ends a draw of about 4 s");
            if LOST[1].load(Ordering::Relaxed) > queues {
                break;
            }
        }
        assert!(
            LOST[0].load(Ordering::Relaxed) > lost,
            "the ended draws were seen"
        );
        assert!(
            LOST[1].load(Ordering::Relaxed) > queues,
            "a new queue replaced the one that stopped"
        );
        tag += 1;
        assert_eq!(
            spin.draw(host, 4, tag).1,
            pixels,
            "a draw on the new queue runs whole"
        );
    }
}

/// The pipelines of the lists the last submission ran, for a hang's log line
/// (`Host::wait_for`): which work the GPU never finished.
static WAITING_ON: Mutex<String> = Mutex::new(String::new());
