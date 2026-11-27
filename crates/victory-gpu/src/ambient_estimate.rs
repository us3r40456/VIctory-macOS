//! A stand-in for the indirect light the title traces rays for, estimated
//! from the frame itself: avatarmanz's (`victory-gpu/src/ambient.rs` in
//! VIctory-7900XTX at 9576c45f, after beloooo's constant). On with
//! `VICTORY_AMBIENT=estimate` (or `estimate:<share>`), beside the constant
//! stand-in of `crate::ambient` (`VICTORY_AMBIENT=1`), which it leaves as it
//! is; off, nothing here runs.
//!
//! The constant stand-in gives every point of the picture one light: far
//! too little outdoors by day, flat grey indoors. This one works the light
//! out from what is on screen, so it follows the hour, the room and its
//! lamps by itself:
//!
//! - The title keeps the light its visible surfaces send out, for its own
//!   screen traces (`CS_CacheScreenSpaceRadiance`, an R11G11B10_FLOAT image
//!   half the picture's size, written near the end of a frame). Right after
//!   that program, in the same host list, the image is averaged in 64x36
//!   cells of a buffer of ours ([`keep`]), and a slow mean of all cells is
//!   moved a share of the way to this frame's.
//! - Right after the program that resolves the screen probes
//!   (`CS_ScreenProbes_TemporalResolve_*`, the next frame), the cells are
//!   placed in space by the depth image the resolve reads; at 160x90 points
//!   of the picture, the cells' light is summed, each cell weighted by the
//!   share of the point's surroundings it covers, as a mean and a direction
//!   per colour (first-order harmonics); and every texel of the resolve's
//!   light image is given what lies between the four points around it,
//!   those at its own depth first, turned by its normal, a fifth of it the
//!   slow mean, darker in creases ([`raise`]).
//!
//! - With `VICTORY_AMBIENT_AIR=<multiple>` (from avatarmanz 7da523a9; off
//!   by default), right after the program that adds the lamps' light to
//!   what the air scatters (`CS_AccumulateLocalLightInscatteringVolume`),
//!   that volume is bounded by a multiple of the light around ([`air`]):
//!   without rays the air of a room is lit by the sky's light, a brown veil
//!   at sunset.
//!
//! - `VICTORY_AMBIENT_SETTLE=<ms>[,<jump>[,<fresh s>]]` (from avatarmanz
//!   b4ba4bbb; unset: 250,3, from 9576c45f): the slow mean follows by a
//!   time constant, not by 0.04 a frame (`0`; 25 frames: 8 s at this line's
//!   3 frames a second), and takes a change by more than the jump factor
//!   at once.
//!
//! What it assumes: what is not on screen sends out what is (with the slow
//! mean, so that turning into a dark corner is not black at once). No
//! occlusion but the creases'.
//! shortcut: stands in for the title's traced probes; gone once rays run.
//!
//! What differs from the fork's file, for this line (as `crate::ambient`
//! differs from Belobug's):
//! - The mode is chosen by `VICTORY_AMBIENT=estimate[:<share>]`; in the
//!   fork a number alone is the share and `constant:<level>` the constant.
//! - The root signatures are built in code, and the shader keeps to what
//!   Wine's compiler takes (`crate::ambient_estimate_set::SOURCE`).
//! - The passes have a view heap of their own, made with other threads'
//!   submissions held off (`Host::quiet`, VIctory-macOS issue #31), and
//!   make their own views of the title's images; the fork uses the views
//!   in the title's heap. The list's heaps, root signature and pipeline
//!   are set again by the next draw (`State::host_state_lost`).
//! - The light visible surfaces send out is read as a shader resource: its
//!   host texture is moved to that state in the list
//!   (`HostList::move_texture`) for the keeping and back after it.
//! - An image is taken if a view of the format wanted can be made of its
//!   host texture (one host texture per texture memory, castable).
//! - What the title's resolve left is replaced unless
//!   `VICTORY_AMBIENT_LEFT` says how much of it counts (the fork: 4 times
//!   the estimate, fixed).
//! - Without a depth image of the light's size the slow mean alone is
//!   written; the fork then leaves the title's output.
//! - The fork's four fixed numbers of a texel's light (the slow mean's
//!   part, the normal's turn, the least and the most) can be set
//!   (`VICTORY_AMBIENT_MIX`); unset they are the fork's.
//! - No texel gets less than a floor: a share of the constant stand-in's
//!   light (`VICTORY_AMBIENT_FLOOR`, 0.3; `set::floor`), worked out once a
//!   frame from the sky dome at the keeping. Where little light is on
//!   screen the estimate has nothing to spread (a room lit only by its own
//!   bounce settles at black); in the fork the title's own resolve is under
//!   it there, which on this line leaves nothing.
//! - Reflections are held to a part of the light around, as the constant
//!   stand-in holds them to a part of its level ([`hold`],
//!   `VICTORY_AMBIENT_REFLECT`); the fork leaves them alone (`0` here).
//!   `VICTORY_AMBIENT_REFLECT_FLOOR` (off unless set) also holds them to
//!   no less than a part of it, for metal, whose light comes all from them.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::ambient_estimate_set as set;
use crate::host::{Host, HostList, host};
use crate::object::This;
use crate::resource::HostTexture;

pub(crate) use set::{AIR, KEEPS, LIGHTS_AIR, RADIANCE, role_name};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The share of the estimate asked for (`VICTORY_AMBIENT=estimate` or
/// `estimate:<share>`), if the estimate is. Read once.
fn share() -> Option<f32> {
    static SHARE: OnceLock<Option<f32>> = OnceLock::new();
    *SHARE.get_or_init(|| set::share(var("VICTORY_AMBIENT").as_deref()))
}

/// Whether the stand-in is the estimate.
pub(crate) fn on() -> bool {
    share().is_some()
}

/// The roles the estimate follows programs in beside the constant
/// stand-in's (bits; 0 while it is off).
pub(crate) fn roles() -> u8 {
    if on() { KEEPS | LIGHTS_AIR | blurred() } else { 0 }
}

/// With the reflection floor, the blur that writes the reflections the
/// lighting reads is followed too (`set::BLUR_PROGRAM`): the hold, and so
/// the floor, runs on its output as well as on the history. Off without the
/// floor, so that the default picture stays as it was.
fn blurred() -> u8 {
    if reflect_floor() > 0.0 { set::BLURS_REFLECTIONS } else { 0 }
}

/// What the program of shader container `container` is to the estimate
/// alone (0 while it is off: the container is not searched).
pub(crate) fn role_of(container: &[u8]) -> u8 {
    if on() { set::role_of(container) & (KEEPS | LIGHTS_AIR | blurred()) } else { 0 }
}

/// The creases' shade, as the constant stand-in's
/// (`VICTORY_AMBIENT_SHADE=s,r,n,t`).
fn shade() -> [f32; 4] {
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| crate::ambient_set::shade(var("VICTORY_AMBIENT_SHADE").as_deref()))
}

/// How much of what the title's resolve left counts
/// (`VICTORY_AMBIENT_LEFT`, `set::left`).
fn left() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::left(var("VICTORY_AMBIENT_LEFT").as_deref()))
}

/// The most the air scatters, as a multiple of the light around
/// (`VICTORY_AMBIENT_AIR`, `set::air`; 0: what the title has).
fn air_bound() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::air(var("VICTORY_AMBIENT_AIR").as_deref()))
}

/// How the slow mean of the light around settles
/// (`VICTORY_AMBIENT_SETTLE`, `set::settle`; by the clock unless it says
/// 0: then `None`, 0.04 of the way a frame).
fn settle() -> Option<set::Settle> {
    static SET: OnceLock<Option<set::Settle>> = OnceLock::new();
    *SET.get_or_init(|| set::settle(var("VICTORY_AMBIENT_SETTLE").as_deref()))
}

/// The milliseconds this keeping counts for (`set::frame_ms` of the time
/// since the one before; once a frame).
fn frame_ms() -> f32 {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let now = std::time::Instant::now();
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).replace(now);
    set::frame_ms(last.map(|last| now.duration_since(last).as_millis() as u64))
}

/// The most a reflection may hold, as a part of the light around: the
/// constant stand-in's `VICTORY_AMBIENT_REFLECT=<part>` (0.5 by default; 0
/// leaves the title's reflections alone, as the fork does).
///
/// Without rays the reflections are the sky's, unoccluded, wherever a
/// surface is glossy (`crate::ambient`): that does not change with how the
/// diffuse light is stood in for.
fn reflect_part() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| crate::ambient_set::keep(var("VICTORY_AMBIENT_REFLECT").as_deref()))
}

/// The least a reflection holds, as a part of the light around
/// (`VICTORY_AMBIENT_REFLECT_FLOOR`, `set::reflect_floor`; 0, the default:
/// none). Read once.
fn reflect_floor() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::reflect_floor(var("VICTORY_AMBIENT_REFLECT_FLOOR").as_deref()))
}

/// How a texel's light is put together (`VICTORY_AMBIENT_MIX`, `set::mix`;
/// the fork's fixed numbers unless set).
fn mix() -> [f32; 4] {
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| set::mix(var("VICTORY_AMBIENT_MIX").as_deref()))
}

/// The floor: the least any texel gets, as a share of the constant
/// stand-in's light (`VICTORY_AMBIENT_FLOOR`, `set::floor`; 0: none).
fn floor_share() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::floor(var("VICTORY_AMBIENT_FLOOR").as_deref()))
}

/// The probe: how many frame lines each of its steps lasts
/// (`VICTORY_AMBIENT_PROBE`, `set::probe`; `None`: no probe).
fn probe() -> Option<u32> {
    static SET: OnceLock<Option<u32>> = OnceLock::new();
    *SET.get_or_init(|| set::probe(var("VICTORY_AMBIENT_PROBE").as_deref()))
}

/// Frame lines logged so far: the probe's clock.
static LINES: AtomicU64 = AtomicU64::new(0);

/// The passes the probe leaves out now (bits of `set::NO_GATHER` and its
/// siblings; 0 without the probe).
fn left_out() -> u8 {
    match probe() {
        Some(each) => set::PROBE[set::probe_step(LINES.load(Ordering::Relaxed), each)].0,
        None => 0,
    }
}

/// Says what is on, once, when the host device is ready.
pub(crate) fn announce() {
    if let Some(each) = probe().filter(|_| on()) {
        tracing::info!(
            each,
            "VICTORY_AMBIENT_PROBE: the estimate's passes are left out in turn, each step for this many \
             frame lines; the estimate's line says which step the frame line above it was measured with"
        );
    }
    if let Some(share) = share() {
        tracing::info!(
            "{}",
            set::start_line(share, shade(), left(), air_bound(), settle(), reflect_part(), mix(), floor_share())
        );
        if reflect_floor() > 0.0 {
            tracing::info!(
                least = reflect_floor(),
                also_after = "CS_BilateralBlur",
                held = reflect_part(),
                "VICTORY_AMBIENT_REFLECT_FLOOR: no reflection holds less than this part of the light around \
                 the point seen there (the light the hold's cap is a part of); a darker one is replaced by it \
                 (nothing while the hold is off, VICTORY_AMBIENT_REFLECT=0)"
            );
        }
    }
}

/// Passes' views, used in turn: more than the passes of every list in
/// flight ([`VIEWS`] views a pass).
const SLOTS: u32 = 1024;
/// Views a pass: the image written, the image read, the sky dome.
const VIEWS: u32 = 3;

/// One pass: its pipeline and the root signature it was made with (the
/// same layout for all).
struct Stage {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct Pass {
    keep: Stage,
    mean: Stage,
    place: Stage,
    gather: Stage,
    raise: Stage,
    air: Stage,
    hold: Stage,
    /// The cells' light, the slow mean and the cells' places
    /// (`set::CELL_BYTES`), and where the CPU sees them.
    cells: ID3D12Resource,
    cells_at: usize,
    /// The points (`set::POINT_BYTES`), and where the CPU sees them.
    points: ID3D12Resource,
    points_at: usize,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded; the mappings live as the buffers.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// A pass of `set::SOURCE`, compiled with the system's compiler (Wine's
/// under Wine); its root signature is built here (`crate::compute`): the
/// constants, a table of the image written, the image read and the sky
/// dome, and the two buffers, each as a structured and as a raw view.
fn stage(pass: &str, name: PCSTR) -> windows::core::Result<Stage> {
    use crate::compute::{Param, srv, uav};
    let ranges = [uav(0, 1), srv(0, 2)];
    let params = [
        Param::Constants { register: 0, count: set::CONSTANTS as u32 },
        Param::Table(&ranges),
        Param::Uav { register: 1 },
        Param::Uav { register: 2 },
        Param::Uav { register: 3 },
        Param::Uav { register: 4 },
    ];
    let (root, pso) = crate::compute::pipeline(&set::part(pass), name, 0, &params)?;
    Ok(Stage { root, pso })
}

fn make(host: &Host) -> windows::core::Result<Pass> {
    let keep = stage(set::KEEP, s!("ambient estimate: keep"))?;
    let mean = stage(set::MEAN, s!("ambient estimate: mean"))?;
    let place = stage(set::PLACE, s!("ambient estimate: place"))?;
    let gather = stage(set::GATHER, s!("ambient estimate: gather"))?;
    let raise = stage(set::RAISE, s!("ambient estimate: raise"))?;
    let air = stage(set::AIR_PASS, s!("ambient estimate: air"))?;
    let hold = stage(set::HOLD, s!("ambient estimate: hold"))?;
    // Buffers the CPU sees too (`log` reads them), all zero to begin with:
    // nothing kept, no light.
    let buffer = |bytes: u64, what: &'static str| {
        let (buffer, data) = host.shared_buffer(bytes).ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the mapping holds the whole buffer.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        crate::host::name(&buffer, || what.into());
        windows::core::Result::Ok((buffer, data as usize))
    };
    let (cells, cells_at) = buffer(set::CELL_BYTES, "ambient estimate: cells")?;
    let (points, points_at) = buffer(set::POINT_BYTES, "ambient estimate: points")?;
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // Under Wine on D3DMetal, lists other threads submit while a view heap
    // is made can lose their views (VIctory-macOS issue #31).
    let _quiet = host.quiet();
    // SAFETY: plain creations.
    unsafe {
        let views = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * VIEWS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            keep,
            mean,
            place,
            gather,
            raise,
            air,
            hold,
            cells,
            cells_at,
            points,
            points_at,
            views,
            step: host.device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

/// The passes, made on first use; `None` if the host does not take them
/// (said once, with the compiler's messages).
fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(host)
            .inspect_err(|e| tracing::warn!(error = %e, "no ambient estimate: its passes were not made"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether views of `wanted` can be made of the host texture `copy`, whose
/// own format is `own` (as `crate::ambient` asks).
fn viewed_as(copy: &HostTexture, own: DXGI_FORMAT, wanted: DXGI_FORMAT) -> bool {
    own == wanted || copy.castable.contains(&(wanted.0 as u32))
}

/// An image a pass writes through `u0` or reads through `t0`, and the
/// format of its view.
#[derive(Clone, Copy)]
struct Image<'a> {
    image: &'a ID3D12Resource,
    format: DXGI_FORMAT,
    /// Slices of a volume written through `u0`; 0: two dimensions.
    slices: u32,
}

impl Pass {
    /// Makes the views of a pass's table in slots no list in flight reads,
    /// and sets the passes' view heap in `list`: the table. A null view
    /// where the pass has no image. `sky`: the first slice of the sky dome
    /// (the keeping's, for the floor).
    ///
    /// # Safety
    /// An open list; live textures the list keeps.
    unsafe fn table(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        written: Option<Image>,
        read: Option<Image>,
        sky: Option<Image>,
    ) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let slot = self.next.fetch_add(1, Ordering::Relaxed) % SLOTS * VIEWS;
        // SAFETY: the caller's; the slots are inside the heap.
        unsafe {
            let cpu = self.views.GetCPUDescriptorHandleForHeapStart();
            let gpu = self.views.GetGPUDescriptorHandleForHeapStart();
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + (slot * self.step) as usize };
            let mut view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: written.map_or(DXGI_FORMAT_R32_UINT, |w| w.format),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            if let Some(slices) = written.map(|w| w.slices).filter(|&slices| slices > 0) {
                view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                view.Anonymous.Texture3D = D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: slices };
            }
            host.device.CreateUnorderedAccessView(
                written.map(|w| w.image),
                None::<&ID3D12Resource>,
                Some(&view),
                at,
            );
            let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: read.map_or(DXGI_FORMAT_R32_FLOAT, |r| r.format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            };
            let beside = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + self.step as usize };
            host.device.CreateShaderResourceView(read.map(|r| r.image), Some(&seen), beside);
            let around = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: sky.map_or(DXGI_FORMAT_R16G16B16A16_FLOAT, |s| s.format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
                },
            };
            let third = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + 2 * self.step as usize };
            host.device.CreateShaderResourceView(sky.map(|s| s.image), Some(&around), third);
            list.SetDescriptorHeaps(&[Some(self.views.clone())]);
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(slot * self.step) }
        }
    }

    /// Records one pass: its root signature and pipeline, the constants,
    /// the table, the buffers, the dispatch and a barrier for what it
    /// wrote.
    ///
    /// # Safety
    /// An open list with the passes' view heap set (`Pass::table`).
    unsafe fn run(
        &self,
        host_list: &HostList,
        name: &str,
        stage: &Stage,
        constants: &[u32; set::CONSTANTS],
        table: D3D12_GPU_DESCRIPTOR_HANDLE,
        groups: [u32; 3],
    ) {
        // The pass is named among the list's pipelines (the slow
        // submission line, the device-removed report), and noted as a
        // command of its own in diagnostics builds.
        host_list.bind_pipeline(&format!("[ambient estimate: {name}]"), &stage.pso, &stage.root);
        #[cfg(feature = "diagnostics")]
        host_list.note_op(|| format!("estimate [ambient estimate: {name}] {}x{}x{}", groups[0], groups[1], groups[2]));
        let list = &host_list.list;
        // SAFETY: the caller's; buffers that live as the passes do.
        unsafe {
            list.SetComputeRootSignature(&stage.root);
            list.SetPipelineState(&stage.pso);
            list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.SetComputeRootDescriptorTable(1, table);
            for (parameter, buffer) in [&self.cells, &self.cells, &self.points, &self.points].into_iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(2 + parameter as u32, buffer.GetGPUVirtualAddress());
            }
            list.Dispatch(groups[0], groups[1], groups[2]);
        }
        uav_barrier(list);
    }

    /// The keeping: the cells' light from `radiance` (a shader resource of
    /// `size` texels in `list` now), then the slow mean and the floor's
    /// light (from `sky`, the sky dome as a shader resource, if the floor
    /// follows it: `given.sky`).
    fn keep(
        &self,
        host: &Host,
        list: &HostList,
        radiance: Image,
        size: [u32; 2],
        sky: Option<Image>,
        given: &set::Pass,
    ) {
        list.keep(&self.cells);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, ..*given });
        uav_barrier(l);
        // SAFETY: an open list; a live texture the caller's list keeps, in
        // a shader-resource state.
        unsafe {
            let table = self.table(host, l, None, Some(radiance), sky);
            self.run(list, "keep", &self.keep, &constants, table, [set::CX / 8, set::CY / 4, 1]);
            self.run(list, "mean", &self.mean, &constants, table, [1, 1, 1]);
        }
    }

    /// The raising: the cells placed by `depth`, the points' light, and the
    /// light image `light` (of `size` texels, in the unordered-access state
    /// in `list` now) given the estimate. Without `depth` (a shader
    /// resource of the same size), the slow mean alone.
    fn raise(
        &self,
        host: &Host,
        list: &HostList,
        light: Image,
        depth: Option<Image>,
        size: [u32; 2],
        given: &set::Pass,
    ) {
        list.keep(&self.cells);
        list.keep(&self.points);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, flat: depth.is_none(), ..*given });
        uav_barrier(l);
        // SAFETY: an open list; live textures the caller's list keeps, in
        // the states named.
        unsafe {
            let table = self.table(host, l, Some(light), depth, None);
            // (The probe may leave the gather out: the points stay.)
            if depth.is_some() && left_out() & set::NO_GATHER == 0 {
                self.run(list, "place", &self.place, &constants, table, [set::CX / 8, set::CY / 4, 1]);
                self.run(list, "gather", &self.gather, &constants, table, [set::PX / 8, set::PY / 2, 1]);
            }
            self.run(list, "raise", &self.raise, &constants, table, [size[0].div_ceil(8), size[1].div_ceil(8), 1]);
        }
    }
}

impl Pass {
    /// The air's bound: the volume `scattered` (of `size` texels and
    /// `slices` slices, in the unordered-access state in `list` now) holds
    /// no more than a multiple of the light around.
    fn air(&self, host: &Host, list: &HostList, scattered: Image, size: [u32; 2], given: &set::Pass) {
        list.keep(&self.cells);
        list.keep(&self.points);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, slices: scattered.slices, ..*given });
        uav_barrier(l);
        // SAFETY: an open list; a live texture the caller's list keeps, in
        // the unordered-access state.
        unsafe {
            let table = self.table(host, l, Some(scattered), None, None);
            let groups = [size[0].div_ceil(8), size[1].div_ceil(8), scattered.slices.div_ceil(2)];
            self.run(list, "air", &self.air, &constants, table, groups);
        }
    }

    /// The hold: the reflections in `image` (of `size` texels, in the
    /// unordered-access state in `list` now) are no brighter than a part of
    /// the light around.
    fn hold(&self, host: &Host, list: &HostList, image: Image, size: [u32; 2], given: &set::Pass) {
        list.keep(&self.cells);
        list.keep(&self.points);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, ..*given });
        uav_barrier(l);
        // SAFETY: an open list; a live texture the caller's list keeps, in
        // the unordered-access state.
        unsafe {
            let table = self.table(host, l, Some(image), None, None);
            self.run(list, "hold", &self.hold, &constants, table, [size[0].div_ceil(8), size[1].div_ceil(8), 1]);
        }
    }
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: AtomicBool = AtomicBool::new(false);

/// Raisings with a depth image so far: the points hold a light.
static RAISED: AtomicU64 = AtomicU64::new(0);

/// Keepings, raisings, bounds of the air and holds of reflections recorded
/// since the last frame line.
static RECORDED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// With the frame line, over `frames` frames: what the estimate is, to
/// compare with the constant stand-in's level in a run log. Read from the
/// buffers as the CPU sees them now (the GPU may be writing them: a
/// diagnostic, not a measurement of one frame). Nothing while the estimate
/// is off.
pub(crate) fn log(frames: f64) {
    let Some(share) = share() else {
        return;
    };
    let per_frame = [0, 1, 2, 3].map(|i| RECORDED[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let made = KEPT.load(Ordering::Relaxed).then(|| host().and_then(pass)).flatten();
    let seen = made.map(|pass| {
        // SAFETY: inside the buffers' mappings, which live as the passes do.
        let float4 = |at: usize, index: u32| unsafe { ((at + index as usize * 16) as *const [f32; 4]).read_volatile() };
        let rgb = |index: u32| {
            let v = float4(pass.cells_at, index);
            [v[0], v[1], v[2]]
        };
        let (mean, floor) = (rgb(set::CELLS), rgb(set::FLOOR_AT));
        let luminance = floor[0] * 0.2126 + floor[1] * 0.7152 + floor[2] * 0.0722;
        let points = (0..set::PX * set::PY).map(|i| [0, 1, 2].map(|colour| float4(pass.points_at, i * 3 + colour)[3]));
        let (points, written, under) = set::over_the_picture(points, share, luminance);
        set::Seen { mean, floor, points, written, under }
    });
    // The probe: the frames of the line above were recorded with the step
    // that is on now; the next lines' with the next, if this one is over.
    let step = probe().map(|each| set::probe_step(LINES.fetch_add(1, Ordering::Relaxed), each));
    tracing::info!("{}", set::frame_line(share, floor_share(), seen, per_frame, step));
}

/// What the passes are given now.
fn settings(share: f32) -> set::Pass {
    let mut shade = shade();
    if left_out() & set::NO_SHADE != 0 {
        shade[0] = 0.0;
    }
    set::Pass {
        level: share,
        shade,
        follow: set::FOLLOW,
        most: left(),
        size: [0, 0],
        flat: false,
        air: air_bound(),
        slices: 0,
        jump: settle().map_or(0.0, |s| s.jump),
        fresh: settle().map_or(0.0, |s| s.fresh),
        dt: 0.0,
        held: reflect_part(),
        mix: mix(),
        sky: [0.0, 0.0, 1.0],
        floor: [0.0; 3],
        sky_size: [0, 0],
        reflect_least: reflect_floor(),
    }
}

/// After the title's program that keeps the light visible surfaces send
/// out, in `list`: the image it wrote, `radiance`, is averaged into the
/// cells. Whether it recorded anything (the list's heaps, compute root
/// signature and pipeline are then ours).
pub(crate) fn keep(list: &HostList, radiance: This) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    if left_out() & set::NO_KEEP != 0 {
        return false;
    }
    let Some(copy) = crate::resource::host_copy(radiance) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let format = DXGI_FORMAT_R11G11B10_FLOAT;
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !viewed_as(&copy, d.Format, format) {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "no ambient estimate: the light visible surfaces send out is not an R11G11B10_FLOAT image"
            );
        });
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            "ambient estimate: the first light it keeps"
        );
    });
    list.keep(&copy.texture);
    // The program before wrote it through an unordered view: read as a
    // shader resource here, and moved back by the title's next use.
    if let Some(barrier) = list.move_texture(&copy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32) {
        // SAFETY: an open list and a live texture the list keeps.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    let radiance = Image { image: &copy.texture, format, slices: 0 };
    // The floor: a share of the constant stand-in's light as it is now,
    // from the sky dome if that light follows it and the title has drawn
    // it (read in the format its program draws it in).
    let (level, tint, follows, dome) = crate::ambient::light_now();
    let dome = dome.filter(|dome| {
        // SAFETY: a live resource.
        let d = unsafe { dome.texture.GetDesc() };
        d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
            && viewed_as(dome, d.Format, DXGI_FORMAT_R16G16B16A16_FLOAT)
    });
    let (sky, floor) = set::floor_of(floor_share(), level, tint, follows, dome.is_some());
    let dome = dome.filter(|_| sky[0] > 0.0);
    let mut sky_size = [0, 0];
    if let Some(dome) = &dome {
        // SAFETY: a live resource.
        let d = unsafe { dome.texture.GetDesc() };
        sky_size = [d.Width as u32, d.Height];
        list.keep(&dome.texture);
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0;
        if let Some(barrier) = list.move_texture(dome, read as u32) {
            // SAFETY: an open list and a live texture the list keeps.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        static FOLLOWS: std::sync::Once = std::sync::Once::new();
        FOLLOWS.call_once(|| {
            tracing::info!(
                share = floor_share(),
                gain = sky[0],
                least = sky[1],
                power = sky[2],
                width = d.Width,
                height = d.Height,
                "ambient estimate: its floor follows the sky dome"
            );
        });
    }
    let dome = dome.as_ref().map(|dome| Image { image: &dome.texture, format: DXGI_FORMAT_R16G16B16A16_FLOAT, slices: 0 });
    // The slow mean goes on by the time since the last keeping.
    let ms = frame_ms();
    let given = set::Pass {
        follow: set::follow(settle(), ms),
        dt: ms / 1000.0,
        sky,
        floor,
        sky_size,
        ..settings(share)
    };
    pass.keep(host, list, radiance, [d.Width as u32, d.Height], dome, &given);
    // And back to the state its program left it in: whatever comes after
    // us, in this list or the next, finds the image as it does without the
    // estimate.
    if let Some(barrier) = list.move_texture(&copy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32) {
        // SAFETY: an open list and a live texture the list keeps.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    KEPT.store(true, Ordering::Relaxed);
    RECORDED[0].fetch_add(1, Ordering::Relaxed);
    true
}

/// After the title's program that resolves the screen probes, in `list`:
/// what `light` holds is replaced by the estimate (or raised to it,
/// `VICTORY_AMBIENT_LEFT`). `depth` is the scene's depth the resolve read.
/// Whether it recorded anything (the list's heaps, compute root signature
/// and pipeline are then ours).
pub(crate) fn raise(list: &HostList, light: This, depth: Option<This>) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    // Nothing kept yet: the title's own output for a frame.
    if !KEPT.load(Ordering::Relaxed) || left_out() & set::NO_RAISE != 0 {
        return false;
    }
    let Some(light) = crate::resource::host_copy(light) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !(d.Format == DXGI_FORMAT_R32_TYPELESS || viewed_as(&light, d.Format, DXGI_FORMAT_R32_UINT))
    {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "no ambient estimate: the light image is not one a 32-bit integer view can be made of"
            );
        });
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the slow mean alone is written.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            share,
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            depth = depth.is_some(),
            "ambient estimate: the first image it writes"
        );
        if depth.is_none() {
            tracing::warn!(
                "ambient estimate: no depth image of the light's size: every texel gets the mean of the light around"
            );
        }
    });
    list.keep(&light.texture);
    if let Some(depth) = &depth {
        list.keep(depth);
    }
    let image = Image { image: &light.texture, format: DXGI_FORMAT_R32_UINT, slices: 0 };
    let depth = depth.as_ref().map(|d| Image { image: d, format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, slices: 0 });
    pass.raise(host, list, image, depth, [d.Width as u32, d.Height], &settings(share));
    RECORDED[1].fetch_add(1, Ordering::Relaxed);
    if depth.is_some() {
        RAISED.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// After the title's program that adds the lamps' light to what the air
/// scatters, in `list`: the volume it wrote, `scattered`, is bounded by a
/// multiple of the light around (`VICTORY_AMBIENT_AIR`; nothing while that
/// is unset). Whether it recorded anything (the list's heaps, compute root
/// signature and pipeline are then ours).
pub(crate) fn air(list: &HostList, scattered: This) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    if air_bound() <= 0.0 || !KEPT.load(Ordering::Relaxed) {
        return false;
    }
    let Some(copy) = crate::resource::host_copy(scattered) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let format = DXGI_FORMAT_R11G11B10_FLOAT;
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || !viewed_as(&copy, d.Format, format) {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "the air is not bounded: what it scatters is not an R11G11B10_FLOAT volume"
            );
        });
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    let slices = u32::from(d.DepthOrArraySize);
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            bound = air_bound(),
            width = d.Width,
            height = d.Height,
            slices,
            format = d.Format.0,
            "ambient estimate: the first volume of air it bounds"
        );
    });
    list.keep(&copy.texture);
    let scattered = Image { image: &copy.texture, format, slices };
    pass.air(host, list, scattered, [d.Width as u32, d.Height], &settings(share));
    RECORDED[2].fetch_add(1, Ordering::Relaxed);
    true
}

/// After the title's program that writes its reflections for the last time
/// in a frame, in `list`: those in `image` are held to a part of the light
/// around (`VICTORY_AMBIENT_REFLECT`; nothing if that is 0). Whether it
/// recorded anything (the list's heaps, compute root signature and pipeline
/// are then ours).
pub(crate) fn hold(list: &HostList, image: This) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    // The points' light is of the last raising: none before one.
    if reflect_part() <= 0.0 || RAISED.load(Ordering::Relaxed) == 0 || left_out() & set::NO_HOLD != 0 {
        return false;
    }
    let Some(copy) = crate::resource::host_copy(image) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !(d.Format == DXGI_FORMAT_R32_TYPELESS || viewed_as(&copy, d.Format, DXGI_FORMAT_R32_UINT))
    {
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            held = reflect_part(),
            least = reflect_floor(),
            width = d.Width,
            height = d.Height,
            "ambient estimate: reflections held to the light around"
        );
    });
    list.keep(&copy.texture);
    let reflections = Image { image: &copy.texture, format: DXGI_FORMAT_R32_UINT, slices: 0 };
    pass.hold(host, list, reflections, [d.Width as u32, d.Height], &settings(share));
    RECORDED[3].fetch_add(1, Ordering::Relaxed);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::transition;
    use set::reference::{self, Scene, V3, V4};

    /// The passes compile with the system's compiler and the host takes
    /// them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(host).expect("the passes");
    }

    /// A value as an unsigned float of `bits` mantissa bits and a 5-bit
    /// exponent (R11G11B10_FLOAT's parts); exact for the values used here.
    fn small_float(v: f32, bits: u32) -> u32 {
        if v <= 0.0 {
            return 0;
        }
        let e = v.log2().floor() as i32;
        let m = ((v / 2f32.powi(e) - 1.0) * (1 << bits) as f32).round() as u32;
        ((e + 15) as u32) << bits | m
    }

    fn r11g11b10(c: V3) -> u32 {
        small_float(c[0], 6) | small_float(c[1], 6) << 11 | small_float(c[2], 5) << 22
    }

    /// A texture of `format` holding `words` (one 32-bit word a texel),
    /// left in `state`; its upload buffer goes to `keep`.
    fn texture(
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        format: DXGI_FORMAT,
        size: [u32; 2],
        words: &[u32],
        state: D3D12_RESOURCE_STATES,
        keep: &mut Vec<ID3D12Resource>,
    ) -> ID3D12Resource {
        let texture = crate::composite::result_texture(&host.device, size[0].into(), size[1], format).unwrap();
        let pitch = (size[0] * 4).next_multiple_of(256);
        let (upload, data) = host.upload_buffer(u64::from(pitch) * u64::from(size[1])).unwrap();
        // SAFETY: a mapped upload buffer written in bounds; an open list
        // and copy locations naming live resources.
        unsafe {
            for y in 0..size[1] {
                let row = &words[(y * size[0]) as usize..((y + 1) * size[0]) as usize];
                std::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), data.add((y * pitch) as usize), row.len() * 4);
            }
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyTextureRegion(&whole(&texture), 0, 0, 0, &rows(&upload, format, size, pitch), None);
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
        keep.push(upload);
        texture
    }

    fn whole(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        }
    }

    fn rows(buffer: &ID3D12Resource, format: DXGI_FORMAT, size: [u32; 2], pitch: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: size[0],
                        Height: size[1],
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        }
    }

    /// The `float4`s of a buffer of ours the CPU sees, from `first` on.
    fn floats(at: usize, first: u32, count: u32) -> Vec<V4> {
        // SAFETY: inside the buffer's mapping; the submission has run.
        (0..count).map(|i| unsafe { ((at + ((first + i) * 16) as usize) as *const V4).read_volatile() }).collect()
    }

    fn near(a: f32, b: f32, part: f32) -> bool {
        (a - b).abs() <= part * a.abs().max(b.abs()) + 1e-5
    }

    /// The scene of the GPU tests: a floor 0.6 m under the eye up to a wall
    /// 3 m ahead, whose left half is lit orange and whose right half is
    /// dim blue, the floor grey; one texel in sixteen brighter than any
    /// light should count for.
    const LIGHT: [u32; 2] = [256, 144];
    const RADIANCE: [u32; 2] = [128, 72];

    fn depth_at(_: u32, y: u32) -> f32 {
        let down = ((y as f32 + 0.5) / LIGHT[1] as f32 * 2.0 - 1.0) * 0.45;
        0.1 / if down > 0.0 { (0.6 / down).min(3.0) } else { 3.0 }
    }

    fn radiance_at(x: u32, y: u32) -> V3 {
        if x % 4 == 1 && y % 4 == 2 {
            [1024.0, 512.0, 256.0]
        } else if y >= RADIANCE[1] * 13 / 18 {
            [0.25, 0.25, 0.25]
        } else if x < RADIANCE[0] / 2 {
            [2.0, 1.0, 0.25]
        } else {
            [0.0625, 0.125, 0.5]
        }
    }

    /// Runs a keeping and a raising on the test scene with `set`, the light
    /// image holding `was` before: the cells' light, the slow mean, the
    /// cells' places, the points and the light image after.
    #[allow(clippy::type_complexity)]
    fn run(host: &Host, pass: &Pass, given: &set::Pass, was: V3, depth: bool) -> (Vec<V4>, V4, Vec<V4>, Vec<V4>, Vec<u32>, V4) {
        let run = host.new_list().unwrap();
        let l = &run.list;
        let mut keep = Vec::new();
        // The sky dome, where the floor follows it.
        let given = &set::Pass { sky_size: SKY, ..*given };
        let dome = (given.sky[0] > 0.0).then(|| sky_texture(host, l, &mut keep));
        let dome = dome.as_ref().map(|dome| Image { image: dome, format: DXGI_FORMAT_R16G16B16A16_FLOAT, slices: 0 });
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let texels = |size: [u32; 2], at: &dyn Fn(u32, u32) -> u32| -> Vec<u32> {
            (0..size[0] * size[1]).map(|i| at(i % size[0], i / size[0])).collect()
        };
        let radiance = texture(
            host,
            l,
            DXGI_FORMAT_R11G11B10_FLOAT,
            RADIANCE,
            &texels(RADIANCE, &|x, y| r11g11b10(radiance_at(x, y))),
            read,
            &mut keep,
        );
        let scene_depth = texture(
            host,
            l,
            DXGI_FORMAT_R32_FLOAT,
            LIGHT,
            &texels(LIGHT, &|x, y| depth_at(x, y).to_bits()),
            read,
            &mut keep,
        );
        let light = texture(
            host,
            l,
            DXGI_FORMAT_R32_UINT,
            LIGHT,
            &texels(LIGHT, &|_, _| reference::pack(was)),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            &mut keep,
        );
        pass.keep(host, &run, Image { image: &radiance, format: DXGI_FORMAT_R11G11B10_FLOAT, slices: 0 }, RADIANCE, dome, given);
        pass.raise(
            host,
            &run,
            Image { image: &light, format: DXGI_FORMAT_R32_UINT, slices: 0 },
            depth.then_some(Image { image: &scene_depth, format: DXGI_FORMAT_R32_FLOAT, slices: 0 }),
            LIGHT,
            given,
        );
        let pitch = (LIGHT[0] * 4).next_multiple_of(256);
        let (back, data) = host.readback_buffer(u64::from(pitch) * u64::from(LIGHT[1])).unwrap();
        // SAFETY: an open list and live resources.
        unsafe {
            l.ResourceBarrier(&[transition(&light, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyTextureRegion(&rows(&back, DXGI_FORMAT_R32_UINT, LIGHT, pitch), 0, 0, 0, &whole(&light), None);
        }
        run.close();
        host.execute(&[&run]);
        // SAFETY: the copy has run; inside the mapped buffer.
        let written = texels(LIGHT, &|x, y| unsafe { data.add((y * pitch + x * 4) as usize).cast::<u32>().read_volatile() });
        let cells = floats(pass.cells_at, 0, set::CELLS);
        let mean = floats(pass.cells_at, set::CELLS, 1)[0];
        let places = floats(pass.cells_at, set::CELLS + 1, set::CELLS);
        let points = floats(pass.points_at, 0, set::PX * set::PY * 3);
        let floor = floats(pass.cells_at, set::FLOOR_AT, 1)[0];
        (cells, mean, places, points, written, floor)
    }

    const BASE: set::Pass = set::Pass {
        level: 0.75,
        shade: [1.0, 0.6, 0.1, 0.45],
        follow: 0.25,
        most: 0.0,
        size: [0, 0],
        flat: false,
        air: 0.0,
        slices: 0,
        jump: 0.0,
        fresh: 0.0,
        dt: 0.25,
        held: 0.5,
        mix: set::MIX,
        sky: [0.0, 0.0, 1.0],
        floor: [0.0; 3],
        sky_size: [0, 0],
        reflect_least: 0.0,
    };

    /// The sky dome of the GPU tests: 16x8, a bright blue half (with texels
    /// brighter than the 8 a texel counts for) and a dim warm half.
    const SKY: [u32; 2] = [16, 8];

    fn sky_at(x: u32, y: u32) -> V3 {
        if x < 8 { [2.0, 4.0, if y == 3 { 32.0 } else { 6.0 }] } else { [0.5, 0.25, 0.125] }
    }

    /// A value as a 16-bit float; exact for the values used here.
    fn half(v: f32) -> u16 {
        if v <= 0.0 {
            return 0;
        }
        let e = v.log2().floor() as i32;
        ((e + 15) as u16) << 10 | ((v / 2f32.powi(e) - 1.0) * 1024.0).round() as u16
    }

    /// The sky dome as an R16G16B16A16_FLOAT texture, left readable by
    /// shaders; its upload buffer goes to `keep`.
    fn sky_texture(host: &Host, list: &ID3D12GraphicsCommandList, keep: &mut Vec<ID3D12Resource>) -> ID3D12Resource {
        let format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let texture = crate::composite::result_texture(&host.device, SKY[0].into(), SKY[1], format).unwrap();
        let pitch = (SKY[0] * 8).next_multiple_of(256);
        let (upload, data) = host.upload_buffer(u64::from(pitch) * u64::from(SKY[1])).unwrap();
        // SAFETY: a mapped upload buffer written in bounds; an open list
        // and copy locations naming live resources.
        unsafe {
            for y in 0..SKY[1] {
                for x in 0..SKY[0] {
                    let [r, g, b] = sky_at(x, y).map(half);
                    data.add((y * pitch + x * 8) as usize).cast::<[u16; 4]>().write_unaligned([r, g, b, half(1.0)]);
                }
            }
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyTextureRegion(&whole(&texture), 0, 0, 0, &rows(&upload, format, SKY, pitch), None);
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        }
        keep.push(upload);
        texture
    }

    /// The dome's mean as the `MEAN` pass takes it: 8x8 texels, each held
    /// to 8.
    fn sky_mean() -> V3 {
        let mut sum = [0.0f32; 3];
        for y in 0..8 {
            for x in 0..8 {
                let there = sky_at((x * 2 + 1) * SKY[0] / 16, (y * 2 + 1) * SKY[1] / 16);
                for k in 0..3 {
                    sum[k] += there[k].clamp(0.0, 8.0);
                }
            }
        }
        sum.map(|v| v / 64.0)
    }

    /// The passes on the GPU give what the arithmetic on the CPU gives
    /// (`set::reference`): the cells' light and places, the slow mean, the
    /// points, and the light image, with and without what the resolve left.
    #[test]
    fn the_passes_match_the_arithmetic() {
        let Some(host) = host() else {
            return;
        };
        let pass = make(host).expect("the passes");
        let scene = Scene { depth: &depth_at, size: LIGHT, near: 0.1, tan_y: 0.45 };
        let base = BASE;
        let want_cells = reference::keep(&radiance_at, RADIANCE);
        let want_places = scene.place();
        let want_points = scene.gather(&want_cells, &want_places);
        let mut slow = [0.0f32; 4];
        for (given, was, depth) in [
            (base, [9.0, 0.0, 9.0], true),
            (set::Pass { most: 4.0, shade: [0.0, 0.6, 0.1, 0.45], ..base }, [9.0, 0.0, 9.0], true),
            // 0.5 s after a jump of 2 fresh seconds: what was left counts
            // for 1.75 times the estimate.
            (set::Pass { most: 4.0, fresh: 2.0, jump: 3.0, ..base }, [9.0, 0.0, 9.0], true),
            // Another mix: half the slow mean, no turn, narrower bounds.
            (set::Pass { mix: [0.5, 0.0, 0.5, 1.5], ..base }, [0.0; 3], true),
            // A floor of a constant light, and one that follows the sky
            // dome (also with what the resolve left, and without depth).
            (set::Pass { floor: [0.4, 0.3, 0.2], ..base }, [0.0; 3], true),
            (set::Pass { sky: [0.108, 0.05, 0.4], ..base }, [0.0; 3], true),
            (set::Pass { sky: [0.36, 0.05, 0.4], most: 4.0, ..base }, [9.0, 0.0, 9.0], true),
            (set::Pass { sky: [0.36, 0.05, 0.4], ..base }, [0.0; 3], false),
            (base, [0.0; 3], false),
        ] {
            let (cells, mean, places, points, written, floor) = run(host, &pass, &given, was, depth);
            // The floor's light: the constant one, or from the dome.
            let least = reference::floor_light(&given, sky_mean());
            assert!((0..3).all(|k| near(floor[k], least[k], 2e-3)), "floor: {floor:?} {least:?}");
            for (i, (got, want)) in cells.iter().zip(&want_cells).enumerate() {
                assert!((0..3).all(|k| near(got[k], want[k], 1e-3)), "cell {i}: {got:?} {want:?}");
            }
            // The second keeping goes a share of the way from the first.
            slow = reference::mean(&want_cells, slow, given.follow, given.jump, given.dt);
            assert!((0..4).all(|k| near(mean[k], slow[k], 1e-3)) && mean[3] > 0.0, "{mean:?} {slow:?}");
            if !depth {
                // The slow mean alone, everywhere, or the floor where that
                // is more.
                let flat = reference::pack([0, 1, 2].map(|k| (slow[k] * given.level).max(least[k])));
                let odd = written.iter().filter(|&&w| !close(reference::unpack(w), reference::unpack(flat))).count();
                assert_eq!(odd, 0, "texels that are not the slow mean");
                continue;
            }
            for (i, (got, want)) in places.iter().zip(&want_places).enumerate() {
                assert!((0..4).all(|k| near(got[k], want[k], 1e-3)), "place {i}: {got:?} {want:?}");
            }
            for (i, want) in want_points.iter().enumerate() {
                for (colour, want) in want.iter().enumerate() {
                    let got = points[i * 3 + colour];
                    // Directions are differences of sums: to a part of the mean.
                    assert!(
                        near(got[3], want[3], 2e-3) && (0..3).all(|k| (got[k] - want[k]).abs() <= 2e-3 * want[3] + 1e-5),
                        "point {i} colour {colour}: {got:?} {want:?}"
                    );
                }
            }
            let want_points: Vec<[V4; 3]> = want_points.clone();
            let mut odd = 0;
            for y in 0..LIGHT[1] {
                for x in 0..LIGHT[0] {
                    let want = scene.raise_over([x as i32, y as i32], &want_points, slow, &given, reference::unpack(reference::pack(was)), least);
                    let got = reference::unpack(written[(y * LIGHT[0] + x) as usize]);
                    if !close(got, reference::unpack(reference::pack(want))) {
                        odd += 1;
                    }
                }
            }
            // A texel at the edge of two points' cells, or with a shade
            // sample on a depth edge, may round the other way on the GPU.
            let most = LIGHT[0] * LIGHT[1] / 100;
            assert!(odd <= most, "{odd} texels differ (most {most}), shade {}", given.shade[0]);
        }
    }

    /// The air's pass reads and writes an R11G11B10_FLOAT volume through
    /// one unordered view, as the title's own program before it does, and
    /// bounds every texel as the arithmetic on the CPU does: by the slow
    /// mean near the eye, by the light of the point seen far away.
    #[test]
    fn the_air_is_bounded_on_the_gpu() {
        let Some(host) = host() else {
            return;
        };
        let pass = make(host).expect("the passes");
        // The light around, as a keeping and a raising leave it.
        let (_, mean, _, points, _, _) = run(host, &pass, &BASE, [0.0; 3], true);
        let size = [40u32, 24, 6];
        let scattered_at = |x: u32, y: u32, z: u32| -> V3 {
            match (x + y + z) % 3 {
                0 => [64.0, 32.0, 16.0],
                1 => [0.5, 0.25, 0.125],
                _ => [0.03125, 0.0625, 0.03125],
            }
        };
        let mut volume: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                        Width: size[0].into(),
                        Height: size[1],
                        DepthOrArraySize: size[2] as u16,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R11G11B10_FLOAT,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut volume,
                )
                .unwrap();
        }
        let volume = volume.unwrap();
        let pitch = (size[0] * 4).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(size[1]) * u64::from(size[2]);
        let (upload, data) = host.upload_buffer(bytes).unwrap();
        let (back, read) = host.readback_buffer(bytes).unwrap();
        let at = |x: u32, y: u32, z: u32| ((z * size[1] + y) * pitch + x * 4) as usize;
        let slices = |buffer: &ID3D12Resource| {
            let mut rows = rows(buffer, DXGI_FORMAT_R11G11B10_FLOAT, [size[0], size[1]], pitch);
            rows.Anonymous.PlacedFootprint.Footprint.Depth = size[2];
            rows
        };
        let list = host.new_list().unwrap();
        let l = &list.list;
        let given = set::Pass { air: 3.0, ..BASE };
        // SAFETY: mapped buffers written and read in bounds; an open list
        // and copy locations naming live resources.
        unsafe {
            for z in 0..size[2] {
                for y in 0..size[1] {
                    for x in 0..size[0] {
                        data.add(at(x, y, z)).cast::<u32>().write(r11g11b10(scattered_at(x, y, z)));
                    }
                }
            }
            l.CopyTextureRegion(&whole(&volume), 0, 0, 0, &slices(&upload), None);
            l.ResourceBarrier(&[transition(&volume, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            let scattered = Image { image: &volume, format: DXGI_FORMAT_R11G11B10_FLOAT, slices: size[2] };
            pass.air(host, &list, scattered, [size[0], size[1]], &given);
            l.ResourceBarrier(&[transition(&volume, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyTextureRegion(&slices(&back), 0, 0, 0, &whole(&volume), None);
        }
        list.close();
        host.execute(&[&list]);
        let part = |word: u32, shift: u32, bits: u32| {
            let v = word >> shift & ((1 << (5 + bits)) - 1);
            let (e, m) = (v >> bits, v & ((1 << bits) - 1));
            if e == 0 { m as f32 / (1 << bits) as f32 * 2f32.powi(-14) } else { (1.0 + m as f32 / (1 << bits) as f32) * 2f32.powi(e as i32 - 15) }
        };
        let (mut bounded, mut kept) = (0, 0);
        for z in 0..size[2] {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    // SAFETY: the copy has run; inside the mapped buffer.
                    let word = unsafe { read.add(at(x, y, z)).cast::<u32>().read_volatile() };
                    let got = [part(word, 0, 6), part(word, 11, 6), part(word, 22, 5)];
                    let q = [0, 1].map(|k| {
                        (((([x, y][k] as f32 + 0.5) / size[k] as f32) * [set::PX, set::PY][k] as f32) as u32).min([set::PX, set::PY][k] - 1)
                    });
                    let point = ((q[1] * set::PX + q[0]) * 3) as usize;
                    let seen_there = [points[point][3], points[point + 1][3], points[point + 2][3]];
                    let was = scattered_at(x, y, z);
                    let want = reference::bound_air(was, 3.0, [mean[0], mean[1], mean[2]], seen_there, z as f32 / size[2] as f32);
                    // (6 and 5 bits of each part are kept.)
                    assert!((0..3).all(|k| near(got[k], want[k], 0.04)), "air at {x},{y},{z}: {got:?} {want:?} (was {was:?})");
                    if want == was { kept += 1 } else { bounded += 1 }
                }
            }
        }
        assert!(bounded > 0 && kept > 0, "{bounded} bounded, {kept} kept");
    }

    /// The hold's pass keeps every reflection to a part of the light around
    /// the point seen there, as the arithmetic on the CPU does; with the
    /// reflection floor also no darker than another part of it.
    #[test]
    fn reflections_are_held_on_the_gpu() {
        let Some(host) = host() else {
            return;
        };
        let pass = make(host).expect("the passes");
        let (_, mean, _, points, _, _) = run(host, &pass, &BASE, [0.0; 3], true);
        // A reflection image of another size than the light's.
        let size = [96u32, 54];
        let there_at = |x: u32, y: u32| -> V3 {
            match (x + 2 * y) % 4 {
                0 => [40.0, 30.0, 20.0],
                1 => [0.75, 0.5, 0.25],
                2 => [0.0078125, 0.015625, 0.0078125],
                _ => [0.0; 3],
            }
        };
        for given in [BASE, set::Pass { reflect_least: 0.15, ..BASE }] {
            let list = host.new_list().unwrap();
            let l = &list.list;
            let mut keep = Vec::new();
            let words: Vec<u32> = (0..size[0] * size[1]).map(|i| reference::pack(there_at(i % size[0], i / size[0]))).collect();
            let image = texture(host, l, DXGI_FORMAT_R32_UINT, size, &words, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mut keep);
            pass.hold(host, &list, Image { image: &image, format: DXGI_FORMAT_R32_UINT, slices: 0 }, size, &given);
            let pitch = (size[0] * 4).next_multiple_of(256);
            let (back, data) = host.readback_buffer(u64::from(pitch) * u64::from(size[1])).unwrap();
            // SAFETY: an open list and live resources.
            unsafe {
                l.ResourceBarrier(&[transition(&image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                l.CopyTextureRegion(&rows(&back, DXGI_FORMAT_R32_UINT, size, pitch), 0, 0, 0, &whole(&image), None);
            }
            list.close();
            host.execute(&[&list]);
            let (mut held, mut kept, mut raised) = (0, 0, 0);
            for y in 0..size[1] {
                for x in 0..size[0] {
                    // SAFETY: the copy has run; inside the mapped buffer.
                    let got = reference::unpack(unsafe { data.add((y * pitch + x * 4) as usize).cast::<u32>().read_volatile() });
                    let q = [0, 1].map(|k| {
                        (((([x, y][k] as f32 + 0.5) / size[k] as f32) * [set::PX, set::PY][k] as f32) as u32).min([set::PX, set::PY][k] - 1)
                    });
                    let point = ((q[1] * set::PX + q[0]) * 3) as usize;
                    let seen_there = [points[point][3], points[point + 1][3], points[point + 2][3]];
                    let was = reference::unpack(reference::pack(there_at(x, y)));
                    let want = reference::hold(was, &given, [mean[0], mean[1], mean[2]], seen_there);
                    assert!(
                        close(got, reference::unpack(reference::pack(want))),
                        "reflection at {x},{y} (floor {}): {got:?} {want:?} (was {was:?})",
                        given.reflect_least
                    );
                    let more = (0..3).all(|k| want[k] >= was[k]);
                    if want == was { kept += 1 } else if more { raised += 1 } else { held += 1 }
                }
            }
            assert!(held > 0 && kept > 0, "{held} held, {kept} kept");
            // Unset nothing is raised; set, the black ones are.
            assert_eq!(raised > 0, given.reflect_least > 0.0, "{raised} raised");
        }
    }

    /// What each pass costs on this GPU at the title's sizes (the light
    /// image 2560x1440, the radiance image 1280x720), each alone in lists
    /// of its own, eight to a list, on a dark and on a bright picture.
    /// Prints the milliseconds; run by name with `--ignored --nocapture`.
    #[test]
    #[ignore = "a measurement: run by name"]
    fn the_passes_cost() {
        let Some(host) = host() else {
            return;
        };
        let pass = make(host).expect("the passes");
        const SIZE: [u32; 2] = [2560, 1440];
        const HALF: [u32; 2] = [1280, 720];
        const EACH: u32 = 8;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let depth_at = |_: u32, y: u32| {
            let down = ((y as f32 + 0.5) / SIZE[1] as f32 * 2.0 - 1.0) * 0.45;
            (0.1 / if down > 0.0 { (0.6 / down).min(3.0) } else { 3.0 }).to_bits()
        };
        let texels = |size: [u32; 2], at: &dyn Fn(u32, u32) -> u32| -> Vec<u32> {
            (0..size[0] * size[1]).map(|i| at(i % size[0], i / size[0])).collect()
        };
        let mut keep = Vec::new();
        let setup = host.new_list().unwrap();
        let depth = texture(host, &setup.list, DXGI_FORMAT_R32_FLOAT, SIZE, &texels(SIZE, &depth_at), read, &mut keep);
        let light = texture(host, &setup.list, DXGI_FORMAT_R32_UINT, SIZE, &texels(SIZE, &|_, _| 0), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mut keep);
        let dark = texture(host, &setup.list, DXGI_FORMAT_R11G11B10_FLOAT, HALF, &texels(HALF, &|_, _| r11g11b10([0.001953125; 3])), read, &mut keep);
        let bright = texture(host, &setup.list, DXGI_FORMAT_R11G11B10_FLOAT, HALF, &texels(HALF, &|x, y| r11g11b10(radiance_at(x % 128, y % 72))), read, &mut keep);
        setup.close();
        host.execute(&[&setup]);
        let light = Image { image: &light, format: DXGI_FORMAT_R32_UINT, slices: 0 };
        let depth = Image { image: &depth, format: DXGI_FORMAT_R32_FLOAT, slices: 0 };
        // The milliseconds one recording of `record` takes, `EACH` to a list.
        let time = |record: &dyn Fn(&HostList)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let list = host.new_list().unwrap();
                for _ in 0..EACH {
                    record(&list);
                }
                list.close();
                let started = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(started.elapsed().as_secs_f64() * 1e3 / f64::from(EACH));
            }
            best
        };
        let empty = time(&|_| {});
        let constants = |given: &set::Pass| set::constants(&set::Pass { size: SIZE, ..*given });
        let unshaded = set::Pass { shade: [0.0, 0.6, 0.1, 0.45], ..BASE };
        for (name, radiance) in [("dark", &dark), ("bright", &bright)] {
            let radiance = Image { image: radiance, format: DXGI_FORMAT_R11G11B10_FLOAT, slices: 0 };
            let keeping = time(&|list| pass.keep(host, list, radiance, HALF, None, &BASE));
            // SAFETY (the three below): an open list; live textures in the
            // states the passes need.
            let placing = time(&|list| unsafe {
                let table = pass.table(host, &list.list, None, Some(depth), None);
                pass.run(list, "place", &pass.place, &constants(&BASE), table, [set::CX / 8, set::CY / 4, 1]);
            });
            let gathering = time(&|list| unsafe {
                let table = pass.table(host, &list.list, None, Some(depth), None);
                pass.run(list, "gather", &pass.gather, &constants(&BASE), table, [set::PX / 8, set::PY / 2, 1]);
            });
            let raise = |given: set::Pass| {
                time(&|list| unsafe {
                    let table = pass.table(host, &list.list, Some(light), Some(depth), None);
                    pass.run(list, "raise", &pass.raise, &constants(&given), table, [SIZE[0] / 8, SIZE[1] / 8, 1]);
                })
            };
            let (shaded, plain) = (raise(BASE), raise(unshaded));
            let holding = time(&|list| pass.hold(host, list, light, SIZE, &BASE));
            println!(
                "ambient estimate cost, {name} picture, ms a recording (an empty list: {empty:.2}): keep and mean \
                 {keeping:.2}, place {placing:.2}, gather {gathering:.2}, raise {shaded:.2} (without the creases' \
                 shade {plain:.2}), hold {holding:.2}"
            );
        }
    }

    /// Two packed lights within a fiftieth of the brighter one's top part
    /// (the packing keeps 9 bits of it).
    fn close(a: V3, b: V3) -> bool {
        let top = a.iter().chain(&b).fold(0.0f32, |m, &v| m.max(v));
        (0..3).all(|k| (a[k] - b[k]).abs() <= top / 50.0 + 1e-5)
    }
}
