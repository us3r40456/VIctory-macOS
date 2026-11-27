//! A least amount of indirect light: a stand-in for the title's ray-traced
//! one, from Belobug's fork (`victory-gpu/src/ambient.rs` at 4f62c2a). Off
//! unless `VICTORY_AMBIENT=1`.
//!
//! Grand Theft Auto VI's indirect diffuse light comes from probes it fills
//! by tracing rays. Where those do not run, or run on nothing, its program
//! that resolves them (`CS_ScreenProbes_TemporalResolve_*`) leaves next to
//! nothing, and whatever the sun does not reach directly is black. Right
//! after that program, its output is replaced by a constant (it used to be
//! a floor under what the program wrote, but that is garbage: see the
//! shader). The constant is the same everywhere: the light in these images
//! does not go with the title's exposure (well-lit surfaces hold about the
//! same in a shot at -5.5 EV and in one at -9.75). The title's own lighting
//! then applies it with the surfaces' colors. Replaced, not added: the
//! result is the next frame's history.
//!
//! shortcut: the same light everywhere, from everywhere; gone once rays run.
//!
//! What differs from the fork's file, for this line (Wine's HLSL compiler
//! and D3DMetal; one host texture per texture memory):
//! - Off by default, behind `VICTORY_AMBIENT=1`; the level is
//!   `VICTORY_AMBIENT_LEVEL` (0.6 by default). Off, nothing here runs: the
//!   title's containers are not searched for names and no pass is made.
//! - The pass's root signature is built in code and its shader keeps to
//!   what Wine's compiler takes (`crate::ambient_set::SOURCE`).
//! - Programs are known per title pipeline (`crate::pipeline::ambient_role`,
//!   from the names found when their containers are first seen), not by
//!   flags on every shader container.
//! - An image is taken if a 32-bit integer view can be made of its host
//!   texture: on this line a texture memory's host texture has the format
//!   of the first title texture there and is castable to the others, where
//!   the fork made such textures `R32_UINT`.
//! - Its view heap is made with other threads' submissions held off
//!   (`Host::quiet`, VIctory-macOS issue #31).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::ambient_set as set;
use crate::host::{Host, HostList, host};
use crate::object::This;
use crate::resource::HostTexture;

pub(crate) use set::{DEPTH, DRAWS_SKY, ENDS_REFLECTIONS, LIGHT, REFLECTIONS, RESOLVES};

/// Whether the stand-in is on: `VICTORY_AMBIENT=1` (off when unset, empty,
/// `0` or `off`). Read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| set::is_on(var("VICTORY_AMBIENT").as_deref()))
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Says what is on, once, when the host device is ready.
pub(crate) fn announce() {
    // The estimate in its place (`VICTORY_AMBIENT=estimate`).
    if crate::ambient_estimate::on() {
        crate::ambient_estimate::announce();
    } else if on() {
        tracing::info!("{}", set::start_line(level(), shade(), keep(), follow()));
    }
}

/// What the containers seen so far are to the stand-in, by their hash as
/// logged: only those of programs it follows (a handful).
static FOLLOWED: Mutex<Vec<(String, u8)>> = Mutex::new(Vec::new());

/// The title passed shader container `container` for the first time, `hash`
/// as logged: remembered if its program is one the stand-in follows.
/// Nothing while the stand-in is off.
pub(crate) fn note_program(hash: &str, container: &[u8]) {
    if !on() {
        return;
    }
    let role = set::role_of(container);
    // The estimate's own programs (none while it is off).
    let more = crate::ambient_estimate::role_of(container);
    if role | more != 0 {
        let name = if more != 0 { crate::ambient_estimate::role_name(more) } else { set::role_name(role) };
        tracing::info!(hash, "a program the ambient light follows: it {name}");
        FOLLOWED.lock().unwrap_or_else(|e| e.into_inner()).push((hash.to_owned(), role | more));
    }
}

/// What a title pipeline of the shaders with these hashes is to the
/// stand-in (bits of `RESOLVES` and its siblings; 0: nothing, and always
/// while the stand-in is off).
pub(crate) fn role<'a>(hashes: impl Iterator<Item = &'a str>) -> u8 {
    if !on() {
        return 0;
    }
    let followed = FOLLOWED.lock().unwrap_or_else(|e| e.into_inner());
    if followed.is_empty() {
        return 0;
    }
    hashes
        .filter_map(|hash| followed.iter().find(|(h, _)| h == hash))
        .fold(0, |role, (_, r)| role | r)
}

/// The level; 0 is no indirect diffuse light at all.
fn level() -> f32 {
    static LEVEL: OnceLock<f32> = OnceLock::new();
    *LEVEL.get_or_init(|| set::level(var("VICTORY_AMBIENT_LEVEL").as_deref()))
}

/// The light's colour: warm white, or what `VICTORY_AMBIENT_TINT=r,g,b`
/// gives. With `VICTORY_AMBIENT_FILE=<file>` (diagnostics builds) the file's
/// `level r g b` is read again twice a second, to tune both by eye in a
/// running game.
fn tint() -> [f32; 3] {
    if let Some((_, tint)) = *LIVE.lock().unwrap_or_else(|e| e.into_inner()) {
        return tint;
    }
    static SET: OnceLock<[f32; 3]> = OnceLock::new();
    *SET.get_or_init(|| set::tint(var("VICTORY_AMBIENT_TINT").as_deref()))
}

/// The level and colour last read from `VICTORY_AMBIENT_FILE`, if any.
static LIVE: Mutex<Option<(f32, [f32; 3])>> = Mutex::new(None);

/// The most a reflection may hold, as a part of the ambient light's level:
/// `VICTORY_AMBIENT_REFLECT=<part>` (0.5 by default; 0 leaves the title's
/// reflections alone); a ninth number in the file sets it in a running game.
///
/// Without rays the reflections are the sky's, unoccluded, wherever a
/// surface is glossy: indoors a wooden floor, a counter and glass tops wore
/// a white glaze five to ten times brighter than anything in the room (the
/// fork's run ciel1, 2026-11-25; it did not move with the ambient level).
/// They are held to a part of the light around, after the program that
/// writes them for the last time in a frame (`set::REFLECTED`); the
/// lighting programs read that image next.
/// shortcut: stands in for what the title's traced reflections would see.
fn keep() -> f32 {
    if let Some(live) = *KEEP.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::keep(var("VICTORY_AMBIENT_REFLECT").as_deref()))
}

static KEEP: Mutex<Option<f32>> = Mutex::new(None);

/// The least a reflection holds, as a part of the light around: a tenth
/// number in the file (0, the default, leaves dark reflections dark).
/// shortcut: without rays, what should mirror the sky (still water) sees
/// nothing and is black; this gives it a part of the ambient light instead.
static LEAST: Mutex<f32> = Mutex::new(0.0);

/// The sky dome the title's sky program (`set::SKY`) last drew into: the
/// host copy, held, since the title's object may go at any time.
///
/// The program draws twice a frame, on two lists recorded at once: the dome
/// (128x128, R16G16B16A16_FLOAT) and another image of the sky (R11G11B10).
/// Following whichever was recorded last made the light jump between the
/// two, three times apart, from frame to frame (the fork's run don1,
/// 2026-11-25).
static SKY_IMAGE: Mutex<Option<std::sync::Arc<HostTexture>>> = Mutex::new(None);

/// The sky program draws into the title texture `image`.
pub(crate) fn sky_drawn(image: usize) {
    let Some(copy) = crate::resource::host_copy(image as This) else { return };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !viewed_as(&copy, d.Format, DXGI_FORMAT_R16G16B16A16_FLOAT)
    {
        return;
    }
    let mut held = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
    if !held.as_ref().is_some_and(|h| std::sync::Arc::ptr_eq(h, &copy)) {
        *held = Some(copy);
    }
}

/// What the constant light is made of now, for the floor under the
/// estimate (`crate::ambient_estimate`): its level and colour, the gain,
/// least level and power with which it follows the sky (if it does), and
/// the sky image the title last drew (if any).
pub(crate) fn light_now() -> (f32, [f32; 3], Option<[f32; 3]>, Option<std::sync::Arc<HostTexture>>) {
    let sky = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    (live_level().unwrap_or_else(level), tint(), follow(), sky)
}

/// The light follows the sky: its level and colour are the sky image's mean
/// times a gain, over a least bluish level. Hand-set levels that looked
/// right in the fork were 0.07 at night, 0.3 at dusk and 1.4 at noon (runs
/// lum1, pr1, av1, 2026-11-25), with sky means of 0.001 at night and 0.65
/// at dawn; one level for all hours either drowned the title's own shadows
/// or left what the sun does not reach black. On unless
/// `VICTORY_AMBIENT_SKY=off`, which keeps the constant; `gain,least,power`
/// sets it; a file of two or three numbers, or six or seven with the
/// shading's, does so in a running game.
/// shortcut: the sky's mean stands in for the light the title's traced
/// probes would gather; indoors it is the sky's too.
///
/// Gain, least level and power: the level is `gain * mean^power + least`.
/// The power is under one because the title's exposure follows the light:
/// the sky's mean goes from 0.001 at night to several units by day, the
/// levels that look right only from 0.07 to 1.4.
fn follow() -> Option<[f32; 3]> {
    if let Some(live) = *FOLLOW.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<Option<[f32; 3]>> = OnceLock::new();
    *SET.get_or_init(|| set::follow(var("VICTORY_AMBIENT_SKY").as_deref()))
}

static FOLLOW: Mutex<Option<Option<[f32; 3]>>> = Mutex::new(None);

/// How much less of the light reaches creases and corners, worked out from
/// the depth image alone (`set::shade`, `VICTORY_AMBIENT_SHADE=s,r,n,t`).
/// The file's values after its first four replace them.
/// shortcut: stands in for the occlusion the title's traced probes carry.
fn shade() -> [f32; 4] {
    if let Some(live) = *SHADE.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| set::shade(var("VICTORY_AMBIENT_SHADE").as_deref()))
}

static SHADE: Mutex<Option<[f32; 4]>> = Mutex::new(None);

/// The level from `VICTORY_AMBIENT_FILE`, if it gave one; starts the thread
/// that reads the file (diagnostics builds).
fn live_level() -> Option<f32> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let Some(file) = std::env::var_os("VICTORY_AMBIENT_FILE") else { return };
        let _ = std::thread::Builder::new().name("victory-ambient".into()).spawn(move || {
            loop {
                // Not `read_to_string`: it asks for the file's metadata,
                // which can hang a thread under Wine (`crate::read_file`).
                let text = crate::read_file(std::path::Path::new(&file)).unwrap_or_default();
                read_live(&String::from_utf8_lossy(&text));
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
    });
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).map(|(level, _)| level)
}

/// Takes what the file's text sets (`set::from_file`).
fn read_live(text: &str) {
    match set::from_file(text) {
        Some(set::FromFile::Follow { follow, shade }) => {
            if shade.is_some() {
                *SHADE.lock().unwrap_or_else(|e| e.into_inner()) = shade;
            }
            let new = Some(Some(follow));
            let mut live = FOLLOW.lock().unwrap_or_else(|e| e.into_inner());
            if *live != new {
                let [gain, least, power] = follow;
                tracing::info!(gain, least, power, "ambient light follows the sky, set from the file");
                *live = new;
            }
        }
        Some(set::FromFile::Constant { level, tint, shade, reflections }) => {
            *FOLLOW.lock().unwrap_or_else(|e| e.into_inner()) = Some(None);
            if shade.is_some() {
                *SHADE.lock().unwrap_or_else(|e| e.into_inner()) = shade;
            }
            if let Some([keep, least]) = reflections {
                *KEEP.lock().unwrap_or_else(|e| e.into_inner()) = Some(keep);
                *LEAST.lock().unwrap_or_else(|e| e.into_inner()) = least;
            }
            let new = Some((level, tint));
            let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            if *live != new {
                let [r, g, b] = tint;
                tracing::info!(level, r, g, b, "ambient light set from the file");
                *live = new;
            }
        }
        None => {}
    }
}

/// Views, used in turn: more than the passes of every list in flight.
const SLOTS: u32 = 1024;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// The pass: compiled with the system's compiler (Wine's under Wine), its
/// root signature built here (`crate::compute`): the constants, then a
/// table of the image, the depth and the sky.
fn make(host: &Host) -> windows::core::Result<Pass> {
    use crate::compute::{Param, srv, uav};
    let ranges = [uav(0, 1), srv(0, 2)];
    let params = [
        Param::Constants { register: 0, count: set::CONSTANTS as u32 },
        Param::Table(&ranges),
    ];
    let (root, pso) = crate::compute::pipeline(set::SOURCE, s!("ambient"), 0, &params)?;
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // Under Wine on D3DMetal, lists other threads submit while a view heap
    // is made can lose their views (VIctory-macOS issue #31).
    let _quiet = host.quiet();
    // SAFETY: plain creations.
    unsafe {
        let views = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * 3,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            root,
            pso,
            views,
            step: host.device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

/// The pass, made on first use; `None` if the host does not take it (said
/// once, with the compiler's messages).
fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(host)
            .inspect_err(|e| tracing::warn!(error = %e, "no ambient light: its pass was not made"))
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
/// own format is `own`: it is that format, or the host took it as castable
/// to it (one host texture per texture memory, in the format of the first
/// title texture there).
fn viewed_as(copy: &HostTexture, own: DXGI_FORMAT, wanted: DXGI_FORMAT) -> bool {
    own == wanted || copy.castable.contains(&(wanted.0 as u32))
}

/// Replaces what `light` holds by the ambient light, in `list`, after the
/// program that wrote it as an image. Whether it recorded anything (the
/// list's heaps, compute root signature and pipeline are then ours).
pub(crate) fn raise(list: &HostList, light: This, depth: Option<This>) -> bool {
    record(list, light, depth, 0.0)
}

/// Holds the reflections in `image` to a part of the ambient light, in
/// `list`, after the program that wrote them last. Whether it recorded
/// anything (the list's heaps, compute root signature and pipeline are then
/// ours).
pub(crate) fn hold(list: &HostList, image: This) -> bool {
    let held = keep();
    if held <= 0.0 || !record(list, image, None, held) {
        return false;
    }
    static HELD: std::sync::Once = std::sync::Once::new();
    HELD.call_once(|| tracing::info!(held, "reflections held to the ambient light"));
    true
}

fn record(list: &HostList, light: This, depth: Option<This>, held: f32) -> bool {
    if !on() {
        return false;
    }
    let Some(host) = host() else {
        return false;
    };
    let Some(light) = crate::resource::host_copy(light) else {
        return false;
    };
    let Some(pass) = pass(host) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.texture.GetDesc() };
    static REFUSED: std::sync::Once = std::sync::Once::new();
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !(d.Format == DXGI_FORMAT_R32_TYPELESS || viewed_as(&light, d.Format, DXGI_FORMAT_R32_UINT))
    {
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "no ambient light: its image is not one a 32-bit integer view can be made of"
            );
        });
        return false;
    }
    let level = live_level().unwrap_or_else(level);
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            level,
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            "ambient light: the first image it writes"
        );
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS * 3;
    // The sky image, if the light follows it and the title has drawn it.
    let sky_image = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let sky = follow().and_then(|f| {
        let copy = sky_image.as_ref()?;
        // SAFETY: a live resource.
        let d = unsafe { copy.texture.GetDesc() };
        Some((copy, f, [d.Width as u32, d.Height], d))
    });
    static FOLLOWS: std::sync::Once = std::sync::Once::new();
    if let Some((copy, f, _, d)) = &sky {
        FOLLOWS.call_once(|| {
            tracing::info!(
                gain = f[0],
                least = f[1],
                power = f[2],
                format = d.Format.0,
                width = d.Width,
                height = d.Height,
                slices = d.DepthOrArraySize,
                "ambient light follows the sky dome"
            );
        });
        list.keep(&copy.texture);
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0;
        if let Some(barrier) = list.move_texture(copy, read as u32) {
            // SAFETY: an open list and a live texture the list keeps.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
    }
    list.keep(&light.texture);
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the shading is left out.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    if let Some(depth) = &depth {
        list.keep(depth);
    }
    let mut shade = shade();
    if depth.is_none() {
        shade[0] = 0.0;
    }
    let constants = set::constants(&set::Pass {
        level,
        tint: tint(),
        shade,
        follow: sky.as_ref().map_or([0.0, 0.0, 1.0], |s| s.1),
        held,
        floor: *LEAST.lock().unwrap_or_else(|e| e.into_inner()),
        size: [d.Width as u32, d.Height],
        sky: sky.as_ref().map_or([0, 0], |s| s.2),
    });
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: an open list; a live texture the list keeps, in the
    // unordered-access state as the program before left it; views in slots
    // no list in flight reads.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(&light.texture, None::<&ID3D12Resource>, Some(&view), at);
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        let beside = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + pass.step as usize };
        host.device.CreateShaderResourceView(depth.as_ref(), Some(&seen), beside);
        // The sky dome's first slice in the format its program draws it in;
        // a null view while the light does not follow it.
        let around = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
            },
        };
        let third = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + 2 * pass.step as usize };
        host.device.CreateShaderResourceView(sky.as_ref().map(|s| &s.0.texture), Some(&around), third);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}

#[cfg(test)]
mod tests {
    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn ambient_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(host).expect("the ambient pass");
    }
}
