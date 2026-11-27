//! Textures shaders write that the title's own code then reads on the CPU
//! (`VICTORY_TEXTURE_BACK`, off by default). The idea and its form are
//! Geckomacabre's (hyphennn 895c0b8f, pull request #999, merge f5c03f3b),
//! with the check of 04df4350 that a landing's memory is still the
//! texture's; the hooks are this line's own.
//!
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it
//! stays in the texture's host copy. The title's water is such a case
//! (fxceless#863, found with `VICTORY_READ_WATCH=1`): its simulation writes
//! a 256x256 height map (R32_FLOAT, `f41`) and displacement map
//! (R32G32_FLOAT, `f16`), its min/max passes a 64x64 R32G32_FLOAT and a
//! 128x128 R16G16_UINT (`f36`) texture, and its CPU code samples all four.
//! Without them, in the mission after the boat cutscene the boat sinks into
//! a pit in the water and the mission fails.
//!
//! For textures of the kinds the switch names:
//!
//! 1. a draw or dispatch that binds one for unordered access through a
//!    table, a root view or a placed view (`crate::draw::readable`) notes
//!    it on its host list, with a [`Ticket`] for the title memory it lies
//!    in ([`Target`]);
//! 2. when the list closes, each one noted is moved to the copy-source
//!    state and copied, subresource by subresource, into a read-back buffer
//!    ([`record`]);
//! 3. once the list has run (`Host::finish`: after the copy-back of
//!    buffers, before the lists' work for after and their bottom-of-pipe
//!    writes, so before any fence of the title's moves past it), the texels
//!    are tiled into the console's layout (`victory_formats`
//!    `Layout::tile`, the inverse of the detile `crate::texture` uploads
//!    with) and written into the texture's title memory as the layer writes
//!    its other results there (`crate::list::read_back`,
//!    `crate::shadow::Shadow::copy_back`): `crate::watch::writing` around a
//!    kernel copy (`crate::host::write_mapped`), so nothing faults on a
//!    guarded page, an unmapped page is left alone, and every copy holding
//!    the memory takes it again ([`land_all`]). Of a texture several lists
//!    of one submission wrote, the last list's copy alone lands.
//!
//! Only textures whose host texture the GPU writes are taken: such a host
//! texture is never brought up to date from title memory
//! (`crate::texture::title_written`), so our write there is not taken for
//! the title's and uploads nothing.
//!
//! `VICTORY_TEXTURE_BACK=256x256f41,256x256f16,64x64f16,128x128f36`: the
//! water's four kinds, as `<width>x<height>f<DXGI format>` separated by
//! commas. `1` (or `on`) stands for the same four: the forks differ on it
//! (hyphennn and avatarmanz take it as these, DexNom as none). Unset,
//! empty, `0` or `off`: nothing is copied back, and nothing here runs but
//! one check of a stored value per texture bound for unordered access. A
//! value with an entry not of that form turns it off, with a warning.
//!
//! A landing is skipped, and counted, when since the dispatch the title
//! released the texture, made another texture at its address, or mapped or
//! unmapped pages under it ([`Ticket`]: looked at, and the release held off,
//! while the write is made); when a page of its memory is not committed
//! (the read of the bytes the texels do not cover fails); and when its
//! layout cannot be tiled (XOR modes: refused when noted, logged once per
//! kind). The bytes written are the texture's console layout, which lies
//! inside the memory the texture was given when the title made it
//! committed or placed (`crate::resource::title_texture`).
//!
//! shortcut: by kind, not by texture: every texture of those kinds that
//! shaders write is copied back, whether the title reads it or not (about
//! a megabyte per list for the water's). Writes through views shaders take
//! from the heap themselves (`crate::bindless`) are not seen, as in every
//! fork: the first-sight lines say which kinds were seen.
//!
//! With the frame line ([`log`]): textures landed per frame, their
//! megabytes and milliseconds (part of the frame line's "copying back"
//! time), the copies recorded and the landings skipped.

// The host side is Windows-only; the rules are plain std, so that their
// tests run everywhere.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(windows)]
use crate::object::This;

/// A title texture's width, height and DXGI format number.
pub(crate) type Kind = (u64, u32, u32);

/// The water's four kinds (fxceless#863), and what `1` stands for.
const WATER: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

/// The kinds value `value` of `VICTORY_TEXTURE_BACK` names (none: off), or
/// the first entry not understood.
fn parse(value: Option<&str>) -> Result<Vec<Kind>, String> {
    let value = value.map(str::trim).unwrap_or_default();
    if value.is_empty() || value == "0" || value.eq_ignore_ascii_case("off") {
        return Ok(Vec::new());
    }
    let value = if value == "1" || value.eq_ignore_ascii_case("on") { WATER } else { value };
    let mut kinds = Vec::new();
    for entry in value.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let kind: Option<Kind> = entry.to_ascii_lowercase().split_once('f').and_then(|(size, format)| {
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        });
        match kind {
            Some(k @ (w, h, _)) if w > 0 && h > 0 => {
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
            _ => return Err(entry.to_owned()),
        }
    }
    Ok(kinds)
}

/// A kind in a few words.
fn describe(&(w, h, f): &Kind) -> String {
    let name = match f {
        16 => " (R32G32_FLOAT)",
        36 => " (R16G16_UINT)",
        41 => " (R32_FLOAT)",
        _ => "",
    };
    format!("{w}x{h} f{f}{name}")
}

/// The line said at start when the switch is on.
fn start_line(kinds: &[Kind]) -> String {
    let kinds: Vec<String> = kinds.iter().map(describe).collect();
    format!(
        "VICTORY_TEXTURE_BACK: textures of these kinds that shaders write are copied back into \
         title memory once their list has run: {}",
        kinds.join(", ")
    )
}

/// Tests: the water's kinds, whatever the environment.
#[cfg(test)]
static FORCED: AtomicBool = AtomicBool::new(false);

/// Turns the water's kinds on for a test, or back to the environment's.
#[cfg(test)]
pub(crate) fn force(on: bool) {
    FORCED.store(on, Ordering::Relaxed);
}

/// The kinds copied back, read once.
fn kinds() -> &'static [Kind] {
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        static WATER_KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
        return WATER_KINDS.get_or_init(|| parse(Some(WATER)).unwrap_or_default());
    }
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let value = std::env::var("VICTORY_TEXTURE_BACK").ok();
        parse(value.as_deref()).unwrap_or_else(|entry| {
            tracing::warn!(
                value = value.as_deref().unwrap_or_default(),
                entry,
                "VICTORY_TEXTURE_BACK takes kinds as <width>x<height>f<format> separated by commas \
                 (the water's: {WATER}), or 1 for those: nothing is copied back"
            );
            Vec::new()
        })
    })
}

/// Whether some kind is copied back.
pub(crate) fn on() -> bool {
    !kinds().is_empty()
}

/// Says what is copied back, once, when the host device is ready.
pub(crate) fn announce() {
    if on() {
        tracing::info!("{}", start_line(kinds()));
    }
}

/// Whether textures of `kind` are copied back.
fn copied_back(kind: Kind) -> bool {
    kinds().contains(&kind)
}

/// A pending landing's claim on title memory: whose texture it is (the
/// title resource object's address), the bytes it lands in, and whether
/// they are still that texture's.
pub(crate) struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

impl Ticket {
    fn valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }
}

/// The tickets alive. Their lock is held while a landing writes
/// ([`Tickets::while_valid`]): the title's release of the texture (whose
/// memory goes with it), a texture made at its address and a mapping
/// changed under it wait for the write to end.
struct Tickets {
    held: Mutex<Vec<Weak<Ticket>>>,
    /// A ticket was made once: until then nothing is looked at.
    any: AtomicBool,
}

impl Tickets {
    const fn new() -> Self {
        Tickets { held: Mutex::new(Vec::new()), any: AtomicBool::new(false) }
    }

    fn track(&self, owner: usize, at: usize, end: usize) -> Arc<Ticket> {
        let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        held.retain(|t| t.strong_count() > 0);
        held.push(Arc::downgrade(&ticket));
        self.any.store(true, Ordering::Release);
        ticket
    }

    /// The tickets `hit` picks no longer hold.
    fn invalidate(&self, hit: impl Fn(&Ticket) -> bool) {
        if !self.any.load(Ordering::Acquire) {
            return;
        }
        self.held.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
            let Some(ticket) = weak.upgrade() else {
                return false;
            };
            if hit(&ticket) {
                ticket.valid.store(false, Ordering::Release);
            }
            true
        });
    }

    /// Runs `write` if `ticket` holds, and nothing invalidates a ticket
    /// meanwhile: whether it ran.
    fn while_valid(&self, ticket: &Ticket, write: impl FnOnce()) -> bool {
        let _held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if !ticket.valid() {
            return false;
        }
        write();
        true
    }

    fn released(&self, owner: usize) {
        self.invalidate(|t| t.owner == owner);
    }

    fn replaced(&self, at: usize) {
        self.invalidate(|t| t.at == at);
    }

    fn remapped(&self, at: usize, len: usize) {
        let end = at.saturating_add(len);
        self.invalidate(|t| at < t.end && t.at < end);
    }
}

static TICKETS: Tickets = Tickets::new();

/// The title released the texture whose object is at `owner`: its pending
/// landings are off (its memory may go with it).
pub(crate) fn released(owner: usize) {
    TICKETS.released(owner);
}

/// A texture was made at title address `at`: pending landings of others
/// there are off.
pub(crate) fn replaced(at: usize) {
    TICKETS.replaced(at);
}

/// Pages of `[at, at + len)` are mapped or unmapped: pending landings into
/// them are off.
pub(crate) fn remapped(at: usize, len: usize) {
    TICKETS.remapped(at, len);
}

/// Since the last frame line: copies recorded; textures landed, their
/// bytes and the microseconds landing took; landings left to a later copy
/// of the same texture in the submission; skipped as the memory changed
/// hands, as it was not all committed, as it could not be tiled.
static COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const RECORDED: usize = 0;
const LANDED: usize = 1;
const BYTES: usize = 2;
const MICROS: usize = 3;
const LATER: usize = 4;
const STALE: usize = 5;
const UNCOMMITTED: usize = 6;
const UNTILED: usize = 7;

fn count(what: usize, n: u64) {
    COUNTS[what].fetch_add(n, Ordering::Relaxed);
}

/// The line logged with the frame line, `n` the counts over `frames`
/// frames.
fn frame_line(n: [u64; 8], frames: f64) -> String {
    let per = |i: usize| n[i] as f64 / frames.max(1.0);
    format!(
        "texture copy-back (VICTORY_TEXTURE_BACK): {:.1} textures/frame landed in title memory, \
         {:.2} MB/frame in {:.1} ms/frame (part of the copying back); {:.1}/frame copied on the GPU; \
         per frame {:.1} left to a later copy in the same submission, {:.1} skipped as the title \
         freed, replaced or remapped the memory, {:.1} as it was not all committed, {:.1} not tiled",
        per(LANDED),
        per(BYTES) / 1e6,
        per(MICROS) / 1e3,
        per(RECORDED),
        per(LATER),
        per(STALE),
        per(UNCOMMITTED),
        per(UNTILED),
    )
}

/// With the frame line, over `frames` frames. Nothing while the switch is
/// off.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let n = COUNTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    tracing::info!("{}", frame_line(n, frames));
}

/// Where a subresource lies in a read-back buffer, as the host's copyable
/// footprints say: offset, row pitch, rows, depth slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sub {
    offset: u64,
    pitch: u32,
    rows: u32,
    depth: u32,
}

/// Tiles the subresources `subs` (in subresource order, `mips` per array
/// slice) of read-back buffer `mapped` into `tiled`, the texture's title
/// memory as it was read, as `layout` lays them out: what no subresource
/// covers keeps its bytes. On an error, the subresource (mip, slice) and
/// why; nothing of it or of those after it is tiled.
fn tile_all(
    layout: &victory_formats::texture::Layout,
    mips: u32,
    subs: &[Sub],
    mapped: &[u8],
    tiled: &mut [u8],
) -> Result<(), (victory_formats::texture::Error, u32, u32)> {
    let mips = mips.max(1);
    for (i, s) in subs.iter().enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let pitch = s.pitch as usize;
        let from = usize::try_from(s.offset).unwrap_or(usize::MAX).min(mapped.len());
        // The host's footprints leave the last row's padding out of the
        // buffer (as `crate::texture::subresource_len` clamps it).
        let len = pitch
            .saturating_mul(s.rows as usize)
            .saturating_mul(s.depth as usize)
            .min(mapped.len() - from);
        let linear = &mapped[from..from + len];
        layout
            .tile(tiled, mip, slice, linear, pitch, pitch * s.rows as usize)
            .map_err(|e| (e, mip, slice))?;
    }
    Ok(())
}

/// Whether this is the first time `kind` is told about as `what` (0: seen,
/// 1: refused).
#[cfg(windows)]
fn first_time(kind: Kind, what: u8) -> bool {
    static TOLD: Mutex<Vec<(Kind, u8)>> = Mutex::new(Vec::new());
    let mut told = TOLD.lock().unwrap_or_else(|e| e.into_inner());
    let first = !told.contains(&(kind, what));
    if first {
        told.push((kind, what));
    }
    first
}

/// Whether title texture `resource` is of a kind copied back (while the
/// switch is off: one stored value looked at).
#[cfg(windows)]
pub(crate) fn wanted(resource: This) -> bool {
    on() && crate::resource::texture_kind(resource).is_some_and(copied_back)
}

/// A texture a list's shaders write, noted when the draw or dispatch is
/// prepared: its host texture (held, whatever the title releases), its
/// console layout and its ticket.
#[cfg(windows)]
pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

#[cfg(windows)]
impl Target {
    /// For title texture `resource` of a kind copied back; `None` (logged
    /// once per kind) for one that cannot land.
    pub(crate) fn new(resource: This) -> Option<Target> {
        let kind = crate::resource::texture_kind(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let refused = |why: std::fmt::Arguments| {
            if first_time(kind, 1) {
                tracing::warn!(kind = describe(&kind), ?resource, "a texture of a kind copied back is not: {why}");
            }
        };
        if crate::texture::title_written(&copy.texture) {
            refused(format_args!("its host texture is brought up to date from title memory"));
            return None;
        }
        if copy.twin.get().is_some_and(Option::is_some) {
            refused(format_args!("a depth texture, written through its twin"));
            return None;
        }
        let Some((layout, at, size)) = crate::resource::title_texture(resource) else {
            refused(format_args!("no console layout of one plane inside its memory"));
            return None;
        };
        if !layout.addressable() {
            refused(format_args!("its layout ({:?}) cannot be tiled", layout.desc.swizzle));
            return None;
        }
        let end = at.checked_add(size).filter(|_| at != 0 && size != 0)?;
        if first_time(kind, 0) {
            tracing::info!(
                kind = describe(&kind),
                ?resource,
                at = format_args!("{at:#x}"),
                size,
                swizzle = ?layout.desc.swizzle,
                mips = layout.desc.mips,
                "the first texture of a kind copied back written through a bound view"
            );
        }
        Some(Target { copy, layout, ticket: TICKETS.track(resource as usize, at, end) })
    }

    /// The title texture it is of (its object's address).
    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }

    /// Whether its memory is still the texture's.
    pub(crate) fn valid(&self) -> bool {
        self.ticket.valid()
    }
}

/// What a list copies of a [`Target`] when it closes: the read-back buffer
/// and its mapping, where each subresource lies in it, and where its texels
/// go.
#[cfg(windows)]
pub(crate) struct Back {
    /// Given back to [`FREE`] when it goes.
    buffer: Option<ID3D12Resource>,
    data: usize,
    bytes: u64,
    subs: Vec<Sub>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

/// Read-back buffers of copies that are done with, for the next copy of the
/// same bytes (lists copy the same textures every frame, and a buffer is a
/// committed resource of its own): bytes, buffer, mapping.
#[cfg(windows)]
struct Free(Vec<(u64, ID3D12Resource, usize)>);

// SAFETY: D3D12 resources are free-threaded; the mappings are persistent
// mappings of the buffers.
#[cfg(windows)]
unsafe impl Send for Free {}

#[cfg(windows)]
static FREE: Mutex<Free> = Mutex::new(Free(Vec::new()));

/// Buffers kept free at most (a few per kind).
#[cfg(windows)]
const KEPT: usize = 32;

/// A read-back buffer of `bytes` and its mapping: a free one, or new.
#[cfg(windows)]
fn read_back_buffer(host: &crate::host::Host, bytes: u64) -> Option<(ID3D12Resource, usize)> {
    {
        let mut free = FREE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = free.0.iter().position(|f| f.0 == bytes) {
            let (_, buffer, data) = free.0.swap_remove(i);
            return Some((buffer, data));
        }
    }
    let (buffer, data) = host.readback_buffer(bytes)?;
    crate::host::name(&buffer, || format!("texture copy-back {bytes:#x}"));
    Some((buffer, data as usize))
}

#[cfg(windows)]
impl Drop for Back {
    /// Its list was reset or let go, so the GPU is done with the buffer
    /// (it ran the list, or never will).
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut free = FREE.lock().unwrap_or_else(|e| e.into_inner());
            if free.0.len() < KEPT {
                free.0.push((self.bytes, buffer, self.data));
            }
        }
    }
}

/// Records into open list `list` the copy of `target`'s host texture (in
/// the copy-source state) into a read-back buffer: what lands once the
/// list has run ([`land_all`]).
#[cfg(windows)]
pub(crate) fn record(list: &crate::host::HostList, target: Target) -> Option<Back> {
    if !target.valid() {
        count(STALE, 1);
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture, held by the target.
    let desc = unsafe { texture.GetDesc() };
    let (mips, slices) = (u32::from(desc.MipLevels.max(1)), u32::from(desc.DepthOrArraySize.max(1)));
    if (mips, slices) != (target.layout.desc.mips, target.layout.slices) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(?desc, layout = ?target.layout.desc, "a texture copied back has other mips or slices on the host: not copied");
        });
        return None;
    }
    let n = (mips * slices) as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: outputs for `n` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            n as u32,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    if total == 0 || total == u64::MAX {
        return None;
    }
    let (buffer, data) = read_back_buffer(host, total)?;
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "texture copy-back of {:#x} ({}x{} f{}, {n} subresources) into a read-back buffer of {total} bytes",
            target.ticket.owner, desc.Width, desc.Height, desc.Format.0
        )
    });
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state and
        // the buffer, a read-back one, in the copy destination state.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    count(RECORDED, 1);
    let subs = footprints
        .iter()
        .zip(&rows)
        .map(|(f, &rows)| Sub { offset: f.Offset, pitch: f.Footprint.RowPitch, rows, depth: f.Footprint.Depth })
        .collect();
    Some(Back { buffer: Some(buffer), data, bytes: total, subs, mips, layout: target.layout, ticket: target.ticket })
}

/// Which of copies `keys` (texture, title address; in the order their lists
/// ran) land: the last of each texture's. The lists have all run, so the
/// last one holds what they left, and landing the others first would only
/// be written over.
fn last_of_each(keys: &[(usize, usize)]) -> Vec<bool> {
    let mut lands = vec![false; keys.len()];
    let mut seen = Vec::new();
    for (i, key) in keys.iter().enumerate().rev() {
        if !seen.contains(key) {
            seen.push(*key);
            lands[i] = true;
        }
    }
    lands
}

/// Once a submission's lists have run (`Host::finish`): the copies `backs`
/// of their textures (in list order) land in title memory, of a texture
/// several lists wrote the last list's alone.
/// mac: the title memory a copy lands in once its list has run (`[at,
/// end)`), for a submission prepared while that list still runs
/// (`crate::host::unwaited::flights`).
#[cfg(windows)]
impl Back {
    pub(crate) fn lands(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

#[cfg(windows)]
pub(crate) fn land_all(backs: &[Arc<Back>]) {
    if backs.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let keys: Vec<(usize, usize)> = backs.iter().map(|b| (b.ticket.owner, b.ticket.at)).collect();
    for (back, lands) in backs.iter().zip(last_of_each(&keys)) {
        if lands {
            land(back);
        } else {
            count(LATER, 1);
        }
    }
    count(MICROS, started.elapsed().as_micros() as u64);
}

/// Lands one copy, unless its memory changed hands, is not all committed,
/// or its texels cannot be tiled.
#[cfg(windows)]
fn land(back: &Back) {
    let ticket = &back.ticket;
    if !ticket.valid() {
        count(STALE, 1);
        return;
    }
    let mut tiled = vec![0u8; ticket.end - ticket.at];
    // The bytes no subresource covers keep the title's; a page not
    // committed fails the read, and nothing lands.
    if !crate::host::read_into(ticket.at as *const u8, &mut tiled) {
        count(UNCOMMITTED, 1);
        return;
    }
    // SAFETY: the read-back buffer's persistent mapping, `bytes` long,
    // which the list that has run copied into; the buffer lives with `back`.
    let mapped = unsafe { std::slice::from_raw_parts(back.data as *const u8, back.bytes as usize) };
    if let Err((e, mip, slice)) = tile_all(&back.layout, back.mips, &back.subs, mapped, &mut tiled) {
        count(UNTILED, 1);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(?e, mip, slice, desc = ?back.layout.desc, "a texture copied back could not be tiled (logged once)"));
        return;
    }
    if TICKETS.while_valid(ticket, || write_title(ticket.at, &tiled)) {
        count(LANDED, 1);
        count(BYTES, tiled.len() as u64);
    } else {
        count(STALE, 1);
    }
}

/// Writes `bytes` into title memory at `at` as the layer writes its other
/// results there: its pages open (no fault on a guarded one) and marked
/// changed in every copy holding them while a kernel copy writes them,
/// which leaves a page that is not mapped alone.
#[cfg(windows)]
fn write_title(at: usize, bytes: &[u8]) {
    let _writing = crate::watch::writing(at, bytes.len());
    crate::host::write_mapped(at as *mut u8, bytes.as_ptr(), bytes.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

    #[test]
    fn the_switch_names_its_kinds() {
        let water = vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)];
        for off in [None, Some(""), Some(" 0 "), Some("OFF")] {
            assert_eq!(parse(off), Ok(Vec::new()), "{off:?}");
        }
        assert_eq!(parse(Some(WATER)), Ok(water.clone()));
        // `1` is the same four, not none (as in DexNom's form).
        assert_eq!(parse(Some("1")), Ok(water.clone()));
        assert_eq!(parse(Some("on")), Ok(water));
        assert_eq!(parse(Some(" 64x64F16 , 128X128f36,64x64f16, ")), Ok(vec![(64, 64, 16), (128, 128, 36)]));
        // Anything else turns it off, naming the entry.
        assert_eq!(parse(Some("yes")), Err("yes".into()));
        assert_eq!(parse(Some("256x256f41,256x256")), Err("256x256".into()));
        assert_eq!(parse(Some("0x256f41")), Err("0x256f41".into()));
        assert_eq!(parse(Some("256x256f")), Err("256x256f".into()));
    }

    #[test]
    fn the_start_line_names_the_kinds() {
        let line = start_line(&parse(Some("1")).unwrap());
        assert!(line.contains("copied back"), "{line}");
        assert!(line.ends_with(
            "256x256 f41 (R32_FLOAT), 256x256 f16 (R32G32_FLOAT), 64x64 f16 (R32G32_FLOAT), 128x128 f36 (R16G16_UINT)"
        ));
    }

    /// Each kind of change invalidates only the landings it touches, and a
    /// ticket made afterwards holds.
    #[test]
    fn only_landings_whose_memory_changed_hands_are_off() {
        let tickets = Tickets::new();
        let first = tickets.track(91, 0x1000, 0x2000);
        let next = tickets.track(92, 0x2000, 0x3000);
        let far = tickets.track(93, 0x4000, 0x5000);
        tickets.remapped(0x0f00, 0x200);
        assert!(!first.valid() && next.valid() && far.valid(), "a mapping over the first");
        tickets.remapped(0x3000, 0x1000);
        assert!(next.valid() && far.valid(), "a mapping between the two");
        tickets.replaced(0x2000);
        assert!(!next.valid() && far.valid(), "a texture made where the next lies");
        tickets.released(93);
        assert!(!far.valid(), "the far one's texture released");
        assert!(tickets.track(94, 0x2000, 0x3000).valid());
        assert!(!tickets.while_valid(&far, || panic!("a landing whose memory changed hands")));
        let mut wrote = false;
        assert!(tickets.while_valid(&tickets.track(95, 0x6000, 0x7000), || wrote = true));
        assert!(wrote);
    }

    /// A release while a landing writes waits until the write is done:
    /// the memory cannot go from under it.
    #[test]
    fn a_release_waits_for_the_landing_under_way() {
        use std::time::{Duration, Instant};
        let tickets = Arc::new(Tickets::new());
        let ticket = tickets.track(7, 0x1000, 0x2000);
        let (started, ended) = (Arc::new(AtomicBool::new(false)), Arc::new(Mutex::new(None)));
        let writer = {
            let (tickets, ticket, started, ended) = (tickets.clone(), ticket.clone(), started.clone(), ended.clone());
            std::thread::spawn(move || {
                tickets.while_valid(&ticket, || {
                    started.store(true, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(50));
                    *ended.lock().unwrap() = Some(Instant::now());
                })
            })
        };
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        tickets.released(7);
        let released = Instant::now();
        assert!(writer.join().unwrap(), "the write began while the ticket held");
        let ended = ended.lock().unwrap().expect("the write ended");
        assert!(ended <= released, "the release went on before the write ended");
        assert!(!ticket.valid());
    }

    /// The host's copyable footprints of a 2D texture of `element` bytes per
    /// texel: offsets aligned to 512, row pitches to 256.
    fn footprints(width: u32, height: u32, element: u32, mips: u32) -> (Vec<Sub>, usize) {
        let (mut subs, mut at) = (Vec::new(), 0u64);
        let mut end = 0;
        for mip in 0..mips {
            let (w, h) = ((width >> mip).max(1), (height >> mip).max(1));
            let pitch = (w * element).next_multiple_of(256);
            at = at.next_multiple_of(512);
            subs.push(Sub { offset: at, pitch, rows: h, depth: 1 });
            // The last row without its padding, as the host counts it.
            end = at as usize + (h as usize - 1) * pitch as usize + (w * element) as usize;
            at += u64::from(pitch) * u64::from(h);
        }
        (subs, end)
    }

    fn layout(swizzle: Swizzle) -> victory_formats::texture::Layout {
        let t = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth_or_array: 1,
            mips: 3,
            format: 16,
            samples: 1,
            swizzle,
        })
        .unwrap();
        t.planes[0].layout.clone()
    }

    /// A 64x64 R32G32_FLOAT texture of three mips (one of the water's
    /// kinds), in the read-back buffer as the host lays it out, tiled into
    /// title memory: each texel where the console's layout puts it, the
    /// bytes no texel covers as the title had them.
    #[test]
    fn every_mip_is_tiled_where_the_console_reads_it() {
        let layout = layout(Swizzle::SW_4KB_S);
        let (subs, total) = footprints(64, 64, 8, 3);
        let texel = |mip: u32, x: u32, y: u32| ((u64::from(mip) + 1) << 48 | u64::from(y) << 16 | u64::from(x)).to_le_bytes();
        let mut mapped = vec![0xeeu8; total];
        for (mip, s) in subs.iter().enumerate() {
            for y in 0..s.rows {
                for x in 0..(64 >> mip) {
                    let at = s.offset as usize + (y * s.pitch + x * 8) as usize;
                    mapped[at..at + 8].copy_from_slice(&texel(mip as u32, x, y));
                }
            }
        }
        let mut tiled = vec![0xcdu8; layout.size as usize];
        tile_all(&layout, 3, &subs, &mapped, &mut tiled).unwrap();
        let mut covered = vec![false; tiled.len()];
        for mip in 0..3 {
            for y in 0..64 >> mip {
                for x in 0..64 >> mip {
                    let at = layout.offset(mip, x, y, 0).unwrap() as usize;
                    assert_eq!(tiled[at..at + 8], texel(mip, x, y), "mip {mip} ({x}, {y})");
                    covered[at..at + 8].fill(true);
                }
            }
        }
        let kept = tiled.iter().zip(&covered).filter(|&(_, &c)| !c).all(|(&b, _)| b == 0xcd);
        assert!(kept, "the bytes between and after the mips are the title's");
        // A buffer that ends short is refused before anything of the
        // subresource is written.
        let mut again = vec![0xcdu8; layout.size as usize];
        let short = &mapped[..subs[2].offset as usize + 100];
        assert!(matches!(tile_all(&layout, 3, &subs, short, &mut again), Err((Error::TooSmall, 2, 0))));
    }

    /// A pipe and bank XOR mode is not tiled (these are refused when
    /// noted, before any copy is made).
    #[test]
    fn xor_modes_are_not_tiled() {
        let layout = layout(Swizzle::SW_64KB_R_X);
        assert!(!layout.addressable());
        let (subs, total) = footprints(64, 64, 8, 3);
        let mut tiled = vec![0xcdu8; layout.size as usize];
        assert!(matches!(tile_all(&layout, 3, &subs, &vec![0; total], &mut tiled), Err((Error::Unsupported, 0, 0))));
        assert!(tiled.iter().all(|&b| b == 0xcd));
    }

    /// Of a texture copied by several lists of a submission, the last
    /// list's copy lands (a list named twice too); others land in order.
    #[test]
    fn the_last_copy_of_each_texture_lands() {
        let (a, b, c) = ((1, 0x1000), (2, 0x2000), (3, 0x1000));
        assert_eq!(last_of_each(&[]), Vec::<bool>::new());
        assert_eq!(last_of_each(&[a, b, a, c, b]), [false, false, true, true, true]);
        assert_eq!(last_of_each(&[a, a]), [false, true]);
    }

    #[test]
    fn the_frame_line_counts_per_frame() {
        let line = frame_line([8, 4, 2_000_000, 30_000, 2, 1, 0, 0], 2.0);
        assert!(line.contains("2.0 textures/frame landed"), "{line}");
        assert!(line.contains("1.00 MB/frame in 15.0 ms/frame"), "{line}");
        assert!(line.contains("4.0/frame copied on the GPU"), "{line}");
        assert!(line.contains("1.0 left to a later copy"), "{line}");
        assert!(line.contains("0.5 skipped as the title"), "{line}");
    }

    /// A landing writes through pages a copy of the memory guards (the
    /// kernel copy alone leaves them unwritten), and that copy takes the
    /// pages again.
    #[cfg(windows)]
    #[test]
    fn a_landing_writes_through_guarded_pages_and_tells_the_copies_there() {
        use crate::watch::PAGE;
        use crate::watch::tests::{free, pool};
        if !crate::watch::fault_tracking() {
            return; // tracking turned off for this run
        }
        let p = pool(2);
        let copy = crate::watch::track(p, 2 * PAGE, true).expect("a pool is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)], "guarded now");
        let bytes: Vec<u8> = (0..PAGE + 100).map(|i| (i % 251) as u8 + 1).collect();
        // SAFETY: the pool's committed pages.
        let memory = |at: usize, n: usize| unsafe { std::slice::from_raw_parts(at as *const u8, n) }.to_vec();
        crate::host::write_mapped((p + 50) as *mut u8, bytes.as_ptr(), bytes.len());
        assert!(memory(p, 2 * PAGE).iter().all(|&b| b == 0), "a guarded page refuses the kernel's copy");
        assert!(!copy.changed(p / PAGE) && !copy.changed(p / PAGE + 1));
        write_title(p + 50, &bytes);
        assert_eq!(memory(p + 50, bytes.len()), bytes);
        assert!(memory(p, 50).iter().all(|&b| b == 0), "nothing before it");
        assert!(copy.changed(p / PAGE) && copy.changed(p / PAGE + 1), "the copy takes both pages again");
        drop(copy);
        free(p);
    }
}
