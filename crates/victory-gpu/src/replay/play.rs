//! The replayer: makes a captured frame's objects again, puts their
//! contents back, and runs the frame as often as asked, with what it costs
//! and experiments on it (`super::edit`).
//!
//! A frame runs as the runtime runs it unless asked otherwise: one
//! submission after the other, each waited for, with what the CPU wrote and
//! the views it made put in place between them by the CPU. Sent without
//! waits (or as one submission), copies on the GPU stand for what the CPU
//! wrote and each submission that replaces views binds heaps of its own.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::File;
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use super::edit;
pub use super::look::{find, listing};
use super::look::hex;
use super::gpu::{self, Seen, Worker};
use super::plain::{self, median};
use super::snap::read_sub;
use super::states::{
    self, COMMON, COPY_DEST, COPY_SOURCE, DEPTH_WRITE, RENDER_TARGET, RESOLVE_DEST, States,
    UNORDERED_ACCESS,
};
use super::wire::{
    self, ALL, Barrier, Cmd, Content, Delta, Frame, Loc, MISSING, NONE, Place, Slot, Va, ZERO, slot,
};

/// An upload buffer contents come back through.
const STAGING: u64 = 256 << 20;

/// The name of the pipeline of the host layer's argument clamp
/// (`crate::indirect`, as `crate::compute::pipeline` names it).
const CLAMP: &str = "victory: clamp";
/// And of its batches of clamps, which `--skip-clamps` leaves in.
const CLAMP_BATCH: &str = "victory: clamps";

/// How the frame's submissions are sent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Waits {
    /// Each one waited for before the next is prepared and sent, as the
    /// runtime does.
    #[default]
    Each,
    /// All sent at once, and one wait at the end.
    None,
    /// All in one `ExecuteCommandLists`.
    OneSubmit,
    /// Each list a submission of its own, waited for (as the runtime's
    /// `VICTORY_GPU_TIMING` sends them): which lists are the heavy ones.
    Lists,
}

/// Which of the frame's barriers are sent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Barriers {
    #[default]
    AsRecorded,
    /// None of buffers (their transitions, and waits for unordered access
    /// to them): what the host layer's own passes add around each call.
    NoBufferOnes,
    /// None at all. No valid D3D12; D3DMetal runs it.
    NoneAtAll,
}

/// How the time of single commands is found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Detail {
    /// Not at all: submissions only.
    #[default]
    Off,
    /// By timestamps after every command, if the host has them and they
    /// tell commands apart; else in pieces.
    Auto,
    /// By timestamps after every command.
    Stamps,
    /// Lists cut into pieces (`edit::pieces`), each sent alone and waited
    /// for.
    Pieces,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub loops: u32,
    /// Put back, before each run, what the run before changed.
    pub restore: bool,
    pub detail: Detail,
    /// Pipelines listed.
    pub top: usize,
    /// Where the picture goes (default: `replay.png` in the capture).
    pub image: Option<PathBuf>,
    pub waits: Waits,
    /// Milliseconds the GPU stands idle after each submission (waited for,
    /// or sent), as in a title that prepares the next one meanwhile.
    pub pace: f64,
    /// Skip what runs a pipeline whose name or shader hash has one of
    /// these in it.
    pub skip_shaders: Vec<String>,
    /// Shaders taken from this folder where it has them ([`shader_file`]).
    pub shaders: Option<PathBuf>,
    /// With `compare`: the folder the shaders of the second are from.
    pub shaders_b: Option<PathBuf>,
    /// Run the frame unchanged too, turn by turn, and compare.
    pub compare: bool,
    /// The debug layer, if there is one.
    pub debug: bool,
    /// The names of the title's shaders for the reports ([`Names`]).
    pub names: Option<PathBuf>,
    /// After the report: for so many of the costliest pipelines, the frame
    /// with and without each, turn by turn.
    pub leave_out: usize,
    pub barriers: Barriers,
    /// Leave out the host layer's argument clamps: the calls they serve
    /// read arguments kept from a run that made them.
    pub keep_clamps: bool,
    /// Every scissor rectangle one texel: draws without their pixels.
    pub one_texel: bool,
    /// Nothing that works: what the lists bind, and their barriers.
    pub binds_only: bool,
    /// Every dispatch one thread group: what dispatches cost whatever
    /// they compute.
    pub one_group: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            loops: 5,
            restore: true,
            detail: Detail::Off,
            top: 20,
            image: None,
            waits: Waits::Each,
            pace: 0.0,
            skip_shaders: Vec::new(),
            shaders: None,
            shaders_b: None,
            compare: false,
            debug: false,
            names: None,
            leave_out: 0,
            barriers: Barriers::AsRecorded,
            keep_clamps: false,
            one_texel: false,
            binds_only: false,
            one_group: false,
        }
    }
}

impl Options {
    fn edited(&self) -> bool {
        self.waits != Waits::Each
            || self.pace > 0.0
            || !self.skip_shaders.is_empty()
            || self.barriers != Barriers::AsRecorded
            || self.keep_clamps
            || self.one_texel
            || self.binds_only
            || self.one_group
    }

    /// The same without the experiments.
    fn unchanged(&self) -> Options {
        Options {
            waits: Waits::Each,
            pace: 0.0,
            skip_shaders: Vec::new(),
            barriers: Barriers::AsRecorded,
            leave_out: 0,
            keep_clamps: false,
            one_texel: false,
            binds_only: false,
            one_group: false,
            ..self.clone()
        }
    }
}

/// How a run went.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// How the submissions were sent.
    pub waits: Waits,
    /// Per loop, in ms: the frame on the wall clock. Waited for one by one:
    /// the sum of the submissions' times (each from its
    /// `ExecuteCommandLists` to the moment its fence is reached); else from
    /// the first `ExecuteCommandLists` to the moment all have run.
    pub frame_ms: Vec<f64>,
    /// Per loop: what the CPU did between submissions (views and buffers
    /// put in place), which is no part of `frame_ms` when each submission
    /// is waited for.
    pub between_ms: Vec<f64>,
    /// Per loop, by the GPU's timestamps (empty without them): from the
    /// first list's start to the last one's end, the sum of the
    /// submissions, and the copies that stand for what the CPU wrote.
    pub stamp_frame_ms: Vec<f64>,
    pub stamp_busy_ms: Vec<f64>,
    pub stamp_delta_ms: Vec<f64>,
    /// By submission: its lists, the median of its time on the wall clock
    /// (only when waited for one by one) and by timestamps.
    pub submits: Vec<(usize, Option<f64>, Option<f64>)>,
    /// With detailed timing: how it was found.
    pub detail: Option<&'static str>,
    /// With detailed timing, by kind of command: its name, ms (median of
    /// the loops), commands, and (timed in pieces) its share of the frame's
    /// own time in ms: each submission's pieces scaled to add up to what
    /// the submission took whole.
    pub categories: Vec<(String, f64, u32, Option<f64>)>,
    /// With detailed timing, the costliest pipelines: ms, commands, name,
    /// and the share of the frame's own time.
    pub shaders: Vec<(f64, u32, String, Option<f64>)>,
    /// For each of `shaders`: its pipelines.
    pub shader_ids: Vec<Vec<u32>>,
    /// All pipelines that ran, the ones with the heaviest work first (by
    /// what their pieces took more than a piece that does next to nothing):
    /// name and pipelines. What `leave_out` tries.
    pub heaviest: Vec<(String, Vec<u32>)>,
    /// Commands left out by `--skip-shader`.
    pub skipped: u32,
    /// Timed in pieces, the heaviest pieces: share of the frame's own time,
    /// ms sent alone, and what it is.
    pub pieces: Vec<(f64, f64, String)>,
    /// With each list waited for: lists by their time, as (lists, ms in
    /// all) of those over 16 ms, of 4 to 16, of 1 to 4 and under 1; and the
    /// heaviest: ms and what it is.
    pub list_times: [(usize, f64); 4],
    pub lists: Vec<(f64, String)>,
    /// The frame with and without each of the costliest pipelines, turn by
    /// turn: name, commands left out, ms with, ms without.
    pub left_out: Vec<(String, u32, f64, f64)>,
    /// Against the presented picture: dB (infinite: the same), and whether
    /// its bytes are the same.
    pub picture: Option<(f64, bool)>,
    /// Pixels of the last loop's picture that are not the presented one's,
    /// and the greatest difference of a channel (of 255).
    pub picture_differs: Option<(usize, u8)>,
    /// Pixels of each loop's picture (but the first) that are not the loop
    /// before's.
    pub loops_differ: Vec<usize>,
    /// Of each loop's picture: dB against the presented one, and its hash.
    pub pictures: Vec<(f64, String)>,
    /// Things to know: from the capture and the replay.
    pub notes: Vec<String>,
}

impl Report {
    pub fn median_ms(&self) -> f64 {
        median(&self.frame_ms)
    }

    /// The report as text.
    pub fn text(&self, what: &str) -> String {
        let mut t = String::new();
        let ms = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let how = match self.waits {
            Waits::Lists => "sum of the lists, each sent alone and waited for",
            Waits::Each => "sum of the submissions, each waited for",
            Waits::None => "all submissions sent at once, first one sent to last one done",
            Waits::OneSubmit => "one submission, sent to done",
        };
        let _ = writeln!(
            t,
            "{what}: {:.2} ms on the wall clock (median of {}; {how})",
            self.median_ms(),
            self.frame_ms.len()
        );
        let _ = writeln!(t, "  loops, ms: {}", ms(&self.frame_ms));
        if self.between_ms.iter().any(|&b| b > 0.0) {
            let _ = writeln!(
                t,
                "  between submissions, the CPU putting views and buffers in place (not counted above): {:.2} ms",
                median(&self.between_ms)
            );
        }
        if !self.stamp_frame_ms.is_empty() {
            let (frame, busy, delta) = (
                median(&self.stamp_frame_ms),
                median(&self.stamp_busy_ms),
                median(&self.stamp_delta_ms),
            );
            let _ = writeln!(
                t,
                "  by the GPU's timestamps: {frame:.2} ms from the first list's start to the last one's end; submissions {busy:.2} ms, copies standing for CPU writes {delta:.2} ms"
            );
        }
        let f = |v: Option<f64>| v.map_or("-".to_owned(), |v| format!("{v:.2}"));
        if self.submits.iter().any(|s| s.2.is_some()) {
            let _ = writeln!(
                t,
                "  by submission (lists: ms on the wall clock / by timestamps): {}",
                self.submits
                    .iter()
                    .enumerate()
                    .map(|(i, (n, wall, stamp))| format!("{i} ({n}): {} / {}", f(*wall), f(*stamp)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else if self.submits.iter().any(|s| s.1.is_some()) {
            let _ = writeln!(
                t,
                "  by submission (lists: ms on the wall clock): {}",
                self.submits
                    .iter()
                    .enumerate()
                    .map(|(i, (n, wall, _))| format!("{i} ({n}): {}", f(*wall)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if let Some(how) = self.detail {
            let _ = writeln!(t, "  commands timed {how}");
        }
        let share = |v: &Option<f64>| v.map_or(String::new(), |v| format!("{v:8.2}  "));
        let shared = self.categories.iter().any(|c| c.3.is_some());
        let heads = if shared {
            "ms of the frame's own time, ms sent alone (commands)"
        } else {
            "ms (commands)"
        };
        if !self.categories.is_empty() {
            let _ = writeln!(t, "  by kind of command, {heads}:");
            for (name, ms, n, of_frame) in &self.categories {
                let _ = writeln!(t, "    {}{ms:8.2}  {n:6}  {name}", share(of_frame));
            }
        }
        if !self.shaders.is_empty() {
            let _ = writeln!(t, "  costliest pipelines, {heads}:");
            for (ms, n, name, of_frame) in &self.shaders {
                let _ = writeln!(t, "    {}{ms:8.2}  {n:6}  {name}", share(of_frame));
            }
        }
        if !self.pieces.is_empty() {
            let _ = writeln!(t, "  heaviest pieces, {heads}:");
            for (of_frame, ms, what) in &self.pieces {
                let _ = writeln!(t, "    {of_frame:8.2}  {ms:8.2}  {what}");
            }
        }
        if self.waits == Waits::Lists {
            let [a, b, c, d] = self.list_times;
            let _ = writeln!(
                t,
                "  lists by their time: {} over 16 ms taking {:.1} ms, {} of 4 to 16 ms taking {:.1} ms, {} of 1 to 4 ms taking {:.1} ms, {} under 1 ms taking {:.1} ms",
                a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1
            );
            let _ = writeln!(t, "  heaviest lists, ms:");
            for (ms, what) in &self.lists {
                let _ = writeln!(t, "    {ms:8.2}  {what}");
            }
        }
        if !self.left_out.is_empty() {
            let _ = writeln!(
                t,
                "  the frame without each, turn by turn with the frame as it is: ms saved (ms with, ms without; commands left out):"
            );
            for (name, n, with, without) in &self.left_out {
                let _ = writeln!(
                    t,
                    "    {:8.2}  ({with:.2}, {without:.2}; {n})  {name}",
                    with - without
                );
            }
        }
        match self.picture {
            Some((_, true)) => _ = writeln!(t, "  picture: the same bytes as presented"),
            Some((db, false)) => {
                _ = writeln!(
                    t,
                    "  picture: differs from the presented one, PSNR {db:.1} dB{}",
                    self.picture_differs.map_or(String::new(), |(n, most)| format!(
                        ": {n} pixels differ, by {most} of 255 at the most"
                    ))
                )
            }
            None => {}
        }
        if !self.loops_differ.is_empty() {
            let _ = writeln!(
                t,
                "  pixels that differ from the loop before: {}",
                self.loops_differ
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        if self.pictures.len() > 1 {
            let all: Vec<String> = self
                .pictures
                .iter()
                .map(|(db, hash)| format!("{db:.1} dB {}", &hash[..8]))
                .collect();
            let kinds: HashSet<&String> = self.pictures.iter().map(|p| &p.1).collect();
            let _ = writeln!(
                t,
                "  pictures of the loops: {} different: {}",
                kinds.len(),
                all.join(", ")
            );
        }
        for n in &self.notes {
            let _ = writeln!(t, "  note: {n}");
        }
        t
    }
}

/// The stages of a pipeline as the runtime names them (`[mesh 0123456789abcdef,
/// pixel ...]`, the hashes those of the title's containers): for each of its
/// shader parts, in order, the stage and hash, if the name has them.
fn title_stages(p: &wire::Pipeline) -> Vec<(u32, Option<(&'static str, &str)>)> {
    p.parts
        .iter()
        .filter_map(|part| match part {
            wire::Part::Code { kind, blob } => Some((*kind, *blob)),
            _ => None,
        })
        .map(|(kind, blob)| {
            let stage = match D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(kind) {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS => "vertex",
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS => "pixel",
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS => "mesh",
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS => "compute",
                _ => return (blob, None),
            };
            let hash = p.name.find(stage).and_then(|at| {
                let at = at + stage.len() + 1;
                let hash = p.name.get(at..at + 16)?;
                hash.bytes().all(|b| b.is_ascii_hexdigit()).then_some(hash)
            });
            (blob, hash.map(|h| (stage, h)))
        })
        .collect()
}

/// The names a folder of shaders may have the code of pipeline `p`'s part
/// `blob` under, the first first: `<hash>@<the other stages' hashes>.dxil`
/// for a translation made for this pipeline alone (a mesh shader without
/// the outputs its pixel shader does not read), `<hash>.dxil` for the
/// shader's own; the hash is the one the runtime names the title's shader
/// by. A pass of ours has no such name: `<the first 8 bytes of its DXIL's
/// BLAKE3>.dxil`.
fn shader_files(frame: &Frame, p: &wire::Pipeline, blob: u32) -> Vec<String> {
    let stages = title_stages(p);
    match stages.iter().find(|s| s.0 == blob).and_then(|s| s.1) {
        Some((_, hash)) => {
            let others: Vec<&str> = stages
                .iter()
                .filter(|s| s.0 != blob)
                .filter_map(|s| Some(s.1?.1))
                .collect();
            let mut names = Vec::new();
            if !others.is_empty() {
                names.push(format!("{hash}@{}.dxil", others.join("+")));
            }
            names.push(format!("{hash}.dxil"));
            names
        }
        None => frame
            .codes
            .get(blob as usize)
            .map(|c| format!("{}.dxil", hex(&c.hash[..8])))
            .into_iter()
            .collect(),
    }
}

/// Loads a capture and runs it: the whole of `victory replay`. The reports,
/// with what each is of.
pub fn replay(folder: &Path, options: &Options) -> Result<Vec<(String, Report)>> {
    let mut player = Player::load(folder, options)?;
    replay_with(&mut player, options)
}

/// Runs a loaded capture as `options` say (those that decide how it is
/// loaded are the ones it was loaded with).
pub fn replay_with(player: &mut Player, options: &Options) -> Result<Vec<(String, Report)>> {
    let mut out = Vec::new();
    let two_sets = options.shaders_b.is_some();
    if options.compare && (options.edited() || options.shaders.is_some() || two_sets) {
        // Turn by turn: what else the machine does falls on both.
        let image = Some(player.folder.join("replay-unchanged.png"));
        let first = Options {
            image,
            // With two folders of shaders the first is one of them; else it
            // is the capture's own.
            shaders: if two_sets {
                options.shaders.clone()
            } else {
                None
            },
            ..options.unchanged()
        };
        let second = Options {
            shaders: options.shaders_b.clone().or(options.shaders.clone()),
            ..options.clone()
        };
        let mut a = player.prepare(&first)?;
        let mut b = player.prepare(&second)?;
        for _ in 0..options.loops.max(1) {
            player.once(&first, &mut a)?;
            player.once(&second, &mut b)?;
        }
        let name = |o: &Options, what: &str| match &o.shaders {
            Some(folder) => format!("{what}, shaders of {}", folder.display()),
            None => what.to_owned(),
        };
        let what = if options.edited() {
            "with the experiments"
        } else {
            "unchanged"
        };
        out.push((name(&first, "unchanged"), player.finish(&first, a)?));
        out.push((name(&second, what), player.finish(&second, b)?));
        return Ok(out);
    }
    let what = if options.edited() {
        "with the experiments"
    } else {
        "unchanged"
    };
    out.push((what.into(), player.run(options)?));
    Ok(out)
}

struct Object {
    res: ID3D12Resource,
    va: u64,
    /// In the capture's `resources`.
    index: usize,
    subs: u32,
    /// Where the CPU writes it, for a buffer it saw (0: nowhere).
    mapped: usize,
}

/// A buffer the CPU maps, as at the frame's start.
struct Mapped {
    at: usize,
    size: usize,
    /// The number of each of its pages (`plain::page_hash`; of a page of
    /// zeros it is zero).
    pages: Vec<u64>,
    /// Where the capture has its bytes at the frame's start: runs of whole
    /// pages (offset in the buffer, bytes, offset in `deltas.bin`), in
    /// order; what is in none is zero.
    runs: Vec<(u64, u64, u64)>,
}

// SAFETY: the mapping is of a buffer the player holds.
unsafe impl Send for Mapped {}
// SAFETY: as above; threads write parts of it that do not overlap.
unsafe impl Sync for Mapped {}

/// What mapped buffers are kept in units of, as the recorder does.
const PAGE: usize = 4096;

/// A shader-visible descriptor heap.
struct Version {
    heap: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
}

/// A recorded shader-visible heap in the replay.
struct Heap {
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
    count: u32,
    step: usize,
    /// Its views as at the frame's start, in a heap shaders do not see:
    /// what the others are copies of.
    _master: ID3D12DescriptorHeap,
    master_cpu: usize,
    /// The one views are written into between submissions that are waited
    /// for one by one.
    live: Version,
    /// One for each version of the frame's views, for submissions sent
    /// without waits: made when first asked for.
    versions: Vec<Version>,
}

pub struct Player {
    pub folder: PathBuf,
    frame: Frame,
    device: ID3D12Device,
    worker: Worker,
    info: Option<ID3D12InfoQueue>,
    /// Ticks of the GPU's timestamps in a second; 0 without them.
    frequency: f64,
    /// Whether timestamps tell the commands of a list apart.
    stamps_tell: bool,
    /// What a submission of an empty list takes, sent and waited for, in ms.
    empty_ms: f64,
    objects: HashMap<u32, Object>,
    _heaps: HashMap<u32, ID3D12Heap>,
    roots: HashMap<u32, ID3D12RootSignature>,
    /// The pipelines in use, and the folder their shaders are from.
    pipelines: HashMap<u32, ID3D12PipelineState>,
    shaders_from: Option<PathBuf>,
    /// Pipelines of other folders, which lists made with them still name.
    sets: Vec<(Option<PathBuf>, HashMap<u32, ID3D12PipelineState>)>,
    signatures: HashMap<u32, ID3D12CommandSignature>,
    queries: HashMap<u32, ID3D12QueryHeap>,
    descriptors: HashMap<u32, Heap>,
    /// What each version of the views changes against the one before
    /// (heap, slot, view; the first changes nothing), each submission's
    /// version, and the slots any version changes as at the frame's start.
    versions: Vec<Vec<(u32, u32, Slot)>>,
    version_of: Vec<usize>,
    first_views: Vec<(u32, u32, Slot)>,
    /// The version the live heaps hold.
    live_version: usize,
    /// Render target and depth views by what they are of.
    targets: HashMap<Slot, D3D12_CPU_DESCRIPTOR_HANDLE>,
    _target_heaps: Vec<ID3D12DescriptorHeap>,
    /// Copies of what the frame changes of the GPU's own memory, as at its
    /// start.
    masters: Vec<(u32, ID3D12Resource)>,
    /// The buffers the CPU maps, with the number of each of their pages at
    /// the frame's start and where the capture has their bytes: what a run
    /// changed is found by the numbers and put back from the capture
    /// itself, so no second copy of them is held.
    mapped: Vec<Mapped>,
    /// What the CPU wrote between submissions: the bytes, for each
    /// submission where in them, and a buffer of the GPU's with the same
    /// bytes (made when submissions are first sent without waits).
    delta_bytes: Vec<u8>,
    delta_parts: Vec<Vec<(Delta, u64)>>,
    deltas: Option<ID3D12Resource>,
    /// States at the start of the frame and after it.
    start: HashMap<u32, Vec<u32>>,
    end: HashMap<u32, Vec<u32>>,
    commands: Vec<Vec<Cmd>>,
    /// The names of the title's shaders.
    names: super::look::Names,
    /// The arguments the clamps of a run made ([`edit::RESULTS`]).
    results: Option<ID3D12Resource>,
    /// Resources in the states of the start (not those of the end).
    fresh: bool,
    /// What the last restore put back of the buffers the CPU maps: bytes,
    /// and the seconds it took.
    restored: (usize, f64),
    notes: Vec<String>,
}

/// What a command of a list is timed as: its timestamp, its kind and the
/// pipeline it runs.
type Stamp = (usize, &'static str, u32);

/// With a stamp of a copy to a texture: this bit and the texture's number
/// stand for the pipeline's.
const COPY_TO: u32 = 1 << 31;

/// A list made from commands, with what its timestamps are of.
struct Built {
    _allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// The query of its start and of its end.
    first: u32,
    last: u32,
    stamps: Vec<Stamp>,
}

/// Everything to send for a frame.
struct Sends {
    /// The frame's commands as they are sent (edited), by submission and
    /// list.
    submits: Vec<Vec<Vec<Cmd>>>,
    /// For each submission: the copies of what the CPU wrote (when sent
    /// without waits), its lists.
    steps: Vec<(Option<Built>, Vec<Built>)>,
    joints: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    resolve: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    _heap: Option<ID3D12QueryHeap>,
    _back: Option<ID3D12Resource>,
    data: usize,
    count: u32,
    /// Whether every command has a timestamp.
    stamped: bool,
}

/// One run of the frame.
struct Ran {
    stamps: Vec<u64>,
    /// Each submission's time on the wall clock, when waited for one by one.
    walls: Vec<f64>,
    /// Each list's, in the frame's order, when each was sent alone.
    lists: Vec<f64>,
    frame: f64,
    between: f64,
}

/// The time of pieces: ms sent alone, ms of the frame's own time (scaled so
/// that each submission's pieces add up to what it took whole), commands.
type Timed = (f64, f64, u32);

/// A loop's times in pieces: by kind, by pipeline, each piece (of the
/// frame's own time, and sent alone), and by pipeline what its pieces took
/// more than a piece that does next to nothing.
type Pieces = (
    BTreeMap<&'static str, Timed>,
    HashMap<u32, Timed>,
    Vec<(f64, f64)>,
    HashMap<u32, f64>,
);

/// A frame made ready to run, and what its runs gave so far.
pub struct Prepared {
    sends: Sends,
    report: Report,
    runs: Vec<Ran>,
    /// Per loop of timing in pieces.
    pieces: Vec<Pieces>,
    /// The picture its last run left ([`Player::picture_now`]).
    picture: Option<Picture>,
}

/// The presented picture after a run.
struct Picture {
    /// Against the recorded one: dB, whether its bytes are the same, the
    /// pixels that differ and the greatest difference of a channel.
    db: f64,
    same: bool,
    differs: (usize, u8),
    hash: String,
    rgb: Vec<u8>,
}

/// The pixels (of three bytes) that differ between two pictures, and the
/// greatest difference of a channel.
fn differing(a: &[u8], b: &[u8]) -> (usize, u8) {
    let (mut n, mut most) = (0, 0);
    for (x, y) in a.chunks_exact(3).zip(b.chunks_exact(3)) {
        if x != y {
            n += 1;
            most = most.max(x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0));
        }
    }
    (n, most)
}

impl Player {
    pub fn load(folder: &Path, options: &Options) -> Result<Player> {
        let folder = find(folder)?;
        let began = Instant::now();
        let frame = Frame::read(&folder.join("frame.bin"))?;
        let (device, info) = make_device(&frame, options.debug)?;
        // SAFETY: plain creation and queries.
        let (queue, frequency) = unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let frequency = queue.GetTimestampFrequency().unwrap_or(0) as f64;
            (queue, frequency)
        };
        let commands: Vec<Vec<Cmd>> = frame
            .lists
            .iter()
            .map(|l| Cmd::all(&l.commands.0))
            .collect::<std::result::Result<_, _>>()?;
        let mut player = Player {
            folder,
            worker: Worker::new(&device, &queue)?,
            device,
            info,
            frequency,
            stamps_tell: false,
            empty_ms: 0.0,
            objects: HashMap::new(),
            _heaps: HashMap::new(),
            roots: HashMap::new(),
            pipelines: HashMap::new(),
            shaders_from: None,
            sets: Vec::new(),
            signatures: HashMap::new(),
            queries: HashMap::new(),
            descriptors: HashMap::new(),
            versions: Vec::new(),
            version_of: Vec::new(),
            first_views: Vec::new(),
            live_version: 0,
            targets: HashMap::new(),
            _target_heaps: Vec::new(),
            masters: Vec::new(),
            mapped: Vec::new(),
            delta_bytes: Vec::new(),
            delta_parts: Vec::new(),
            deltas: None,
            start: HashMap::new(),
            end: HashMap::new(),
            commands,
            names: super::look::Names::read(options.names.as_deref())?,
            results: None,
            fresh: true,
            restored: (0, 0.0),
            notes: frame
                .notes
                .iter()
                .map(|n| format!("recorder: {n}"))
                .collect(),
            frame,
        };
        player.check()?;
        let mut steps: Vec<(&str, Duration)> = Vec::new();
        let mut step = |what: &'static str, at: &mut Instant| {
            steps.push((what, at.elapsed()));
            *at = Instant::now();
        };
        let mut at = Instant::now();
        player.measure_host()?;
        step("the host's clocks", &mut at);
        player.make_resources()?;
        step("resources", &mut at);
        player.shaders_from = options.shaders.clone();
        player.make_pipelines(options.shaders.as_deref())?;
        step("pipelines", &mut at);
        player.make_descriptors()?;
        step("descriptors", &mut at);
        player.load_contents()?;
        step("contents", &mut at);
        player.make_masters(options.restore)?;
        step("copies to restore from", &mut at);
        let steps: Vec<String> = steps
            .iter()
            .map(|(w, d)| format!("{w} {:.1} s", d.as_secs_f64()))
            .collect();
        if let Some([budget, used, other]) = player.memory() {
            player.notes.push(format!(
                "{used} MB of video memory used (and {other} MB of the other kind), budget {budget} MB"
            ));
        }
        player.notes.push(format!(
            "loaded in {:.1} s: {}",
            began.elapsed().as_secs_f64(),
            steps.join(", ")
        ));
        Ok(player)
    }

    fn note(&mut self, n: String) {
        tracing::warn!("{n}");
        self.notes.push(n);
    }

    /// Says what of the frame is not there.
    fn check(&mut self) -> Result<()> {
        if self.frame.partial {
            self.note(
                "the capture is not whole: it ends before the frame's present (the title ended in the middle of it, or the capture failed)"
                    .into(),
            );
        }
        let missing = self
            .frame
            .submits
            .iter()
            .flat_map(|s| &s.lists)
            .filter(|&&l| l == MISSING)
            .count();
        if missing > 0 {
            self.note(format!(
                "{missing} lists of the frame were not recorded: the frame is incomplete"
            ));
        }
        if self.frame.submits.is_empty() {
            bail!("the capture has no submission");
        }
        let unknown = self
            .commands
            .iter()
            .flatten()
            .filter(|c| {
                matches!(
                    c,
                    Cmd::RootView {
                        va: Va::Unknown { .. },
                        ..
                    }
                )
            })
            .count();
        if unknown > 0 {
            self.note(format!(
                "{unknown} root views name memory in no known buffer: they are null in the replay"
            ));
        }
        Ok(())
    }

    /// What the host's clocks give: whether timestamps tell the commands of
    /// a list apart, and what a submission costs that does nothing.
    fn measure_host(&mut self) -> Result<()> {
        // An empty list, sent and waited for.
        let mut empties = Vec::new();
        for _ in 0..40 {
            let began = Instant::now();
            self.worker.run(|_| {})?;
            empties.push(began.elapsed().as_secs_f64() * 1e3);
        }
        self.empty_ms = median(&empties[8..]);
        let mut note = format!(
            "a submission that does nothing takes {:.3} ms sent and waited for",
            self.empty_ms
        );
        if self.frequency <= 0.0 {
            note.push_str("; the host has no timestamps");
            self.notes.push(note);
            return Ok(());
        }
        // Timestamps around nothing, a copy of 64 MB, and four of them.
        const SIZE: u64 = 64 << 20;
        let make = |kind| gpu::buffer(&self.device, kind, SIZE, D3D12_RESOURCE_FLAG_NONE);
        let (a, b) = (make(D3D12_HEAP_TYPE_DEFAULT)?.0, make(D3D12_HEAP_TYPE_DEFAULT)?.0);
        let (back, data) = gpu::buffer(
            &self.device,
            D3D12_HEAP_TYPE_READBACK,
            64,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        let mut heap: Option<ID3D12QueryHeap> = None;
        let desc = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: 4,
            NodeMask: 0,
        };
        // SAFETY: plain creation.
        if unsafe { self.device.CreateQueryHeap(&desc, &mut heap) }.is_err() {
            self.frequency = 0.0;
            note.push_str("; the host makes no timestamp queries");
            self.notes.push(note);
            return Ok(());
        }
        let heap = heap.context("no timestamps")?;
        let mut tells = Vec::new();
        let mut walls = Vec::new();
        for _ in 0..3 {
            let began = Instant::now();
            self.worker.run(|list| {
                // SAFETY: an open list; live buffers and queries of the heap.
                unsafe {
                    let stamp = |i| list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
                    stamp(0);
                    stamp(1);
                    list.CopyBufferRegion(&b, 0, &a, 0, SIZE);
                    stamp(2);
                    for _ in 0..4 {
                        list.CopyBufferRegion(&a, 0, &b, 0, SIZE);
                    }
                    stamp(3);
                    list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, &back, 0);
                }
            })?;
            walls.push(began.elapsed().as_secs_f64() * 1e3);
            // SAFETY: the mapping of four resolved timestamps.
            let s = unsafe { data.cast::<[u64; 4]>().read_unaligned() };
            let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * 1e3 / self.frequency;
            tells.push([ms(s[0], s[1]), ms(s[1], s[2]), ms(s[2], s[3])]);
        }
        let last = tells[tells.len() - 1];
        if last == [0.0; 3] {
            // D3DMetal (4.0 beta 3) answers a frequency and takes the
            // queries, and says "Unsupported" for each: no timestamps.
            self.frequency = 0.0;
            let _ = write!(
                note,
                "; the host's timestamp queries do nothing (all read zero): only the wall clock tells times, and commands are timed in pieces (the list of copies of 320 MB took {:.2} ms sent and waited for)",
                walls[walls.len() - 1]
            );
            self.notes.push(note);
            return Ok(());
        }
        // A copy takes time, four take more, nothing takes next to none.
        self.stamps_tell = last[1] > 0.0 && last[2] > last[1] * 1.5 && last[0] < last[1] * 0.5;
        let _ = write!(
            note,
            "; timestamps ({} ticks a second) around nothing, a copy of 64 MB and four such copies: {:.3}, {:.3} and {:.3} ms (the list took {:.2} ms sent and waited for): {}",
            self.frequency,
            last[0],
            last[1],
            last[2],
            walls[walls.len() - 1],
            if self.stamps_tell {
                "they tell the commands of a list apart"
            } else {
                "they do not tell the commands of a list apart"
            }
        );
        self.notes.push(note);
        Ok(())
    }

    fn resource(&self, id: u32) -> Option<&ID3D12Resource> {
        if id == edit::RESULTS {
            return self.results.as_ref();
        }
        self.objects.get(&id).map(|o| &o.res)
    }

    fn va(&self, va: &Va) -> u64 {
        match va {
            Va::At { res, offset } if *res == edit::RESULTS => {
                // SAFETY: a plain query on a live buffer.
                self.results.as_ref().map_or(0, |r| unsafe { r.GetGPUVirtualAddress() } + offset)
            }
            Va::At { res, offset } => self.objects.get(res).map_or(0, |o| o.va + offset),
            _ => 0,
        }
    }

    fn is_buffer(&self, id: u32) -> bool {
        self.objects.get(&id).is_some_and(|o| {
            self.frame.resources[o.index].desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0
        })
    }

    fn make_resources(&mut self) -> Result<()> {
        // A buffer the CPU saw becomes one the CPU maps and the GPU may
        // write too (as the host layer's copies of title memory are): what
        // the CPU wrote is put there by either.
        let seen_by_cpu = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let props = |p: &wire::HeapProps, buffer: bool| -> D3D12_HEAP_PROPERTIES {
            match gpu::seen(p) {
                Seen::Cached | Seen::Combined if buffer => seen_by_cpu,
                _ => gpu::to_props(p),
            }
        };
        // Flags a replay has no use for: sharing, display, not zeroed.
        const DROP: i32 = 0x1 | 0x8 | 0x20 | 0x1000;
        for h in &self.frame.heaps {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: h.size,
                Properties: props(&h.props, false),
                Alignment: h.alignment,
                Flags: D3D12_HEAP_FLAGS(h.flags & !DROP),
            };
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: plain creation.
            unsafe { self.device.CreateHeap(&desc, &mut heap) }
                .with_context(|| format!("heap {}", h.id))?;
            self._heaps.extend(heap.map(|heap| (h.id, heap)));
        }
        let newer = self.device.cast::<ID3D12Device10>().ok();
        let mut failed = 0;
        let mut plain_made = 0;
        for (index, r) in self.frame.resources.iter().enumerate() {
            let is_buffer = r.desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0;
            let desc = gpu::to_desc(&r.desc);
            let mut made: Option<ID3D12Resource> = None;
            let clear: Option<D3D12_CLEAR_VALUE> = r.clear.map(|c| {
                // SAFETY: 20 bytes that were a clear value.
                unsafe { std::mem::transmute::<[u32; 5], D3D12_CLEAR_VALUE>(c) }
            });
            let clear = clear.as_ref().map(|c| c as *const _);
            let castable: Vec<DXGI_FORMAT> = r.castable.iter().map(|&f| DXGI_FORMAT(f)).collect();
            let desc1 = D3D12_RESOURCE_DESC1 {
                Dimension: desc.Dimension,
                Alignment: desc.Alignment,
                Width: desc.Width,
                Height: desc.Height,
                DepthOrArraySize: desc.DepthOrArraySize,
                MipLevels: desc.MipLevels,
                Format: desc.Format,
                SampleDesc: desc.SampleDesc,
                Layout: desc.Layout,
                Flags: desc.Flags,
                ..Default::default()
            };
            let mut mapped = false;
            // SAFETY: plain creation from a description the host took.
            let result = unsafe {
                match (&r.place, &newer) {
                    // With formats to view it in: made as the host layer
                    // makes it, in the common layout.
                    (Place::Placed { heap, offset }, Some(device)) if !castable.is_empty() => {
                        match self._heaps.get(heap) {
                            Some(h) => device.CreatePlacedResource2(
                                h,
                                *offset,
                                &desc1,
                                D3D12_BARRIER_LAYOUT_COMMON,
                                clear,
                                Some(&castable),
                                &mut made,
                            ),
                            None => Err(windows::core::Error::empty()),
                        }
                    }
                    (Place::Committed { props: p, flags }, Some(device))
                        if !castable.is_empty() =>
                    {
                        device.CreateCommittedResource3(
                            &props(p, is_buffer),
                            D3D12_HEAP_FLAGS(flags & !DROP),
                            &desc1,
                            D3D12_BARRIER_LAYOUT_COMMON,
                            clear,
                            None::<&ID3D12ProtectedResourceSession>,
                            Some(&castable),
                            &mut made,
                        )
                    }
                    (Place::Placed { heap, offset }, _) => match self._heaps.get(heap) {
                        Some(h) => self.device.CreatePlacedResource(
                            h,
                            *offset,
                            &desc,
                            D3D12_RESOURCE_STATE_COMMON,
                            clear,
                            &mut made,
                        ),
                        None => Err(windows::core::Error::empty()),
                    },
                    (Place::Committed { props: p, flags }, _) => {
                        let p = props(p, is_buffer);
                        mapped = is_buffer && p.Type == D3D12_HEAP_TYPE_CUSTOM;
                        let state = if p.Type == D3D12_HEAP_TYPE_READBACK {
                            D3D12_RESOURCE_STATE_COPY_DEST
                        } else {
                            D3D12_RESOURCE_STATE_COMMON
                        };
                        self.device.CreateCommittedResource(
                            &p,
                            D3D12_HEAP_FLAGS(flags & !DROP),
                            &desc,
                            state,
                            clear,
                            &mut made,
                        )
                    }
                    (Place::BackBuffer {}, _) => {
                        let p = D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        };
                        self.device.CreateCommittedResource(
                            &p,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COMMON,
                            None,
                            &mut made,
                        )
                    }
                }
            };
            plain_made += u32::from(!castable.is_empty() && newer.is_none());
            match (result, made) {
                (Ok(()), Some(res)) => {
                    // SAFETY: a live resource.
                    let va = if is_buffer {
                        unsafe { res.GetGPUVirtualAddress() }
                    } else {
                        0
                    };
                    let mut data = std::ptr::null_mut();
                    if mapped {
                        // SAFETY: a buffer in memory the CPU sees.
                        unsafe { res.Map(0, None, Some(&mut data)) }
                            .with_context(|| format!("buffer {} not mapped", r.id))?;
                    }
                    self.objects.insert(
                        r.id,
                        Object {
                            res,
                            va,
                            index,
                            subs: gpu::subresources(&r.desc),
                            mapped: data as usize,
                        },
                    );
                }
                (e, _) => {
                    if failed == 0 {
                        tracing::warn!("resource {} not made: {e:?}: {r:?}", r.id);
                    }
                    failed += 1;
                }
            }
            if !is_buffer {
                self.start.insert(r.id, r.state.clone());
            }
        }
        if failed > 0 {
            self.note(format!("{failed} resources could not be made again"));
        }
        if plain_made > 0 {
            self.note(format!(
                "{plain_made} textures were recorded with other formats to view them in, which this host does not take"
            ));
        }
        Ok(())
    }

    /// Makes the root signatures, command signatures, query heaps and
    /// pipelines; the pipelines with shaders of `shaders` where it has them.
    fn make_pipelines(&mut self, shaders: Option<&Path>) -> Result<()> {
        for (id, blob) in &self.frame.roots {
            if self.roots.contains_key(id) {
                // (Made for another set of shaders: the root signatures are
                // the same.)
                continue;
            }
            // SAFETY: a blob the host took.
            let root = unsafe { self.device.CreateRootSignature(0, &blob.0) }
                .with_context(|| format!("root signature {id}"))?;
            self.roots.insert(*id, root);
        }
        for s in &self.frame.signatures {
            if self.signatures.contains_key(&s.id) {
                continue;
            }
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: s.stride,
                NumArgumentDescs: s.args.len() as u32,
                pArgumentDescs: s.args.as_ptr().cast(),
                NodeMask: 0,
            };
            let mut made: Option<ID3D12CommandSignature> = None;
            // SAFETY: a description the host took.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, self.roots.get(&s.root), &mut made)?
            };
            self.signatures.extend(made.map(|m| (s.id, m)));
        }
        for q in &self.frame.query_heaps {
            if self.queries.contains_key(&q.id) {
                continue;
            }
            let desc = D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE(q.kind),
                Count: q.count,
                NodeMask: 0,
            };
            let mut made: Option<ID3D12QueryHeap> = None;
            // SAFETY: plain creation.
            unsafe { self.device.CreateQueryHeap(&desc, &mut made)? };
            self.queries.extend(made.map(|m| (q.id, m)));
        }
        let file = File::open(self.folder.join("shaders.bin"))?;
        let device: ID3D12Device2 = self.device.cast()?;
        let next = AtomicUsize::new(0);
        type Made = std::result::Result<ID3D12PipelineState, String>;
        let made: std::sync::Mutex<Vec<(u32, Made)>> = std::sync::Mutex::default();
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .max(1);
        let (frame, roots) = (&self.frame, &self.roots);
        let (swapped, kept) = (AtomicUsize::new(0), AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    while let Some(p) = frame.pipelines.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let recorded = |i: u32| {
                            let c = frame.codes.get(i as usize)?;
                            let mut bytes = vec![0; c.len as usize];
                            read_at(&file, &mut bytes, c.at).ok()?;
                            Some(bytes)
                        };
                        let root = |id: u32| roots.get(&id).cloned();
                        let taken = std::cell::Cell::new(false);
                        let mut m = super::pso::make(&device, &p.parts, root, |i| {
                            let other = shaders.and_then(|dir| {
                                shader_files(frame, p, i)
                                    .iter()
                                    .find_map(|name| crate::read_file(&dir.join(name)).ok())
                            });
                            taken.set(taken.get() || other.is_some());
                            other.or_else(|| recorded(i))
                        });
                        if taken.get() && m.is_err() {
                            // Not for the root signature or the other stage
                            // recorded.
                            kept.fetch_add(1, Ordering::Relaxed);
                            m = super::pso::make(&device, &p.parts, root, recorded);
                        } else if taken.get() {
                            swapped.fetch_add(1, Ordering::Relaxed);
                        }
                        made.lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push((p.id, m));
                    }
                });
            }
        });
        let mut refused = Vec::new();
        let mut pipelines = HashMap::new();
        for (id, m) in made.into_inner().unwrap_or_else(|e| e.into_inner()) {
            match m {
                Ok(p) => drop(pipelines.insert(id, p)),
                Err(e) => refused.push((id, e)),
            }
        }
        self.pipelines = pipelines;
        if let Some(dir) = shaders {
            let n = format!(
                "{} of {} pipelines made with shaders of {}, {} refused with them (made as recorded)",
                swapped.into_inner(),
                self.frame.pipelines.len(),
                dir.display(),
                kept.into_inner()
            );
            self.note(n);
        }
        if let Some((id, e)) = refused.first() {
            let n = format!(
                "{} pipelines could not be made again, first {}: {e}",
                refused.len(),
                self.pipeline_name(*id)
            );
            self.note(n);
        }
        Ok(())
    }

    /// The pipelines made with the shaders of `shaders` are the ones in use
    /// from now on; those in use before are kept (lists made with them name
    /// them).
    fn use_shaders(&mut self, shaders: Option<&Path>) -> Result<()> {
        if self.shaders_from.as_deref() == shaders {
            return Ok(());
        }
        let old = (
            self.shaders_from.take(),
            std::mem::take(&mut self.pipelines),
        );
        self.sets.push(old);
        self.shaders_from = shaders.map(Path::to_owned);
        match self.sets.iter().position(|s| s.0.as_deref() == shaders) {
            Some(i) => self.pipelines = self.sets.swap_remove(i).1,
            None => self.make_pipelines(shaders)?,
        }
        Ok(())
    }

    /// A pipeline's name for reports: what the host layer called it and the
    /// hashes of its shaders.
    fn pipeline_name(&self, id: u32) -> String {
        let Some(p) = self.frame.pipelines.iter().find(|p| p.id == id) else {
            return format!("pipeline {id}");
        };
        let code: Vec<String> = self.hashes(p);
        format!(
            "{} (DXIL {})",
            if p.name.is_empty() {
                "unnamed".to_owned()
            } else {
                self.names.of(&p.name)
            },
            code.join(" ")
        )
    }

    /// What pipelines are counted together by: the shaders the host layer
    /// named them by (several pipelines of one set of shaders differ in
    /// their states only).
    fn group(&self, id: u32) -> String {
        match self.frame.pipelines.iter().find(|p| p.id == id) {
            Some(p) if !p.name.is_empty() => p.name.clone(),
            _ => format!("pipeline {id}"),
        }
    }

    /// A resource in a few words.
    fn resource_name(&self, id: u32) -> String {
        match self.objects.get(&id) {
            Some(o) => {
                let d = &self.frame.resources[o.index].desc;
                if d.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0 {
                    format!("buffer #{id}")
                } else if d.depth > 1 {
                    format!("#{id} {}x{}x{} f{}", d.width, d.height, d.depth, d.format)
                } else {
                    format!("#{id} {}x{} f{}", d.width, d.height, d.format)
                }
            }
            None => format!("#{id}"),
        }
    }

    /// What commands do, in a line: how many of which kind, with which
    /// pipelines, into which targets.
    fn describe(&self, cmds: &[Cmd]) -> String {
        let (mesh, dispatches) = self.kinds();
        let mut kinds: BTreeMap<&str, u32> = BTreeMap::new();
        let mut pipelines: HashMap<String, u32> = HashMap::new();
        let mut targets: Vec<String> = Vec::new();
        let (mut groups, mut largest) = (0u64, [0u32; 3]);
        let mut bound = NONE;
        let mut now = String::new();
        for c in cmds {
            match c {
                Cmd::Pipeline { pso } => bound = *pso,
                Cmd::Targets { colors, depth } => {
                    let mut all: Vec<String> =
                        colors.iter().map(|s| self.resource_name(s.res)).collect();
                    all.extend(depth.iter().map(|s| format!("depth {}", self.resource_name(s.res))));
                    now = match all.len() {
                        0..=2 => all.join(" + "),
                        n => format!("{} + {} more", all[0], n - 1),
                    };
                }
                Cmd::Dispatch { groups: g } => {
                    let n: u64 = g.iter().map(|&g| u64::from(g)).product();
                    groups += n;
                    if n > largest.iter().map(|&g| u64::from(g)).product() {
                        largest = *g;
                    }
                }
                _ => {}
            }
            if !c.works() {
                continue;
            }
            *kinds.entry(edit::kind(c, &dispatches, &mesh, bound)).or_default() += 1;
            if c.runs_shaders() {
                *pipelines.entry(self.group(bound)).or_default() += 1;
                let draws = !matches!(c, Cmd::Dispatch { .. })
                    && !matches!(c, Cmd::Indirect { signature, .. } if dispatches.contains(signature));
                if draws && !now.is_empty() && !targets.contains(&now) {
                    targets.push(now.clone());
                }
            }
        }
        let mut kinds: Vec<(&str, u32)> = kinds.into_iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(&a.1));
        let mut text = kinds
            .iter()
            .take(4)
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(", ");
        if groups > 0 {
            let _ = write!(
                text,
                " ({groups} thread groups, the largest {}x{}x{})",
                largest[0], largest[1], largest[2]
            );
        }
        let mut pipelines: Vec<(String, u32)> = pipelines.into_iter().collect();
        pipelines.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if !pipelines.is_empty() {
            let _ = write!(
                text,
                "; {}{}",
                pipelines
                    .iter()
                    .take(2)
                    .map(|(name, n)| format!("{n} with {}", self.names.of(name)))
                    .collect::<Vec<_>>()
                    .join(", "),
                match pipelines.len() {
                    0..=2 => String::new(),
                    n => format!(", {} other pipelines", n - 2),
                }
            );
        }
        if !targets.is_empty() {
            let _ = write!(
                text,
                "; into {}{}",
                targets[0],
                match targets.len() {
                    1 => String::new(),
                    n => format!(" and {} other sets of targets", n - 1),
                }
            );
        }
        text
    }

    fn hashes(&self, p: &wire::Pipeline) -> Vec<String> {
        p.parts
            .iter()
            .filter_map(|part| match part {
                wire::Part::Code { blob, .. } => {
                    let c = self.frame.codes.get(*blob as usize)?;
                    Some(hex(&c.hash[..8]))
                }
                _ => None,
            })
            .collect()
    }

    /// Makes a view; one of a resource that is not there is null.
    ///
    /// # Safety
    /// `at` is a slot of a heap of the view's type.
    unsafe fn view(&self, s: &Slot, at: D3D12_CPU_DESCRIPTOR_HANDLE, sampler: bool) {
        let gone = s.res != NONE && !self.objects.contains_key(&s.res);
        let res = self.objects.get(&s.res).map(|o| &o.res);
        let counter = self.objects.get(&s.counter).map(|o| &o.res);
        // The descriptions lie in the slot unaligned.
        // SAFETY: the slot's description is of its kind; the resources live.
        unsafe {
            match (s.kind, s.described != 0 && !gone) {
                (slot::CBV, described) => {
                    let mut d: D3D12_CONSTANT_BUFFER_VIEW_DESC = Default::default();
                    if described && let Some(o) = self.objects.get(&s.res) {
                        d = s
                            .desc
                            .as_ptr()
                            .cast::<D3D12_CONSTANT_BUFFER_VIEW_DESC>()
                            .read_unaligned();
                        d.BufferLocation = o.va + s.address();
                    }
                    self.device.CreateConstantBufferView(Some(&d), at);
                }
                (slot::SRV, true) => {
                    let d = s
                        .desc
                        .as_ptr()
                        .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
                        .read_unaligned();
                    self.device.CreateShaderResourceView(res, Some(&d), at);
                }
                (slot::SRV, false) if res.is_some() => {
                    self.device.CreateShaderResourceView(res, None, at)
                }
                (slot::UAV, true) => {
                    let d = s
                        .desc
                        .as_ptr()
                        .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
                        .read_unaligned();
                    self.device
                        .CreateUnorderedAccessView(res, counter, Some(&d), at);
                }
                (slot::UAV, false) if res.is_some() => {
                    self.device
                        .CreateUnorderedAccessView(res, None::<&ID3D12Resource>, None, at)
                }
                (slot::SAMPLER, true) => {
                    let d = s
                        .desc
                        .as_ptr()
                        .cast::<D3D12_SAMPLER_DESC>()
                        .read_unaligned();
                    self.device.CreateSampler(&d, at);
                }
                _ if sampler => {
                    let d = D3D12_SAMPLER_DESC {
                        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        ..Default::default()
                    };
                    self.device.CreateSampler(&d, at);
                }
                (slot::UAV, _) => {
                    let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT(41),
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        ..Default::default()
                    };
                    self.device.CreateUnorderedAccessView(
                        None::<&ID3D12Resource>,
                        None::<&ID3D12Resource>,
                        Some(&d),
                        at,
                    );
                }
                _ => {
                    let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT(41),
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        ..Default::default()
                    };
                    self.device
                        .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), at);
                }
            }
        }
    }

    /// A shader-visible heap with the views of `from` (a heap of as many
    /// slots that shaders do not see).
    fn visible(
        &self,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        count: u32,
        from: usize,
    ) -> windows::core::Result<Version> {
        let d = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: count,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: plain creation; whole heaps of one size and kind.
        unsafe {
            let heap: ID3D12DescriptorHeap = self.device.CreateDescriptorHeap(&d)?;
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            self.device.CopyDescriptorsSimple(
                count,
                cpu,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: from },
                kind,
            );
            let gpu = heap.GetGPUDescriptorHandleForHeapStart().ptr;
            Ok(Version {
                heap,
                cpu: cpu.ptr,
                gpu,
            })
        }
    }

    fn make_descriptors(&mut self) -> Result<()> {
        // A submission whose views replace ones that were in use starts a
        // version of the views: sent without waits, it binds heaps of its
        // own (a heap cannot be written while lists that use it wait).
        let mut now: HashMap<(u32, u32), Slot> = HashMap::new();
        let replaced: HashSet<(u32, u32)> = self
            .frame
            .submits
            .iter()
            .flat_map(|s| s.views.iter().map(|v| (v.0, v.1)))
            .collect();
        for h in &self.frame.descriptor_heaps {
            for (id, i) in replaced.iter().filter(|k| k.0 == h.id) {
                let k = h.runs.partition_point(|run| run.0 <= *i);
                let first = match k.checked_sub(1).and_then(|k| h.runs.get(k)) {
                    Some((first, n, s)) if i - first < *n => *s,
                    _ => Slot::EMPTY,
                };
                now.insert((*id, *i), first);
            }
        }
        self.first_views = now.iter().map(|(k, s)| (k.0, k.1, *s)).collect();
        let mut versions: Vec<Vec<(u32, u32, Slot)>> = vec![Vec::new()];
        let mut touched: HashSet<(u32, u32)> = HashSet::new();
        for s in &self.frame.submits {
            let live =
                |old: Option<&Slot>| old.is_some_and(|o| o.kind == slot::SAMPLER || o.res != NONE);
            let news: Vec<&(u32, u32, Slot)> = s
                .views
                .iter()
                .filter(|(h, i, v)| now.get(&(*h, *i)) != Some(v))
                .collect();
            if news
                .iter()
                .any(|(h, i, _)| live(now.get(&(*h, *i))) || touched.contains(&(*h, *i)))
            {
                versions.push(Vec::new());
                touched.clear();
            }
            for (h, i, v) in news {
                now.insert((*h, *i), *v);
                touched.insert((*h, *i));
                if let Some(last) = versions.last_mut() {
                    last.push((*h, *i, *v));
                }
            }
            self.version_of.push(versions.len() - 1);
        }
        let mut made = HashMap::new();
        for h in &self.frame.descriptor_heaps {
            let sampler = h.kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0;
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE(h.kind);
            let d = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: h.count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            // SAFETY: plain creation; slots of the heap just made.
            unsafe {
                let master: ID3D12DescriptorHeap = self.device.CreateDescriptorHeap(&d)?;
                let step = self.device.GetDescriptorHandleIncrementSize(kind) as usize;
                let start = master.GetCPUDescriptorHandleForHeapStart();
                let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + i as usize * step,
                };
                // Null views everywhere: one, doubled.
                self.view(&Slot::EMPTY, start, sampler);
                let mut filled = 1;
                while filled < h.count {
                    let n = filled.min(h.count - filled).min(1 << 16);
                    // Sources and destinations do not overlap.
                    self.device
                        .CopyDescriptorsSimple(n, at(filled), start, kind);
                    filled += n;
                }
                for (first, n, s) in &h.runs {
                    if *first >= h.count {
                        continue;
                    }
                    self.view(s, at(*first), sampler);
                    let n = (*n).min(h.count - first);
                    let mut filled = 1;
                    while filled < n {
                        let m = filled.min(n - filled).min(1 << 16);
                        self.device.CopyDescriptorsSimple(
                            m,
                            at(first + filled),
                            at(*first),
                            kind,
                        );
                        filled += m;
                    }
                }
                let live = self.visible(kind, h.count, start.ptr)?;
                made.insert(
                    h.id,
                    Heap {
                        kind,
                        count: h.count,
                        step,
                        _master: master,
                        master_cpu: start.ptr,
                        live,
                        versions: Vec::new(),
                    },
                );
            }
        }
        self.descriptors = made;
        if versions.len() > 1 {
            self.notes.push(format!(
                "views in use are replaced during the frame: {} versions of them ({} slots)",
                versions.len(),
                self.first_views.len()
            ));
        }
        self.versions = versions;
        if let Some(first) = self.versions.first() {
            self.write_live(first);
        }
        // Render target and depth views, and the views unordered-access
        // clears name by a CPU handle.
        let mut views: [HashSet<Slot>; 3] = Default::default();
        for c in self.commands.iter().flatten() {
            match c {
                Cmd::Targets { colors, depth } => {
                    views[0].extend(colors);
                    views[1].extend(depth);
                }
                Cmd::ClearTarget { view, .. } => _ = views[0].insert(*view),
                Cmd::ClearDepth { view, .. } => _ = views[1].insert(*view),
                Cmd::ClearUnordered { view, .. } => _ = views[2].insert(*view),
                _ => {}
            }
        }
        for (i, all) in views.iter().enumerate() {
            if all.is_empty() {
                continue;
            }
            let kind = [
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ][i];
            let d = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: all.len() as u32,
                ..Default::default()
            };
            // SAFETY: plain creation; slots of the heap just made.
            unsafe {
                let heap: ID3D12DescriptorHeap = self.device.CreateDescriptorHeap(&d)?;
                let step = self.device.GetDescriptorHandleIncrementSize(kind) as usize;
                let mut at = heap.GetCPUDescriptorHandleForHeapStart();
                for s in all {
                    let res = self.objects.get(&s.res).map(|o| &o.res);
                    if i == 2 {
                        self.view(s, at, false);
                    } else if i == 0 {
                        let d = s
                            .desc
                            .as_ptr()
                            .cast::<D3D12_RENDER_TARGET_VIEW_DESC>()
                            .read_unaligned();
                        // Without either there is nothing to make a view of.
                        let null = D3D12_RENDER_TARGET_VIEW_DESC {
                            Format: DXGI_FORMAT(28),
                            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                            ..Default::default()
                        };
                        let d = if s.described != 0 {
                            Some(&d as *const _)
                        } else if res.is_none() {
                            Some(&null as *const _)
                        } else {
                            None
                        };
                        self.device.CreateRenderTargetView(res, d, at);
                    } else {
                        let d = s
                            .desc
                            .as_ptr()
                            .cast::<D3D12_DEPTH_STENCIL_VIEW_DESC>()
                            .read_unaligned();
                        let null = D3D12_DEPTH_STENCIL_VIEW_DESC {
                            Format: DXGI_FORMAT(40),
                            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                            ..Default::default()
                        };
                        let d = if s.described != 0 {
                            Some(&d as *const _)
                        } else if res.is_none() {
                            Some(&null as *const _)
                        } else {
                            None
                        };
                        self.device.CreateDepthStencilView(res, d, at);
                    }
                    self.targets.insert(*s, at);
                    at.ptr += step;
                }
                self._target_heaps.push(heap);
            }
        }
        Ok(())
    }

    /// A heap for each version of the views, for submissions sent without
    /// waits; made once.
    fn make_versions(&mut self) -> Result<()> {
        if self.descriptors.values().all(|h| !h.versions.is_empty()) {
            return Ok(());
        }
        let ids: Vec<u32> = self.descriptors.keys().copied().collect();
        let mut full = 0;
        for id in ids {
            let h = &self.descriptors[&id];
            let (kind, count, step, master) = (h.kind, h.count, h.step, h.master_cpu);
            let sampler = kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            let mut all: Vec<Version> = Vec::new();
            for changes in &self.versions {
                let mine: Vec<&(u32, u32, Slot)> =
                    changes.iter().filter(|(heap, _, _)| *heap == id).collect();
                for (_, i, s) in &mine {
                    if *i < count {
                        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: master + *i as usize * step,
                        };
                        // SAFETY: a slot of the master heap.
                        unsafe { self.view(s, at, sampler) };
                    }
                }
                let again = |v: &Version| Version {
                    heap: v.heap.clone(),
                    cpu: v.cpu,
                    gpu: v.gpu,
                };
                if mine.is_empty()
                    && let Some(last) = all.last()
                {
                    all.push(again(last));
                    continue;
                }
                match self.visible(kind, count, master) {
                    Ok(v) => all.push(v),
                    // Room for no more of them: the changes go into the
                    // last one, and lists before see them too.
                    Err(e) => {
                        let Some(last) = all.last() else {
                            return Err(e.into());
                        };
                        full += 1;
                        // SAFETY: whole heaps of one size and kind.
                        unsafe {
                            self.device.CopyDescriptorsSimple(
                                count,
                                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: last.cpu },
                                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: master },
                                kind,
                            );
                        }
                        all.push(again(last));
                    }
                }
            }
            // The master holds the frame's start again.
            for (heap, i, s) in &self.first_views {
                if *heap == id && *i < count {
                    let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: master + *i as usize * step,
                    };
                    // SAFETY: a slot of the master heap.
                    unsafe { self.view(s, at, sampler) };
                }
            }
            if let Some(h) = self.descriptors.get_mut(&id) {
                h.versions = all;
            }
        }
        if full > 0 {
            self.note(format!(
                "{full} versions of descriptor heaps had no room: earlier lists see later views"
            ));
        }
        Ok(())
    }

    /// Writes the views of `changes` into the live heaps.
    fn write_live(&self, changes: &[(u32, u32, Slot)]) {
        for (heap, i, s) in changes {
            let Some(h) = self.descriptors.get(heap) else {
                continue;
            };
            if *i < h.count {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: h.live.cpu + *i as usize * h.step,
                };
                // SAFETY: a slot of the live heap, which no list in flight
                // uses (the caller waited).
                unsafe { self.view(s, at, h.kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) };
            }
        }
    }

    /// Brings the live heaps to version `to` of the views. (The first
    /// version is the frame's start with the views that came into slots
    /// nothing used before any view in use was replaced: lists sent before
    /// them do not name those slots.)
    fn live_views(&mut self, to: usize) {
        if to < self.live_version {
            self.write_live(&self.first_views);
            if let Some(first) = self.versions.first() {
                self.write_live(first);
            }
            self.live_version = 0;
        }
        for v in self.live_version + 1..=to.min(self.versions.len().saturating_sub(1)) {
            self.write_live(&self.versions[v]);
        }
        self.live_version = to;
    }

    /// Reads parts of files into upload memory and has `copy` record what
    /// is done with each.
    fn upload<T: Sync>(
        &mut self,
        parts: &[(T, u64, u64)],
        file: &File,
        mut copy: impl FnMut(&Player, &ID3D12GraphicsCommandList, &[(&T, u64, u64)], &ID3D12Resource),
    ) -> Result<()> {
        // Batches of parts with their place in the buffer.
        let mut batches: Vec<Vec<(usize, u64)>> = vec![Vec::new()];
        let (mut at, mut most) = (0u64, 0u64);
        for (i, (_, len, _)) in parts.iter().enumerate() {
            at = at.next_multiple_of(512);
            if at + len > STAGING && at > 0 {
                batches.push(Vec::new());
                at = 0;
            }
            if let Some(b) = batches.last_mut() {
                b.push((i, at));
            }
            at += len;
            most = most.max(at);
        }
        if parts.is_empty() {
            return Ok(());
        }
        let mut stagings: Vec<(ID3D12Resource, usize, u64)> = Vec::new();
        for _ in 0..batches.len().min(2) {
            let (b, data) = gpu::buffer(
                &self.device,
                D3D12_HEAP_TYPE_UPLOAD,
                most,
                D3D12_RESOURCE_FLAG_NONE,
            )?;
            stagings.push((b, data as usize, 0));
        }
        for (n, batch) in batches.iter().enumerate() {
            let k = n % stagings.len();
            self.worker.wait(stagings[k].2)?;
            let map = stagings[k].1;
            let next = AtomicUsize::new(0);
            let failed: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::default();
            std::thread::scope(|scope| {
                for _ in 0..8.min(batch.len()) {
                    scope.spawn(|| {
                        while let Some(&(i, at)) = batch.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let (_, len, from) = &parts[i];
                            // SAFETY: this part's own bytes of the mapping.
                            let to = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (map + at as usize) as *mut u8,
                                    *len as usize,
                                )
                            };
                            if *from == ZERO {
                                to.fill(0);
                            } else if let Err(e) = read_at(file, to, *from) {
                                failed
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .get_or_insert(e);
                            }
                        }
                    });
                }
            });
            if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                return Err(e).context("the capture's contents");
            }
            let list: Vec<(&T, u64, u64)> = batch
                .iter()
                .map(|&(i, at)| (&parts[i].0, parts[i].1, at))
                .collect();
            let buffer = stagings[k].0.clone();
            // The worker is another part of the player than `copy` uses.
            let other = Worker::new(&self.device, &self.worker.queue)?;
            let mut worker = std::mem::replace(&mut self.worker, other);
            let sent = worker.send(|l| copy(self, l, &list, &buffer));
            self.worker = worker;
            stagings[k].2 = sent?;
        }
        self.worker.idle()?;
        Ok(())
    }

    fn load_contents(&mut self) -> Result<()> {
        // Idle aliases first: a change of their state may write memory the
        // one in use holds.
        let known: HashSet<u32> = self.frame.contents.iter().map(|c| c.res).collect();
        let mut early = Vec::new();
        let mut into = Vec::new();
        for r in &self.frame.resources {
            let (Some(o), Some(state)) = (self.objects.get(&r.id), self.start.get(&r.id)) else {
                continue;
            };
            let loaded = !r.idle && known.contains(&r.id);
            gpu::moves(
                &o.res,
                &[COMMON],
                if loaded { &[COPY_DEST] } else { state },
                if loaded { &mut into } else { &mut early },
            );
        }
        let textures: Vec<(Content, u64, u64)> = self
            .frame
            .contents
            .iter()
            .filter(|c| self.objects.contains_key(&c.res))
            .map(|c| (c.clone(), c.len, c.file))
            .collect();
        let missing = self.frame.contents.len() - textures.len();
        self.worker.run(|list| {
            // SAFETY: an open list; live resources.
            unsafe {
                barriers(list, &early);
                list.ResourceBarrier(&[alias()]);
                barriers(list, &into);
            }
        })?;
        let file = File::open(self.folder.join("contents.bin"))?;
        self.upload(&textures, &file, |player, list, parts, staging| {
            for (c, len, at) in parts {
                let Some(o) = player.objects.get(&c.res) else {
                    continue;
                };
                let r = &player.frame.resources[o.index];
                // SAFETY: an open list; live resources.
                unsafe {
                    if r.desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0 {
                        list.CopyBufferRegion(&o.res, c.first, staging, *at, *len);
                        continue;
                    }
                    let (all, _) = gpu::footprints(
                        &player.device,
                        &gpu::to_desc(&r.desc),
                        c.first as u32,
                        c.count,
                    );
                    for (i, mut f) in all.into_iter().enumerate() {
                        f.Offset += at;
                        let to = gpu::sub_location(&o.res, c.first as u32 + i as u32);
                        list.CopyTextureRegion(
                            &to,
                            0,
                            0,
                            0,
                            &gpu::placed_location(staging, f),
                            None,
                        );
                    }
                }
            }
        })?;
        // To the states of the frame's start.
        let mut back = Vec::new();
        for r in &self.frame.resources {
            let (Some(o), Some(state)) = (self.objects.get(&r.id), self.start.get(&r.id)) else {
                continue;
            };
            if !r.idle && known.contains(&r.id) {
                gpu::moves(&o.res, &[COPY_DEST], state, &mut back);
            }
        }
        self.worker.run(|list| {
            // SAFETY: an open list; live resources.
            unsafe { barriers(list, &back) };
        })?;
        if missing > 0 {
            self.note(format!(
                "contents of {missing} parts of resources that could not be made are left out"
            ));
        }
        // What the CPU had put into the memory both see by the first
        // submission is part of the frame's start: read into the buffers'
        // own mappings.
        let file = File::open(self.folder.join("deltas.bin"))?;
        let named = self.named_buffers();
        let of = |s: &wire::Submit| {
            s.deltas
                .iter()
                .filter(|d| named.contains(&d.res))
                .cloned()
                .collect::<Vec<Delta>>()
        };
        let first = of(&self.frame.submits[0]);
        let next = AtomicUsize::new(0);
        let failed: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::default();
        let unmapped = AtomicUsize::new(0);
        let (objects, file_ref) = (&self.objects, &file);
        std::thread::scope(|scope| {
            for _ in 0..8.min(first.len()) {
                scope.spawn(|| {
                    while let Some(d) = first.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let Some(o) = objects.get(&d.res) else {
                            continue;
                        };
                        if o.mapped == 0 {
                            unmapped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        // SAFETY: bytes of the buffer's mapping (the
                        // recorder took them from inside the buffer), which
                        // no other thread writes.
                        let to = unsafe {
                            std::slice::from_raw_parts_mut(
                                (o.mapped + d.at as usize) as *mut u8,
                                d.len as usize,
                            )
                        };
                        if let Err(e) = read_at(file_ref, to, d.file) {
                            failed
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .get_or_insert(e);
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e).context("what the CPU wrote by the first submission");
        }
        let unmapped = unmapped.into_inner();
        if unmapped > 0 {
            self.note(format!(
                "{unmapped} runs of bytes the CPU wrote are of buffers the replay does not map: left out"
            ));
        }
        // The later ones wait in memory.
        self.delta_parts = vec![Vec::new(); self.frame.submits.len()];
        let mut total = 0u64;
        for (i, s) in self.frame.submits.iter().enumerate().skip(1) {
            for d in of(s) {
                self.delta_parts[i].push((d.clone(), total));
                total += d.len;
            }
        }
        self.delta_bytes = vec![0; total as usize];
        for (d, at) in self.delta_parts.iter().flatten() {
            let to = &mut self.delta_bytes[*at as usize..(*at + d.len) as usize];
            read_at(&file, to, d.file).context("what the CPU wrote between submissions")?;
        }
        // What the mapped buffers hold now, page by page: the frame's start,
        // to find what a run changed. Only pages the capture has bytes of
        // are not zero.
        let mut by_buffer: HashMap<u32, Vec<(u64, u64, u64)>> = HashMap::new();
        for d in &first {
            by_buffer
                .entry(d.res)
                .or_default()
                .push((d.at, d.len, d.file));
        }
        let mut mapped: Vec<Mapped> = self
            .objects
            .iter()
            .filter(|(_, o)| o.mapped != 0)
            .map(|(id, o)| {
                let size = self.frame.resources[o.index].desc.width as usize;
                let mut runs = by_buffer.remove(id).unwrap_or_default();
                runs.sort_unstable();
                Mapped {
                    at: o.mapped,
                    size,
                    pages: vec![0; size.div_ceil(PAGE)],
                    runs,
                }
            })
            .collect();
        let all = std::sync::Mutex::new(mapped.iter_mut());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    loop {
                        let Some(m) = all.lock().unwrap_or_else(|e| e.into_inner()).next() else {
                            break;
                        };
                        for &(at, len, _) in &m.runs {
                            let (from, to) = (at as usize, ((at + len) as usize).min(m.size));
                            for page in (from..to).step_by(PAGE) {
                                let n = PAGE.min(m.size - page);
                                // SAFETY: a page of the buffer's mapping.
                                let bytes = unsafe {
                                    std::slice::from_raw_parts((m.at + page) as *const u8, n)
                                };
                                m.pages[page / PAGE] = plain::page_hash(bytes);
                            }
                        }
                    }
                });
            }
        });
        drop(all);
        self.mapped = mapped;
        Ok(())
    }

    /// The buffer of the GPU's that holds what the CPU wrote between
    /// submissions, for copies that stand for it; made once.
    fn make_deltas(&mut self) -> Result<()> {
        if self.deltas.is_some() || self.delta_bytes.is_empty() {
            return Ok(());
        }
        let total = self.delta_bytes.len() as u64;
        let (buffer, _) = gpu::buffer(
            &self.device,
            D3D12_HEAP_TYPE_DEFAULT,
            total,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        let mut at = 0;
        while at < total {
            let n = STAGING.min(total - at);
            let (staging, data) = gpu::buffer(
                &self.device,
                D3D12_HEAP_TYPE_UPLOAD,
                n,
                D3D12_RESOURCE_FLAG_NONE,
            )?;
            // SAFETY: the mapping of `n` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.delta_bytes.as_ptr().add(at as usize),
                    data,
                    n as usize,
                )
            };
            self.worker.run(|list| {
                // SAFETY: an open list; live buffers.
                unsafe { list.CopyBufferRegion(&buffer, at, &staging, 0, n) };
            })?;
            at += n;
        }
        self.deltas = Some(buffer);
        Ok(())
    }

    /// The buffers the frame names, in commands or descriptors: what the CPU
    /// writes to others (buffers it has results fetched to) is of no use.
    fn named_buffers(&self) -> HashSet<u32> {
        let mut named = HashSet::new();
        let mut va = |va: &Va| {
            if let Va::At { res, .. } = va {
                named.insert(*res);
            }
        };
        for c in self.commands.iter().flatten() {
            match c {
                Cmd::RootView { va: v, .. } => va(v),
                Cmd::IndexBuffer {
                    view: Some((v, _, _)),
                } => va(v),
                Cmd::VertexBuffers {
                    views: Some(all), ..
                } => all.iter().for_each(|(v, _, _)| va(v)),
                Cmd::WriteImmediate { all } => all.iter().for_each(|(v, _, _)| va(v)),
                _ => {}
            }
        }
        for c in self.commands.iter().flatten() {
            match c {
                Cmd::CopyBuffer { dst, src, .. } | Cmd::CopyResource { dst, src } => {
                    named.extend([*dst, *src])
                }
                Cmd::CopyTexture { dst, src, .. } => {
                    for Loc::Sub { res, .. } | Loc::Placed { res, .. } in [dst, src] {
                        named.insert(*res);
                    }
                }
                Cmd::Indirect { args, count, .. } => named.extend([*args, *count]),
                Cmd::ResolveQuery { dst, .. } => _ = named.insert(*dst),
                _ => {}
            }
        }
        for h in &self.frame.descriptor_heaps {
            named.extend(h.runs.iter().flat_map(|(_, _, s)| [s.res, s.counter]));
        }
        for s in &self.frame.submits {
            named.extend(s.views.iter().flat_map(|(_, _, s)| [s.res, s.counter]));
        }
        named
    }

    /// Copies of what the frame changes; and the states it leaves.
    fn make_masters(&mut self, wanted: bool) -> Result<()> {
        let mut states = States::default();
        for r in &self.frame.resources {
            if r.desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0 {
                continue;
            }
            let subs = gpu::subresources(&r.desc);
            let decays = r.desc.flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0 != 0;
            states.add(r.id, subs, r.state[0], decays);
            if r.state.len() > 1 {
                for (i, s) in r.state.iter().enumerate() {
                    states.change(&states::Change {
                        res: r.id,
                        sub: i as u32,
                        after: *s,
                        promoted: false,
                        alias: false,
                    });
                }
            }
        }
        // Resources in a state they are written in at some time, or written
        // some other way.
        const WRITE: u32 =
            RENDER_TARGET | UNORDERED_ACCESS | DEPTH_WRITE | COPY_DEST | RESOLVE_DEST;
        // (A texture that was filled long ago and left a copy's destination
        // is not written in the frame for that.)
        let mut written: HashSet<u32> = self
            .frame
            .resources
            .iter()
            .filter(|r| r.state.iter().any(|s| s & WRITE & !COPY_DEST != 0))
            .map(|r| r.id)
            .collect();
        let mut by_cpu: HashSet<u32> = HashSet::new();
        for (i, s) in self.frame.submits.iter().enumerate() {
            for &l in s.lists.iter().filter(|&&l| l != MISSING) {
                for c in &self.commands[l as usize] {
                    let mut changes = Vec::new();
                    states::changes(c, &mut changes);
                    for ch in &changes {
                        states.change(ch);
                    }
                    match c {
                        Cmd::Barriers { all } => {
                            for b in all {
                                if let Barrier::Transition {
                                    res, before, after, ..
                                } = b
                                    && (*before | *after) as u32 & WRITE != 0
                                {
                                    written.insert(*res);
                                }
                            }
                        }
                        Cmd::CopyBuffer { dst, .. }
                        | Cmd::CopyResource { dst, .. }
                        | Cmd::ResolveQuery { dst, .. } => _ = written.insert(*dst),
                        Cmd::CopyTexture {
                            dst: Loc::Sub { res, .. } | Loc::Placed { res, .. },
                            ..
                        } => _ = written.insert(*res),
                        Cmd::Resolve { dst, .. } | Cmd::ResolveRegion { dst, .. } => {
                            _ = written.insert(*dst)
                        }
                        Cmd::RootView {
                            kind: slot::UAV,
                            va: Va::At { res, .. },
                            ..
                        } => _ = written.insert(*res),
                        Cmd::WriteImmediate { all } => {
                            written.extend(all.iter().filter_map(|(va, _, _)| {
                                if let Va::At { res, .. } = va {
                                    Some(*res)
                                } else {
                                    None
                                }
                            }));
                        }
                        _ => {}
                    }
                }
            }
            states.decay();
            by_cpu.extend(self.delta_parts[i].iter().map(|(d, _)| d.res));
        }
        // Buffers any unordered-access view names.
        let simultaneous = |id: u32| {
            self.objects.get(&id).is_some_and(|o| {
                self.frame.resources[o.index].desc.flags
                    & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0
                    != 0
            })
        };
        let views = self
            .frame
            .descriptor_heaps
            .iter()
            .flat_map(|h| h.runs.iter().map(|(_, _, s)| s));
        let late = self
            .frame
            .submits
            .iter()
            .flat_map(|s| s.views.iter().map(|(_, _, s)| s));
        for s in views.chain(late).filter(|s| s.kind == slot::UAV) {
            for id in [s.res, s.counter] {
                if self.is_buffer(id) || simultaneous(id) {
                    written.insert(id);
                }
            }
        }
        written.extend(by_cpu);
        for id in self.start.keys() {
            self.end
                .insert(*id, states.get(*id).unwrap_or(vec![COMMON]));
        }
        let mut ids: Vec<u32> = written
            .into_iter()
            .filter(|id| {
                self.objects.get(id).is_some_and(|o| {
                    let r = &self.frame.resources[o.index];
                    let back = matches!(&r.place, Place::Committed { props, .. } if gpu::seen(props) == Seen::ReadBack);
                    // Buffers the CPU maps are put back another way
                    // ([`Player::restore_mapped`]).
                    !r.idle && !back && o.mapped == 0 && !matches!(r.place, Place::BackBuffer {})
                })
            })
            .collect();
        ids.sort_unstable();
        // Without restoring they are of no use.
        if !wanted {
            ids.clear();
        }
        let mut total = 0u64;
        for id in ids {
            let o = &self.objects[&id];
            let r = &self.frame.resources[o.index];
            let mut desc = gpu::to_desc(&r.desc);
            desc.Alignment = 0;
            // Copies are only copied to and from; a depth format needs its
            // flag.
            if r.desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0 {
                desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            }
            // SAFETY: a plain query.
            let size = unsafe { self.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
            let props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            let result = unsafe {
                self.device.CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut made,
                )
            };
            match (result, made) {
                (Ok(()), Some(m)) => {
                    total += if size == u64::MAX { r.desc.width } else { size };
                    self.masters.push((id, m));
                }
                (e, _) => self.note(format!(
                    "no copy of resource {id} to restore it from ({e:?})"
                )),
            }
        }
        tracing::info!(
            resources = self.masters.len(),
            mb = total >> 20,
            "copies of what the frame changes"
        );
        self.notes.push(format!(
            "{} resources the frame changes are put back between loops from copies of {} MB",
            self.masters.len(),
            total >> 20
        ));
        // Filled now: the resources are as at the frame's start. Masters
        // stay in the copy-source state (buffers decay).
        let (masters, objects, start) = (&self.masters, &self.objects, &self.start);
        for some in masters.chunks(256) {
            let (mut before, mut after) = (Vec::new(), Vec::new());
            for (id, m) in some {
                let o = &objects[id];
                if let Some(state) = start.get(id) {
                    let to: Vec<u32> = state
                        .iter()
                        .map(|&s| if s & COPY_SOURCE != 0 { s } else { COPY_SOURCE })
                        .collect();
                    gpu::moves(&o.res, state, &to, &mut before);
                    gpu::moves(&o.res, &to, state, &mut after);
                }
                before.push(gpu::transition(m, ALL, COMMON, COPY_DEST));
                after.push(gpu::transition(m, ALL, COPY_DEST, COPY_SOURCE));
            }
            self.worker.run(|list| {
                // SAFETY: an open list; live resources of one size.
                unsafe {
                    barriers(list, &before);
                    for (id, m) in some {
                        list.CopyResource(m, &objects[id].res);
                    }
                    barriers(list, &after);
                }
            })?;
        }
        Ok(())
    }

    /// The budget and use of video memory, and the use of the other kind
    /// (D3DMetal counts everything as local), in MB.
    fn memory(&self) -> Option<[u64; 3]> {
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut other = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter3 = factory
                .EnumAdapterByLuid(self.device.GetAdapterLuid())
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut other)
                .ok()?;
        }
        Some([
            own.Budget >> 20,
            own.CurrentUsage >> 20,
            other.CurrentUsage >> 20,
        ])
    }

    /// Puts back what a run changed.
    fn restore(&mut self, contents: bool) -> Result<()> {
        if self.fresh {
            return Ok(());
        }
        self.live_views(0);
        let restored: HashMap<u32, &ID3D12Resource> = if contents {
            self.masters.iter().map(|(id, m)| (*id, m)).collect()
        } else {
            HashMap::new()
        };
        let (mut idle, mut into, mut back, mut other) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (id, start) in &self.start {
            let Some(o) = self.objects.get(id) else {
                continue;
            };
            let end = self.end.get(id).unwrap_or(start);
            if restored.contains_key(id) {
                gpu::moves(&o.res, end, &[COPY_DEST], &mut into);
                gpu::moves(&o.res, &[COPY_DEST], start, &mut back);
            } else if self.frame.resources[o.index].idle {
                gpu::moves(&o.res, end, start, &mut idle);
            } else {
                gpu::moves(&o.res, end, start, &mut other);
            }
        }
        let objects = &self.objects;
        self.worker.run(|list| {
            // SAFETY: an open list; live resources, each copied from one of
            // its own description.
            unsafe {
                barriers(list, &idle);
                barriers(list, &[alias()]);
                barriers(list, &other);
                barriers(list, &into);
                for (id, m) in &restored {
                    list.CopyResource(&objects[id].res, *m);
                }
                barriers(list, &back);
            }
        })?;
        if contents {
            self.restore_mapped()?;
        }
        self.fresh = true;
        Ok(())
    }

    /// Puts back what a run changed of the buffers the CPU maps (what the
    /// CPU wrote between submissions, what shaders and copies wrote,
    /// whatever an experiment made them write): every page whose number is
    /// not the one of the frame's start is read from the capture again, or
    /// zeroed. The GPU is idle (the caller waited).
    fn restore_mapped(&mut self) -> Result<()> {
        let file = File::open(self.folder.join("deltas.bin"))?;
        let failed: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::default();
        let bytes = AtomicUsize::new(0);
        let began = Instant::now();
        // Pieces of 4 MB of the buffers, taken by the threads in turn.
        const PIECE: usize = 1024;
        let pieces: Vec<(usize, usize)> = self
            .mapped
            .iter()
            .enumerate()
            .flat_map(|(i, m)| (0..m.pages.len()).step_by(PIECE).map(move |p| (i, p)))
            .collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let (next, all, pieces, file, failed, bytes) =
                (&next, &self.mapped, &pieces, &file, &failed, &bytes);
            for _ in 0..8 {
                scope.spawn(move || {
                    while let Some(&(i, first)) = pieces.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let m = &all[i];
                        let last = (first + PIECE).min(m.pages.len());
                        for (p, &was) in m.pages[first..last].iter().enumerate() {
                            let page = (first + p) * PAGE;
                            let n = PAGE.min(m.size - page);
                            // SAFETY: a page of the buffer's mapping, which
                            // this thread alone has; the GPU is idle.
                            let now = unsafe {
                                std::slice::from_raw_parts_mut((m.at + page) as *mut u8, n)
                            };
                            if plain::page_hash(now) == was {
                                continue;
                            }
                            bytes.fetch_add(n, Ordering::Relaxed);
                            // The run of the capture that holds it, if any.
                            let k = m.runs.partition_point(|r| r.0 <= page as u64);
                            match k.checked_sub(1).map(|k| m.runs[k]) {
                                Some((at, len, from)) if (page as u64) < at + len => {
                                    if let Err(e) = read_at(file, now, from + page as u64 - at) {
                                        failed
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .get_or_insert(e);
                                    }
                                }
                                _ => now.fill(0),
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e).context("putting back the buffers the CPU maps");
        }
        self.restored = (bytes.into_inner(), began.elapsed().as_secs_f64());
        Ok(())
    }

    /// The frame's lists, edited: for each submission, its lists' commands.
    fn edited(
        &self,
        o: &Options,
        report: &mut Report,
        also: &HashSet<u32>,
    ) -> Vec<Vec<Vec<Cmd>>> {
        let mut skip: HashSet<u32> = self
            .frame
            .pipelines
            .iter()
            .filter(|p| {
                let hashes = self.hashes(p);
                o.skip_shaders.iter().any(|s| {
                    !s.is_empty()
                        && (p.name.contains(s.as_str())
                            || hashes.iter().any(|h| h.contains(s.as_str())))
                })
            })
            .map(|p| p.id)
            .collect();
        skip.extend(also);
        let mut barriers = 0;
        let clamps = self.clamps();
        let mut kept = edit::Clamped::default();
        let mut clamped = (0, 0);
        if !o.skip_shaders.is_empty() {
            report
                .notes
                .push(format!("{} pipelines match --skip-shader", skip.len()));
        }
        let mut skipped = 0;
        let mut out = Vec::new();
        for s in &self.frame.submits {
            let mut lists = Vec::new();
            for &l in s.lists.iter().filter(|&&l| l != MISSING) {
                let mut cmds = self.commands[l as usize].clone();
                if !skip.is_empty() {
                    let n;
                    (cmds, n) = edit::skip_pipelines(cmds, &skip);
                    skipped += n;
                }
                if o.barriers != Barriers::AsRecorded {
                    let n;
                    (cmds, n) = edit::drop_barriers(cmds, |id| {
                        o.barriers == Barriers::NoneAtAll || id == NONE || self.is_buffer(id)
                    });
                    barriers += n;
                }
                if o.keep_clamps {
                    let (passes, n);
                    (cmds, passes, n) = edit::skip_clamped(
                        cmds,
                        &clamps.0,
                        &clamps.1,
                        &mut kept,
                    );
                    clamped = (clamped.0 + passes, clamped.1 + n);
                }
                if o.one_texel {
                    cmds = edit::one_texel(cmds);
                }
                if o.binds_only {
                    cmds = edit::binds_only(cmds);
                }
                if o.one_group {
                    cmds = edit::one_group(cmds);
                }
                lists.push(cmds);
            }
            out.push(lists);
        }
        if !skip.is_empty() {
            report
                .notes
                .push(format!("--skip-shader left out {skipped} commands"));
        }
        if o.barriers != Barriers::AsRecorded {
            report
                .notes
                .push(format!("{barriers} barriers were left out"));
        }
        if o.keep_clamps {
            report.notes.push(format!(
                "{} argument clamps were left out, with {} barriers of theirs; their calls read {} KB of arguments kept from a run with them",
                clamped.0,
                clamped.1,
                kept.next >> 10
            ));
            // This line's batches of clamps are another pipeline's work
            // (`crate::indirect`, "victory: clamps").
            if self.frame.pipelines.iter().any(|p| p.name == CLAMP_BATCH) {
                report.notes.push(format!(
                    "batches of clamps (\"{CLAMP_BATCH}\", VICTORY_BATCH_CLAMPS=1) stay, and their calls read what they write: record with VICTORY_BATCH_CLAMPS=0 to leave out every clamp"
                ));
            }
        }
        if o.one_texel {
            report
                .notes
                .push("every scissor rectangle is one texel: draws without their pixels".into());
        }
        if o.binds_only {
            report
                .notes
                .push("nothing that works is sent: what the lists bind, and their barriers".into());
        }
        if o.one_group {
            report
                .notes
                .push("every dispatch is of one thread group (those by arguments from a buffer are as they were)".into());
        }
        report.skipped = skipped;
        out
    }

    /// The pipelines of the host layer's argument clamp, and their root
    /// signatures.
    fn clamps(&self) -> (HashSet<u32>, HashSet<u32>) {
        let all: Vec<&wire::Pipeline> = self
            .frame
            .pipelines
            .iter()
            .filter(|p| p.name == CLAMP)
            .collect();
        let roots = all
            .iter()
            .flat_map(|p| &p.parts)
            .filter_map(|part| match part {
                wire::Part::Root { root } => Some(*root),
                _ => None,
            })
            .collect();
        (all.iter().map(|p| p.id).collect(), roots)
    }

    /// Runs the frame once with every clamp writing its arguments to a
    /// buffer of ours ([`edit::RESULTS`]), from which the calls of a frame
    /// without the clamps read them; once for good.
    fn keep_clamps(&mut self) -> Result<()> {
        if self.results.is_some() {
            return Ok(());
        }
        let clamps = self.clamps();
        let mut kept = edit::Clamped::default();
        let mut submits = Vec::new();
        for s in &self.frame.submits {
            let mut lists = Vec::new();
            for &l in s.lists.iter().filter(|&&l| l != MISSING) {
                lists.push(edit::keep_clamped(
                    self.commands[l as usize].clone(),
                    &clamps.0,
                    &mut kept,
                ));
            }
            submits.push(lists);
        }
        let next = kept.next;
        let (results, _) = gpu::buffer(
            &self.device,
            D3D12_HEAP_TYPE_DEFAULT,
            next.max(16),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )?;
        self.results = Some(results);
        let plain = Options::default();
        let sends = self.build(submits, &plain, false)?;
        self.restore(true)?;
        self.fresh = false;
        self.send(&sends, &plain)?;
        Ok(())
    }

    /// The pipelines with a mesh shader, and the command signatures that
    /// dispatch: what commands are counted by.
    fn kinds(&self) -> (HashSet<u32>, HashSet<u32>) {
        let mesh = self
            .frame
            .pipelines
            .iter()
            .filter(|p| p.parts.iter().any(|part| matches!(part, wire::Part::Code { kind, .. } if *kind == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0)))
            .map(|p| p.id)
            .collect();
        let dispatches = self
            .frame
            .signatures
            .iter()
            .filter(|s| {
                s.args
                    .iter()
                    .any(|a| a[0] == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32)
            })
            .map(|s| s.id)
            .collect();
        (mesh, dispatches)
    }

    /// Makes the lists of all submissions and, for submissions sent without
    /// waits, before each one's the copies of what the CPU wrote.
    fn build(&self, submits: Vec<Vec<Vec<Cmd>>>, o: &Options, stamped: bool) -> Result<Sends> {
        let waited = matches!(o.waits, Waits::Each | Waits::Lists);
        // Room for the timestamps of every command that has one.
        let count: usize = submits
            .iter()
            .flatten()
            .map(|c| {
                2 + if stamped {
                    c.iter()
                        .filter(|c| c.works() || matches!(c, Cmd::Barriers { .. }))
                        .count()
                } else {
                    0
                }
            })
            .sum::<usize>()
            + 2 * submits.len();
        let mut heap: Option<ID3D12QueryHeap> = None;
        let mut back = None;
        let mut data = 0;
        if self.frequency > 0.0 {
            let desc = D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: count.max(1) as u32,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            unsafe { self.device.CreateQueryHeap(&desc, &mut heap)? };
            let (b, d) = gpu::buffer(
                &self.device,
                D3D12_HEAP_TYPE_READBACK,
                count.max(1) as u64 * 8,
                D3D12_RESOURCE_FLAG_NONE,
            )?;
            (back, data) = (Some(b), d as usize);
        }
        let mut next = 0u32;
        let mut steps = Vec::new();
        for (i, lists) in submits.iter().enumerate() {
            let parts = self.delta_parts.get(i).filter(|p| !p.is_empty());
            let delta = match (parts, &self.deltas) {
                (Some(parts), Some(from)) if !waited => {
                    let open = self.open(heap.as_ref(), &mut next)?;
                    // SAFETY: an open list; live buffers.
                    unsafe {
                        let mut seen = HashSet::new();
                        let mut after = Vec::new();
                        for (d, at) in parts {
                            let Some(ob) = self.objects.get(&d.res) else {
                                continue;
                            };
                            open.1.CopyBufferRegion(&ob.res, d.at, from, *at, d.len);
                            if seen.insert(d.res) {
                                // Back to the common state by itself, for
                                // lists in the same submission.
                                after.push(gpu::transition(&ob.res, ALL, COPY_DEST, COMMON));
                            }
                        }
                        barriers(&open.1, &after);
                    }
                    Some(self.close(open, heap.as_ref(), &mut next, Vec::new())?)
                }
                _ => None,
            };
            let version = if waited { None } else { Some(self.version_of.get(i).copied().unwrap_or(0)) };
            let mut built = Vec::new();
            for cmds in lists {
                let open = self.open(heap.as_ref(), &mut next)?;
                let stamps = self.translate(
                    &open.1,
                    cmds,
                    version,
                    match (&heap, stamped) {
                        (Some(heap), true) => Some((heap, &mut next)),
                        _ => None,
                    },
                );
                built.push(self.close(open, heap.as_ref(), &mut next, stamps)?);
            }
            steps.push((delta, built));
        }
        // Between submissions that are sent as one: everything waits. A list
        // each: one cannot stand twice in a submission.
        let mut joints = Vec::new();
        if o.waits == Waits::OneSubmit {
            for _ in 0..2 * submits.len() {
                let open = self.plain_list()?;
                // SAFETY: an open list.
                unsafe {
                    open.1.ResourceBarrier(&[alias(), uav_barrier(None)]);
                    open.1.Close()?;
                }
                joints.push(open);
            }
        }
        let resolve = match (&heap, &back) {
            (Some(heap), Some(back)) => {
                let open = self.plain_list()?;
                // SAFETY: an open list; `next` queries were written.
                unsafe {
                    open.1.ResolveQueryData(
                        heap,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        next.max(1),
                        back,
                        0,
                    );
                    open.1.Close()?;
                }
                Some(open)
            }
            _ => None,
        };
        Ok(Sends {
            submits,
            steps,
            joints,
            resolve,
            _heap: heap,
            _back: back,
            data,
            count: next,
            stamped,
        })
    }

    fn plain_list(&self) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        // SAFETY: plain creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            Ok((allocator, list))
        }
    }

    fn open(
        &self,
        heap: Option<&ID3D12QueryHeap>,
        next: &mut u32,
    ) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList, u32)> {
        let (allocator, list) = self.plain_list()?;
        if let Some(heap) = heap {
            // SAFETY: an open list; a query of the heap.
            unsafe { list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, *next) };
        }
        *next += 1;
        Ok((allocator, list, *next - 1))
    }

    fn close(
        &self,
        (allocator, list, first): (ID3D12CommandAllocator, ID3D12GraphicsCommandList, u32),
        heap: Option<&ID3D12QueryHeap>,
        next: &mut u32,
        stamps: Vec<Stamp>,
    ) -> Result<Built> {
        // SAFETY: as above.
        unsafe {
            if let Some(heap) = heap {
                list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, *next);
            }
            list.Close()
                .context("a list of the frame could not be recorded again")?;
        }
        *next += 1;
        Ok(Built {
            _allocator: allocator,
            list,
            first,
            last: *next - 1,
            stamps,
        })
    }

    /// Records commands into a list; with `version`, the heaps of that
    /// version of the views are bound, else the live ones.
    fn translate(
        &self,
        list: &ID3D12GraphicsCommandList,
        cmds: &[Cmd],
        version: Option<usize>,
        mut stamp: Option<(&ID3D12QueryHeap, &mut u32)>,
    ) -> Vec<Stamp> {
        // What commands are counted by, where they are timed.
        let kinds = stamp.is_some().then(|| self.kinds());
        let mut stamps = Vec::new();
        let mut bound = NONE;
        for c in cmds {
            if let Cmd::Pipeline { pso } = c {
                bound = *pso;
            }
            // SAFETY: an open list; the objects are the frame's, alive with
            // the player; commands the host recorded the same way.
            unsafe { self.command(list, c, version) };
            let timed = c.works() || matches!(c, Cmd::Barriers { .. });
            if timed
                && let Some((heap, next)) = &mut stamp
                && let Some((mesh, dispatches)) = &kinds
            {
                // SAFETY: as above.
                unsafe { list.EndQuery(*heap, D3D12_QUERY_TYPE_TIMESTAMP, **next) };
                let kind = edit::kind(c, dispatches, mesh, bound);
                let runs = match c {
                    c if c.runs_shaders() => bound,
                    // Copies to textures by what they go to.
                    Cmd::CopyResource { dst, .. }
                    | Cmd::CopyTexture {
                        dst: Loc::Sub { res: dst, .. },
                        ..
                    } => COPY_TO | dst,
                    _ => NONE,
                };
                stamps.push((**next as usize, kind, runs));
                **next += 1;
            }
        }
        stamps
    }

    unsafe fn command(&self, l: &ID3D12GraphicsCommandList, c: &Cmd, version: Option<usize>) {
        let res = |id: u32| self.resource(id);
        let view = |s: &Slot| self.targets.get(s).copied().unwrap_or_default();
        let heap = |id: &u32| -> Option<(&Version, u64)> {
            let h = self.descriptors.get(id)?;
            let v = match version {
                Some(v) => h.versions.get(v)?,
                None => &h.live,
            };
            Some((v, h.step as u64))
        };
        // SAFETY: the caller's.
        unsafe {
            match c {
                Cmd::Draw {
                    vertices,
                    instances,
                    first,
                    first_instance,
                } => l.DrawInstanced(*vertices, *instances, *first, *first_instance),
                Cmd::DrawIndexed {
                    indices,
                    instances,
                    first,
                    base,
                    first_instance,
                } => {
                    l.DrawIndexedInstanced(*indices, *instances, *first, *base, *first_instance);
                }
                Cmd::Dispatch { groups } => l.Dispatch(groups[0], groups[1], groups[2]),
                Cmd::DispatchMesh { groups } => {
                    if let Ok(l) = l.cast::<ID3D12GraphicsCommandList6>() {
                        l.DispatchMesh(groups[0], groups[1], groups[2]);
                    }
                }
                Cmd::CopyBuffer {
                    dst,
                    dst_at,
                    src,
                    src_at,
                    len,
                } => {
                    if let (Some(dst), Some(src)) = (res(*dst), res(*src)) {
                        l.CopyBufferRegion(dst, *dst_at, src, *src_at, *len);
                    }
                }
                Cmd::CopyTexture { dst, at, src, part } => {
                    let side = |loc: &Loc| match loc {
                        Loc::Sub { res: id, sub } => Some(gpu::sub_location(res(*id)?, *sub)),
                        Loc::Placed {
                            res: id,
                            offset,
                            format,
                            size,
                            pitch,
                        } => {
                            let f = D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT(*format),
                                Width: size[0],
                                Height: size[1],
                                Depth: size[2],
                                RowPitch: *pitch,
                            };
                            Some(gpu::placed_location(
                                res(*id)?,
                                D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                    Offset: *offset,
                                    Footprint: f,
                                },
                            ))
                        }
                    };
                    if let (Some(dst), Some(src)) = (side(dst), side(src)) {
                        let part = part.map(|p| D3D12_BOX {
                            left: p[0],
                            top: p[1],
                            front: p[2],
                            right: p[3],
                            bottom: p[4],
                            back: p[5],
                        });
                        l.CopyTextureRegion(
                            &dst,
                            at[0],
                            at[1],
                            at[2],
                            &src,
                            part.as_ref().map(|p| p as *const _),
                        );
                    }
                }
                Cmd::CopyResource { dst, src } => {
                    if let (Some(dst), Some(src)) = (res(*dst), res(*src)) {
                        l.CopyResource(dst, src);
                    }
                }
                Cmd::Resolve {
                    dst,
                    dst_sub,
                    src,
                    src_sub,
                    format,
                } => {
                    if let (Some(dst), Some(src)) = (res(*dst), res(*src)) {
                        l.ResolveSubresource(dst, *dst_sub, src, *src_sub, DXGI_FORMAT(*format));
                    }
                }
                Cmd::ResolveRegion {
                    dst,
                    dst_sub,
                    at,
                    src,
                    src_sub,
                    part,
                    format,
                    mode,
                } => {
                    if let (Some(dst), Some(src), Ok(l)) =
                        (res(*dst), res(*src), l.cast::<ID3D12GraphicsCommandList1>())
                    {
                        let part = part.map(to_rect);
                        let part = part.as_ref().map(|r| r as *const _);
                        l.ResolveSubresourceRegion(
                            dst,
                            *dst_sub,
                            at[0],
                            at[1],
                            src,
                            *src_sub,
                            part,
                            DXGI_FORMAT(*format),
                            D3D12_RESOLVE_MODE(*mode),
                        );
                    }
                }
                Cmd::Topology { kind } => l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(*kind)),
                Cmd::Viewports { all } => {
                    let all: Vec<D3D12_VIEWPORT> = all
                        .iter()
                        .map(|v| D3D12_VIEWPORT {
                            TopLeftX: v[0],
                            TopLeftY: v[1],
                            Width: v[2],
                            Height: v[3],
                            MinDepth: v[4],
                            MaxDepth: v[5],
                        })
                        .collect();
                    l.RSSetViewports(&all);
                }
                Cmd::Scissors { all } => {
                    l.RSSetScissorRects(&all.iter().map(|r| to_rect(*r)).collect::<Vec<_>>())
                }
                Cmd::BlendFactor { factor } => l.OMSetBlendFactor(factor.as_ref()),
                Cmd::StencilRef { value } => l.OMSetStencilRef(*value),
                Cmd::DepthBounds { min, max } => {
                    if let Ok(l) = l.cast::<ID3D12GraphicsCommandList1>() {
                        l.OMSetDepthBounds(*min, *max);
                    }
                }
                Cmd::Pipeline { pso } => {
                    if let Some(p) = self.pipelines.get(pso) {
                        l.SetPipelineState(p);
                    }
                }
                Cmd::Barriers { all } => {
                    let all: Vec<D3D12_RESOURCE_BARRIER> =
                        all.iter().filter_map(|b| self.barrier(b)).collect();
                    barriers(l, &all);
                }
                Cmd::Heaps { heaps } => {
                    let all: Vec<Option<ID3D12DescriptorHeap>> = heaps
                        .iter()
                        .filter_map(|id| Some(Some(heap(id)?.0.heap.clone())))
                        .collect();
                    l.SetDescriptorHeaps(&all);
                }
                Cmd::RootSignature { graphics, root } => {
                    let root = self.roots.get(root);
                    if *graphics {
                        l.SetGraphicsRootSignature(root)
                    } else {
                        l.SetComputeRootSignature(root)
                    }
                }
                Cmd::Table {
                    graphics,
                    param,
                    heap: id,
                    slot,
                } => {
                    if let Some((v, step)) = heap(id) {
                        let at = D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: v.gpu + u64::from(*slot) * step,
                        };
                        if *graphics {
                            l.SetGraphicsRootDescriptorTable(*param, at)
                        } else {
                            l.SetComputeRootDescriptorTable(*param, at)
                        }
                    }
                }
                Cmd::Constants {
                    graphics,
                    param,
                    offset,
                    values,
                } => {
                    let (n, data) = (values.len() as u32, values.as_ptr().cast());
                    if *graphics {
                        l.SetGraphicsRoot32BitConstants(*param, n, data, *offset);
                    } else {
                        l.SetComputeRoot32BitConstants(*param, n, data, *offset);
                    }
                }
                Cmd::RootView {
                    graphics,
                    kind,
                    param,
                    va,
                } => {
                    let va = self.va(va);
                    match (*kind, *graphics) {
                        (slot::CBV, true) => l.SetGraphicsRootConstantBufferView(*param, va),
                        (slot::CBV, false) => l.SetComputeRootConstantBufferView(*param, va),
                        (slot::SRV, true) => l.SetGraphicsRootShaderResourceView(*param, va),
                        (slot::SRV, false) => l.SetComputeRootShaderResourceView(*param, va),
                        (_, true) => l.SetGraphicsRootUnorderedAccessView(*param, va),
                        (_, false) => l.SetComputeRootUnorderedAccessView(*param, va),
                    }
                }
                Cmd::IndexBuffer { view } => {
                    let v = view
                        .as_ref()
                        .map(|(va, size, format)| D3D12_INDEX_BUFFER_VIEW {
                            BufferLocation: self.va(va),
                            SizeInBytes: *size,
                            Format: DXGI_FORMAT(*format),
                        });
                    l.IASetIndexBuffer(v.as_ref().map(|v| v as *const _));
                }
                Cmd::VertexBuffers {
                    start,
                    views,
                    count,
                } => match views {
                    Some(views) => {
                        let all: Vec<D3D12_VERTEX_BUFFER_VIEW> = views
                            .iter()
                            .map(|(va, size, stride)| D3D12_VERTEX_BUFFER_VIEW {
                                BufferLocation: self.va(va),
                                SizeInBytes: *size,
                                StrideInBytes: *stride,
                            })
                            .collect();
                        l.IASetVertexBuffers(*start, Some(&all));
                    }
                    None => l.IASetVertexBuffers(
                        *start,
                        Some(&vec![D3D12_VERTEX_BUFFER_VIEW::default(); *count as usize]),
                    ),
                },
                Cmd::Targets { colors, depth } => {
                    let colors: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> =
                        colors.iter().map(view).collect();
                    let depth = depth.as_ref().map(view);
                    l.OMSetRenderTargets(
                        colors.len() as u32,
                        (!colors.is_empty()).then_some(colors.as_ptr()),
                        false,
                        depth.as_ref().map(|d| d as *const _),
                    );
                }
                Cmd::ClearDepth {
                    view: v,
                    flags,
                    depth,
                    stencil,
                    rects,
                } => {
                    let rects: Vec<RECT> = rects.iter().map(|r| to_rect(*r)).collect();
                    let rects = (!rects.is_empty()).then_some(&rects[..]);
                    if self.objects.contains_key(&v.res) {
                        l.ClearDepthStencilView(
                            view(v),
                            D3D12_CLEAR_FLAGS(*flags),
                            *depth,
                            *stencil,
                            rects,
                        );
                    }
                }
                Cmd::ClearTarget {
                    view: v,
                    color,
                    rects,
                } => {
                    let rects: Vec<RECT> = rects.iter().map(|r| to_rect(*r)).collect();
                    let rects = (!rects.is_empty()).then_some(&rects[..]);
                    if self.objects.contains_key(&v.res) {
                        l.ClearRenderTargetView(view(v), color, rects);
                    }
                }
                Cmd::ClearUnordered {
                    heap: id,
                    slot,
                    view: v,
                    res: cleared,
                    values,
                    float,
                    rects,
                } => {
                    let rects: Vec<RECT> = rects.iter().map(|r| to_rect(*r)).collect();
                    // Both handles name a view of the live resource, the
                    // first in the heap the list has bound.
                    if let (Some((h, step)), Some(r), Some(cpu)) =
                        (heap(id), res(*cleared), self.targets.get(v))
                    {
                        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: h.gpu + u64::from(*slot) * step,
                        };
                        if *float {
                            let values = values.map(f32::from_bits);
                            l.ClearUnorderedAccessViewFloat(gpu, *cpu, r, &values, &rects);
                        } else {
                            l.ClearUnorderedAccessViewUint(gpu, *cpu, r, values, &rects);
                        }
                    }
                }
                Cmd::Discard {
                    res: id,
                    rects,
                    subs,
                } => {
                    if let Some(r) = res(*id) {
                        let rects: Vec<RECT> = rects.iter().map(|r| to_rect(*r)).collect();
                        let region = subs.map(|s| D3D12_DISCARD_REGION {
                            NumRects: rects.len() as u32,
                            pRects: if rects.is_empty() {
                                std::ptr::null()
                            } else {
                                rects.as_ptr()
                            },
                            FirstSubresource: s[0],
                            NumSubresources: s[1],
                        });
                        l.DiscardResource(r, region.as_ref().map(|r| r as *const _));
                    }
                }
                Cmd::BeginQuery { heap, kind, index } => {
                    if let Some(h) = self.queries.get(heap) {
                        l.BeginQuery(h, D3D12_QUERY_TYPE(*kind), *index);
                    }
                }
                Cmd::EndQuery { heap, kind, index } => {
                    if let Some(h) = self.queries.get(heap) {
                        l.EndQuery(h, D3D12_QUERY_TYPE(*kind), *index);
                    }
                }
                Cmd::ResolveQuery {
                    heap,
                    kind,
                    first,
                    count,
                    dst,
                    at,
                } => {
                    if let (Some(h), Some(dst)) = (self.queries.get(heap), res(*dst)) {
                        l.ResolveQueryData(h, D3D12_QUERY_TYPE(*kind), *first, *count, dst, *at);
                    }
                }
                Cmd::Indirect {
                    signature,
                    most,
                    args,
                    at,
                    count,
                    count_at,
                } => {
                    if let (Some(s), Some(a)) = (self.signatures.get(signature), res(*args)) {
                        l.ExecuteIndirect(s, *most, a, *at, res(*count), *count_at);
                    }
                }
                Cmd::WriteImmediate { all } => {
                    if let Ok(l) = l.cast::<ID3D12GraphicsCommandList2>() {
                        let all: Vec<&(Va, u32, i32)> =
                            all.iter().filter(|(va, _, _)| self.va(va) != 0).collect();
                        let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = all
                            .iter()
                            .map(|(va, v, _)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                                Dest: self.va(va),
                                Value: *v,
                            })
                            .collect();
                        let modes: Vec<D3D12_WRITEBUFFERIMMEDIATE_MODE> = all
                            .iter()
                            .map(|(_, _, m)| D3D12_WRITEBUFFERIMMEDIATE_MODE(*m))
                            .collect();
                        if !params.is_empty() {
                            l.WriteBufferImmediate(
                                params.len() as u32,
                                params.as_ptr(),
                                Some(modes.as_ptr()),
                            );
                        }
                    }
                }
            }
        }
    }

    fn barrier(&self, b: &Barrier) -> Option<D3D12_RESOURCE_BARRIER> {
        Some(match b {
            Barrier::Transition {
                flags,
                res,
                sub,
                before,
                after,
            } => {
                let mut t =
                    gpu::transition(self.resource(*res)?, *sub, *before as u32, *after as u32);
                t.Flags = D3D12_RESOURCE_BARRIER_FLAGS(*flags);
                t
            }
            Barrier::Uav { flags, res } => {
                let mut u = uav_barrier(if *res == NONE {
                    None
                } else {
                    Some(self.resource(*res)?)
                });
                u.Flags = D3D12_RESOURCE_BARRIER_FLAGS(*flags);
                u
            }
            Barrier::Alias {
                flags,
                before,
                after,
            } => {
                let side = |id: &u32| -> std::mem::ManuallyDrop<Option<ID3D12Resource>> {
                    // SAFETY: a borrowed pointer (or none) never released.
                    self.resource(*id)
                        .map_or(std::mem::ManuallyDrop::new(None), |r| unsafe {
                            std::mem::transmute_copy(r)
                        })
                };
                D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                    Flags: D3D12_RESOURCE_BARRIER_FLAGS(*flags),
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                            pResourceBefore: side(before),
                            pResourceAfter: side(after),
                        }),
                    },
                }
            }
        })
    }

    /// Puts what the CPU wrote by submission `i` into the buffers' mappings.
    fn write_deltas(&self, i: usize) {
        for (d, at) in self.delta_parts.get(i).into_iter().flatten() {
            let Some(o) = self.objects.get(&d.res).filter(|o| o.mapped != 0) else {
                continue;
            };
            // SAFETY: bytes of the buffer's mapping (the recorder took them
            // from inside the buffer) and of what was read of them; no list
            // in flight uses the buffer (the caller waited).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.delta_bytes.as_ptr().add(*at as usize),
                    (o.mapped + d.at as usize) as *mut u8,
                    d.len as usize,
                )
            };
        }
    }

    /// Stands idle for `ms` milliseconds.
    fn pace(ms: f64) {
        if ms > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(ms / 1e3));
        }
    }

    /// Sends the frame, waits for it, and reads the timestamps.
    fn send(&mut self, s: &Sends, o: &Options) -> Result<Ran> {
        let cast = |l: &ID3D12GraphicsCommandList| l.cast::<ID3D12CommandList>().map(Some);
        let mut walls = Vec::new();
        let mut between = Duration::ZERO;
        let frame;
        let mut list_walls = Vec::new();
        if matches!(o.waits, Waits::Each | Waits::Lists) {
            // As the runtime does: the CPU's part, then the submission,
            // waited for (or each of its lists).
            let mut sum = Duration::ZERO;
            for (i, (_, lists)) in s.steps.iter().enumerate() {
                let began = Instant::now();
                self.live_views(self.version_of.get(i).copied().unwrap_or(0));
                self.write_deltas(i);
                between += began.elapsed();
                let mut all = Vec::new();
                for l in lists {
                    all.push(cast(&l.list)?);
                }
                let mut took = Duration::ZERO;
                if o.waits == Waits::Lists {
                    for l in &all {
                        let began = Instant::now();
                        // SAFETY: a closed list whose last run has finished.
                        unsafe { self.worker.queue.ExecuteCommandLists(std::slice::from_ref(l)) };
                        self.worker
                            .idle()
                            .context("the device was lost running the frame")?;
                        took += began.elapsed();
                        list_walls.push(began.elapsed().as_secs_f64() * 1e3);
                    }
                } else {
                    let began = Instant::now();
                    if !all.is_empty() {
                        // SAFETY: closed lists whose last run has finished.
                        unsafe { self.worker.queue.ExecuteCommandLists(&all) };
                    }
                    self.worker
                        .idle()
                        .context("the device was lost running the frame")?;
                    took = began.elapsed();
                }
                sum += took;
                walls.push(took.as_secs_f64() * 1e3);
                Self::pace(o.pace);
            }
            frame = sum;
            if let Some(resolve) = &s.resolve {
                // SAFETY: a closed list.
                unsafe {
                    self.worker
                        .queue
                        .ExecuteCommandLists(&[cast(&resolve.1)?])
                };
                self.worker.idle()?;
            }
        } else {
            let mut submissions: Vec<Vec<Option<ID3D12CommandList>>> = Vec::new();
            for (delta, lists) in &s.steps {
                if let Some(d) = delta {
                    submissions.push(vec![cast(&d.list)?]);
                }
                let mut all = Vec::new();
                for l in lists {
                    all.push(cast(&l.list)?);
                }
                if !all.is_empty() {
                    submissions.push(all);
                }
            }
            if o.waits == Waits::OneSubmit {
                let mut all = Vec::new();
                for (i, lists) in submissions.into_iter().enumerate() {
                    if i > 0
                        && let Some(joint) = s.joints.get(i)
                    {
                        all.push(cast(&joint.1)?);
                    }
                    all.extend(lists);
                }
                submissions = vec![all];
            }
            let began = Instant::now();
            // SAFETY: closed lists whose last run has finished.
            unsafe {
                for all in &submissions {
                    self.worker.queue.ExecuteCommandLists(all);
                    Self::pace(o.pace);
                }
            }
            self.worker
                .idle()
                .context("the device was lost running the frame")?;
            frame = began.elapsed();
            if let Some(resolve) = &s.resolve {
                // SAFETY: a closed list.
                unsafe {
                    self.worker
                        .queue
                        .ExecuteCommandLists(&[cast(&resolve.1)?])
                };
                self.worker.idle()?;
            }
        }
        let stamps = if s.data != 0 && s.resolve.is_some() {
            // SAFETY: the mapping of `count` resolved timestamps.
            unsafe { std::slice::from_raw_parts(s.data as *const u64, s.count as usize) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(Ran {
            stamps,
            walls,
            lists: list_walls,
            frame: frame.as_secs_f64() * 1e3,
            between: between.as_secs_f64() * 1e3,
        })
    }

    /// Runs the frame as the options say.
    pub fn run(&mut self, o: &Options) -> Result<Report> {
        let mut made = self.prepare(o)?;
        for _ in 0..o.loops.max(1) {
            self.once(o, &mut made)?;
        }
        let mut report = self.finish(o, made)?;
        if o.leave_out > 0 {
            self.leave_out(o, &mut report)?;
        }
        Ok(report)
    }

    /// For the costliest pipelines of `report`: the frame as it is and
    /// without what runs them, turn by turn; what leaving each out saves is
    /// what it costs in the frame (and what it leaves undone for what comes
    /// after it: a pass that draws less is cheaper too).
    fn leave_out(&mut self, o: &Options, report: &mut Report) -> Result<()> {
        let plain = Options {
            detail: Detail::Off,
            leave_out: 0,
            ..o.clone()
        };
        let mut with = self.prepare(&plain)?;
        let groups: Vec<(String, Vec<u32>)> =
            report.heaviest.iter().take(o.leave_out).cloned().collect();
        for (name, ids) in groups {
            let mut without = self.prepare_without(&plain, &ids.iter().copied().collect())?;
            let from = with.runs.len();
            for _ in 0..o.loops.max(1) {
                self.time(&plain, &mut with)?;
                self.time(&plain, &mut without)?;
            }
            let of = |runs: &[Ran]| median(&runs.iter().map(|r| r.frame).collect::<Vec<_>>());
            report.left_out.push((
                name,
                without.report.skipped,
                of(&with.runs[from..]),
                of(&without.runs),
            ));
        }
        report
            .left_out
            .sort_by(|a, b| (b.2 - b.3).total_cmp(&(a.2 - a.3)));
        Ok(())
    }

    /// One run of the frame for its time alone.
    fn time(&mut self, o: &Options, p: &mut Prepared) -> Result<()> {
        self.use_shaders(o.shaders.as_deref())?;
        self.restore(o.restore)?;
        self.fresh = false;
        let ran = self.send(&p.sends, o)?;
        p.runs.push(ran);
        Ok(())
    }

    /// Whether commands are timed by timestamps or in pieces (or not).
    fn detail(&self, o: &Options) -> Detail {
        match o.detail {
            Detail::Auto if self.stamps_tell => Detail::Stamps,
            Detail::Auto => Detail::Pieces,
            Detail::Stamps if self.frequency <= 0.0 => Detail::Pieces,
            other => other,
        }
    }

    /// The lists for a run as the options say.
    pub fn prepare(&mut self, o: &Options) -> Result<Prepared> {
        self.prepare_without(o, &HashSet::new())
    }

    /// As [`Player::prepare`], with what runs the pipelines `also` left out
    /// too.
    fn prepare_without(&mut self, o: &Options, also: &HashSet<u32>) -> Result<Prepared> {
        let mut report = Report {
            notes: self.notes.clone(),
            waits: o.waits,
            ..Default::default()
        };
        self.use_shaders(o.shaders.as_deref())?;
        if !matches!(o.waits, Waits::Each | Waits::Lists) {
            self.make_versions()?;
            self.make_deltas()?;
        }
        if o.keep_clamps {
            self.keep_clamps()?;
        }
        let submits = self.edited(o, &mut report, also);
        let stamped = self.detail(o) == Detail::Stamps;
        Ok(Prepared {
            sends: self.build(submits, o, stamped)?,
            report,
            runs: Vec::new(),
            pieces: Vec::new(),
            picture: None,
        })
    }

    /// One run of the frame, noted for the report.
    pub fn once(&mut self, o: &Options, p: &mut Prepared) -> Result<()> {
        self.use_shaders(o.shaders.as_deref())?;
        self.restore(o.restore)?;
        self.fresh = false;
        let ran = self.send(&p.sends, o)?;
        // The picture now: another frame may run before the report is made.
        let before = p.picture.take().map(|p| p.rgb);
        p.picture = self.picture_now()?;
        if let Some(now) = &p.picture {
            p.report.pictures.push((now.db, now.hash.clone()));
            if let Some(before) = &before {
                p.report.loops_differ.push(differing(&now.rgb, before).0);
            }
        }
        if self.detail(o) == Detail::Pieces {
            self.restore(o.restore)?;
            self.fresh = false;
            let timed = self.in_pieces(&p.sends, &ran)?;
            p.pieces.push(timed);
        }
        p.runs.push(ran);
        Ok(())
    }

    /// Runs the frame cut into pieces, each sent alone and waited for: the
    /// time of each by kind and by pipeline, less what a submission that
    /// does nothing takes. `whole` is the run of the frame whose time the
    /// pieces share out.
    #[allow(clippy::type_complexity)]
    fn in_pieces(
        &mut self,
        s: &Sends,
        whole: &Ran,
    ) -> Result<Pieces> {
        let (mesh, dispatches) = self.kinds();
        // By submission: each piece's kind, pipeline, commands and ms.
        let mut all: Vec<Vec<(&'static str, u32, u32, f64)>> = Vec::new();
        // Each piece's time: of the frame's own, and sent alone.
        let mut each = Vec::new();
        let (allocator, list) = self.plain_list()?;
        // SAFETY: a list just made, with nothing recorded.
        unsafe { list.Close()? };
        for (i, lists) in s.submits.iter().enumerate() {
            self.live_views(self.version_of.get(i).copied().unwrap_or(0));
            self.write_deltas(i);
            let mut timed = Vec::new();
            for cmds in lists {
                for piece in edit::pieces(cmds, &dispatches, &mesh) {
                    // SAFETY: a closed list of ours whose last run has
                    // finished; the piece's commands as the host recorded
                    // them.
                    let took = unsafe {
                        allocator.Reset()?;
                        list.Reset(&allocator, None)?;
                        self.translate(&list, &piece.cmds, None, None);
                        list.Close()
                            .context("a piece of a list could not be recorded")?;
                        let began = Instant::now();
                        self.worker
                            .queue
                            .ExecuteCommandLists(&[Some(list.cast()?)]);
                        self.worker
                            .idle()
                            .context("the device was lost running a piece of the frame")?;
                        began.elapsed().as_secs_f64() * 1e3
                    };
                    let ms = (took - self.empty_ms).max(0.0);
                    timed.push((piece.kind, piece.pso, piece.works, ms));
                }
            }
            all.push(timed);
        }
        // The frame's own time shared out: by submission where each was
        // waited for, else the whole frame's.
        let alone: f64 = all.iter().flatten().map(|p| p.3).sum();
        let mut kinds: BTreeMap<&'static str, Timed> = BTreeMap::new();
        let mut shaders: HashMap<u32, Timed> = HashMap::new();
        for (i, timed) in all.iter().enumerate() {
            let of_these: f64 = timed.iter().map(|p| p.3).sum();
            let scale = match whole.walls.get(i) {
                Some(wall) if of_these > 0.0 => wall / of_these,
                None if alone > 0.0 => whole.frame / alone,
                _ => 0.0,
            };
            for &(kind, pso, works, ms) in timed {
                each.push((ms * scale, ms));
                let k = kinds.entry(kind).or_default();
                *k = (k.0 + ms, k.1 + ms * scale, k.2 + works);
                if pso != NONE {
                    let p = shaders.entry(pso).or_default();
                    *p = (p.0 + ms, p.1 + ms * scale, p.2 + works);
                }
            }
        }
        // What of each pipeline's pieces is more than a piece takes that does
        // next to nothing (the middle one of all): its heavy work.
        let floor = median(&all.iter().flatten().map(|p| p.3).collect::<Vec<_>>());
        let mut heavy: HashMap<u32, f64> = HashMap::new();
        for &(_, pso, _, ms) in all.iter().flatten() {
            if pso != NONE {
                *heavy.entry(pso).or_default() += (ms - floor).max(0.0);
            }
        }
        Ok((kinds, shaders, each, heavy))
    }

    /// The report after the last run.
    pub fn finish(&mut self, o: &Options, p: Prepared) -> Result<Report> {
        let Prepared {
            sends,
            mut report,
            runs,
            pieces,
            picture,
        } = p;
        let ms = |stamps: &[u64], a: u32, b: u32| match (stamps.get(a as usize), stamps.get(b as usize)) {
            (Some(a), Some(b)) if self.frequency > 0.0 => {
                b.saturating_sub(*a) as f64 * 1e3 / self.frequency
            }
            _ => 0.0,
        };
        let lists: Vec<&Built> = sends.steps.iter().flat_map(|(_, l)| l).collect();
        let (Some(first), Some(last)) = (lists.first(), lists.last()) else {
            bail!("no lists to run");
        };
        let each = |stamps: &[u64], l: &[Built]| match (l.first(), l.last()) {
            (Some(a), Some(b)) => ms(stamps, a.first, b.last),
            _ => 0.0,
        };
        let mut stamp_submits: Vec<Vec<f64>> = vec![Vec::new(); sends.steps.len()];
        let mut wall_submits: Vec<Vec<f64>> = vec![Vec::new(); sends.steps.len()];
        for ran in &runs {
            report.frame_ms.push(ran.frame);
            report.between_ms.push(ran.between);
            for (i, w) in ran.walls.iter().enumerate() {
                wall_submits[i].push(*w);
            }
            if ran.stamps.is_empty() {
                continue;
            }
            report
                .stamp_frame_ms
                .push(ms(&ran.stamps, first.first, last.last));
            let mut busy = 0.0;
            for (i, (_, l)) in sends.steps.iter().enumerate() {
                let t = each(&ran.stamps, l);
                stamp_submits[i].push(t);
                busy += t;
            }
            report.stamp_busy_ms.push(busy);
            report.stamp_delta_ms.push(
                sends
                    .steps
                    .iter()
                    .filter_map(|(d, _)| d.as_ref())
                    .map(|d| ms(&ran.stamps, d.first, d.last))
                    .sum(),
            );
        }
        // The first loop is slower: left out of the medians where there are
        // others.
        let later = |v: &[f64]| -> Option<f64> {
            match v.len() {
                0 => None,
                1 | 2 => Some(median(v)),
                _ => Some(median(&v[1..])),
            }
        };
        report.submits = sends
            .steps
            .iter()
            .enumerate()
            .map(|(i, (_, l))| (l.len(), later(&wall_submits[i]), later(&stamp_submits[i])))
            .collect();
        if o.waits == Waits::Lists {
            // Each list's time: the median of the loops but the first.
            let count = runs.iter().map(|r| r.lists.len()).min().unwrap_or(0);
            let mut all: Vec<(f64, usize, usize)> = Vec::new();
            let mut n = 0;
            for (i, lists) in sends.submits.iter().enumerate() {
                for j in 0..lists.len() {
                    if n < count {
                        let of: Vec<f64> = runs.iter().map(|r| r.lists[n]).collect();
                        all.push((later(&of).unwrap_or(0.0), i, j));
                    }
                    n += 1;
                }
            }
            for &(ms, _, _) in &all {
                let bucket = match ms {
                    ms if ms > 16.0 => 0,
                    ms if ms > 4.0 => 1,
                    ms if ms > 1.0 => 2,
                    _ => 3,
                };
                let b = &mut report.list_times[bucket];
                *b = (b.0 + 1, b.1 + ms);
            }
            all.sort_by(|a, b| b.0.total_cmp(&a.0));
            report.lists = all
                .iter()
                .take(o.top)
                .map(|&(ms, i, j)| {
                    (
                        ms,
                        format!(
                            "submission {i} list {j}: {}",
                            self.describe(&sends.submits[i][j])
                        ),
                    )
                })
                .collect();
        }
        match self.detail(o) {
            Detail::Stamps if sends.stamped => {
                report.detail = Some(
                    "by a timestamp after each: a command's time is from the end of the one before to its own end",
                );
                let all: Vec<&Vec<u64>> = runs.iter().map(|r| &r.stamps).collect();
                self.by_stamps(&sends, &all, o, &mut report);
            }
            Detail::Pieces => {
                report.detail = Some(
                    "in pieces: each run of commands of one kind with one pipeline was sent as a submission of its own and waited for (less what a submission that does nothing takes); the frame's own time is shared out by those times, submission by submission",
                );
                self.by_pieces(&sends, &pieces, o, &mut report);
            }
            _ => {}
        }
        self.picture(o, picture, &mut report)?;
        if self.restored.0 > 0 {
            report.notes.push(format!(
                "between loops, {} MB of the buffers the CPU maps are put back from the capture ({:.2} s)",
                self.restored.0 >> 20,
                self.restored.1
            ));
        }
        let missed = gpu::missed();
        if missed > 0 {
            report.notes.push(format!(
                "{missed} waits since the start found their work done without having been woken for it (D3DMetal did not set the event): each took up to a second too long, and a loop with one is an outlier"
            ));
        }
        if let Some(q) = &self.info {
            report.notes.extend(debug_messages(q));
        }
        Ok(report)
    }

    fn name_of(&self, pso: u32) -> String {
        match self
            .objects
            .get(&(pso & !COPY_TO))
            .filter(|_| pso & COPY_TO != 0)
        {
            Some(o) => {
                let r = &self.frame.resources[o.index];
                format!("copies to #{} {}", pso & !COPY_TO, r.name)
            }
            None => self.pipeline_name(pso),
        }
    }

    /// The time by command from timestamps: the median of the loops,
    /// without the first (which is slower) where there are others.
    fn by_stamps(&self, built: &Sends, stamps: &[&Vec<u64>], o: &Options, report: &mut Report) {
        let mut kinds: BTreeMap<&str, (Vec<f64>, u32)> = BTreeMap::new();
        let mut shaders: HashMap<u32, (Vec<f64>, u32)> = HashMap::new();
        let loops = if stamps.len() > 2 { &stamps[1..] } else { stamps };
        for stamps in loops {
            let mut of_kind: BTreeMap<&str, (f64, u32)> = BTreeMap::new();
            let mut of_shader: HashMap<u32, (f64, u32)> = HashMap::new();
            for l in built.steps.iter().flat_map(|(_, l)| l) {
                let Some(mut before) = stamps.get(l.first as usize).copied() else {
                    continue;
                };
                for &(at, kind, pso) in &l.stamps {
                    let Some(&now) = stamps.get(at) else { continue };
                    let ms = now.saturating_sub(before) as f64 * 1e3 / self.frequency;
                    before = before.max(now);
                    let k = of_kind.entry(kind).or_default();
                    *k = (k.0 + ms, k.1 + 1);
                    if pso != NONE {
                        let s = of_shader.entry(pso).or_default();
                        *s = (s.0 + ms, s.1 + 1);
                    }
                }
            }
            for (k, (ms, n)) in of_kind {
                let all = kinds.entry(k).or_default();
                all.0.push(ms);
                all.1 = n;
            }
            for (k, (ms, n)) in of_shader {
                let all = shaders.entry(k).or_default();
                all.0.push(ms);
                all.1 = n;
            }
        }
        self.detailed(
            kinds
                .into_iter()
                .map(|(k, (ms, n))| (k, median(&ms), n, None))
                .collect(),
            shaders
                .into_iter()
                .map(|(k, (ms, n))| (k, median(&ms), n, None))
                .collect(),
            &HashMap::new(),
            o,
            report,
        );
    }

    /// The time by command from runs in pieces: the median of the loops,
    /// without the first where there are others.
    #[allow(clippy::type_complexity)]
    fn by_pieces(
        &self,
        sends: &Sends,
        pieces: &[Pieces],
        o: &Options,
        report: &mut Report,
    ) {
        let loops = if pieces.len() > 2 { &pieces[1..] } else { pieces };
        // The heaviest pieces: the median of each over the loops.
        let (mesh, dispatches) = self.kinds();
        let mut each: Vec<(f64, f64, usize, usize, usize)> = Vec::new();
        let mut n = 0;
        for (i, lists) in sends.submits.iter().enumerate() {
            for (j, cmds) in lists.iter().enumerate() {
                for k in 0..edit::pieces(cmds, &dispatches, &mesh).len() {
                    let of = |which: fn(&(f64, f64)) -> f64| {
                        median(&loops.iter().filter_map(|l| l.2.get(n).map(which)).collect::<Vec<_>>())
                    };
                    each.push((of(|t| t.0), of(|t| t.1), i, j, k));
                    n += 1;
                }
            }
        }
        each.sort_by(|a, b| b.0.total_cmp(&a.0));
        report.pieces = each
            .iter()
            .take(o.top)
            .map(|&(of_frame, alone, i, j, k)| {
                let all = edit::pieces(&sends.submits[i][j], &dispatches, &mesh);
                let piece = &all[k];
                // What it does, not what it binds again first.
                (
                    of_frame,
                    alone,
                    format!(
                        "submission {i} list {j} piece {k} of {}: {}",
                        all.len(),
                        self.describe(&piece.cmds)
                    ),
                )
            })
            .collect();
        let mut kinds: BTreeMap<&str, (Vec<f64>, Vec<f64>, u32)> = BTreeMap::new();
        let mut shaders: HashMap<u32, (Vec<f64>, Vec<f64>, u32)> = HashMap::new();
        let mut heavy: HashMap<u32, Vec<f64>> = HashMap::new();
        for (_, _, _, of_heavy) in loops {
            for (k, ms) in of_heavy {
                heavy.entry(*k).or_default().push(*ms);
            }
        }
        let heavy: HashMap<u32, f64> = heavy.into_iter().map(|(k, v)| (k, median(&v))).collect();
        for (of_kind, of_shader, _, _) in loops {
            for (k, (ms, of_frame, n)) in of_kind {
                let all = kinds.entry(k).or_default();
                all.0.push(*ms);
                all.1.push(*of_frame);
                all.2 = *n;
            }
            for (k, (ms, of_frame, n)) in of_shader {
                let all = shaders.entry(*k).or_default();
                all.0.push(*ms);
                all.1.push(*of_frame);
                all.2 = *n;
            }
        }
        self.detailed(
            kinds
                .into_iter()
                .map(|(k, (ms, of_frame, n))| (k, median(&ms), n, Some(median(&of_frame))))
                .collect(),
            shaders
                .into_iter()
                .map(|(k, (ms, of_frame, n))| (k, median(&ms), n, Some(median(&of_frame))))
                .collect(),
            &heavy,
            o,
            report,
        );
    }

    fn detailed(
        &self,
        kinds: Vec<(&str, f64, u32, Option<f64>)>,
        shaders: Vec<(u32, f64, u32, Option<f64>)>,
        heavy: &HashMap<u32, f64>,
        o: &Options,
        report: &mut Report,
    ) {
        // By the share of the frame's own time where there is one.
        let key = |ms: f64, of_frame: Option<f64>| of_frame.unwrap_or(ms);
        report.categories = kinds
            .into_iter()
            .map(|(k, ms, n, of_frame)| (k.to_owned(), ms, n, of_frame))
            .collect();
        report
            .categories
            .sort_by(|a, b| key(b.1, b.3).total_cmp(&key(a.1, a.3)));
        // Pipelines of the same shaders together.
        let mut groups: Vec<(String, Vec<u32>, f64, u32, Option<f64>)> = Vec::new();
        let mut at: HashMap<String, usize> = HashMap::new();
        for (pso, ms, n, of_frame) in shaders {
            let key = if pso & COPY_TO != 0 {
                self.name_of(pso)
            } else {
                self.group(pso)
            };
            let i = *at.entry(key.clone()).or_insert_with(|| {
                groups.push((key, Vec::new(), 0.0, 0, of_frame.map(|_| 0.0)));
                groups.len() - 1
            });
            let g = &mut groups[i];
            g.1.push(pso);
            (g.2, g.3) = (g.2 + ms, g.3 + n);
            g.4 = g.4.zip(of_frame).map(|(a, b)| a + b);
        }
        groups.sort_by(|a, b| key(b.2, b.4).total_cmp(&key(a.2, a.4)));
        // What to try leaving out: by their heavy work, where that is
        // known (many small pieces are mostly what a piece costs alone).
        let weight = |ids: &[u32]| ids.iter().filter_map(|id| heavy.get(id)).sum::<f64>();
        let mut heaviest: Vec<(f64, String, Vec<u32>)> = groups
            .iter()
            .filter(|g| g.1.iter().all(|id| id & COPY_TO == 0))
            .map(|g| {
                let name = match g.1.len() {
                    1 => self.name_of(g.1[0]),
                    several => format!("{} ({several} pipelines)", self.name_of(g.1[0])),
                };
                (if heavy.is_empty() { key(g.2, g.4) } else { weight(&g.1) }, name, g.1.clone())
            })
            .collect();
        heaviest.sort_by(|a, b| b.0.total_cmp(&a.0));
        report.heaviest = heaviest.into_iter().map(|h| (h.1, h.2)).collect();
        groups.truncate(o.top);
        report.shader_ids = groups.iter().map(|g| g.1.clone()).collect();
        report.shaders = groups
            .into_iter()
            .map(|(_, ids, ms, n, of_frame)| {
                let name = match ids.len() {
                    1 => self.name_of(ids[0]),
                    several => format!("{} ({several} pipelines)", self.name_of(ids[0])),
                };
                (ms, n, name, of_frame)
            })
            .collect();
    }

    /// The presented picture as it is now, against the recorded one.
    fn picture_now(&mut self) -> Result<Option<Picture>> {
        let Some(p) = self.frame.presents.last().cloned() else {
            return Ok(None);
        };
        let Some(bytes) = self.contents(p.res)? else {
            return Ok(None);
        };
        let Some(rgb) = plain::rgb(p.width, p.height, p.pitch, p.format, &bytes) else {
            return Ok(None);
        };
        let mut recorded = vec![0; bytes.len()];
        read_at(
            &File::open(self.folder.join("present.bin"))?,
            &mut recorded,
            p.file,
        )?;
        let was = plain::rgb(p.width, p.height, p.pitch, p.format, &recorded).unwrap_or_default();
        let hash = blake3::hash(&rgb).to_hex().to_string();
        Ok(Some(Picture {
            db: plain::psnr(&rgb, &was),
            same: rows_equal(&p, &bytes, &recorded),
            differs: differing(&rgb, &was),
            hash,
            rgb,
        }))
    }

    /// Saves the picture a run left (`picture`) and says how it compares
    /// with the recorded one.
    fn picture(
        &mut self,
        o: &Options,
        picture: Option<Picture>,
        report: &mut Report,
    ) -> Result<()> {
        let Some(p) = self.frame.presents.last() else {
            report.notes.push("no picture was presented".into());
            return Ok(());
        };
        let Some(now) = picture else {
            report
                .notes
                .push("the presented texture is not there, or its format is not shown".into());
            return Ok(());
        };
        let path = o
            .image
            .clone()
            .unwrap_or_else(|| self.folder.join("replay.png"));
        std::fs::write(&path, plain::png(p.width, p.height, &now.rgb))?;
        report.picture = Some((now.db, now.same));
        report.picture_differs = (!now.same).then_some(now.differs);
        report.notes.push(format!(
            "picture {} (blake3 {})",
            path.display(),
            &now.hash[..16],
        ));
        Ok(())
    }

    /// What a resource holds now (after a run): a buffer's bytes, or a
    /// texture's first subresource in its copyable layout.
    pub fn contents(&mut self, id: u32) -> Result<Option<Vec<u8>>> {
        self.contents_of(id, 0)
    }

    /// As [`Player::contents`], of a texture's subresource `sub`.
    pub fn contents_of(&mut self, id: u32, sub: u32) -> Result<Option<Vec<u8>>> {
        let Some(o) = self.objects.get(&id) else {
            return Ok(None);
        };
        let (res, r, subs) = (o.res.clone(), &self.frame.resources[o.index], o.subs);
        let desc = gpu::to_desc(&r.desc);
        let is_buffer = r.desc.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0;
        let states = if self.fresh { &self.start } else { &self.end };
        let state = states.get(&id).cloned().unwrap_or(vec![COMMON]);
        let (f, total) = if is_buffer {
            (Vec::new(), r.desc.width)
        } else {
            gpu::footprints(&self.device, &desc, sub, 1)
        };
        let (back, data) = gpu::buffer(
            &self.device,
            D3D12_HEAP_TYPE_READBACK,
            total,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        self.worker.run(|list| {
            if is_buffer {
                // SAFETY: an open list; live buffers.
                unsafe { list.CopyBufferRegion(&back, 0, &res, 0, total) };
            } else if state.len() == 1 || subs == 1 {
                read_sub(list, &res, sub, state[0], &back, f[0]);
            } else {
                let (mut before, mut after) = (Vec::new(), Vec::new());
                gpu::moves(&res, &state, &[COPY_SOURCE], &mut before);
                gpu::moves(&res, &[COPY_SOURCE], &state, &mut after);
                // SAFETY: an open list; live resources.
                unsafe { barriers(list, &before) };
                read_sub(list, &res, sub, COPY_SOURCE, &back, f[0]);
                // SAFETY: as above.
                unsafe { barriers(list, &after) };
            }
        })?;
        // SAFETY: the buffer's mapping; the copy into it has run.
        Ok(Some(
            unsafe { std::slice::from_raw_parts(data, total as usize) }.to_vec(),
        ))
    }

    /// The id of the resource of a name (tests).
    pub fn named(&self, name: &str) -> Option<u32> {
        self.frame
            .resources
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.id)
    }

    /// The frame as the capture has it (tests).
    pub fn frame(&self) -> &FrameInfo {
        // The same thing under a name of this module's.
        &self.frame
    }
}

/// What a capture holds (`frame.bin`).
pub type FrameInfo = Frame;

/// What `--retranslate` did.
#[derive(Clone, Debug, Default)]
pub struct Retranslated {
    /// Pipelines of the title's in the capture, and those whose containers
    /// were all found.
    pub pipelines: usize,
    pub found: usize,
    /// Files written, and those of them that are the capture's own shaders
    /// bit for bit (all of them, with the build and switches of the run
    /// that was recorded).
    pub written: usize,
    pub same: usize,
    /// Stages that did not translate: stage, hash and why.
    pub refused: Vec<String>,
    pub seconds: f64,
}

/// Translates the shaders of the capture's pipelines again, with this build
/// and the switches of this run (`VICTORY_*`), into `into`: files a replay
/// takes with `--shaders` ([`shader_files`]). The title's containers are
/// those the recorder kept beside the capture (`containers`).
pub fn retranslate(folder: &Path, into: &Path) -> Result<Retranslated> {
    let capture = find(folder)?;
    let frame = Frame::read(&capture.join("frame.bin"))?;
    let kept = capture.with_file_name("containers");
    crate::make_dir(into)?;
    let began = Instant::now();
    let mut out = Retranslated::default();
    let mut written: HashSet<String> = HashSet::new();
    for p in &frame.pipelines {
        let stages = title_stages(p);
        let named: Vec<(u32, &'static str, &str)> = stages
            .iter()
            .filter_map(|(blob, s)| s.map(|(stage, hash)| (*blob, stage, hash)))
            .collect();
        if named.is_empty() {
            // A pass of ours: its code is the build's own.
            continue;
        }
        out.pipelines += 1;
        let containers: Option<Vec<Vec<u8>>> = named
            .iter()
            .map(|(_, _, hash)| crate::read_file(&kept.join(format!("{hash}.cdxil"))).ok())
            .collect();
        let Some(containers) = containers else {
            continue;
        };
        out.found += 1;
        let stages: Vec<(&'static str, &[u8])> = named
            .iter()
            .zip(&containers)
            .map(|((_, stage, _), bytes)| (*stage, &bytes[..]))
            .collect();
        for ((blob, stage, hash), (made, own)) in named
            .iter()
            .zip(crate::pipeline::host::retranslated(&stages))
        {
            let names = shader_files(&frame, p, *blob);
            // A translation made for this pipeline alone goes under the
            // pipeline's name, the shader's own under the shader's.
            let name = if own { names.first() } else { names.last() };
            let Some(name) = name else { continue };
            match made {
                Ok(dxil) => {
                    if written.insert(name.clone()) {
                        let recorded = frame.codes.get(*blob as usize).map(|c| c.hash);
                        out.same += usize::from(recorded == Some(*blake3::hash(&dxil).as_bytes()));
                        std::fs::write(into.join(name), dxil)?;
                        out.written += 1;
                    }
                }
                Err(e) => {
                    let line = format!("{stage} {hash}: {e}");
                    if !out.refused.contains(&line) {
                        out.refused.push(line);
                    }
                }
            }
        }
    }
    out.seconds = began.elapsed().as_secs_f64();
    Ok(out)
}

fn rows_equal(p: &wire::Present, a: &[u8], b: &[u8]) -> bool {
    let texel = match p.format {
        9..=14 => 8,
        _ => 4,
    };
    (0..p.height as usize).all(|y| {
        let from = y * p.pitch as usize;
        let to = from + p.width as usize * texel;
        a.get(from..to) == b.get(from..to)
    })
}

fn to_rect(r: [i32; 4]) -> RECT {
    RECT {
        left: r[0],
        top: r[1],
        right: r[2],
        bottom: r[3],
    }
}

/// # Safety
/// An open list; barriers of live resources.
unsafe fn barriers(list: &ID3D12GraphicsCommandList, all: &[D3D12_RESOURCE_BARRIER]) {
    for some in all.chunks(1000) {
        // SAFETY: the caller's.
        unsafe { list.ResourceBarrier(some) };
    }
}

fn uav_barrier(res: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer (or none) never released.
                pResource: res.map_or(std::mem::ManuallyDrop::new(None), |r| unsafe {
                    std::mem::transmute_copy(r)
                }),
            }),
        },
    }
}

/// Any placed resource to any other: waits for all writes.
fn alias() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                pResourceAfter: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

fn read_at(file: &File, mut to: &mut [u8], mut at: u64) -> std::io::Result<()> {
    while !to.is_empty() {
        let n = file.seek_read(to, at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        to = &mut to[n..];
        at += n as u64;
    }
    Ok(())
}

fn make_device(frame: &Frame, debug: bool) -> Result<(ID3D12Device, Option<ID3D12InfoQueue>)> {
    // SAFETY: plain creation and queries.
    unsafe {
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!("no debug layer: {e}"),
            }
        }
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let mut adapter = None;
        for i in 0.. {
            let Ok(a) = factory.EnumAdapters1(i) else {
                break;
            };
            let d = a.GetDesc1()?;
            if (d.VendorId, d.DeviceId) == (frame.vendor, frame.device) {
                adapter = Some(a);
                break;
            }
        }
        if adapter.is_none() {
            tracing::warn!(
                "the adapter recorded on ({}) is not here: the default one is used",
                frame.adapter
            );
        }
        let mut device: Option<ID3D12Device> = None;
        match &adapter {
            Some(a) => D3D12CreateDevice(a, D3D_FEATURE_LEVEL_12_0, &mut device)?,
            None => D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?,
        }
        let device = device.context("no device")?;
        let info = if debug {
            device.cast::<ID3D12InfoQueue>().ok()
        } else {
            None
        };
        Ok((device, info))
    }
}

/// The debug layer's errors: how many of each kind, and its first text.
fn debug_messages(q: &ID3D12InfoQueue) -> Vec<String> {
    let mut seen: BTreeMap<i32, (u32, String)> = BTreeMap::new();
    // SAFETY: messages read into buffers of the size asked for.
    unsafe {
        for i in 0..q.GetNumStoredMessages() {
            let mut size = 0;
            if q.GetMessage(i, None, &mut size).is_err() {
                continue;
            }
            let mut buffer = vec![0u64; size.div_ceil(8)];
            let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if q.GetMessage(i, Some(message), &mut size).is_err() {
                continue;
            }
            let m = &*message;
            if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                let text = || {
                    String::from_utf8_lossy(text)
                        .trim_end_matches('\0')
                        .to_owned()
                };
                seen.entry(m.ID.0).or_insert_with(|| (0, text())).0 += 1;
            }
        }
        q.ClearStoredMessages();
    }
    seen.into_iter()
        .map(|(id, (n, text))| format!("debug layer: {n} errors {id}, first: {text}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_are_looked_up_by_the_title_s_names() {
        let code = |i: u8| wire::Code {
            at: 0,
            len: 0,
            hash: [i; 32],
        };
        let part = |kind: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, blob| wire::Part::Code {
            kind: kind.0,
            blob,
        };
        let pipeline = |name: &str, parts| wire::Pipeline {
            id: 1,
            parts,
            name: name.into(),
        };
        let frame = Frame {
            adapter: String::new(),
            vendor: 0,
            device: 0,
            number: 0,
            world: 0,
            partial: false,
            notes: Vec::new(),
            heaps: Vec::new(),
            resources: Vec::new(),
            roots: Vec::new(),
            codes: vec![code(0xab), code(0xcd)],
            pipelines: Vec::new(),
            signatures: Vec::new(),
            descriptor_heaps: Vec::new(),
            query_heaps: Vec::new(),
            lists: Vec::new(),
            submits: Vec::new(),
            contents: Vec::new(),
            presents: Vec::new(),
        };
        let draw = pipeline(
            "[mesh 0123456789abcdef, pixel fedcba9876543210]",
            vec![
                part(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, 0),
                part(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, 1),
            ],
        );
        assert_eq!(
            shader_files(&frame, &draw, 0),
            [
                "0123456789abcdef@fedcba9876543210.dxil",
                "0123456789abcdef.dxil"
            ]
        );
        assert_eq!(
            shader_files(&frame, &draw, 1),
            [
                "fedcba9876543210@0123456789abcdef.dxil",
                "fedcba9876543210.dxil"
            ]
        );
        let compute = pipeline(
            "[compute 00112233445566ff]",
            vec![part(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, 1)],
        );
        assert_eq!(shader_files(&frame, &compute, 1), ["00112233445566ff.dxil"]);
        // A pass of ours: by its own code.
        let ours = pipeline(
            "indirect.rs:377",
            vec![part(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, 0)],
        );
        assert_eq!(shader_files(&frame, &ours, 0), ["abababababababab.dxil"]);
    }

    #[test]
    fn reports() {
        let r = Report {
            frame_ms: vec![9.0, 5.0, 7.0],
            submits: vec![(2, Some(1.5), None)],
            picture: Some((f64::INFINITY, true)),
            notes: vec!["one".into()],
            ..Default::default()
        };
        assert_eq!(r.median_ms(), 7.0);
        let text = r.text("unchanged");
        assert!(text.contains("unchanged: 7.00 ms"));
        assert!(text.contains("by submission (lists: ms on the wall clock): 0 (2): 1.50"));
        assert!(text.contains("the same bytes"));
        assert!(text.contains("note: one"));
    }
}
