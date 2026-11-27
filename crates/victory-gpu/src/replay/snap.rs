//! One frame written down: what its submissions run, what the card's
//! memory held at its start, and what the CPU put into memory both see
//! before each submission.
//!
//! Nothing of the frame's memory is held a second time: contents go from
//! read-back buffers and from the buffers' own mappings straight to the
//! files, and of the memory both sides see only a number of each 4 KiB page
//! is kept ([`plain::page_hash`]) to tell what changed by the next
//! submission. `frame.bin` is written anew after the first submission and
//! every other second from then on (marked partial until the frame's
//! present), so what was captured of a frame outlives a title that ends in
//! the middle of it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::os::windows::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::gpu::{self, Seen, Worker};
use super::plain;
use super::record::{Heap, Ours, Recorder, Res, WORLD_DRAWS, lock, not_recorded_calls, read};
use super::states::{COMMON, COPY_SOURCE, Change, States};
use super::wire::{
    self, ALL, Cmd, Content, Delta, Loc, MISSING, NONE, Place, Slot, Va, ZERO, slot,
};

type Fault = Box<dyn std::error::Error + Send + Sync>;

/// A read-back buffer contents go through, and how many there are at the
/// most: what the capture holds of the card's memory at a time.
const STAGING: u64 = 256 << 20;
const STAGINGS: usize = 3;
/// The most of a buffer in one piece.
const PIECE: u64 = 32 << 20;
/// What the CPU changes is kept in units of.
const PAGE: usize = 4096;
/// How long `frame.bin` is left as it is while submissions of the captured
/// frame go by.
const CHECKPOINT: std::time::Duration = std::time::Duration::from_secs(2);
/// What one thread compares at a time.
const CHUNK: usize = 1 << 20;

#[derive(Default)]
enum Stage {
    #[default]
    Waiting,
    /// Lists are recorded, since this present; the frame asked for (0: the
    /// first whose lists are whole).
    Armed(u64, u64),
    Capturing(Box<Capture>),
    Done,
}

/// Where the recorder is.
#[derive(Default)]
pub(super) struct Phase {
    /// The number of the last present (the title's frame number).
    presents: u64,
    /// Presents seen by the recorder.
    seen: u64,
    /// Draws run on the host by the last present, and the world's first
    /// frame (0: not seen yet).
    drawn: u64,
    world: u64,
    stage: Stage,
    /// Submissions of this frame so far, and lists among them that were not
    /// recorded.
    submits: u32,
    missing: u32,
}

impl Phase {
    pub(super) fn done(&self) -> bool {
        matches!(self.stage, Stage::Done)
    }
}

/// A submission to a queue other than the drawing one: left out.
pub(super) fn other_queue(r: &Recorder, lists: u32) {
    if r.watching.load(Ordering::SeqCst) {
        r.problem_times(
            "lists sent to a queue other than the drawing one are not in the capture",
            u64::from(lists),
        );
    }
}

/// A submission to the drawing queue, made by `go`.
pub(super) fn submit(
    r: &Recorder,
    queue: &ID3D12CommandQueue,
    lists: &[*mut std::ffi::c_void],
    go: impl FnOnce(),
) {
    let mut phase = lock(&r.submits);
    type Closed = (Option<Arc<Vec<u8>>>, Arc<Vec<Change>>);
    let closed: Vec<Option<Closed>> = lists
        .iter()
        .map(|&p| r.list(p).and_then(|l| l.closed()))
        .collect();
    phase.submits += 1;
    let whole = |c: &Option<Closed>| c.as_ref().is_some_and(|c| c.0.is_some());
    phase.missing += lists
        .iter()
        .zip(&closed)
        .filter(|(p, c)| !whole(c) && r.knows(**p))
        .count() as u32;
    if let Stage::Capturing(c) = &mut phase.stage {
        let commands: Vec<Option<Arc<Vec<u8>>>> = closed
            .iter()
            .map(|c| c.as_ref().and_then(|c| c.0.clone()))
            .collect();
        let _ours = Ours::new();
        // `frame.bin` so far, after the first submission (the long one) and
        // then every other second: a title of 20,000 resources and a million
        // views has 60 MB of it.
        let done = c.submit(r, queue, &commands).and_then(|()| {
            if c.submits.len() > 1 && c.written.elapsed() < CHECKPOINT {
                return Ok(());
            }
            c.timed("frame.bin between submissions", |c| c.write(r, true))?;
            c.written = Instant::now();
            Ok(())
        });
        if let Err(e) = done {
            stop(r, &mut phase, &*e);
        }
    }
    go();
    let mut states = lock(&r.states);
    for change in closed.iter().flatten().flat_map(|c| c.1.iter()) {
        states.change(change);
    }
    states.decay();
}

/// The capture failed.
fn stop(r: &Recorder, phase: &mut Phase, e: &(dyn std::error::Error + Send + Sync)) {
    tracing::error!("frame recorder: the capture failed: {e}");
    let _ = std::fs::write(r.folder.join("failed.txt"), e.to_string());
    phase.stage = Stage::Done;
    r.watching.store(false, Ordering::SeqCst);
    r.arm(false);
    let held = std::mem::take(&mut *lock(&r.held));
    drop(held);
}

/// Whether file `go` is in the capture's folder (asked by opening it:
/// `crate::read_file` says why not for its metadata).
fn triggered(r: &Recorder) -> bool {
    File::open(r.settings.folder.join("go")).is_ok()
}

pub(super) fn present(
    r: &Recorder,
    number: u64,
    shown: Option<(&ID3D12Resource, u32)>,
    drawn: u64,
) {
    let mut phase = lock(&r.submits);
    phase.presents = number;
    phase.seen += 1;
    let presents = number;
    let draws = drawn.saturating_sub(std::mem::replace(&mut phase.drawn, drawn));
    if phase.world == 0 && draws >= WORLD_DRAWS {
        phase.world = presents;
        tracing::info!(
            frame = presents,
            draws,
            "frame recorder: the world's first frame"
        );
    }
    let (submits, missing) = (
        std::mem::take(&mut phase.submits),
        std::mem::take(&mut phase.missing),
    );
    let world = phase.world;
    let mut finished = None;
    match &mut phase.stage {
        Stage::Waiting => {
            let seconds = r.began.elapsed().as_secs_f64();
            let asked = plain::due(r.settings.at, presents, world, seconds);
            let by_file = asked.is_none() && phase.seen % 16 == 0 && triggered(r);
            if let Some(target) = asked.or(by_file.then_some(0)) {
                tracing::info!(
                    presents,
                    target,
                    "frame recorder: a frame is captured soon (lists are recorded from now on, if they were not)"
                );
                r.arm(true);
                phase.stage = Stage::Armed(presents, target);
            }
        }
        Stage::Armed(since, target) => {
            let whole = submits > 0 && missing == 0;
            if plain::captures_next(presents, *since, *target, whole) {
                if !whole {
                    r.problem("no frame was found whose lists were all recorded");
                }
                tracing::info!(presents, "frame recorder: the next frame is captured");
                match Capture::new(r, presents + 1, world) {
                    Ok(c) => phase.stage = Stage::Capturing(Box::new(c)),
                    Err(e) => stop(r, &mut phase, &*e),
                }
            }
        }
        Stage::Capturing(c) if c.started => {
            let _ours = Ours::new();
            let done = c.present(r, shown).and_then(|()| {
                if c.frame < r.settings.frames {
                    return c.write(r, true).map(|()| false);
                }
                c.write(r, false)?;
                Ok(true)
            });
            match done {
                Ok(false) => {}
                Ok(true) => {
                    r.watching.store(false, Ordering::SeqCst);
                    r.arm(false);
                    if let Stage::Capturing(c) = std::mem::replace(&mut phase.stage, Stage::Done) {
                        finished = Some((c, std::mem::take(&mut *lock(&r.held))));
                    }
                }
                Err(e) => stop(r, &mut phase, &*e),
            }
        }
        // A frame without a submission: the next one.
        Stage::Capturing(_) | Stage::Done => {}
    }
    drop(phase);
    // What was kept alive goes without a lock held.
    drop(finished);
}

/// The process is about to end: waits for what is being written of a
/// capture (a submission's part of it, with `frame.bin` after it), for a
/// minute at the most.
pub(super) fn ending(r: &Recorder) {
    let began = Instant::now();
    loop {
        match r.submits.try_lock() {
            Ok(phase) => {
                if let Stage::Capturing(c) = &phase.stage
                    && c.started
                {
                    tracing::warn!(
                        submissions = c.submits.len(),
                        "frame recorder: the process ends in the middle of the captured frame; frame.bin holds its submissions so far"
                    );
                }
                return;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => return,
            Err(std::sync::TryLockError::WouldBlock) => {
                if began.elapsed() > std::time::Duration::from_secs(60) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
}

/// What a submission reads of buffers, by id: parts of them (copied from),
/// or all (read some other way).
#[derive(Default)]
struct Reads {
    all: HashSet<u32>,
    parts: HashMap<u32, Vec<(u64, u64)>>,
}

impl Reads {
    fn note(&mut self, cmds: &[Cmd]) {
        let (all, parts) = (&mut self.all, &mut self.parts);
        let mut whole = |va: &Va| {
            if let Va::At { res, .. } = va {
                all.insert(*res);
            }
        };
        for c in cmds {
            match c {
                Cmd::RootView { va, .. }
                | Cmd::IndexBuffer {
                    view: Some((va, _, _)),
                } => whole(va),
                Cmd::VertexBuffers {
                    views: Some(views), ..
                } => views.iter().for_each(|(va, _, _)| whole(va)),
                Cmd::Indirect { args, count, .. } => {
                    for res in [*args, *count] {
                        whole(&Va::At { res, offset: 0 });
                    }
                }
                Cmd::CopyResource { src, .. } => whole(&Va::At {
                    res: *src,
                    offset: 0,
                }),
                Cmd::CopyBuffer {
                    src, src_at, len, ..
                } => parts.entry(*src).or_default().push((*src_at, src_at + len)),
                Cmd::CopyTexture {
                    src:
                        Loc::Placed {
                            res,
                            offset,
                            size,
                            pitch,
                            ..
                        },
                    ..
                } => {
                    // At the most: compressed formats have fewer rows.
                    let len = u64::from(*pitch) * u64::from(size[1]) * u64::from(size[2].max(1));
                    parts.entry(*res).or_default().push((*offset, offset + len));
                }
                _ => {}
            }
        }
    }

    /// The whole pages read of a buffer of `size` bytes; `viewed` by a
    /// descriptor, all of it is.
    fn pages(&self, id: u32, size: usize, viewed: bool) -> Vec<(usize, usize)> {
        if viewed || self.all.contains(&id) {
            return vec![(0, size)];
        }
        let Some(parts) = self.parts.get(&id) else {
            return Vec::new();
        };
        let mut pages: Vec<(usize, usize)> = parts
            .iter()
            .map(|&(from, to)| {
                (
                    from as usize / PAGE * PAGE,
                    (to as usize).next_multiple_of(PAGE).min(size),
                )
            })
            .filter(|(from, to)| from < to)
            .collect();
        pages.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (from, to) in pages {
            match out.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => out.push((from, to)),
            }
        }
        out
    }
}

/// A buffer the CPU sees.
struct CpuBuffer {
    id: u32,
    /// Where it is read: its mapping.
    live: *const u8,
    size: usize,
    /// The number ([`plain::page_hash`]) of each of its pages as it was
    /// last written down: zero, a page of zeros', before the first time.
    kept: Vec<u64>,
}

// SAFETY: the mapping is of a buffer the recorder holds while the capture
// lasts.
unsafe impl Send for CpuBuffer {}
// SAFETY: as above; the mapping is only read.
unsafe impl Sync for CpuBuffer {}

struct Capture {
    worker: Worker,
    others: Vec<Worker>,
    started: bool,
    frame: u32,
    states: States,
    idle: HashSet<u32>,
    heaps: Vec<(Arc<Heap>, Vec<(u32, u32, Slot)>)>,
    contents: Vec<Content>,
    buffers: Vec<CpuBuffer>,
    known: HashSet<u32>,
    /// Resources descriptors name, or named during the frame.
    viewed: HashSet<u32>,
    /// The frame's number (its present's), and the world's first frame.
    number: u64,
    world: u64,
    deltas: File,
    delta_at: AtomicU64,
    lists: Vec<Arc<Vec<u8>>>,
    submits: Vec<wire::Submit>,
    presents: Vec<wire::Present>,
    present_file: File,
    present_at: u64,
    calls: Vec<(String, u64)>,
    began: Instant,
    took: BTreeMap<&'static str, std::time::Duration>,
    /// Bytes of memory both sides see that were looked at for changes.
    looked: u64,
    /// How long the title stood still in the frame's first submission.
    first: std::time::Duration,
    /// Bytes of the read-back buffers the contents went through.
    stagings: u64,
    /// When `frame.bin` was last written.
    written: Instant,
}

impl Capture {
    fn new(r: &Recorder, number: u64, world: u64) -> Result<Capture, Fault> {
        let _ours = Ours::new();
        let device = r.device().ok_or("no device")?;
        let queues = lock(&r.queues).clone();
        // The drawing queue made last: the host layer replaces one that
        // macOS stopped running work on (`Host::replace_queue`), and what
        // is sent to the old one then never runs.
        let direct = queues
            .iter()
            .rev()
            .find(|q| q.1 == D3D12_COMMAND_LIST_TYPE_DIRECT.0)
            .ok_or("no drawing queue")?;
        let mut others = Vec::new();
        for (queue, _) in queues.iter().filter(|q| q.0 != direct.0) {
            others.push(Worker::new(&device, queue)?);
        }
        Ok(Capture {
            worker: Worker::new(&device, &direct.0)?,
            others,
            started: false,
            frame: 0,
            states: States::default(),
            idle: HashSet::new(),
            heaps: Vec::new(),
            contents: Vec::new(),
            buffers: Vec::new(),
            known: HashSet::new(),
            viewed: HashSet::new(),
            number,
            world,
            deltas: File::create(r.folder.join("deltas.bin"))?,
            delta_at: AtomicU64::new(0),
            lists: Vec::new(),
            submits: Vec::new(),
            presents: Vec::new(),
            present_file: File::create(r.folder.join("present.bin"))?,
            present_at: 0,
            calls: not_recorded_calls(),
            began: Instant::now(),
            took: BTreeMap::new(),
            looked: 0,
            first: std::time::Duration::ZERO,
            stagings: 0,
            written: Instant::now(),
        })
    }

    fn timed<T>(
        &mut self,
        what: &'static str,
        f: impl FnOnce(&mut Self) -> Result<T, Fault>,
    ) -> Result<T, Fault> {
        let began = Instant::now();
        let out = f(self);
        *self.took.entry(what).or_default() += began.elapsed();
        out
    }

    /// Before the first submission of the frame: what the card holds.
    fn start(&mut self, r: &Recorder) -> Result<(), Fault> {
        self.began = Instant::now();
        r.watching.store(true, Ordering::SeqCst);
        self.started = true;
        self.worker.idle()?;
        for w in &mut self.others {
            w.idle()?;
        }
        let mut all: Vec<Arc<Res>> = Vec::new();
        {
            let live = read(&r.resources);
            let mut held = lock(&r.held);
            for res in live.values().filter(|res| !res.late) {
                let object = res.object();
                // SAFETY: a live resource, or one whose end waits for the
                // lock held here (`Recorder::gone`): the count tells which.
                let count = unsafe {
                    (Interface::vtable(&*object)
                        .base__
                        .base__
                        .base__
                        .base__
                        .AddRef)(object.as_raw())
                };
                if count <= 1 {
                    // Dying: the reference does not bring it back, and
                    // giving it back would end it a second time.
                    continue;
                }
                // SAFETY: the reference just taken. (The count does not tell
                // all that are dying: `Recorder::gone` takes those out.)
                held.insert(res.id, unsafe { ID3D12Resource::from_raw(object.as_raw()) });
                all.push(res.clone());
            }
        }
        all.sort_by_key(|res| res.id);
        self.states = lock(&r.states).clone();
        self.idle = self.find_idle(&all);
        self.timed("contents of the card's memory", |c| c.snapshot(r, &all))?;
        Ok(())
    }

    /// Of placed resources that overlap, those that are not the one used
    /// last.
    fn find_idle(&self, all: &[Arc<Res>]) -> HashSet<u32> {
        let mut heaps: HashMap<u32, Vec<(u64, u64, u64, u32)>> = HashMap::new();
        for res in all {
            if let Place::Placed { heap, offset } = res.place {
                // SAFETY: a plain query.
                let info = unsafe {
                    self.worker
                        .device
                        .GetResourceAllocationInfo(0, &[gpu::to_desc(&res.desc)])
                };
                let size = if info.SizeInBytes == u64::MAX {
                    res.desc.width
                } else {
                    info.SizeInBytes
                };
                let used = self.states.used.get(&res.id).copied().unwrap_or(0);
                heaps
                    .entry(heap)
                    .or_default()
                    .push((used, offset, offset + size, res.id));
            }
        }
        let mut idle = HashSet::new();
        for (_, mut list) in heaps {
            list.sort_by(|a, b| b.cmp(a));
            let mut taken: BTreeMap<u64, u64> = BTreeMap::new();
            for (_, from, to, id) in list {
                let overlaps = taken
                    .range(..to)
                    .next_back()
                    .is_some_and(|(_, &end)| end > from);
                if overlaps {
                    idle.insert(id);
                } else {
                    taken.insert(from, to);
                }
            }
        }
        idle
    }

    fn snapshot(&mut self, r: &Recorder, all: &[Arc<Res>]) -> Result<(), Fault> {
        let device = self.worker.device.clone();
        let mut pieces: Vec<Piece> = Vec::new();
        for res in all.iter().filter(|res| res.seen == Seen::No) {
            if matches!(res.place, Place::BackBuffer {}) || self.idle.contains(&res.id) {
                continue;
            }
            if res.is_buffer() {
                let mut at = 0;
                while at < res.desc.width {
                    let len = PIECE.min(res.desc.width - at);
                    pieces.push(Piece {
                        res: res.clone(),
                        first: at,
                        count: 1,
                        len,
                        at: 0,
                    });
                    at += len;
                }
                continue;
            }
            if res.desc.samples[0] > 1 {
                r.problem("contents of multisampled textures are not in the capture");
                continue;
            }
            let desc = gpu::to_desc(&res.desc);
            let (_, total) = gpu::footprints(&device, &desc, 0, res.subs);
            if total == u64::MAX || total == 0 {
                r.problem("contents of a texture without a copyable layout are not in the capture");
            } else if total <= STAGING {
                pieces.push(Piece {
                    res: res.clone(),
                    first: 0,
                    count: res.subs,
                    len: total,
                    at: 0,
                });
            } else {
                for sub in 0..res.subs {
                    let (_, len) = gpu::footprints(&device, &desc, sub, 1);
                    if len <= STAGING {
                        pieces.push(Piece {
                            res: res.clone(),
                            first: u64::from(sub),
                            count: 1,
                            len,
                            at: 0,
                        });
                    } else {
                        r.problem(
                            "contents of a texture level of over 256 MB are not in the capture",
                        );
                    }
                }
            }
        }
        // Into batches, one read-back buffer each.
        let mut batches: Vec<Vec<Piece>> = vec![Vec::new()];
        let (mut at, mut most) = (0u64, 0u64);
        for mut p in pieces {
            at = at.next_multiple_of(512);
            if at + p.len > STAGING {
                batches.push(Vec::new());
                at = 0;
            }
            p.at = at;
            at += p.len;
            most = most.max(at);
            if let Some(b) = batches.last_mut() {
                b.push(p);
            }
        }
        // As large as the fullest batch, and no more of them than batches.
        let most = most.max(1).next_multiple_of(64 << 10);
        let mut stagings = Vec::new();
        for _ in 0..STAGINGS.min(batches.len()) {
            stagings.push(gpu::buffer(
                &device,
                D3D12_HEAP_TYPE_READBACK,
                most,
                D3D12_RESOURCE_FLAG_NONE,
            )?);
        }
        self.stagings = stagings.len() as u64 * most;
        let file = File::create(r.folder.join("contents.bin"))?;
        let end = AtomicU64::new(0);
        let contents: std::sync::Mutex<Vec<Content>> = std::sync::Mutex::default();
        let failed: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::default();
        let (todo, work) = std::sync::mpsc::channel::<(usize, Vec<Piece>)>();
        let work = std::sync::Mutex::new(work);
        let (free, freed) = std::sync::mpsc::channel::<usize>();
        for i in 0..stagings.len() {
            free.send(i)?;
        }
        let maps: Vec<usize> = stagings.iter().map(|s| s.1 as usize).collect();
        std::thread::scope(|scope| -> Result<(), Fault> {
            for _ in 0..stagings.len() {
                let (work, free, maps, file, end, contents, failed) =
                    (&work, free.clone(), &maps, &file, &end, &contents, &failed);
                scope.spawn(move || {
                    loop {
                        let next = lock(work).recv();
                        let Ok((i, pieces)) = next else { break };
                        for p in pieces {
                            // SAFETY: inside the buffer's mapping; the copy
                            // into it has run.
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    (maps[i] + p.at as usize) as *const u8,
                                    p.len as usize,
                                )
                            };
                            let at = if zero(bytes) {
                                ZERO
                            } else {
                                end.fetch_add(p.len, Ordering::Relaxed)
                            };
                            if at != ZERO
                                && let Err(e) = write_at(file, bytes, at)
                            {
                                lock(failed).get_or_insert(e);
                            }
                            lock(contents).push(Content {
                                res: p.res.id,
                                first: p.first,
                                count: p.count,
                                len: p.len,
                                file: at,
                            });
                        }
                        let _ = free.send(i);
                    }
                });
            }
            // Moved in: the writers end when it goes.
            let todo = todo;
            for mut pieces in batches {
                {
                    let held = lock(&r.held);
                    pieces.retain(|p| held.contains_key(&p.res.id));
                }
                if pieces.is_empty() {
                    continue;
                }
                let i = freed.recv()?;
                let value = self
                    .worker
                    .send(|list| copy_out(&device, list, &self.states, &pieces, &stagings[i].0))?;
                self.worker.wait(value)?;
                todo.send((i, pieces))?;
            }
            Ok(())
        })?;
        if let Some(e) = lock(&failed).take() {
            return Err(e.into());
        }
        self.contents = contents.into_inner().unwrap_or_else(|e| e.into_inner());
        Ok(())
    }

    /// Buffers the CPU sees that were not known yet.
    fn new_buffers(&mut self, r: &Recorder) -> Result<(), Fault> {
        let fresh: Vec<Arc<Res>> = read(&r.resources)
            .values()
            .filter(|res| {
                matches!(res.seen, Seen::Cached | Seen::Combined) && !self.known.contains(&res.id)
            })
            .cloned()
            .collect();
        for res in fresh {
            self.known.insert(res.id);
            if !res.is_buffer() {
                r.problem("contents of textures in memory the CPU sees are not in the capture");
                continue;
            }
            // Those alive at the start are held, those made since too; what
            // is not is dying.
            if !lock(&r.held).contains_key(&res.id) {
                continue;
            }
            let object = res.object();
            let size = res.desc.width as usize;
            // Read where the host layer writes it, whichever kind of memory
            // it is: upload memory reads as fast as any on a Mac. (Mapped
            // once more than the host layer has it, and left so.)
            let mut data = std::ptr::null_mut();
            // SAFETY: a live buffer in memory the CPU sees.
            unsafe { object.Map(0, None, Some(&mut data))? };
            if data.is_null() {
                r.problem("a buffer the CPU sees has no mapping: what the CPU wrote to it is missing");
                continue;
            }
            self.buffers.push(CpuBuffer {
                id: res.id,
                live: data.cast::<u8>().cast_const(),
                size,
                kept: vec![0; size.div_ceil(PAGE)],
            });
        }
        Ok(())
    }

    /// What changed in the buffers the CPU sees, of what a submission reads
    /// of them, since they were last looked at (the first time: what is not
    /// zero).
    fn changed(&mut self, r: &Recorder, reads: &Reads) -> Result<Vec<Delta>, Fault> {
        self.new_buffers(r)?;
        {
            let held = lock(&r.held);
            self.buffers.retain(|b| held.contains_key(&b.id));
        }
        let parts: Vec<(usize, usize, usize)> = self
            .buffers
            .iter()
            .enumerate()
            .flat_map(|(i, b)| {
                reads
                    .pages(b.id, b.size, self.viewed.contains(&b.id))
                    .into_iter()
                    .map(move |(from, to)| (i, from, to))
            })
            .collect();
        self.looked += parts.iter().map(|p| (p.2 - p.1) as u64).sum::<u64>();
        let chunks: Vec<(usize, usize, usize)> = parts
            .iter()
            .flat_map(|&(i, from, to)| {
                (from..to)
                    .step_by(CHUNK)
                    .map(move |at| (i, at, CHUNK.min(to - at)))
            })
            .collect();
        // Each chunk is one thread's; threads share what is kept by parts.
        struct Kept(*mut u64);
        // SAFETY: parts of it are used by one thread each.
        unsafe impl Sync for Kept {}
        let kept: Vec<Kept> = self
            .buffers
            .iter_mut()
            .map(|b| Kept(b.kept.as_mut_ptr()))
            .collect();
        let next = AtomicUsize::new(0);
        let out: std::sync::Mutex<Vec<Delta>> = std::sync::Mutex::default();
        let failed: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::default();
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .clamp(1, 16)
            .min(chunks.len().max(1));
        let (buffers, file, end) = (&self.buffers, &self.deltas, &self.delta_at);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    // The changed pages of a chunk, as they were read.
                    let mut taken = vec![0u8; CHUNK];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(b, at, len)) = chunks.get(i) else {
                            break;
                        };
                        // SAFETY: a chunk of the buffer's memory (it may
                        // change meanwhile: what is kept is of what was
                        // taken) and its pages' numbers, which no other
                        // thread has (chunks start at multiples of a page).
                        let runs = unsafe {
                            changed_runs(
                                buffers[b].live.add(at),
                                kept[b].0.add(at / PAGE),
                                len,
                                &mut taken,
                            )
                        };
                        let mut from = 0;
                        for (start, n) in runs {
                            let bytes = &taken[from..from + n];
                            from += n;
                            let file_at = end.fetch_add(n as u64, Ordering::Relaxed);
                            if let Err(e) = write_at(file, bytes, file_at) {
                                lock(&failed).get_or_insert(e);
                            }
                            mine.push(Delta {
                                res: buffers[b].id,
                                at: (at + start) as u64,
                                len: n as u64,
                                file: file_at,
                            });
                        }
                    }
                    lock(&out).extend(mine);
                });
            }
        });
        if let Some(e) = lock(&failed).take() {
            return Err(e.into());
        }
        let mut out = out.into_inner().unwrap_or_else(|e| e.into_inner());
        out.sort_by_key(|d| (d.res, d.at));
        Ok(out)
    }

    fn submit(
        &mut self,
        r: &Recorder,
        queue: &ID3D12CommandQueue,
        lists: &[Option<Arc<Vec<u8>>>],
    ) -> Result<(), Fault> {
        let first = !self.started;
        let began = Instant::now();
        // The capture reads the card's memory through the queue the frame
        // is sent on: one the host layer moved to since the capture was
        // prepared takes the place of the one found then.
        if self.worker.queue != *queue {
            if first {
                self.worker = Worker::new(&self.worker.device.clone(), queue)?;
            } else {
                r.problem("the frame's submissions went to more than one drawing queue: what the capture read of the card's memory may not be in their order");
            }
        }
        if first {
            self.start(r)?;
        } else {
            self.timed("waiting for the card", |c| Ok(c.worker.idle()?))?;
        }
        // Descriptors first: what they name is looked at below.
        let mut written = std::mem::take(&mut *lock(&r.writes));
        let mut views = Vec::new();
        if first {
            let heaps: Vec<Arc<Heap>> = read(&r.descriptors).visible().cloned().collect();
            for h in heaps {
                let runs = runs_of(r, &h.all());
                self.viewed
                    .extend(runs.iter().flat_map(|(_, _, s)| [s.res, s.counter]));
                self.heaps.push((h, runs));
            }
        } else {
            written.sort_unstable();
            let mut seen = HashSet::new();
            for (heap, at, n) in written {
                let h = match self.heaps.iter().find(|(h, _)| h.id == heap) {
                    Some((h, _)) => h.clone(),
                    None => {
                        let found = read(&r.descriptors)
                            .visible()
                            .find(|h| h.id == heap)
                            .cloned();
                        let Some(h) = found else { continue };
                        self.heaps.push((h.clone(), Vec::new()));
                        h
                    }
                };
                for i in at..at + n {
                    if seen.insert((heap, i)) {
                        let s = resolved(r, h.get(i));
                        self.viewed.extend([s.res, s.counter]);
                        views.push((heap, i, s));
                    }
                }
            }
        }
        let mut reads = Reads::default();
        for l in lists.iter().flatten() {
            reads.note(&Cmd::all(l)?);
        }
        let deltas = self.timed("what the CPU changed", |c| c.changed(r, &reads))?;
        if first {
            self.first = began.elapsed();
        }
        let mut numbers = Vec::new();
        for l in lists {
            numbers.push(match l {
                None => {
                    r.problem(
                        "a list was submitted that was not recorded whole: its work is missing",
                    );
                    MISSING
                }
                Some(commands) => match self
                    .lists
                    .iter()
                    .position(|have| Arc::ptr_eq(have, commands))
                {
                    Some(i) => i as u32,
                    None => {
                        self.lists.push(commands.clone());
                        self.lists.len() as u32 - 1
                    }
                },
            });
        }
        self.submits.push(wire::Submit {
            frame: self.frame,
            lists: numbers,
            deltas,
            views,
        });
        Ok(())
    }

    fn present(
        &mut self,
        r: &Recorder,
        shown: Option<(&ID3D12Resource, u32)>,
    ) -> Result<(), Fault> {
        self.worker.idle()?;
        let Some((texture, state)) = shown else {
            r.problem("a frame was presented without a picture the host layer shows");
            self.frame += 1;
            return Ok(());
        };
        // SAFETY: a live texture.
        let desc = unsafe { texture.GetDesc() };
        let (f, total) = gpu::footprints(&self.worker.device, &desc, 0, 1);
        let (back, data) = gpu::buffer(
            &self.worker.device,
            D3D12_HEAP_TYPE_READBACK,
            total,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        self.worker
            .run(|list| read_sub(list, texture, 0, state, &back, f[0]))?;
        // SAFETY: the buffer's mapping; the copy into it has run.
        let bytes = unsafe { std::slice::from_raw_parts(data, total as usize) };
        write_at(&self.present_file, bytes, self.present_at)?;
        let id = r.resource(texture.as_raw());
        let (width, height, pitch) = (desc.Width as u32, desc.Height, f[0].Footprint.RowPitch);
        self.presents.push(wire::Present {
            submits: self.submits.len() as u32,
            res: id,
            width,
            height,
            format: desc.Format.0,
            pitch,
            file: self.present_at,
        });
        self.present_at += total;
        match plain::rgb(width, height, pitch, desc.Format.0, bytes) {
            Some(rgb) => std::fs::write(
                r.folder.join(format!("frame-{}.png", self.frame)),
                plain::png(width, height, &rgb),
            )?,
            None => r.problem("the presented format is not shown in a PNG"),
        }
        self.frame += 1;
        Ok(())
    }

    /// Writes `frame.bin` and `notes.txt`: after every submission (`partial`)
    /// and when the frame is done.
    fn write(&mut self, r: &Recorder, partial: bool) -> Result<(), Fault> {
        let mut resources: Vec<wire::Resource> = Vec::new();
        {
            let names = lock(&r.names);
            for res in read(&r.resources).values() {
                let state = match (res.is_buffer(), res.late) {
                    (true, _) => vec![COMMON],
                    (_, true) => vec![res.made_in as u32],
                    _ => self.states.get(res.id).unwrap_or(vec![COMMON]),
                };
                resources.push(wire::Resource {
                    id: res.id,
                    place: res.place.clone(),
                    desc: res.desc.clone(),
                    made_in: res.made_in,
                    clear: res.clear,
                    state,
                    idle: self.idle.contains(&res.id),
                    late: res.late,
                    name: names.get(&res.id).cloned().unwrap_or_default(),
                    castable: res.castable.clone(),
                });
            }
        }
        resources.sort_by_key(|res| res.id);
        let mut heaps: Vec<wire::Heap> = {
            let all = lock(&r.heaps);
            let used: HashSet<u32> = resources
                .iter()
                .filter_map(|res| {
                    if let Place::Placed { heap, .. } = res.place {
                        Some(heap)
                    } else {
                        None
                    }
                })
                .collect();
            used.iter().filter_map(|id| all.get(id).cloned()).collect()
        };
        heaps.sort_by_key(|h| h.id);
        // What the lists name.
        let (mut psos, mut roots, mut signatures, mut queries) = (
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
        );
        for l in &self.lists {
            for c in Cmd::all(l)? {
                match c {
                    Cmd::Pipeline { pso } => _ = psos.insert(pso),
                    Cmd::RootSignature { root, .. } => _ = roots.insert(root),
                    Cmd::Indirect { signature, .. } => _ = signatures.insert(signature),
                    Cmd::BeginQuery { heap, .. }
                    | Cmd::EndQuery { heap, .. }
                    | Cmd::ResolveQuery { heap, .. } => _ = queries.insert(heap),
                    _ => {}
                }
            }
        }
        let pipelines: Vec<wire::Pipeline> = {
            let all = lock(&r.pipelines);
            let names = lock(&r.names);
            let mut out: Vec<wire::Pipeline> = psos
                .iter()
                .filter_map(|id| {
                    let parts = all.get(id)?;
                    Some(wire::Pipeline {
                        id: *id,
                        parts: parts.to_vec(),
                        name: names.get(id).cloned().unwrap_or_default(),
                    })
                })
                .collect();
            out.sort_by_key(|p| p.id);
            out
        };
        roots.extend(pipelines.iter().flat_map(|p| &p.parts).filter_map(|p| {
            if let wire::Part::Root { root } = p {
                Some(*root)
            } else {
                None
            }
        }));
        let signatures: Vec<wire::Signature> = {
            let all = lock(&r.signatures);
            signatures
                .iter()
                .filter_map(|id| all.get(id).cloned())
                .collect()
        };
        roots.extend(signatures.iter().map(|s| s.root));
        roots.remove(&NONE);
        let roots: Vec<(u32, wire::Blob)> = {
            let all = lock(&r.roots);
            roots
                .iter()
                .filter_map(|id| Some((*id, wire::Blob(all.get(id)?.to_vec()))))
                .collect()
        };
        let mut notes = r.problems();
        for ((what, before), (_, now)) in self.calls.iter().zip(not_recorded_calls()) {
            if now > *before {
                let n = now - before;
                notes.push(if n == 1 {
                    what.clone()
                } else {
                    format!("{what} ({n} times)")
                });
            }
        }
        if partial {
            notes.push(format!(
                "the frame is not whole: written after {} of its submissions, before its present (while a frame is captured, this is written after its first submission and every other second)",
                self.submits.len()
            ));
        }
        let (adapter, vendor, device) = gpu::adapter(&self.worker.device);
        let bytes: u64 = self
            .contents
            .iter()
            .filter(|c| c.file != ZERO)
            .map(|c| c.len)
            .sum();
        let views: u64 = self
            .heaps
            .iter()
            .map(|(h, _)| u64::from(h.count))
            .sum();
        notes.push(format!(
            "frame {} ({}); {} submissions, {} lists, {} resources, {} pipelines, {} views in {} heaps",
            self.number,
            match self.world {
                0 => "the world's first frame was not seen".to_owned(),
                w => format!("world frame {}", self.number.saturating_sub(w)),
            },
            self.submits.len(),
            self.lists.len(),
            resources.len(),
            pipelines.len(),
            views,
            self.heaps.len(),
        ));
        notes.push(format!(
            "on disk: {} MB of the card's memory, {} MB of memory both sides see (what the CPU wrote by each submission), {} MB of pictures",
            bytes >> 20,
            self.delta_at.load(Ordering::Relaxed) >> 20,
            self.present_at >> 20,
        ));
        notes.push(format!(
            "the title stood still {:.1} s in the frame's first submission, {:.1} s in all: {}",
            self.first.as_secs_f64(),
            self.took.values().sum::<std::time::Duration>().as_secs_f64(),
            self.took
                .iter()
                .map(|(what, d)| format!("{what} {:.1} s", d.as_secs_f64()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        notes.push(format!(
            "memory the capture took: {} MB of read-back buffers at the most, {} MB of page numbers for {} MB of memory both sides see, {} MB of views' descriptions for the whole run; {} MB were looked at for changes",
            self.stagings >> 20,
            self.buffers.iter().map(|b| b.kept.len() as u64 * 8).sum::<u64>() >> 20,
            self.buffers.iter().map(|b| b.size as u64).sum::<u64>() >> 20,
            (views * size_of::<Slot>() as u64) >> 20,
            self.looked >> 20,
        ));
        if !partial {
            for n in &notes {
                tracing::info!("frame recorder: {n}");
            }
        }
        let frame = wire::Frame {
            adapter,
            vendor,
            device,
            number: self.number,
            world: self.world,
            partial,
            notes: notes.clone(),
            heaps,
            resources,
            roots,
            codes: lock(&r.codes).all.clone(),
            pipelines,
            signatures,
            descriptor_heaps: self
                .heaps
                .iter()
                .map(|(h, runs)| wire::DescriptorHeap {
                    id: h.id,
                    kind: h.kind,
                    count: h.count,
                    flags: h.flags,
                    runs: runs.clone(),
                })
                .collect(),
            query_heaps: {
                let all = lock(&r.queries);
                queries
                    .iter()
                    .filter_map(|id| all.get(id).cloned())
                    .collect()
            },
            lists: self
                .lists
                .iter()
                .map(|l| wire::List {
                    kind: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                    commands: wire::Blob(l.to_vec()),
                })
                .collect(),
            submits: self.submits.clone(),
            contents: self.contents.clone(),
            presents: self.presents.clone(),
        };
        frame.write(&r.folder.join("frame.bin"))?;
        std::fs::write(r.folder.join("notes.txt"), notes.join("\n"))?;
        if !partial {
            tracing::info!(folder = %r.folder.display(), "frame recorder: the frame is captured");
        }
        Ok(())
    }
}

/// A part of a resource in a read-back buffer, `at` bytes in.
struct Piece {
    res: Arc<Res>,
    first: u64,
    count: u32,
    len: u64,
    at: u64,
}

/// Records the copies of `pieces` into `to`. Textures are moved to a state
/// they can be copied from and back.
fn copy_out(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    states: &States,
    pieces: &[Piece],
    to: &ID3D12Resource,
) {
    let (mut before, mut after) = (Vec::new(), Vec::new());
    let mut seen = HashSet::new();
    for p in pieces.iter().filter(|p| !p.res.is_buffer()) {
        if !seen.insert(p.res.id) {
            continue;
        }
        let from = states.get(p.res.id).unwrap_or(vec![COMMON]);
        // From the common state a texture is promoted by the copy.
        let to: Vec<u32> = from
            .iter()
            .map(|&s| {
                if s == COMMON || s & COPY_SOURCE != 0 {
                    s
                } else {
                    COPY_SOURCE
                }
            })
            .collect();
        gpu::moves(&p.res.object(), &from, &to, &mut before);
        gpu::moves(&p.res.object(), &to, &from, &mut after);
    }
    // SAFETY: an open list; live resources.
    unsafe {
        if !before.is_empty() {
            list.ResourceBarrier(&before);
        }
        for p in pieces {
            let object = p.res.object();
            if p.res.is_buffer() {
                list.CopyBufferRegion(to, p.at, &*object, p.first, p.len);
                continue;
            }
            let (all, _) =
                gpu::footprints(device, &gpu::to_desc(&p.res.desc), p.first as u32, p.count);
            for (i, mut f) in all.into_iter().enumerate() {
                f.Offset += p.at;
                list.CopyTextureRegion(
                    &gpu::placed_location(to, f),
                    0,
                    0,
                    0,
                    &gpu::sub_location(&object, p.first as u32 + i as u32),
                    None,
                );
            }
        }
        if !after.is_empty() {
            list.ResourceBarrier(&after);
        }
    }
}

/// Records the copy of a texture's subresource, in `state`, into `to`.
pub(super) fn read_sub(
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    sub: u32,
    state: u32,
    to: &ID3D12Resource,
    f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let there = state == COMMON || state & COPY_SOURCE != 0;
    // SAFETY: an open list; live resources.
    unsafe {
        if !there {
            list.ResourceBarrier(&[gpu::transition(texture, ALL, state, COPY_SOURCE)]);
        }
        list.CopyTextureRegion(
            &gpu::placed_location(to, f),
            0,
            0,
            0,
            &gpu::sub_location(texture, sub),
            None,
        );
        if !there {
            list.ResourceBarrier(&[gpu::transition(texture, ALL, COPY_SOURCE, state)]);
        }
    }
}

/// A slot as it goes into the file: a constant buffer view's address as
/// buffer and offset.
fn resolved(r: &Recorder, mut s: Slot) -> Slot {
    let kind = i32::from_le_bytes([s.desc[4], s.desc[5], s.desc[6], s.desc[7]]);
    if s.kind == slot::SRV
        && s.described != 0
        && kind == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0
    {
        r.problem("views of acceleration structures are null in the capture");
        return Slot {
            kind: slot::SRV,
            ..Slot::EMPTY
        };
    }
    if s.kind == slot::CBV && s.described != 0 {
        match r.va(s.address()) {
            Va::At { res, offset } => {
                s.res = res;
                s.set_address(offset);
            }
            Va::Null {} => {}
            Va::Unknown { .. } => {
                // Its buffer is gone: nothing may read it.
                s = Slot {
                    kind: slot::CBV,
                    ..Slot::EMPTY
                };
            }
        }
    }
    s
}

/// The slots as runs of equal ones, empty ones left out.
fn runs_of(r: &Recorder, all: &[Slot]) -> Vec<(u32, u32, Slot)> {
    let mut runs: Vec<(u32, u32, Slot)> = Vec::new();
    for (i, s) in all.iter().enumerate() {
        if s.kind == slot::EMPTY {
            continue;
        }
        match runs.last_mut() {
            Some((first, n, same)) if *first + *n == i as u32 && same == s => *n += 1,
            _ => runs.push((i as u32, 1, *s)),
        }
    }
    for run in &mut runs {
        run.2 = resolved(r, run.2);
    }
    runs
}

/// Whether all bytes are zero.
fn zero(bytes: &[u8]) -> bool {
    // SAFETY: any bytes are a number.
    let (a, b, c) = unsafe { bytes.align_to::<u128>() };
    a.iter().chain(c).all(|&x| x == 0) && b.iter().all(|&x| x == 0)
}

fn write_at(file: &File, mut bytes: &[u8], mut at: u64) -> std::io::Result<()> {
    while !bytes.is_empty() {
        let n = file.seek_write(bytes, at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        bytes = &bytes[n..];
        at += n as u64;
    }
    Ok(())
}

/// Looks at `len` bytes at `live` page by page: a page whose number is not
/// the one at `kept` is copied to the end of what `taken` holds so far (from
/// its start on; it has room for `len` bytes), and `kept` gets the number of
/// that copy. The runs of pages taken: where each starts in `live`, and its
/// bytes.
///
/// # Safety
/// `live` is `len` readable bytes; `kept` has a number for each of their
/// pages and is the caller's alone.
unsafe fn changed_runs(
    live: *const u8,
    kept: *mut u64,
    len: usize,
    taken: &mut [u8],
) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let (mut at, mut have) = (0, 0);
    while at < len {
        let n = PAGE.min(len - at);
        // SAFETY: the caller's.
        unsafe {
            let page = std::slice::from_raw_parts(live.add(at), n);
            let was = kept.add(at / PAGE);
            if plain::page_hash(page) != *was {
                // What is written down is what the number is of, whatever
                // the page turns into meanwhile.
                let copy = &mut taken[have..have + n];
                copy.copy_from_slice(page);
                *was = plain::page_hash(copy);
                have += n;
                match runs.last_mut() {
                    Some(last) if last.0 + last.1 == at => last.1 += n,
                    _ => runs.push((at, n)),
                }
            }
        }
        at += n;
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_are_found_by_page() {
        let mut live = vec![0u8; 5 * PAGE + 100];
        let mut kept = vec![0u64; 6];
        let mut taken = vec![0u8; live.len()];
        let mut find = |live: &[u8], taken: &mut [u8]| {
            // SAFETY: bytes and a number for each of their pages.
            unsafe { changed_runs(live.as_ptr(), kept.as_mut_ptr(), live.len(), taken) }
        };
        assert!(find(&live, &mut taken).is_empty(), "zeros are not written");
        live[PAGE + 5] = 1;
        live[2 * PAGE] = 2;
        live[5 * PAGE + 99] = 3;
        assert_eq!(
            find(&live, &mut taken),
            [(PAGE, 2 * PAGE), (5 * PAGE, 100)]
        );
        // The pages taken, one after the other.
        assert_eq!(taken[..2 * PAGE], live[PAGE..3 * PAGE]);
        assert_eq!(taken[2 * PAGE..2 * PAGE + 100], live[5 * PAGE..]);
        assert!(find(&live, &mut taken).is_empty());
        // Back to zeros is a change too.
        live[2 * PAGE] = 0;
        assert_eq!(find(&live, &mut taken), [(2 * PAGE, PAGE)]);
        assert!(taken[..PAGE].iter().all(|&b| b == 0));
    }

    #[test]
    fn what_a_submission_reads() {
        let mut reads = Reads::default();
        let part = |from, len| Cmd::CopyBuffer {
            dst: 9,
            dst_at: 0,
            src: 1,
            src_at: from,
            len,
        };
        reads.note(&[
            part(100, 50),
            part(4000, 200),
            part(20_000, 1),
            Cmd::CopyTexture {
                dst: Loc::Sub { res: 9, sub: 0 },
                at: [0; 3],
                src: Loc::Placed {
                    res: 2,
                    offset: 4096,
                    format: 28,
                    size: [16, 16, 1],
                    pitch: 256,
                },
                part: None,
            },
            Cmd::Indirect {
                signature: 1,
                most: 1,
                args: 3,
                at: 8,
                count: 0,
                count_at: 0,
            },
            Cmd::RootView {
                graphics: true,
                kind: slot::CBV,
                param: 0,
                va: Va::At { res: 4, offset: 5 },
            },
        ]);
        assert_eq!(reads.pages(1, 1 << 20, false), [(0, 8192), (16384, 20480)]);
        assert_eq!(
            reads.pages(1, 20_100, false),
            [(0, 8192), (16384, 20_100)],
            "its end"
        );
        assert_eq!(reads.pages(2, 1 << 20, false), [(4096, 8192)]);
        assert_eq!(reads.pages(3, 100, false), [(0, 100)]);
        assert_eq!(reads.pages(4, 100, false), [(0, 100)]);
        assert_eq!(
            reads.pages(5, 100, true),
            [(0, 100)],
            "a descriptor names it"
        );
        assert!(reads.pages(5, 100, false).is_empty() && reads.pages(9, 100, false).is_empty());
    }

    #[test]
    fn zeros() {
        assert!(zero(&[]));
        assert!(zero(&[0; 1001]));
        for i in [0, 7, 500, 1000] {
            let mut b = [0u8; 1001];
            b[i] = 1;
            assert!(!zero(&b));
        }
    }
}
