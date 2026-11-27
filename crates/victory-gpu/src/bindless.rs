//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). Grand Theft Auto VI is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::census_fast::{Fast, Tickets, Unordered};
use crate::census_maps::{Counts, Hashing};
use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
pub(crate) enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
pub(crate) struct Viewed {
    inner: Mutex<Inner>,
    /// mac: with `VICTORY_CENSUS_FAST=1` (`crate::census_fast`): the kept
    /// answer of [`unordered_writes`] and its generation.
    fast: Option<Fast<Shadow>>,
}

impl Default for Viewed {
    fn default() -> Self {
        Viewed::new(crate::mac_switches::census_fast())
    }
}

impl Viewed {
    /// A census, with the fast parts (`fast`) or as before.
    pub(crate) fn new(fast: bool) -> Self {
        Viewed { inner: Mutex::new(Inner::new(fast)), fast: fast.then(Fast::default) }
    }
}

pub(crate) type Range = (usize, u64, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

struct Inner {
    /// These two in the form `VICTORY_QUICK_SYNC` chooses (`crate::census_maps`).
    slots: HashMap<u32, Seen, Hashing>,
    ranges: Counts<Range, (u32, Arc<Shadow>)>,
    textures: BTreeMap<usize, u32>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4).
    hulls: HashMap<(usize, bool), Hull>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: HashMap<(usize, bool), Ends>,
    /// mac: with `VICTORY_CENSUS_FAST=1`: the unordered ranges of `ranges`
    /// alone, changed where `ranges` gains or loses one ([`Inner::count`]).
    unordered: Option<Unordered<Shadow>>,
    /// mac: with `VICTORY_CENSUS_FAST=1`: per slot, the ticket of the making
    /// its entry is from ([`Inner::enter`]).
    tickets: Tickets,
}

impl Inner {
    fn new(fast: bool) -> Self {
        let [quick, _] = crate::mac_switches::quick_sync(crate::shader_cache::under_wine);
        Inner {
            slots: HashMap::with_hasher(Hashing::new(quick)),
            ranges: Counts::new(quick),
            textures: BTreeMap::new(),
            hulls: HashMap::new(),
            ends: HashMap::new(),
            unordered: fast.then(Unordered::new),
            tickets: Tickets::default(),
        }
    }
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    /// One more (`add`) or one fewer slot views what `seen` reads. Whether
    /// the census's unordered ranges changed (mac: told only with
    /// `VICTORY_CENSUS_FAST=1`; false without it).
    fn count(&mut self, seen: &Seen, add: bool) -> bool {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        let mut unordered_changed = false;
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry_or(key, || (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                // mac: an unordered range came or went (`crate::census_fast`).
                if (was == 0) != (now == 0)
                    && key.3
                    && let Some(unordered) = &mut self.unordered
                {
                    if now > 0 {
                        unordered.came(key, shadow);
                    } else {
                        unordered.went(&key);
                    }
                    unordered_changed = true;
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let ends = self
                        .ends
                        .entry((key.0, key.3))
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let hull = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => Some((shadow.clone(), from, to - from, key.3, shadow.dirty_bit())),
                        _ => None,
                    };
                    match hull {
                        Some(hull) => {
                            self.hulls.insert((key.0, key.3), hull);
                        }
                        None => {
                            self.ends.remove(&(key.0, key.3));
                            self.hulls.remove(&(key.0, key.3));
                        }
                    }
                }
            }
            Seen::Texture(r) => {
                let n = self.textures.entry(*r).or_insert(0);
                step(n);
                if *n == 0 {
                    self.textures.remove(r);
                }
            }
            Seen::Left(_) => {}
        }
        unordered_changed
    }

    /// mac (`VICTORY_CENSUS_FAST=1`): slot `index` holds `seen`, as of the
    /// making that drew `ticket`, unless a later making of the slot is in
    /// already (`crate::census_fast::Tickets`).
    fn enter(&mut self, fast: &Fast<Shadow>, index: u32, ticket: u64, seen: Option<Seen>) {
        if !self.tickets.newer(index, ticket) {
            return;
        }
        let mut changed = false;
        if let Some(old) = self.slots.remove(&index) {
            changed |= self.count(&old, false);
        }
        if let Some(seen) = seen {
            changed |= self.count(&seen, true);
            self.slots.insert(index, seen);
        }
        // Told before the census lock goes.
        if changed {
            fast.changed();
        }
    }
}

/// mac: the census updates of a heap sync pass made the quick way
/// (`crate::descriptor::host`, `sync_quick`), entered under one hold of
/// the census lock [`crate::census_fast::BATCH`] at a time instead of one
/// lock a slot (`VICTORY_CENSUS_FAST=1`); without the switch each goes in
/// at once, as before. Each update carries the ticket of its slot's making,
/// so a slot made again meanwhile by another thread (entered at once, with
/// a later ticket) keeps that making's entry. Entered at the latest when
/// the batch goes: the pass's own thread reads the census after it (the
/// submission's refreshes) only once the sync has returned.
pub(crate) struct Batch<'a> {
    viewed: &'a Viewed,
    updates: Vec<(u32, u64, Option<Seen>)>,
}

impl Batch<'_> {
    /// Where the pass makes slot `index` (under its lock): what
    /// [`Viewed::set_seen`] enters, now (no switch) or with the batch.
    pub(crate) fn set_seen(&mut self, index: u32, seen: Option<Seen>) {
        match &self.viewed.fast {
            Some(fast) => self.updates.push((index, fast.ticket(), seen)),
            None => self.viewed.set_seen(index, seen),
        }
    }

    /// With no slot lock held: the updates so far, if there are a batch's.
    pub(crate) fn enter_if_full(&mut self) {
        if self.updates.len() >= crate::census_fast::BATCH {
            self.enter();
        }
    }

    /// With no slot lock held: the updates so far, under one hold of the
    /// census lock (waits counted with `VICTORY_HEAP_LOCKS`, as
    /// [`Viewed::set_seen`]'s).
    pub(crate) fn enter(&mut self) {
        let Some(fast) = &self.viewed.fast else {
            return;
        };
        if self.updates.is_empty() {
            return;
        }
        let mut inner = crate::descriptor::slots::locked(&self.viewed.inner, crate::descriptor::slots::CENSUS);
        for (index, ticket, seen) in self.updates.drain(..) {
            inner.enter(fast, index, ticket, seen);
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.enter();
    }
}

impl Viewed {
    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        let seen = view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) });
        self.set_seen(index, seen);
    }

    /// [`Viewed::set`] with what the slot's view reads already worked out
    /// (`crate::descriptor::host`, "quick").
    pub(crate) fn set_seen(&self, index: u32, seen: Option<Seen>) {
        // mac: with `VICTORY_CENSUS_FAST=1` in the order of the slot's
        // makings, by a ticket drawn here (callers hold the slot's lock), as
        // a heap sync's batch enters them ([`Batch`]).
        if let Some(fast) = &self.fast {
            let ticket = fast.ticket();
            let mut inner = crate::descriptor::slots::locked(&self.inner, crate::descriptor::slots::CENSUS);
            inner.enter(fast, index, ticket, seen);
            return;
        }
        // mac: a wait for it is counted with `VICTORY_HEAP_LOCKS`.
        let mut inner = crate::descriptor::slots::locked(&self.inner, crate::descriptor::slots::CENSUS);
        if let Some(old) = inner.slots.remove(&index) {
            inner.count(&old, false);
        }
        if let Some(seen) = seen {
            inner.count(&seen, true);
            inner.slots.insert(index, seen);
        }
    }

    /// mac: census updates entered together (`VICTORY_CENSUS_FAST=1`).
    pub(crate) fn batch(&self) -> Batch<'_> {
        Batch { viewed: self, updates: Vec::new() }
    }

    /// The range slot `index` views of a buffer copy (the copy, the title
    /// address, the bytes, unordered access), if it views one (tests).
    #[cfg(test)]
    pub(crate) fn range_at(&self, index: u32) -> Option<Range> {
        match self.inner.lock().unwrap_or_else(|e| e.into_inner()).slots.get(&index) {
            Some(Seen::Buffer(shadow, address, len, uav)) => Some((Arc::as_ptr(shadow) as usize, *address, *len, *uav)),
            _ => None,
        }
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).hulls.values().cloned().collect()
    }
}

impl Seen {
    /// What [`seen`] finds for a made shader view of `len` bytes (not none)
    /// at title address `address`, whose host copy is `shadow`.
    pub(crate) fn shader_buffer(shadow: Arc<Shadow>, address: u64, len: u64) -> Self {
        let len = len.min(shadow.left(address));
        Seen::Buffer(shadow, address, len, false)
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, _) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        return texture.title_written().then_some(Seen::Texture(*resource));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
) -> Option<Uploads> {
    static LIST: Mutex<Holder> = Mutex::new(Holder(None));
    let mut guard = LIST.lock().unwrap_or_else(|e| e.into_inner());
    crate::texture::recycle();
    match &guard.0 {
        // Closed by the last submission, which has run.
        Some(list) => list.reset(),
        // Made open.
        None => {
            let list = crate::host::host()?.new_list().ok()?;
            list.set_ours();
            guard.0 = Some(list);
        }
    }
    let list = guard.0.as_ref()?;
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Collected under the heaps'
    // locks and brought up without them, several at a time: reading title
    // memory, detiling and making staging buffers took half the submitting
    // thread in a new shot, and title threads setting views waited on the
    // lock meanwhile (run evict5).
    let mut work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)> = Vec::new();
    let mut staged = std::collections::HashSet::new();
    for (staging, copy) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded).
        if copy.host_written.load(Relaxed) {
            continue;
        }
        if staged.insert(Arc::as_ptr(staging)) {
            work.push((staging.clone(), copy.clone()));
        }
    }
    let mut done = std::collections::HashSet::new();
    for viewed in heaps {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        for &r in inner.textures.keys() {
            if !done.insert(r) {
                continue;
            }
            // Not made (given up, or never read): nothing to bring up.
            let Some(texture) = crate::resource::made_copy(r as This) else {
                continue;
            };
            // Only what shaders read lately (the eviction pass's stamps):
            // the heaps view thousands of textures a frame never reads, and
            // bringing each up whenever its pages change re-uploaded about
            // 4 GB in 30-46 world frames (runs evict2, noevict1).
            // shortcut: a texture read again after a gap is brought up after
            // the submission that read it, which can see bytes one refresh
            // old (a slot the title rewrites is made fresh anyway); upload on
            // the mark itself if that shows.
            if !texture.used_recently() {
                continue;
            }
            // Its contents are the host's since a host copy wrote it.
            if texture.host_written.load(Relaxed) {
                continue;
            }
            if let Some(staging) = crate::texture::staging(r as This, &texture.texture)
                && staged.insert(Arc::as_ptr(&staging))
            {
                work.push((staging, texture));
            }
        }
    }
    let mut recorded = false;
    // The textures copied into stay alive in `work` until recorded.
    let mut readable = Vec::new();
    for ((staging, copy), upload) in work.iter().zip(crate::texture::refresh_all(&work)) {
        if let Some(upload) = upload {
            readable.extend(crate::texture::record_upload(list, copy, staging, &upload));
            recorded = true;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    if !readable.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&readable) };
    }
    list.close();
    // A list recording nothing is not run (its reset next time is fine).
    recorded.then_some(Uploads(guard))
}

/// The upload list, used under its lock only.
pub(crate) struct Holder(Option<HostList>);
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, under the lock.
unsafe impl Send for Holder {}

/// The upload list of a submission, ours until it has run.
pub(crate) struct Uploads(std::sync::MutexGuard<'static, Holder>);

impl Uploads {
    pub(crate) fn list(&self) -> &HostList {
        self.0.0.as_ref().expect("made before it is handed out")
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, address, len, uav, bit) in inner.hulls.values() {
        if !crate::watch::clean_bit(*bit) {
            let (address, len) = if *uav && shadow.marked() {
                shadow.page_range(*address, *len)
            } else {
                (*address, *len)
            };
            add(shadow, address as usize, len as usize);
        }
    }
}

/// The copies the heap views for unordered access: what shaders that take
/// views from the heap itself may write with no list declaring it
/// (`crate::host::unwaited`, for a submission still running).
pub(crate) fn written(viewed: &Viewed, mut each: impl FnMut(&Arc<Shadow>)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, _, _, uav, _) in inner.hulls.values() {
        if *uav {
            each(shadow);
        }
    }
}

/// The buffer range (title address, bytes) of the view currently in
/// `index`, changed or not (diagnostics: what a table draw reads there).
#[cfg(feature = "diagnostics")]
pub(crate) fn slot_range(viewed: &Viewed, index: u32) -> Option<(usize, usize)> {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    match inner.slots.get(&index) {
        Some(Seen::Buffer(_, address, len, _)) => Some((*address as usize, *len as usize)),
        _ => None,
    }
}

/// Per submission: the buffer range of the view currently in `index`. A
/// table draw may have been recorded with another view in that slot.
pub(crate) fn refresh_slot(
    viewed: &Viewed,
    index: u32,
    mut add: impl FnMut(&Arc<Shadow>, usize, usize),
) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Seen::Buffer(shadow, address, len, _)) = inner.slots.get(&index) else {
        return;
    };
    if !crate::watch::clean_bit(shadow.dirty_bit()) {
        add(shadow, *address as usize, *len as usize);
    }
}

/// Heap UAV ranges that may need to refresh a widened SRV after a later
/// buffer barrier. Computed heap indices do not name ordinary write steps.
pub(crate) fn widened_writes(viewed: &Viewed) -> Vec<(Arc<Shadow>, usize, usize)> {
    if !crate::under_wine() {
        return Vec::new();
    }
    viewed
        .inner
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .ranges
        .iter()
        .filter(|((_, address, end, uav), _)| *uav && address < end)
        .map(|((_, address, end, _), (_, shadow))| {
            (shadow.clone(), *address as usize, (end - address) as usize)
        })
        .collect()
}

/// mac: with `VICTORY_CENSUS_FAST=1`, [`widened_writes`] without walking
/// every range under the census lock (`crate::census_fast`): the answer
/// kept since the heap's unordered ranges last changed, with no census lock
/// at all, or one made from those ranges alone. The same ranges (in another
/// order where the census is in hash maps, whose order is none anyway).
/// None without the switch.
pub(crate) fn unordered_writes(viewed: &Viewed) -> Option<Arc<[(Arc<Shadow>, usize, usize)]>> {
    let fast = viewed.fast.as_ref()?;
    if !crate::under_wine() {
        return Some(Arc::new([]));
    }
    if let Some(kept) = fast.kept() {
        return Some(kept);
    }
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    inner.unordered.as_ref().map(|unordered| fast.answer(unordered))
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
        if !HEAP_WRITTEN.swap(true, Relaxed) {
            REMAKE.store(true, Relaxed);
        }
    }
}

/// A shader wrote through a heap's views (ever): textures viewed for
/// unordered access there may hold what the GPU wrote, which no state move
/// shows (`crate::resource::evictable`).
static HEAP_WRITTEN: AtomicBool = AtomicBool::new(false);
/// The unordered views left null are to be made again before the next
/// submission ([`take_remake`]).
static REMAKE: AtomicBool = AtomicBool::new(false);

/// Whether any shader wrote through a heap yet.
pub(crate) fn heap_written() -> bool {
    HEAP_WRITTEN.load(Relaxed)
}

/// Whether the null unordered views are to be made again now (once).
pub(crate) fn take_remake() -> bool {
    REMAKE.swap(false, Relaxed)
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.textures.remove(&resource).is_some() {
            inner.slots.retain(|_, seen| !matches!(seen, Seen::Texture(r) if *r == resource));
        }
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = HashMap::new();
        for seen in inner.slots.values() {
            match seen {
                Seen::Buffer(..) => buffers += 1,
                Seen::Texture(_) => textures += 1,
                Seen::Left((k, address, bytes)) => {
                    left[*k] += 1;
                    if *k == 2 {
                        let b = placed.entry(*address).or_insert(0u64);
                        *b = (*b).max(*bytes);
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let bytes: u64 = inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum();
        drop(inner);
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let copies = {
            let mut c: Vec<usize> = inner.ranges.keys().map(|k| k.0).collect();
            // The ranges come in order only from the B-tree form.
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame; slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            inner.ranges.len(),
            bytes as f64 / 1e6,
            inner.textures.len(),
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        use crate::compute::{Param, srv, uav};
        let params = &[Param::Table(&[srv(0, 1), uav(0, 1), srv(1, 1)])];
        crate::compute::pipeline(SOURCE, s!("heap"), 0, params).unwrap()
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            let mut inner = viewed.inner.lock().unwrap();
            if let Some(old) = inner.slots.remove(&slot) {
                inner.count(&old, false);
            }
            if let Some((from, len)) = range {
                let seen = Seen::Buffer(copy.clone(), at + from, len, false);
                inner.count(&seen, true);
                inner.slots.insert(slot, seen);
            }
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// Three host copies of test memory, registered (none: no host GPU
    /// here), and what unregisters them.
    fn three_copies(memory: &[u8]) -> (Option<Vec<Arc<Shadow>>>, impl Fn() + use<>) {
        let at = memory.as_ptr() as u64;
        for k in 0..3u64 {
            crate::shadow::register((at + 0x1000 * k) as usize, at + 0x1000 * k, 0x1000);
        }
        let release = move || {
            for k in 0..3u64 {
                crate::shadow::unregister((at + 0x1000 * k) as usize, at + 0x1000 * k);
            }
        };
        let copies: Option<Vec<_>> = (0..3u64).map(|k| crate::shadow::find(at + 0x1000 * k, 0x1000)).collect();
        (copies, release)
    }

    /// A census's answers as plain numbers, sorted: the ranges unordered
    /// views write, the hulls, each of `slots` slots' range.
    type Plain = (Vec<(usize, usize, usize)>, Vec<(usize, u64, u64, bool)>, Vec<Option<Range>>);

    fn plain(writes: &[(Arc<Shadow>, usize, usize)], viewed: &Viewed, slots: u32) -> Plain {
        let mut w: Vec<_> = writes.iter().map(|(s, a, l)| (Arc::as_ptr(s) as usize, *a, *l)).collect();
        w.sort_unstable();
        let mut h: Vec<_> = viewed.ranges().iter().map(|h| (Arc::as_ptr(&h.0) as usize, h.1, h.2, h.3)).collect();
        h.sort_unstable();
        (w, h, (0..slots).map(|s| viewed.range_at(s)).collect())
    }

    /// mac: a census with `VICTORY_CENSUS_FAST`'s parts answers as one
    /// without, after slots rewritten at random with views of three copies
    /// (shader and unordered views, the same range again, empty ranges,
    /// none), half of them entered at once and half in a batch entered
    /// later (as a heap sync pass's): the ranges unordered views write (kept
    /// or made anew), the hulls, each slot's range.
    #[test]
    fn the_fast_census_answers_as_the_old_one() {
        let memory = vec![0u8; 0x3000];
        let at = memory.as_ptr() as u64;
        let (Some(copies), release) = three_copies(&memory) else {
            return; // no host GPU here
        };
        let (old, fast) = (Viewed::new(false), Viewed::new(true));
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut kept = 0;
        let mut batch = fast.batch();
        for step in 0..20_000u32 {
            let slot = (next() % 200) as u32;
            let seen = (step % 7 != 0).then(|| {
                let k = next() % 3;
                let from = at + 0x1000 * k + (next() % 64) * 16;
                let len = if next() % 40 == 0 { 0 } else { 16 + next() % 256 };
                Seen::Buffer(copies[k as usize].clone(), from, len, next() % 3 == 0)
            });
            old.set_seen(slot, seen.clone());
            if next() % 2 == 0 {
                batch.set_seen(slot, seen);
            } else {
                fast.set_seen(slot, seen);
            }
            if next() % 8 == 0 {
                batch.enter_if_full();
                batch.enter();
            }
            if step % 3 == 0 {
                batch.enter();
                kept += usize::from(fast.fast.as_ref().expect("the fast parts").kept().is_some());
                let answer = unordered_writes(&fast).expect("the fast parts");
                let slots = if step % 101 == 0 { 200 } else { 0 };
                assert_eq!(plain(&answer, &fast, slots), plain(&widened_writes(&old), &old, slots), "step {step}");
            }
        }
        drop(batch);
        assert!(unordered_writes(&old).is_none(), "nothing without the switch");
        assert!(!crate::under_wine() || kept > 100, "kept answers are tested: {kept}");
        drop((old, fast));
        release();
    }

    /// mac: a pass's batch entered after a later making of one of its slots
    /// (another thread's, entered at once) leaves that slot as the later
    /// making left it; an update of a batch made after one entered at once
    /// wins, and a batch is entered when it goes. Without the switch a
    /// batch enters each update at once.
    #[test]
    fn a_batch_entered_late_leaves_a_later_making_in() {
        let memory = vec![0u8; 0x3000];
        let at = memory.as_ptr() as u64;
        let (Some(copies), release) = three_copies(&memory) else {
            return; // no host GPU here
        };
        let view = |k: usize, uav: bool| Some(Seen::Buffer(copies[k].clone(), at + 0x1000 * k as u64, 0x100, uav));
        let range = |k: usize, uav: bool| Some((Arc::as_ptr(&copies[k]) as usize, at + 0x1000 * k as u64, 0x100, uav));
        let (old, fast) = (Viewed::new(false), Viewed::new(true));
        for viewed in [&old, &fast] {
            viewed.set_seen(0, view(0, true));
        }
        // A pass makes slots 0 and 1, then another thread makes slot 0 again
        // before the pass's batch is entered.
        let mut batch = fast.batch();
        batch.set_seen(0, view(1, false));
        batch.set_seen(1, view(1, true));
        assert_eq!(fast.range_at(0), range(0, true), "not entered yet");
        fast.set_seen(0, view(2, true));
        batch.enter();
        drop(batch);
        for (slot, seen) in [(0, view(1, false)), (1, view(1, true)), (0, view(2, true))] {
            old.set_seen(slot, seen);
        }
        assert_eq!(fast.range_at(0), range(2, true), "the later making stays");
        assert_eq!(fast.range_at(1), range(1, true), "the pass's other slot is in");
        let answer = |v: &Viewed| match unordered_writes(v) {
            Some(writes) => plain(&writes, v, 2),
            None => plain(&widened_writes(v), v, 2),
        };
        assert_eq!(answer(&fast), answer(&old), "the counts follow the entries");
        // A batch's update made after an update entered at once wins, when
        // the batch goes.
        fast.set_seen(1, view(0, false));
        let mut batch = fast.batch();
        batch.set_seen(1, None);
        batch.set_seen(0, None);
        assert_eq!(fast.range_at(1), range(0, false));
        drop(batch);
        assert_eq!((fast.range_at(0), fast.range_at(1)), (None, None));
        assert!(answer(&fast).0.is_empty() && answer(&fast).1.is_empty());
        // Without the switch: at once.
        let mut batch = old.batch();
        batch.set_seen(0, None);
        assert_eq!(old.range_at(0), None, "entered at once");
        drop(batch);
        drop((old, fast));
        release();
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }
}
