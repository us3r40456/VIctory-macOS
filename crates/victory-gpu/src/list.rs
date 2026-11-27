//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Upload>>,
    /// The host lists the calls record into, in order: one, and one more
    /// for every wait the title records after commands ([`wait`]). Shared
    /// with the queues that hold them until their waits hold
    /// (`crate::hold`), so they outlive a list the title releases first.
    #[cfg(windows)]
    hosts: std::sync::Mutex<Vec<std::sync::Arc<crate::host::HostList>>>,
    /// Which of them the calls record into.
    #[cfg(windows)]
    segment: std::sync::atomic::AtomicUsize,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
pub(crate) fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list a title list's calls record into (its current segment),
/// if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let current: *const crate::host::HostList = std::sync::Arc::as_ptr(hosts.get(k)?);
    // SAFETY: the list holds this host list until it is reset or released
    // (an `Arc` does not move it), and the title records into a list from
    // one thread at a time.
    Some(unsafe { &*current })
}

/// A title list's host lists, in order: what a submission of it runs.
#[cfg(windows)]
pub(crate) fn segments(this: This) -> Vec<std::sync::Arc<crate::host::HostList>> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    hosts.iter().take(k + 1).cloned().collect()
}

/// Waits recorded after commands: the title list goes on in a new host
/// list, which runs once the wait holds.
static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lists reset or released while a queue still held some of their host
/// lists: D3D12 lets a list be reset as soon as it is submitted (Grand
/// Theft Auto VI does, 0-21 times a world frame).
static EARLY_RESETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Cuts and early resets since the last call (the frame line).
pub(crate) fn take_cuts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CUTS.swap(0, Relaxed), EARLY_RESETS.swap(0, Relaxed))
}

/// Title lists cut at a copy that changes what their earlier commands read
/// (`VICTORY_COPY_ORDER=split`, [`copy_into`]).
static COPY_CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Those cuts since the last call (the frame line).
pub(crate) fn take_copy_cuts() -> u64 {
    COPY_CUTS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Cuts for hoisted clamps since the last call (the frame line).
pub(crate) fn take_hoists() -> u64 {
    HOISTS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Runs of indirect draws whose clamps went to the host list before
/// ([`hoist`]): each cut the title list once more.
static HOISTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Of those cuts since the last call, the ones made at an indirect
/// dispatch (the frame line).
pub(crate) fn take_dispatch_hoists() -> u64 {
    DISPATCH_HOISTS.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static DISPATCH_HOISTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Before an indirect draw (`crate::indirect::hoisting`): unless the clamps
/// of the list's draws already go to the host list before it and nothing
/// since may have written arguments, the title list goes on in a new host
/// list here, as at a wait. The one it leaves stays open for the clamps of
/// the draws to come, so that no dispatch of ours comes between the draws;
/// it ends where the run does ([`end_hoist`]).
///
/// mac: the same before an indirect dispatch with
/// `crate::indirect::hoisting_dispatches`, unless the clamps already go to
/// the host list before and no barrier, sync, copy, clear or immediate
/// write was recorded since (`State::hoists_dispatches_into`).
#[cfg(windows)]
fn hoist(this: This, kind: u32) {
    if !crate::indirect::hoisting() {
        return;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let mut state = list.state.lock().unwrap_or_else(|e| e.into_inner());
    let dispatch = kind == crate::draw::INDIRECT_DISPATCH;
    let hoistable = if dispatch {
        crate::indirect::hoisting_dispatches() && state.dispatch_hoistable()
    } else {
        state.hoistable(kind)
    };
    if hoistable {
        start_run(this, &mut state, dispatch);
    }
}

/// [`hoist`] for an indirect draw, or a `dispatch`, whose clamp can be
/// hoisted, with the list's state.
#[cfg(windows)]
fn start_run(this: This, state: &mut crate::draw::State, dispatch: bool) {
    let Some(current) = host_list(this) else { return };
    let joins = if dispatch {
        state.hoists_dispatches_into(current)
    } else {
        state.hoists_into(current)
    };
    if joins {
        return;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    state.end_hoist(current);
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let Some(clamps) = list.hosts.lock().unwrap_or_else(|e| e.into_inner()).get(k).cloned() else {
        return;
    };
    let Some(next) = next_segment(this) else { return };
    HOISTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if dispatch {
        DISPATCH_HOISTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    state.cut(next);
    // One budget on both sides of this cut, as without it.
    next.continue_budget(&clamps);
    state.hoist(clamps, next);
}

/// Ends the run whose clamps go to the host list before the current one,
/// if any: that list closes. Before anything that ends the current host
/// list, or counts who holds the list's host lists.
#[cfg(windows)]
fn end_hoist(this: This) {
    with_state(this, |list, s| s.end_hoist(list));
}

/// The host list a title copy into `len` bytes at `dst` is recorded into,
/// noted there as one of its copies.
///
/// With `VICTORY_COPY_ORDER=split`, when a command the title recorded
/// earlier in the current host list reads some of those bytes
/// (`HostList::read_before`), the title list goes on in a new host list,
/// as at a wait after commands ([`wait`]), and that one is marked: the
/// queue runs it in a host submission after the one the first part is in
/// (`crate::host::copy_order_parts`), so the first part's commands have
/// run on the GPU with the old bytes when the copy is made. On the console
/// a list's copies and commands run in their order; here a submission's
/// copies are all made before its first GPU command.
#[cfg(windows)]
fn copy_into<'a>(this: This, dst: *const u8, len: u64) -> Option<&'a crate::host::HostList> {
    let current = host_list(this)?;
    let (at, len) = (dst as usize, usize::try_from(len).unwrap_or(usize::MAX));
    // (A run of hoisted clamps ends before the list it draws in does.)
    let cut = current.read_before(at, len);
    if cut {
        end_hoist(this);
    }
    let into = match cut.then(|| next_segment(this)).flatten() {
        Some(next) => {
            COPY_CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            current.close();
            // SAFETY: the title passes one of our lists.
            let list = unsafe { &*this.cast::<List>() };
            list.state.lock().unwrap_or_else(|e| e.into_inner()).cut(next);
            next.starts_after_copy();
            next
        }
        None => current,
    };
    into.title_copies(at, len);
    Some(into)
}

/// `Wait32BitValueX(address, comparison, reference, mask, flags)` and
/// `Wait64BitValueX(address, comparison, reference, flags)`: the GPU waits
/// until the value compares true. Recorded at the start of the current host
/// list, or, after commands, at the start of a new one the rest of the
/// title list records into (the queue runs each once its waits hold,
/// `crate::hold`); bottom-of-pipe writes recorded before it land when the
/// part before it has run.
extern "system" fn wait32(this: This, address: u64, comparison: u32, reference: u32, mask: u32, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 4, comparison, reference: u64::from(reference), mask: u64::from(mask) });
}
pub(crate) extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 8, comparison, reference, mask: u64::MAX });
}

fn wait(this: This, wait: crate::hold::ValueWait) {
    #[cfg(windows)]
    {
        let Some(current) = host_list(this) else { return };
        // Clamps hoisted from here on would run before the wait holds.
        end_hoist(this);
        if !current.is_empty() {
            let Some(next) = next_segment(this) else { return };
            CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            current.close();
            // SAFETY: the title passes one of our lists.
            let list = unsafe { &*this.cast::<List>() };
            list.state.lock().unwrap_or_else(|e| e.into_inner()).cut(next);
            next.defer_wait(wait);
            return;
        }
        // (What follows a wait has a budget of its own, as in a new list.)
        current.own_budget();
        current.defer_wait(wait);
    }
    #[cfg(not(windows))]
    let _ = (this, wait);
}

/// Moves a title list's recording on to its next host list (made, or one
/// it held since an earlier recording, reset then): that host list.
#[cfg(windows)]
fn next_segment<'a>(this: This) -> Option<&'a crate::host::HostList> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    if hosts.len() <= k {
        let made = crate::host::host()?.new_list().ok()?;
        hosts.push(std::sync::Arc::new(made));
    }
    list.segment.store(k, std::sync::atomic::Ordering::Relaxed);
    let next: *const crate::host::HostList = std::sync::Arc::as_ptr(&hosts[k]);
    drop(hosts);
    // SAFETY: as in `host_list`.
    Some(unsafe { &*next })
}

/// After the title list's last submission: its host lists start again,
/// and a host list a queue still holds (`crate::hold`) is left to the
/// queue and replaced. The last one left so, which should keep what the
/// title list owned.
#[cfg(windows)]
fn restart(this: This) -> Option<std::sync::Arc<crate::host::HostList>> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let used = list.segment.swap(0, std::sync::atomic::Ordering::Relaxed);
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let mut left = None;
    for (k, h) in hosts.iter_mut().enumerate() {
        if std::sync::Arc::strong_count(h) > 1 {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(segment = k, "a list reset while a queue still holds part of it: that part keeps its own host list");
            let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            match spare.map(Ok).or_else(|| crate::host::host().map(|host| host.new_list())) {
                Some(Ok(fresh)) => left = Some(std::mem::replace(h, std::sync::Arc::new(fresh))),
                _ => continue,
            }
        } else if k <= used {
            h.reset();
        }
    }
    left
}

/// Host lists [`restart`] left to a queue, back once they ran: the next
/// ones it hands out (a host list and its allocators are not free to make;
/// Grand Theft Auto VI resets up to 21 lists a frame while they are held).
#[cfg(windows)]
static SPARE: std::sync::Mutex<Vec<crate::host::HostList>> = std::sync::Mutex::new(Vec::new());

/// A host list a queue has run: one no title list holds any more is reset
/// and kept for [`restart`].
#[cfg(windows)]
pub(crate) fn recycle(list: std::sync::Arc<crate::host::HostList>) {
    const KEPT: usize = 64;
    if let Ok(list) = std::sync::Arc::try_unwrap(list) {
        list.reset();
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < KEPT {
            spare.push(list);
        }
    }
}

/// What a title list owned that the host list a queue still holds, if any,
/// may read: its upload memory (list 0x4a0) and its draw state's dynamic
/// descriptor slots and title-memory page table. Freed once that host list
/// (the list's last held one, so the last to run) is gone.
#[cfg(windows)]
fn leave(to: Option<std::sync::Arc<crate::host::HostList>>, uploads: Vec<Upload>, state: crate::draw::State) {
    struct Kept(#[allow(dead_code)] Vec<Upload>, #[allow(dead_code)] crate::draw::State);
    // SAFETY: only dropped, with the host list, after it has run.
    unsafe impl Send for Kept {}
    if let Some(held) = to {
        let kept = Kept(uploads, state);
        held.defer_after(move || {
            let _ = &kept;
        });
    }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        hosts: std::sync::Mutex::new(
            crate::host::host()
                .and_then(|h| match h.new_list() {
                    Ok(list) => Some(std::sync::Arc::new(list)),
                    Err(e) => {
                        tracing::warn!(error = %e, "no host command list");
                        None
                    }
                })
                .into_iter()
                .collect(),
        ),
        #[cfg(windows)]
        segment: Default::default(),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
    });
    let hr = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if hr == S_OK {
        crate::pipeline::bound(pipeline);
    }
    hr
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    // A queue may still hold some of its host lists; they are its own, and
    // the last of them takes what the list owned.
    #[cfg(windows)]
    if !this.is_null() && object::refs(this) == 1 {
        end_hoist(this);
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        let held = list.hosts.lock().unwrap_or_else(|e| e.into_inner()).iter().rfind(|h| std::sync::Arc::strong_count(h) > 1).cloned();
        if held.is_some() {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!("a list released while a queue still holds part of it: that part runs on its own");
            let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
            leave(held, uploads, state);
        }
    }
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        end_hoist(this);
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A queue may still hold what the list recorded (`restart`, `leave`).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let uploads = std::mem::take(&mut *uploads.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(windows)]
    {
        end_hoist(this);
        let held = restart(this);
        // SAFETY: the title passes one of our lists.
        let mut state = unsafe { &(*this.cast::<List>()).state }.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::mem::take(&mut *state);
        state.reset(pipeline);
        drop(state);
        leave(held, uploads, old);
    }
    #[cfg(not(windows))]
    drop(uploads);
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(&texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// A transition barrier between two read-only states (diagnostics: such a
/// barrier on a buffer orders nothing, but still becomes a global UAV
/// barrier). A transition has the subresource and the state before packed
/// in `second` (low, high 32 bits) and the state after in `last`. States
/// are the PC bits (the console's compression hints are higher); common
/// (none) is not counted, as it may stand for a write the title promoted
/// from.
#[cfg(any(all(windows, feature = "diagnostics"), test))]
fn read_only_transition(b: &Barrier) -> bool {
    // Vertex and constant buffer, index buffer, depth read, non-pixel and
    // pixel shader resource, indirect argument, copy source, resolve
    // source, ray tracing acceleration structure read, shading rate source.
    const READ_ONLY: u32 =
        0x1 | 0x2 | 0x20 | 0x40 | 0x80 | 0x200 | 0x800 | 0x2000 | 0x40_0000 | 0x100_0000;
    const PC: u32 = 0x01ff_ffff;
    let read_only = |state: u32| {
        let state = state & PC;
        state != 0 && state & !READ_ONLY == 0
    };
    b.kind == 0 && read_only((b.second >> 32) as u32) && read_only(b.last as u32)
}

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // A run of hoisted clamps of indirect dispatches ends here.
        list.count_order();
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("barrier kinds {:?}", barriers.iter().map(|b| b.kind).collect::<Vec<_>>()));
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        let on_buffer = |b: &Barrier| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        };
        let buffer_barrier = barriers.iter().any(on_buffer);
        let global_uav = barriers.iter().any(|b| b.kind == 2 && b.first.is_null());
        if buffer_barrier {
            host.push(UAV_BARRIER);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let (buffers, read_only) = barriers
                    .iter()
                    .filter(|b| on_buffer(b))
                    .fold((0, 0), |(n, r), b| {
                        (n + 1, r + u64::from(read_only_transition(b)))
                    });
                add(C::BufferBarriers, buffers);
                add(C::BufferReadOnly, read_only);
                add(C::BufferUav, 1);
                add(C::BufferUavSkippable, u64::from(buffers == read_only));
            }
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
        if buffer_barrier || global_uav {
            list.sync_widened();
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(&texture, state);
        let barrier = crate::host::transition(&texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if host_list(this).is_some()
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
        && let Some(list) = copy_into(this, to, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(&from) {
                crate::texture::upload(list, src, &from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_writes(list, dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(&to, &from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            let list = copy_into(this, to, a.min(b)).unwrap_or(list);
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission; texture to buffer through readback memory
/// ([`read_back`]).
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            if dst.kind == FOOTPRINT && src.kind != FOOTPRINT {
                read_back(list, &dst, [x, y, z], &src, source_box);
            }
            return;
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let Some(bytes) = src.footprint.bytes() else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, at, staging)) = list.upload(bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = at;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() } + at, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(&texture) {
                crate::texture::upload(list, src.resource, &texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_writes(list, dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

#[cfg(test)]
impl CopyLocation {
    /// A placed footprint in `resource` (tests).
    pub(crate) fn placed(resource: This, footprint: crate::resource::PlacedFootprint) -> Self {
        Self { resource, kind: 1, footprint }
    }
    /// Subresource 0 of `resource` (tests).
    pub(crate) fn whole(resource: This) -> Self {
        Self { resource, kind: 0, footprint: crate::resource::PlacedFootprint::new(0, 0, 0, 0, 0) }
    }
}

/// Texture read backs since the last frame-rate line: how many, bytes
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static READ_BACKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Buffer footprint to texture copies since the last frame-rate line: how
/// many, bytes, and how many of them were larger than a scratch chunk (a
/// buffer of their own; the rest take the list's scratch) (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static FOOTPRINT_UPLOADS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// (texture, buffer) pairs read back so far, each logged once (diagnostics).
#[cfg(feature = "diagnostics")]
static READ_BACK_PAIRS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Texture to buffer (a read back): the region is copied on the host into
/// readback memory laid out as the title's footprint, and once the list has
/// run its rows go into the title's buffer (the title reads them on the CPU
/// after its fence).
#[cfg(windows)]
pub(crate) fn read_back(list: &crate::host::HostList, dst: &CopyLocation, at: [u32; 3], src: &CopyLocation, source_box: *const [u32; 6]) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_RESOURCE_DIMENSION_TEXTURE3D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    };
    use windows::core::Interface;
    macro_rules! refused {
        ($why:literal) => {{
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(concat!("texture to buffer copy refused: ", $why)));
            return;
        }};
    }
    let (Some(host), Some(texture)) = (crate::host::host(), crate::resource::host_resource(src.resource)) else {
        refused!("no host texture");
    };
    // SAFETY: the title's box (same layout as the host's) or null; a live texture.
    let (b, desc) = unsafe { (source_box.cast::<D3D12_BOX>().as_ref(), texture.GetDesc()) };
    let size = match b {
        Some(b) => [b.right.saturating_sub(b.left), b.bottom.saturating_sub(b.top), b.back.saturating_sub(b.front)],
        None => {
            // A subresource index shares the footprint's first dword.
            let mip = src.footprint.offset as u32 % u32::from(desc.MipLevels.max(1));
            let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u32::from(desc.DepthOrArraySize) } else { 1 };
            [((desc.Width >> mip) as u32).max(1), (desc.Height >> mip).max(1), (depth >> mip).max(1)]
        }
    };
    // The host takes footprints of 256-byte pitches only (as the console).
    if !dst.footprint.row_pitch.is_multiple_of(256) {
        refused!("a row pitch not a multiple of 256");
    }
    // shortcut: depth and stencil planes copy out in typeless formats of
    // their own; refused until a title is seen reading one back.
    if desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
        refused!("a depth or stencil texture");
    }
    // The host copies into a footprint only within a buffer that holds all
    // of it, so readback memory spans the footprint, not just the region.
    let (Some(rows), Some(len)) = (dst.footprint.rows(at, size), dst.footprint.bytes()) else {
        refused!("a region outside its footprint");
    };
    let (Some(title), Some((buffer, data))) = (buffer_range(dst.resource, dst.footprint.offset, len), list.readback(host, len)) else {
        return;
    };
    if crate::texture::title_written(&texture) {
        crate::texture::upload(list, src.resource, &texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let (mut to, mut from) = (*dst, *src);
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint.offset = 0;
    from.resource = texture.as_raw().cast_const().cast();
    #[cfg(feature = "diagnostics")]
    {
        let text = || {
            let mem = crate::resource::texture_memory(src.resource).map_or(0, |(at, _)| at as usize);
            format!(
                "read back {:?} (mem {mem:#x}, {}x{} f{} sub {}) {size:?} at {at:?} into {:?}+{:#x}, {len} bytes",
                src.resource,
                desc.Width,
                desc.Height,
                desc.Format.0,
                src.footprint.offset as u32,
                dst.resource,
                dst.footprint.offset
            )
        };
        READ_BACKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        READ_BACKS[1].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        let pair = (src.resource as usize, dst.resource as usize);
        let mut seen = READ_BACK_PAIRS.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&pair) {
            seen.push(pair);
            tracing::info!("a texture read back (first of this pair): {}", text());
        }
        list.note_op(text);
    }
    // SAFETY: an open list; both locations in the host's layout, naming live
    // resources; the region lies in the footprint (checked).
    unsafe {
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            at[0],
            at[1],
            at[2],
            (&raw const from).cast(),
            b.map(std::ptr::from_ref),
        )
    };
    let (data, title) = (data as usize, title as usize);
    list.note_landing(title, len as usize);
    list.defer_after(move || {
        for &(at, n) in &rows {
            let (at, n) = (at as usize, n as usize);
            let _writing = crate::watch::writing(title + at, n);
            crate::host::write_mapped((title + at) as *mut u8, (data + at) as *const u8, n);
        }
    });
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_writes(list, dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(&to, dst_sub, &from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    &to,
                    dst_sub,
                    0,
                    0,
                    &from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(&texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(&texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAGS(flags as i32 & 3),
                depth,
                stencil,
                rects,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (`copy_mapped`:
/// an unmapped source page reads as zeros). Overlapping ranges move with
/// `memmove` when both ranges are committed.
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if dst != 0
        && src != 0
        && let Some(list) = copy_into(this, dst as *const u8, len)
    {
        if src.abs_diff(dst) >= len {
            list.defer_copy(dst as *mut u8, src as *const u8, len as usize);
        } else {
            list.defer_move(dst as *mut u8, src as *const u8, len as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do but end a run of hoisted clamps of indirect dispatches, as a
/// barrier does.
extern "system" fn sync(this: This, _flags: u32, _control: u32, _address: u64, _size: u64) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.count_order();
    }
    #[cfg(not(windows))]
    let _ = this;
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size)
        .ok()
        .and_then(|s| s.checked_add(align))
    else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Memory [`allocate_upload`] hands out, zeroed, freed when the list is
/// reset. On Windows it lies in [4 GiB, 2^40), where translated shaders
/// find title memory by address (`crate::memory`; the console's GPU
/// addresses have 40 bits), and is a buffer of its own to the host copies
/// (`crate::shadow`), so its copy is exactly it.
struct Upload {
    #[cfg(windows)]
    at: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    block: Vec<u8>,
}

impl Upload {
    #[cfg(windows)]
    fn new(len: usize) -> Option<Upload> {
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // Watched like the title's pools: copies take only written pages.
        // SAFETY: a fresh allocation; the parameter points at `requirements`,
        // which outlives the call.
        let at = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        } as usize;
        if at == 0 {
            return None;
        }
        crate::shadow::register(at, at as u64, len as u64);
        UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        Some(Upload { at, len })
    }

    #[cfg(not(windows))]
    fn new(len: usize) -> Option<Upload> {
        Some(Upload {
            block: vec![0; len],
        })
    }

    fn start(&self) -> *mut u8 {
        #[cfg(windows)]
        return self.at as *mut u8;
        #[cfg(not(windows))]
        return self.block.as_ptr().cast_mut();
    }
}

#[cfg(windows)]
impl Drop for Upload {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::shadow::unregister(self.at, self.at as u64);
        UPLOADS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_sub(self.len as u64, std::sync::atomic::Ordering::Relaxed);
        // Not guarded any more: memory allocated here next is new.
        crate::watch::title_memory_remapped(self.at, self.len);
        // SAFETY: our allocation; lists that still use its copy hold that.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

// SAFETY: plain memory, handed out once.
#[cfg(windows)]
unsafe impl Send for Upload {}

/// Live list upload blocks and their bytes (the statistics).
#[cfg(windows)]
pub(crate) static UPLOADS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
    #[cfg(windows)]
    with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    #[cfg(not(windows))]
    let _ = (this, min, max);
}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        with_state(this, |list, s| s.viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        // SAFETY: the title passes four floats or null.
        with_state(this, |list, s| s.blend_factor(list, unsafe { factor.as_ref() }));
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.stencil_ref(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    /// `ClearUnorderedAccessViewUint(GPU handle, CPU handle, resource,
    /// values, rectangle count, rectangles)`.
    pub(super) extern "system" fn clear_unordered_uint(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, false, items(rects, count));
    }
    /// `ClearUnorderedAccessViewFloat`, the same with four floats.
    pub(super) extern "system" fn clear_unordered_float(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, true, items(rects, count));
    }
    fn clear_unordered(this: This, cpu: u64, values: *const [u32; 4], float: bool, rects: &[RECT]) {
        // SAFETY: the title passes four values (null is tolerated).
        let Some(&values) = (unsafe { values.as_ref() }) else {
            return;
        };
        with_state(this, |list, s| {
            s.clear_unordered(list, cpu, values, float, rects);
        });
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as clear_unordered_float;
    pub(super) use super::ignored_state as clear_unordered_uint;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE {
            DISPATCH
        } else {
            *kind
        };
        hoist(this, kind);
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}


/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        list.count_order();
        // Which kinds of barriers (0 global, 1 texture, 2 buffer) and how many.
        #[cfg(feature = "diagnostics")]
        if !_groups.is_null() {
            // SAFETY: the title passes `count` groups of 16 bytes: kind, count, pointer.
            let groups = unsafe { std::slice::from_raw_parts(_groups.cast::<[u32; 4]>(), count as usize) };
            list.note_op(|| format!("barrier groups {:?}", groups.iter().map(|g| (g[0], g[1])).collect::<Vec<_>>()));
            crate::copystat::add(
                crate::copystat::C::GroupBarriers,
                groups.iter().map(|g| u64::from(g[1])).sum(),
            );
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GroupCalls, 1);
        // SAFETY: the same layout; no resource named.
        let barrier =
            unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        list.sync_widened();
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
pub(crate) extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, depth_bounds as *const ()),              // 0x220
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait32 as *const ()),                    // 0x238 (newer 0x248)
    (72, wait64 as *const ()),                    // 0x240 (newer 0x250)
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 12] = [
    (53, state::clear_unordered_uint as *const ()), // 0x1a8
    (54, state::clear_unordered_float as *const ()), // 0x1b0
    (55, ignored_state as *const ()),               // 0x1b8 DiscardResource
    (28, query_mark as *const ()),                  // 0xe0 BeginQuery
    (29, query_mark as *const ()),                  // 0xe8 EndQuery
    (30, resolve_queries as *const ()),             // 0xf0
    (78, dispatch_x as *const ()),                  // 0x270
    (98, state::dispatch_mesh_x as *const ()),      // 0x310
    (99, driver_hint as *const ()),                 // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(all(test, windows))]
pub(crate) fn has_newer_table(list: This) -> bool {
    // SAFETY: one of our lists; its first field is the table. The device
    // tests can switch layouts, so use this list's table.
    let table = unsafe { *list.cast::<*const *const ()>() };
    table == NEWER_LIST_FUNCTIONS.0.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{Event, Metadata, Subscriber, span};

    struct PipelineBinds(Arc<AtomicUsize>);

    impl Subscriber for PipelineBinds {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_event()
                && metadata.target() == "victory_gpu::pipeline"
                && metadata.fields().field("pipeline").is_some()
                && metadata.fields().field("frame").is_some()
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            unreachable!()
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if self.enabled(event.metadata()) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn lists_bind_the_initial_pipeline_on_creation() {
        use crate::pipeline::{ComputeHead, create_compute};

        // A synthetic compute description matching the public ABI.
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let desc = ListDesc {
            kind: 0,
            flags: 0,
            nodes: 0,
            barrier_batch: 0,
        };
        let bytes = [0xa5, 0xb6, 0xc7];
        let compute = ComputeDesc {
            root: null,
            code: bytes.as_ptr(),
            size: bytes.len(),
        };
        let compute = std::ptr::from_ref(&compute).cast::<ComputeHead>();
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let binds = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing::Dispatch::new(PipelineBinds(binds.clone()));

        tracing::dispatcher::with_default(&subscriber, || {
            for console in [false, true] {
                let create = |pipeline, out| {
                    if console {
                        create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                    } else {
                        create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                    }
                };
                let mut pipeline = null;
                let mut list = null;
                assert_eq!(
                    create_compute(null, compute, &PAGEABLE_IIDS[0], &mut pipeline),
                    S_OK
                );
                let before = binds.load(Ordering::Relaxed);

                assert_eq!(create(null, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before);
                assert_eq!(release(list), 0);

                assert_eq!(create(pipeline, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before + 1);
                assert_eq!(release(list), 0);
                assert_eq!(release(pipeline), 0);
            }
        });
        assert_eq!(release(allocator), 0);
    }

    /// A title buffer of `bytes` bytes.
    #[cfg(windows)]
    fn title_buffer(bytes: u32) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made = crate::resource::create_committed(
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

    /// The whole way with `VICTORY_COPY_ORDER=split`: a title list that
    /// reads 32 bytes, copies over them and goes on is submitted to a
    /// queue; the part before the copy has run and been finished (its work
    /// for after is done, title memory still holding the old bytes) when
    /// the second part's steps make the copy.
    #[cfg(windows)]
    #[test]
    fn a_list_cut_at_a_copy_runs_as_two_submissions_through_its_queue() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let _split = crate::mac_switches::tests::ForcedCopyOrder::on();
        let null = std::ptr::null();
        let (target, source) = (title_buffer(4096), title_buffer(256));
        let at = crate::resource::buffer_bytes(target).unwrap().0 as u64 + 0x100;
        let from = crate::resource::buffer_bytes(source).unwrap().0 as u64;
        let copy = crate::shadow::find(at, 32).expect("the buffer's host copy");
        // SAFETY: inside the two title buffers.
        unsafe {
            std::ptr::write_bytes(at as *mut u8, 0xaa, 32);
            std::ptr::write_bytes(from as *mut u8, 0xbb, 32);
        }
        let queue_iid = guid(0x0ec870a6, 0x5d7e, 0x4c22, [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed]);
        let mut queue = null;
        assert_eq!(crate::queue::create(null, [0u32; 4].as_ptr().cast(), &queue_iid, &mut queue), S_OK);
        let mut allocator = null;
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        // What title memory holds there, as each part sees it.
        // SAFETY: inside the title buffer, which outlives the submission.
        let now = move || unsafe { (at as *const u8).read_volatile() };

        let first = host_list(list).unwrap();
        copy.refresh(first, at, 32);
        let log = order.clone();
        first.defer_after(move || log.lock().unwrap().push(("the first part has run", now())));
        copy_memory(list, at, from, 32, 0);
        let second = host_list(list).unwrap();
        assert!(!std::ptr::eq(first, second) && second.after_copy());
        let log = order.clone();
        second.defer_host(move || log.lock().unwrap().push(("the second part's steps", now())));
        // A command of the second part reads the bytes too.
        copy.refresh(second, at, 32);
        assert_eq!(close(list), S_OK);

        let splits = crate::host::take_copy_splits();
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        // (Another test may have submissions not waited for just now.)
        crate::host::unwaited::settle(crate::host::unwaited::By::Signal);
        assert_eq!(
            *order.lock().unwrap(),
            [("the first part has run", 0xaa), ("the second part's steps", 0xbb)]
        );
        assert!(crate::host::take_copy_splits() + splits >= 1);
        assert_eq!(copy.read_host(at, 1), [0xbbbb_bbbb], "the second part's commands read the new bytes");

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
        assert_eq!(release(queue), 0);
    }

    /// With `VICTORY_COPY_ORDER=split`, a title copy into bytes that a
    /// command recorded earlier in the list reads moves the recording to a
    /// new host list, marked as starting after a copy and holding the copy;
    /// a copy before any read, a copy elsewhere, and a second copy into
    /// the same bytes (the new part has read nothing yet) do not.
    #[cfg(windows)]
    #[test]
    fn a_copy_into_what_the_list_read_cuts_it_with_the_split_on() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let _split = crate::mac_switches::tests::ForcedCopyOrder::on();
        let null = std::ptr::null();
        let (target, source) = (title_buffer(4096), title_buffer(256));
        let at = crate::resource::buffer_bytes(target).unwrap().0 as u64;
        let from = crate::resource::buffer_bytes(source).unwrap().0 as u64;
        let copy = crate::shadow::find(at, 4096).expect("the buffer's host copy");
        let mut allocator = null;
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        let parts = || segments(list).len();
        let _ = take_copy_cuts();

        // Before any read.
        copy_memory(list, at + 0x100, from, 32, 0);
        assert_eq!(parts(), 1);
        // A command reads 32 bytes at 0x100 (a table's constant buffer).
        copy.refresh(host_list(list).unwrap(), at + 0x100, 32);
        // A copy next to them, to the byte.
        copy_memory(list, at + 0x120, from, 32, 0);
        copy_memory(list, at + 0xe0, from, 32, 0);
        assert_eq!(parts(), 1);
        // A copy into eight of them.
        copy_memory(list, at + 0x110, from, 8, 0);
        assert_eq!(parts(), 2);
        let hosts = segments(list);
        assert!(!hosts[0].after_copy() && hosts[1].after_copy());
        assert!(hosts[0].read_before(at as usize + 0x100, 1) && !hosts[1].read_before(at as usize, 4096));
        // The new part has read nothing: the same copy again stays in it.
        copy_memory(list, at + 0x100, from, 32, 0);
        assert_eq!(parts(), 2);
        // The other calls that copy title buffers: a region, a whole buffer.
        copy.refresh(host_list(list).unwrap(), at + 0x300, 16);
        copy_buffer_region(list, target, 0x308, source, 0, 16);
        assert_eq!(parts(), 3);
        // (The shorter buffer's 256 bytes are copied: a read past them
        // does not count, one inside them does.)
        copy.refresh(host_list(list).unwrap(), at + 0xf00, 4);
        copy_resource(list, target, source);
        assert_eq!(parts(), 3);
        copy.refresh(host_list(list).unwrap(), at + 0xfc, 4);
        copy_resource(list, target, source);
        assert_eq!(parts(), 4);
        assert!(segments(list)[3].after_copy());
        // (Counted for the frame line; another test may cut lists too.)
        assert!(take_copy_cuts() >= 3);

        assert_eq!(close(list), S_OK);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// Hoisted clamps (`crate::indirect::hoisting`): the first indirect
    /// draw of a run cuts the title list, the host list before staying open
    /// for the run's clamps; draws of the same run add no cut until
    /// something may have written arguments, and a wait ends the run. The
    /// host list after such a cut goes on with the budget of the one
    /// before; after a wait it has its own. The host lists run in order,
    /// and a reset starts from one again.
    #[cfg(windows)]
    #[test]
    fn indirect_draws_cut_lists_for_hoisted_clamps() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        // SAFETY: our list.
        let state = unsafe { &(*list.cast::<List>()).state };
        let start = || start_run(list, &mut state.lock().unwrap(), false);
        for _ in 0..2 {
            let viewport = windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT {
                Width: 64.0,
                Height: 64.0,
                MaxDepth: 1.0,
                ..Default::default()
            };
            state::viewports(list, 1, &viewport);
            state::stencil_ref(list, 3);
            start();
            assert_eq!(segments(list).len(), 2);
            let draws = host_list(list).unwrap();
            assert!(state.lock().unwrap().hoists_into(draws));
            assert_eq!(draws.budget(), segments(list)[0].budget(), "one budget across the cut");
            start();
            assert_eq!(segments(list).len(), 2, "the same run");
            draws.count_write();
            assert!(!state.lock().unwrap().hoists_into(draws));
            start();
            assert_eq!(segments(list).len(), 3, "arguments may have been written");
            // A wait ends the run; with nothing recorded since the cut it
            // adds none.
            wait64(list, 0x1000, 0, 0, 0);
            assert!(!state.lock().unwrap().hoists_into(host_list(list).unwrap()));
            assert_eq!(segments(list).len(), 3);
            assert_ne!(host_list(list).unwrap().budget(), segments(list)[1].budget(), "a budget of its own after a wait");
            start();
            assert_eq!(segments(list).len(), 4);
            assert_eq!(close(list), S_OK);
            let all = segments(list);
            host.execute(&all.iter().map(|s| &**s).collect::<Vec<_>>());
            drop(all);
            assert_eq!(reset(list, allocator, null), S_OK);
            assert_eq!(segments(list).len(), 1);
        }
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// Hoisted clamps of indirect dispatches (`VICTORY_DISPATCH_CLAMPS`): the
    /// first of a run cuts the title list as a draw's does. The title's
    /// dispatches do not end the run (what they write is read as arguments
    /// only after a barrier), though they end a run of draws; its next
    /// barrier, enhanced barriers, pipeline sync, copy or clamp in the list
    /// itself does. A run of draws takes the clamps of dispatches after it
    /// until one of those. One budget across the cuts.
    #[cfg(windows)]
    #[test]
    fn indirect_dispatches_cut_lists_for_hoisted_clamps() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        // SAFETY: our list.
        let state = unsafe { &(*list.cast::<List>()).state };
        let start = |dispatch| start_run(list, &mut state.lock().unwrap(), dispatch);
        let joins = || state.lock().unwrap().hoists_dispatches_into(host_list(list).unwrap());
        let draws_join = || state.lock().unwrap().hoists_into(host_list(list).unwrap());
        let parts = || segments(list).len();
        for _ in 0..2 {
            start(true);
            assert_eq!(parts(), 2);
            assert!(joins());
            assert_eq!(host_list(list).unwrap().budget(), segments(list)[0].budget(), "one budget across the cut");
            host_list(list).unwrap().count_dispatch();
            assert!(joins(), "a dispatch does not end the run");
            start(true);
            assert_eq!(parts(), 2, "the same run");
            assert!(!draws_join(), "but it ends a run of draws");
            barrier(list, 1, &UAV_BARRIER);
            assert!(!joins(), "a barrier ends it");
            start(true);
            assert_eq!(parts(), 3);
            sync(list, 0, 0, 0, 0);
            assert!(!joins(), "a pipeline sync ends it");
            start(true);
            assert_eq!(parts(), 4);
            barrier_groups(list, 1, std::ptr::null());
            assert!(!joins(), "enhanced barriers end it");
            start(true);
            assert_eq!(parts(), 5);
            host_list(list).unwrap().count_write();
            assert!(!joins(), "a copy, clear, immediate write or clamp in place ends it");
            // A run of draws, and a dispatch after its draws.
            start(false);
            assert_eq!(parts(), 6);
            barrier(list, 1, &UAV_BARRIER);
            assert!(draws_join() && !joins(), "a barrier ends only a run of dispatches");
            start(false);
            assert_eq!(parts(), 6);
            start(true);
            assert_eq!(parts(), 7);
            start(false);
            assert_eq!(parts(), 7, "draws join a run of dispatches with nothing written since");
            host_list(list).unwrap().count_dispatch();
            start(true);
            assert_eq!(parts(), 7);
            start(false);
            assert_eq!(parts(), 8);
            // A wait ends the run, with a budget of its own after it.
            wait64(list, 0x1000, 0, 0, 0);
            assert!(!joins());
            assert_ne!(host_list(list).unwrap().budget(), segments(list)[6].budget(), "a budget of its own after a wait");
            assert_eq!(close(list), S_OK);
            let all = segments(list);
            host.execute(&all.iter().map(|s| &**s).collect::<Vec<_>>());
            drop(all);
            assert_eq!(reset(list, allocator, null), S_OK);
            assert_eq!(parts(), 1);
        }
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// The order with hoisted clamps of indirect dispatches, on the host
    /// GPU: in one title list a dispatch writes two dispatch records into
    /// a title buffer, a barrier, two indirect dispatches run from them;
    /// then two dispatches write four more records and a count of 2, a
    /// barrier, and an indirect dispatch runs up to 4 records under that
    /// count. Every indirect dispatch runs the groups the dispatch before
    /// it wrote, alone and batched: each run's clamps are recorded after
    /// the barrier that began it, so after the writes. (Hoisted at the
    /// first run's start, the last clamps would read the stale records the
    /// CPU left: 6 groups; with no count, 9.)
    #[cfg(windows)]
    #[test]
    fn hoisted_dispatch_clamps_read_what_an_earlier_dispatch_wrote() {
        use crate::compute::Param;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        const WRITE: &str = r#"
cbuffer Words : register(b0) { uint4 a; uint4 b; uint4 c; };
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() {
    dst.Store4(0, a);
    dst.Store4(16, b);
    dst.Store4(32, c);
}
"#;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store(g.x * 4, 1);
}
"#;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (write_root, write) =
            crate::compute::pipeline(WRITE, s!("write"), 0, &[Param::Constants { register: 0, count: 12 }, Param::Uav { register: 0 }]).unwrap();
        let (mark_root, mark) = crate::compute::pipeline(MARK, s!("marks"), 0, &[Param::Uav { register: 0 }]).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
        let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &argument, NodeMask: 0 };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a description of one argument; no root arguments.
        unsafe { host.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature) }.unwrap();
        let signature = signature.unwrap();
        let target = title_buffer(4096);
        let at = crate::resource::buffer_bytes(target).unwrap().0 as u64;
        let copy = crate::shadow::find(at, 4096).expect("the buffer's host copy");
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        // SAFETY: our list.
        let state = unsafe { &(*list.cast::<List>()).state };
        for batched in [false, true] {
            // What the CPU left: records of 6 groups everywhere.
            // SAFETY: the title buffer's own bytes, before the list runs.
            unsafe {
                for i in 0..1024 {
                    (at as *mut u32).add(i).write_volatile([6, 1, 1][i % 3]);
                }
            }
            copy.refresh(host_list(list).unwrap(), at, 4096);
            let marks: Vec<_> = (0..3).map(|_| host.shared_buffer(64 * 4).unwrap()).collect();
            // SAFETY: our mapped buffers, before the lists run.
            unsafe { marks.iter().for_each(|(_, m)| std::ptr::write_bytes(*m, 0, 64 * 4)) };
            // A dispatch of the title's that writes 12 words at `to`.
            let writes = |to: u64, words: [u32; 12]| {
                let l = host_list(list).unwrap();
                // SAFETY: an open list; our pipeline, the title buffer's copy.
                unsafe {
                    l.list.SetComputeRootSignature(&write_root);
                    l.list.SetPipelineState(&write);
                    l.list.SetComputeRoot32BitConstants(0, 12, words.as_ptr().cast(), 0);
                    l.list.SetComputeRootUnorderedAccessView(1, copy.gpu(to));
                    l.list.Dispatch(1, 1, 1);
                }
                l.count_dispatch();
                assert!(!state.lock().unwrap().hoists_into(l), "no run of draws past it");
            };
            // An indirect dispatch of `records` records at `from`, under the
            // count at `count` if any, marking `into`, as `execute_indirect`
            // and `State::execute_indirect` record it.
            let runs = |from: u64, records: u32, count: Option<u64>, into: &ID3D12Resource| {
                start_run(list, &mut state.lock().unwrap(), true);
                let current = host_list(list).unwrap();
                assert!(state.lock().unwrap().hoists_dispatches_into(current));
                let clamps = state.lock().unwrap().hoist_clamps().unwrap();
                let start = |a: u64| (copy.gpu(a) - copy.offset(a), copy.gpu(a));
                let (out, offset) =
                    crate::indirect::bound_before_as(&clamps, current, start(from), (2, 12, records), count.map(start), batched).unwrap();
                current.count_dispatch();
                // SAFETY: an open list; our pipeline and buffers, the title
                // buffer's copy.
                unsafe {
                    current.list.SetComputeRootSignature(&mark_root);
                    current.list.SetPipelineState(&mark);
                    current.list.SetComputeRootUnorderedAccessView(0, into.GetGPUVirtualAddress());
                    current.list.ExecuteIndirect(
                        &signature,
                        records,
                        &out,
                        offset,
                        count.map(|_| &copy.buffer),
                        count.map_or(0, |c| copy.offset(c)),
                    );
                }
            };
            writes(at, [3, 1, 1, 4, 1, 1, 0, 0, 0, 0, 0, 0]);
            barrier(list, 1, &UAV_BARRIER);
            runs(at, 1, None, &marks[0].0);
            runs(at + 12, 1, None, &marks[1].0);
            assert_eq!(segments(list).len(), 2, "one run");
            writes(at + 0x40, [2, 1, 1, 5, 1, 1, 7, 1, 1, 9, 1, 1]);
            writes(at + 0x100, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            assert!(state.lock().unwrap().hoists_dispatches_into(host_list(list).unwrap()), "dispatches leave the run open");
            barrier(list, 1, &UAV_BARRIER);
            runs(at + 0x40, 4, Some(at + 0x100), &marks[2].0);
            assert_eq!(segments(list).len(), 3, "a run of its own after the barrier");
            assert_eq!(close(list), S_OK);
            let all = segments(list);
            host.execute(&all.iter().map(|s| &**s).collect::<Vec<_>>());
            drop(all);
            let ran: Vec<usize> = marks
                .iter()
                // SAFETY: our mapped buffers, after the lists ran.
                .map(|(_, m)| (0..64).filter(|&i| unsafe { m.cast::<u32>().add(i).read_volatile() } != 0).count())
                .collect();
            // The last runs records of 2 and 5 groups: 5 words marked.
            assert_eq!(ran, [3, 4, 5], "batched {batched}");
            assert_eq!(reset(list, allocator, null), S_OK);
        }
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// The same records are cut with `VICTORY_BATCH_CLAMPS` as without, in
    /// every order: five indirect executions of a title list (three draws
    /// whose clamps are hoisted where a run allows, two dispatches clamped
    /// in place, which end a run), which between them take more than a
    /// budget. In each of their 120 orders, recorded through the list's own
    /// cuts with the clamps alone and in batches, every execution's records
    /// and the budget are what the clamps recorded each before its
    /// execution in one host list give (the model's `alone`, which the host
    /// GPU's clamps are compared with in `crate::indirect`).
    ///
    /// In half of the orders the dispatches' clamps are hoisted too
    /// (`VICTORY_DISPATCH_CLAMPS`), every other one after a title barrier,
    /// which ends the run there.
    #[cfg(windows)]
    #[test]
    fn the_same_records_are_cut_however_the_list_is_cut_for_runs() {
        use crate::batch::tests::{FRESH, RECORD_BUFFERS, Spec, alone, draws, laid_out, orders};
        use windows::core::Interface;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let dispatch = |words: [u32; 3], source| Spec { kind: 2, stride: 12, records: 1, words: words.to_vec(), count: None, source, counter: 0 };
        // (Whether its clamp may be hoisted, the execution.)
        let executions = [
            (true, draws(0, 3, 3 << 18, 40, 0)),
            (true, draws(1, 2, u32::MAX, u32::MAX, 1)),
            (false, dispatch([64, 64, 4], 0)),
            (true, Spec { kind: 10, stride: 12, records: 6, words: [70_000, 1, 1].repeat(6), count: Some(4), source: 2, counter: 1 }),
            (false, dispatch([1 << 10, 1 << 9, 1], 2)),
        ];
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        // SAFETY: our list.
        let state = unsafe { &(*list.cast::<List>()).state };
        let (mut cuts, mut outcomes) = (0, std::collections::HashSet::new());
        for (n, order) in orders(&executions).into_iter().enumerate() {
            let batched = n % 2 == 0;
            let hoist_dispatches = n % 4 >= 2;
            let specs: Vec<Spec> = order.iter().map(|(_, s)| s.clone()).collect();
            let (contents, places, _) = laid_out(&specs);
            let model = alone(&specs, 0, crate::indirect::widest());
            assert_eq!(model.1, 1 << 27, "they take more than a budget");
            let buffers: Vec<_> = contents.iter().map(|c| host.shared_buffer(c.len() as u64 * 4).unwrap()).collect();
            // Where each execution's clamped records are (the buffer and the
            // byte offset), and the last clamp in place of each host list,
            // whose records its argument buffer still holds.
            let mut outs = Vec::new();
            let mut in_place = std::collections::HashMap::new();
            // SAFETY: our mapped buffers, written before the lists run and
            // read after they have; live buffers.
            unsafe {
                for (c, (_, data)) in contents.iter().zip(&buffers) {
                    std::ptr::copy_nonoverlapping(c.as_ptr(), data.cast::<u32>(), c.len());
                }
                let mut dispatches = 0;
                for (i, ((hoisted, s), &(from, count_at, _))) in order.iter().zip(&places).enumerate() {
                    let base = buffers[s.source].0.GetGPUVirtualAddress();
                    let counter = buffers[RECORD_BUFFERS + s.counter].0.GetGPUVirtualAddress();
                    let counted = s.count.map(|_| (counter, counter + u64::from(count_at)));
                    let spec = (s.kind, s.stride, s.records);
                    let gpu = (base, base + u64::from(from));
                    let dispatch = s.kind == crate::draw::INDIRECT_DISPATCH;
                    if dispatch && hoist_dispatches {
                        dispatches += 1;
                        if dispatches % 2 == 0 {
                            barrier(list, 1, &UAV_BARRIER);
                        }
                        start_run(list, &mut state.lock().unwrap(), true);
                        let current = host_list(list).unwrap();
                        assert!(state.lock().unwrap().hoists_dispatches_into(current));
                        let clamps = state.lock().unwrap().hoist_clamps().unwrap();
                        outs.push(crate::indirect::bound_before_as(&clamps, current, gpu, spec, counted, batched).unwrap());
                        current.count_dispatch();
                    } else if *hoisted {
                        // As `execute_indirect` and `State::execute_indirect` do.
                        start_run(list, &mut state.lock().unwrap(), false);
                        let current = host_list(list).unwrap();
                        assert!(state.lock().unwrap().hoists_into(current));
                        let clamps = state.lock().unwrap().hoist_clamps().unwrap();
                        outs.push(crate::indirect::bound_before_as(&clamps, current, gpu, spec, counted, batched).unwrap());
                    } else {
                        let current = host_list(list).unwrap();
                        let out = crate::indirect::bound(current, gpu.1, spec, counted.map(|c| c.1)).unwrap();
                        crate::indirect::done(current, &out);
                        current.count_write();
                        in_place.insert(out.as_raw() as usize, i);
                        outs.push((out, 0));
                    }
                }
                assert_eq!(close(list), S_OK);
                let all = segments(list);
                cuts += all.len() - 1;
                host.execute(&all.iter().map(|s| &**s).collect::<Vec<_>>());
                assert_eq!(all.last().unwrap().budget_spent(), model.1, "order {n}: the budget");
                drop(all);
                for (i, (((_, s), (out, at)), &(_, _, to))) in order.iter().zip(&outs).zip(&places).enumerate() {
                    if in_place.get(&(out.as_raw() as usize)).is_some_and(|&last| last != i) {
                        continue; // a later clamp in place wrote over it
                    }
                    let mut data = std::ptr::null_mut();
                    out.Map(0, None, Some(&mut data)).unwrap();
                    let data = data.cast::<u8>().add(*at as usize).cast::<u32>();
                    for w in 0..(s.stride * s.records / 4) as usize {
                        let expected = model.0[to as usize / 4 + w];
                        if expected != FRESH {
                            assert_eq!(data.add(w).read_volatile(), expected, "order {n} ({batched}): execution {i}, word {w}");
                        }
                    }
                }
            }
            outcomes.insert(model.0);
            assert_eq!(reset(list, allocator, null), S_OK);
        }
        assert!(outcomes.len() > 10, "the order decides what is cut");
        assert!(cuts >= 120, "the lists were cut for runs: {cuts}");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// With `VICTORY_COPY_ORDER=split`, the arguments of an indirect draw
    /// whose clamp is hoisted are read by the list before the one that
    /// draws; they are noted in the drawing list too, so a title copy into
    /// them recorded there cuts it as it does without hoisting. The cut
    /// ends the run (its list of clamps closes) and the part after the copy
    /// has a budget of its own, as without hoisting.
    #[cfg(windows)]
    #[test]
    fn a_copy_into_the_arguments_of_a_hoisted_draw_cuts_the_list() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _split = crate::mac_switches::tests::ForcedCopyOrder::on();
        let null = std::ptr::null();
        let (target, source) = (title_buffer(4096), title_buffer(256));
        let at = crate::resource::buffer_bytes(target).unwrap().0 as u64;
        let from = crate::resource::buffer_bytes(source).unwrap().0 as u64;
        let copy = crate::shadow::find(at, 4096).expect("the buffer's host copy");
        let (mut allocator, mut list) = (null, null);
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        // SAFETY: our list.
        let state = unsafe { &(*list.cast::<List>()).state };
        start_run(list, &mut state.lock().unwrap(), false);
        let hosts = segments(list);
        assert_eq!(hosts.len(), 2);
        // An indirect draw with 16 bytes of arguments at 0x100, as
        // `State::execute_indirect` records it with its clamp hoisted.
        let gpu = copy.gpu(at + 0x100);
        let clamped = crate::indirect::bound_before(&hosts[0], &hosts[1], (gpu - copy.offset(at + 0x100), gpu), (0, 16, 1), None);
        assert!(clamped.is_some());
        copy.refresh(&hosts[0], at + 0x100, 16);
        hosts[1].note_hoisted_read(at as usize + 0x100, 16);
        // A copy elsewhere stays in the drawing list; one into the
        // arguments cuts it.
        copy_memory(list, at + 0x200, from, 32, 0);
        assert_eq!(segments(list).len(), 2);
        assert!(!state.lock().unwrap().hoists_into(&hosts[1]), "a copy ends the run");
        copy_memory(list, at + 0x108, from, 4, 0);
        let hosts = segments(list);
        assert_eq!(hosts.len(), 3);
        assert!(hosts[2].after_copy());
        assert_eq!(hosts[1].budget(), hosts[0].budget());
        assert_ne!(hosts[2].budget(), hosts[0].budget(), "a budget of its own after the copy");
        assert_eq!(close(list), S_OK);
        // Every part is closed and runs, the clamps before their draw.
        for part in crate::host::copy_order_parts(hosts) {
            host.execute(&part.iter().map(|s| &**s).collect::<Vec<_>>());
        }
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// Without the switch nothing is noted and no copy cuts a list.
    #[cfg(windows)]
    #[test]
    fn a_copy_cuts_no_list_without_the_split() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        // Another test may have the split forced on just now.
        if crate::mac_switches::copy_order_split().is_some() {
            return;
        }
        let list = crate::host::host().unwrap().new_list().unwrap();
        assert!(!list.read_before(0, usize::MAX));
        list.title_copies(0x1000, 32);
        assert!(crate::host::copy_order_parts(vec![Arc::new(list)]).len() == 1);
    }

    /// ClearUnorderedAccessViewUint/Float and DiscardResource come before
    /// the methods newer lists insert: the AMD upscaler calls them there.
    #[test]
    fn unordered_access_clears_have_the_same_slots_in_both_layouts() {
        for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
            assert_eq!(table.0[0x1a8 / 8], state::clear_unordered_uint as *const ());
            assert_eq!(
                table.0[0x1b0 / 8],
                state::clear_unordered_float as *const ()
            );
            assert_eq!(table.0[0x1b8 / 8], ignored_state as *const ());
        }
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, below 2^40 (where
        // shaders find title memory), gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        #[cfg(windows)]
        assert!((1 << 32..1 << 40).contains(&(p as usize)), "{p:p}");
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(
            allocate_upload(list, 8, 0x30).is_null(),
            "not a power of two"
        );
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(
            // SAFETY: the list is ours.
            unsafe { &(*list.cast::<List>()).uploads }
                .lock()
                .unwrap()
                .is_empty()
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    /// Transitions between two read-only states are told from those that
    /// write, from common, and from other barrier kinds.
    #[test]
    fn read_only_transitions_need_both_states_read_only() {
        const COMMON: u32 = 0;
        const INDEX: u32 = 0x2;
        const UNORDERED: u32 = 0x8;
        const NON_PIXEL: u32 = 0x40;
        const PIXEL: u32 = 0x80;
        const COPY_DEST: u32 = 0x400;
        const GENERIC_READ: u32 = 0xac3;
        // The console's compression hints live above the PC bits.
        const HINT: u32 = 1 << 26;
        let transition = |kind, before: u32, after: u32| Barrier {
            kind,
            flags: 0,
            first: 8 as This,
            second: 0xffff_ffff | (u64::from(before) << 32),
            last: u64::from(after),
        };
        assert!(read_only_transition(&transition(0, NON_PIXEL, PIXEL)));
        let both = |before, after| read_only_transition(&transition(0, before, after));
        assert!(both(GENERIC_READ, INDEX | PIXEL));
        assert!(both(NON_PIXEL | HINT, PIXEL));
        assert!(!read_only_transition(&transition(0, UNORDERED, PIXEL)));
        assert!(!read_only_transition(&transition(0, PIXEL, UNORDERED)));
        assert!(!read_only_transition(&transition(0, NON_PIXEL, COPY_DEST)));
        assert!(!read_only_transition(&transition(0, COMMON, PIXEL)));
        assert!(!read_only_transition(&transition(0, NON_PIXEL, COMMON)));
        assert!(!read_only_transition(&transition(2, NON_PIXEL, PIXEL)));
        assert!(!read_only_transition(&UAV_BARRIER));
    }
}
