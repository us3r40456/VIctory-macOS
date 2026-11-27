//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! With a count buffer ExecuteIndirect runs only the first min(max, count)
//! records, so the clamp reads the count word and stops there: the maximum
//! can be far above the count, and the records between never run.
//!
//! With `VICTORY_BATCH_CLAMPS=1` the clamps of a run of indirect draws go
//! to the host list before the one that draws ([`bound_before`]): each
//! clamp before its draw puts a dispatch of ours and three barriers between
//! every two draws, which ends and restarts the host's render pass each
//! time (from 64bitMango 18872f48, after Belobug ccab973). With `1` a
//! run's clamps are then one dispatch ([`Pending`], from 64bitMango
//! 131721b7): one thread walks a table of them in the order of the draws,
//! with the results and the budget of one dispatch each.
//!
//! With `VICTORY_DISPATCH_CLAMPS=1` as well, the clamps of indirect
//! dispatches go there too ([`hoisting_dispatches`]). A run of them does
//! not end at the title's dispatches but at its next barrier, pipeline
//! sync, copy, clear or immediate write: by D3D12's rules arguments a
//! dispatch writes are read as such only after a barrier. The title's
//! compute lists write a table of records, put a barrier, and run hundreds
//! of indirect dispatches from it (a frame at the prison: 961 in 104 such
//! runs by the barriers a frame capture holds, each clamp a dispatch of
//! ours and three barriers before).
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension except in x where [`widest`] allows more, 2^20
//! per dispatch, one-dimensional mesh dispatches, the budget) cut
//! legitimate work beyond them; raise them if the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, host, transition, uav_barrier};

const SOURCE: &str = include_str!("indirect-clamp.hlsl");

/// The most groups in x the clamp lets a dispatch run (2^20 groups in all
/// still bound it). The title's tile passes are dispatches of one group per
/// 8x4 tile of the picture, 67,440 and 47,760 for its two tile classes in a
/// frame at 2560x1440 (measured on Windows in VIctory-7900XTX, 84eae35);
/// cut to 65,535, the last tiles, the lowest of the picture, never run.
/// The console runs what it is asked, and so does D3DMetal: an M5 Pro ran
/// every group of indirect dispatches up to 2^20 wide. D3D12's own limit,
/// 65,535, stays for other hosts. `VICTORY_INDIRECT_WIDE=0` or `1`
/// overrides.
pub(crate) fn widest() -> u32 {
    static WIDEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WIDEST.get_or_init(|| {
        let wide = match std::env::var("VICTORY_INDIRECT_WIDE").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => crate::shader_cache::under_wine(),
        };
        if wide { 1 << 20 } else { 65_535 }
    })
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Built from [`WORDS_SOURCE`], which also reads the budget as words
    /// through root view 5 (the other views are those of [`SOURCE`]).
    words: bool,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    #[cfg(victory_dxc_clamp)]
    match precompiled() {
        Ok(clamp) => {
            tracing::info!("argument clamp: using precompiled 64-thread shader");
            return Ok(clamp);
        }
        Err(e) => tracing::warn!(error = %e, "precompiled argument clamp unavailable"),
    }
    make_fxc().or_else(|e| {
        tracing::info!(error = %e, "argument clamp: building the one-thread form");
        make_words()
    })
}

/// The clamp compiled by DXC at build time. D3DMetal finds a shader by the
/// `HASH` part of its container, and DXC leaves that and the container
/// digest as zeros when it cannot find `dxil.dll` to sign with, which would
/// make this shader share a cache entry with every other unsigned one (issue
/// #4). So both are checked here and set when they are zero.
#[cfg(victory_dxc_clamp)]
fn precompiled() -> windows::core::Result<Clamp> {
    let mut bytes = include_bytes!(concat!(env!("OUT_DIR"), "/indirect-clamp.dxil")).to_vec();
    match victory_shader::emit::sign_container(&mut bytes) {
        Ok(true) => tracing::info!("argument clamp: the compiled shader was unsigned, its digests are set"),
        Ok(false) => {}
        Err(e) => tracing::warn!(error = %e, "argument clamp: could not check the compiled shader's digests"),
    }
    from_bytes(&bytes)
}

/// The clamp for Wine's HLSL compiler (vkd3d-shader), which has no `Load`
/// on `RWByteAddressBuffer`, no stores to indexed group-shared memory and
/// no `[RootSignature]`: [`SOURCE`]'s clamp, read through structured views
/// of the same buffers, and one thread charging the records in order
/// instead of a group of 64. The results are the same.
const WORDS_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
uint3 load3(uint at) { return uint3(argWords[at / 4], argWords[at / 4 + 1], argWords[at / 4 + 2]); }
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        uint count = min(argWords[at / 4], 1u << 24);
        uint instances = min(argWords[at / 4 + 1], (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1) {
            dest.Store3(at + 8, uint3(argWords[at / 4 + 2], argWords[at / 4 + 3], argWords[at / 4 + 4]));
        } else {
            dest.Store2(at + 8, uint2(argWords[at / 4 + 2], argWords[at / 4 + 3]));
        }
    } else if (kind == 2) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        uint3 g = load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint walked = records;
    if (counted != 0) {
        walked = min(records, countWords[0]);
    }
    for (uint record = 0; record < walked; record++) {
        uint at = record * stride;
        uint cost = clamped(at);
        if (cost > left) {
            // The first record that does not fit takes what is left, so
            // those after it run empty too.
            dest.Store(at, 0);
            left = 0;
        } else {
            left -= cost;
        }
    }
    budget.Store(0, BUDGET - left);
}
"#;

/// Many clamps in one dispatch ([`Batch`]): [`WORDS_SOURCE`]'s clamp, one
/// thread walking a table of clamps in order, each as one dispatch of it
/// would (the same records, results and budget). A clamp's records and its
/// count are in one of four buffers by offset; its clamped records go to
/// one buffer by offset. Written for Wine's compiler, as that source is
/// (words read through structured views, stored through byte-address
/// ones, one return a function); `crate::batch::model` is this in Rust.
const BATCH_SOURCE: &str = r#"
cbuffer Batch : register(b0) { uint entries; uint widest; };
StructuredBuffer<uint> s0 : register(t0);
StructuredBuffer<uint> s1 : register(t1);
StructuredBuffer<uint> s2 : register(t2);
StructuredBuffer<uint> s3 : register(t3);
// Per clamp, ENTRY words: kind, stride, records, flags (1: under a count),
// the buffer and byte offset of its records, of its count, and the byte
// offset of its clamped records in `dest`.
StructuredBuffer<uint> table : register(t4);
RWByteAddressBuffer dest : register(u0);
// The budget, read as words and written as bytes.
RWByteAddressBuffer budget : register(u1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint ENTRY = 10;
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
uint word(uint slot, uint at) {
    uint w;
    if (slot == 0) {
        w = s0[at / 4];
    } else if (slot == 1) {
        w = s1[at / 4];
    } else if (slot == 2) {
        w = s2[at / 4];
    } else {
        w = s3[at / 4];
    }
    return w;
}
uint3 load3(uint slot, uint at) { return uint3(word(slot, at), word(slot, at + 4), word(slot, at + 8)); }
uint clamped(uint kind, uint slot, uint at, uint put) {
    uint cost;
    if (kind <= 1) {
        uint count = min(word(slot, at), 1u << 24);
        uint instances = min(word(slot, at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(put, uint2(count, instances));
        if (kind == 1) {
            dest.Store3(put + 8, load3(slot, at + 8));
        } else {
            dest.Store2(put + 8, uint2(word(slot, at + 8), word(slot, at + 12)));
        }
    } else if (kind == 2) {
        uint3 g = load3(slot, at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(put, g);
    } else {
        uint3 g = load3(slot, at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(put, g);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    for (uint e = 0; e < entries; e++) {
        uint kind = table[e * ENTRY];
        uint stride = table[e * ENTRY + 1];
        uint records = table[e * ENTRY + 2];
        uint flags = table[e * ENTRY + 3];
        uint slot = table[e * ENTRY + 4];
        uint from = table[e * ENTRY + 5];
        uint countSlot = table[e * ENTRY + 6];
        uint countAt = table[e * ENTRY + 7];
        uint to = table[e * ENTRY + 8];
        uint walked = records;
        if ((flags & 1) != 0) {
            walked = min(records, word(countSlot, countAt));
        }
        for (uint record = 0; record < walked; record++) {
            uint at = from + record * stride;
            uint put = to + record * stride;
            uint cost = clamped(kind, slot, at, put);
            if (cost > left) {
                // The first record that does not fit takes what is left, so
                // those after it run empty too.
                dest.Store(put, 0);
                left = 0;
            } else {
                left -= cost;
            }
        }
    }
    budget.Store(0, BUDGET - left);
}
"#;
const _: () = assert!(crate::batch::ENTRY == 10 && crate::batch::SOURCES == 4);

/// The pipeline that runs a batch of clamps. Root parameters: the count of
/// clamps and [`widest`] (constants), the four buffers of records, the
/// table (shader resource views), the clamped records' buffer and the
/// budget twice (unordered access).
struct Batch {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Batch {}
// SAFETY: as above.
unsafe impl Sync for Batch {}

/// The batch pipeline, made once; `None` if it does not build: every
/// clamp is then recorded alone.
fn batch() -> Option<&'static Batch> {
    use crate::compute::Param;
    static BATCH: std::sync::OnceLock<Option<Batch>> = std::sync::OnceLock::new();
    BATCH
        .get_or_init(|| {
            let params = [
                Param::Constants { register: 0, count: 2 },
                Param::Srv { register: 0, space: 0 },
                Param::Srv { register: 1, space: 0 },
                Param::Srv { register: 2, space: 0 },
                Param::Srv { register: 3, space: 0 },
                Param::Srv { register: 4, space: 0 },
                Param::Uav { register: 0 },
                Param::Uav { register: 1 },
                Param::Uav { register: 2 },
            ];
            match crate::compute::pipeline(BATCH_SOURCE, s!("clamps"), 0, &params) {
                Ok((root, pso)) => Some(Batch { root, pso }),
                Err(e) => {
                    tracing::warn!(error = %e, "no batches of argument clamps: each is recorded alone");
                    None
                }
            }
        })
        .as_ref()
}

fn make_words() -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 5 },
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Uav { register: 0 },
        Param::Srv { register: 1, space: 0 },
        Param::Uav { register: 2 },
    ];
    let (root, pso) = crate::compute::pipeline(WORDS_SOURCE, s!("clamp"), 0, &params)?;
    Ok(Clamp { root, pso, words: true })
}

fn make_fxc() -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the compiler owns the blob until this call returns.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    from_bytes(bytes)
}

fn from_bytes(bytes: &[u8]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    // SAFETY: D3D12 validates the bytecode and its embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        // By the name the one-thread form gets from `compute::pipeline`:
        // what a frame replay knows the clamp by (`--skip-clamps`).
        crate::host::name(&pso, || "victory: clamp".to_owned());
        Ok(Clamp { root, pso, words: false })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// `count` is the host address of the title's record count (a title
/// buffer's copy, only read): the clamp then stops at the lesser of it and
/// `records`, which is where ExecuteIndirect stops. A count that is not
/// 4-byte aligned is not read: every record is clamped.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    bound_with(clamp()?, list, gpu, (kind, stride, records), count)
}

/// [`bound`] with a given clamp.
fn bound_with(
    c: &Clamp,
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    let budget = list.budget()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    if kind == DISPATCH {
        FOR_DISPATCHES[0].fetch_add(1, Ordering::Relaxed);
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    // SAFETY: an open list; our buffers, and title buffers' copies the
    // caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        run(l, c, budget, (gpu, count), (kind, stride, records), out.GetGPUVirtualAddress());
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// Records the clamp's dispatch: the records at `gpu`, under the count at
/// `count` if any, clamped into the memory at host address `dest`, in
/// unordered access.
///
/// # Safety
/// `l` is open; `budget`, `gpu`, `count` and `dest` are addresses in live
/// buffers the list holds.
unsafe fn run(
    l: &ID3D12GraphicsCommandList,
    c: &Clamp,
    budget: u64,
    (gpu, count): (u64, Option<u64>),
    (kind, stride, records): (u32, u32, u32),
    dest: u64,
) {
    let constants = [kind, stride, records, u32::from(count.is_some()), widest()];
    // SAFETY: the caller's; our root signature and pipeline.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, dest);
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// Whether the clamps of a run of indirect draws go to the host list
/// before the one that draws (`VICTORY_BATCH_CLAMPS=1` or `hoist`,
/// `crate::mac_switches::batch_clamps`); else each is recorded before its
/// draw.
pub(crate) fn hoisting() -> bool {
    crate::mac_switches::batch_clamps()[0]
}

/// Whether the clamps of runs of indirect dispatches go to the host list
/// before too (`VICTORY_DISPATCH_CLAMPS=1` with [`hoisting`],
/// `crate::mac_switches::dispatch_clamps`); else each is recorded before
/// its dispatch.
pub(crate) fn hoisting_dispatches() -> bool {
    crate::mac_switches::dispatch_clamps()
}

/// As [`bound`], recorded in `clamps`, the host list that runs before
/// `into`, for an execution `into` records: the clamped records go to one
/// of `into`'s own buffers ([`HostList::clamp_room`]), which the execution
/// reads at the offset returned once `clamps` has ended
/// ([`HostList::clamp_end`]). `None` also when there is no room: the
/// caller clamps in `into`.
///
/// The clamps of a run charge the budget of `clamps` one after another, in
/// the order of the draws (a barrier for shader writes before each): the
/// records past the budget are the same ones from frame to frame. Clamps
/// that did not wait for each other made whole mesh objects flicker in
/// Belobug's fork (leaves, chairs, clothes; fixed there in 76387c2).
///
/// `gpu` and `count` are host addresses with the start of the buffer each
/// is in (`(start, address)`): batches read the records from there.
pub(crate) fn bound_before(
    clamps: &HostList,
    into: &HostList,
    gpu: (u64, u64),
    spec: (u32, u32, u32),
    count: Option<(u64, u64)>,
) -> Option<(ID3D12Resource, u64)> {
    bound_before_as(clamps, into, gpu, spec, count, crate::mac_switches::batch_clamps()[1])
}

/// [`bound_before`], the clamp `batched` or recorded at once.
pub(crate) fn bound_before_as(
    clamps: &HostList,
    into: &HostList,
    (base, gpu): (u64, u64),
    (kind, stride, records): (u32, u32, u32),
    count: Option<(u64, u64)>,
    batched: bool,
) -> Option<(ID3D12Resource, u64)> {
    let (c, budget) = (clamp()?, clamps.budget()?);
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|(_, at)| at.is_multiple_of(4));
    let (out, at) = into.clamp_room(clamps, u64::from(stride) * u64::from(records.max(1)))?;
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    HOISTED.fetch_add(1, Ordering::Relaxed);
    if kind == DISPATCH {
        FOR_DISPATCHES[0].fetch_add(1, Ordering::Relaxed);
        FOR_DISPATCHES[1].fetch_add(1, Ordering::Relaxed);
    }
    if batched
        && batch().is_some()
        && Pending::add(clamps, (base, gpu), count, (kind, stride, records), &out, at, budget).is_some()
    {
        BATCHED.fetch_add(1, Ordering::Relaxed);
        if kind == DISPATCH {
            FOR_DISPATCHES[2].fetch_add(1, Ordering::Relaxed);
        }
        return Some((out, at));
    }
    // Alone, after the clamps batched before it: the budget is charged in
    // the order of the draws.
    flush(clamps);
    let l = &clamps.list;
    // SAFETY: an open list; our buffers (`out` in unordered access), and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes, the clamp
        // before) first.
        l.ResourceBarrier(&[uav_barrier()]);
        run(l, c, budget, (gpu, count.map(|c| c.1)), (kind, stride, records), out.GetGPUVirtualAddress() + at);
    }
    Some((out, at))
}

/// Hoisted clamps batched since the last [`flush`], recorded in a list
/// ([`HostList::clamp_batch`]) as one dispatch where their run ends: all
/// into one buffer of the drawing list's and charging one budget, reading
/// at most [`crate::batch::SOURCES`] buffers.
pub(crate) struct Pending {
    table: crate::batch::Table,
    dest: ID3D12Resource,
    budget: u64,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pending {}

impl Pending {
    /// Adds a clamp to `clamps`' batch, recording the batch so far first
    /// where this one does not fit it (another destination, a fifth
    /// buffer); `None` (nothing added) where it fits no batch: an offset
    /// that is not of whole words or is past 32 bits.
    fn add(
        clamps: &HostList,
        (base, gpu): (u64, u64),
        count: Option<(u64, u64)>,
        spec: (u32, u32, u32),
        dest: &ID3D12Resource,
        at: u64,
        budget: u64,
    ) -> Option<()> {
        use crate::batch::offset;
        let from = (base, offset(base, gpu)?);
        let count = match count {
            Some((start, address)) => Some((start, offset(start, address)?)),
            None => None,
        };
        let to = u32::try_from(at).ok()?;
        let mut pending = clamps.clamp_batch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = pending.as_mut()
            && p.dest == *dest
            && p.budget == budget
            && p.table.add(spec, from, count, to)
        {
            return Some(());
        }
        if let Some(p) = pending.take() {
            p.record(clamps);
        }
        let mut table = crate::batch::Table::default();
        table.add(spec, from, count, to).then_some(())?;
        *pending = Some(Pending { table, dest: dest.clone(), budget });
        Some(())
    }

    /// Records the batch's dispatch in `clamps`, after an unordered-access
    /// barrier for the title's argument passes and the clamps before it.
    fn record(self, clamps: &HostList) {
        let (Some(b), Some(c), Some(sources)) = (batch(), clamp(), self.table.bound()) else {
            return;
        };
        // SAFETY: a live buffer.
        let dest = unsafe { self.dest.GetGPUVirtualAddress() };
        // A batch of one is the clamp alone.
        if let [e] = &self.table.entries[..] {
            let at = |(slot, offset): (u32, u32)| sources[slot as usize] + u64::from(offset);
            let count = (e.flags & 1 != 0).then(|| at(e.count));
            // SAFETY: an open list; our buffers (the destination in
            // unordered access) and title buffers' copies the list holds.
            unsafe {
                clamps.list.ResourceBarrier(&[uav_barrier()]);
                run(&clamps.list, c, self.budget, (at(e.from), count), (e.kind, e.stride, e.records), dest + u64::from(e.to));
            }
            return;
        }
        let words = self.table.words();
        let Some((table, data)) = clamps.scratch(words.len() as u64 * 4) else {
            tracing::warn!("argument clamps: no room for a batch's table; its draws read what their buffer held");
            return;
        };
        let constants = [self.table.entries.len() as u32, widest()];
        let l = &clamps.list;
        // SAFETY: upload memory of the list's, as long as the table; an
        // open list; our root signature, pipeline and buffers (the
        // destination in unordered access) and title buffers' copies the
        // list holds.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), words.len());
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(&b.root);
            l.SetPipelineState(&b.pso);
            l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            for (i, source) in sources.iter().enumerate() {
                l.SetComputeRootShaderResourceView(1 + i as u32, *source);
            }
            l.SetComputeRootShaderResourceView(5, table);
            l.SetComputeRootUnorderedAccessView(6, dest);
            l.SetComputeRootUnorderedAccessView(7, self.budget);
            l.SetComputeRootUnorderedAccessView(8, self.budget);
            l.Dispatch(1, 1, 1);
        }
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        BATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Records the clamps batched in `clamps` that are not yet: where their
/// run ends ([`HostList::clamp_end`]), before a clamp recorded alone, or
/// where `clamps` closes.
pub(crate) fn flush(clamps: &HostList) {
    let pending = clamps.clamp_batch.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(p) = pending {
        p.record(clamps);
    }
}

/// Clamps that went into batches, and batches of more than one recorded,
/// since the last call (the frame line).
pub(crate) fn take_batched() -> (u64, u64) {
    (BATCHED.swap(0, Ordering::Relaxed), BATCHES.swap(0, Ordering::Relaxed))
}
static BATCHED: AtomicU64 = AtomicU64::new(0);
static BATCHES: AtomicU64 = AtomicU64::new(0);

/// Clamps recorded in the host list before their draw's since the last
/// call (the frame line).
pub(crate) fn take_hoisted() -> u64 {
    HOISTED.swap(0, Ordering::Relaxed)
}
static HOISTED: AtomicU64 = AtomicU64::new(0);

/// Dispatches of ours recorded for clamps since the last call (the frame
/// line): one before every indirect execution, each ending the host's
/// render pass where it stands between two draws.
pub(crate) fn take_dispatches() -> u64 {
    DISPATCHES.swap(0, Ordering::Relaxed)
}
static DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// The kind of an indirect dispatch's records (PC numbering).
const DISPATCH: u32 = 2;

/// Clamps of indirect dispatches since the last call (the frame line), of
/// all the clamps: recorded, in the list before their dispatch's, in
/// batches.
pub(crate) fn take_for_dispatches() -> [u64; 3] {
    FOR_DISPATCHES.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static FOR_DISPATCHES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Records the clamps were asked to walk (the maximum of each execution)
/// since the last call, and the part of them under a count buffer, where
/// the walk stops at the count.
pub(crate) fn take_records() -> [u64; 2] {
    RECORDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static RECORDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Every word the clamp `c` writes for `records` records of `kind` made
    /// from `seed` by a small generator, in one list with no count buffer.
    /// `None` without a host GPU.
    #[cfg(victory_dxc_clamp)]
    fn clamp_all(c: &super::Clamp, kind: u32, seed: u32, records: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let all = per * records as usize;
        let (buffer, data) = host.shared_buffer(all as u64 * 4 + 16).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        let mut state = seed;
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Mostly small values, now and then a wild one.
                let word = if state >> 28 == 0 { state } else { (state >> 8) % 9000 };
                data.add(at).write_volatile(word);
            }
            let spec = (kind, per as u32 * 4, records);
            let out = super::bound_with(c, &list, buffer.GetGPUVirtualAddress(), spec, None).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut out_data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut out_data)).unwrap();
            Some((0..all).map(|i| out_data.cast::<u32>().add(i).read_volatile()).collect())
        }
    }

    /// The precompiled clamp is the one in use when it builds, and it writes
    /// what the clamp compiled at run time writes for the same records of
    /// every kind.
    #[cfg(victory_dxc_clamp)]
    #[test]
    #[ignore = "requires a hardware D3D12 queue and the build-time DXC shader"]
    fn hardware_uses_precompiled_clamp() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let precompiled = super::precompiled().expect("create pipeline from the compiled clamp");
        assert!(!precompiled.words);
        assert!(!super::clamp().expect("argument clamp").words);
        let reference = super::make_fxc().or_else(|_| super::make_words()).expect("the existing clamp");
        for (kind, seed, records) in [(0, 1, 200), (1, 2, 200), (2, 3, 130), (10, 4, 70), (0, 5, 3)] {
            let new = clamp_all(&precompiled, kind, seed, records).unwrap();
            let old = clamp_all(&reference, kind, seed, records).unwrap();
            assert_eq!(new, old, "kind {kind}, {records} records");
        }
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_under(kind, words, records, calls, None)
    }

    /// As [`clamped`], with the title's record count `count` (a word in a
    /// buffer of its own) if there is one.
    fn clamped_under(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        count: Option<u32>,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let (counter, word) = host.shared_buffer(4).unwrap();
            word.cast::<u32>().write_volatile(count.unwrap_or(0));
            let counted = count.map(|_| counter.GetGPUVirtualAddress());
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, counted).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [70_000.min(super::widest()), 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// Hoisted clamps (`bound_before`) leave the clamped records in the
    /// drawing list's own buffer, one run of records after the other, and
    /// charge the budget of the list before in the order of the draws: of
    /// two draws of four garbage records (2^26 vertex instances each once
    /// clamped) the first two records of the first fit, none of the second;
    /// under a count of 1 only the first is clamped. Every barrier starts
    /// from the buffer's current state, the title's copy is never
    /// transitioned and all end common.
    #[test]
    fn hoisted_clamps_fill_the_drawing_lists_buffer() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (per, records) = (4usize, 4u32);
        let bytes = (per * records as usize * 4) as u64;
        let (title, data) = host.shared_buffer(bytes).unwrap();
        let (counter, word) = host.shared_buffer(4).unwrap();
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for count in [None, Some(1u32)] {
            let (clamps, draws) = (host.new_list().unwrap(), host.new_list().unwrap());
            issued();
            // SAFETY: our mapped buffers, written before the lists run and
            // read after they have; live buffers.
            unsafe {
                for at in 0..per * records as usize {
                    data.cast::<u32>().add(at).write_volatile(u32::MAX);
                }
                word.cast::<u32>().write_volatile(count.unwrap_or(0));
                let gpu = (title.GetGPUVirtualAddress(), title.GetGPUVirtualAddress());
                let counted = count.map(|_| (counter.GetGPUVirtualAddress(), counter.GetGPUVirtualAddress()));
                let spec = (0, per as u32 * 4, records);
                let first = super::bound_before_as(&clamps, &draws, gpu, spec, counted, false).expect("room");
                let second = super::bound_before_as(&clamps, &draws, gpu, spec, counted, false).expect("room");
                assert!(first.0 == second.0 && second.1 == first.1 + bytes, "one buffer, in turn");
                draws.clamp_end(&clamps);
                assert!(super::bound_before_as(&clamps, &draws, gpu, spec, counted, false).is_none(), "the run has ended");
                clamps.close();
                draws.close();
                let run = issued();
                host.execute(&[&clamps, &draws]);
                let mut states: std::collections::HashMap<usize, u32> =
                    [key(&title), key(&counter), key(&first.0)].into_iter().map(|k| (k, 0)).collect();
                replay(&run, &mut states).unwrap();
                assert!(run.iter().all(|t| t.0 == key(&first.0)), "only the list's buffer is transitioned");
                assert_eq!(run.len(), 3, "to unordered access, indirect argument and common");
                assert!(states.values().all(|&s| s == 0), "all common again");
                let mut out = std::ptr::null_mut();
                first.0.Map(0, None, Some(&mut out)).unwrap();
                let out = out.cast::<u32>();
                let counts: Vec<u32> = (0..2 * records as usize).map(|i| out.add(i * per).read_volatile()).collect();
                match count {
                    None => assert_eq!(counts, [1 << 24, 1 << 24, 0, 0, 0, 0, 0, 0]),
                    // The words past the count stay as the fresh buffer's.
                    Some(_) => assert_eq!(counts, [1 << 24, 0, 0, 0, 1 << 24, 0, 0, 0]),
                }
                for at in 0..per * records as usize {
                    assert_eq!(data.cast::<u32>().add(at).read_volatile(), u32::MAX, "the title's copy as written");
                }
            }
        }
    }

    /// Runs `specs` as the hoisted clamps of one run, `batched` or each
    /// alone: the words of the drawing list's buffer after, laid out as
    /// `crate::batch::tests::laid_out` does (words no clamp wrote are
    /// `FRESH`), and the budget word. `None` without a host GPU.
    fn hoisted_run(specs: &[crate::batch::tests::Spec], batched: bool) -> Option<(Vec<u32>, u32)> {
        use crate::batch::tests::{FRESH, RECORD_BUFFERS, laid_out};
        let host = super::host()?;
        let (contents, places, words) = laid_out(specs);
        let buffers: Vec<_> = contents.iter().map(|c| host.shared_buffer(c.len() as u64 * 4).unwrap()).collect();
        let (clamps, draws) = (host.new_list().unwrap(), host.new_list().unwrap());
        let mut dest = None;
        // SAFETY: our mapped buffers, written before the lists run and read
        // after they have; live buffers.
        unsafe {
            for (c, (_, data)) in contents.iter().zip(&buffers) {
                std::ptr::copy_nonoverlapping(c.as_ptr(), data.cast::<u32>(), c.len());
            }
            for (s, &(from, count_at, to)) in specs.iter().zip(&places) {
                let base = buffers[s.source].0.GetGPUVirtualAddress();
                let counter = buffers[RECORD_BUFFERS + s.counter].0.GetGPUVirtualAddress();
                let counted = s.count.map(|_| (counter, counter + u64::from(count_at)));
                let spec = (s.kind, s.stride, s.records);
                let gpu = (base, base + u64::from(from));
                let (out, at) = super::bound_before_as(&clamps, &draws, gpu, spec, counted, batched).expect("room");
                assert_eq!(at, u64::from(to), "the drawing list's buffer is filled in turn");
                let first = dest.get_or_insert_with(|| {
                    let mut data = std::ptr::null_mut();
                    out.Map(0, None, Some(&mut data)).unwrap();
                    let data = data.cast::<u32>();
                    (0..words).for_each(|i| data.add(i).write_volatile(FRESH));
                    (out.clone(), data)
                });
                assert!(first.0 == out, "one buffer");
            }
            draws.clamp_end(&clamps);
            clamps.close();
            draws.close();
            host.execute(&[&clamps, &draws]);
            let (_, data) = dest?;
            Some(((0..words).map(|i| data.add(i).read_volatile()).collect(), clamps.budget_spent()))
        }
    }

    /// The host GPU's clamps are the model's (`crate::batch::model`, whose
    /// tests run everywhere), each alone and in batches: the same words
    /// written and no others, the same budget spent, over runs of random
    /// clamps of every kind, with and without counts, from five buffers
    /// (batches of at most four). So a run of clamps in batches gives what
    /// each alone gives.
    #[test]
    fn a_batch_clamps_as_each_alone_does() {
        use crate::batch::tests::{alone, generator, random};
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::batch().is_some(), "the batch program builds");
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        let (batches, mut spent) = (super::BATCHES.load(super::Ordering::Relaxed), 0);
        for trial in 0..40 {
            let specs = random(&mut next, 24);
            let model = alone(&specs, 0, super::widest());
            let each = hoisted_run(&specs, false).unwrap();
            assert_eq!(each, model, "trial {trial}: each clamp alone, and the model");
            let batched = hoisted_run(&specs, true).unwrap();
            assert_eq!(batched, model, "trial {trial}: in batches, and the model");
            spent += usize::from(model.1 == 1 << 27);
        }
        assert!(spent > 0, "some runs spend the whole budget");
        assert!(super::BATCHES.load(super::Ordering::Relaxed) > batches + 40, "batches ran, some runs in two");
    }

    /// The budget order on the host GPU: five executions that take more
    /// than a budget between them, in each of their 120 orders, cut the
    /// same records in batches as alone, the ones the model cuts (its test
    /// `batches_cut_the_same_records_in_every_order` has all the orders of
    /// six).
    #[test]
    fn batches_cut_the_same_records_in_every_order() {
        use crate::batch::tests::{Spec, alone, draws, orders};
        if super::host().is_none() {
            return; // no host GPU here
        }
        let executions = [
            draws(0, 3, 3 << 18, 40, 0),
            draws(1, 2, u32::MAX, u32::MAX, 1),
            Spec { count: Some(8), ..draws(0, 8, 300, 2, 2) },
            Spec { kind: 10, stride: 12, records: 6, words: [70_000, 1, 1].repeat(6), count: Some(4), source: 2, counter: 1 },
            draws(0, 3, 3 << 18, 40, 1),
        ];
        let mut outcomes = std::collections::HashSet::new();
        for order in orders(&executions) {
            let model = alone(&order, 0, super::widest());
            assert_eq!(model.1, 1 << 27, "they take more than a budget");
            assert_eq!(hoisted_run(&order, false).unwrap(), model);
            assert_eq!(hoisted_run(&order, true).unwrap(), model);
            outcomes.insert(model.0);
        }
        assert!(outcomes.len() > 10, "the order decides what is cut");
    }

    /// A title list cut for runs of indirect draws charges one budget in
    /// the title's order, as the one host list it would be without the cuts
    /// does: a dispatch clamped in the first list, a draw of two garbage
    /// records and one of small records hoisted into it, a dispatch clamped
    /// in the second list and a small draw hoisted into that. The dispatch
    /// and the first garbage record run and take the budget; everything
    /// after runs empty, in the lists after the cuts too. With a budget for
    /// each list (`chained` off: lists cut at a wait) the second list's
    /// executions run.
    #[test]
    fn a_list_cut_for_a_run_goes_on_with_the_budget_before_it() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let words = |b: &super::ID3D12Resource, at: u64, n: usize| -> Vec<u32> {
            let mut data = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours, read once its lists have run.
            unsafe {
                b.Map(0, None, Some(&mut data)).unwrap();
                (0..n).map(|i| data.cast::<u8>().add(at as usize).cast::<u32>().add(i * 4).read_volatile()).collect()
            }
        };
        let (title, data) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffer, written before the lists run.
        let gpu = unsafe {
            // A dispatch of 4 groups at 0; two garbage draws at 16; two
            // small draws at 48.
            let w = data.cast::<u32>();
            for (i, v) in [4, 1, 1, 0].into_iter().chain([u32::MAX; 8]).chain([3, 1, 0, 0, 3, 1, 0, 0]).enumerate() {
                w.add(i).write_volatile(v);
            }
            title.GetGPUVirtualAddress()
        };
        for (chained, batched) in [(true, false), (false, false), (true, true), (false, true)] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap(), host.new_list().unwrap()];
            if chained {
                lists[1].continue_budget(&lists[0]);
                lists[2].continue_budget(&lists[1]);
            }
            let a = super::bound(&lists[0], gpu, (2, 12, 1), None).unwrap();
            super::done(&lists[0], &a);
            let garbage = super::bound_before_as(&lists[0], &lists[1], (gpu, gpu + 16), (0, 16, 2), None, batched).unwrap();
            let small = super::bound_before_as(&lists[0], &lists[1], (gpu, gpu + 48), (0, 16, 2), None, batched).unwrap();
            lists[1].clamp_end(&lists[0]);
            lists[0].close();
            let b = super::bound(&lists[1], gpu, (2, 12, 1), None).unwrap();
            super::done(&lists[1], &b);
            let last = super::bound_before_as(&lists[1], &lists[2], (gpu, gpu + 48), (0, 16, 2), None, batched).unwrap();
            lists[2].clamp_end(&lists[1]);
            lists[1].close();
            lists[2].close();
            host.execute(&[&lists[0], &lists[1], &lists[2]]);
            assert_eq!(words(&a, 0, 1), [4], "the dispatch before the run");
            assert_eq!(words(&garbage.0, garbage.1, 2), [1 << 24, 0], "the budget ends in the first draw");
            assert_eq!(words(&small.0, small.1, 2), [0, 0], "the draw after it runs empty");
            if chained {
                assert_eq!(words(&b, 0, 1), [0], "and the dispatch after the cut");
                assert_eq!(words(&last.0, last.1, 2), [0, 0], "and the draw of the next run");
                assert_eq!(lists[2].budget_spent(), 1 << 27);
                assert_eq!(lists[0].budget_spent(), 1 << 27, "one budget");
            } else {
                assert_eq!(words(&b, 0, 1), [4], "a budget of its own");
                assert_eq!(words(&last.0, last.1, 2), [3, 3]);
                assert_eq!(lists[1].budget_spent(), 4 * 64 + 6);
            }
            // A list reset has its own budget again.
            lists[2].reset();
            let again = super::bound(&lists[2], gpu, (2, 12, 1), None).unwrap();
            super::done(&lists[2], &again);
            lists[2].close();
            host.execute(&[&lists[2]]);
            assert_eq!((words(&again, 0, 1), lists[2].budget_spent()), (vec![4], 4 * 64));
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// The walk stops at the count: of 128 garbage draws under a count of 3
    /// the first two run and the third runs empty, as they do without one,
    /// and the others are never read (the argument buffer's fresh words stay
    /// zero). A count above the records changes nothing; a count of zero
    /// walks none.
    #[test]
    fn the_walk_stops_at_the_count() {
        let Some(under) = clamped_under(0, &[u32::MAX], 128, 1, Some(3)) else {
            return; // no host GPU here
        };
        assert_eq!(under[..3], [1 << 24, 1 << 24, 0]);
        assert!(under[3..].iter().all(|&count| count == 0));
        let all = clamped(0, &[u32::MAX], 128, 1).unwrap();
        assert_eq!(under[..3], all[..3]);
        let above = clamped_under(0, &[u32::MAX], 128, 1, Some(1 << 20)).unwrap();
        assert_eq!(above, all);
        let none = clamped_under(0, &[u32::MAX], 128, 1, Some(0)).unwrap();
        assert!(none.iter().all(|&count| count == 0));
    }

    /// Only the records that run are charged to the budget: after a call
    /// under a count of 1 a second finds room for that record again, where
    /// the same draws with no count spent it all on records that never ran.
    #[test]
    fn records_past_the_count_are_not_charged() {
        let Some(twice) = clamped_under(0, &[u32::MAX], 128, 2, Some(1)) else {
            return; // no host GPU here
        };
        assert_eq!(twice[0], 1 << 24);
        let spent = clamped(0, &[u32::MAX], 128, 2).unwrap();
        assert!(spent.iter().all(|&count| count == 0));
    }

    /// The groups of one indirect dispatch of `groups` x 1 x 1 that run on
    /// the host, through the clamp when `clamped`, else straight from the
    /// record. Each group marks its own word. `None` without a host GPU.
    fn groups_that_run(groups: u32, clamped: bool) -> Option<usize> {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store(g.x * 4, 1);
}
"#;
        let host = host()?;
        let (root, pso) =
            crate::compute::pipeline(MARK, s!("marks"), 0, &[Param::Uav { register: 0 }]).unwrap();
        let (record, words) = host.shared_buffer(12).unwrap();
        let (marks, seen) = host.shared_buffer(u64::from(groups) * 4).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; our pipeline, signature and buffers.
        unsafe {
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
                .unwrap();
            let signature = signature.unwrap();
            words.cast::<[u32; 3]>().write_volatile([groups, 1, 1]);
            std::ptr::write_bytes(seen, 0, groups as usize * 4);
            let arguments = if clamped {
                bound(&list, record.GetGPUVirtualAddress(), (2, 12, 1), None).unwrap()
            } else {
                list.list.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                record.clone()
            };
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
            l.ExecuteIndirect(&signature, 1, &arguments, 0, None, 0);
            if clamped {
                done(&list, &arguments);
            } else {
                l.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let seen = seen.cast::<u32>();
            Some(
                (0..groups as usize)
                    .filter(|&i| seen.add(i).read_volatile() != 0)
                    .count(),
            )
        }
    }

    /// A tile pass of the title (67,440 groups, one class of its 8x4 tiles at
    /// 2560x1440) runs every group where the host runs dispatches that wide,
    /// and its first 65,535 elsewhere. The host runs the widest dispatch the
    /// clamp lets through, so a host that stops running them fails here.
    #[test]
    fn a_wide_tile_pass_runs_every_group_the_host_runs() {
        let Some(ran) = groups_that_run(67_440, true) else {
            return; // no host GPU here
        };
        let widest = super::widest();
        assert_eq!(ran, 67_440.min(widest) as usize);
        assert_eq!(
            groups_that_run(widest, false).unwrap(),
            widest as usize,
            "the host runs the widest dispatch"
        );
        assert_eq!(
            groups_that_run(widest, true).unwrap(),
            widest as usize,
            "the clamp lets it through"
        );
    }

    /// Dispatches wider than [`super::widest`] are cut to 65,535 groups in x
    /// as before, and one whose groups pass 2^20 in all runs empty, also
    /// where x times y would wrap 32 bits.
    #[test]
    fn dispatches_past_the_bounds_are_cut() {
        let widest = super::widest();
        let Some(past) = clamped(2, &[widest + 1, 1, 1], 1, 1) else {
            return; // no host GPU here
        };
        assert_eq!(past, [65_535]);
        assert_eq!(clamped(2, &[widest, 1, 1], 1, 1).unwrap(), [widest]);
        // Past 2^20 groups in all: 65,535 by 17, 2^10 by 2^10 by 2, and
        // 2^20 by 4,096, whose x * y wraps 32 bits to zero.
        assert_eq!(clamped(2, &[65_535, 17, 1], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 10, 1 << 10, 2], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 20, 4_096, 1], 1, 1).unwrap(), [0]);
    }
}
