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
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, host, transition};

const SOURCE: &str = include_str!("indirect-clamp.hlsl");

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
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; };
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
        uint3 g = min(load3(at), uint3(65535, 65535, 65535));
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
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

fn make_words() -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 4 },
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
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(count.is_some())];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

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

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
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
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
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
}
