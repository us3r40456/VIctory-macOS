//! Console wave64 compute programs translated for a host of 32-lane waves
//! (`lower::halves`): what reads within a 32-lane half runs exactly in two
//! host waves of 32; what reads across them passes between the host waves
//! through group-shared memory where every lane of the group runs it once,
//! and is refused with its reason elsewhere.
//!
//! The GPU test runs one program on every hardware adapter: in halves
//! everywhere (`WaveSize(32)`), and also as one wave64 (`WaveSize(64)`)
//! where the adapter has 64-lane waves (AMD), which is the console's own
//! form; the results must be the console's, computed here, in both.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{gfx10, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// Per thread `l` of a group of 64 (one console wave64), eight dwords at
/// `l * 32`: its thread id, `v_mbcnt` of all lanes (its lane), `l ^ 5`
/// through `ds_swizzle` (xor mode, within 32 lanes), `l ^ 9` through
/// `ds_bpermute_b32` (within the half), `0x777` written by `v_writelane` to
/// lane 40 only, whether the constant mask -4 (every lane but 0 and 1; lane
/// 32 is bit 0 of its high dword) has its lane, and `l ^ 1` through DPP
/// (`quad_perm:[1,0,3,2]`).
fn in_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0xD8D4_141F,
        0x0200_0000, // ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM xor 5)
        0x3A08_0089, // v_xor_b32 v4, 9, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xDACC_0000,
        0x0300_0004, // ds_bpermute_b32 v3, v4, v0
        0xBE88_03FF,
        0x0000_0777, // s_mov_b32 s8, 0x777
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD761_0005,
        0x0001_5008, // v_writelane_b32 v5, s8, 40
        0xBE94_04C4, // s_mov_b64 s[20:21], -4
        0xD501_0006,
        0x0051_0280, // v_cndmask_b32 v6, 0, 1, s[20:21]
        0x7E0E_02FA,
        0xFF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] (DPP)
        S_WAITCNT_LGKM,
    ]);
    for (vdata, slot) in [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`in_halves`]'s lane `l`.
fn console(l: u32) -> [u32; 8] {
    [
        l,
        l,
        l ^ 5,
        l ^ 9,
        if l == 40 { 0x777 } else { 0 },
        u32::from(l >= 2),
        l ^ 1,
        0,
    ]
}

fn native(words: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            // One user-data SGPR; LDS in blocks of 512 bytes (bits 15-23).
            rsrc2: 1 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lowered(
    native: &NativeProgram,
    host_wave32: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    lowered_with(native, host_wave32, false)
}

/// [`lowered`], with LDS synced for narrow host waves where `narrow_waves`
/// (`emit::Options::narrow_waves`, D3DMetal).
fn lowered_with(
    native: &NativeProgram,
    host_wave32: bool,
    narrow_waves: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves,
        halves_local: Default::default(),
        mac: Default::default(),
        host_wave32,
    })
}

fn shader(l: lower::Lowered) -> super::HostShader {
    super::HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: Vec::new(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_srv_bytes: l.root_srv_bytes,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    }
}

fn refusal(words: &[u32], threads: u32, lds_bytes: u32) -> String {
    match lowered(&native(words, true, threads, lds_bytes), true) {
        Ok(_) => "translated".into(),
        Err(e) => e.to_string(),
    }
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            in_halves(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_mbcnt_hi_u32_b32 v1, -1, v1",
                "ds_swizzle_b32 v2, v0",
                "v_xor_b32 v4, 9, v0",
                "ds_bpermute_b32 v3, v4, v0",
                "v_writelane_b32 v5, s8, 40",
                "v_cndmask_b32 v6, 0, 1, s[20:21]",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2]",
                "buffer_store_dword v7, v16, s[40:43], 0 offen offset:24",
            ],
        ),
        (
            dpp_row_mask(),
            &[
                "v_mov_b32 v7, -1",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc",
                "buffer_store_dword v7, v16, s[40:43], 0 offen",
            ],
        ),
        (
            first_lane_of_the_thread_id(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_readfirstlane_b32 s12, v0",
                "v_mov_b32 v2, s12",
            ],
        ),
        (
            first_lane_of_a_buffer_load(3),
            &[
                "s_load_dwordx4 s[44:47], s[4:5], 0x000020",
                "v_mov_b32 v3, 4",
                "buffer_load_dword v4, v3, s[44:47], 0 offen",
                "s_waitcnt vmcnt(0)",
                "v_readfirstlane_b32 s12, v4",
                "v_mov_b32 v5, s12",
            ],
        ),
        (
            first_lane_alike(),
            &[
                "v_lshrrev_b32 v2, 6, v0",
                "v_readfirstlane_b32 s12, v2",
                "v_mov_b32 v3, s12",
                "buffer_store_dword v1, v16, s[40:43], 0 offen offset:4",
            ],
        ),
        (
            first_active_lane(),
            &["s_ff1_i32_b64 s12, exec", "v_mov_b32 v1, s12"],
        ),
        (
            lanes_below(100),
            &[
                "v_cmp_gt_u32 vcc, lit(0x00000064), v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_mov_b32 v2, vcc_hi",
                "v_mbcnt_lo_u32_b32 v3, vcc_lo, 0",
                "v_mbcnt_hi_u32_b32 v3, vcc_hi, v3",
                "buffer_store_dword v3, v16, s[40:43], 0 offen offset:8",
            ],
        ),
        (
            lanes_below_in_a_loop(),
            &[
                "s_mov_b32 s20, 0",
                "v_cmp_gt_u32 vcc, 40, v0",
                "s_add_u32 s20, s20, 1",
                "s_cmp_lt_u32 s20, 2",
                "s_cbranch_scc1",
            ],
        ),
        (
            lane_reads(40, 33),
            &[
                "v_readlane_b32 s12, v0, 40",
                "v_readlane_b32 s14, v0, s13",
                "v_mov_b32 v1, s12",
                "v_mov_b32 v2, s14",
                "buffer_store_dword v2, v16, s[40:43], 0 offen offset:4",
            ],
        ),
        (
            reduction(V_ADD_F32, 0x80, 0x7E06_0D00),
            &[
                "v_cvt_f32_u32 v3, v0",
                "s_or_saveexec_b64 s[12:13], -1",
                "v_cndmask_b32 v1, 0, v3, s[12:13]",
                "v_permlanex16_b32 v2, v1, 0, 0",
                "v_add_f32 v1, v2, v1",
                "v_readlane_b32 s14, v1, 32",
                "v_add_f32 v1, s14, v1",
                "v_readlane_b32 s15, v1, 0",
                "s_mov_b64 exec, s[12:13]",
                "v_mov_b32 v1, s15",
            ],
        ),
        (
            reduction(V_MIN_U32, 0xC1, V_MOV_V3_V0),
            &[
                "v_mov_b32 v3, v0",
                "v_cndmask_b32 v1, -1, v3, s[12:13]",
                "v_min_u32 v1, v2, v1",
                "v_min_u32 v1, s14, v1",
            ],
        ),
        (
            reduction(V_MAX_U32, 0x80, V_MOV_V3_V0),
            &["v_max_u32 v1, v2, v1", "v_max_u32 v1, s14, v1"],
        ),
        (
            exclusive_prefix(),
            &[
                "s_or_saveexec_b64 s[0:1], -1",
                "v_cndmask_b32 v1, 0, v0, s[0:1]",
                "v_add_nc_u32 v1, v1, v1 row_shr:8",
                "v_readlane_b32 s2, v1, 47",
                "v_mov_b32 v3, v1 row_shr:1",
                "s_mov_b64 exec, -2",
                "v_mov_b32 v2, v3",
                "s_mov_b64 exec, s[0:1]",
                "buffer_store_dword v2, v16, s[40:43], 0 offen",
            ],
        ),
        (
            in_a_loop(&[0xD760_000C, 0x0001_5100], 12, Count::Two),
            &[
                "v_readlane_b32 s12, v0, 40",
                "s_add_u32 s20, s20, 1",
                "s_cmp_lt_u32 s20, 2",
                "s_cbranch_scc1",
                "v_mov_b32 v1, s12",
            ],
        ),
        (
            in_a_loop(&[0x7D88_00A8, 0xBE8C_106A], 12, Count::FirstLane),
            &[
                "v_readfirstlane_b32 s21, v0",
                "v_cmp_gt_u32 vcc, 40, v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "s_cmp_lt_u32 s20, s21",
                "s_cbranch_scc1",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// In halves the program declares waves of 32; without, the console's 64.
#[test]
fn a_wave64_program_reading_lanes_declares_32_lane_waves_in_halves() {
    let n = native(&in_halves(), true, 64, 0);
    let halves = lowered(&n, true).expect("halves");
    let whole = lowered(&n, false).expect("one wave64");
    assert_eq!(halves.wave_size, Some(32));
    assert_eq!(whole.wave_size, Some(64));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut halves.dxil.clone())
            .expect("DXIL validation failed");
    }
    // A wave32 program is the console's either way.
    let n32 = native(&in_halves(), false, 32, 0);
    assert_eq!(lowered(&n32, true).expect("wave32").wave_size, Some(32));
}

/// `s_ff1_i32_b64 s12, exec` at the start: the console wave's first active
/// lane, 0 in every lane.
fn first_active_lane() -> Vec<u32> {
    [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xBE8C_147E, // s_ff1_i32_b64 s12, exec
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat()
}

/// The lanes below `limit` (`v_cmp_gt_u32 vcc, limit, v0`) as data: per
/// lane, three dwords at `l * 32`: the console wave's count of them
/// (`s_bcnt1_i32_b64`), their high dword (`vcc_hi`, lanes 32-63), and how
/// many are below the lane (`v_mbcnt` of the mask, as a compaction counts
/// its place).
fn lanes_below(limit: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_00FF,
        limit, // v_cmp_gt_u32 vcc, limit, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
        0xD765_0003,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v3, vcc_lo, 0
        0xD766_0003,
        0x0002_066B, // v_mbcnt_hi_u32_b32 v3, vcc_hi, v3
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.extend(store(3, 8));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lanes_below`]`(limit)`'s lane `l` in a group of
/// `threads`.
fn console_lanes_below(limit: u32, threads: u32, l: u32) -> [u32; 3] {
    let wave = l & !63;
    let below = |end: u32| limit.min(threads).min(end).saturating_sub(wave);
    let high = below(wave + 64).saturating_sub(32);
    [
        below(wave + 64),
        if high == 32 { u32::MAX } else { (1 << high) - 1 },
        below(l),
    ]
}

/// The lanes below 40, as in [`lanes_below`], counted in a scalar loop
/// that runs twice: the mask is written in a block each lane runs more than
/// once, as often as every other lane of the group.
fn lanes_below_in_a_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFFB, // s_cbranch_scc1 (back to the v_cmp)
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Per thread `l`, two dwords at `l * 32`: `v_readlane_b32` of the thread
/// id at console lane `k` (a constant) and at lane `s` (in an SGPR).
fn lane_reads(k: u32, s: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD760_000C,
        0x0001_0100 | (128 + k) << 9, // v_readlane_b32 s12, v0, k
        0xBE8D_03FF,
        s, // s_mov_b32 s13, s
        0xD760_000E,
        0x0000_1B00, // v_readlane_b32 s14, v0, s13
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_020E, // v_mov_b32 v2, s14
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// How many passes [`in_a_loop`] makes.
#[derive(Clone, Copy)]
enum Count {
    /// Two: a count every host wave computes alike.
    Two,
    /// Up to the thread id of the host wave's first lane
    /// (`v_readfirstlane_b32`), at least one: a count the host waves need
    /// not share.
    FirstLane,
}

/// `body` in a scalar loop (so each lane runs its block more than once),
/// then the scalar `s<result>` stored by every lane.
fn in_a_loop(body: &[u32], result: u32, count: Count) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    let compare = match count {
        Count::Two => 0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        Count::FirstLane => {
            w.push(0x7E2A_0500); // v_readfirstlane_b32 s21, v0
            0xBF0A_1514 // s_cmp_lt_u32 s20, s21
        }
    };
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        compare,                              // s_cmp_lt_u32 s20, <count>
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the body)
        0x7E02_0200 | result,                 // v_mov_b32 v1, s<result>
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// VOP2 opcodes of the operations [`reduction`] takes.
const V_ADD_F32: u32 = 0x03;
const V_MIN_U32: u32 = 0x13;
const V_MAX_U32: u32 = 0x14;
/// `v_mov_b32 v3, v0`.
const V_MOV_V3_V0: u32 = 0x7E06_0300;

/// The console compiler's whole-wave reduction of `v3` (`input`, from the
/// thread id) by the VOP2 operation `op`, with the inline constant `ident`
/// in inactive lanes: within rows of 16 by DPP, across rows by
/// `v_permlanex16`, across the halves by `v_readlane` of lane 32, and read
/// back from lane 0. Per thread, one dword at `l * 32`: its console wave's
/// result.
fn reduction(op: u32, ident: u32, input: u32) -> Vec<u32> {
    let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        input,       // v3 from v0
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
    ]);
    for control in [0x161, 0x162, 0x164, 0x168] {
        w.extend([
            vop2(0x0602_02FA),
            0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
        ]);
    }
    w.extend([
        0xD778_0002,
        0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
        vop2(0x0602_0302), // op v1, v2, v1
        0xD760_000E,
        0x0001_4101,       // v_readlane_b32 s14, v1, 32
        vop2(0x0602_020E), // op v1, s14, v1
        0xD760_000F,
        0x0001_0101, // v_readlane_b32 s15, v1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_020F, // v_mov_b32 v1, s15
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`reduction`] of `f` (folding the inputs of a
/// console wave's lanes) in lane `l` of a group of `threads`.
fn console_reduction(
    threads: u32,
    l: u32,
    f: impl Fn(u32, u32) -> u32,
    input: impl Fn(u32) -> u32,
) -> u32 {
    let wave = l & !63;
    (wave + 1..(wave + 64).min(threads)).fold(input(wave), |a, t| f(a, input(t)))
}

/// The console compiler's exclusive prefix sum of the thread id over the
/// wave64 (`WavePrefixSum`): inclusive within rows of 16 by DPP, rows
/// joined by `v_readlane` of each row's last lane under EXEC of the rows
/// after it, shifted one lane on (the first lane of each row from the
/// previous row's last), lane 0 the identity. One dword per thread at
/// `l * 32`.
fn exclusive_prefix() -> Vec<u32> {
    const EXEC: u32 = 126;
    let sop1 = |sdst: u32, op: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
    let vop1 = |vdst: u32, src0: u32| 0x7E00_0200 | vdst << 17 | src0;
    let add = |src0: u32| 0x4A02_0200 | src0; // v_add_nc_u32 v1, src0, v1
    let dpp = |ctrl: u32| 0xFF00_0001 | ctrl << 8; // of v1, row and bank masks 0xf
    let readlane = |lane: u32| [0xD760_0002, 0x0000_0101 | (128 + lane) << 9]; // s2, v1
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,        // v_lshlrev_b32 v16, 5, v0
        sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
    ]);
    for shift in [1, 2, 4, 8] {
        w.extend([add(0xFA), dpp(0x110 + shift)]); // v_add_nc_u32 v1, v1, v1 row_shr
    }
    // Each row's sum so far into the rows after it.
    for (last, exec_lo, exec_hi) in [
        (15, 0xFFFF_0000, 0),
        (31, 0, 0x0000_FFFF),
        (47, 0, 0xFFFF_0000),
    ] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(add(2)); // v_add_nc_u32 v1, s2, v1
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(3, 257),          // v_mov_b32 v3, v1
        vop1(3, 0xFA),
        dpp(0x111), // v_mov_b32 v3, v1 row_shr:1
    ]);
    // The first lane of each row from the last of the row before.
    for (last, exec_lo, exec_hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(vop1(3, 2)); // v_mov_b32 v3, s2
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(2, 128),          // v_mov_b32 v2, 0
        sop1(EXEC, 0x04, 194), // s_mov_b64 exec, -2
        vop1(2, 259),          // v_mov_b32 v2, v3
        sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// A lane mask read as data in a group of 33 to 64 lanes or more holds
/// lanes of both host waves: they pass their halves to each other where
/// every lane of the group runs the block writing it together (once, or in
/// a loop on a count every host wave computes alike), and it is refused in
/// a loop counted to a lane's value. In a group of 32 or fewer, one host
/// wave holds the whole mask.
#[test]
fn lane_masks_read_as_data_pass_between_halves() {
    assert_eq!(refusal(&first_active_lane(), 32, 0), "translated");
    assert_eq!(refusal(&first_active_lane(), 64, 0), "translated");
    for threads in [64, 96, 128] {
        assert_eq!(refusal(&lanes_below(40), threads, 0), "translated");
        assert_eq!(refusal(&lanes_below_in_a_loop(), threads, 0), "translated");
    }
    // v_cmp_gt_u32 vcc, 40, v0; s_bcnt1_i32_b64 s12, vcc
    let counted = in_a_loop(&[0x7D88_00A8, 0xBE8C_106A], 12, Count::FirstLane);
    let why = refusal(&counted, 64, 0);
    assert!(why.contains("a lane mask read as data (wave64 on a 32-lane host)"), "{why}");
    assert_eq!(refusal(&counted, 32, 0), "translated");
    // For 64-lane hosts it is read in one wave, as before.
    assert!(lowered(&native(&counted, true, 64, 0), false).is_ok());

    let halves = lowered(&native(&lanes_below(40), true, 64, 0), true).expect("halves");
    assert_eq!(halves.wave_size, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut halves.dxil.clone())
            .expect("DXIL validation failed");
    }
}

/// [`lanes_below`] and [`first_active_lane`] on every hardware adapter give
/// the console's masks: in one console wave of 64 (both halves' lanes), in
/// two (one mask each), and in a group of 96, whose second console wave has
/// no upper half (a high dword of 0).
#[test]
fn lane_masks_passed_between_halves_match_the_console_on_every_gpu() {
    let mut failed = differences(&first_active_lane(), 64, 1, &[0; 64]);
    for (limit, threads) in [(40, 64), (100, 128), (80, 96), (96, 96)] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| console_lanes_below(limit, threads, l))
            .collect();
        failed.extend(differences(&lanes_below(limit), threads, 3, &want));
    }
    // Counted in a loop: the count of lanes below 40, each pass.
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| console_lanes_below(40, threads, l)[0])
            .collect();
        failed.extend(differences(&lanes_below_in_a_loop(), threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Whole-wave reductions and prefix sums pass between the halves where
/// every lane of the group runs them together (once, or in a loop on a
/// count every host wave computes alike); in a group of 32 or fewer, one
/// host wave holds the whole console wave.
#[test]
fn reductions_and_prefixes_pass_between_halves() {
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D00);
    for threads in [32, 64, 96, 128] {
        assert_eq!(refusal(&sum, threads, 0), "translated");
        assert_eq!(refusal(&exclusive_prefix(), threads, 0), "translated");
    }
    let body = &sum[PROLOGUE.len() + 1..sum.len() - 3];
    for threads in [64, 96, 128] {
        assert_eq!(refusal(&in_a_loop(body, 15, Count::Two), threads, 0), "translated");
    }
    let looped = in_a_loop(body, 15, Count::FirstLane);
    let why = refusal(&looped, 64, 0);
    assert!(
        why.contains("a wave reduction over both 32-lane halves"),
        "{why}"
    );
    assert_eq!(refusal(&looped, 32, 0), "translated");

    for words in [sum, exclusive_prefix()] {
        let halves = lowered(&native(&words, true, 96, 0), true).expect("halves");
        assert_eq!(halves.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut halves.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

/// [`reduction`]s and [`exclusive_prefix`] on every hardware adapter give
/// the console's results: in a group of 32 (one host wave), of 64 (one
/// console wave in two halves), of 128 (two), and of 96, whose second
/// console wave has no upper half.
#[test]
fn reductions_and_prefixes_across_halves_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [32, 64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                let sum = console_reduction(threads, l, |a, b| a + b, |t| t);
                (sum as f32).to_bits()
            })
            .collect();
        failed.extend(differences(
            &reduction(V_ADD_F32, 0x80, 0x7E06_0D00),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, u32::min, |t| t ^ 0x2A))
            .collect();
        let xor = 0x3A06_00AA; // v_xor_b32 v3, 42, v0
        failed.extend(differences(
            &reduction(V_MIN_U32, 0xC1, xor),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, u32::max, |t| t ^ 0x2A))
            .collect();
        failed.extend(differences(
            &reduction(V_MAX_U32, 0x80, xor),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                let (wave, k) = (l & !63, l & 63);
                wave * k + k * k.saturating_sub(1) / 2
            })
            .collect();
        failed.extend(differences(&exclusive_prefix(), threads, 1, &want));
    }
    // The sum in a loop: the same each pass.
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D00);
    let looped = in_a_loop(&sum[PROLOGUE.len() + 1..sum.len() - 3], 15, Count::Two);
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t) as f32).to_bits())
            .collect();
        failed.extend(differences(&looped, threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Lane reads and first-lane reads that may be in the other half pass
/// between the halves where every lane of the group runs them together
/// (once, or in a loop on a count every host wave computes alike), and are
/// refused with a reason naming the host's waves in a loop counted to a
/// lane's value; the same programs translate for 64-lane hosts.
#[test]
fn lane_reads_pass_between_halves() {
    for threads in [32, 64, 96, 128] {
        assert_eq!(refusal(&lane_reads(40, 33), threads, 0), "translated");
        assert_eq!(
            refusal(&first_lane_of_the_thread_id(), threads, 0),
            "translated"
        );
    }
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(0), 64, 0),
        "translated"
    );

    let readlane = [0xD760_000C, 0x0001_5100]; // v_readlane_b32 s12, v0, 40
    let first = [0x7E18_0500]; // v_readfirstlane_b32 s12, v0
    for threads in [64, 96, 128] {
        for body in [&readlane[..], &first] {
            let words = in_a_loop(body, 12, Count::Two);
            assert_eq!(refusal(&words, threads, 0), "translated");
        }
    }

    let readlane = in_a_loop(&readlane, 12, Count::FirstLane);
    let why = refusal(&readlane, 64, 0);
    assert!(
        why.contains("a lane read that may cross 32-lane halves (wave64 on a 32-lane host)"),
        "{why}"
    );
    assert_eq!(refusal(&readlane, 32, 0), "translated");
    assert!(lowered(&native(&readlane, true, 64, 0), false).is_ok());

    let first = in_a_loop(&first, 12, Count::FirstLane);
    let why = refusal(&first, 64, 0);
    assert!(
        why.contains("a first-lane read of a value that may differ between 32-lane halves"),
        "{why}"
    );
    assert_eq!(refusal(&first, 32, 0), "translated");
    assert!(lowered(&native(&first, true, 64, 0), false).is_ok());

    // A first-lane read alone needs no console lanes: the host wave's
    // first lane, in halves or not.
    let first_alone = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x7E18_0500, // v_readfirstlane_b32 s12, v0
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    assert_eq!(refusal(&first_alone, 64, 0), "translated");
    // Nor does a value the same in every lane: a buffer dword loaded at an
    // offset the same in every lane, through a scalar descriptor.
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(3), 64, 0),
        "translated"
    );

    for words in [lane_reads(20, 5), first_lane_of_the_thread_id()] {
        let halves = lowered(&native(&words, true, 96, 0), true).expect("halves");
        assert_eq!(halves.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut halves.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

/// [`lane_reads`] and [`first_lane_of_the_thread_id`] on every hardware
/// adapter give the console's lanes: in a group of one console wave, of
/// two, of 96 (the second console wave has no upper half), and of 32.
#[test]
fn lane_reads_across_halves_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for (k, s, threads) in [
        (40, 33, 64),
        (40, 63, 128),
        (7, 50, 128),
        (20, 5, 96),
        (20, 5, 32),
    ] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| [(l & !63) + k, (l & !63) + s])
            .collect();
        failed.extend(differences(&lane_reads(k, s), threads, 2, &want));
    }
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads).map(|l| l & !63).collect();
        failed.extend(differences(
            &first_lane_of_the_thread_id(),
            threads,
            1,
            &want,
        ));
        // In a loop, the same each pass.
        let first = in_a_loop(&[0x7E18_0500], 12, Count::Two); // v_readfirstlane_b32 s12, v0
        failed.extend(differences(&first, threads, 1, &want));
    }
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).map(|l| (l & !63) + 40).collect();
        // v_readlane_b32 s12, v0, 40
        let readlane = in_a_loop(&[0xD760_000C, 0x0001_5100], 12, Count::Two);
        failed.extend(differences(&readlane, threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// LDS shared by a group that is one console wave64 (no barriers between
/// its lanes) cannot be split into two host waves; one of 32 threads or
/// fewer is one host wave, and a group of several console waves keeps the
/// console's barriers.
#[test]
fn lds_of_a_one_wave64_group_is_refused_in_halves() {
    let lds = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0xD834_0000,
            0x0000_0102, // ds_write_b32 v2, v1
            S_WAITCNT_LGKM,
            0xD8D8_0000,
            0x0300_0002, // ds_read_b32 v3, v2
            S_WAITCNT_LGKM,
        ],
        &store(3, 0),
        &[S_ENDPGM],
    ]
    .concat();
    let why = refusal(&lds, 64, 256);
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
    assert_eq!(refusal(&lds, 32, 256), "translated");
    assert_eq!(refusal(&lds, 128, 512), "translated");
}

/// Runs `words`, one group of `threads` of a wave64 program whose lanes store
/// `per_lane` dwords at `l * 32`, on every hardware adapter: in halves, and
/// as one wave64 where the adapter runs 64-lane waves. A line per lane whose
/// dwords differ from `want` (the console's, `per_lane` per lane).
fn differences(words: &[u32], threads: u32, per_lane: usize, want: &[u32]) -> Vec<String> {
    differences_with(words, threads, per_lane, want, 0, false)
}

/// [`differences`] for a program with `lds_bytes` of LDS, lowered with
/// `narrow_waves` ([`lowered_with`]).
fn differences_with(
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
    lds_bytes: u32,
    narrow_waves: bool,
) -> Vec<String> {
    differences_in(
        true,
        words,
        threads,
        per_lane,
        want,
        lds_bytes,
        narrow_waves,
    )
}

/// [`differences_with`] for a program of console waves of 64 lanes
/// (`wave64`) or of 32, which a host of any wave size runs as it is.
fn differences_in(
    wave64: bool,
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
    lds_bytes: u32,
    narrow_waves: bool,
) -> Vec<String> {
    let n = native(words, wave64, threads, lds_bytes);
    let forms = if wave64 {
        vec![
            (
                "halves",
                shader(lowered_with(&n, true, narrow_waves).expect("halves")),
            ),
            (
                "wave64",
                shader(lowered_with(&n, false, narrow_waves).expect("one wave64")),
            ),
        ]
    } else {
        vec![(
            "wave32",
            shader(lowered_with(&n, false, narrow_waves).expect("one wave32")),
        )]
    };
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(threads * 8),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
    }
    let mut failed = Vec::new();
    for (adapter, host) in &hosts {
        let wave32 = host.wave32().unwrap_or(false);
        for (form, shader) in &forms {
            let label = format!("{adapter} ({form})");
            // A host of 32-lane waves only ever runs the halves. NVIDIA
            // refuses the console's own form; D3DMetal makes its pipeline
            // and then runs nothing.
            if *form == "wave64" && wave32 {
                eprintln!("{label}: one wave64 on a host of 32-lane waves; skipped");
                continue;
            }
            let Ok((root, pso)) = host.compute_pipeline(shader) else {
                panic!("{label}: no pipeline");
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let wrong: Vec<String> = out[0]
                .chunks(8)
                .zip(want.chunks(per_lane))
                .enumerate()
                .filter(|(_, (got, want))| got[..per_lane] != **want)
                .map(|(l, (got, want))| {
                    format!(
                        "{label}: lane {l}: {:?}, the console {want:?}",
                        &got[..per_lane]
                    )
                })
                .collect();
            eprintln!(
                "{label}: {} lanes of {threads} differ from the console",
                wrong.len()
            );
            failed.extend(wrong);
        }
    }
    failed
}

/// A DPP row mask names the console's rows 0-3: with `row_mask:0xc` only
/// rows 2 and 3 (lanes 32-63) take `l ^ 1`, rows 0 and 1 keep -1.
fn dpp_row_mask() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0E_02C1, // v_mov_b32 v7, -1
        0x7E0E_02FA,
        0xCF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc (DPP)
    ]);
    w.extend(store(7, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn dpp_row_masks_name_the_console_rows_on_every_gpu() {
    let want: Vec<u32> = (0..64)
        .map(|l| if l >= 32 { l ^ 1 } else { u32::MAX })
        .collect();
    let failed = differences(&dpp_row_mask(), 64, 1, &want);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_readfirstlane_b32` of the thread id, with console lanes (`v_mbcnt`):
/// the console's first lane is lane 0 of the wave.
fn first_lane_of_the_thread_id() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readfirstlane_b32` of a dword loaded from the buffer at table dword 8,
/// at the byte offset in `v<offset>`: `v3` holds 4 in every lane, `v0` the
/// thread id.
fn first_lane_of_a_buffer_load(offset: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E06_0284, // v_mov_b32 v3, 4
        0xE030_1000,
        0x800B_0400 | offset, // buffer_load_dword v4, v<offset>, s[44:47], 0 offen
        0xBF8C_3F70,         // s_waitcnt vmcnt(0)
        0x7E18_0504,         // v_readfirstlane_b32 s12, v4
        0x7E0A_020C,         // v_mov_b32 v5, s12
    ]);
    w.extend(store(5, 0));
    w.push(S_ENDPGM);
    w
}

/// Per thread `l` of a group of 128 (two console wave64s), two dwords at
/// `l * 32`: the first lane of `l >> 6` (one value per console wave, the
/// same in both its halves) and the lane `v_mbcnt` counts.
fn first_lane_alike() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E18_0502, // v_readfirstlane_b32 s12, v2
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(1, 4));
    w.push(S_ENDPGM);
    w
}

/// First-lane reads kept in halves give the console's value on every
/// hardware adapter: the thread id's in a group of 32 (one host wave holds
/// the whole console wave), and one value per console wave in a group of
/// 128, where each half reads its console wave's.
#[test]
fn first_lanes_kept_in_halves_match_the_console_on_every_gpu() {
    let mut failed = differences(&first_lane_of_the_thread_id(), 32, 1, &[0; 32]);
    let want: Vec<u32> = (0..128).flat_map(|l| [l >> 6, l & 63]).collect();
    failed.extend(differences(&first_lane_alike(), 128, 2, &want));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`in_halves`] on every hardware adapter: in halves, and as one wave64
/// where the adapter runs 64-lane waves; both give the console's results.
#[test]
fn wave64_programs_in_halves_match_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64).flat_map(console).collect();
    let failed = differences(&in_halves(), 64, 8, &want);
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
}

/// Each lane of a one-wave group of 64 writes its `v_mbcnt` count to LDS and
/// reads the dword of the lane 32 away, in the other half: the console's
/// lockstep orders the two without a barrier.
fn lds_across_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_lds_program_is_what_its_comments_say() {
    let code: Vec<u8> = lds_across_halves()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "ds_write_b32 v2, v1",
        "v_xor_b32 v4, 32, v0",
        "v_lshlrev_b32 v4, 2, v4",
        "ds_read_b32 v3, v4",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// With narrow host waves (#148, D3DMetal), a one-wave group whose LDS gets
/// its barrier runs in halves instead of being refused; without them it is
/// refused, as before.
#[test]
fn lds_of_a_one_wave64_group_synced_by_barriers_runs_in_halves() {
    let n = native(&lds_across_halves(), true, 64, 256);
    let synced = lowered_with(&n, true, true).expect("synced, in halves");
    assert_eq!(synced.wave_size, Some(32));
    assert_eq!(synced.split_barriers, 1);
    let why = lowered_with(&n, true, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
}

/// The synced program on every hardware adapter, in halves: each lane reads
/// the count the lane in the other half wrote (32 for the low half, the
/// lane less 32 for the high one).
#[test]
fn lds_synced_in_halves_matches_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64u32).map(|l| (l ^ 32).min(32)).collect();
    let failed = differences_with(&lds_across_halves(), 64, 1, &want, 256, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Per thread `l` of a one-wave group of 64: four passes of a loop on a
/// scalar count, each writing the lane's sum to LDS and adding the sum of
/// the lane 32 away, in the other half (`v_mbcnt` asks for the console's
/// lanes). The console's lockstep orders the accesses; split, each pass
/// needs a barrier before its read, and one before its write, which the
/// other half's read of the last pass must not miss.
fn lds_in_a_loop() -> Vec<u32> {
    let body = [
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
    ];
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0005,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E02_0300, // v_mov_b32 v1, v0
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        0xBF0A_8414,                          // s_cmp_lt_u32 s20, 4
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the write)
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lds_in_a_loop`]: each pass, a lane's sum and
/// that of the lane 32 away both become their total.
fn console_lds_in_a_loop() -> Vec<u32> {
    let mut sum: Vec<u32> = (0..64).collect();
    for _ in 0..4 {
        sum = (0..64).map(|l| sum[l] + sum[l ^ 32]).collect();
    }
    sum
}

#[test]
fn the_lds_loop_is_what_its_comments_say() {
    let code: Vec<u8> = lds_in_a_loop()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_mbcnt_lo_u32_b32 v5, -1, 0",
        "v_xor_b32 v4, 32, v0",
        "v_mov_b32 v1, v0",
        "s_mov_b32 s20, 0",
        "ds_write_b32 v2, v1",
        "ds_read_b32 v3, v4",
        "v_add_nc_u32 v1, v1, v3",
        "s_add_u32 s20, s20, 1",
        "s_cmp_lt_u32 s20, 4",
        "s_cbranch_scc1",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// With narrow host waves (D3DMetal), LDS a one-wave group shares in a loop
/// on a scalar count gets its barriers inside the loop and runs in halves;
/// without them it is refused, as before.
#[test]
fn lds_of_a_one_wave64_group_in_a_loop_runs_in_halves() {
    let n = native(&lds_in_a_loop(), true, 64, 256);
    let synced = lowered_with(&n, true, true).expect("synced, in halves");
    assert_eq!(synced.wave_size, Some(32));
    assert_eq!(synced.split_barriers, 2);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut synced.dxil.clone())
            .expect("DXIL validation failed");
    }
    let why = lowered_with(&n, true, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
}

/// The loop on every hardware adapter, in halves: every pass gives the
/// console's sums.
#[test]
fn lds_in_a_loop_in_halves_matches_the_console_on_every_gpu() {
    let failed = differences_with(&lds_in_a_loop(), 64, 1, &console_lds_in_a_loop(), 256, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_permlanex16_b32 v1, v0, s8, s9` with the FI and BOUND_CTRL bits
/// (`op_sel` bits 0 and 1, word bits 11 and 12): lane `l` takes the value
/// of `v0` from the other row of 16 in its 32-lane half, the lane its
/// nibble of `s8` (lanes 0 to 7 of a row) or `s9` (lanes 8 to 15) names.
const fn permlanex16(fetch_inactive: bool, bound_ctrl: bool) -> [u32; 2] {
    [
        0xD778_0001 | (fetch_inactive as u32) << 11 | (bound_ctrl as u32) << 12,
        0x0024_1100,
    ]
}

/// The selectors: every nibble of `s8` is odd and every one of `s9` is even.
const PERM_LOW: u32 = 0x1357_9BDF;
const PERM_HIGH: u32 = 0x2468_ACE0;
/// What `v1` holds before an instruction that may leave it alone.
const PERM_OLD: u32 = 0xAAAA;

/// Per lane `l` of one console wave of `lanes`: `v0` (the lane's number)
/// permuted with every lane active, then with only the even lanes active
/// for FI clear and BOUND_CTRL clear, FI clear and BOUND_CTRL set, FI set
/// and BOUND_CTRL clear, and FI and BOUND_CTRL both set, each over a `v1`
/// holding [`PERM_OLD`]. EXEC is whole again for every store.
fn permlanex16_program(lanes: u32) -> Vec<u32> {
    let wave64 = lanes == 64;
    let full = |w: &mut Vec<u32>| {
        w.push(if wave64 {
            0xBEFE_04C1 // s_mov_b64 exec, -1
        } else {
            0xBEFE_03C1 // s_mov_b32 exec_lo, -1
        });
    };
    let even = |w: &mut Vec<u32>| {
        w.push(if wave64 {
            0xBEFE_040A // s_mov_b64 exec, s[10:11]
        } else {
            0xBEFE_030A // s_mov_b32 exec_lo, s10
        });
    };
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE88_03FF,
        PERM_LOW, // s_mov_b32 s8, PERM_LOW
        0xBE89_03FF,
        PERM_HIGH, // s_mov_b32 s9, PERM_HIGH
        0xBE8A_03FF,
        0x5555_5555, // s_mov_b32 s10, 0x55555555
        0xBE8B_03FF,
        0x5555_5555, // s_mov_b32 s11, 0x55555555
    ]);
    w.extend(permlanex16(false, false));
    w.extend(store(1, 0));
    for (slot, (fi, bc)) in [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .enumerate()
    {
        w.extend([0x7E02_02FF, PERM_OLD]); // v_mov_b32 v1, PERM_OLD
        even(&mut w);
        w.extend(permlanex16(fi, bc));
        full(&mut w);
        w.extend(store(1, 4 * (slot as u32 + 1)));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`permlanex16_program`]'s lane `l`.
fn permlanex16_console(l: u32) -> [u32; 5] {
    let in_row = l & 15;
    let selectors = if in_row >= 8 { PERM_HIGH } else { PERM_LOW };
    let source = (l & !15 ^ 16) | (selectors >> (4 * (in_row & 7)) & 15);
    let active = |lane: u32| lane.is_multiple_of(2);
    let with_exec = |inactive_source: u32, fetch_inactive: bool| {
        if !active(l) {
            PERM_OLD
        } else if fetch_inactive || active(source) {
            source
        } else {
            inactive_source
        }
    };
    [
        source,
        with_exec(PERM_OLD, false),
        with_exec(0, false),
        with_exec(PERM_OLD, true),
        with_exec(0, true),
    ]
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_permlanex16_program_is_what_its_comments_say() {
    let mut names = Vec::new();
    for words in [permlanex16(false, false), permlanex16(true, true)] {
        let p = ir::build(
            &[words[0], words[1], S_ENDPGM]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            false,
            32,
        )
        .expect("ir::build");
        names.push(
            p.blocks[0]
                .insts
                .iter()
                .map(|i| i.name().to_string())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(names[0][0], "v_permlanex16_b32");
    assert_eq!(names[1][0], "v_permlanex16_b32");
}

/// The lanes of each row read the other row of their 32-lane half through
/// their selector nibbles, and an inactive source lane gives the old value,
/// zero or its register as FI and BOUND_CTRL say, on every hardware
/// adapter, in console waves of 32 and of 64 (in halves and whole).
#[test]
fn permlanex16_picks_lanes_as_the_console_does_on_every_gpu() {
    let mut failed = Vec::new();
    for lanes in [32, 64] {
        let want: Vec<u32> = (0..lanes).flat_map(permlanex16_console).collect();
        failed.extend(differences_in(
            lanes == 64,
            &permlanex16_program(lanes),
            lanes,
            5,
            &want,
            0,
            false,
        ));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The model itself: lane 0 reads lane 16 | 15 (the low nibble of `s8` is
/// 15, an odd and so inactive lane), lane 8 reads 16 | 0 (the low nibble of `s9`), the second half's
/// lane 32 reads 48 | 15, and FI and BOUND_CTRL change only what an
/// inactive source gives.
#[test]
fn the_permlanex16_model_follows_the_instruction() {
    assert_eq!(permlanex16_console(0), [31, 0xAAAA, 0, 31, 31]);
    assert_eq!(permlanex16_console(8), [16, 16, 16, 16, 16]);
    assert_eq!(permlanex16_console(1), [29, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA]);
    assert_eq!(permlanex16_console(16)[0], 15);
    assert_eq!(permlanex16_console(32)[0], 63);
    assert_eq!(permlanex16_console(48)[0], 47);
}
