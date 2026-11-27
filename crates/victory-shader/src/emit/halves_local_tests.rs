//! Half-local reads (`lower::halves`, `Options::halves_local`): a console
//! wave64 compute program that reads across its two 32-lane halves, on a
//! host of 32-lane waves. Without the setting each such read is refused with
//! its reason, as before; with it the program translates and each half
//! answers from its own lanes.
//!
//! The translation tests run on any host. The GPU test (Windows) runs each
//! program in halves on every hardware adapter and compares with the
//! half-local values worked out here by hand, which are not the console's
//! where the console's wave reached across its halves.

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{gfx10, io, ir, resource, structure};

pub(super) const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)
const EXEC: u8 = 126;

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
pub(super) const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// Each lane's store address: its thread id times 32.
pub(super) const V16_IS_32_TIMES_V0: u32 = 0x3420_0085; // v_lshlrev_b32 v16, 5, v0

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
pub(super) const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

pub(super) const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
}

pub(super) const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
    (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

pub(super) const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// The second word of a DPP instruction: `vsrc0` fetched as `ctrl` says,
/// every row and bank written.
const fn dpp16(vsrc0: u8, ctrl: u16) -> u32 {
    0xFF00_0000 | (ctrl as u32) << 8 | vsrc0 as u32
}

pub(super) const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xD400_0000 | (op as u32) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
    ]
}

const V_ADD_NC_U32: u8 = 0x25;
const V_READLANE_B32: u16 = 0x360;
/// Operand codes: a VGPR, the constant `k` (0 to 64), src0 taken by DPP.
pub(super) const fn v(reg: u16) -> u16 {
    256 + reg
}
pub(super) const fn k(n: u16) -> u16 {
    128 + n
}
const DPP: u16 = 0xFA;

/// A lane mask read as data: per thread `l` of a group of 64, the two
/// dwords of `29 < l` over the wave (lanes 30 to 63 on the console:
/// `0xC000_0000`, `0xFFFF_FFFF`).
fn mask_as_data() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0x7D82_009D, // v_cmp_lt_u32 vcc, 29, v0
        0x7E02_026A, // v_mov_b32 v1, vcc_lo
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// A lane read: the thread id of lane 40 (40 on the console).
fn lane_read() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A first-lane read of a value that differs between the halves: the
/// thread id (0 on the console), with console lanes (`v_mbcnt`).
fn first_lane() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// The start of a whole-wave section over the thread ids: EXEC saved in
/// `s[0:1]` and made every lane, `v1` the thread id.
fn section_start() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.push(sop1(0, 0x25, 193)); // s_or_saveexec_b64 s[0:1], -1
    w.extend(vop3(1, 0x101, k(0), v(0), 0)); // v_cndmask_b32 v1, 0, v0, s[0:1]
    w
}

/// A wave reduction: the sum of the thread ids of the wave's 64 lanes
/// (2,016 on the console), as the console compiler expands it: within
/// quads, rows of 16 and the two rows of a half by DPP and
/// `v_permlanex16_b32`, then one lane of each half added.
fn wave_sum() -> Vec<u32> {
    let mut w = section_start();
    // quad_perm:[1,0,3,2], quad_perm:[2,3,0,1], row_xmask:4, row_xmask:8
    for ctrl in [0x00B1u16, 0x004E, 0x0164, 0x0168] {
        w.push(vop2(1, V_ADD_NC_U32, DPP, 1)); // v_add_nc_u32 v1, v1, v1 (DPP)
        w.push(dpp16(1, ctrl));
    }
    w.extend(vop3(2, 0x378, v(1), k(0), k(0))); // v_permlanex16_b32 v2, v1, 0, 0
    w.push(vop2(1, V_ADD_NC_U32, v(2), 1)); // v_add_nc_u32 v1, v2, v1
    w.extend(vop3(2, V_READLANE_B32, v(1), k(0), 0)); // v_readlane_b32 s2, v1, 0
    w.extend(vop3(3, V_READLANE_B32, v(1), k(32), 0)); // v_readlane_b32 s3, v1, 32
    w.push(vop1(3, 0x01, 2)); // v_mov_b32 v3, s2
    w.push(vop2(3, V_ADD_NC_U32, 3, 3)); // v_add_nc_u32 v3, s3, v3
    w.push(sop1(EXEC, 0x04, 0)); // s_mov_b64 exec, s[0:1]
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

/// A wave prefix sum: per lane the sum of the thread ids of the lanes
/// below it (`l * (l - 1) / 2` on the console), as the console compiler
/// expands it: row shifts by DPP, then each row of 16 takes the row
/// below's total through `v_readlane_b32`, and the whole shifted up a lane.
fn wave_prefix_sum() -> Vec<u32> {
    let mut w = section_start();
    for shift in [1u16, 2, 4, 8] {
        w.push(vop2(1, V_ADD_NC_U32, DPP, 1)); // v_add_nc_u32 v1, v1, v1 row_shr:<shift>
        w.push(dpp16(1, 0x110 + shift));
    }
    // Rows 1 to 3 add the total of the rows below.
    for (lane, lo, hi) in [
        (15, Some(0xFFFF_0000), None),
        (31, None, Some(0x0000_FFFF)),
        (47, None, Some(0xFFFF_0000)),
    ] {
        w.extend(vop3(2, V_READLANE_B32, v(1), k(lane), 0)); // v_readlane_b32 s2, v1, <lane>
        exec_is(&mut w, lo, hi);
        w.push(vop2(1, V_ADD_NC_U32, 2, 1)); // v_add_nc_u32 v1, s2, v1
    }
    w.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1
    // v3: v1 a lane up (lane 0 keeps its own, and is zeroed below).
    w.push(vop1(3, 0x01, v(1))); // v_mov_b32 v3, v1
    w.push(vop1(3, 0x01, DPP)); // v_mov_b32 v3, v1 row_shr:1
    w.push(dpp16(1, 0x111));
    for (lane, lo, hi) in [
        (15, Some(0x0001_0000), None),
        (31, None, Some(1)),
        (47, None, Some(0x0001_0000)),
    ] {
        w.extend(vop3(2, V_READLANE_B32, v(1), k(lane), 0)); // v_readlane_b32 s2, v1, <lane>
        exec_is(&mut w, lo, hi);
        w.push(vop1(3, 0x01, 2)); // v_mov_b32 v3, s2
    }
    w.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1
    w.push(vop1(2, 0x01, k(0))); // v_mov_b32 v2, 0
    w.push(sop1(EXEC, 0x04, 194)); // s_mov_b64 exec, -2
    w.push(vop1(2, 0x01, v(3))); // v_mov_b32 v2, v3
    w.push(sop1(EXEC, 0x04, 0)); // s_mov_b64 exec, s[0:1]
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// EXEC becomes the lanes of `lo` (its low dword) or of `hi` (its high
/// dword), the other dword 0.
fn exec_is(w: &mut Vec<u32>, lo: Option<u32>, hi: Option<u32>) {
    let dword = |w: &mut Vec<u32>, reg: u8, x: u32| match x {
        0 | 1 => w.push(sop1(reg, 0x03, 128 + x as u8)), // s_mov_b32 <reg>, <x>
        x => w.extend([sop1(reg, 0x03, 255), x]),        // s_mov_b32 <reg>, <literal>
    };
    match (lo, hi) {
        // s_mov_b64 exec, <literal> (zero-extended)
        (Some(lo), None) => w.extend([sop1(EXEC, 0x04, 255), lo]),
        (lo, hi) => {
            dword(w, EXEC, lo.unwrap_or(0));
            dword(w, EXEC + 1, hi.unwrap_or(0));
        }
    }
}

/// LDS a group of one console wave64 shares: no read across the halves in
/// the lowering's sense, and not what the setting is about.
fn shared_lds() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0xD8D8_0000,
        0x0300_0002, // ds_read_b32 v3, v2
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

/// A program that stays within its halves: `v_mbcnt` of all lanes (the
/// lane's number) and a lane write to lane 40.
fn within_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0xBE88_03FF,
        0x0000_0777, // s_mov_b32 s8, 0x777
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD761_0005,
        0x0001_5008, // v_writelane_b32 v5, s8, 40
    ]);
    w.extend(store(1, 0));
    w.extend(store(5, 4));
    w.push(S_ENDPGM);
    w
}

pub(super) fn native(words: &[u32], threads: u32, lds_bytes: u32) -> NativeProgram {
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
        wave64: true,
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

/// `words` as a wave64 compute program of one row of `threads`, lowered for
/// a host of 32-lane waves (`host_wave32`) or of 64, with half-local reads
/// (`local`) or without.
fn lowered(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    host_wave32: bool,
    local: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    let local = lower::HalvesLocal {
        reads: local,
        lds: false,
    };
    lowered_with(words, threads, lds_bytes, host_wave32, local)
}

/// [`lowered`] with each of the halves' settings as given.
fn lowered_with(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    host_wave32: bool,
    local: lower::HalvesLocal,
) -> Result<lower::Lowered, crate::ShaderError> {
    let native = native(words, threads, lds_bytes);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
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
        narrow_waves: false,
        halves_local: local,
        mac: Default::default(),
        host_wave32,
    })
}

/// "translated", or why not: one group of 64 threads in halves.
fn outcome(words: &[u32], lds_bytes: u32, local: bool) -> String {
    match lowered(words, 64, lds_bytes, true, local) {
        Ok(_) => "translated".into(),
        Err(e) => e.to_string(),
    }
}

/// The programs that read across the halves, each with the refusal it gets
/// without the setting.
pub(super) fn across() -> [(&'static str, Vec<u32>, &'static str); 5] {
    [
        (
            "a lane mask read as data",
            mask_as_data(),
            "a lane mask read as data (wave64 on a 32-lane host)",
        ),
        (
            "a lane read",
            lane_read(),
            "a lane read that may cross 32-lane halves (wave64 on a 32-lane host)",
        ),
        (
            "a first-lane read",
            first_lane(),
            "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)",
        ),
        (
            "a wave reduction",
            wave_sum(),
            "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)",
        ),
        (
            "a wave prefix sum",
            wave_prefix_sum(),
            "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)",
        ),
    ]
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 5] = [
        (
            mask_as_data(),
            &[
                "v_cmp_lt_u32 vcc, 29, v0",
                "v_mov_b32 v1, vcc_lo",
                "v_mov_b32 v2, vcc_hi",
                "buffer_store_dword v2, v16, s[40:43], 0 offen offset:4",
            ],
        ),
        (
            lane_read(),
            &["v_readlane_b32 s12, v0, 40", "v_mov_b32 v1, s12"],
        ),
        (
            first_lane(),
            &["v_readfirstlane_b32 s12, v0", "v_mov_b32 v2, s12"],
        ),
        (
            wave_sum(),
            &[
                "s_or_saveexec_b64 s[0:1], -1",
                "v_cndmask_b32 v1, 0, v0, s[0:1]",
                "v_add_nc_u32 v1, v1, v1 quad_perm:[1,0,3,2]",
                "v_add_nc_u32 v1, v1, v1 quad_perm:[2,3,0,1]",
                "v_permlanex16_b32 v2, v1, 0, 0",
                "v_add_nc_u32 v1, v2, v1",
                "v_readlane_b32 s2, v1, 0",
                "v_readlane_b32 s3, v1, 32",
                "v_mov_b32 v3, s2",
                "v_add_nc_u32 v3, s3, v3",
                "s_mov_b64 exec, s[0:1]",
            ],
        ),
        (
            wave_prefix_sum(),
            &[
                "v_add_nc_u32 v1, v1, v1 row_shr:8",
                "v_readlane_b32 s2, v1, 47",
                "s_mov_b64 exec, lit(0xffff0000)",
                "s_mov_b32 exec_hi, lit(0x0000ffff)",
                "v_add_nc_u32 v1, s2, v1",
                "v_mov_b32 v3, v1 row_shr:1",
                "v_mov_b32 v3, s2",
                "s_mov_b64 exec, -2",
                "v_mov_b32 v2, v3",
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

/// Without the setting a read across the halves is answered by the Mac
/// port's hand-over where every lane of the group runs together (before
/// that was merged every one was refused with its reason; a read where
/// only some lanes run still is, `walk_tests`). With the setting the same
/// programs translate with the half's own answer, in waves of 32, and
/// not through that hand-over. On a host of 64-lane waves the setting
/// changes nothing.
#[test]
fn reads_across_halves_are_refused_without_the_setting_and_translate_with_it() {
    for (what, words, refusal) in across() {
        // Each was refused with `refusal` until VIctory-macOS's hand-over
        // was merged (its #309 and #360, `lower::halves`): every lane of
        // the group runs these reads together, so that answers now.
        let _ = refusal;
        let off = lowered(&words, 64, 0, true, false)
            .unwrap_or_else(|e| panic!("{what}, setting off: {e}"));
        assert!(off.port_exchange && off.exchanges == 0, "{what}, setting off");
        let on = lowered(&words, 64, 0, true, true)
            .unwrap_or_else(|e| panic!("{what}, setting on: {e}"));
        assert_eq!(on.wave_size, Some(32), "{what}");
        assert!(!on.port_exchange, "{what}: half-local reads come first");
        assert_ne!(on.dxil, off.dxil, "{what}");
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut on.dxil.clone())
                .unwrap_or_else(|e| panic!("{what}: DXIL validation failed: {e}"));
        }
        let whole = |local| {
            lowered(&words, 64, 0, false, local)
                .unwrap_or_else(|e| panic!("{what}, one wave64: {e}"))
        };
        assert_eq!(whole(false).wave_size, Some(64), "{what}");
        assert_eq!(whole(true).dxil, whole(false).dxil, "{what}");
    }
}

/// A group of 32 threads or fewer is one host wave: a first-lane read is
/// the console's there, with the setting or without.
#[test]
fn a_group_of_one_host_wave_reads_its_first_lane_either_way() {
    let dxil = |local| {
        lowered(&first_lane(), 32, 0, true, local)
            .expect("one host wave")
            .dxil
    };
    assert_eq!(dxil(true), dxil(false));
}

/// A program that stays within its halves is the same translation with the
/// setting and without, byte for byte.
#[test]
fn the_setting_changes_nothing_in_a_program_that_stays_in_its_halves() {
    let dxil = |local| {
        lowered(&within_halves(), 64, 0, true, local)
            .expect("in halves")
            .dxil
    };
    assert_eq!(dxil(true), dxil(false));
}

/// LDS shared by a group of one console wave64 stays refused: the setting
/// is about reads across the halves, not about memory they share.
#[test]
fn shared_lds_of_a_one_wave64_group_stays_refused_with_the_setting() {
    for local in [false, true] {
        let why = outcome(&shared_lds(), 256, local);
        assert!(
            why.contains("LDS one console wave64 shares without barriers"),
            "setting {local}: {why}"
        );
    }
}

/// With its own setting such a group translates, in waves of 32 and with
/// no barrier added; that setting lifts no refusal of a read across the
/// halves, and changes nothing in a program without such LDS.
#[test]
fn shared_lds_of_a_one_wave64_group_runs_unsynced_with_its_own_setting() {
    let lds = |reads| lower::HalvesLocal { reads, lds: true };
    for reads in [false, true] {
        let l = lowered_with(&shared_lds(), 64, 256, true, lds(reads)).expect("shared LDS");
        assert_eq!(l.wave_size, Some(32));
        assert_eq!(l.split_barriers, 0);
    }
    // The LDS setting answers no read across the halves: without half-local
    // reads those are the Mac port's hand-over's (refused before it was
    // merged), with the setting or without, the same translation.
    for (what, words, _) in across() {
        let l = lowered_with(&words, 64, 0, true, lds(false))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(l.port_exchange, "{what}");
        let plain = lowered_with(&words, 64, 0, true, Default::default()).expect("handed over");
        assert_eq!(l.dxil, plain.dxil, "{what}");
    }
    let dxil = |local| {
        lowered_with(&within_halves(), 64, 0, true, local)
            .expect("in halves")
            .dxil
    };
    assert_eq!(dxil(lds(false)), dxil(lower::HalvesLocal::default()));
}

/// What each half answers on its own, per lane `l` of the group of 64.
#[cfg(windows)]
pub(super) mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;

    fn shader(l: lower::Lowered) -> super::super::HostShader {
        super::super::HostShader {
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

    /// Runs `words`, one group of 64 whose lanes store `per_lane` dwords
    /// at `l * 32`, in halves with half-local reads on every hardware
    /// adapter (an adapter of 64-lane waves runs the two waves of 32 the
    /// program declares). A line per lane whose dwords differ from `want`.
    fn differences(what: &str, words: &[u32], per_lane: usize, want: &[u32]) -> Vec<String> {
        let reads = lower::HalvesLocal {
            reads: true,
            lds: false,
        };
        differences_with(what, words, 0, reads, per_lane, want)
    }

    /// [`differences`] for a program with `lds_bytes` of LDS, with the
    /// halves' settings as given.
    pub(in super::super) fn differences_with(
        what: &str,
        words: &[u32],
        lds_bytes: u32,
        local: lower::HalvesLocal,
        per_lane: usize,
        want: &[u32],
    ) -> Vec<String> {
        let shader = shader(lowered_with(words, 64, lds_bytes, true, local).expect("in halves"));
        let views = [
            View::Texture(1, 1, &[[0.0; 4]]),
            View::Buffer {
                data: &[0; 4],
                stride: Stride::Structured(16),
            },
            View::RawUav(64 * 8),
        ];
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let hosts = Host::each_hardware();
        if hosts.is_empty() {
            eprintln!("no SM 6.6 hardware device; skipped");
        }
        let mut failed = Vec::new();
        for (adapter, host) in &hosts {
            let label = format!("{adapter} ({what})");
            let Ok((root, pso)) = host.compute_pipeline(&shader) else {
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
                        "{label}: lane {l}: {:?}, each half on its own {want:?}",
                        &got[..per_lane]
                    )
                })
                .collect();
            eprintln!("{label}: {} lanes of 64 differ", wrong.len());
            failed.extend(wrong);
        }
        failed
    }

    /// Each half answers from its own 32 lanes, the upper half's lanes
    /// numbered 32 to 63 as on the console:
    /// - the mask of `29 < l` holds lanes 30 and 31 in the lower half's
    ///   low dword and nothing above; in the upper half its 32 lanes in the
    ///   high dword and nothing below;
    /// - a read of lane 40 is the console's in the upper half, which holds
    ///   that lane, and lane 8 (40 less 32) in the lower half;
    /// - the first lane is lane 0 in the lower half and lane 32 in the upper;
    /// - the sum of the thread ids is the half's: 496 and 1,520, the
    ///   console's 2,016 in two parts;
    /// - the prefix sum counts from the half's first lane: the console's in
    ///   the lower half, and less the lower half's 496 in the upper.
    #[test]
    fn each_half_answers_reads_across_the_halves_on_its_own_on_every_gpu() {
        let lanes = |f: &dyn Fn(u32) -> Vec<u32>| (0..64).flat_map(f).collect::<Vec<u32>>();
        let upper = |l: u32| l >= 32;
        let mut failed = Vec::new();
        let want = lanes(&|l| {
            if upper(l) {
                vec![0, u32::MAX]
            } else {
                vec![0xC000_0000, 0]
            }
        });
        failed.extend(differences("mask", &mask_as_data(), 2, &want));
        let want = lanes(&|l| vec![if upper(l) { 40 } else { 8 }]);
        failed.extend(differences("lane read", &lane_read(), 1, &want));
        let want = lanes(&|l| vec![if upper(l) { 32 } else { 0 }]);
        failed.extend(differences("first lane", &first_lane(), 1, &want));
        let want = lanes(&|l| vec![if upper(l) { 1520 } else { 496 }]);
        failed.extend(differences("sum", &wave_sum(), 1, &want));
        let want = lanes(&|l| vec![(l & !31..l).sum()]);
        failed.extend(differences("prefix sum", &wave_prefix_sum(), 1, &want));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// Shared LDS run without sync, in the one case with a sure answer:
    /// each lane reads back the dword it wrote itself (the lanes below it
    /// in the mask's low dword: `l`, and 32 from lane 32 on, as on the
    /// console), so no order between the halves matters.
    #[test]
    fn unsynced_shared_lds_keeps_what_a_lane_wrote_itself_on_every_gpu() {
        let lds = lower::HalvesLocal {
            reads: false,
            lds: true,
        };
        let want: Vec<u32> = (0..64u32).map(|l| l.min(32)).collect();
        let failed = differences_with("shared LDS", &shared_lds(), 256, lds, 1, &want);
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
