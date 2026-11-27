//! A lane mask a loop leaves alone, joined at the loop's header with a
//! value nothing else reads as a mask (`structure::Structured::joined`):
//! the mask comes back through the join of a branch inside the loop, or
//! from the two lanes of a VGPR the console compiler parked it in (the
//! water transmission programs).
//!
//! The join is each lane's own bit, so it is the same on a host of 64-lane
//! waves and in two halves of 32, with no setting of the halves. The
//! translation tests run on any host. The GPU test (Windows) runs the
//! programs in halves on every hardware adapter and compares with the
//! console's values.

use super::halves_local_tests::{
    PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, k, native, sop1, store, v, vop1, vop2, vop3,
};
use super::lower;
use crate::native::NativeProgram;
use crate::{gfx10, io, ir, resource, structure};

pub(super) const fn sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (ssrc1 as u32) << 8 | ssrc0 as u32
}

/// `s_cbranch_scc1` (5) and the like, `words` instructions on from the next.
pub(super) const fn branch(op: u8, words: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | (words as u16) as u32
}

const S_MOV_B32: u8 = 0x03;
const S_MOV_B64: u8 = 0x04;
const V_MOV_B32: u8 = 0x01;
const V_AND_B32: u8 = 0x1B;
const V_ADD_NC_U32: u8 = 0x25;
const V_CMP_EQ_U32: u16 = 0xC2;
const V_CNDMASK_B32: u16 = 0x101;
const V_READLANE_B32: u16 = 0x360;
const V_WRITELANE_B32: u16 = 0x361;
const S_CBRANCH_SCC1: u8 = 5;
/// `s_add_u32 s4, s4, 1`, `s_cmp_lt_u32 s4, 4`, `s_cmp_eq_u32 s4, 1`.
const S4_PLUS_1: u32 = sop2(0x00, 4, 4, 0x81);
const S4_BELOW_4: u32 = 0xBF0A_8404;
const S4_IS_1: u32 = 0xBF06_8104;

/// Before the loop: `s[2:3]` is the mask of the even threads, `v1` the sum
/// (0) and `s4` the loop's count (0).
fn start() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.push(vop2(7, V_AND_B32, k(1), 0)); // v_and_b32 v7, 1, v0
    w.extend(vop3(2, V_CMP_EQ_U32, k(0), v(7), 0)); // v_cmp_eq_u32 s[2:3], 0, v7
    w.push(vop1(1, V_MOV_B32, k(0))); // v_mov_b32 v1, 0
    w.push(vop1(5, V_MOV_B32, k(0))); // v_mov_b32 v5, 0
    w.push(sop1(4, S_MOV_B32, 128)); // s_mov_b32 s4, 0
    w
}

/// The loop's first instructions: a lane in the mask adds 1, another 2.
fn add() -> Vec<u32> {
    let mut w = vop3(2, V_CNDMASK_B32, k(2), k(1), 2).to_vec(); // v_cndmask_b32 v2, 2, 1, s[2:3]
    w.push(vop2(1, V_ADD_NC_U32, v(2), 1)); // v_add_nc_u32 v1, v2, v1
    w
}

/// The loop's end, `back` instructions back to its first, and the store of
/// each lane's sum: four rounds, so 4 in an even thread and 8 in an odd one.
fn end(w: &mut Vec<u32>, back: i16) {
    w.extend([S4_PLUS_1, S4_BELOW_4, branch(S_CBRANCH_SCC1, -back - 3)]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
}

/// The mask is parked in lanes 0 and 1 of `v5`, its register reused, and
/// read back before the loop's end: the back edge brings two lane reads.
fn parked() -> Vec<u32> {
    let mut w = start();
    w.extend(add());
    w.extend(vop3(5, V_WRITELANE_B32, 2, k(0), 0)); // v_writelane_b32 v5, s2, 0
    w.extend(vop3(5, V_WRITELANE_B32, 3, k(1), 0)); // v_writelane_b32 v5, s3, 1
    w.push(sop1(2, S_MOV_B64, 128)); // s_mov_b64 s[2:3], 0
    w.extend(vop3(2, V_READLANE_B32, v(5), k(0), 0)); // v_readlane_b32 s2, v5, 0
    w.extend(vop3(3, V_READLANE_B32, v(5), k(1), 0)); // v_readlane_b32 s3, v5, 1
    end(&mut w, 12);
    w
}

/// A branch inside the loop copies the mask from `source` back into its
/// register in every round but the second: the back edge brings the join
/// after that branch, which nothing reads as a mask. With `source` 6 the
/// copy is of the mask itself (saved in `s[6:7]`); with 8 it is of another
/// mask, thread 0 alone, so the join after the branch is one of its own.
fn branched(source: u8) -> Vec<u32> {
    let mut w = start();
    w.extend(vop3(8, V_CMP_EQ_U32, k(0), v(0), 0)); // v_cmp_eq_u32 s[8:9], 0, v0
    w.extend(add());
    w.push(sop1(6, S_MOV_B64, 2)); // s_mov_b64 s[6:7], s[2:3]
    w.push(S4_IS_1);
    w.push(branch(S_CBRANCH_SCC1, 1)); // s_cbranch_scc1 over the copy
    w.push(sop1(2, S_MOV_B64, source)); // s_mov_b64 s[2:3], s[<source>:..]
    end(&mut w, 7);
    w
}

/// `native` lowered for a host of 32-lane waves (`host_wave32`) or of 64,
/// with half-local reads (`reads`) or without.
pub(super) fn lowered(
    native: &NativeProgram,
    host_wave32: bool,
    reads: bool,
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
        narrow_waves: false,
        halves_local: lower::HalvesLocal { reads, lds: false },
        mac: Default::default(),
        host_wave32,
    })
}

/// Every instruction of `words`, one a line.
pub(super) fn listing(words: &[u32], wave64: bool) -> String {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let insts = gfx10::decode(&code).expect("decodes");
    let lines: Vec<String> = insts
        .iter()
        .map(|i| format!("{:#x}: {}", i.offset, i.text(wave64)))
        .collect();
    lines.join("\n")
}

/// The hand-encoded words are the instructions their comments name, and
/// each loop's branch goes back to the loop's first instruction.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 3] = [
        (
            parked(),
            &[
                "v_and_b32 v7, 1, v0",
                "v_cmp_eq_u32 s[2:3], 0, v7",
                "s_mov_b32 s4, 0",
                "0x30: v_cndmask_b32 v2, 2, 1, s[2:3]",
                "v_add_nc_u32 v1, v2, v1",
                "v_writelane_b32 v5, s2, 0",
                "v_writelane_b32 v5, s3, 1",
                "s_mov_b64 s[2:3], 0",
                "v_readlane_b32 s2, v5, 0",
                "v_readlane_b32 s3, v5, 1",
                "s_add_u32 s4, s4, 1",
                "s_cmp_lt_u32 s4, 4",
                "s_cbranch_scc1 label_0030",
            ],
        ),
        (
            branched(6),
            &[
                "v_cmp_eq_u32 s[8:9], 0, v0",
                "0x38: v_cndmask_b32 v2, 2, 1, s[2:3]",
                "s_mov_b64 s[6:7], s[2:3]",
                "s_cmp_eq_u32 s4, 1",
                "s_cbranch_scc1 label_0054",
                "0x50: s_mov_b64 s[2:3], s[6:7]",
                "0x54: s_add_u32 s4, s4, 1",
                "s_cbranch_scc1 label_0038",
            ],
        ),
        (branched(8), &["0x50: s_mov_b64 s[2:3], s[8:9]"]),
    ];
    for (words, wants) in programs {
        let all = listing(&words, true);
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// Both programs translate whatever the host's waves are: 64 lanes, two
/// halves of 32 (with half-local reads or without), or a group of one host
/// wave. The branched one uses no wave operation and declares no wave
/// size, so it is one translation for all of them: the join is the lane's
/// own bit. The parked one is lowered in waves of the host's size for its
/// lane writes; the lanes read back are the mask itself either way.
#[test]
fn a_mask_that_comes_back_to_its_loop_through_a_join_or_a_spill_is_joined() {
    let l = |what: &str, words: &[u32], threads, host_wave32, reads| {
        lowered(&native(words, threads, 0), host_wave32, reads)
            .unwrap_or_else(|e| panic!("{what}, {threads} threads, {host_wave32}, {reads}: {e}"))
    };
    let one = l("branched", &branched(6), 64, false, false);
    assert_eq!((one.wave_size, one.wave_ops), (None, false));
    for reads in [false, true] {
        let halves = l("branched", &branched(6), 64, true, reads);
        assert!(halves.dxil == one.dxil, "branched in halves, {reads}");
    }
    assert!(!l("branched", &branched(6), 32, true, false).wave_ops);
    assert_eq!(l("parked", &parked(), 64, false, false).wave_size, Some(64));
    for reads in [false, true] {
        assert_eq!(l("parked", &parked(), 64, true, reads).wave_size, Some(32));
    }
    l("parked", &parked(), 32, true, false);
}

/// A back edge that brings a join of two different masks stays refused,
/// by the same words as before: only what is one mask whichever way a
/// lane came is taken.
#[test]
fn a_join_of_its_own_on_the_way_back_stays_refused() {
    for host_wave32 in [false, true] {
        let why = lowered(&native(&branched(8), 64, 0), host_wave32, true)
            .err()
            .expect("refused")
            .to_string();
        assert!(
            why.contains("a lane-mask join of an untracked mask"),
            "{why}"
        );
    }
}

#[cfg(windows)]
mod gpu {
    use super::super::halves_local_tests::gpu::differences_with;
    use super::*;

    /// In halves, on every hardware adapter, each thread of the group of
    /// 64 has the console's sum: 4 where its id is even (in the mask in
    /// each of the four rounds), 8 where it is odd.
    #[test]
    fn a_mask_joined_at_its_loop_is_the_consoles_in_each_half_on_every_gpu() {
        let want: Vec<u32> = (0..64u32).map(|l| if l % 2 == 0 { 4 } else { 8 }).collect();
        let local = lower::HalvesLocal::default();
        let mut failed = differences_with("parked", &parked(), 0, local, 1, &want);
        failed.extend(differences_with("branched", &branched(6), 0, local, 1, &want));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
