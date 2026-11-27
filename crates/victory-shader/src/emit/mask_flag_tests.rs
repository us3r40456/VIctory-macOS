//! The flag a lane loop ends with, read by a select after the loop
//! (`structure`, `clear_after_loop`; the blend-shape skinning programs).
//!
//! The console compiler walks the distinct values of a per-lane index with
//! `do { EXEC = lanes equal to the first lane's; EXEC = rest & ~EXEC }
//! while (EXEC)`. `s_andn2_wrexec` sets SCC to "the EXEC it leaves is not
//! empty", and the loop is left when it is empty: after the loop the flag
//! is clear, whatever the lanes hold, so `s_cselect X, exec, 0` there is 0.
//! No lane has the value otherwise: each leaves the loop in a round of its
//! own.
//!
//! The translation tests run on any host. The GPU test (Windows) runs the
//! wave64 program in halves on every hardware adapter and compares with
//! the console's values.

use super::halves_local_tests::{
    PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, k, native, sop1, store, v, vop1, vop2, vop3,
};
use super::mask_join_tests::{branch, listing, lowered, sop2};
use crate::native::NativeProgram;

const EXEC: u8 = 126;
const VCC: u8 = 106;
const V_AND_B32: u8 = 0x1B;
const V_READFIRSTLANE_B32: u8 = 0x02;
const V_CNDMASK_B32: u16 = 0x101;
const S_CBRANCH_EXECNZ: u8 = 9;

/// Each thread stores 2 where its id is above 29 and 1 elsewhere, chosen
/// by the mask `(29 < id) | (flag ? EXEC : 0)`, where `flag` is what the
/// loop over the four values of `id & 3` ended with: clear. Were it taken
/// as set, every thread would store 2.
///
/// `wide`: a wave64 program (`_b64` masks); else wave32. `inside`: the
/// select sits in the loop, before its branch, where the flag is whatever
/// the round left.
fn flag(wide: bool, inside: bool) -> Vec<u32> {
    // s_mov, s_andn2_wrexec, s_cselect, s_or: `_b32` or `_b64`.
    let (mov, wrexec, cselect, or) = if wide {
        (0x04, 0x3A, 0x0B, 0x11)
    } else {
        (0x03, 0x47, 0x0A, 0x10)
    };
    let select = sop2(cselect, 2, EXEC, 128); // s_cselect s[2:3], exec, 0
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    // The lane's number, unused: the program needs the console's lanes, so
    // a wave64 one runs in halves on a host of 32-lane waves.
    w.extend([0xD765_0003, 0x0001_00C1]); // v_mbcnt_lo_u32_b32 v3, -1, 0
    w.push(vop2(7, V_AND_B32, k(3), 0)); // v_and_b32 v7, 3, v0
    w.push(sop1(4, mov, EXEC)); // s_mov s[4:5], exec
    w.push(sop1(28, mov, EXEC)); // s_mov s[28:29], exec
    w.push(vop1(30, V_READFIRSTLANE_B32, v(7))); // loop: v_readfirstlane_b32 s30, v7
    w.push(0x7DA4_0E1E); // v_cmpx_eq_u32 exec, s30, v7
    w.push(sop1(28, wrexec, 28)); // s_andn2_wrexec s[28:29], s[28:29]
    if inside {
        w.push(select);
    }
    w.push(branch(S_CBRANCH_EXECNZ, if inside { -5 } else { -4 })); // s_cbranch_execnz loop
    w.push(sop1(EXEC, mov, 4)); // s_mov exec, s[4:5]
    w.push(0x7D82_009D); // v_cmp_lt_u32 vcc, 29, v0
    if !inside {
        w.push(select);
    }
    w.push(sop2(or, 2, VCC, 2)); // s_or s[2:3], vcc, s[2:3]
    w.extend(vop3(1, V_CNDMASK_B32, k(1), k(2), 2)); // v_cndmask_b32 v1, 1, 2, s[2:3]
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The wave32 program of one group of 32 threads, as the game's are.
fn narrow(words: &[u32]) -> NativeProgram {
    NativeProgram {
        wave64: false,
        ..native(words, 32, 0)
    }
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let wide = listing(&flag(true, false), true);
    for want in [
        "v_and_b32 v7, 3, v0",
        "s_mov_b64 s[4:5], exec",
        "s_mov_b64 s[28:29], exec",
        "v_mbcnt_lo_u32_b32 v3, -1, 0",
        "0x2c: v_readfirstlane_b32 s30, v7",
        "v_cmpx_eq_u32 exec, s30, v7",
        "s_andn2_wrexec_b64 s[28:29], s[28:29]\n0x38: s_cbranch_execnz label_002C",
        "s_mov_b64 exec, s[4:5]",
        "v_cmp_lt_u32 vcc, 29, v0",
        "s_cselect_b64 s[2:3], exec, 0",
        "s_or_b64 s[2:3], vcc, s[2:3]",
        "v_cndmask_b32 v1, 1, 2, s[2:3]",
    ] {
        assert!(wide.contains(want), "{want:?} not in:\n{wide}");
    }
    let narrow = listing(&flag(false, false), false);
    for want in [
        "s_mov_b32 s28, exec_lo",
        "v_cmpx_eq_u32 exec_lo, s30, v7",
        "s_andn2_wrexec_b32 s28, s28\n0x38: s_cbranch_execnz label_002C",
        "s_mov_b32 exec_lo, s4",
        "s_cselect_b32 s2, exec_lo, 0",
        "s_or_b32 s2, vcc_lo, s2",
        "v_cndmask_b32 v1, 1, 2, s2",
    ] {
        assert!(narrow.contains(want), "{want:?} not in:\n{narrow}");
    }
    let inside = listing(&flag(true, true), true);
    let want = "s_andn2_wrexec_b64 s[28:29], s[28:29]\n0x38: s_cselect_b64 s[2:3], exec, 0\n0x3c: s_cbranch_execnz label_002C";
    assert!(inside.contains(want), "{want:?} not in:\n{inside}");
}

/// The select after the loop translates: in a wave32 program of 32 threads
/// (the game's form), in a wave64 program on a host of 64-lane waves, and
/// in two halves of 32. The halves need half-local reads for the loop's
/// first-lane read, as before; the flag needs nothing from them.
#[test]
fn a_select_on_the_flag_a_lane_loop_ended_with_translates() {
    for host_wave32 in [false, true] {
        let l = lowered(&narrow(&flag(false, false)), host_wave32, false)
            .unwrap_or_else(|e| panic!("wave32, {host_wave32}: {e}"));
        assert_eq!(l.wave_size, Some(32));
    }
    let one = lowered(&native(&flag(true, false), 64, 0), false, false).expect("wave64");
    assert_eq!(one.wave_size, Some(64));
    let halves = lowered(&native(&flag(true, false), 64, 0), true, true).expect("halves");
    assert_eq!(halves.wave_size, Some(32));
    let why = lowered(&native(&flag(true, false), 64, 0), true, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        why.contains("a first-lane read of a value that may differ between 32-lane halves"),
        "{why}"
    );
}

/// A select on that flag inside the loop reads what each round left: no
/// lane has that value, and the program is refused, by the flag's name.
#[test]
fn a_select_on_the_flag_inside_the_loop_stays_refused() {
    for (what, l) in [
        ("wave32", lowered(&narrow(&flag(false, true)), false, false)),
        ("wave64", lowered(&native(&flag(true, true), 64, 0), false, false)),
        ("halves", lowered(&native(&flag(true, true), 64, 0), true, true)),
    ] {
        let why = l.err().unwrap_or_else(|| panic!("{what}: translated")).to_string();
        assert!(why.contains(REFUSAL), "{what}: {why}");
    }
}

/// What the flag of a lane-mask operation read as a number stops with.
const REFUSAL: &str = "the flag of a lane-mask operation read as data";

#[cfg(windows)]
mod gpu {
    use super::super::halves_local_tests::gpu::differences_with;
    use super::super::lower;
    use super::*;

    /// In halves, on every hardware adapter, each thread of the group of
    /// 64 stores the console's value: 2 where its id is above 29, else 1.
    /// Each half walks the values of its own lanes (half-local reads); the
    /// result does not depend on the walk.
    #[test]
    fn the_flag_after_a_lane_loop_is_clear_in_each_half_on_every_gpu() {
        let want: Vec<u32> = (0..64u32).map(|l| if l > 29 { 2 } else { 1 }).collect();
        let local = lower::HalvesLocal {
            reads: true,
            lds: false,
        };
        let failed = differences_with("flag", &flag(true, false), 0, local, 1, &want);
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
