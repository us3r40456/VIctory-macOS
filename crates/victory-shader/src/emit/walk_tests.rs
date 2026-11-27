//! Hand-overs across the 32-lane halves in code only some lanes ran
//! (`walk`, `lower::together`, `Options::halves_walk`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs"). The
//! translation tests and the model run on any host. The GPU tests (Windows)
//! run the programs on every hardware adapter and compare with the
//! console's values.
//!
//! The programs have the shape of the title's tiled sun-shadow programs
//! (`CS_ShadowRenderTiled*`): lanes outside the target leave by an
//! early-out, and the lanes left count a compare's mask ("does any lane of
//! the wave ..."), also inside an `if` and under a branch on a flag that is
//! the same for the whole group.

use super::halves_local_tests::{PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, native, sop1, store};
use super::lower::{self, HalvesLocal, Mac};
use super::walk;
use crate::{gfx10, io, ir, resource, structure};

const EXEC: u8 = 126;
const VCC: u8 = 106;
const REFUSAL: &str = "a lane mask read as data (wave64 on a 32-lane host)";
/// `v_mov_b32 v1, 0`.
const V1_IS_0: u32 = 0x7E02_0280;
/// `v_cmpx_lt_u32 exec, 29, v0`: the lanes inside are those from 30 on.
const INSIDE: u32 = 0x7DA2_009D;
/// `v_cmp_lt_u32 vcc, 29, v0` and `v_cmp_lt_u32 vcc, 40, v0`.
const ABOVE_29: u32 = 0x7D82_009D;
const ABOVE_40: u32 = 0x7D82_00A8;
/// `s_bcnt1_i32_b64 s12, vcc`.
const COUNT: u32 = sop1(12, 16, VCC);
/// `v_mov_b32 v1, s12`.
const V1_IS_S12: u32 = 0x7E02_020C;
/// `s_mov_b64 s[2:3], exec` and back.
const SAVE: u32 = sop1(2, 0x04, EXEC);
const RESTORE: u32 = sop1(EXEC, 0x04, 2);
/// `s_mov_b32 s6, 1`, `s_cmp_eq_u32 s6, 0`: a flag every thread of a group
/// holds alike, and its test.
const FLAG: [u32; 2] = [sop1(6, 0x03, 129), 0xBF06_8006];

/// `s_cbranch_execz`, `s_cbranch_scc0` and `s_cbranch_scc1` over `n` words.
const fn execz(n: u32) -> u32 {
    0xBF88_0000 | n
}
const fn scc0(n: u32) -> u32 {
    0xBF84_0000 | n
}
const fn scc1(n: u32) -> u32 {
    0xBF85_0000 | n
}

/// The start of every program: each lane stores 0, so that a lane that
/// leaves has stored something.
fn start() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([V16_IS_32_TIMES_V0, V1_IS_0]);
    w.extend(store(1, 0));
    w
}

/// The lanes from 30 on stay, the others leave by the early-out; the lanes
/// left store how many of them are above 40. On the console lanes 0 to 29
/// keep 0 and lanes 30 to 63 store 23; a second wave (lanes 64 to 127)
/// stores 64.
fn behind_an_early_out() -> Vec<u32> {
    let mut w = start();
    w.extend([INSIDE, execz(5), ABOVE_40, COUNT, V1_IS_S12]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The same count inside an `if`, the store after it: the same values.
fn in_an_if() -> Vec<u32> {
    let mut w = start();
    w.extend([SAVE, INSIDE, execz(3), ABOVE_40, COUNT, V1_IS_S12, RESTORE]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A count under a branch on the group's flag: every lane stores how many
/// lanes of its wave are above 29 (34 in the first wave, 64 in a second).
fn under_a_group_branch() -> Vec<u32> {
    let mut w = start();
    w.extend(FLAG);
    w.extend([scc1(3), ABOVE_29, COUNT, V1_IS_S12]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Both: inside an `if`, under the group's flag (the see-through variants
/// of the title's programs count there). The values of [`in_an_if`].
fn in_an_if_under_a_group_branch() -> Vec<u32> {
    let mut w = start();
    w.extend([SAVE, INSIDE, execz(6)]);
    w.extend(FLAG);
    w.extend([scc1(3), ABOVE_40, COUNT, V1_IS_S12, RESTORE]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// An `if` that leaves its count in `s12` for every lane to store after
/// it: a lane that went round the `if` stores the 0 from before.
fn a_scalar_read_after_the_if() -> Vec<u32> {
    let mut w = start();
    w.extend([
        sop1(12, 0x03, 128), // s_mov_b32 s12, 0
        SAVE,
        INSIDE,
        execz(2),
        ABOVE_40,
        COUNT,
        RESTORE,
        V1_IS_S12,
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A count under a branch each wave decides for itself (on another count):
/// two waves of a group can take it differently.
fn under_a_wave_branch() -> Vec<u32> {
    let mut w = start();
    w.extend([ABOVE_29, COUNT, scc0(3), ABOVE_40, COUNT, V1_IS_S12]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 5] = [
        (
            behind_an_early_out(),
            &[
                "v_mov_b32 v1, 0",
                "v_cmpx_lt_u32 exec, 29, v0\ns_cbranch_execz label_0040",
                "v_cmp_lt_u32 vcc, 40, v0\ns_bcnt1_i32_b64 s12, vcc\nv_mov_b32 v1, s12",
                "label_0040\ns_endpgm",
            ],
        ),
        (
            in_an_if(),
            &[
                "s_mov_b64 s[2:3], exec\nv_cmpx_lt_u32 exec, 29, v0\ns_cbranch_execz label_003C",
                "v_mov_b32 v1, s12\nlabel_003C\ns_mov_b64 exec, s[2:3]",
            ],
        ),
        (
            under_a_group_branch(),
            &[
                "s_mov_b32 s6, 1\ns_cmp_eq_u32 s6, 0\ns_cbranch_scc1 label_003C",
                "v_cmp_lt_u32 vcc, 29, v0\ns_bcnt1_i32_b64 s12, vcc",
            ],
        ),
        (
            in_an_if_under_a_group_branch(),
            &[
                "s_cbranch_execz label_0048",
                "s_cbranch_scc1 label_0048",
                "label_0048\ns_mov_b64 exec, s[2:3]",
            ],
        ),
        (
            under_a_wave_branch(),
            &["s_bcnt1_i32_b64 s12, vcc\ns_cbranch_scc0 label_003C"],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let insts = gfx10::decode(&code).expect("decodes");
        // Each instruction on a line, a `label_<offset>` line before a
        // branch target.
        let targets: Vec<String> = insts
            .iter()
            .filter_map(|i| i.text(true).split("label_").nth(1).map(str::to_owned))
            .collect();
        let all: Vec<String> = insts
            .iter()
            .flat_map(|i| {
                let at = format!("{:04X}", i.offset);
                let label = targets.contains(&at).then(|| format!("label_{at}"));
                label.into_iter().chain([i.text(true)])
            })
            .collect();
        let all = all.join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// `words` as a wave64 compute program of one row of `threads`, translated
/// as `emit` does: with `mac.walk` through [`walk::open`], and as it is
/// where that declines.
fn translated(
    words: &[u32],
    threads: u32,
    host_wave32: bool,
    reads: bool,
    mac: Mac,
) -> Result<lower::Lowered, crate::ShaderError> {
    let native = native(words, threads, 0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lower = |(p, st, io, res): walk::Stages<'_>, reads: bool| {
        lower::lower(&lower::Parts {
            native: &native,
            p,
            st,
            io,
            res,
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
            halves_local: HalvesLocal { reads, lds: false },
            mac,
            host_wave32,
        })
    };
    let start = (&p, &st, &io, &res);
    if mac.walk
        && host_wave32
        && let Some(exact) = walk::open(&native, start, lower)
    {
        return Ok(exact);
    }
    lower(start, reads)
}

fn halves(words: &[u32], reads: bool, mac: Mac) -> Result<lower::Lowered, crate::ShaderError> {
    translated(words, 64, true, reads, mac)
}

fn exchange() -> Mac {
    Mac {
        exchange: true,
        ..Mac::default()
    }
}

fn walking() -> Mac {
    Mac {
        walk: true,
        ..exchange()
    }
}

/// With the exchange alone a count behind an early-out or inside an `if`
/// is refused as before the walk existed, and one under a group's branch
/// alone is the Mac port's hand-over's since that was merged (and with
/// half-local reads all are translated as before). With the walk the program
/// translates in waves of 32 with the count handed over, the same
/// translation with half-local reads or without: nothing in it is a half's
/// own answer.
#[test]
fn a_count_where_only_some_lanes_ran_is_handed_over() {
    for (what, words) in [
        ("early-out", behind_an_early_out()),
        ("if", in_an_if()),
        ("group branch", under_a_group_branch()),
        ("if and group branch", in_an_if_under_a_group_branch()),
    ] {
        // Without the walk: refused, except under a group's branch alone,
        // where every lane of the group runs together and the Mac port's
        // hand-over answers since it was merged (its #376; refused before).
        match halves(&words, false, exchange()) {
            Err(off) => {
                assert_ne!(what, "group branch", "{off}");
                assert!(off.to_string().contains(REFUSAL), "{what}: {off}");
            }
            Ok(off) => {
                assert_eq!(what, "group branch");
                assert!(off.port_exchange && off.exchanges == 0, "{what}");
            }
        }
        let on = halves(&words, false, walking());
        let on = on.unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!((on.wave_size, on.exchanges), (Some(32), 1), "{what}");
        assert!(!on.port_exchange, "{what}: this line's exchange, not the Mac port's");
        let with_local = halves(&words, true, walking()).expect("handed over");
        assert_eq!(with_local.dxil, on.dxil, "{what}");
        let local = halves(&words, true, exchange()).expect("half-local");
        assert_eq!(local.exchanges, 0, "{what}");
        assert_ne!(local.dxil, on.dxil, "{what}");
        // Two console waves in a group: a hand-over per wave.
        let two = translated(&words, 128, true, false, walking());
        assert_eq!(two.unwrap_or_else(|e| panic!("{what}: {e}")).exchanges, 1);
        // A host of 64-lane waves answers by itself, walk or not.
        let whole = |mac| {
            translated(&words, 64, false, false, mac)
                .expect("one wave64")
                .dxil
        };
        assert_eq!(whole(walking()), whole(Mac::default()), "{what}");
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut on.dxil.clone())
                .unwrap_or_else(|e| panic!("{what}: DXIL validation failed: {e}"));
        }
    }
}

/// The walk without the exchange changes nothing: there is nothing to walk
/// to.
#[test]
fn the_walk_alone_changes_nothing() {
    let alone = Mac {
        walk: true,
        ..Mac::default()
    };
    for words in [behind_an_early_out(), in_an_if(), under_a_group_branch()] {
        let why = halves(&words, false, alone).err().expect("refused");
        assert!(why.to_string().contains(REFUSAL), "{why}");
        let dxil = |mac| halves(&words, true, mac).expect("half-local").dxil;
        assert_eq!(dxil(alone), dxil(Mac::default()));
    }
}

/// What the walk leaves alone stays as it was, refused or half-local, bit
/// for bit: an `if` whose scalar the code after it reads (a lane that went
/// round had the old one), and a count under a branch that each wave of a
/// group decides for itself (no barrier can go where part of a group
/// comes).
#[test]
fn what_is_not_exact_stays_as_it_was() {
    for (what, words) in [
        ("scalar read after the if", a_scalar_read_after_the_if()),
        ("wave branch", under_a_wave_branch()),
    ] {
        for mac in [exchange(), walking()] {
            let why = halves(&words, false, mac)
                .err()
                .unwrap_or_else(|| panic!("{what}: translated"))
                .to_string();
            assert!(why.contains(REFUSAL), "{what}: {why}");
        }
        let dxil = |mac| halves(&words, true, mac).expect("half-local").dxil;
        assert_eq!(dxil(walking()), dxil(exchange()), "{what}");
    }
}

/// The walked form itself: the skip around the count is gone, the count's
/// block is one every lane runs, and the instructions behind the early-out
/// carry the EXEC guard that reaching them no longer proves.
#[test]
fn every_lane_reaches_the_count_and_runs_nothing_there() {
    let words = behind_an_early_out();
    let native = native(&words, 64, 0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let count = |p: &ir::Program| {
        let has = |b: &ir::Block| b.insts.iter().any(|i| i.name() == "s_bcnt1_i32_b64");
        p.blocks.iter().position(has).expect("the count")
    };
    assert!(!lower::lds_order::once(&p)[count(&p)]);
    assert!(st.guards[count(&p)].iter().all(Option::is_none));
    let mut around = vec![false; p.blocks.len()];
    around[count(&p)] = true;
    let (q, skips) = ir::walk::walked(&p, &around).expect("walked");
    let st = structure::structure(&q).expect("structure");
    assert_eq!(skips.len(), 1);
    assert!(lower::lds_order::once(&q)[count(&q)]);
    let guarded = |name: &str| {
        let b = &q.blocks[count(&q)];
        let i = b.insts.iter().position(|i| i.name() == name).expect(name);
        st.guards[count(&q)][i].is_some()
    };
    assert!(guarded("v_cmp_lt_u32"));
    assert!(guarded("v_mov_b32"));
    assert!(guarded("buffer_store_dword"));
}

// ---- Loops the group goes round together, and the thread that runs alone ----
//
// The shape of the title's tile-list programs (`CS_CalcRasterClassificationListCounts`,
// `CS_WriteRasterClassificationResults`): behind an early-out, a loop on a
// scalar count in which a mask of the wave's lanes is counted each round;
// and thread 0 alone summing up through first-lane reads.

/// `s_mov_b32 s20, 0`; and the loop's end: `s_add_u32 s20, s20, 1`,
/// `s_cmp_lt_u32 s20, 2`, `s_cbranch_scc1` back over `body` words.
const S20_IS_0: u32 = sop1(20, 0x03, 128);
fn twice(body: u32) -> [u32; 3] {
    [0x8014_8114, 0xBF0A_8214, scc1(0x1_0000 - (body + 3))]
}

/// Behind the early-out, twice round a loop: the lanes left count those of
/// them above 40 and keep the count. The values of [`behind_an_early_out`].
fn in_a_loop_behind_an_early_out() -> Vec<u32> {
    let mut w = start();
    w.extend([INSIDE, execz(9), S20_IS_0, ABOVE_40, COUNT, V1_IS_S12]);
    w.extend(twice(3));
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The same loop counted to the first lane's thread id (at least once): a
/// count the host waves do not share, so they go round apart.
fn in_a_loop_each_wave_counts() -> Vec<u32> {
    let mut w = start();
    w.extend([S20_IS_0, 0x7E2A_0500, ABOVE_40, COUNT, V1_IS_S12]);
    w.extend([0x8014_8114, 0xBF0A_1514, scc1(0x1_0000 - 6)]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Thread 0 alone (`v_cmp_eq_u32 vcc, 0, v0`, `s_and_saveexec_b64 s[2:3],
/// vcc`) reads the first lane of a value that differs by lane (its thread
/// id plus 5) and stores it: 5 in thread 0, the 0 from before elsewhere.
/// (`v_mbcnt` makes it a program of console lanes, as the lane operations
/// around such a read do.)
fn thread_0_reads_its_first_lane() -> Vec<u32> {
    let mut w = start();
    w.extend([
        0xD765_0003,
        0x0001_00C1,         // v_mbcnt_lo_u32_b32 v3, -1, 0
        0x7D84_0080,         // v_cmp_eq_u32 vcc, 0, v0
        0x4A04_0085,         // v_add_nc_u32 v2, 5, v0
        sop1(2, 0x24, VCC),  // s_and_saveexec_b64 s[2:3], vcc
        execz(2),
        0x7E18_0502,         // v_readfirstlane_b32 s12, v2
        V1_IS_S12,
        RESTORE,
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_loop_programs_are_what_their_comments_say() {
    for (words, wants) in [
        (
            in_a_loop_behind_an_early_out(),
            &[
                "s_cbranch_execz label_0050",
                "s_mov_b32 s20, 0\nlabel_0030\nv_cmp_lt_u32 vcc, 40, v0\ns_bcnt1_i32_b64 s12, vcc",
                "s_add_u32 s20, s20, 1\ns_cmp_lt_u32 s20, 2\ns_cbranch_scc1 label_0030",
            ][..],
        ),
        (
            in_a_loop_each_wave_counts(),
            &["v_readfirstlane_b32 s21, v0", "s_cmp_lt_u32 s20, s21\ns_cbranch_scc1 label_002C"][..],
        ),
        (
            thread_0_reads_its_first_lane(),
            &[
                "v_cmp_eq_u32 vcc, 0, v0\nv_add_nc_u32 v2, 5, v0\ns_and_saveexec_b64 s[2:3], vcc\ns_cbranch_execz label_0044",
                "v_readfirstlane_b32 s12, v2\nv_mov_b32 v1, s12\nlabel_0044\ns_mov_b64 exec, s[2:3]",
            ][..],
        ),
    ] {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let insts = gfx10::decode(&code).expect("decodes");
        let targets: Vec<String> = insts
            .iter()
            .filter_map(|i| i.text(true).split("label_").nth(1).map(str::to_owned))
            .collect();
        let all: Vec<String> = insts
            .iter()
            .flat_map(|i| {
                let at = format!("{:04X}", i.offset);
                let label = targets.contains(&at).then(|| format!("label_{at}"));
                label.into_iter().chain([i.text(true)])
            })
            .collect();
        let all = all.join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// A count in a loop the whole group goes round together is handed over
/// each round, with a second barrier after the loads (the next round stores
/// to the same slots): refused with the exchange alone, exact with the
/// walk, the same translation with half-local reads. The count outside a
/// loop has the one barrier.
#[test]
fn a_count_in_a_loop_the_group_goes_round_together_is_handed_over() {
    let words = in_a_loop_behind_an_early_out();
    let off = halves(&words, false, exchange()).err().expect("refused");
    assert!(off.to_string().contains(REFUSAL), "{off}");
    let on = halves(&words, false, walking()).expect("handed over");
    assert_eq!((on.wave_size, on.exchanges, on.barriers), (Some(32), 1, 2));
    assert_eq!(halves(&words, true, walking()).expect("handed over").dxil, on.dxil);
    let once = halves(&behind_an_early_out(), false, walking()).expect("handed over");
    assert_eq!((once.exchanges, once.barriers), (1, 1));
    let two = translated(&words, 128, true, false, walking()).expect("two console waves");
    assert_eq!((two.exchanges, two.barriers), (1, 2));
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut on.dxil.clone())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
}

/// A loop each host wave counts for itself is not gone round together: no
/// barrier can go in it, and the program stays as it was.
#[test]
fn a_loop_each_wave_counts_for_itself_stays_as_it_was() {
    let words = in_a_loop_each_wave_counts();
    for mac in [exchange(), walking()] {
        assert!(halves(&words, false, mac).is_err());
    }
    let dxil = |mac| halves(&words, true, mac).expect("half-local").dxil;
    assert_eq!(dxil(walking()), dxil(exchange()));
}

/// A first-lane read that only thread 0 can run is that thread's own value:
/// refused with the exchange alone (the first lane may be in the other
/// half), translated with the walk without a hand-over, and not on a host
/// of 64-lane waves or where the compare is not of the thread id.
#[test]
fn a_first_lane_read_only_one_thread_runs_is_its_own_value() {
    let words = thread_0_reads_its_first_lane();
    let off = halves(&words, false, exchange()).err().expect("refused");
    assert!(off.to_string().contains("a first-lane read of a value that may differ"), "{off}");
    let on = halves(&words, false, walking()).expect("its own value");
    assert_eq!((on.wave_size, on.exchanges, on.barriers), (Some(32), 0, 0));
    assert_eq!(halves(&words, true, walking()).expect("its own value").dxil, on.dxil);
    // Compared with a value that is not the thread id (v2): refused.
    let mut other = words.clone();
    let at = other.iter().position(|w| *w == 0x7D84_0080).expect("the compare");
    other.swap(at, at + 1);
    other[at + 1] = 0x7D84_0480; // v_cmp_eq_u32 vcc, 0, v2
    for mac in [exchange(), walking()] {
        assert!(halves(&other, false, mac).is_err());
    }
}

// ---- A model of the hand-over in a loop ----
//
// Host waves as units, every order of their steps. Each round every wave
// stores its value in its slot, the group meets, every wave loads its
// partner's slot; with `two` the group meets again before the next round.

/// Every result of `rounds` rounds: per round and wave the value loaded.
fn looped(waves: usize, rounds: usize, two: bool) -> Vec<Vec<u32>> {
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct State {
        /// Per wave: the round, and 0 stores, 1 waits, 2 loads, 3 waits again.
        pcs: Vec<(usize, u8)>,
        slots: Vec<u32>,
        got: Vec<u32>,
    }
    let value = |round: usize, w: usize| (10 * (round + 1) + w) as u32;
    let start = State {
        pcs: vec![(0, 0); waves],
        slots: vec![7; waves],
        got: vec![0; rounds * waves],
    };
    let (mut seen, mut stack, mut ends) = (std::collections::HashSet::new(), vec![start], Vec::new());
    while let Some(s) = stack.pop() {
        if !seen.insert(s.clone()) {
            continue;
        }
        if s.pcs.iter().all(|pc| pc.0 == rounds) {
            ends.push(s.got.clone());
            continue;
        }
        // A barrier lets the group on when every wave has arrived at it.
        let (round, step) = s.pcs[0];
        if (step == 1 || step == 3) && s.pcs.iter().all(|pc| *pc == (round, step)) {
            let mut next = s.clone();
            next.pcs = vec![if step == 1 { (round, 2) } else { (round + 1, 0) }; waves];
            stack.push(next);
            continue;
        }
        for w in 0..waves {
            let (round, step) = s.pcs[w];
            let mut next = s.clone();
            match step {
                0 if round < rounds => {
                    next.slots[w] = value(round, w);
                    next.pcs[w] = (round, 1);
                }
                2 => {
                    next.got[round * waves + w] = s.slots[w ^ 1];
                    next.pcs[w] = if two { (round, 3) } else { (round + 1, 0) };
                }
                _ => continue,
            }
            stack.push(next);
        }
    }
    ends.sort();
    ends.dedup();
    ends
}

/// With the second barrier every order gives each wave its partner's value
/// of the round; with one barrier alone some order gives a wave the value
/// its partner has already stored for the next round.
#[test]
fn a_hand_over_in_a_loop_needs_the_second_barrier() {
    for waves in [2, 4] {
        let want: Vec<u32> = (0..3)
            .flat_map(|r| (0..waves).map(move |w| (10 * (r + 1) + (w ^ 1)) as u32))
            .collect();
        assert_eq!(looped(waves, 3, true), [want.clone()]);
        let one = looped(waves, 3, false);
        assert!(one.len() > 1 && one.contains(&want));
    }
}

// ---- A model of the hand-over in walked code ----
//
// A group of host waves of 32 lanes, each a unit that runs its lanes in
// step; two of them are a console wave. The program is the tiled shadow
// programs' count: under the group's flag, the lanes inside EXEC count
// their flagged lanes over the console wave. Every order of the host waves'
// steps is tried.

/// A host wave: the lanes inside EXEC where the count is taken, and their
/// flags (a bit per lane).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Half {
    inside: u32,
    flags: u32,
}

/// What a lane ends with: the count it got, or nothing (it was outside, or
/// the group's flag was off).
type Got = Vec<Option<u32>>;

/// The console: a wave of 64 lanes counts once, and takes the count only
/// where some lane is inside (else the wave has left).
fn console(group: &[Half], flag_on: bool) -> Got {
    let mut out = Vec::new();
    for wave in group.chunks(2) {
        let count: u32 = wave.iter().map(|h| (h.inside & h.flags).count_ones()).sum();
        for h in wave {
            out.extend((0..32).map(|l| (flag_on && h.inside >> l & 1 == 1).then_some(count)));
        }
    }
    out
}

/// How the host waves run the count.
#[derive(Clone, Copy, PartialEq, Eq)]
enum How {
    /// Every lane walks to the count; each half stores its ballot, the
    /// group meets, each half loads both ballots of its console wave.
    Walked,
    /// The same hand-over with the lanes outside EXEC gone round it: a
    /// host wave with no lane inside never comes to the barrier.
    NotWalked,
    /// No hand-over: each half counts its own lanes.
    Local,
}

/// Every result the group can end with, and whether some order of the
/// steps leaves it waiting at the barrier for ever.
fn results(group: &[Half], flag_on: bool, how: How) -> (Vec<Got>, bool) {
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct State {
        /// Per host wave: 0 stores, 1 waits at the barrier, 2 loads, 3 done.
        pcs: Vec<u8>,
        slots: Vec<u32>,
        got: Got,
    }
    let n = group.len();
    let comes = |h: &Half| flag_on && (how != How::NotWalked || h.inside != 0);
    let start = State {
        pcs: group.iter().map(|h| if comes(h) { 0 } else { 3 }).collect(),
        // Stale values, as memory nothing has stored to.
        slots: vec![0x5555_5555; n],
        got: vec![None; 32 * n],
    };
    let (mut seen, mut stack) = (std::collections::HashSet::new(), vec![start]);
    let (mut ends, mut stuck) = (Vec::new(), false);
    while let Some(s) = stack.pop() {
        if !seen.insert(s.clone()) {
            continue;
        }
        if s.pcs.iter().all(|pc| *pc == 3) {
            ends.push(s.got.clone());
            continue;
        }
        let mut moved = false;
        // The barrier lets the group on when every thread has arrived.
        if s.pcs.iter().all(|pc| *pc == 1) {
            let mut next = s.clone();
            next.pcs = vec![2; n];
            stack.push(next);
            moved = true;
        }
        for w in 0..n {
            let mut next = s.clone();
            let ballot = group[w].inside & group[w].flags;
            match (s.pcs[w], how) {
                (0, How::Local) => {
                    for l in 0..32 {
                        if group[w].inside >> l & 1 == 1 {
                            next.got[32 * w + l] = Some(ballot.count_ones());
                        }
                    }
                    next.pcs[w] = 3;
                }
                (0, _) => {
                    next.slots[w] = ballot;
                    next.pcs[w] = 1;
                }
                (2, _) => {
                    let count = s.slots[w & !1].count_ones() + s.slots[w | 1].count_ones();
                    for l in 0..32 {
                        if group[w].inside >> l & 1 == 1 {
                            next.got[32 * w + l] = Some(count);
                        }
                    }
                    next.pcs[w] = 3;
                }
                _ => continue,
            }
            stack.push(next);
            moved = true;
        }
        stuck |= !moved;
    }
    ends.sort();
    ends.dedup();
    (ends, stuck)
}

/// Halves of every kind: no lane inside, all inside with no flag, with
/// every flag, and mixtures.
const HALVES: [Half; 6] = [
    Half {
        inside: 0,
        flags: 0xFFFF_0000,
    },
    Half {
        inside: u32::MAX,
        flags: 0,
    },
    Half {
        inside: u32::MAX,
        flags: u32::MAX,
    },
    Half {
        inside: 0x0000_FFFF,
        flags: 0x0000_10F0,
    },
    Half {
        inside: 0xC000_0000,
        flags: 0x8000_0001,
    },
    Half {
        inside: 0x7FFF_FFFE,
        flags: 1,
    },
];

/// Every group of `waves` host waves made of [`HALVES`].
fn groups(waves: usize) -> Vec<Vec<Half>> {
    let mut all = vec![Vec::new()];
    for _ in 0..waves {
        all = all
            .into_iter()
            .flat_map(|g| HALVES.iter().map(move |h| [&g[..], &[*h]].concat()))
            .collect();
    }
    all
}

/// One console wave as two halves, and two console waves in a group, with
/// the group's flag on and off: walked, every order of the halves' steps
/// gives the console's counts and none waits for ever.
#[test]
fn the_walked_hand_over_gives_the_consoles_count_in_every_order() {
    for waves in [2, 4] {
        for group in groups(waves) {
            for flag_on in [true, false] {
                let (ends, stuck) = results(&group, flag_on, How::Walked);
                assert!(!stuck, "{group:?}");
                assert_eq!(ends, [console(&group, flag_on)], "{group:?} {flag_on}");
            }
        }
    }
}

/// The model can tell: without the hand-over a wave whose flagged lanes
/// are in both halves gets two counts, and the hand-over without the walk
/// waits for ever for a half that has no lane inside.
#[test]
fn without_the_walk_the_count_is_a_halfs_or_the_barrier_never_opens() {
    let both = [HALVES[2], HALVES[3]];
    let (ends, _) = results(&both, true, How::Local);
    assert_ne!(ends, [console(&both, true)]);
    // A wave that needs no hand-over is the console's either way.
    let one = [HALVES[2], HALVES[0]];
    assert_eq!(results(&one, true, How::Local).0, [console(&one, true)]);
    let (_, stuck) = results(&one, true, How::NotWalked);
    assert!(stuck);
    let (ends, stuck) = results(&both, true, How::NotWalked);
    assert!(!stuck && ends == [console(&both, true)]);
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// Runs `words` (one group of `threads`, each lane storing a dword at
    /// `l * 32`) walked on every hardware adapter, 16 times. A line per
    /// lane whose dword is not `want`'s.
    fn differences(what: &str, words: &[u32], threads: u32, want: &[u32]) -> Vec<String> {
        let l = translated(words, threads, true, false, walking()).expect("handed over");
        let shader = super::super::HostShader {
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
        };
        let views = [
            View::Texture(1, 1, &[[0.0; 4]]),
            View::Buffer {
                data: &[0; 4],
                stride: Stride::Structured(16),
            },
            View::RawUav(threads * 8),
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
            // Many times: a hand-over that is only usually right shows.
            for round in 0..16 {
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud.clone()))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                let wrong: Vec<String> = out[0]
                    .chunks(8)
                    .zip(want)
                    .enumerate()
                    .filter(|(_, (got, want))| got[0] != **want)
                    .map(|(l, (got, want))| {
                        format!(
                            "{label}, round {round}: lane {l}: {}, the console {want}",
                            got[0]
                        )
                    })
                    .collect();
                if !wrong.is_empty() {
                    eprintln!("{label}: {} lanes differ in round {round}", wrong.len());
                    failed.extend(wrong);
                    break;
                }
            }
        }
        failed
    }

    /// The count behind an early-out and inside an `if` is the console's in
    /// both halves (23: lanes 41 to 63, of which the lower half holds none)
    /// and the lanes outside keep their 0; a second console wave in the
    /// group counts its own 64.
    #[test]
    fn a_count_behind_a_skip_is_the_consoles_in_both_halves_on_every_gpu() {
        let one: Vec<u32> = (0..64).map(|l| if l < 30 { 0 } else { 23 }).collect();
        let two: Vec<u32> = one.iter().copied().chain([64; 64]).collect();
        let mut failed = Vec::new();
        for (what, words) in [
            ("early-out", behind_an_early_out()),
            ("if", in_an_if()),
            ("if and group branch", in_an_if_under_a_group_branch()),
        ] {
            failed.extend(differences(what, &words, 64, &one));
            let what = format!("{what}, two waves");
            failed.extend(differences(&what, &words, 128, &two));
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// The count in a loop behind an early-out is the console's in both
    /// halves each round (two hand-overs through the same slots), and the
    /// thread that runs alone reads its own first lane.
    #[test]
    fn a_count_in_a_loop_is_the_consoles_in_both_halves_on_every_gpu() {
        let one: Vec<u32> = (0..64).map(|l| if l < 30 { 0 } else { 23 }).collect();
        let two: Vec<u32> = one.iter().copied().chain([64; 64]).collect();
        let words = in_a_loop_behind_an_early_out();
        let mut failed = differences("loop", &words, 64, &one);
        failed.extend(differences("loop, two waves", &words, 128, &two));
        let alone: Vec<u32> = (0..64).map(|l| if l == 0 { 5 } else { 0 }).collect();
        failed.extend(differences("thread 0", &thread_0_reads_its_first_lane(), 64, &alone));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// The count under a branch on the group's flag is the console's in
    /// both halves: 34 lanes above 29 in the first wave, 64 in a second.
    #[test]
    fn a_count_under_a_group_branch_is_the_consoles_on_every_gpu() {
        let words = under_a_group_branch();
        let two: Vec<u32> = [[34; 64], [64; 64]].concat();
        let mut failed = differences("group branch", &words, 64, &[34; 64]);
        failed.extend(differences("group branch, two waves", &words, 128, &two));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
