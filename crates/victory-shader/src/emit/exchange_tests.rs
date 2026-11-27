//! Reads across the two 32-lane halves handed over through group-shared
//! memory (`lower::exchange`, `Options::halves_exchange`;
//! `notes/ray-tracing-plan.md`): lane reads, first-lane reads, lane masks
//! read as data and integer reductions. The translation tests and the model of the hand-over
//! run on any host. The GPU test (Windows) runs the programs in halves with
//! the setting on every hardware adapter and compares with the console's
//! values.
//!
//! The programs have the shape of the prefix sums in the title's ray-tracing
//! structure builder (f6eb6498409ca974 and four more): each half sums its own
//! lanes, then the lower half's total is read from lane 31 and added in the
//! upper half.

use super::halves_local_tests::{
    PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, k, native, sop1, store, v,
};
use super::halves_local_tests::{across, vop2, vop3};
use super::lower::{self, HalvesLocal, Mac};
use crate::{gfx10, io, ir, resource, structure};

const EXEC: u8 = 126;
const V_ADD_NC_U32: u8 = 0x25;
const V_READLANE_B32: u16 = 0x360;
const V_CNDMASK_B32: u16 = 0x101;
const REFUSAL: &str = "a lane read that may cross 32-lane halves (wave64 on a 32-lane host)";

fn exchange() -> Mac {
    Mac {
        exchange: true,
        uav_level0: false,
        ..Mac::default()
    }
}

/// `v_readlane_b32 s<dst>, v<src>, <lane>`.
const fn readlane(dst: u8, src: u16, lane: u16) -> [u32; 2] {
    vop3(dst, V_READLANE_B32, v(src), k(lane), 0)
}

/// `v_mov_b32 v<dst>, s<src>`.
const fn v_mov_s(dst: u8, src: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | src
}

/// A read of one lane: every thread stores the thread id of lane `lane` of
/// its wave (on the console `lane` in the first wave, `64 + lane` in the
/// second).
fn one_read(lane: u16) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.extend(readlane(12, 0, lane));
    w.push(v_mov_s(1, 12));
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Two reads of one value: the thread ids of lanes 40 and 7.
fn one_value_twice() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.extend(readlane(12, 0, 40));
    w.extend(readlane(13, 0, 7));
    w.extend([v_mov_s(1, 12), v_mov_s(2, 13)]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// Reads of three values in turn: the thread id at lane 40, twice the
/// thread id at lane 31, and their per-lane sum at lane 63 (on the console
/// 40, 62 and 63 + 126).
fn three_values() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.push(vop2(2, V_ADD_NC_U32, v(0), 0)); // v_add_nc_u32 v2, v0, v0
    w.push(vop2(3, V_ADD_NC_U32, v(2), 0)); // v_add_nc_u32 v3, v2, v0
    w.extend(readlane(12, 0, 40));
    w.extend(readlane(13, 2, 31));
    w.extend(readlane(14, 3, 63));
    w.extend([v_mov_s(4, 12), v_mov_s(5, 13), v_mov_s(6, 14)]);
    w.extend(store(4, 0));
    w.extend(store(5, 4));
    w.extend(store(6, 8));
    w.push(S_ENDPGM);
    w
}

/// The end of the builder's prefix sum: lane 31's value, added in the
/// lanes of the upper half only (the mask `s[6:7]` is 0 and all ones). On
/// the console lane `l` stores `l` below 32 and `l + 31` from 32 on.
fn upper_half_adds_lane_31() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.push(sop1(6, 0x03, 128)); // s_mov_b32 s6, 0
    w.push(sop1(7, 0x03, 193)); // s_mov_b32 s7, -1
    w.extend(readlane(8, 0, 31));
    w.extend(vop3(6, V_CNDMASK_B32, k(0), 8, 6)); // v_cndmask_b32 v6, 0, s8, s[6:7]
    w.push(vop2(1, V_ADD_NC_U32, v(6), 0)); // v_add_nc_u32 v1, v6, v0
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A lane mask held as a number, its two dwords written by two
/// instructions: the upper half's lanes (`s[6:7]` is 0 and all ones) take
/// their thread id, the lower half's 0. (`v_mbcnt` makes it a program of
/// console lanes, as the lane operations around such a mask do.)
fn numeric_mask() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.extend([0xD765_0001, 0x0001_00C1]); // v_mbcnt_lo_u32_b32 v1, -1, 0
    w.push(sop1(6, 0x03, 128)); // s_mov_b32 s6, 0
    w.push(sop1(7, 0x03, 193)); // s_mov_b32 s7, -1
    w.extend(vop3(6, V_CNDMASK_B32, k(0), v(0), 6)); // v_cndmask_b32 v6, 0, v0, s[6:7]
    w.extend(store(6, 0));
    w.push(S_ENDPGM);
    w
}

/// A read that only some lanes reach: the lanes whose thread id is not 5
/// read lane 40 under a branch.
fn under_a_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0x7E02_0280,         // v_mov_b32 v1, 0
        sop1(2, 0x04, EXEC), // s_mov_b64 s[2:3], exec
        0x7DAA_0085,         // v_cmpx_ne_u32 exec, 5, v0
        0xBF88_0003,         // s_cbranch_execz +3
    ]);
    w.extend(readlane(12, 0, 40));
    w.push(v_mov_s(1, 12));
    w.push(sop1(EXEC, 0x04, 2)); // s_mov_b64 exec, s[2:3]
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 6] = [
        (
            numeric_mask(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "s_mov_b32 s6, 0",
                "s_mov_b32 s7, -1",
                "v_cndmask_b32 v6, 0, v0, s[6:7]",
            ],
        ),
        (
            one_read(40),
            &["v_readlane_b32 s12, v0, 40", "v_mov_b32 v1, s12"],
        ),
        (
            one_value_twice(),
            &[
                "v_readlane_b32 s12, v0, 40",
                "v_readlane_b32 s13, v0, 7",
                "v_mov_b32 v2, s13",
            ],
        ),
        (
            three_values(),
            &[
                "v_add_nc_u32 v2, v0, v0",
                "v_add_nc_u32 v3, v2, v0",
                "v_readlane_b32 s13, v2, 31",
                "v_readlane_b32 s14, v3, 63",
            ],
        ),
        (
            upper_half_adds_lane_31(),
            &[
                "s_mov_b32 s6, 0",
                "s_mov_b32 s7, -1",
                "v_readlane_b32 s8, v0, 31",
                "v_cndmask_b32 v6, 0, s8, s[6:7]",
                "v_add_nc_u32 v1, v6, v0",
            ],
        ),
        (
            under_a_branch(),
            &[
                "s_mov_b64 s[2:3], exec",
                "v_cmpx_ne_u32 exec, 5, v0",
                "s_cbranch_execz",
                "v_readlane_b32 s12, v0, 40",
                "s_mov_b64 exec, s[2:3]",
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

/// `words` as a wave64 compute program of one row of `threads` with
/// `lds_bytes` of LDS, lowered for a host of 32-lane waves (`host_wave32`)
/// or of 64.
fn lowered(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    host_wave32: bool,
    reads: bool,
    mac: Mac,
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
        halves_local: HalvesLocal { reads, lds: false },
        mac,
        host_wave32,
    })
}

fn halves(words: &[u32], reads: bool, mac: Mac) -> Result<lower::Lowered, crate::ShaderError> {
    lowered(words, 64, 0, true, reads, mac)
}

/// Without the setting a lane read in halves is not this line's to answer
/// (the Mac port's hand-over does, since the merge of its #360; before
/// that it was refused). With it the program translates in waves of 32, with half-local
/// reads or without: the read is the console's either way, so the two
/// translations are the same.
#[test]
fn a_lane_read_where_every_lane_runs_translates_with_the_setting() {
    for words in [
        one_read(40),
        one_read(31),
        one_value_twice(),
        three_values(),
        upper_half_adds_lane_31(),
    ] {
        // Without the setting the read was refused ([`REFUSAL`]) until
        // VIctory-macOS's own hand-over was merged (its #360,
        // `lower::halves`): that answers now, with none of this line's
        // exchanges. With the setting this line's exchange comes first.
        let off = halves(&words, false, Mac::default()).expect("the Mac port's hand-over");
        assert!(off.port_exchange && off.exchanges == 0);
        let on = halves(&words, false, exchange()).expect("handed over");
        assert_eq!(on.wave_size, Some(32));
        assert!(on.exchanges > 0);
        assert!(!on.port_exchange);
        let with_local = halves(&words, true, exchange()).expect("handed over");
        assert_eq!(with_local.dxil, on.dxil);
        // Half-local reads alone are another translation: the half's answer.
        let local = halves(&words, true, Mac::default()).expect("half-local");
        assert_eq!(local.exchanges, 0);
        assert_ne!(local.dxil, on.dxil);
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut on.dxil.clone())
                .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
        }
    }
}

/// One hand-over per value: reads of the same value one after the other
/// share it, each other value has its own.
#[test]
fn each_value_is_handed_over_once() {
    let count = |words: &[u32]| {
        halves(words, false, exchange())
            .expect("handed over")
            .exchanges
    };
    assert_eq!(count(&one_read(40)), 1);
    assert_eq!(count(&one_value_twice()), 1);
    assert_eq!(count(&three_values()), 3);
    assert_eq!(count(&upper_half_adds_lane_31()), 1);
}

/// A read that only some lanes reach gets no hand-over (no barrier can go
/// there): refused as before, and with half-local reads the translation
/// that had before.
#[test]
fn a_lane_read_in_divergent_code_is_left_as_it_was() {
    let words = under_a_branch();
    for mac in [Mac::default(), exchange()] {
        let why = halves(&words, false, mac)
            .err()
            .expect("refused")
            .to_string();
        assert!(why.contains(REFUSAL), "{why}");
    }
    let dxil = |mac| halves(&words, true, mac).expect("half-local").dxil;
    assert_eq!(dxil(exchange()), dxil(Mac::default()));
}

/// The setting changes nothing on a host of 64-lane waves, in a group of
/// several console waves it hands over per wave, and where the slots do
/// not fit beside the program's LDS in a group's shared memory the read is
/// refused as before.
#[test]
fn the_setting_acts_only_in_halves_and_within_shared_memory() {
    let words = one_read(40);
    let whole = |mac| {
        lowered(&words, 64, 0, false, false, mac)
            .expect("one wave64")
            .dxil
    };
    assert_eq!(whole(exchange()), whole(Mac::default()));
    let two_waves = lowered(&words, 128, 0, true, false, exchange()).expect("two console waves");
    assert_eq!((two_waves.wave_size, two_waves.exchanges), (Some(32), 1));
    // 1,024 threads: two sets of 1,024 dwords, 8,192 bytes. They fit
    // beside 24,576 bytes of LDS and not beside 25,088.
    let beside = |lds_bytes| lowered(&words, 1024, lds_bytes, true, false, exchange());
    assert_eq!(beside(24576).expect("fits").exchanges, 1);
    // Past that this line's slots do not fit. The read was refused there
    // ([`REFUSAL`]) until the Mac port's hand-over was merged: its two
    // dwords a console wave fit, and it answers.
    let full = beside(25088).expect("the Mac port's hand-over");
    assert!(full.port_exchange && full.exchanges == 0);
    // With neither there is the refusal: half-local reads off, and LDS
    // that leaves no room for the Mac port's dwords either.
    let none = beside(32768).err().expect("refused").to_string();
    assert!(none.contains("no group-shared memory left"), "{none}");
}

/// A first-lane read under a narrowed EXEC: the lanes above 29 are left
/// (no branch), and the first of them is read (lane 30 on the console).
fn first_lane_of_some() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        0xD765_0001,
        0x0001_00C1,         // v_mbcnt_lo_u32_b32 v1, -1, 0
        sop1(2, 0x04, EXEC), // s_mov_b64 s[2:3], exec
        0x7DA2_009D,         // v_cmpx_lt_u32 exec, 29, v0
        0x7E18_0500,         // v_readfirstlane_b32 s12, v0
        sop1(EXEC, 0x04, 2), // s_mov_b64 exec, s[2:3]
        v_mov_s(2, 12),
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// A lane mask read as data, a wave sum and a wave prefix sum, each where
/// every lane runs, translate with the setting, each with one hand-over of
/// a dword per half: the same translation with half-local reads or
/// without. So does a first-lane read where EXEC is every lane (lane 0's
/// value, a hand-over per lane); under a narrowed EXEC it stays refused.
#[test]
fn ballots_and_integer_reductions_are_handed_over_per_half() {
    let narrowed = first_lane_of_some();
    let code: Vec<u8> = narrowed.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect();
    assert!(
        text.contains(&"v_cmpx_lt_u32 exec, 29, v0".to_owned()),
        "{text:?}"
    );
    assert!(
        text.contains(&"v_readfirstlane_b32 s12, v0".to_owned()),
        "{text:?}"
    );
    // Not this line's to hand over, with the setting or without (the first
    // lane with its EXEC bit set is not known here). It was refused ("a
    // first-lane read of a value that may differ") until the Mac port's
    // hand-over was merged; that answers now, with the lower half's first
    // lane, which is what a host of 64-lane waves reads too.
    for mac in [Mac::default(), exchange()] {
        let l = halves(&narrowed, false, mac).expect("the Mac port's hand-over");
        assert!(l.port_exchange && l.exchanges == 0);
    }
    for (what, words, refusal) in across() {
        // Refused with `refusal` without the setting, until the Mac port's
        // hand-over was merged: that answers now.
        let _ = refusal;
        let off = halves(&words, false, Mac::default())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(off.port_exchange && off.exchanges == 0, "{what}");
        let on = halves(&words, false, exchange());
        let on = on.unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!((on.wave_size, on.exchanges), (Some(32), 1), "{what}");
        assert!(!on.port_exchange, "{what}: this line's exchange comes first");
        assert_ne!(on.dxil, off.dxil, "{what}");
        let with_local = halves(&words, true, exchange()).expect("handed over");
        assert_eq!(with_local.dxil, on.dxil, "{what}");
        let local = halves(&words, true, Mac::default()).expect("half-local");
        assert_ne!(local.dxil, on.dxil, "{what}");
        // A host of 64-lane waves answers by itself.
        let whole = |mac| {
            lowered(&words, 64, 0, false, false, mac)
                .expect("one wave64")
                .dxil
        };
        assert_eq!(whole(exchange()), whole(Mac::default()), "{what}");
        // A group that ends in a lower half (96 threads) translates too.
        let cut = lowered(&words, 96, 0, true, false, exchange());
        assert_eq!(cut.unwrap_or_else(|e| panic!("{what}: {e}")).exchanges, 1);
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut on.dxil.clone())
                .unwrap_or_else(|e| panic!("{what}: DXIL validation failed: {e}"));
        }
    }
}

/// With half-local reads the Mac port's hand-over (`lower::halves`) comes
/// before them only with its setting ([`Mac::across`],
/// `VICTORY_HALVES_ACROSS=1`): the translation is then the one without
/// half-local reads. Without the setting half-local reads answer as before
/// that hand-over was merged, and this line's exchange comes before either.
#[test]
fn the_ports_hand_over_comes_before_half_local_reads_only_when_asked() {
    let asked = Mac {
        across: true,
        ..Mac::default()
    };
    for (what, words, _) in across() {
        let lower = |reads, mac| halves(&words, reads, mac).unwrap_or_else(|e| panic!("{what}: {e}"));
        let (port, local) = (lower(false, Mac::default()), lower(true, Mac::default()));
        assert!(port.port_exchange && !local.port_exchange, "{what}");
        assert_ne!(port.dxil, local.dxil, "{what}");
        let first = lower(true, asked);
        assert!(first.port_exchange, "{what}");
        assert_eq!(first.dxil, port.dxil, "{what}");
        // Without half-local reads the setting changes nothing.
        assert_eq!(lower(false, asked).dxil, port.dxil, "{what}");
        // This line's exchange still comes first.
        let ours = lower(true, Mac { across: true, ..exchange() });
        assert!(!ours.port_exchange && ours.exchanges == 1, "{what}");
        assert_eq!(ours.dxil, lower(true, exchange()).dxil, "{what}");
    }
}

/// A lane read, a mask read as data and another lane read: the two lane
/// reads use the two sets of slots in turn whatever is handed over per half
/// between them (that exchange has slots of its own and is not counted in
/// the turn).
#[test]
fn lane_reads_take_turns_whatever_is_handed_over_between() {
    let mut w = PROLOGUE.to_vec();
    w.push(V16_IS_32_TIMES_V0);
    w.push(vop2(2, V_ADD_NC_U32, v(0), 0)); // v_add_nc_u32 v2, v0, v0
    w.extend(readlane(12, 0, 40));
    w.extend([
        0x7D82_009D, // v_cmp_lt_u32 vcc, 29, v0
        0x7E06_026A, // v_mov_b32 v3, vcc_lo
    ]);
    w.extend(readlane(13, 2, 31));
    w.extend([v_mov_s(4, 12), v_mov_s(5, 13)]);
    w.extend(store(4, 0));
    w.extend(store(3, 4));
    w.extend(store(5, 8));
    w.push(S_ENDPGM);
    let l = halves(&w, false, exchange()).expect("handed over");
    assert_eq!((l.exchanges, l.lane_exchanges), (3, 2));
}

/// A numeric mask whose dwords two instructions write is known as a pair
/// from its use, and with the setting the lane tests the dword of its own
/// half instead of the low one for every lane: another translation, in
/// halves and on a host of 64-lane waves alike. (The builder's prefix sums
/// end with this mask: without the high dword the upper half never adds
/// the lower half's total.)
#[test]
fn a_numeric_mask_written_a_dword_at_a_time_keeps_its_high_dword() {
    for words in [numeric_mask(), upper_half_adds_lane_31()] {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 32).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        assert_eq!(st.other_highs.len(), 1);
    }
    let words = numeric_mask();
    for host_wave32 in [true, false] {
        let dxil = |mac| {
            lowered(&words, 64, 0, host_wave32, false, mac)
                .expect("translates")
                .dxil
        };
        assert_ne!(dxil(exchange()), dxil(Mac::default()));
    }
}

// ---- A model of the hand-over ----
//
// A group of host waves, each a unit that runs its lanes in step. A program
// of several exchanges: per exchange every lane stores its value in its own
// slot of a set, the group meets at a barrier, every lane loads the slot of
// the lane read. Every order of the waves' steps is tried.

/// Where a wave is: the exchange, and its step in it (0 store, 1 at the
/// barrier, 2 load).
type Pc = (usize, u8);

/// The console's value of `lane` at exchange `e`.
fn value(e: usize, lane: usize) -> u32 {
    (100 * (e + 1) + lane) as u32
}

/// Every result the group can end with: per exchange and lane the value
/// loaded. `waves` host waves of `lanes` lanes each; exchange `e` reads lane
/// `reads[e]` and uses set `e % sets`.
fn results(waves: usize, lanes: usize, reads: &[usize], sets: usize) -> Vec<Vec<u32>> {
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct Group {
        pcs: Vec<Pc>,
        slots: Vec<u32>,
        got: Vec<u32>,
    }
    let all = waves * lanes;
    let start = Group {
        pcs: vec![(0, 0); waves],
        // Stale values, as memory nothing has stored to.
        slots: vec![7; sets * all],
        got: vec![0; reads.len() * all],
    };
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start];
    let mut ends = Vec::new();
    while let Some(g) = stack.pop() {
        if !seen.insert(g.clone()) {
            continue;
        }
        if g.pcs.iter().all(|pc| pc.0 == reads.len()) {
            ends.push(g.got.clone());
            continue;
        }
        // The barrier lets the group on when every wave has arrived at it.
        if let Some(&(e, 1)) = g.pcs.first()
            && g.pcs.iter().all(|pc| *pc == (e, 1))
        {
            let mut n = g.clone();
            n.pcs = vec![(e, 2); waves];
            stack.push(n);
            continue;
        }
        for w in 0..waves {
            let (e, step) = g.pcs[w];
            if e == reads.len() || step == 1 {
                continue;
            }
            let mut n = g.clone();
            let set = (e % sets) * all;
            for l in w * lanes..(w + 1) * lanes {
                if step == 0 {
                    n.slots[set + l] = value(e, l);
                } else {
                    n.got[e * all + l] = n.slots[set + reads[e]];
                }
            }
            n.pcs[w] = if step == 0 { (e, 1) } else { (e + 1, 0) };
            stack.push(n);
        }
    }
    ends.sort();
    ends.dedup();
    ends
}

/// The console's result: every lane gets the read lane's value.
fn console(all: usize, reads: &[usize]) -> Vec<u32> {
    (0..reads.len())
        .flat_map(|e| std::iter::repeat(value(e, reads[e])).take(all))
        .collect()
}

/// Two halves and four exchanges that read lanes of both halves: with two
/// sets used in turn every order of the halves' steps gives the console's
/// values. With one set some order does not: a half that is past the
/// barrier stores the next exchange's value over a slot the other half has
/// yet to load.
#[test]
fn two_sets_in_turn_give_the_consoles_values_in_every_order() {
    let reads = [3, 0, 2, 1];
    assert_eq!(results(2, 2, &reads, 2), vec![console(4, &reads)]);
    let one_set = results(2, 2, &reads, 1);
    assert!(one_set.contains(&console(4, &reads)));
    assert!(one_set.len() > 1, "one set is enough?");
    // One exchange alone needs no second set.
    assert_eq!(results(2, 2, &[3], 1), vec![console(4, &[3])]);
    // Nor do exchanges with slots of their own (a half's answer).
    assert_eq!(results(2, 2, &reads, 4), vec![console(4, &reads)]);
}

/// The same with a group of two console waves in four host waves, and with
/// a console wave in four quarter waves.
#[test]
fn more_host_waves_give_the_consoles_values_in_every_order() {
    let reads = [3, 0, 2];
    assert_eq!(results(4, 1, &reads, 2), vec![console(4, &reads)]);
    assert_eq!(results(3, 2, &[5, 0, 3], 2), vec![console(6, &[5, 0, 3])]);
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// Runs `words` (one group of `threads`, each lane storing `per_lane`
    /// dwords at `l * 32`) in halves with the setting on every hardware
    /// adapter, 16 times. A line per lane whose dwords are not `want`'s.
    fn differences(
        what: &str,
        words: &[u32],
        threads: u32,
        per_lane: usize,
        want: &[u32],
    ) -> Vec<String> {
        let l = lowered(words, threads, 0, true, false, exchange()).expect("handed over");
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
                    .zip(want.chunks(per_lane))
                    .enumerate()
                    .filter(|(_, (got, want))| got[..per_lane] != **want)
                    .map(|(l, (got, want))| {
                        format!(
                            "{label}, round {round}: lane {l}: {:?}, the console {want:?}",
                            &got[..per_lane]
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

    /// Every lane reads what the console's lane holds, in either half:
    /// lane 40 from the lower half and lane 31 from the upper, one value at
    /// two lanes, three values in turn (both sets of slots, the first one
    /// twice), the builder's upper half adding lane 31, and two console
    /// waves in one group each reading its own lane 40.
    #[test]
    fn both_halves_read_the_consoles_lane_on_every_gpu() {
        let lanes = |n: u32, f: &dyn Fn(u32) -> Vec<u32>| (0..n).flat_map(f).collect::<Vec<u32>>();
        let mut failed = Vec::new();
        failed.extend(differences("lane 40", &one_read(40), 64, 1, &[40; 64]));
        failed.extend(differences("lane 31", &one_read(31), 64, 1, &[31; 64]));
        let want = lanes(64, &|_| vec![40, 7]);
        failed.extend(differences(
            "one value twice",
            &one_value_twice(),
            64,
            2,
            &want,
        ));
        let want = lanes(64, &|_| vec![40, 62, 189]);
        failed.extend(differences("three values", &three_values(), 64, 3, &want));
        let want = lanes(64, &|l| vec![if l < 32 { l } else { l + 31 }]);
        failed.extend(differences(
            "upper half adds lane 31",
            &upper_half_adds_lane_31(),
            64,
            1,
            &want,
        ));
        let want = lanes(128, &|l| vec![(l & !63) + 40]);
        failed.extend(differences(
            "two console waves",
            &one_read(40),
            128,
            1,
            &want,
        ));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// A half's answer handed over: the mask of `29 < l` is the console's
    /// two dwords in every lane; the sum of the thread ids is 2,016 in both
    /// halves; the prefix sum counts from lane 0 in the upper half too. In
    /// a group of 96 threads the second console wave has a lower half only:
    /// its sum is that half's (64 to 95: 2,544), its mask has no high bits.
    #[test]
    fn both_halves_get_the_consoles_mask_and_reductions_on_every_gpu() {
        let lanes = |n: u32, f: &dyn Fn(u32) -> Vec<u32>| (0..n).flat_map(f).collect::<Vec<u32>>();
        let [mask, _, first, sum, prefix] = across().map(|p| p.1);
        let mut failed = Vec::new();
        // The first lane is lane 0 of the console wave in both halves.
        failed.extend(differences("first lane", &first, 64, 1, &[0; 64]));
        let want = lanes(128, &|l| vec![l & !63]);
        failed.extend(differences("first lane, two waves", &first, 128, 1, &want));
        let want = lanes(64, &|_| vec![0xC000_0000, u32::MAX]);
        failed.extend(differences("mask", &mask, 64, 2, &want));
        failed.extend(differences("sum", &sum, 64, 1, &[2016; 64]));
        let want = lanes(64, &|l| vec![(0..l).sum()]);
        failed.extend(differences("prefix sum", &prefix, 64, 1, &want));
        let want = lanes(96, &|l| vec![if l < 64 { 2016 } else { 2544 }]);
        failed.extend(differences("sum, a wave cut short", &sum, 96, 1, &want));
        let want = lanes(96, &|l| {
            if l < 64 {
                vec![0xC000_0000, u32::MAX]
            } else {
                vec![u32::MAX, 0]
            }
        });
        failed.extend(differences("mask, a wave cut short", &mask, 96, 2, &want));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// A numeric mask of the upper half, its dwords written one at a time:
    /// lanes 32 to 63 take their thread id, the others 0.
    #[test]
    fn a_numeric_mask_selects_the_upper_half_on_every_gpu() {
        let want: Vec<u32> = (0..64).map(|l| if l < 32 { 0 } else { l }).collect();
        let failed = differences("numeric mask", &numeric_mask(), 64, 1, &want);
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
