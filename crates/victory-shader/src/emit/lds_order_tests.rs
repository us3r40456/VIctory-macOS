//! The exact order of LDS that a group of one console wave shares
//! (`lower::lds_order`, `Options::lds_order`; `notes/cross-half-programs.md`,
//! "Shared LDS"). The translation tests and the CPU model of the two halves
//! run on any host. The GPU test (Windows) runs the programs in halves with
//! the setting on every hardware adapter and compares with the console's
//! values.
//!
//! The programs have the shape of the title's tile classifier
//! (2cc4c197ee29e291): lane 0 clears a dword, lanes that find something set
//! it, sometimes in a loop the others leave when they see it set, and then
//! every lane reads it.

use super::halves_local_tests::{PROLOGUE, V16_IS_32_TIMES_V0, native, sop1, store};
use super::lower::{self, HalvesLocal, Mac};
use crate::{gfx10, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F;
const EXEC: u8 = 126;
const ORDER: Mac = Mac {
    early_depth: false,
    lock_helpers: false,
    lock_keys: false,
    lds_order: true,
    exchange: false,
    uav_level0: false,
    walk: false,
    across: false,
};

/// `v_mov_b32 v<dst>, <k>`.
const fn v_mov(dst: u8, k: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | (128 + k)
}

/// `v_cmpx_eq_u32 exec, <k>, v0` and `v_cmpx_ne_u32 exec, <k>, v<reg>`.
const fn cmpx_eq_v0(k: u32) -> u32 {
    0x7DA4_0000 | (128 + k)
}
const fn cmpx_ne(k: u32, reg: u32) -> u32 {
    0x7DAA_0000 | reg << 9 | (128 + k)
}

/// A DS instruction: `op` on the dword at `v<addr> + offset`, with data
/// `v<data>` (and a second, for `ds_write2`) and result `v<dst>`.
const fn ds(op: u32, offset: u32, addr: u8, data: u8, data1: u8, dst: u8) -> [u32; 2] {
    [
        0xD800_0000 | op << 18 | offset,
        (dst as u32) << 24 | (data1 as u32) << 16 | (data as u32) << 8 | addr as u32,
    ]
}
const DS_ADD_U32: u32 = 0;
const DS_MIN_U32: u32 = 7;
const DS_MAX_U32: u32 = 8;
const DS_WRITE: u32 = 13;
const DS_WRITE2: u32 = 14;
const DS_ADD_RTN: u32 = 32;
const DS_READ: u32 = 54;

const fn sop2(op: u32, dst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | op << 23 | (dst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
const S_ADD_I32: u32 = 2;
const S_AND_B64: u32 = 15;
const S_ANDN2_B64: u32 = 21;

/// `s_cbranch_execz` and `s_cbranch_scc0` by `words` from the next word.
const fn execz(words: i16) -> u32 {
    0xBF88_0000 | words as u16 as u32
}
const fn scc0(words: i16) -> u32 {
    0xBF84_0000 | words as u16 as u32
}

/// The start: the store address, EXEC saved in `s[2:3]`, and lane 0 (the
/// thread with id 0) storing 0 to dwords 0 and 1, in the entry block.
fn cleared() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        sop1(2, 0x04, EXEC), // s_mov_b64 s[2:3], exec
        cmpx_eq_v0(0),
        v_mov(2, 0),
    ]);
    w.extend(ds(DS_WRITE2, 1 << 8, 2, 2, 2, 0)); // ds_write2_b32 v2, v2, v2 offset1:1
    w.push(sop1(EXEC, 0x04, 2)); // s_mov_b64 exec, s[2:3]
    w
}

/// The end: every lane reads dword `dword` and stores it at its address.
fn all_read(w: &mut Vec<u32>, dword: u32) {
    w.extend([sop1(EXEC, 0x04, 2), v_mov(4, 0)]);
    w.extend(ds(DS_READ, 4 * dword, 4, 0, 0, 4)); // ds_read_b32 v4, v4 offset:<4 dword>
    w.push(S_WAITCNT_LGKM);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
}

/// Writers before, readers after: lane 40 sets dword 1 under a branch, then
/// every lane reads it (1 in every lane on the console).
fn writers_then_readers() -> Vec<u32> {
    let mut w = cleared();
    w.extend([cmpx_eq_v0(40), execz(4), v_mov(6, 0), v_mov(7, 1)]);
    w.extend(ds(DS_WRITE, 4, 6, 7, 0, 0)); // ds_write_b32 v6, v7 offset:4
    all_read(&mut w, 1);
    w
}

/// The flag loop: up to four times, lane 40 sets dword 0 and the other
/// lanes read it and leave once it is set; then every lane reads it (1 in
/// every lane on the console, where all leave in the first round). With
/// `leak` the last store is of what a lane read in the loop instead.
fn flag_loop(leak: bool) -> Vec<u32> {
    let mut w = cleared();
    w.extend([
        sop1(8, 0x03, 128),  // s_mov_b32 s8, 0
        sop1(4, 0x04, EXEC), // s_mov_b64 s[4:5], exec
    ]);
    let top = w.len();
    w.extend([
        sop1(6, 0x04, EXEC),
        cmpx_eq_v0(40),
        v_mov(6, 1),
        v_mov(8, 0),
    ]);
    w.extend(ds(DS_WRITE, 0, 8, 6, 0, 0)); // ds_write_b32 v8, v6
    w.extend([sop2(S_ANDN2_B64, EXEC, 6, EXEC), v_mov(6, 0)]);
    w.extend(ds(DS_READ, 0, 6, 0, 0, 6)); // ds_read_b32 v6, v6
    w.extend([
        S_WAITCNT_LGKM,
        sop1(EXEC, 0x04, 6), // s_mov_b64 exec, s[6:7]
        cmpx_ne(0, 6),
        sop2(S_ANDN2_B64, 4, 4, EXEC), // s_andn2_b64 s[4:5], s[4:5], exec
        scc0(4),
        sop2(S_AND_B64, EXEC, 6, 4), // s_and_b64 exec, s[6:7], s[4:5]
        sop2(S_ADD_I32, 8, 8, 129),  // s_add_i32 s8, s8, 1
        0xBF02_8308,                 // s_cmp_gt_i32 s8, 3
    ]);
    w.push(scc0(top as i16 - (w.len() as i16 + 1)));
    if leak {
        w.push(sop1(EXEC, 0x04, 2));
        w.extend(store(6, 0));
        w.push(S_ENDPGM);
    } else {
        all_read(&mut w, 0);
    }
    w
}

/// A maximum: the lanes other than lane 5 raise dword 0 to their thread id
/// under a branch (`also_min`: and lower dword 0 to it), then every lane
/// reads it (63 in every lane on the console).
fn group_maximum(also_min: bool) -> Vec<u32> {
    let mut w = cleared();
    w.extend([
        cmpx_ne(5, 0),
        execz(if also_min { 5 } else { 3 }),
        v_mov(6, 0),
    ]);
    w.extend(ds(DS_MAX_U32, 0, 6, 0, 0, 0)); // ds_max_u32 v6, v0
    if also_min {
        w.extend(ds(DS_MIN_U32, 0, 6, 0, 0, 0)); // ds_min_u32 v6, v0
    }
    all_read(&mut w, 0);
    w
}

/// Clear, add and read all under one branch: no block every lane runs lies
/// between them.
fn all_under_one_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        V16_IS_32_TIMES_V0,
        sop1(2, 0x04, EXEC),
        cmpx_ne(5, 0),
        execz(8),
        v_mov(6, 0),
    ]);
    w.extend(ds(DS_WRITE, 0, 6, 6, 0, 0)); // ds_write_b32 v6, v6
    w.extend(ds(DS_ADD_U32, 0, 6, 0, 0, 0)); // ds_add_u32 v6, v0
    w.extend(ds(DS_READ, 0, 6, 0, 0, 4)); // ds_read_b32 v4, v6
    w.push(S_WAITCNT_LGKM);
    w.push(sop1(EXEC, 0x04, 2));
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// An atomic that hands each lane the value before its own addition.
fn counted() -> Vec<u32> {
    let mut w = cleared();
    w.push(v_mov(6, 0));
    w.extend(ds(DS_ADD_RTN, 0, 6, 0, 0, 4)); // ds_add_rtn_u32 v4, v6, v0
    w.push(S_WAITCNT_LGKM);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 5] = [
        (
            writers_then_readers(),
            &[
                "s_mov_b64 s[2:3], exec\nv_cmpx_eq_u32 exec, 0, v0\nv_mov_b32 v2, 0\n\
                 ds_write2_b32 v2, v2, v2 offset1:1\ns_mov_b64 exec, s[2:3]\n\
                 v_cmpx_eq_u32 exec, 40, v0\ns_cbranch_execz",
                "v_mov_b32 v6, 0\nv_mov_b32 v7, 1\nds_write_b32 v6, v7 offset:4\n\
                 s_mov_b64 exec, s[2:3]\nv_mov_b32 v4, 0\nds_read_b32 v4, v4 offset:4",
            ],
        ),
        (
            flag_loop(false),
            &[
                "s_mov_b32 s8, 0\ns_mov_b64 s[4:5], exec\ns_mov_b64 s[6:7], exec\n\
                 v_cmpx_eq_u32 exec, 40, v0\nv_mov_b32 v6, 1\nv_mov_b32 v8, 0\n\
                 ds_write_b32 v8, v6\ns_andn2_b64 exec, s[6:7], exec\nv_mov_b32 v6, 0\n\
                 ds_read_b32 v6, v6",
                "s_mov_b64 exec, s[6:7]\nv_cmpx_ne_u32 exec, 0, v6\n\
                 s_andn2_b64 s[4:5], s[4:5], exec\ns_cbranch_scc0",
                "s_and_b64 exec, s[6:7], s[4:5]\ns_add_i32 s8, s8, 1\ns_cmp_gt_i32 s8, 3\n\
                 s_cbranch_scc0",
                "s_mov_b64 exec, s[2:3]\nv_mov_b32 v4, 0\nds_read_b32 v4, v4",
            ],
        ),
        (
            group_maximum(true),
            &[
                "v_cmpx_ne_u32 exec, 5, v0\ns_cbranch_execz",
                "ds_max_u32 v6, v0\nds_min_u32 v6, v0",
            ],
        ),
        (
            all_under_one_branch(),
            &["ds_write_b32 v6, v6\nds_add_u32 v6, v0\nds_read_b32 v4, v6"],
        ),
        (counted(), &["ds_add_rtn_u32 v4, v6, v0"]),
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

/// `words` as a wave64 compute program of one row of 64 threads with 512
/// bytes of LDS, lowered for a host of 32-lane waves.
fn lowered(
    words: &[u32],
    mac: Mac,
    narrow_waves: bool,
    halves_local: HalvesLocal,
) -> Result<lower::Lowered, crate::ShaderError> {
    let native = native(words, 64, 512);
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
        narrow_waves,
        halves_local,
        mac,
        host_wave32: true,
    })
}

/// The barriers the analysis places, as (the instruction's name, or "end"
/// for a block's end), in program order; or why it places none.
fn placed(words: &[u32]) -> Result<Vec<String>, &'static str> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 32).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let mut at: Vec<_> = lower::lds_order::barriers(&p, &st)?.into_iter().collect();
    at.sort_unstable_by_key(|&(b, i)| (p.blocks[b].offset, i));
    Ok(at
        .into_iter()
        .map(|(b, i)| match p.blocks[b].insts.get(i) {
            Some(inst) => inst.name().to_owned(),
            None => "end".to_owned(),
        })
        .collect())
}

fn refusal(words: &[u32], mac: Mac) -> String {
    match lowered(words, mac, false, HalvesLocal::default()) {
        Ok(_) => "translated".into(),
        Err(e) => e.to_string(),
    }
}

/// Barriers go where every lane runs: after the clear in the entry block
/// (divergent code follows that sets what it cleared) and before the read
/// at the join. The flag loop's own read gets none and is left as it is.
#[test]
fn barriers_go_after_the_clear_and_before_the_read_every_lane_makes() {
    for words in [
        writers_then_readers(),
        flag_loop(false),
        group_maximum(false),
    ] {
        assert_eq!(placed(&words).unwrap(), ["end", "ds_read_b32"]);
    }
}

/// What has no exact order with barriers alone stays out, each with its
/// reason: a flag read whose value leaves the loop, accesses under one
/// branch with no common block between them, two kinds of atomics on one
/// dword in divergent code, an atomic that returns the old value.
#[test]
fn what_has_no_exact_order_is_refused_with_its_reason() {
    let cases = [
        (
            flag_loop(true),
            "a flag read in divergent code decides more than the flag",
        ),
        (
            all_under_one_branch(),
            "LDS accesses in divergent code whose order matters",
        ),
        (
            group_maximum(true),
            "LDS accesses in divergent code whose order matters",
        ),
        (counted(), "an LDS atomic that returns the old value"),
    ];
    for (words, why) in cases {
        let got = placed(&words).expect_err("no order");
        assert!(got.contains(why), "{got}");
        assert!(got.starts_with("LDS one console wave64 shares without barriers"));
        // In halves the program is refused with that reason.
        assert!(refusal(&words, ORDER).contains(why));
    }
}

/// Without the setting the programs are refused as before it existed. With
/// it they translate in waves of 32 with their two barriers; under
/// `narrow_waves` (D3DMetal) they no longer ask for a wave size at all.
/// The unsynced setting is not needed, and where both are on the order
/// wins.
#[test]
fn the_setting_orders_what_was_refused() {
    let unsynced = HalvesLocal {
        reads: false,
        lds: true,
    };
    for words in [
        writers_then_readers(),
        flag_loop(false),
        group_maximum(false),
    ] {
        let before = refusal(&words, Mac::default());
        assert!(
            before.ends_with(
                "LDS one console wave64 shares without barriers, on a host of 32-lane waves is not supported"
            ),
            "{before}"
        );
        let l = lowered(&words, ORDER, false, HalvesLocal::default()).expect("ordered");
        assert_eq!((l.wave_size, l.split_barriers), (Some(32), 2));
        let narrow = lowered(&words, ORDER, true, HalvesLocal::default()).expect("ordered");
        assert_eq!((narrow.wave_size, narrow.split_barriers), (None, 2));
        let raced = lowered(&words, Mac::default(), false, unsynced).expect("unsynced");
        assert_eq!(raced.split_barriers, 0);
        let both = lowered(&words, ORDER, false, unsynced).expect("ordered");
        assert_eq!(both.dxil, l.dxil);
    }
}

/// The setting changes nothing where the barriers of `narrow_waves` already
/// order the LDS, nor in a program it cannot order.
#[test]
fn the_setting_leaves_other_programs_as_they_were() {
    // A write by every lane, then a read by every lane, in the entry block.
    let mut plain = PROLOGUE.to_vec();
    plain.extend([V16_IS_32_TIMES_V0, 0x3404_0082]); // v_lshlrev_b32 v2, 2, v0
    plain.extend(ds(DS_WRITE, 0, 2, 0, 0, 0));
    plain.extend(ds(DS_READ, 4, 2, 0, 0, 3));
    plain.push(S_WAITCNT_LGKM);
    plain.extend(store(3, 0));
    plain.push(S_ENDPGM);
    let unsynced = HalvesLocal {
        reads: false,
        lds: true,
    };
    for (words, narrow, local) in [
        (plain, true, HalvesLocal::default()),
        (all_under_one_branch(), false, unsynced),
        (counted(), true, unsynced),
    ] {
        let dxil = |mac| {
            lowered(&words, mac, narrow, local)
                .expect("translates")
                .dxil
        };
        assert_eq!(dxil(ORDER), dxil(Mac::default()));
    }
}

// ---- a CPU model of the tile classifier's LDS in two halves ----

/// What a lane brings: whether its quad has a valid texel, and the round
/// (0 to 3) in which one of its texels is of the class looked for.
#[derive(Clone, Copy, Default)]
struct Quad {
    valid: bool,
    class_at: Option<u32>,
}

/// What a lane ends with: the "tile has something" dword it read, and for
/// lane 0 the class dword it read when the tile had something.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
struct Out {
    any: u32,
    class: Option<u32>,
}

/// One wave of the model: `lanes` consecutive lanes from `first`, run in
/// lockstep, at step `pc` of the program below, past `passed` barriers.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Wave {
    first: usize,
    lanes: usize,
    pc: usize,
    passed: usize,
    /// Lanes still in the class loop.
    looking: Vec<bool>,
}

/// A group of 64 in the middle of a run.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Group {
    waves: Vec<Wave>,
    lds: [u32; 2],
    out: Vec<Out>,
}

/// The steps: 0 the clear (lane 0), 1 valid lanes set dword 1, 2 to 9 the
/// four rounds of the class loop (even: lanes with the class set dword 0;
/// odd: the other lanes still looking read it and leave if it is set), 10
/// every lane reads dword 1, 11 lane 0 reads dword 0 if the tile has
/// something. Barriers, where the form under test has them, are before
/// steps 1 and 10: the two `lower::lds_order` places.
const STEPS: usize = 12;
const BARRIERS: [usize; 2] = [1, 10];

fn step(g: &mut Group, n: usize, quads: &[Quad; 64]) {
    let w = &mut g.waves[n];
    let lanes = w.first..w.first + w.lanes;
    match w.pc {
        0 => {
            if lanes.contains(&0) {
                g.lds = [0, 0];
            }
        }
        1 => {
            for l in lanes {
                if quads[l].valid {
                    g.lds[1] = 1;
                }
                w.looking[l - w.first] = quads[l].valid;
            }
        }
        2..=9 => {
            let round = (w.pc as u32 - 2) / 2;
            let sets = w.pc % 2 == 0;
            for l in lanes {
                let i = l - w.first;
                let has = quads[l].class_at == Some(round);
                if !w.looking[i] {
                    continue;
                }
                if sets && has {
                    g.lds[0] = 1;
                    w.looking[i] = false;
                } else if !sets && !has && g.lds[0] != 0 {
                    w.looking[i] = false;
                }
            }
        }
        10 => {
            for l in lanes {
                g.out[l].any = g.lds[1];
            }
        }
        _ => {
            if lanes.contains(&0) && g.out[0].any != 0 {
                g.out[0].class = Some(g.lds[0]);
            }
        }
    }
    w.pc += 1;
}

/// Every result a group of 64 can end with when it runs as waves of
/// `width` lanes whose steps come in any order: with `barriers` no wave
/// passes one until every wave is at it. The LDS starts with what the group
/// before left (`stale`), on the console as on the host.
fn results(quads: &[Quad; 64], width: usize, barriers: &[usize], stale: [u32; 2]) -> Vec<Vec<Out>> {
    use std::collections::HashSet;
    let start = Group {
        waves: (0..64 / width)
            .map(|n| Wave {
                first: n * width,
                lanes: width,
                pc: 0,
                passed: 0,
                looking: vec![false; width],
            })
            .collect(),
        lds: stale,
        out: vec![Out::default(); 64],
    };
    let mut seen: HashSet<Group> = HashSet::from([start.clone()]);
    let mut ends: HashSet<Vec<Out>> = HashSet::new();
    let mut todo = vec![start];
    while let Some(mut g) = todo.pop() {
        let waits = |w: &Wave| barriers.get(w.passed) == Some(&w.pc);
        if g.waves.iter().all(waits) {
            for w in &mut g.waves {
                w.passed += 1;
            }
        }
        let ready: Vec<usize> = (0..g.waves.len())
            .filter(|&n| {
                g.waves[n].pc < STEPS && barriers.get(g.waves[n].passed) != Some(&g.waves[n].pc)
            })
            .collect();
        if ready.is_empty() {
            assert!(g.waves.iter().all(|w| w.pc == STEPS), "stuck at a barrier");
            ends.insert(g.out);
            continue;
        }
        for n in ready {
            let mut next = g.clone();
            step(&mut next, n, quads);
            if seen.insert(next.clone()) {
                todo.push(next);
            }
        }
    }
    ends.into_iter().collect()
}

fn tiles() -> Vec<[Quad; 64]> {
    let tile = |valid: &[usize], class: &[(usize, u32)]| {
        let mut q = [Quad::default(); 64];
        for &l in valid {
            q[l].valid = true;
        }
        for &(l, round) in class {
            q[l].class_at = Some(round);
        }
        q
    };
    vec![
        // Nothing in the tile.
        tile(&[], &[]),
        // Something in the upper half only, of the class in round 2.
        tile(&[40], &[(40, 2)]),
        // Something in both halves; the class only in the upper half.
        tile(&[3, 50], &[(50, 0)]),
        // Something in the lower half; the class nowhere.
        tile(&[0, 7], &[]),
        // The class in both halves in different rounds.
        tile(&[1, 33, 60], &[(1, 3), (60, 1)]),
        // The class in a lane that another lane's earlier find stops short.
        tile(&[2, 20, 45], &[(20, 0), (45, 3)]),
    ]
}

/// The model's console: one wave of 64 in lockstep, which has one result.
fn console(quads: &[Quad; 64], stale: [u32; 2]) -> Vec<Out> {
    let all = results(quads, 64, &[], stale);
    assert_eq!(all.len(), 1);
    all.into_iter().next().expect("one")
}

/// The dependency the title's tile classifier has
/// (`notes/cross-half-programs.md`): the clear must come before every set,
/// and every set before the reads all lanes make. Over every order of the
/// two halves' steps: with nothing between the halves some order gives a
/// wrong tile wherever the tile has something or the group before left the
/// dword set; with only one of the two barriers some order is still wrong
/// where one half alone has something; with both every order gives the
/// console's result, the flag read in the loop left unordered.
#[test]
fn two_halves_give_the_consoles_tile_in_every_order_only_with_both_barriers() {
    for (n, quads) in tiles().iter().enumerate() {
        for stale in [[0, 0], [1, 1], [0, 1], [1, 0]] {
            let want = console(quads, stale);
            let what = format!("tile {n}, stale {stale:?}");
            let wrong = |barriers: &[usize]| {
                (results(quads, 32, barriers, stale).iter()).any(|got| *got != want)
            };
            assert!(!wrong(&BARRIERS), "{what}: ordered");
            let something = quads.iter().any(|q| q.valid);
            assert_eq!(wrong(&[]), something || stale[1] != 0, "{what}: unsynced");
            // Something in one half only: the other half reads before it is
            // set, or lane 0's half clears it after it was.
            let halves = [&quads[..32], &quads[32..]].map(|h| h.iter().any(|q| q.valid));
            if halves[0] != halves[1] {
                assert!(wrong(&BARRIERS[..1]), "{what}: only after the clear");
            }
            if halves == [false, true] {
                assert!(wrong(&BARRIERS[1..]), "{what}: only before the read");
            }
        }
    }
}

/// The same with the group as four waves of 16: the proof does not turn on
/// where the host splits the console's wave.
#[test]
fn four_quarter_waves_give_the_consoles_tile_in_every_order() {
    for (n, quads) in tiles().iter().enumerate() {
        let want = console(quads, [1, 1]);
        let all = results(quads, 16, &BARRIERS, [1, 1]);
        assert_eq!(all, [want], "tile {n}");
    }
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// Runs `words` (one group of 64, each lane storing a dword at
    /// `l * 32`) in halves with the LDS order on every hardware adapter. A
    /// line per lane whose dword is not `want`.
    fn differences(what: &str, words: &[u32], want: u32) -> Vec<String> {
        let l = lowered(words, ORDER, false, HalvesLocal::default()).expect("ordered");
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
            // Many groups, many times: an order that is only usually right
            // shows.
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
                let wrong: Vec<String> = (out[0].chunks(8).enumerate())
                    .filter(|(_, got)| got[0] != want)
                    .map(|(l, got)| {
                        format!(
                            "{label}: round {round}, lane {l}: {}, console {want}",
                            got[0]
                        )
                    })
                    .collect();
                failed.extend(wrong);
            }
            eprintln!("{label}: {} lane results differ", failed.len());
        }
        failed
    }

    /// With the two barriers every lane of both halves reads what the
    /// console's one wave reads: the dword lane 40 set (1), the flag (1),
    /// the maximum of the thread ids that take part (63).
    #[test]
    fn both_halves_read_what_the_console_reads_on_every_gpu() {
        let mut failed = differences("writers then readers", &writers_then_readers(), 1);
        failed.extend(differences("flag loop", &flag_loop(false), 1));
        failed.extend(differences("maximum", &group_maximum(false), 63));
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
