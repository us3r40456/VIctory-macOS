//! The Mac line's settings (`lower::mac`: `Options::early_depth`,
//! `atomic_lock_helpers`, `atomic_lock_keys`). These tests look at the
//! translation, and at a CPU model of the lock loop's steps; they run on any
//! host. The GPU test of the lock settings is in `valu_tests` (Windows).

use super::container::SigElement;
use super::lower::{self, Mac};
use super::{Options, container, emit_with};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `DB_SHADER_CONTROL` of the title's alpha pre-pass: early then late Z
/// order (bit 4), kill (6), run without colour targets (10), depth before
/// the shader (12).
const PREPASS: u32 = 0x1450;
/// `SPI_PS_INPUT_ADDR`: perspective centre and the position's four floats,
/// as the pre-pass has them; and with the coverage mask.
const POSITION: u32 = 0xF02;
const COVERAGE: u32 = 1 << 14;

/// `v_mov_b32 v<dst>, <src>` (128 + k: the constant k; 256 + n: VGPR n).
const fn v_mov(dst: u8, src: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | src
}

fn element(name: &str, kind: u8, comp: u8, row: Option<u32>, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row,
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

fn target() -> SigElement {
    element("SV_Target", 16, 9, Some(0), 4)
}

/// A pixel program: four constants to render target 0.
fn colour() -> Vec<u32> {
    let mut w: Vec<u32> = (0..4).map(|k| v_mov(k, 128 + u32::from(k))).collect();
    w.extend([0xF800_180F, 0x0302_0100]); // exp mrt0, v0, v1, v2, v3 done vm
    w.push(S_ENDPGM);
    w
}

/// A pixel program that also exports depth (`exp mrtz v0`), not last.
fn colour_and_depth() -> Vec<u32> {
    let mut w: Vec<u32> = (0..4).map(|k| v_mov(k, 128 + u32::from(k))).collect();
    w.extend([0xF800_0081, 0x0000_0000]); // exp mrtz v0, off, off, off
    w.extend([0xF800_180F, 0x0302_0100]);
    w.push(S_ENDPGM);
    w
}

fn pixel(words: &[u32], db_shader_control: u32, input_addr: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr,
            db_shader_control,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lowered(native: &NativeProgram, outputs: Vec<SigElement>, mac: Mac) -> lower::Lowered {
    lowered_with(native, outputs, mac, false)
}

/// With 64-bit image atomics under locks.
fn lowered_locked(native: &NativeProgram, outputs: Vec<SigElement>, mac: Mac) -> lower::Lowered {
    lowered_with(native, outputs, mac, true)
}

fn lowered_with(
    native: &NativeProgram,
    outputs: Vec<SigElement>,
    mac: Mac,
    atomic_locks: bool,
) -> lower::Lowered {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs,
        prims: vec![],
        mesh_stride: 0,
        atomic_locks,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac,
        view_marks: false,
    })
    .unwrap()
}

const EARLY: Mac = Mac {
    early_depth: true,
    lock_helpers: false,
    lock_keys: false,
    lds_order: false,
    exchange: false,
    uav_level0: false,
    walk: false,
    across: false,
};

/// The rule by itself: bit 12, and nothing that D3D's flag would change.
#[test]
fn the_early_depth_rule_is_bit_12_without_depth_stencil_or_coverage() {
    let rule = lower::early_depth_rule;
    let colours = [target()];
    // The title's four values with bit 12 (`notes/hair.md`).
    for control in [PREPASS, 0x1050, 0x1010, 0x1410] {
        assert!(rule(control, POSITION, &colours), "{control:#x}");
        assert!(
            rule(control, POSITION, &[]),
            "{control:#x}, no colour target"
        );
    }
    // Without bit 12 the title leaves the order to the hardware: no flag.
    for control in [0, 0x10, 0x50, 0x70, 0x200, 0x240, 0x10240] {
        assert!(!rule(control, POSITION, &colours), "{control:#x}");
    }
    // Depth, stencil and coverage exports the register enables.
    for bit in [0, 1, 2, 8] {
        assert!(!rule(PREPASS | 1 << bit, POSITION, &colours), "bit {bit}");
    }
    // The same as outputs of the translated program.
    for (name, kind) in [
        ("SV_Coverage", 14),
        ("SV_Depth", 17),
        ("SV_DepthLessEqual", 18),
        ("SV_DepthGreaterEqual", 19),
        ("SV_StencilRef", 20),
    ] {
        let outputs = [target(), element(name, kind, 9, None, 1)];
        assert!(!rule(PREPASS, POSITION, &outputs), "{name}");
    }
    // A program that reads the coverage mask.
    assert!(!rule(PREPASS, POSITION | COVERAGE, &colours));
}

/// The flag is set where the setting is on and the rule holds, and the
/// translation is byte for byte the same wherever either is not so.
#[test]
fn the_early_depth_flag_follows_the_setting_and_the_rule() {
    let off = Mac::default();
    let prepass = pixel(&colour(), PREPASS, POSITION);
    let with = lowered(&prepass, vec![target()], EARLY);
    let without = lowered(&prepass, vec![target()], off);
    assert!(with.early_depth);
    assert!(!without.early_depth);
    assert_ne!(with.dxil, without.dxil);
    // The setting off is the translation of a program without the bit.
    let plain = pixel(&colour(), 0x50, POSITION);
    assert_eq!(lowered(&plain, vec![target()], off).dxil, without.dxil);

    // The rule does not hold: no flag, the same bytes.
    let depth = [target(), element("SV_Depth", 17, 9, None, 1)];
    let cases: [(&str, NativeProgram, Vec<SigElement>); 4] = [
        ("no bit 12", plain, vec![target()]),
        (
            "a depth export",
            pixel(&colour_and_depth(), PREPASS | 1, POSITION),
            depth.to_vec(),
        ),
        (
            "a depth output the register does not name",
            pixel(&colour_and_depth(), PREPASS, POSITION),
            depth.to_vec(),
        ),
        (
            "a coverage input",
            pixel(&colour(), PREPASS, POSITION | COVERAGE),
            vec![target()],
        ),
    ];
    for (label, native, outputs) in cases {
        let with = lowered(&native, outputs.clone(), EARLY);
        assert!(!with.early_depth, "{label}");
        assert_eq!(with.dxil, lowered(&native, outputs, off).dxil, "{label}");
    }
}

/// Only pixel programs: a compute program is the same with the setting.
#[test]
fn early_depth_leaves_other_stages_alone() {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: [v_mov(0, 128), S_ENDPGM]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let with = lowered(&native, vec![], EARLY);
    assert!(!with.early_depth);
    assert_eq!(with.dxil, lowered(&native, vec![], Mac::default()).dxil);
}

/// `Options::early_depth` reaches the lowering through `emit_with`.
#[test]
fn the_early_depth_option_reaches_the_translation() {
    let native = pixel(&colour(), PREPASS, POSITION);
    let osg1 = container::signature(&[target()], false);
    let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let emit = |options| emit_with(&c, &native, &p, &st, &io, &res, options).unwrap();
    let off = emit(Options::default());
    let on = emit(Options {
        early_depth: true,
        ..Default::default()
    });
    assert_ne!(on.dxil, off.dxil);
    // The flag has no `SFI0` feature bit: the two differ in the module only.
    let part = |dxil: &[u8], cc| {
        Container::parse(dxil)
            .unwrap()
            .part(cc)
            .unwrap()
            .data
            .to_vec()
    };
    for cc in [FourCC::SFI0, FourCC::OSG1, FourCC::PSV0] {
        assert_eq!(part(&on.dxil, cc), part(&off.dxil, cc));
    }
    // dxil.dll, where there is one, checks the flag against the module.
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut on.dxil.clone()).unwrap();
    }
}

// ---- Lock writers (`Mac::lock_helpers`, `Mac::lock_keys`) ----

const WRITERS: Mac = Mac {
    early_depth: false,
    lock_helpers: true,
    lock_keys: true,
    lds_order: false,
    exchange: false,
    uav_level0: false,
    walk: false,
    across: false,
};

/// `image_atomic_<op> v[vdata..], v[vaddr:vaddr+1], s[srsrc:srsrc+7]` on a
/// 2D image (0x17 umax, 0x0F swap; glc returns the old value).
const fn image_atomic(op: u32, vdata: u8, vaddr: u8, srsrc: u8, dmask: u32, glc: bool) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | (glc as u32) << 13 | 1 << 12 | dmask << 8 | 1 << 3,
        vaddr as u32 | (vdata as u32) << 8 | (srsrc as u32 >> 2) << 16,
    ]
}

/// A program with one image atomic of the constants v[1:2] at texel (0, 0)
/// of the unordered view at table dword 0 (s0 is the table's address).
fn atomic_words(op: u32, dmask: u32, glc: bool, pixel: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF40C_0000,
        0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
        v_mov(1, 128 + 5),
        v_mov(2, 128 + 7),
        v_mov(3, 128),
        v_mov(4, 128),
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    ];
    w.extend(image_atomic(op, 1, 3, 0, dmask, glc));
    if pixel {
        w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
        w.extend([0xF800_180F, 0x0201_0201]); // exp mrt0, v1, v2, v1, v2 done vm
    }
    w.push(S_ENDPGM);
    w
}

fn with_uav(mut native: NativeProgram) -> NativeProgram {
    native.layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    native.used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }];
    native
}

fn atomic_pixel(op: u32, dmask: u32, glc: bool) -> NativeProgram {
    with_uav(pixel(
        &atomic_words(op, dmask, glc, true),
        PREPASS,
        POSITION,
    ))
}

fn atomic_compute(op: u32, dmask: u32, glc: bool) -> NativeProgram {
    with_uav(NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: atomic_words(op, dmask, glc, false)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    })
}

/// (helper-lane tests, plain texel loads, 32-bit lock exchanges) written.
fn lock_calls(l: &lower::Lowered) -> [usize; 3] {
    l.lock_calls
}

const UMAX: u32 = 0x17;
const SWAP: u32 = 0x0F;

/// What the settings add to a 64-bit atomic under locks, by stage,
/// operation and whether the old value is read; and that everything else
/// is byte for byte what it was.
#[test]
fn only_lanes_that_can_change_the_texel_reach_the_lock() {
    let off = Mac::default();
    let helpers = Mac {
        lock_helpers: true,
        ..off
    };
    let keys = Mac {
        lock_keys: true,
        ..off
    };
    let out = vec![target()];
    // The lock loop as it was: one load (under the lock), one exchange.
    let px = atomic_pixel(UMAX, 3, false);
    let was = lowered_locked(&px, out.clone(), off);
    assert_eq!(lock_calls(&was), [0, 1, 1]);
    // A pixel's umax whose old value no one reads: both tests, one load more.
    let both = lowered_locked(&px, out.clone(), WRITERS);
    assert_eq!(lock_calls(&both), [1, 2, 1]);
    assert_eq!(
        lock_calls(&lowered_locked(&px, out.clone(), helpers)),
        [1, 1, 1]
    );
    assert_eq!(
        lock_calls(&lowered_locked(&px, out.clone(), keys)),
        [0, 2, 1]
    );
    // Its old value read (glc): no key test; a helper lane's old value is
    // the plain load.
    let glc = atomic_pixel(UMAX, 3, true);
    assert_eq!(
        lock_calls(&lowered_locked(&glc, out.clone(), WRITERS)),
        [1, 2, 1]
    );
    let glc_keys = lowered_locked(&glc, out.clone(), keys);
    assert_eq!(glc_keys.dxil, lowered_locked(&glc, out.clone(), off).dxil);
    // Another operation: the helper test only (an exchange whose old value
    // no one reads loads nothing).
    let swap = atomic_pixel(SWAP, 3, false);
    assert_eq!(
        lock_calls(&lowered_locked(&swap, out.clone(), off)),
        [0, 0, 1]
    );
    assert_eq!(
        lock_calls(&lowered_locked(&swap, out.clone(), WRITERS)),
        [1, 0, 1]
    );
    let swap_keys = lowered_locked(&swap, out.clone(), keys);
    assert_eq!(swap_keys.dxil, lowered_locked(&swap, out.clone(), off).dxil);
    // Compute has no helper lanes: the key test only, and nothing for
    // another operation.
    let cs = atomic_compute(UMAX, 3, false);
    assert_eq!(lock_calls(&lowered_locked(&cs, vec![], WRITERS)), [0, 2, 1]);
    let cs_helpers = lowered_locked(&cs, vec![], helpers);
    assert_eq!(cs_helpers.dxil, lowered_locked(&cs, vec![], off).dxil);
    let cs_swap = atomic_compute(SWAP, 3, false);
    assert_eq!(
        lowered_locked(&cs_swap, vec![], WRITERS).dxil,
        lowered_locked(&cs_swap, vec![], off).dxil
    );
    // Without locks (the host's own 64-bit atomic), and for a 32-bit atomic
    // with or without them, the settings change nothing.
    let narrow = atomic_pixel(UMAX, 1, false);
    for (native, locks) in [(&px, false), (&narrow, false), (&narrow, true)] {
        let a = lowered_with(native, out.clone(), WRITERS, locks);
        let b = lowered_with(native, out.clone(), off, locks);
        assert_eq!(a.dxil, b.dxil, "locks {locks}");
        assert_eq!(lock_calls(&a)[0], 0);
    }
}

// A CPU model of the operations the translation emits for a 64-bit umax
// under locks, to check the argument in `lower::mac` ("Lock writers") over
// interleavings. It is a model: it does not run the emitted DXIL. The
// texel is two 32-bit cells; a store writes them one after the other, so a
// plain load between the two sees one writer's high half with another
// value's low half, the worst a host may do.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    /// The lock loop for every lane.
    Locked,
    /// The key test first: a lane whose high half is below the texel's
    /// stays out ([`Mac::lock_keys`]).
    KeyFirst,
    /// Wrong on purpose: stay out when the whole 64-bit value read without
    /// the lock is not below the lane's.
    WholeFirst,
    /// Wrong on purpose: no lock, each half raised by a 32-bit atomic max.
    HalvesMax,
}

#[derive(Clone)]
struct Lane {
    value: (u32, u32),
    /// A helper lane: the host drops what it writes.
    helper: bool,
    pc: u8,
    old: (u32, u32),
}

struct Texel {
    hi: u32,
    lo: u32,
    lock: bool,
}

/// One step of `lane`; false if it could not move (the lock is taken).
/// The steps: 0 key test, 1 take the lock, 2 load high, 3 load low,
/// 4 store high, 5 store low, 6 free the lock, 7 done.
fn step(t: &mut Texel, l: &mut Lane, form: Form) -> bool {
    let (vhi, vlo) = l.value;
    match (form, l.pc) {
        (Form::HalvesMax, 0) => {
            if !l.helper {
                t.hi = t.hi.max(vhi);
            }
            l.pc = 1;
        }
        (Form::HalvesMax, _) => {
            if !l.helper {
                t.lo = t.lo.max(vlo);
            }
            l.pc = 7;
        }
        (Form::Locked, 0) => l.pc = 1,
        (Form::KeyFirst, 0) => l.pc = if t.hi > vhi { 7 } else { 1 },
        // The two halves of one plain load, read in two steps.
        (Form::WholeFirst, 0) => {
            l.old.0 = t.hi;
            l.pc = 8;
        }
        (Form::WholeFirst, 8) => l.pc = if (l.old.0, t.lo) >= l.value { 7 } else { 1 },
        (_, 1) => {
            // A helper lane's exchange is dropped; it goes on as if it won.
            if !l.helper {
                if t.lock {
                    return false;
                }
                t.lock = true;
            }
            l.pc = 2;
        }
        (_, 2) => {
            l.old.0 = t.hi;
            l.pc = 3;
        }
        (_, 3) => {
            l.old.1 = t.lo;
            l.pc = 4;
        }
        (_, 4) => {
            if !l.helper {
                t.hi = l.old.max(l.value).0;
            }
            l.pc = 5;
        }
        (_, 5) => {
            if !l.helper {
                t.lo = l.old.max(l.value).1;
            }
            l.pc = 6;
        }
        _ => {
            if !l.helper {
                t.lock = false;
            }
            l.pc = 7;
        }
    }
    true
}

/// Runs the lanes to the end, the next lane to move chosen by `pick`
/// (given how many can move); the texel at the end.
fn run(
    start: (u32, u32),
    lanes: &[Lane],
    form: Form,
    mut pick: impl FnMut(usize) -> usize,
) -> (u32, u32) {
    let mut t = Texel {
        hi: start.0,
        lo: start.1,
        lock: false,
    };
    let mut lanes = lanes.to_vec();
    loop {
        let ready: Vec<usize> = (0..lanes.len())
            .filter(|&i| lanes[i].pc != 7 && !(lanes[i].pc == 1 && t.lock && !lanes[i].helper))
            .collect();
        if ready.is_empty() {
            assert!(
                lanes.iter().all(|l| l.pc == 7),
                "lanes wait for a lock no one holds"
            );
            assert!(!t.lock, "a lock left taken");
            return (t.hi, t.lo);
        }
        let i = ready[pick(ready.len())];
        assert!(step(&mut t, &mut lanes[i], form));
    }
}

/// What the console leaves: the maximum of the start and the values of
/// the lanes that are no helpers.
fn want(start: (u32, u32), lanes: &[Lane]) -> (u32, u32) {
    lanes
        .iter()
        .filter(|l| !l.helper)
        .map(|l| l.value)
        .fold(start, |a, b| a.max(b))
}

fn lane(value: (u32, u32), helper: bool) -> Lane {
    Lane {
        value,
        helper,
        pc: 0,
        old: (0, 0),
    }
}

/// Every interleaving of two lanes (by depth-first search over the
/// choices), for `form`: the first start, values and schedule that end
/// with another texel than [`want`].
fn first_wrong_pair(form: Form) -> Option<String> {
    let values = [(0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2)];
    for start in [(0, 0), (1, 2), (2, 1)] {
        for a in values {
            for b in values {
                let lanes = [lane(a, false), lane(b, false)];
                // Depth first over the choices between two ready lanes.
                let mut path: Vec<bool> = Vec::new();
                loop {
                    let mut k = 0;
                    let got = run(start, &lanes, form, |n| {
                        if n < 2 {
                            return 0;
                        }
                        if k == path.len() {
                            path.push(false);
                        }
                        k += 1;
                        usize::from(path[k - 1])
                    });
                    if got != want(start, &lanes) {
                        return Some(format!(
                            "start {start:?}, values {a:?} and {b:?}, choices {path:?}: {got:?}"
                        ));
                    }
                    path.truncate(k);
                    while path.last() == Some(&true) {
                        path.pop();
                    }
                    match path.last_mut() {
                        Some(last) => *last = true,
                        None => break,
                    }
                }
            }
        }
    }
    None
}

#[test]
fn the_key_test_keeps_the_maximum_in_every_interleaving_of_two_lanes() {
    assert_eq!(first_wrong_pair(Form::Locked), None);
    assert_eq!(first_wrong_pair(Form::KeyFirst), None);
    // The model finds what the argument rules out: a whole-value test on
    // a load torn between two stores, and a maximum taken by halves (the
    // halves of two writers stitched together).
    assert!(first_wrong_pair(Form::WholeFirst).is_some());
    assert!(first_wrong_pair(Form::HalvesMax).is_some());
}

/// Many lanes, helper lanes among them, in random order: the lock loop
/// and the two settings end with the same texel, the true maximum, and it
/// is the start or one lane's whole value.
#[test]
fn many_lanes_in_random_order_leave_the_maximum() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut random = move |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 33) as usize % n
    };
    for round in 0..20_000 {
        let n = 2 + random(15);
        // Few high halves, so that equal and near keys are common.
        let highs = 1 + random(4) as u32;
        let start = (random(highs as usize + 1) as u32, random(8) as u32);
        let lanes: Vec<Lane> = (0..n)
            .map(|_| {
                let value = (random(highs as usize + 2) as u32, random(8) as u32);
                lane(value, random(4) == 0)
            })
            .collect();
        let want = want(start, &lanes);
        assert!(want == start || lanes.iter().any(|l| !l.helper && l.value == want));
        // The lock loop with helper lanes in it, the key test with them in
        // it, and with them left out (`Mac::lock_helpers`).
        let live: Vec<Lane> = lanes.iter().filter(|l| !l.helper).cloned().collect();
        for (label, form, lanes) in [
            ("locked", Form::Locked, &lanes),
            ("keys", Form::KeyFirst, &lanes),
            ("keys, no helpers", Form::KeyFirst, &live),
            ("no helpers", Form::Locked, &live),
        ] {
            let got = run(start, lanes, form, &mut random);
            assert_eq!(
                got,
                want,
                "round {round}, {label}: {lanes_n} lanes",
                lanes_n = lanes.len()
            );
        }
    }
}
