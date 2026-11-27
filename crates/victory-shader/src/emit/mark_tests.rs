//! Write marks: what the translated code of a store holds. The bitmaps they
//! set are checked on a GPU in `memory_tests`; a program run nowhere has
//! only the shape of its module to check.

use super::lower::{self, Shape};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0.
const ZERO: u16 = 128;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

fn lower(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    view_marks: bool,
) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
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
        atomic_locks: false,
        halves_local: Default::default(),
        mac: Default::default(),
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        host_wave32: false,
        view_marks,
    })
    .expect("lower")
}

fn param(root: RootType, user_data: u8, sgprs: u32, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    }
}

/// A program that stores `stores` dwords through a pointer in the root
/// constants, each in its own dword: title memory.
fn title_stores(stores: u32) -> Shape {
    let mut w = vec![v_mov_b32(1, ZERO), v_mov_b32(4, 2)];
    for k in 0..stores {
        w.extend(global(28, 0, 1, 4, 0, 4 * k)); // global_store_dword v1, v4, s[0:1]
    }
    w.push(S_ENDPGM);
    let l = lower(&w, vec![param(RootType::Constants, 0, 4, 0)], vec![], false);
    assert!(l.memory_writes || stores == 0);
    l.shape
}

/// A program that stores `stores` dwords through the buffer view in a
/// table: a heap view with write marks.
fn view_stores(stores: u32) -> Shape {
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    w.extend(s_load(2, 8, 4, 0));
    w.push(S_WAITCNT_ALL);
    w.extend((0..4).map(|v| v_mov_b32(v, 0)));
    for k in 0..stores {
        w.extend(mubuf(0x1C, 0, 8, 4 * k as u16)); // buffer_store_dword v0, off, s[8:11]
    }
    w.push(S_ENDPGM);
    let uav = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let layout = vec![
        param(RootType::Constants, 0, 4, 0),
        param(RootType::Table, 4, 1, 1),
    ];
    let l = lower(&w, layout, vec![uav], true);
    assert!(l.view_marks || stores == 0);
    l.shape
}

/// A store to title memory sets its page's bit where a load finds it clear,
/// so none of its atomics runs in every lane: they sit behind branches.
#[test]
fn title_stores_mark_with_an_atomic_only_behind_a_branch() {
    for stores in 1..3 {
        let shape = title_stores(stores);
        // The page of its first byte and the page of its last.
        assert_eq!(shape.atomics.len() as u32, 2 * stores, "{shape:?}");
        assert!(shape.atomics.iter().all(|&b| b != 0), "{shape:?}");
    }
}

/// A store through a buffer view reads the view's base bit and phase in one
/// load and the bitmap dword holding its pages' bits in one more (the
/// store's first and last byte nearly always share it), and takes one branch
/// for both pages' bits, with the atomic for a write across two dwords
/// nested in it.
#[test]
fn view_stores_read_their_marks_once() {
    let (one, two) = (view_stores(1), view_stores(2));
    assert_eq!(two.loads - one.loads, 2, "{one:?} {two:?}");
    assert_eq!(two.branches - one.branches, 2, "{one:?} {two:?}");
    assert_eq!(two.atomics.len() - one.atomics.len(), 2, "{one:?} {two:?}");
    assert!(two.atomics.iter().all(|&b| b != 0), "{two:?}");
}
