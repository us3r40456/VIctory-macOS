//! Title memory at addresses the program computes (decision 0005, the
//! unified-memory item), run on the host GPU and WARP with the test host
//! serving the contract (`emit` "Title memory"): a synthetic GFX10 program
//! takes a 64-bit pointer from root constants, reads through it with a
//! global and a scalar load, follows a pointer it loaded with another
//! scalar load, stores through a pointer it computes (64-bit multiply-add)
//! and does float min/max atomics, reads through the address it takes
//! from a buffer view's descriptor and an image descriptor's size fields
//! (the host's view addresses), reads
//! pointers into memory no copy holds (zeros, marked missed) and stores and
//! loads through a root UAV (title memory in such a program).

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_FLOAT,
};

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, VIEW_STRIDES, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0, 64.
const ZERO: u16 = 128;
const SIXTY_FOUR: u16 = 192;
/// The title address of the memory the program reads: above 4 GiB.
const TITLE: u64 = 0x12_3450_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// A full typed store to element `s0` through the typed UAV at table slot 0,
/// with write marks and `soffset` in the instruction's scalar-source field.
fn typed_store_program(soffset: u8) -> Result<HostShader, crate::ShaderError> {
    let mut words = vec![
        s_mov_b32(4, 1), // s[4:5]: the table at s1
        s_mov_b32(5, ZERO as u8),
    ];
    words.extend(s_load(2, 8, 4, 0));
    words.push(v_mov_b32(0, 0)); // v0 = s0, the element index
    words.push(S_WAITCNT_ALL);
    for (v, bits) in [
        (1, 0x3F80_0000),
        (2, 0x4000_0000),
        (3, 0x4040_0000),
        (4, 0x4080_0000),
    ] {
        words.extend([0x7E00_0000 | (v as u32) << 17 | 0x01 << 9 | 0xFF, bits]);
    }
    words.extend([
        0xE01C_2000,
        0x8002_0100 | (soffset as u32) << 24, // buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen
        S_ENDPGM,
    ]);
    let param = |root, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 0, 0),
            param(RootType::Table, 1, 1),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 1,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
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
        atomic_lock_rounds: false,
            atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        host_wave32: false,
        view_marks: true,
    })?;
    assert!(lowered.view_marks, "the typed store marks its view");
    assert_eq!(
        lowered.root_srvs,
        [VIEW_STRIDES],
        "the view's run-time stride"
    );
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_srv_bytes: lowered.root_srv_bytes,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    Ok(shader)
}

fn program() -> HostShader {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(v_mov_b32(1, ZERO));
    w.extend(global(12, 2, 1, 0, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(s_load(1, 12, 0, 8)); // s[12:13] = the pointer at +8
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 14, 12, 0)); // s14 = what it points at
    w.push(v_mov_b32(3, SIXTY_FOUR));
    w.push(v_mov_b32(4, 2)); // v4 = s2
    // v_mad_u64_u32 v[8:9], s[20:21], v3, 1, s[0:1]: the pointer plus 0x40,
    // stored through as a VGPR pair.
    w.extend([0xD576_1408, 0x0001_0303]);
    w.extend(global(28, 0, 8, 4, 0x7D, 0)); // global_store_dword v[8:9], v4, off
    w.push(v_mov_b32(6, 3)); // v6 = s3 (a float)
    w.extend(global(63, 0, 3, 6, 0, 0x40)); // global_atomic_fmin at +0x80
    w.extend(global(64, 0, 3, 6, 0, 0x44)); // global_atomic_fmax at +0x84
    // The address of the view at heap index 1 from its descriptor (as the
    // tracers take their BVH's), read through.
    w.extend(s_load(1, 16, 4, 0x20)); // s[16:17] = descriptor 1, dwords 0-1
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 18, 16, 0x100)); // s18 = the dword at +0x100
    w.push(S_WAITCNT_ALL);
    // Pointers into a page no copy holds and into a 256 MiB region with no
    // block, read through: zeros, and both marked missed.
    w.extend(s_load(1, 22, 0, 0x10)); // s[22:23]
    w.extend(s_load(1, 24, 0, 0x18)); // s[24:25]
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 26, 22, 0));
    w.extend(s_load(0, 27, 24, 0));
    w.push(S_WAITCNT_ALL);
    w.push(v_mov_b32(5, 14));
    w.push(v_mov_b32(7, 18));
    w.extend(buffer_store_dword(2, 8, 0));
    w.extend(buffer_store_dword(5, 8, 4));
    w.extend(buffer_store_dword(7, 8, 8));
    w.push(v_mov_b32(5, 26));
    w.push(v_mov_b32(7, 27));
    w.extend(buffer_store_dword(5, 8, 12));
    w.extend(buffer_store_dword(7, 8, 16));
    // A root UAV (s[6:7]), title memory in such a program: a store and a
    // load at its address.
    w.extend(global(28, 0, 1, 4, 6, 0x20)); // global_store_dword v1, v4, s[6:7]
    w.extend(global(12, 9, 1, 0, 6, 0x24)); // global_load_dword v9, v1, s[6:7]
    w.push(S_WAITCNT_ALL);
    w.extend(buffer_store_dword(9, 8, 20));
    // The image descriptor at heap index 2 loaded whole and read only for
    // its size fields (dwords 1-3).
    w.extend(s_load(3, 28, 4, 0x40)); // s[28:35]
    w.push(S_WAITCNT_ALL);
    for (k, at) in [(29, 24), (30, 28), (31, 32)] {
        w.push(v_mov_b32(5, k));
        w.extend(buffer_store_dword(5, 8, at));
    }
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 10,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let memory = res.scalar.iter().filter(|s| s.1.param == resource::MEMORY).count();
    assert_eq!(memory, 11, "every access through the pointer is title memory");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes && l.view_addresses);
    assert!(l.root_uavs.is_empty(), "the root UAV is read as title memory");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// The page table for one copy of title memory at [`TITLE`] whose views
/// are at heap index `view` (null pair at `null`).
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

#[test]
fn title_memory_is_read_and_written_at_computed_addresses() {
    let shader = program();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut memory = vec![0u32; 256];
    memory[0] = 0x1111_1111;
    let pointer = TITLE + 0x100;
    memory[2] = pointer as u32;
    memory[3] = (pointer >> 32) as u32;
    // The next page (no copy) and another 256 MiB region (no block).
    let (next, far) = (TITLE + 0x1_0010, 0x55_0000_0040u64);
    memory[4..8].copy_from_slice(&[next as u32, (next >> 32) as u32, far as u32, (far >> 32) as u32]);
    memory[0x40] = 0x2222_2222;
    // The root UAV's address, and what it loads there.
    let root_uav = TITLE + 0x200;
    memory[(0x200 + 0x24) / 4] = 0x3333_3333;
    memory[0x80 / 4] = 1.0f32.to_bits();
    memory[0x84 / 4] = (-1.0f32).to_bits();
    let value = 0xCAFE_F00D;
    // Heap index 1 (the memory's view) has the title address, index 2 a
    // texture's size fields.
    let mut view_addresses = vec![0u32; 4 * 5];
    view_addresses[4..6].copy_from_slice(&[TITLE as u32, (TITLE >> 32) as u32]);
    let image = [0xC000_0000, 0x0957_C0F9, 0x9004_2FAC];
    view_addresses[9..12].copy_from_slice(&image);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // fmin with 1.0 and fmax with -1.0 in memory, against a negative
        // and a positive operand.
        for operand in [-2.0f32, 0.5] {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                let constants = [TITLE as u32, (TITLE >> 32) as u32, value, operand.to_bits()];
                ud[u.index as usize] = match u.root {
                    RootType::Constants => constants[u.dword as usize],
                    RootType::Descriptor => (root_uav >> (32 * u.dword)) as u32,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: page_table(1, 3) }),
                            (2, Root::Uav(2 * 4096 / 32)),
                            (3, Root::View { srv: true, data: view_addresses.clone() }),
                        ],
                        views: &[
                            View::RawUav(9),
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                out[0],
                [0x1111_1111, 0x2222_2222, 0x2222_2222, 0, 0, 0x3333_3333, image[0], image[1], image[2]],
                "{label}: reads"
            );
            assert_eq!(out[1][0x40 / 4], value, "{label}: the store");
            assert_eq!(out[1][(0x200 + 0x20) / 4], value, "{label}: the root UAV store");
            let floats = [f32::from_bits(out[1][0x20]), f32::from_bits(out[1][0x21])];
            let want = [operand.min(1.0), operand.max(-1.0)];
            assert_eq!(floats, want, "{label}: float min and max");
            // The far region by its level-1 index, the page written, the
            // page missed after it; nothing else.
            let entry = 4096 + (TITLE as u32 >> 16 & 0xFFF) as usize;
            let marked: Vec<usize> = (0..out[2].len() * 32)
                .filter(|&b| out[2][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, [(far >> 28) as usize, entry, entry + 1], "{label}: pages");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A root SRV in a program that does not use title memory is read through
/// the host's view of its host copy (`emit`, "Root SRVs and root UAVs"): the
/// view's index and the root address's offset in it come as user data, and
/// the shader adds the offset. Lanes read 64 bytes apart from the root
/// address 16 bytes into a 1 KiB copy: those inside it read the copy, those
/// past its end read 0 (a host root descriptor read whatever lay there).
#[test]
fn root_views_read_their_copy_and_zeros_past_it() {
    const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
    }
    /// MUBUF `op` (offen) of `vdata` at `vaddr` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [0x38 << 26 | op << 18 | 1 << 12, 0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32]
    }
    const V_LSHLREV_B32: u32 = 0x1A;
    let mut w = vec![
        s_mov_b32(5, ZERO as u8), // s[4:5]: the table at s4
        s_mov_b32(0, 6),          // s[0:3]: the root SRV's buffer descriptor
        0x8801_FF07,
        0x0001_0000, // s_or_b32 s1, s7, stride 1
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, (the console's buffer word 3)
    ];
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(vop2(V_LSHLREV_B32, 1, 128 + 6, 0)); // v1 = lane << 6
    let load = w.len();
    w.extend(mubuf(0x0C, 2, 1, 0)); // buffer_load_dword v2, v1, s[0:3] offen
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane << 2
    w.push(S_WAITCNT_ALL);
    w.extend(mubuf(0x1C, 2, 3, 8)); // buffer_store_dword v2, v3, s[8:11] offen
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lower_shader = |native: &NativeProgram| {
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
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            host_wave32: false,
            narrow_waves: false,
            halves_local: Default::default(),
            mac: Default::default(),
            view_marks: false,
        })
        .expect("lower")
    };
    let mut fixed = native.clone();
    let [word, registers] = mubuf(0x0C, 2, 0, 0);
    fixed.code[4 * load..4 * load + 4].copy_from_slice(&((word & !(1 << 12)) | 48).to_le_bytes());
    fixed.code[4 * load + 4..4 * load + 8].copy_from_slice(&registers.to_le_bytes());
    assert_eq!(lower_shader(&fixed).root_srv_bytes, [(2, 52)]);
    let l = lower_shader(&native);
    assert!(l.root_srv_bytes.is_empty(), "a shader with a dynamic load has no bound");
    assert!(l.root_srvs.is_empty() && !l.memory, "no root descriptor, no title memory");
    let marks: Vec<(u32, u32)> = l
        .user_data
        .iter()
        .filter(|u| u.root == RootType::Descriptor)
        .map(|u| (u.index, u.dword))
        .collect();
    assert_eq!(marks, [(6, super::ROOT_SRV_VIEW), (7, super::ROOT_VIEW_OFFSET)]);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let copy: Vec<u32> = (0..256).map(|k| 0x1000 + k).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        // The table at heap index 0, the copy's view at 1, the root address
        // 16 bytes into it.
        (ud[4], ud[6], ud[7]) = (0, 1, 16);
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(32),
                        View::Buffer {
                            data: &copy,
                            stride: super::host::Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let want: Vec<u32> = (0..32).map(|lane| if lane < 16 { 0x1000 + 4 + 16 * lane } else { 0 }).collect();
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Writes through unordered buffer views mark their pages (`emit`, "Write
/// marks"): a dword store and a 16-byte store through view 0 (base bit 100,
/// its first byte 16 bytes before a 64 KiB page ends, so the second store
/// crosses into the next page), a store through view 1 (not marked) and an
/// atomic add through view 2 (base bit 200). The stores land; exactly bits
/// 100, 101 and 200 are set (31, 32 and 200 with view 0 at base bit 31,
/// whose two bits are in two dwords of the bitmap).
#[test]
fn writes_through_views_mark_their_pages() {
    /// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }
    const BUFFER_STORE_DWORDX4: u32 = 0x1E;
    const BUFFER_ATOMIC_ADD: u32 = 0x32;
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    for (k, sdata) in [8u8, 12, 16].into_iter().enumerate() {
        w.extend(s_load(2, sdata, 4, 32 * k as u32));
    }
    w.push(S_WAITCNT_ALL);
    w.extend((0..4).map(|v| v_mov_b32(v, 0)));
    w.extend(buffer_store_dword(0, 8, 0));
    w.extend(mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12));
    w.extend(buffer_store_dword(0, 12, 4));
    w.extend(mubuf(BUFFER_ATOMIC_ADD, 0, 16, 8));
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
        used: (0..3).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        view_marks: true,
    })
    .expect("lower");
    assert!(l.view_marks && !l.memory, "marks without the page table");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let value = 0x1234_5678;
    let marks = vec![100, 0xFFF0, super::VIEW_MARK_NONE, 0, 200, 0];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // View 0's second store crosses into the next page: from bit 100 both
        // bits are in one bitmap dword, from bit 31 in two.
        for (first, pages) in [(100, [100, 101, 200]), (31, [31, 32, 200])] {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => value,
                    _ => 0,
                };
            }
            let mut marks = marks.clone();
            marks[0] = first;
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: marks,
                                },
                            ),
                            (2, Root::Uav(8)),
                        ],
                        views: &[View::RawUav(8), View::RawUav(8), View::RawUav(8)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let view_0 = [value, 0, 0, value, value, value, value, 0];
            assert_eq!(out[0], view_0, "{label}: view 0");
            assert_eq!(out[1][1], value, "{label}: view 1");
            assert_eq!(out[2][2], value, "{label}: the atomic");
            let marked: Vec<usize> = (0..out[3].len() * 32)
                .filter(|&b| out[3][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, pages, "{label}: pages from bit {first}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A typed store's write mark spans the view's real element width. Each store
/// below ends exactly at a 64 KiB page boundary: 4-, 8-, 12- and 16-byte
/// elements must all set only the first page's bit, where a hardcoded 16-byte
/// width would also mark the next page. Inline zero is the no-offset soffset.
#[test]
fn typed_store_marks_use_the_element_stride_at_page_boundaries() {
    let shader = typed_store_program(128).expect("lower");
    let cases: [(&str, DXGI_FORMAT, u32); 4] = [
        ("R32F", DXGI_FORMAT_R32_FLOAT, 4),
        ("R32G32F", DXGI_FORMAT_R32G32_FLOAT, 8),
        ("RGB32F", DXGI_FORMAT_R32G32B32_FLOAT, 12),
        ("RGBA32F", DXGI_FORMAT_R32G32B32A32_FLOAT, 16),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        for &(name, format, stride) in &cases {
            // WARP on some hosts loses the device on RGB32 typed views, so
            // it runs only the formats it says it can load and store.
            if warp && !host.typed_buffer_access(format, warp) {
                eprintln!("{label} {name}: no typed UAV load and store; skipped");
                continue;
            }
            let elements = 65_536 / stride;
            let index = elements - 1;
            // Put the element's last byte at 65,535. 65 KiB is not a
            // multiple of 12, so RGB32 needs a four-byte mark phase.
            let phase = 65_536 - elements * stride;
            let base = 100;
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => index,
                    RootType::Table => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let views = [View::TypedUav(format, elements, stride)];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: vec![stride],
                                },
                            ),
                            (
                                2,
                                Root::View {
                                    srv: true,
                                    data: vec![base, phase],
                                },
                            ),
                            (3, Root::Uav(8)),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} {name}: {e}"));
            let marked: Vec<usize> = (0..out[1].len() * 32)
                .filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, [base as usize], "{label} {name}: pages");
            // D3DMetal substitutes R32G32 for RGB32 typed buffers and drops
            // the store. WARP still checks those bytes; both check the mark.
            if !warp && super::host::under_wine() && stride == 12 {
                eprintln!(
                    "{label}: D3DMetal's RGB32 typed-buffer substitution; store bytes not compared"
                );
                continue;
            }
            let channels = stride / 4;
            let values = [
                1.0f32.to_bits(),
                2.0f32.to_bits(),
                3.0f32.to_bits(),
                4.0f32.to_bits(),
            ];
            let mut want = vec![0; (elements * stride).div_ceil(4) as usize];
            let at = (index * stride / 4) as usize;
            want[at..at + channels as usize].copy_from_slice(&values[..channels as usize]);
            assert_eq!(out[0], want, "{label} {name}: the typed store");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Reads of texture views the program takes from the resource heap itself
/// mark their heap index (`emit`, "Read marks"): an `image_load` of the view
/// at heap byte offset 32 (index 1) sets exactly the bit one past
/// [`super::VIEW_READS`]. The same read through a table marks nothing (a
/// table read makes its slot on the host).
#[test]
fn heap_direct_texture_reads_mark_their_heap_index() {
    let lower_with = |root: RootType, used: Vec<UsedResource>, offset: u32| {
        let mut w = vec![s_mov_b32(5, ZERO as u8)];
        w.extend(s_load(3, 12, 4, offset));
        w.push(S_WAITCNT_ALL);
        w.extend((0..2).map(|v| v_mov_b32(v, ZERO)));
        // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        w.extend([0xF000_1F08, 0x0003_0200]);
        w.push(S_WAITCNT_ALL);
        w.push(S_ENDPGM);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
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
            host_wave32: false,
            view_marks: true,
            atomic_locks: false,
            atomic_lock_rounds: false,
        atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: false,
            halves_local: Default::default(),
            mac: Default::default(),
        })
        .expect("lower")
    };
    let table = lower_with(
        RootType::Table,
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 0,
        }],
        0,
    );
    assert!(!table.view_marks, "a table read marks nothing");
    let l = lower_with(RootType::ResourceHeap, Vec::new(), 32);
    assert!(l.view_marks && l.heap, "a heap-direct read is marked");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let words = (super::VIEW_READS + 64) / 32;
    let texel = [[0.5f32; 4]];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // The heap's base is 0 in the user data (`emit`, "Binding").
        let ud = vec![0u32; shader.user_data_dwords() as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: vec![super::VIEW_MARK_NONE; 4] }),
                        (2, Root::Uav(words)),
                    ],
                    views: &[View::Texture(1, 1, &texel), View::Texture(1, 1, &texel)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let bits = &out[0];
        let marked: Vec<usize> = (0..bits.len() * 32).filter(|&b| bits[b / 32] >> (b % 32) & 1 != 0).collect();
        assert_eq!(marked, [super::VIEW_READS as usize + 1], "{label}: the slot read");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Multi-byte writes into title memory that cross a 256 MiB region boundary
/// or a 4 GiB boundary look up their last byte's region and page in the
/// page table rather than offsetting the first page's entry or adding the
/// low dword alone (`Lower::memory`).
#[test]
fn straddling_writes_across_256mib_and_4gib_boundaries_mark_exact_pages() {
    let mut w = vec![v_mov_b32(0, ZERO)];
    for k in 0..4u8 {
        // Inline constants 129..=132 are 1..=4.
        w.push(v_mov_b32(1 + k, 129 + u16::from(k)));
    }
    w.extend(global(30, 0, 0, 1, 0, 0)); // global_store_dwordx4 v0, v[1:4], s[0:1]
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        host_wave32: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }

    // 513 blocks (block 0 null + 512 capacity blocks), 4096 entries each,
    // View::Memory at heap slot 0 (UAV at 1) and View::NullPair at slot 2.
    const BLOCKS: usize = 513;
    const ENTRIES: usize = 4096;
    let mut table = vec![0u32; ENTRIES];
    for _ in 0..BLOCKS * ENTRIES {
        table.extend([2, 0]);
    }
    let set_entry = |table: &mut Vec<u32>, block: usize, page: usize, start_lo: u32| {
        let at = ENTRIES + 2 * (block * ENTRIES + page);
        table[at..at + 2].copy_from_slice(&[0, start_lo]);
    };
    // 4 GiB boundary: 0x1_FFFF_0000 (l1 0x1F -> block 2) -> 0x2_0000_0000 (l1 0x20 -> block 1).
    table[0x1F] = 2;
    table[0x20] = 1;
    set_entry(&mut table, 2, 0xFFF, 0xFFFF_0000);
    set_entry(&mut table, 1, 0x000, 0xFFFF_0000);
    // 256 MiB / capacity edge: 0x1_0FFF_0000 (l1 0x10 -> block 512) -> 0x1_1000_0000 (l1 0x11 -> block 0, unmapped).
    table[0x10] = 512;
    set_entry(&mut table, 512, 0xFFF, 0x0FFF_0000);

    // 128 KiB copy spanning two 64 KiB pages; the store writes at +0xFFF8..+0x10008.
    let memory = vec![0u32; 32 * 1024];
    let bitmap_dwords = BLOCKS * ENTRIES / 32;
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let cases: [(&str, u64, Vec<usize>); 3] = [
            (
                "4 GiB boundary",
                0x1_FFFF_FFF8,
                vec![4096, 2 * ENTRIES + 0xFFF],
            ),
            (
                "256 MiB boundary into unmapped region at block capacity",
                0x1_0FFF_FFF8,
                vec![0x11, 512 * ENTRIES + 0xFFF],
            ),
            (
                "same-page write inside 0x1_FFFF_0000",
                0x1_FFFF_FF00,
                vec![2 * ENTRIES + 0xFFF],
            ),
        ];
        for (case, addr, want_bits) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants {
                    ud[u.index as usize] = (addr >> (32 * u.dword)) as u32;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: table.clone(),
                                },
                            ),
                            (2, Root::Uav(bitmap_dwords as u32)),
                        ],
                        views: &[
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} ({case}): {e}"));
            let off = ((addr as u32) & 0xFFFF) as usize / 4;
            assert_eq!(
                &out[0][off..off + 4],
                &[1, 2, 3, 4],
                "{label} ({case}): store data"
            );
            let marked: Vec<usize> = (0..out[1].len() * 32)
                .filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, want_bits, "{label} ({case}): dirty bits");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Hardware proof that `EARLY_DEPTH_STENCIL` lets D3D12 / D3DMetal skip
/// occluded pixels on translated opaque pixel shaders that only bind `u1`
/// (`VIEW_WRITES`) for `mark_view_read` texture tracking.
#[test]
fn early_depth_stencil_skips_occluded_read_mark_pixel_shaders() {
    use super::host::Host;
    use crate::native::{HwStage, NativeProgram, RootParam, RootType, StageIo, Usage, VsInput};
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    let vs_code_words: [u32; 3] = [
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        S_ENDPGM,
    ];
    let vs_native = NativeProgram {
        stage: HwStage::Vs,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: vec![VsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0xf,
                vgpr: 4,
            }],
            outputs: Vec::new(),
        },
        code: vs_code_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let vp = ir::build(&vs_native.code, false, vs_native.vgprs).expect("vs ir");
    let vst = structure::structure(&vp).expect("vs structure");
    let vres = resource::track(&vp, &vs_native.layout, vs_native.stage, Some(&vst)).expect("vs res");
    let vio = io::raise(&vp, &vs_native).expect("vs io");
    let vs_lowered = lower::lower(&lower::Parts {
        native: &vs_native,
        p: &vp,
        st: &vst,
        io: &vio,
        res: &vres,
        inputs: vec![super::container::SigElement {
            name: "POSITION".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xf,
            half: false,
        }],
        outputs: vec![super::container::SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xf,
            half: false,
        }],
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
        view_marks: false,
    })
    .expect("vs lower");
    let vs_shader = HostShader {
        stage: HwStage::Vs,
        dxil: vs_lowered.dxil,
        user_data: vs_lowered.user_data,
        samplers: vs_lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: vs_lowered.root_cbvs,
        root_srvs: vs_lowered.root_srvs,
        root_uavs: vs_lowered.root_uavs,
        root_srv_bytes: vs_lowered.root_srv_bytes,
        memory: vs_lowered.memory,
        memory_writes: vs_lowered.memory_writes,
        view_addresses: vs_lowered.view_addresses,
        heap: vs_lowered.heap,
        heap_writes: vs_lowered.heap_writes,
        view_marks: vs_lowered.view_marks,
    };

    let ps_words: [u32; 12] = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302, // s_load_dwordx8 s[12:19], s[4:5], 32 (heap slot 1)
        0xFA00_0020,
        S_WAITCNT_ALL,
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_1F08, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0x0003_0200,
        S_WAITCNT_ALL,
        0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
        0x0504_0302,
        S_ENDPGM,
    ];
    let ps_native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let pp = ir::build(&ps_native.code, false, ps_native.vgprs).expect("ps ir");
    let pst = structure::structure(&pp).expect("ps structure");
    let pres = resource::track(&pp, &ps_native.layout, ps_native.stage, Some(&pst)).expect("ps res");
    let pio = io::raise(&pp, &ps_native).expect("ps io");
    let ps_lowered = lower::lower(&lower::Parts {
        native: &ps_native,
        p: &pp,
        st: &pst,
        io: &pio,
        res: &pres,
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
        view_marks: true,
    })
    .expect("ps lower");
    assert!(ps_lowered.early_depth_stencil);
    let ps_shader = HostShader {
        stage: HwStage::Ps,
        dxil: ps_lowered.dxil,
        user_data: ps_lowered.user_data,
        samplers: ps_lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: ps_lowered.root_cbvs,
        root_srvs: ps_lowered.root_srvs,
        root_uavs: ps_lowered.root_uavs,
        root_srv_bytes: ps_lowered.root_srv_bytes,
        memory: ps_lowered.memory,
        memory_writes: ps_lowered.memory_writes,
        view_addresses: ps_lowered.view_addresses,
        heap: ps_lowered.heap,
        heap_writes: ps_lowered.heap_writes,
        view_marks: ps_lowered.view_marks,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut vs_shader.dxil.clone()).expect("VS DXIL validation");
        v.validate(&mut ps_shader.dxil.clone()).expect("PS DXIL validation");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let words = (super::VIEW_READS + 64) / 32;
    let texel = [[0.75f32, 0.25, 0.5, 1.0]];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let root = host
            .root_signature_for(&[
                (&vs_shader, D3D12_SHADER_VISIBILITY_VERTEX),
                (&ps_shader, D3D12_SHADER_VISIBILITY_PIXEL),
            ])
            .expect("root signature");
        let dev = &host.device;
        let layout = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed pointer for PSO creation.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_shader.dxil.as_ptr().cast(),
                BytecodeLength: vs_shader.dxil.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps_shader.dxil.as_ptr().cast(),
                BytecodeLength: ps_shader.dxil.len(),
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: layout.as_ptr(),
                NumElements: layout.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        pso_desc.DepthStencilState.DepthEnable = true.into();
        pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        // SAFETY: valid PSO description.
        let pso: ID3D12PipelineState = unsafe { dev.CreateGraphicsPipelineState(&pso_desc) }
            .unwrap_or_else(|e| panic!("{label}: PSO: {e}"));

        let run_pass = |z: f32| -> Vec<u32> {
            // SAFETY: standalone D3D12 resources for one pass.
            unsafe {
                let rtv_heap: ID3D12DescriptorHeap = dev
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let dsv_heap: ID3D12DescriptorHeap = dev
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let srv_heap: ID3D12DescriptorHeap = dev
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 4,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        ..Default::default()
                    })
                    .unwrap();
                let rt_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 16,
                    Height: 16,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let dsv_desc = D3D12_RESOURCE_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    ..rt_desc
                };
                let default_heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                };
                let mut rt: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
                let rt = rt.unwrap();
                let mut dsv: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &dsv_desc,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    None,
                    &mut dsv,
                )
                .unwrap();
                let dsv = dsv.unwrap();
                dev.CreateRenderTargetView(&rt, None, rtv_heap.GetCPUDescriptorHandleForHeapStart());
                dev.CreateDepthStencilView(&dsv, None, dsv_heap.GetCPUDescriptorHandleForHeapStart());

                let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                let tex_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 1,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let mut tex: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut tex,
                )
                .unwrap();
                let tex = tex.unwrap();

                let buf = |heap_type, bytes: u64, flags, state| -> ID3D12Resource {
                    let mut r: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: bytes,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            Flags: flags,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut r,
                    )
                    .unwrap();
                    r.unwrap()
                };

                let tex_up = buf(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                let mut p = std::ptr::null_mut();
                tex_up.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(texel.as_ptr().cast::<u8>(), p.cast(), 16);
                tex_up.Unmap(0, None);

                let verts = [
                    -1.0f32, -1.0, z, 1.0,
                    -1.0, 3.0, z, 1.0,
                    3.0, -1.0, z, 1.0,
                ];
                let vb = buf(D3D12_HEAP_TYPE_UPLOAD, 48, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                vb.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(verts.as_ptr().cast::<u8>(), p.cast(), 48);
                vb.Unmap(0, None);

                let ud_buf = buf(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                ud_buf.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::write_bytes(p, 0, 256);
                ud_buf.Unmap(0, None);

                let marks_buf = buf(D3D12_HEAP_TYPE_UPLOAD, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                marks_buf.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping([super::VIEW_MARK_NONE; 4].as_ptr().cast::<u8>(), p.cast(), 16);
                marks_buf.Unmap(0, None);

                let bitmap_bytes = u64::from(words) * 4;
                let bitmap = buf(D3D12_HEAP_TYPE_DEFAULT, bitmap_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                let readback = buf(D3D12_HEAP_TYPE_READBACK, bitmap_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

                // Put texture SRV at heap slot 1 (slot 0 left null).
                let slot1 = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: srv_heap.GetCPUDescriptorHandleForHeapStart().ptr + inc as usize,
                };
                dev.CreateShaderResourceView(&tex, None, slot1);

                let queue: ID3D12CommandQueue = dev
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                let alloc: ID3D12CommandAllocator = dev
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = dev
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso))
                    .unwrap();

                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::ManuallyDrop::new(Some(tex_up.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                                Width: 1,
                                Height: 1,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                let mut tex_bar = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                            StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        }),
                    },
                };
                list.ResourceBarrier(&[tex_bar.clone()]);
                std::mem::ManuallyDrop::drop(&mut tex_bar.Anonymous.Transition);

                let rtv_h = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                let dsv_h = dsv_heap.GetCPUDescriptorHandleForHeapStart();
                list.ClearRenderTargetView(rtv_h, &[0.0, 0.0, 0.0, 1.0], None);
                list.ClearDepthStencilView(dsv_h, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
                list.OMSetRenderTargets(1, Some(&rtv_h), false, Some(&dsv_h));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 16.0,
                    Height: 16.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    right: 16,
                    bottom: 16,
                    ..Default::default()
                }]);
                list.SetGraphicsRootSignature(&root);
                list.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
                // PS root indices: 0 = VS b0, 1 = PS b0, 2 = PS t2 (marks), 3 = PS u1 (bitmap).
                list.SetGraphicsRootConstantBufferView(0, ud_buf.GetGPUVirtualAddress());
                list.SetGraphicsRootConstantBufferView(1, ud_buf.GetGPUVirtualAddress());
                list.SetGraphicsRootShaderResourceView(2, marks_buf.GetGPUVirtualAddress());
                list.SetGraphicsRootUnorderedAccessView(3, bitmap.GetGPUVirtualAddress());
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: vb.GetGPUVirtualAddress(),
                        SizeInBytes: 48,
                        StrideInBytes: 16,
                    }]),
                );
                list.DrawInstanced(3, 1, 0, 0);

                let mut bm_bar = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(bitmap.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                        }),
                    },
                };
                list.ResourceBarrier(&[bm_bar.clone()]);
                std::mem::ManuallyDrop::drop(&mut bm_bar.Anonymous.Transition);
                list.CopyResource(&readback, &bitmap);
                list.Close().unwrap();

                let cmd: ID3D12CommandList = windows::core::Interface::cast(&list).unwrap();
                queue.ExecuteCommandLists(&[Some(cmd)]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                while fence.GetCompletedValue() < 1 {
                    assert!(dev.GetDeviceRemovedReason().is_ok());
                    std::thread::yield_now();
                }

                let mut out = vec![0u32; words as usize];
                readback.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(p.cast::<u32>(), out.as_mut_ptr(), words as usize);
                readback.Unmap(0, None);
                out
            }
        };

        let marked = |bits: &[u32]| -> Vec<usize> {
            (0..bits.len() * 32)
                .filter(|&b| bits[b / 32] >> (b % 32) & 1 != 0)
                .collect()
        };

        // Pass 1: z = 0.9 > cleared depth 0.5 -> occluded, Early-Z skips PS & u1 write.
        let occluded = run_pass(0.9);
        if warp {
            // WARP, the software device, does not always skip the shader of
            // an occluded pixel, and CI has only WARP: the result is
            // reported there, and asserted on a hardware device.
            eprintln!("{label}: occluded draw marked {:?}", marked(&occluded));
        } else {
            assert_eq!(
                marked(&occluded),
                Vec::<usize>::new(),
                "{label}: occluded draw must not mark texture reads"
            );
        }

        // Pass 2: z = 0.2 < cleared depth 0.5 -> visible, PS marks texture slot 1.
        let visible = run_pass(0.2);
        assert_eq!(
            marked(&visible),
            vec![super::VIEW_READS as usize + 1],
            "{label}: visible draw must mark texture slot 1"
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
