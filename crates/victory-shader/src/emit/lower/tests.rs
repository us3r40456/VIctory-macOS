//! Tests of the lowering (`super`). In a file of their own so that
//! `lower.rs` stays under the asset policy's file size limit (512 KiB).

use super::{
    Lowered, Numeric, Parts, barycentric_signature, buffer_format, code_table_start, half_shift,
    lower, packed_half,
};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A program's constant table starts after its instructions, unless a
/// load at a fixed offset reads below that.
#[test]
fn the_code_table_starts_after_the_instructions() {
    // s_nop 0, s_endpgm, then 1,000 dwords of constants.
    let mut words = vec![0xBF80_0000, S_ENDPGM];
    words.extend(0..1000u32);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let at = |offset: u32, dynamic: bool| resource::Address {
        param: resource::CODE,
        offset,
        dynamic: dynamic.then_some(ir::Value(0)),
    };
    let (computed, fixed_after, fixed_inside) = (at(0, true), at(12, false), at(4, false));
    // A load at a computed offset from the code's start: the table is
    // the constants only, not the whole program.
    assert_eq!(code_table_start(&p, &[&computed], 0), 8);
    assert_eq!(code_table_start(&p, &[&computed, &fixed_after], 0), 8);
    // A load at a fixed offset inside the instructions keeps the start.
    assert_eq!(code_table_start(&p, &[&computed, &fixed_inside], 0), 0);
    // Loads past the instructions only: from the lowest of them.
    assert_eq!(code_table_start(&p, &[&fixed_after], 12), 12);
}

#[test]
fn direct_barycentric_modes_match_dxil_semantics() {
    use crate::io::Barycentric::*;

    assert_eq!(barycentric_signature(PerspectiveSample), Some((0, 6)));
    assert_eq!(barycentric_signature(PerspectiveCenter), Some((0, 2)));
    assert_eq!(barycentric_signature(PerspectiveCentroid), Some((0, 3)));
    assert_eq!(barycentric_signature(LinearSample), Some((1, 7)));
    assert_eq!(barycentric_signature(LinearCenter), Some((1, 4)));
    assert_eq!(barycentric_signature(LinearCentroid), Some((1, 5)));
    assert_eq!(barycentric_signature(PerspectivePullModel), None);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
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
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
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
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

/// Lowers a one-thread compute program that copies four root constants
/// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
/// descriptor table, so the vector instructions in `body` have an effect.
fn lower_valu(body: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    // v_mov_b32 v1..v4, s0..s3
    words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
        0x8002_0000,
        S_ENDPGM,
    ]);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
}

fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
    lower_valu(body).map(|_| ())
}

#[test]
fn vop3_conversion_modifiers_affect_the_lowering() {
    for (op, first, second, modifier) in [
        (0x0Au32, 1 << 15, 0, "clamp"),
        (0x0B, 1 << 8, 0, "abs"),
        (0x0B, 0, 1 << 29, "neg"),
    ] {
        let vop1 = lower_valu(&[0x7E00_0000 | op << 9 | 0x101])
            .unwrap()
            .dxil;
        let vop3 = |first: u32, second: u32| {
            lower_valu(&[
                0xD400_0000 | (0x180 + op) << 16 | first,
                0x101 | second,
            ])
            .unwrap()
            .dxil
        };
        assert_eq!(vop1, vop3(0, 0), "VOP1/VOP3 control for op {op:#x}");
        assert_ne!(
            vop1,
            vop3(first, second),
            "VOP3 {modifier} was dropped for op {op:#x}"
        );
    }
}

#[test]
fn the_24_bit_multiplies_lower_both_halves() {
    // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
    // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
    for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
        let word = op << 25 | 2 << 9 | 0x101;
        lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
    }
}

#[test]
fn sums_of_absolute_differences_lower() {
    // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
    // v_sad_u32 and v_msad_u8.
    for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
        let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
        lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
    }
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

/// A full typed store to element `s0` through the table at `s1`, with
/// `soffset` in the MUBUF instruction's scalar-source field.
fn typed_store_words(soffset: u8) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0301, // s_mov_b32 s4, s1
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0x7E00_0200, // v_mov_b32 v0, s0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend((1..=4).map(|v| 0x7E00_0000 | (v as u32) << 17 | 0x01 << 9 | 0x80));
    words.extend([
        0xE01C_2000,
        0x8002_0100 | (soffset as u32) << 24, // buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen
        S_ENDPGM,
    ]);
    words
}

#[test]
fn typed_stores_refuse_register_and_immediate_soffsets() {
    let layout = || vec![root(RootType::Constants, 0, 0), root(RootType::Table, 1, 1)];
    // RGBA32 and RGB32 use the same full-store MUBUF form. The mark width
    // comes from the host's stride table; the used-resource stride here
    // keeps both descriptor layouts in the regression matrix.
    for (format, stride) in [("RGBA32", 16), ("RGB32", 12)] {
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride,
                dwords: 4,
                user_data: 1,
                slot: 1,
            }]
        };
        // A register and an inline constant both make an instruction-valued
        // offset. Neither may be silently dropped until offsets can fold into
        // the typed element index. Inline zero is the accepted no-offset form.
        for (source, soffset) in [("register", 6), ("immediate", 132)] {
            let Err(crate::ShaderError::Unsupported { what, .. }) =
                lower_with(&typed_store_words(soffset), layout(), used())
            else {
                panic!("translated a {format} typed store with a {source} soffset");
            };
            assert_eq!(
                what, "a typed buffer store at an offset",
                "{format} {source} soffset"
            );
        }
        lower_with(&typed_store_words(128), layout(), used())
            .unwrap_or_else(|e| panic!("{format} inline zero soffset: {e}"));
    }
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn v_cndmask_accepts_uniform_and_constant_masks() {
    // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
    // (resolved in `Structured::other` or directly as constants).
    let words = [
        0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
        0x8500_80C1,    // s_cselect_b32 s0, -1, 0
        0xD501_0001,
        0x0001_0080,    // v_cndmask_b32 v1, 0, 1, s0
        0xD501_0002,
        0x0201_02C1,    // v_cndmask_b32 v2, 1, v1, -1
        0xD834_0000,
        0x0000_0200,    // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    lowers(&words, 0x8001).unwrap();
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32")
    );

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
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
    .unwrap();

    let container = crate::Container::parse(&emitted.dxil).unwrap();

    let features = container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn a_descriptor_loaded_only_for_its_size_takes_the_kind_of_its_reload() {
    // The record count of the buffer at s0 (count - 1 as an index), then
    // the same descriptor loaded again for the indexed read.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x810C_C10A, // s_add_i32 s12, s10, -1
        0xF408_0402,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], null
        0x7E04_020C, // v_mov_b32 v2, s12
        0xE030_2000,
        0x8004_0102, // buffer_load_dword v1, v2, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_with(&words, layout, used).unwrap();
}

#[test]
fn a_waterfall_loop_that_only_sizes_takes_the_kind_of_the_sampled_one() {
    // Each lane names table entry (lane & 1) by its byte offset in v7.
    // A first waterfall loop only reads the width bits of the entry
    // (`v_alignbit_b32`), a second one samples it; the width goes to the
    // buffer at dword 16 of the table.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // L1: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD54E_0008,
        0x0278_1A0E, // v_alignbit_b32 v8, s14, s13, 30
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF7, // s_cbranch_execnz L1
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // L2: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz L2
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0810, // buffer_store_dword v8, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (layout, used) = texture_and_uav_tables();
    lower_wave(&words, layout, used, false).unwrap();
}

#[test]
fn permlanex16_lowers_in_wave32_and_wave64() {
    let words = [
        0xD778_0001,
        0x0004_0100, // v_permlanex16_b32 v1, v0, s0, s1
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![RootParam {
        sgprs: 2,
        ..root(RootType::Constants, 0, 0)
    }];
    for wave64 in [false, true] {
        let result = lower_wave(&words, layout.clone(), vec![], wave64).unwrap();
        assert!(result.wave_ops);
    }
}

#[test]
fn compute_spills_beside_a_descriptor_word_read_back() {
    // Dword 0 of a buffer descriptor (no host value) and a root constant
    // spilled to lanes 0 and 1 of v5 in a compute program; reading lane
    // 1 back is the constant (the descriptor's lane is never read).
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xD761_0005,
        0x0001_0008, // v_writelane_b32 v5, s8, 0
        0xD761_0005,
        0x0001_0201, // v_writelane_b32 v5, s1, 1
        0xD760_000C,
        0x0001_0305, // v_readlane_b32 s12, v5, 1
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0x7E06_020C, // v_mov_b32 v3, s12
        0xD834_0000,
        0x0000_0300, // ds_write_b32 v0, v3
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_with(&words, layout, used).unwrap();
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn an_address_half_set_where_only_lanes_that_never_run_skip_is_known() {
    // As `indirect_light_probes:CS_Alpha_Schedule` after its texture: a
    // table address whose high half (s25 = 0) is set inside the region
    // that only a skip no lane takes bypasses; the skip edge would bring
    // the stale input s25.
    let words = [
        0xBE9A_037E, // s_mov_b32 s26, exec_lo
        0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
        0xBE98_0300, // s_mov_b32 s24, s0
        0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
        0xBF88_0001, // s_cbranch_execz -> join
        0xBE99_0380, // s_mov_b32 s25, 0
        0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
        0xF408_0A0C,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[24:25], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E06_0280, // v_mov_b32 v3, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    lower_with(&words, layout, used).expect("the high half is 0 wherever a lane runs");
}

#[test]
fn a_descriptor_skipped_only_by_lanes_that_never_run_needs_no_source() {
    // As `indirect_light_probes:CS_Alpha_Schedule`: the texture is
    // loaded under the lanes that ran before `s_wqm`, skipped when there
    // are none, and sampled with the helper lanes after the join. With
    // every lane running from the start nothing takes the skip, so the
    // registers it would bring (no descriptor) are never read.
    let words = [
        0xBE9A_037E, // s_mov_b32 s26, exec_lo
        0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
        0xBF88_0002, // s_cbranch_execz -> join
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E04_02F0, // v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    let l = lower_with(&words, layout, used).unwrap();
    assert_eq!(l.nonuniform_handles, 0);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
    // entry 1 if it is 1, else entry 0 again, the inner if/else joined
    // before the outer one. The join's edges are a load and a join.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0003, // s_cbranch_scc0 -> 0x38
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF82_0008, // s_branch -> 0x58
        0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
        0xBF84_0003, // s_cbranch_scc0 -> 0x4c
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xBF82_0002, // s_branch -> 0x54
        0xF40C_0302,
        0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE86_0301, // 0x54: s_mov_b32 s6, s1
        0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle from the joins' heap index, the same in all lanes.
    assert_eq!(l.nonuniform_handles, 0);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn descriptors_reloaded_from_a_root_table_share_one_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1. Entry 0 is loaded and sampled
    // in a branch, then loaded again and sampled after it (the console
    // compiler reloads a descriptor before each use rather than keep it).
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E04_02F0, // v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0005, // s_cbranch_scc0 -> the reload
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0 (again)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle made in the entry block for both loads, and the
    // UAV's.
    assert_eq!(l.heap_handles, 2);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn a_value_no_lane_from_its_edge_reads_needs_no_form() {
    // s6 is a timestamp (no DXIL form) where the load is skipped and 5
    // where it runs; a flag in v1, set only with the load, guards the
    // read after the join (the console compiler's form, where the
    // skipped path leaves a stale lane mask in the register).
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    lowers(&words, group_ids_xy).expect("the timestamp is never read");
    // With the flag set on both paths the timestamp is read: as refused
    // (or accepted) as a direct read.
    words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[9..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
}

#[test]
fn a_mask_joined_with_data_needs_a_uniform_join() {
    // s6: lanes below 24, then shifted (data) on each back edge; read
    // after the loop. Lanes leaving the loop at different iterations
    // (per lane) see different numbers than the console's register:
    // refused. The uniform loop is `synthetic_tests::
    // a_mask_joined_with_data_at_a_uniform_loop_is_its_number`.
    let mut words = [
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
        0xBE87_0380, // s_mov_b32 s7, 0
        0x9006_8406, // header: s_lshr_b32 s6, s6, 4
        0x8007_8107, // s_add_u32 s7, s7, 1
        0x7D88_0007, // v_cmp_gt_u32 vcc_lo, s7, v0
        0x8A7E_6A7E, // s_andn2_b32 exec_lo, exec_lo, vcc_lo
        0xBF89_FFFB, // s_cbranch_execnz header
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowers(&words, 1 << 7) else {
        panic!("a per-lane loop's mask join translated");
    };
    assert!(what.contains("lane mask"), "{what}");
    // Leaving on a uniform condition instead: the mask's number.
    words[5] = 0xBF0A_8207; // s_cmp_lt_u32 s7, 2
    words[6] = 0xBF80_0000; // s_nop 0
    words[7] = 0xBF85_FFFB; // s_cbranch_scc1 header
    lowers(&words, 1 << 7).expect("a uniform loop's mask join");
}

#[test]
fn a_scalar_result_needs_no_exec() {
    // s6 is 0 where every lane skips and the saved EXEC where some run:
    // no lane mask at the per-lane join. EXEC taken from it guards only
    // a `v_readlane` (written whatever EXEC holds) before it is restored
    // (`CS_IrradianceCache_Shade`).
    let words = [
        0xBE86_0380, // s_mov_b32 s6, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
        0xBF88_0001, // s_cbranch_execz +1
        0xBE86_0304, // s_mov_b32 s6, s4
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD760_0002, // v_readlane_b32 s2, v0, 3
        0x0001_0700,
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        S_ENDPGM,
    ];
    lowers(&words, 1 << 7).expect("a lane read under an EXEC with no mask");
}

#[test]
fn guarded_or_modified_v_mov_b32_is_not_folded_as_constant() {
    let group_ids_xy = 3 << 7;
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert!(refusal(&direct).is_some());

    // Guarded `v_mov_b32 v1, 0`: inactive lanes (v0 != 0) preserve v1 = -1,
    // so they pass the compare after the join and read s6 (`s_memtime`).
    let mut guarded = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0x7E02_0280, // v_mov_b32 v1, 0 (guarded)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert_eq!(refusal(&guarded), refusal(&direct));
    // Without the EXEC guard, `v_mov_b32 v1, 0` writes every lane and s6 is dead.
    guarded[3] = 0xBF80_0000; // s_nop 0
    guarded[5] = 0xBF80_0000; // s_nop 0
    lowers(&guarded, group_ids_xy).expect("unguarded v_mov_b32 still folds");

    // VOP3 `v_mov_b32_e64 v1, -(1.0)`: neg modifier flips 1.0f to -1.0f,
    // matching the compare after the join so s6 is live.
    let mut negated = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xD581_0001,
        0x2000_00F2, // v_mov_b32_e64 v1, -(1.0)
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02F3, // v_mov_b32 v1, -1.0
        0x7D84_02F3, // join: v_cmp_eq_u32 vcc_lo, -1.0, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert_eq!(refusal(&negated), refusal(&direct));
    // Clearing the neg bit yields unmodified 1.0f (!= -1.0f), so s6 is dead.
    negated[3] = 0x0000_00F2; // v_mov_b32_e64 v1, 1.0
    lowers(&negated, group_ids_xy).expect("unmodified v_mov_b32_e64 still folds");
}

#[test]
fn a_missing_value_on_a_path_no_lane_takes_is_dead() {
    let refusal = |code: &[u32]| match lowers(code, 1 << 7) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    // The timestamp stays in s6 only where s0 == 0, and those lanes
    // overwrite it before the second join: it never reaches the read.
    let mut joins = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF84_0001, // s_cbranch_scc0: s0 != 0 keeps that 0
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001,
        0xBE83_0380,
        S_ENDPGM,
    ];
    assert_eq!(refusal(&joins), None);
    // Tested on the other side, the timestamp reaches the read.
    joins[5] = 0xBF85_0001; // s_cbranch_scc1
    assert!(refusal(&joins).is_some());
    // The same through a VGPR whose old value a guarded write keeps
    // (lanes outside it get 0 for a value with no DXIL form, as for any
    // such write): keeping does not make the join live.
    let kept = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0206, // v_mov_b32 v1, s6
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF84_0004, // s_cbranch_scc0: s0 != 0 keeps that 0
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0x7E02_0282, // v_mov_b32 v1, 2 (the other lanes keep v1)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0x7E02_0283, // v_mov_b32 v1, 3
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0xBF87_0001, // s_cbranch_vccnz
        0xBE83_0380,
        S_ENDPGM,
    ];
    assert_eq!(refusal(&kept), None);
}

#[test]
fn a_guarded_scalar_write_kills_a_stale_sgpr() {
    // s6 holds a missing value (timestamp) where s0 == 0 and 5 where
    // s0 != 0; the flag in v1 skips the first read on the s0 == 0 path.
    // After that, inside a guarded region (EXEC narrowed without a
    // branch), `v_readfirstlane_b32` overwrites s6 unconditionally before
    // s6 is read again.
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> after
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> after
        0xBE84_0380, // s_mov_b32 s4, 0
        0x7DA4_0080, // after: v_cmpx_eq_u32 exec_lo, 0, v0 (narrows EXEC)
        0x7E0C_0500, // v_readfirstlane_b32 s6, v0 (guarded, writes SGPR)
        0x7DA4_0006, // v_cmpx_eq_u32 exec_lo, s6, v0 (reads new s6)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    lowers(&words, group_ids_xy).expect("guarded v_readfirstlane overwrites stale s6");
    // Without the overwrite, the guarded compare reads the stale s6.
    let mut no_kill = words;
    no_kill[13] = 0xBF80_0000; // s_nop 0
    assert!(lowers(&no_kill, group_ids_xy).is_err());
    // With the flag set on both paths, the earlier read sees the stale s6.
    words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
    assert!(lowers(&words, group_ids_xy).is_err());
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn packed_halves_fill_parameters_first_fit() {
    // 16-bit rows as a title's signature has them: row 0 holds two
    // elements (x y and z w), row 3 is 32-bit, row 4 has one channel at w.
    let el = |row: u32, col: u8, cols: u8, half: bool| super::SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half,
    };
    let sig = [
        el(0, 0, 2, true),
        el(0, 2, 2, true),
        el(1, 0, 4, true),
        el(2, 0, 3, true),
        el(3, 0, 4, false),
        el(4, 0, 3, true),
        el(4, 3, 1, true),
        el(5, 0, 3, true),
    ];
    // Each half as row and channel ("-": padding or past the end).
    let param = |a| {
        (0..8)
            .map(|h| match packed_half(&sig, a, h) {
                Some((r, c)) => format!("{r}{}", ["x", "y", "z", "w"][usize::from(c)]),
                None => "-".into(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    // By first channel, then row: row 0's x y, all of row 1, then (last,
    // as it starts at z) row 0's z w.
    assert_eq!(param(0), "0x 0y 1x 1y 1z 1w 0z 0w");
    // Three channels take two dwords; row 4's w fills the first
    // parameter with room, after row 5.
    assert_eq!(param(2), "2x 2y 2z - 4x 4y 4z -");
    assert_eq!(param(5), "5x 5y 5z - 4w - - -");
    // No parameter starts in rows 1, 3 or 4.
    for a in [1, 3, 4] {
        assert_eq!(param(a), "- - - - - - - -");
    }
}

#[test]
fn ds_bpermute_refuses_gds() {
    // ds_bpermute_b32 with gds=1 (bit 17), consumed by a branch, must fail with GDS.
    let words = [
        0x7E02_0285,           // v_mov_b32 v1, 5
        0xDACC_0000 | 1 << 17, // ds_bpermute_b32 gds
        2 << 24 | 1 << 8,
        0x7E00_0502, // v_readfirstlane_b32 s0, v2
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    assert!(matches!(
        lowers(&words, 0),
        Err(crate::ShaderError::Unsupported { what: "GDS", .. })
    ));
}

// ---- one console wave split across host waves ----

const V_LSHLREV_V1_2_V0: u32 = 0x3402_0082; // v_lshlrev_b32 v1, 2, v0
const DS_WRITE_V1_V0: [u32; 2] = [0xD834_0000, 0x0000_0001]; // ds_write_b32 v1, v0
const DS_READ_V2_V1_4: [u32; 2] = [0xD8D8_0004, 0x0200_0001]; // ds_read_b32 v2, v1 offset:4
const SAVE_EXEC: u32 = 0xBE80_047E; // s_mov_b64 s[0:1], exec
const CMPX_V0_EQ_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec, 0, v0
const RESTORE_EXEC: u32 = 0xBEFE_0400; // s_mov_b64 exec, s[0:1]

/// `s_cbranch_execz` over the next `words` words.
const fn skip(words: u32) -> u32 {
    0xBF88_0000 | words
}

fn wave64(words: &[u32]) -> ir::Program {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, true, 256).unwrap()
}

/// [`super::split_wave_barriers`] of a program.
fn barriers_at(p: &ir::Program) -> Option<Vec<(usize, usize)>> {
    let st = structure::structure(p).unwrap();
    let mut at: Vec<_> = super::split_wave_barriers(p, &st)?.into_iter().collect();
    at.sort_unstable();
    Some(at)
}

/// Lowers a wave64 compute program of `lanes` threads with 512 bytes of
/// LDS: the barriers it got for narrow host waves.
fn split_barriers(words: &[u32], lanes: u32, narrow_waves: bool) -> usize {
    lowered(words, lanes, narrow_waves).split_barriers
}

/// The same program, lowered.
fn lowered(words: &[u32], lanes: u32, narrow_waves: bool) -> super::Lowered {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 2 << 1,
            threads: [lanes, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves,
        halves_local: Default::default(),
        mac: Default::default(),
        host_wave32: false,
        view_marks: false,
    })
    .expect("lower")
}

#[test]
fn a_read_after_another_lanes_write_syncs_in_a_split_wave() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), [(0, 2)]);
    assert_eq!(split_barriers(&w, 64, true), 1);
    // Option off, or a group one host wave holds: as before.
    assert_eq!(split_barriers(&w, 64, false), 0);
    assert_eq!(split_barriers(&w, 32, true), 0);
    // Synced, it no longer asks for 64-lane waves a 32-lane host lacks
    // (D3DMetal runs nothing then); without the option it still does.
    assert_eq!(lowered(&w, 64, true).wave_size, None);
    assert_eq!(lowered(&w, 64, false).wave_size, Some(64));
    assert_eq!(lowered(&w, 32, true).wave_size, None);
}

#[test]
fn a_divergent_write_syncs_at_the_join_before_the_read() {
    // 34d88edf583f3778's shape: lanes with v0 = 0 write, all read.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(2)];
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    let at = barriers_at(&p).unwrap();
    assert_eq!(at.len(), 1);
    let (b, i) = at[0];
    assert!(p.blocks[b].insts[i].name().starts_with("ds_read"));
    assert_eq!(p.blocks[b].preds.len(), 2, "the join");
    assert_eq!(split_barriers(&w, 64, true), 1);
}

#[test]
fn a_write_before_divergent_reads_syncs_where_all_lanes_run() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend([SAVE_EXEC, CMPX_V0_EQ_0, skip(2)]);
    w.extend(DS_READ_V2_V1_4);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    let at = barriers_at(&p).unwrap();
    assert_eq!(at, [(0, p.blocks[0].insts.len())], "the end of the entry");
    assert_eq!(split_barriers(&w, 64, true), 1);
}

#[test]
fn no_barrier_in_divergent_code() {
    // Write and read both under the branch: no barrier can go there.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(4)];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    assert!(barriers_at(&wave64(&w)).is_none());
    assert_eq!(split_barriers(&w, 64, true), 0);
}

const S_MOV_S20_0: u32 = 0xBE94_0380; // s_mov_b32 s20, 0
const S_ADD_S20_1: u32 = 0x8014_8114; // s_add_u32 s20, s20, 1
const S_CMP_LT_S20_4: u32 = 0xBF0A_8414; // s_cmp_lt_u32 s20, 4
const S_CMP_LT_S20_S21: u32 = 0xBF0A_1514; // s_cmp_lt_u32 s20, s21
const READFIRSTLANE_S21_V0: u32 = 0x7E2A_0500; // v_readfirstlane_b32 s21, v0

/// `body` in a loop on s20, ending at the compare `cmp` (`s_cbranch_scc1`
/// back to the body's start).
fn looped(before: &[u32], body: &[u32], cmp: u32) -> Vec<u32> {
    let mut w = before.to_vec();
    w.push(S_MOV_S20_0);
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([S_ADD_S20_1, cmp, 0xBF85_0000 | (back as u32 & 0xFFFF)]);
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_loop_programs_are_what_their_comments_say() {
    let w = looped(
        &[V_LSHLREV_V1_2_V0, READFIRSTLANE_S21_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_S21,
    );
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_readfirstlane_b32 s21, v0",
        "s_mov_b32 s20, 0",
        "ds_write_b32 v1, v0",
        "ds_read_b32 v2, v1 offset:4",
        "s_add_u32 s20, s20, 1",
        "s_cmp_lt_u32 s20, s21",
        "s_cbranch_scc1",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    let w = looped(&[], &[], S_CMP_LT_S20_4);
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(all.contains("s_cmp_lt_u32 s20, 4"), "{all}");
}

#[test]
fn lds_in_a_loop_on_a_scalar_count_syncs_inside_the_loop() {
    // Each pass writes, then reads what another lane wrote: a barrier
    // before the read, and one before the write, which the last pass's read
    // of another lane must not miss.
    let w = looped(
        &[V_LSHLREV_V1_2_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_4,
    );
    let p = wave64(&w);
    assert!(p.blocks[1].insts[0].name().starts_with("ds_write"));
    assert_eq!(barriers_at(&p).unwrap(), [(1, 0), (1, 1)]);
    assert_eq!(split_barriers(&w, 64, true), 2);
    assert_eq!(lowered(&w, 64, true).wave_size, None);
}

#[test]
fn lds_in_a_loop_on_a_lane_value_gets_no_barrier() {
    // The count comes from a lane (`v_readfirstlane`): the host waves need
    // not agree on it, so no barrier can go in the loop.
    let w = looped(
        &[V_LSHLREV_V1_2_V0, READFIRSTLANE_S21_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_S21,
    );
    assert!(barriers_at(&wave64(&w)).is_none());
    assert_eq!(split_barriers(&w, 64, true), 0);
    // A loop on a lane's own condition (`s_cbranch_vccnz`) neither.
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.extend([
        0x7D88_0480, // v_cmp_gt_u32 vcc, 0, v2
        0xBF87_FFFA, // s_cbranch_vccnz (back to the write)
        S_ENDPGM,
    ]);
    assert!(barriers_at(&wave64(&w)).is_none());
}

#[test]
fn a_write_after_another_lanes_read_syncs_before_it() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), [(0, 2), (0, 3), (0, 4)]);
}

#[test]
fn a_read_before_divergent_writes_syncs_where_all_lanes_run() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_READ_V2_V1_4);
    w.extend([SAVE_EXEC, CMPX_V0_EQ_0, skip(2)]);
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    assert_eq!(barriers_at(&p).unwrap(), [(0, p.blocks[0].insts.len())]);
    // A write after a read, both under the branch: no barrier can go
    // there, and the program runs as before.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(4)];
    w.extend(DS_READ_V2_V1_4);
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), []);
}

#[test]
fn ds_bpermute_wave64_preserves_32_lane_half() {
    // Without reading v0 (`thread_id`), only the wave64 half-wave base
    // (`lane_index & !31`) introduces `dx.op.waveGetLaneIndex`.
    let words = [
        0x7E02_02FF,
        0x0000_0104, // v_mov_b32 v1, 0x104
        0x7E04_0285, // v_mov_b32 v2, 5
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
        0x7E00_0503, // v_readfirstlane_b32 s0, v3
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for wave64 in [false, true] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code: code.clone(),
        };
        let p = ir::build(&native.code, wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let mut lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
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
        .unwrap();
        assert!(lowered.wave_ops);
        assert_eq!(lowered.uses_lane_index, wave64, "wave64={wave64}");
        assert_eq!(lowered.has_half_wave_mask, wave64, "wave64={wave64}");
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut lowered.dxil).unwrap();
        }
    }
}

fn lower_ps_for_early_z(
    extra: &[u32],
    export: [u32; 2],
    outputs: Vec<crate::emit::container::SigElement>,
    db_shader_control: u32,
    view_marks: bool,
) -> Lowered {
    use crate::native::{HwStage, NativeProgram, RootParam, RootType, StageIo};
    const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302, // s_load_dwordx8 s[12:19], s[4:5], 32
        0xFA00_0020,
        S_WAITCNT_ALL,
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_1F08, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0x0003_0200,
        S_WAITCNT_ALL,
    ];
    w.extend_from_slice(extra);
    w.extend_from_slice(&export);
    w.push(S_ENDPGM);
    let native = NativeProgram {
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
            db_shader_control,
            patches: Vec::new(),
        },
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = crate::structure::structure(&p).expect("structure");
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = crate::io::raise(&p, &native).expect("io::raise");
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
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

#[test]
fn pixel_shader_with_only_read_marks_enables_early_depth_stencil() {
    let lowered = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 0, true);
    assert!(lowered.view_marks);
    assert!(lowered.early_depth_stencil);
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut lowered.dxil.clone()).unwrap();
    }
}

#[test]
fn pixel_shader_skips_early_depth_stencil_with_discard_depth_or_uav() {
    // 1. No view_marks -> no injected UAV, no need for ForceEarlyDepthStencil.
    let no_marks = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 0, false);
    assert!(!no_marks.early_depth_stencil);

    // 2. Depth export (exp mrtz v2, off, off, off done vm).
    let depth_out = vec![crate::emit::container::SigElement {
        name: "SV_Depth".into(),
        index: 0,
        kind: 17,
        comp: 9,
        interp: 0,
        row: None,
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    let depth = lower_ps_for_early_z(&[], [0xF800_1881, 0x0000_0002], depth_out, 0, true);
    assert!(!depth.early_depth_stencil);

    // 3. Stencil / POPS (db_shader_control bit 16).
    let pops = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 1 << 16, true);
    assert!(!pops.early_depth_stencil);

    // 4. Conditional discard (divergent forward exit to `exp null done`).
    let discard = lower_ps_for_early_z(
        &[
            0xBE88_037E, // s_mov_b32 s8, exec_lo
            0x7C22_0480, // v_cmpx_lt_f32 exec, 0, v2
            0x8A08_7E08, // s_andn2_b32 s8, s8, exec_lo
            0xBF84_0004, // s_cbranch_scc0 -> kill (past s_mov + exp mrt0 + s_endpgm)
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
            0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
            0x0504_0302,
            S_ENDPGM,
        ],
        [0xF800_1890, 0x0000_0000], // kill: exp null done vm
        Vec::new(),
        0,
        true,
    );
    assert!(!discard.early_depth_stencil);

    // 5. Title UAV store (`buffer_store_dword` through heap slot 2).
    let uav = lower_ps_for_early_z(
        &[
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 64
            0xFA00_0040,
            0xBF8C_0000, // s_waitcnt lgkmcnt(0)
            0xE070_0000, // buffer_store_dword v2, off, s[8:11], 0
            0x8002_0000,
        ],
        [0xF800_180F, 0x0504_0302],
        Vec::new(),
        0,
        true,
    );
    assert!(!uav.early_depth_stencil);
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// Lowers a one-thread compute program with a pointer to title memory in
/// root constants s[0:1] (and s2, s3), the raw UAV of table entry 0 (s4) in
/// s[8:11], around `body`.
fn lower_memory(body: &[u32]) -> Lowered {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let l = lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
    .expect("lower");
    assert!(l.memory);
    l
}

/// Each title-memory access reads the page table twice, its level-1 entry
/// and then the page's entry (view and start in one load), and the null
/// pair's view, the same for all, is read once in the entry block: here
/// two loads behind a branch from it.
#[test]
fn the_null_pair_is_read_once_in_the_entry_block() {
    let mut w = vec![
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBF85_000A, // s_cbranch_scc1 -> s_endpgm
        0x7E02_0280, // v_mov_b32 v1, 0
    ];
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(global(0x0C, 3, 1, 0, 0x100));
    w.push(0xBF8C_0000); // s_waitcnt
    w.extend([0xE070_0000, 0x8002_0200]); // buffer_store_dword v2, off, s[8:11], 0
    w.extend([0xE070_0004, 0x8002_0300]); // v3 at 4
    let l = lower_memory(&w);
    let [loads, table] = &l.raw_loads;
    assert_eq!(table.len(), 1 + 2 * 2, "{table:?}");
    assert_eq!(loads.len(), 1 + 2 * 3, "{loads:?}");
    assert_eq!(table.iter().filter(|&&b| b == 0).count(), 1, "{table:?}");
    assert_eq!(loads.iter().filter(|&&b| b == 0).count(), 1, "{loads:?}");
}

/// A box32 node's last 48 bytes are read through its first 64 bytes'
/// page-table lookup, and looked up again only in lanes whose node starts
/// in the last 64 bytes of a page (`raw_load_past`): the head's lookup and
/// the null pair's view in the entry block, then the head's 16 dwords and
/// the tail's 12 through that lookup, and the second lookup in a block of
/// its own.
#[test]
fn a_box32_nodes_tail_reuses_its_heads_lookup() {
    // The descriptor in s[12:15], based at the pointer.
    let mut w = vec![
        0x908C_8800, // s_lshr_b64 s[12:13], s[0:1], 8
        0x880D_FF0D, // s_or_b32 s13, s13, 0x80000000
        0x8000_0000,
        0xBE8E_03C1, // s_mov_b32 s14, -1
        0xBE8F_03FF, // s_mov_b32 s15: type 8, barycentrics, no bound
        8 << 28 | 1 << 24 | 1 << 27 | 0x3FF,
    ];
    // v2..v12: the node pointer and the ray, from s2 and s3.
    w.extend((0..11u32).map(|k| 0x7E00_0200 | (2 + k) << 17 | (2 + k % 2)));
    // image_bvh_intersect_ray v[20:23], v[2:12], s[12:15]
    w.extend([0xF198_9F01, 20 << 8 | (12 >> 2) << 16 | 2]);
    // buffer_store_dwordx4 v[20:23], off, s[8:11], 0
    w.extend([0xE078_0000, 0x8002_1400]);
    let l = lower_memory(&w);
    let [loads, table] = &l.raw_loads;
    let at = |b: usize| loads.iter().filter(|&&x| x == b).count();
    assert_eq!(table.len(), 1 + 2 + 2, "{table:?}");
    assert_eq!(loads.len(), 1 + (2 + 4) + 3 + (2 + 3), "{loads:?}");
    assert_eq!(table[..3], [0; 3], "{table:?}");
    let again = table[3];
    assert!(table[4] == again && at(again) == 2, "{loads:?} {table:?}");
    let through_head = loads.iter().filter(|&&b| b != 0 && b < again);
    assert_eq!(through_head.count(), 4 + 3, "{loads:?} {table:?}");
}
