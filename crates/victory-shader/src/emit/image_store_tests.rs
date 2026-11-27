//! End-to-end tests of synthetic GFX10 image stores lowered to DXIL,
//! validated by `dxil.dll` when present, and executed on the host GPU and
//! WARP.
//!
//! Each program fills texel (1, 1) of a 2x2 texture UAV with a four-channel
//! store, then stores to it again with the dmask (and d16) under test; the
//! texture is read back. The console writes the channels the dmask leaves
//! out as 0.

use super::host::{Dispatch, Host, Root, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, lower, validator};
use crate::ShaderError;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const IMAGE_STORE: u8 = 0x08;
const IMAGE_STORE_MIP: u8 = 0x09;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], offset` (soffset =
/// null).
const fn s_load_dwordx8(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 3 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0` (`src0`: 0..105 = SGPR, 128 = inline 0, 129
/// = inline 1).
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// MIMG `op vdata, vaddr, s[srsrc:srsrc+7] dmask dim:2D unorm [d16]`
/// (non-NSA).
const fn mimg_2d(op: u8, vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, d16: bool) -> [u32; 2] {
    [
        0x3C << 26
            | (op as u32 & 0x7F) << 18
            | 1 << 12 // unorm
            | (dmask as u32 & 0xF) << 8
            | 1 << 3 // dim 2D
            | (op as u32 >> 7),
        (d16 as u32) << 31 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// Where the second store's mip level comes from (`image_store_mip`).
#[derive(Clone, Copy)]
enum Mip {
    /// `image_store`: no level.
    None,
    /// An inline constant (128 = 0, 129 = 1).
    Inline(u16),
    /// Root constant s10, unknown at translation.
    Computed,
}

/// Root constants s0..s3: the fill; s4..s7: the data under test; s8, s9:
/// the texel; s10: a mip level. s12 is the table of the texture UAV.
fn program(dmask: u8, d16: bool, mip: Mip) -> NativeProgram {
    let mut words = vec![s_mov_b32(13, 0x80)]; // s13 = 0: the table's high dword
    words.extend(s_load_dwordx8(16, 12, 0));
    words.push(v_mov_b32(4, 8));
    words.push(v_mov_b32(5, 9));
    match mip {
        Mip::None => {}
        Mip::Inline(k) => words.push(v_mov_b32(6, k)),
        Mip::Computed => words.push(v_mov_b32(6, 10)),
    }
    for k in 0..4u8 {
        words.push(v_mov_b32(k, u16::from(k)));
    }
    words.extend(mimg_2d(IMAGE_STORE, 0, 4, 16, 0xF, false));
    for k in 0..4u8 {
        words.push(v_mov_b32(k, u16::from(4 + k)));
    }
    let op = match mip {
        Mip::None => IMAGE_STORE,
        _ => IMAGE_STORE_MIP,
    };
    words.extend(mimg_2d(op, 0, 4, 16, dmask, d16));
    words.push(S_ENDPGM);

    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 12,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 12,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 12,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 13 << 1, // 13 user SGPRs: s0..s12
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
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
        halves_local: Default::default(),
        mac: Default::default(),
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        view_marks: false,
    })?;
    Ok(HostShader {
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
    })
}

/// The program lowered and validated.
fn build(dmask: u8, d16: bool, mip: Mip) -> HostShader {
    let shader = lower(&program(dmask, d16, mip)).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// The texture's format.
#[derive(Clone, Copy, Debug)]
enum Format {
    Rgba32Float,
    Rgba32Uint,
    R32Uint,
}

impl Format {
    fn view(self) -> View<'static> {
        match self {
            Format::Rgba32Float => View::TextureUav(2, 2),
            Format::Rgba32Uint => View::Uint4TextureUav(2, 2),
            Format::R32Uint => View::UintTextureUav(2, 2),
        }
    }

    fn channels(self) -> usize {
        match self {
            Format::R32Uint => 1,
            _ => 4,
        }
    }
}

/// A store under test: its dmask, whether its data is d16, and the data
/// dwords (as the VGPRs hold them, the enabled channels' in order).
struct Case {
    dmask: u8,
    d16: bool,
    format: Format,
    data: [u32; 4],
}

/// Texel (1, 1) after the program: the enabled channels from the data VGPRs
/// in order (d16: two per VGPR, low half first, widened to f32), the others
/// 0.
fn expected(case: &Case) -> [u32; 4] {
    let mut texel = [0; 4];
    let mut k = 0;
    for (ch, t) in texel.iter_mut().enumerate() {
        if case.dmask >> ch & 1 == 0 {
            continue;
        }
        *t = if case.d16 {
            let half = (case.data[k / 2] >> (16 * (k % 2))) as u16;
            f16_to_f32(half).to_bits()
        } else {
            case.data[k]
        };
        k += 1;
    }
    texel
}

/// An f16 (normal, zero or infinite) as an f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from(h >> 10 & 0x1F), f32::from(h & 0x3FF));
    match e {
        0 => sign * m * 2f32.powi(-24),
        0x1F => sign * f32::INFINITY,
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The fill: four distinct values, finite floats as f32 and nonzero as
/// integers.
const FILL: [u32; 4] = [0x3F80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000];

fn run(cases: &[Case]) {
    run_with(cases, Mip::None);
}

/// Runs each case's shader (its second store's level from `mip`) on the
/// hardware adapter and WARP (each skipped without an SM 6.6 device) and
/// checks the whole texture.
fn run_with(cases: &[Case], mip: Mip) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let shaders: Vec<_> = cases.iter().map(|c| build(c.dmask, c.d16, mip)).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (case, shader) in cases.iter().zip(&shaders) {
            let what = format!(
                "{label} {:?} dmask {:#x}{}",
                case.format,
                case.dmask,
                if case.d16 { " d16" } else { "" }
            );
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{what}: compute_pipeline: {e}"));
            let mut k = [0u32; 12];
            k[..4].copy_from_slice(&FILL);
            k[4..8].copy_from_slice(&case.data);
            k[8..10].copy_from_slice(&[1, 1]);
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => k[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[case.format.view()],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{what}: dispatch: {e}"));
            let n = case.format.channels();
            let want = expected(case);
            let mut texture = vec![0u32; 4 * n];
            texture[3 * n..].copy_from_slice(&want[..n]);
            assert_eq!(
                out[0].iter().map(|x| format!("{x:#x}")).collect::<Vec<_>>(),
                texture
                    .iter()
                    .map(|x| format!("{x:#x}"))
                    .collect::<Vec<_>>(),
                "{what}: texture"
            );
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no host device could run these; translation and validation only");
    }
}

/// Float data: distinct finite values, none of them the fill's.
const FLOATS: [u32; 4] = [0x3FC0_0000, 0xC000_0000, 0x3E80_0000, 0x447A_0000]; // 1.5, -2, 0.25, 1000

#[test]
fn float_stores_of_each_dmask_run_on_the_host() {
    let cases: Vec<_> = (1..16u8)
        .map(|dmask| Case {
            dmask,
            d16: false,
            format: Format::Rgba32Float,
            data: FLOATS,
        })
        .collect();
    run(&cases);
}

#[test]
fn d16_float_stores_of_some_channels_run_on_the_host() {
    // f16 pairs: (1.5, -2), (0.25, 1000).
    let data = [0xC000_3E00, 0x63D0_3400, 0, 0];
    let cases: Vec<_> = [0x1, 0x5, 0x6, 0x8, 0xB, 0xE, 0xF]
        .into_iter()
        .map(|dmask| Case {
            dmask,
            d16: true,
            format: Format::Rgba32Float,
            data,
        })
        .collect();
    run(&cases);
}

/// Integer data: denormal, NaN and negative-zero bit patterns as floats.
const INTS: [u32; 4] = [0x0000_0001, 0x007F_FFFF, 0x7FC0_0001, 0x8000_0000];

#[test]
fn integer_stores_run_on_the_host() {
    let mut cases = Vec::new();
    for dmask in [0x1, 0x6, 0xA, 0xF] {
        cases.push(Case {
            dmask,
            d16: false,
            format: Format::Rgba32Uint,
            data: INTS,
        });
    }
    for x in [1, 0x7F_FFFF, 0x7FC0_0001, 0xFFFF_FFFF, 0x8000_0000] {
        cases.push(Case {
            dmask: 0x1,
            d16: false,
            format: Format::R32Uint,
            data: [x, 0, 0, 0],
        });
    }
    run(&cases);
}

#[test]
fn mip_stores_to_the_views_level_run_on_the_host() {
    let cases = [
        Case {
            dmask: 0xF,
            d16: false,
            format: Format::Rgba32Float,
            data: FLOATS,
        },
        Case {
            dmask: 0x9,
            d16: false,
            format: Format::Rgba32Float,
            data: FLOATS,
        },
        Case {
            dmask: 0x1,
            d16: false,
            format: Format::R32Uint,
            data: INTS,
        },
    ];
    run_with(&cases, Mip::Inline(128));
}

#[test]
fn mip_stores_past_the_views_level_or_computed_stop_the_program() {
    for mip in [Mip::Inline(129), Mip::Computed] {
        let Err(e) = lower(&program(0xF, false, mip)) else {
            panic!("a store to another level lowered");
        };
        assert!(
            matches!(
                e,
                ShaderError::Unsupported {
                    what: "an image store to a mip level other than 0 or unknown",
                    ..
                }
            ),
            "{e}"
        );
    }
}
