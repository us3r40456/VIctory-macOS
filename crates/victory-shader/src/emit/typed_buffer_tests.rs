//! Synthetic typed-buffer accesses: the view's format is chosen at runtime,
//! so integer views need `bufferLoad.i32`/`bufferStore.i32` without putting
//! mismatched component bits through the float overloads.

#[cfg(windows)]
use super::{HostShader, VIEW_INTEGER, VIEW_SIGNED, validator};
use super::{VIEW_STRIDES, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 a typed UAV, s1 a raw SRV, s2 a raw UAV, s3 a typed SRV.
/// Each lane loads four dwords from element `lane` of the raw SRV, stores them
/// to the typed UAV, loads the typed element back from both the UAV and SRV,
/// and writes the three copies to dwords `12 * lane .. 12 * lane + 11` of the
/// raw UAV.
const ROUND_TRIP_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0x0
    0xBE94_0302, // s_mov_b32 s20, s2
    0xBE95_0380, // s_mov_b32 s21, 0
    0xF408_040A,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
    0xBE96_0303, // s_mov_b32 s22, s3
    0xBE97_0380, // s_mov_b32 s23, 0
    0xF408_060B,
    0xFA00_0000, // s_load_dwordx4 s[24:27], s[22:23], 0x0
    0x3406_0084, // v_lshlrev_b32 v3, 4, v0
    0x1614_00B0, // v_mul_u32_u24 v10, 48, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE038_1000,
    0x8003_0103, // buffer_load_dwordx4 v[1:4], v3, s[12:15], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE078_1000,
    0x8004_010A, // buffer_store_dwordx4 v[1:4], v10, s[16:19], 0 offen
    0xE01C_2000,
    0x8002_0100, // buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE00C_2000,
    0x8002_0500, // buffer_load_format_xyzw v[5:8], v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE00C_2000,
    0x8006_0C00, // buffer_load_format_xyzw v[12:15], v0, s[24:27], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE078_1010,
    0x8004_050A, // buffer_store_dwordx4 v[5:8], v10, s[16:19], 0 offset:16
    0xE078_1020,
    0x8004_0C0A, // buffer_store_dwordx4 v[12:15], v10, s[16:19], 0 offset:32
    0xBF81_0000, // s_endpgm
];

fn lowered() -> lower::Lowered {
    let range = |kind, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: slot,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: (0..4)
            .map(|slot| RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: slot,
                slot,
            })
            .collect(),
        used: vec![
            range(ResourceType::Uav, 0),
            range(ResourceType::Srv, 1),
            range(ResourceType::Uav, 2),
            range(ResourceType::Srv, 3),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [32, 1, 1],
        }),
        code: ROUND_TRIP_WAVE32
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
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
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        host_wave32: false,
        view_marks: false,
    })
    .expect("lower")
}

#[cfg(windows)]
fn shader() -> HostShader {
    let lowered = lowered();
    HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
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
    }
}

#[test]
fn typed_accesses_emit_both_component_overloads() {
    let lowered = lowered();
    assert_eq!(
        lowered.root_srvs,
        [VIEW_STRIDES],
        "the runtime view-kind table is read"
    );
    let calls = lowered.typed_buffer_calls;
    let [load_f32, load_i32] = calls.loads;
    let [store_f32, store_i32] = calls.stores;
    assert_eq!(
        (load_f32.len(), load_i32.len()),
        (2, 2),
        "typed SRV/UAV loads: f32, i32"
    );
    assert_eq!(
        (store_f32.len(), store_i32.len()),
        (1, 1),
        "typed UAV stores: f32, i32"
    );
    for (float, integer) in [(load_f32, load_i32), (store_f32, store_i32)] {
        for (float, integer) in float.into_iter().zip(integer) {
            assert_ne!(float, integer, "the component overloads must be guarded");
        }
    }
}

#[cfg(windows)]
#[test]
fn rgb32_and_rgba32_integer_views_have_kind_metadata() {
    use super::host::{Stride, View, view_strides};
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32A32_SINT,
        DXGI_FORMAT_R32G32B32A32_UINT,
    };

    let views = [
        View::Buffer {
            data: &[0; 3],
            stride: Stride::Format(DXGI_FORMAT_R32G32B32_UINT, 12),
        },
        View::Buffer {
            data: &[0; 3],
            stride: Stride::Format(DXGI_FORMAT_R32G32B32_SINT, 12),
        },
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Format(DXGI_FORMAT_R32G32B32A32_UINT, 16),
        },
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Format(DXGI_FORMAT_R32G32B32A32_SINT, 16),
        },
    ];
    assert_eq!(
        view_strides(&views),
        [
            12 | VIEW_INTEGER,
            12 | VIEW_INTEGER | VIEW_SIGNED,
            16 | VIEW_INTEGER,
            16 | VIEW_INTEGER | VIEW_SIGNED,
        ]
    );
}

#[cfg(windows)]
#[test]
fn synthetic_typed_accesses_validate() {
    let mut shader = shader();
    let Ok(mut validator) = validator::Validator::load(None) else {
        eprintln!("no dxil.dll; DXIL validation skipped");
        return;
    };
    validator
        .validate(&mut shader.dxil)
        .expect("the guarded typed-buffer module validates");
}

#[cfg(windows)]
#[test]
fn integer_typed_bits_survive_srv_and_uav_round_trips() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_SINT,
        DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_SINT,
        DXGI_FORMAT_R32G32B32A32_UINT,
    };

    use super::host::{Dispatch, Host, Root, Stride, View};

    let shader = shader();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let integer = [
        0xFFFF_FFFF, // UINT_MAX / SINT -1
        0x8000_0000, // INT_MIN
        0x7FC0_0000, // quiet NaN bits
        0x7F80_0001, // signalling NaN bits
        0xFFC0_0001, // negative signalling NaN bits
        0,
        1,
        0x3F80_0000,
    ];
    let float = [
        0,
        1,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x4049_0FDB,
    ];
    // (format, bytes per element, integer, signed)
    let cases: [(DXGI_FORMAT, u32, bool, bool); 6] = [
        (DXGI_FORMAT_R32G32B32A32_UINT, 16, true, false),
        (DXGI_FORMAT_R32G32B32A32_SINT, 16, true, true),
        (DXGI_FORMAT_R32G32B32_UINT, 12, true, false),
        (DXGI_FORMAT_R32G32B32_SINT, 12, true, true),
        (DXGI_FORMAT_R32G32B32A32_FLOAT, 16, false, false),
        (DXGI_FORMAT_R32G32B32_FLOAT, 12, false, false),
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for &(format, bytes, is_integer, signed) in &cases {
            if !host.typed_buffer_access(format, warp) {
                eprintln!("{label}: typed buffer access to {format:?} unsupported; skipped");
                continue;
            }
            let channels = bytes / 4;
            let patterns = if is_integer { &integer } else { &float };
            let data: Vec<u32> = (0..32u32)
                .flat_map(|lane| {
                    (0..channels)
                        .map(move |ch| patterns[(lane as usize + ch as usize) % patterns.len()])
                })
                .collect();
            // The raw source is four dwords an element even for RGB32; the
            // typed SRV and UAV keep their tightly packed three-channel layout.
            let mut raw_data = Vec::with_capacity(128);
            for lane in 0..32 {
                raw_data.extend_from_slice(
                    &data[lane * channels as usize..(lane + 1) * channels as usize],
                );
                if channels == 3 {
                    raw_data.push(0);
                }
            }
            let kind = if is_integer {
                VIEW_INTEGER | if signed { VIEW_SIGNED } else { 0 }
            } else {
                0
            };
            let views = [
                View::TypedUav(format, 32, bytes),
                View::Buffer {
                    data: &raw_data,
                    stride: Stride::Raw,
                },
                View::RawUav(384),
                View::Buffer {
                    data: &data,
                    stride: Stride::Format(format, bytes),
                },
            ];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = 32 * u32::from(u.slot);
            }
            let words = host
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
                                    data: vec![bytes | kind, 4, 4, bytes | kind],
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} {format:?}: dispatch failed: {e}"));
            for copy in 0..3 {
                for lane in 0..32 {
                    for ch in 0..channels as usize {
                        assert_eq!(
                            words[1][lane * 12 + copy * 4 + ch],
                            data[lane * channels as usize + ch],
                            "{label} {format:?}: copy {copy}, lane {lane}, channel {ch}"
                        );
                    }
                }
            }
            for lane in 0..32 {
                for ch in 0..channels as usize {
                    assert_eq!(
                        words[0][lane * channels as usize + ch],
                        data[lane * channels as usize + ch],
                        "{label} {format:?}: UAV lane {lane}, channel {ch}"
                    );
                }
            }
            ran += 1;
        }
    }
    assert!(ran > 0, "no adapter supports typed buffer loads and stores");
}
