//! Size queries on writable images (`lower::mac`, `Mac::uav_level0`,
//! `Options::uav_size_level0`; `notes/uav-size.md`). The console compiler
//! reads an image's size from its descriptor or with `image_get_resinfo`;
//! both become DXIL's `getDimensions`, for a writable image with an
//! undefined level (as DXC writes it) or, with the setting, with level 0.
//!
//! The translation tests run on any host. The GPU test (Windows) asks every
//! hardware adapter for the sizes both ways and prints what it answers; it
//! fails only where level 0 does not give the view's size, which is what the
//! setting relies on.

use super::lower::{self, Mac};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `v_mov_b32 v<dst>, <src>` (an SGPR's number; 128 + k: the constant k).
const fn v_mov(dst: u8, src: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | src
}

/// MIMG `<op> v<vdata>.., v<vaddr>.., s[8:15] dmask:<dmask> dim:2D unorm`.
const fn image(op: u32, dmask: u32, vdata: u8, vaddr: u8) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | 1 << 12 | dmask << 8 | 1 << 3,
        vaddr as u32 | (vdata as u32) << 8 | 2 << 16,
    ]
}

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// Per thread, four dwords at 32 times its number: the width and height
/// `image_get_resinfo` gives for the image at table dword 0 (at level 0),
/// then dwords 1 and 2 of its descriptor read as numbers (width - 1 in
/// dword 1 bits 30-31 and dword 2 bits 0-13, height - 1 in dword 2 bits
/// 14-29), or without `numbers` two zeros. With `writes` the program stores
/// to the image (texel 0, 0), else it loads from it.
fn size_queries(writes: bool, numbers: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
        0xF408_0A02,
        0xFA00_0020, // s_load_dwordx4 s[40:43], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        v_mov(1, 128),
        v_mov(2, 128),
    ];
    w.extend(image(0x0E, 3, 4, 1)); // image_get_resinfo v[4:5], v1
    if writes {
        w.extend((8..12).map(|r| v_mov(r, 128)));
        w.extend(image(0x08, 0xF, 8, 1)); // image_store v[8:11], v[1:2]
    } else {
        w.extend(image(0x00, 0xF, 8, 1)); // image_load v[8:11], v[1:2]
    }
    if numbers {
        w.extend([v_mov(6, 9), v_mov(7, 10)]);
    } else {
        w.extend([v_mov(6, 128), v_mov(7, 128)]);
    }
    for (k, r) in (4..8).enumerate() {
        w.extend(store(r, 4 * k as u32));
    }
    w.push(S_ENDPGM);
    w
}

/// The width and height in a descriptor's dwords 1 and 2.
#[cfg(windows)]
fn descriptor_size(d1: u32, d2: u32) -> (u32, u32) {
    ((d1 >> 30 | (d2 & 0x3FFF) << 2) + 1, (d2 >> 14 & 0xFFFF) + 1)
}

/// One table in `s0`: the image at dword 0, the raw buffer the threads
/// store to at dword 8. A wave32 program of 32 threads.
fn native(writes: bool, numbers: bool) -> NativeProgram {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let image = if writes {
        ResourceType::Uav
    } else {
        ResourceType::Srv
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![range(image, 0, 8), range(ResourceType::Uav, 8, 4)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: size_queries(writes, numbers)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    }
}

fn lowered(writes: bool, level0: bool) -> lower::Lowered {
    lowered_with(writes, true, level0)
}

fn lowered_with(writes: bool, numbers: bool, level0: bool) -> lower::Lowered {
    let native = native(writes, numbers);
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
        halves_local: Default::default(),
        mac: Mac {
            uav_level0: level0,
            ..Mac::default()
        },
        host_wave32: false,
    })
    .expect("lowered")
}

/// The setting changes the translation of a program that asks a writable
/// image for its size, and of no other: a read-only image is asked for
/// level 0 either way.
#[test]
fn the_setting_changes_only_queries_on_writable_images() {
    assert_ne!(lowered(true, false).dxil, lowered(true, true).dxil);
    assert_eq!(lowered(false, false).dxil, lowered(false, true).dxil);
}

/// What `translate --list` reports: a query per path for a writable image
/// whose size the program uses, with the setting or without; none for a
/// read-only image; and none from the descriptor of a writable image that
/// the program only writes through (its dwords are loaded all the same).
#[test]
fn size_queries_on_writable_images_are_counted_where_the_size_is_used() {
    let _ = lower::take_uav_size_queries();
    for level0 in [false, true] {
        lowered(true, level0);
        assert_eq!(lower::take_uav_size_queries(), [1, 1]);
    }
    lowered(false, false);
    assert_eq!(lower::take_uav_size_queries(), [0, 0]);
    lowered_with(true, false, false);
    assert_eq!(lower::take_uav_size_queries(), [0, 1]);
}

#[cfg(windows)]
mod gpu {
    use windows::Win32::Graphics::Dxgi::Common::*;

    use super::super::host::{Dispatch, Host, Root, View};
    use super::*;

    /// What thread 0 stored, and whether every thread stored the same.
    fn run(host: &Host, label: &str, l: lower::Lowered, image: View) -> ([u32; 4], bool) {
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
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            panic!("{label}: no pipeline");
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[image, View::RawUav(32 * 8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // The raw buffer is the last view read back (a zeroed texture
        // view is read back before it).
        let stored = out.last().expect("the raw buffer");
        let first: [u32; 4] = stored[..4].try_into().unwrap();
        (first, stored.chunks(8).all(|t| t[..4] == first))
    }

    /// What each hardware adapter answers a size query on a writable 2D
    /// image, with an undefined level and with level 0, by both of the
    /// translator's paths (`image_get_resinfo`, descriptor dwords): views
    /// of one-level textures in four formats and two sizes (the second the
    /// title's reduced shadow atlas), a view of level 1 of three (D3D12:
    /// the size of the view's level), and a view without a description.
    /// A read-only image is the control. Printed for `notes/uav-size.md`;
    /// the test fails where level 0 does not give the view's size.
    #[test]
    fn level_0_gives_a_writable_images_size_on_every_gpu() {
        let sized = |format, size, levels, uav| View::Sized {
            format,
            size,
            levels,
            uav,
        };
        let one = |format, size| sized(format, size, 1, Some(0));
        // (what, the view, writable, the size D3D12 defines)
        let cases = [
            ("R16_UNORM 48x20", one(DXGI_FORMAT_R16_UNORM, (48, 20)), true, (48, 20)),
            ("R8_UNORM 48x20", one(DXGI_FORMAT_R8_UNORM, (48, 20)), true, (48, 20)),
            ("R16G16_UNORM 48x20", one(DXGI_FORMAT_R16G16_UNORM, (48, 20)), true, (48, 20)),
            ("R32_FLOAT 48x20", one(DXGI_FORMAT_R32_FLOAT, (48, 20)), true, (48, 20)),
            ("R16_UNORM 512x2048", one(DXGI_FORMAT_R16_UNORM, (512, 2048)), true, (512, 2048)),
            ("R8_UNORM 2560x1440", one(DXGI_FORMAT_R8_UNORM, (2560, 1440)), true, (2560, 1440)),
            (
                "R16_UNORM 64x32, level 1 of 3",
                sized(DXGI_FORMAT_R16_UNORM, (64, 32), 3, Some(1)),
                true,
                (32, 16),
            ),
            (
                "R8_UNORM 64x32, level 2 of 3",
                sized(DXGI_FORMAT_R8_UNORM, (64, 32), 3, Some(2)),
                true,
                (16, 8),
            ),
            ("RGBA32F 48x20, no description", View::TextureUav(48, 20), true, (48, 20)),
            (
                "read-only R16_UNORM 48x20",
                sized(DXGI_FORMAT_R16_UNORM, (48, 20), 1, None),
                false,
                (48, 20),
            ),
            (
                "read-only R8_UNORM 64x32, 3 levels",
                sized(DXGI_FORMAT_R8_UNORM, (64, 32), 3, None),
                false,
                (64, 32),
            ),
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
            for (what, view, writes, want) in &cases {
                for level0 in [false, true] {
                    // A read-only image is asked for level 0 either way.
                    if !writes && level0 {
                        continue;
                    }
                    let level = if *writes && !level0 { "undefined level" } else { "level 0" };
                    let label = format!("{adapter}: {what}, {level}");
                    let view = match *view {
                        View::Sized { format, size, levels, uav } => {
                            View::Sized { format, size, levels, uav }
                        }
                        View::TextureUav(w, h) => View::TextureUav(w, h),
                        _ => unreachable!(),
                    };
                    let (got, same) = run(host, &label, lowered(*writes, level0), view);
                    let query = (got[0], got[1]);
                    let descriptor = descriptor_size(got[2], got[3]);
                    eprintln!(
                        "{label}: query {} x {}, descriptor {} x {} (dwords {:#x}, {:#x}){}",
                        query.0,
                        query.1,
                        descriptor.0,
                        descriptor.1,
                        got[2],
                        got[3],
                        if same { "" } else { "; threads differ" },
                    );
                    if level == "level 0" && (query != *want || descriptor != *want || !same) {
                        failed.push(format!(
                            "{label}: query {query:?}, descriptor {descriptor:?}, D3D12 {want:?}"
                        ));
                    }
                }
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
