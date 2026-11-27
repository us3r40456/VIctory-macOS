//! GPU compute texture detiling (`VICTORY_GPU_DETILE`, ported from
//! `fxceless/victory-fork` 3461f52 with `dxc-d3dmetal` SM 6.0 precompilation).
//!
//! Without it, every changed title texture is detiled element-run by
//! element-run on the CPU (under Rosetta 2 on macOS). With it, the CPU does a
//! single contiguous copy of the tiled bytes + pattern tables into the upload
//! staging buffer, and a 64-thread compute pass on the upload command list
//! detiles only the changed subresources directly into a linear UAV buffer
//! before `CopyTextureRegion`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;

/// `VICTORY_GPU_DETILE`: 0 (unset) off; 1 on; `alt` twenty seconds on,
/// twenty seconds off for live A/B comparison.
pub(crate) fn on() -> bool {
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_GPU_DETILE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                tracing::info!(
                    "title textures uploaded now are detiled {} (VICTORY_GPU_DETILE=alt)",
                    if on { "ON THE GPU" } else { "on the CPU" }
                );
            }
            on
        }
    }
}

/// Number of 32-bit root constants passed per subresource dispatch.
pub(crate) const CONSTANTS: usize = 22;

#[cfg(not(victory_dxc_detile))]
const SOURCE: &str = include_str!("gpudetile.hlsl");

#[cfg(victory_dxc_detile)]
const PRECOMPILED_DXIL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpudetile.dxil"));

/// Compiled GPU detiler root signature and compute pipeline.
pub(crate) struct Detiler {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) pso: ID3D12PipelineState,
}

// SAFETY: immutable D3D12 COM objects are free-threaded.
unsafe impl Send for Detiler {}
// SAFETY: as above.
unsafe impl Sync for Detiler {}

fn make_from_bytecode(device: &ID3D12Device, bytes: &[u8]) -> windows::core::Result<Detiler> {
    // SAFETY: `bytes` is a valid DXIL/DXBC container carrying an embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        Ok(Detiler { root, pso })
    }
}

/// Builds the detiler pipeline on `device`, using the precompiled SM 6.0 DXIL
/// when built with `VICTORY_HELPER_DXC` (`dxc-d3dmetal`) or `D3DCompile`
/// (`cs_5_1`) otherwise.
pub(crate) fn make(device: &ID3D12Device) -> windows::core::Result<Detiler> {
    #[cfg(victory_dxc_detile)]
    {
        make_from_bytecode(device, PRECOMPILED_DXIL)
    }
    #[cfg(not(victory_dxc_detile))]
    {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source and entry strings live through the call; output pointers are local.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                windows::core::s!("texture detile"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            if let Some(errors) = errors {
                // SAFETY: blob owns its buffer bytes.
                let text = unsafe {
                    std::slice::from_raw_parts(
                        errors.GetBufferPointer().cast::<u8>(),
                        errors.GetBufferSize(),
                    )
                };
                tracing::warn!("GPU texture detiler compile failed: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: blob owns its buffer bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            )
        };
        make_from_bytecode(device, bytes)
    }
}

/// Lazily initializes the host GPU detiler once; returns `None` if unavailable
/// (falling back seamlessly to CPU detiling).
pub(crate) fn detiler() -> Option<&'static Detiler> {
    static MADE: OnceLock<Option<Detiler>> = OnceLock::new();
    MADE.get_or_init(|| {
        let host = crate::host::host()?;
        match make(&host.device) {
            Ok(d) => {
                tracing::info!(
                    precompiled = cfg!(victory_dxc_detile),
                    "GPU texture detiling active (VICTORY_GPU_DETILE)"
                );
                Some(d)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "VICTORY_GPU_DETILE: detile compute pipeline unavailable; falling back to CPU detiling"
                );
                None
            }
        }
    })
    .as_ref()
}

/// Offsets inside the raw staging slice: tiled bytes at 0, held-levels mask,
/// pattern tables, and total required bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Raw {
    pub(crate) mask_at: u64,
    pub(crate) tables_at: u64,
    pub(crate) bytes: u64,
}

pub(crate) fn raw_layout(layout: &Layout) -> Raw {
    let mask_at = layout.size.next_multiple_of(16);
    let tables_at = mask_at + 16;
    let words: usize = layout.pattern_tables().iter().map(|t| t.len()).sum();
    Raw {
        mask_at,
        tables_at,
        bytes: tables_at + 4 * words as u64,
    }
}

/// Writes the held-levels mask and pattern tables into the mapped staging
/// slice at `data`.
///
/// # Safety
/// `data` must point to at least `raw.bytes` writable bytes owned exclusively
/// for this upload.
pub(crate) unsafe fn write_metadata(layout: &Layout, raw: &Raw, data: *mut u8, held: u32) {
    let mut at = raw.tables_at as usize;
    // SAFETY: offsets lie strictly within `raw.bytes`.
    unsafe {
        data.add(raw.mask_at as usize)
            .cast::<u32>()
            .write_unaligned(held);
        for table in layout.pattern_tables() {
            for &v in table {
                data.add(at).cast::<u32>().write_unaligned(v);
                at += 4;
            }
        }
    }
}

/// Dispatch constants and thread-group counts for one subresource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Job {
    pub(crate) constants: [u32; CONSTANTS],
    pub(crate) groups: [u32; 3],
}

/// Precomputes the dispatch parameters for every subresource of `layout`, or
/// returns `None` if any subresource must stay on the CPU detile path (e.g.
/// pipe/bank XOR swizzles or >4 GiB linear offsets).
pub(crate) fn jobs(
    layout: &Layout,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    mips: u32,
    raw: &Raw,
) -> Option<Vec<Job>> {
    let tables_at = u32::try_from(raw.tables_at).ok()?;
    let mask_at = u32::try_from(raw.mask_at).ok()?;
    let mut out = Vec::with_capacity(footprints.len());
    for (i, (footprint, &row_count)) in footprints.iter().zip(rows).enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let p = layout.detile_params(mip, slice)?;
        let f = footprint.Footprint;
        let bytes = u64::from(p.width) * u64::from(p.element_bytes);
        let depth_pitch = u64::from(f.RowPitch) * u64::from(row_count);
        let end = footprint.Offset + depth_pitch * u64::from(f.Depth);
        if bytes > u64::from(f.RowPitch)
            || p.height > row_count
            || p.depth > f.Depth
            || end > u64::from(u32::MAX)
            || p.height > 65535
            || p.depth > 65535
        {
            return None;
        }
        let words = bytes.div_ceil(4) as u32;
        out.push(Job {
            constants: [
                u32::from(p.linear),
                p.element_bytes,
                p.width,
                p.height,
                p.depth,
                p.first,
                p.slab,
                p.slice_size,
                p.mip_offset,
                p.pitch,
                p.block_w,
                p.block_h,
                p.block_log2,
                p.tail_x,
                p.tail_y,
                tables_at,
                footprint.Offset as u32,
                f.RowPitch,
                depth_pitch as u32,
                words,
                mask_at,
                mip,
            ],
            groups: [words.div_ceil(64), p.height, p.depth],
        });
    }
    Some(out)
}

/// Records GPU compute detiling of `subresources` from `raw + raw_offset` into
/// `linear` (`COMMON` -> `UAV` -> dispatches -> `COPY_SOURCE`).
pub(crate) fn record_subset(
    list: &ID3D12GraphicsCommandList,
    d: &Detiler,
    raw: &ID3D12Resource,
    raw_offset: u64,
    linear: &ID3D12Resource,
    all_jobs: &[Job],
    subresources: &[u32],
) {
    use crate::host::transition;
    // SAFETY: open command list and live buffers kept alive by the caller.
    unsafe {
        list.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        list.SetComputeRootSignature(&d.root);
        list.SetPipelineState(&d.pso);
        list.SetComputeRootShaderResourceView(1, raw.GetGPUVirtualAddress() + raw_offset);
        list.SetComputeRootUnorderedAccessView(2, linear.GetGPUVirtualAddress());
        for &sub in subresources {
            let job = &all_jobs[sub as usize];
            list.SetComputeRoot32BitConstants(0, CONSTANTS as u32, job.constants.as_ptr().cast(), 0);
            list.Dispatch(job.groups[0], job.groups[1], job.groups[2]);
        }
        list.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(subresources.len() as u64, Relaxed);
}

/// Counters: [textures_detiled, dispatches, copied_bytes, copy_us, made, refused].
pub(crate) static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Swizzle};
    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    fn make_buffer(
        device: &ID3D12Device,
        heap: D3D12_HEAP_TYPE,
        bytes: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes.max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: valid D3D12 buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    /// Runs the GPU detile compute shader on the host D3D12 device (D3DMetal
    /// under Wine on macOS) and asserts byte-for-byte equality against CPU
    /// `Layout::detile` across linear, 256B, 4KB, and 64KB swizzles + mips.
    #[test]
    fn gpu_detile_matches_cpu_byte_for_byte() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let detiler = make(&h.device).expect("compile gpudetile shader");
        let queue: ID3D12CommandQueue = unsafe {
            h.device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap()
        };
        let allocator: ID3D12CommandAllocator = unsafe {
            h.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap()
        };
        let list: ID3D12GraphicsCommandList = unsafe {
            h.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap()
        };
        let fence: ID3D12Fence = unsafe { h.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
        let event: HANDLE = unsafe { CreateEventW(None, false, false, None).unwrap() };
        let mut submitted = 0u64;

        for swizzle in [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ] {
            let desc = Desc {
                dimension: Dimension::Tex2d,
                width: 97,
                height: 61,
                depth: 1,
                mips: 5,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle,
            };
            let layout = Layout::new(&desc).unwrap();
            let tiled: Vec<u8> = (0..layout.size as usize)
                .map(|i| ((i.wrapping_mul(37) ^ (i >> 7)) & 0xff) as u8)
                .collect();

            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(desc.width),
                Height: desc.height,
                DepthOrArraySize: 1,
                MipLevels: desc.mips as u16,
                Format: DXGI_FORMAT(DXGI_FORMAT_R8G8B8A8_UNORM.0),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let count = desc.mips as usize;
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count];
            let mut rows = vec![0u32; count];
            let mut total = 0u64;
            // SAFETY: valid descriptor and output slices.
            unsafe {
                h.device.GetCopyableFootprints(
                    &tex_desc,
                    0,
                    count as u32,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    None,
                    Some(&mut total),
                );
            }

            let raw = raw_layout(&layout);
            let all_jobs = jobs(&layout, &footprints, &rows, desc.mips, &raw).expect("gpu jobs");
            let upload = make_buffer(
                &h.device,
                D3D12_HEAP_TYPE_UPLOAD,
                raw.bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let linear = make_buffer(
                &h.device,
                D3D12_HEAP_TYPE_DEFAULT,
                total.next_multiple_of(4),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let readback = make_buffer(
                &h.device,
                D3D12_HEAP_TYPE_READBACK,
                total.next_multiple_of(4),
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            // Upload raw tiled bytes + tables.
            let mut mapped = std::ptr::null_mut();
            unsafe {
                upload.Map(0, None, Some(&mut mapped)).unwrap();
                std::ptr::copy_nonoverlapping(tiled.as_ptr(), mapped.cast::<u8>(), tiled.len());
                write_metadata(&layout, &raw, mapped.cast::<u8>(), u32::MAX);
                upload.Unmap(0, None);
            }

            let subs: Vec<u32> = (0..count as u32).collect();
            record_subset(&list, &detiler, &upload, 0, &linear, &all_jobs, &subs);
            unsafe {
                list.CopyBufferRegion(&readback, 0, &linear, 0, total);
                list.Close().unwrap();
                let cmd: ID3D12CommandList = windows::core::Interface::cast(&list).unwrap();
                queue.ExecuteCommandLists(&[Some(cmd)]);
                submitted += 1;
                queue.Signal(&fence, submitted).unwrap();
                fence.SetEventOnCompletion(submitted, event).unwrap();
                assert_eq!(WaitForSingleObject(event, 30_000), WAIT_OBJECT_0);
                allocator.Reset().unwrap();
                list.Reset(&allocator, None).unwrap();
            }

            // Compare every texel against CPU Layout::detile.
            let mut out_ptr = std::ptr::null_mut();
            unsafe { readback.Map(0, None, Some(&mut out_ptr)).unwrap() };
            let gpu_bytes = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u8>(), total as usize) };
            for mip in 0..desc.mips {
                let fp = &footprints[mip as usize];
                let pitch = fp.Footprint.RowPitch as usize;
                let r = rows[mip as usize] as usize;
                let mut cpu = vec![0u8; pitch * r];
                layout.detile(&tiled, mip, 0, &mut cpu, pitch, pitch * r).unwrap();
                let row_bytes = layout.mips[mip as usize].width as usize * desc.element_bytes as usize;
                for y in 0..layout.mips[mip as usize].height as usize {
                    let off = fp.Offset as usize + y * pitch;
                    assert_eq!(
                        &gpu_bytes[off..off + row_bytes],
                        &cpu[y * pitch..y * pitch + row_bytes],
                        "mismatch for swizzle {:?} mip {mip} row {y}",
                        swizzle
                    );
                }
            }
            unsafe { readback.Unmap(0, None) };
        }
        unsafe {
            let _ = CloseHandle(event);
        }
    }
}
