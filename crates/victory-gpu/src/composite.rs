//! Draws a presented overlay plane over the picture plane (#15).
//!
//! The title can present a scene and a UI plane at different resolutions.
//! VIctory blends the other plane over the selected picture when
//! `VICTORY_PRESENT_COMPOSITE=1`. The overlay uses straight alpha and
//! nearest-texel sampling in picture coordinates. `VICTORY_PRESENT_MAX`
//! fits the final picture into the requested size, with or without an overlay.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Result, s};

/// Whether `VICTORY_PRESENT_COMPOSITE=1` asks for the pass.
pub(crate) fn wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PRESENT_COMPOSITE").is_ok_and(|v| v == "1"))
}

/// Fit the final presentation texture inside a requested host size, keeping
/// its aspect ratio. Guest render targets and upscaler inputs keep their sizes.
fn fit_output(width: u32, height: u32, limit: (u32, u32)) -> (u32, u32) {
    if width <= limit.0 && height <= limit.1 {
        return (width, height);
    }
    let (w, h) = (u64::from(width), u64::from(height));
    let (mw, mh) = (u64::from(limit.0), u64::from(limit.1));
    if mw * h <= mh * w {
        (limit.0, (h * mw / w).max(1) as u32)
    } else {
        ((w * mh / h).max(1) as u32, limit.1)
    }
}

pub(crate) fn output_size(width: u32, height: u32) -> (u32, u32) {
    static LIMIT: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
    let limit = LIMIT.get_or_init(|| {
        let value = std::env::var("VICTORY_PRESENT_MAX").ok()?;
        let size = value.trim().split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .filter(|&(w, h)| w > 0 && h > 0);
        if size.is_none() {
            tracing::warn!(%value, "invalid VICTORY_PRESENT_MAX (expected positive WIDTHxHEIGHT)");
        }
        size
    });
    limit.map_or((width, height), |limit| fit_output(width, height, limit))
}

const SOURCE: &str = r#"
cbuffer C : register(b0) { uint width; uint height; uint overWidth; uint overHeight; uint outWidth; uint outHeight; };
Texture2D<float4> picture : register(t0);
Texture2D<float4> overlay : register(t1);
RWTexture2D<float4> result : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= outWidth || id.y >= outHeight) return;
    uint2 pixel = id.xy;
    if (width != outWidth || height != outHeight)
        pixel = (2 * id.xy + 1) * uint2(width, height) / (2 * uint2(outWidth, outHeight));
    float4 p = picture.Load(int3(pixel, 0));
    if (overWidth == 0) {
        result[id.xy] = p;
        return;
    }
    float4 o = overlay.Load(int3(pixel.x * overWidth / width, pixel.y * overHeight / height, 0));
    result[id.xy] = float4(lerp(p.rgb, o.rgb, o.a), 1.0);
}
"#;

/// The pass's pipeline and its shader-visible heap of three views
/// (picture, overlay, result). Presents run one at a time, under the
/// presenter's lock, and each waits for its list, so one heap serves all.
pub(crate) struct Compositor {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: u32,
}

impl Compositor {
    pub(crate) fn new(device: &ID3D12Device) -> Result<Self> {
        use crate::compute::{Param, srv, uav};
        let ranges = [srv(0, 2), uav(0, 1)];
        let params = [
            Param::Table(&ranges),
            Param::Constants {
                register: 0,
                count: 6,
            },
        ];
        let (root, pso) = crate::compute::pipeline(SOURCE, s!("composite"), 0, &params)?;
        // SAFETY: plain object creation.
        unsafe {
            let heap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })?;
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Self {
                root,
                pso,
                heap,
                step,
            })
        }
    }

    /// Records the pass. The picture and optional overlay must be in a non-pixel
    /// shader resource state and the output-sized `result` (in `format`) in
    /// the unordered access state; they stay so.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        picture: (&ID3D12Resource, DXGI_FORMAT),
        overlay: Option<(&ID3D12Resource, DXGI_FORMAT)>,
        result: &ID3D12Resource,
        format: DXGI_FORMAT,
    ) {
        // SAFETY: live resources and our heap of three views.
        unsafe {
            let at = self.heap.GetCPUDescriptorHandleForHeapStart();
            let slot = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: at.ptr + (i * self.step) as usize,
            };
            for (i, (texture, format)) in [picture, overlay.unwrap_or(picture)].into_iter().enumerate() {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                device.CreateShaderResourceView(texture, Some(&view), slot(i as u32));
            }
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            device.CreateUnorderedAccessView(result, None, Some(&view), slot(2));
            let p = picture.0.GetDesc();
            let out = result.GetDesc();
            let (ow, oh) = overlay.map_or((0, 0), |(texture, _)| {
                let d = texture.GetDesc();
                (d.Width as u32, d.Height)
            });
            let constants = [p.Width as u32, p.Height, ow, oh, out.Width as u32, out.Height];
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootDescriptorTable(0, self.heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRoot32BitConstants(1, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.Dispatch((out.Width as u32).div_ceil(8), out.Height.div_ceil(8), 1);
        }
    }
}

/// An output-sized texture the pass writes, in `format`.
pub(crate) fn result_texture(
    device: &ID3D12Device,
    width: u64,
    height: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource> {
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut texture,
        )?;
    }
    texture.ok_or_else(windows::core::Error::empty)
}

/// The format the pass writes and the swap chain shows for a picture whose
/// swap format is `format`: typed stores to BGRA are optional in D3D12, so
/// BGRA becomes RGBA (the shader reads and writes colors by name, so the
/// picture keeps its colors).
pub(crate) fn result_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    if format == DXGI_FORMAT_B8G8R8A8_UNORM {
        DXGI_FORMAT_R8G8B8A8_UNORM
    } else {
        format
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::transition;

    /// A buffer of `len` bytes in `heap`, in `state`.
    fn buffer(heap: D3D12_HEAP_TYPE, len: u64, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let device = &crate::host::host().unwrap().device;
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: len,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut buffer,
                )
                .unwrap();
        }
        buffer.unwrap()
    }

    /// A copy location for `texture`'s first subresource.
    fn whole(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        }
    }

    /// A copy location for 8-bit RGBA rows at a 256-byte pitch in `buffer`.
    fn rows(buffer: &ID3D12Resource, width: u32, height: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: (width * 4).next_multiple_of(256),
                    },
                },
            },
        }
    }

    /// An 8-bit RGBA texture of `width` x `height` holding `texels`, left
    /// in the non-pixel shader resource state; its upload buffer is kept.
    fn texture(
        width: u32,
        height: u32,
        texels: &[[u8; 4]],
        list: &ID3D12GraphicsCommandList,
        keep: &mut Vec<ID3D12Resource>,
    ) -> ID3D12Resource {
        let device = &crate::host::host().unwrap().device;
        let texture =
            result_texture(device, width.into(), height, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let upload = buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            u64::from((width * 4).next_multiple_of(256)) * u64::from(height),
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a mapped upload buffer written in bounds; an open list and
        // copy locations naming live resources.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            for y in 0..height {
                for x in 0..width {
                    let at = data.cast::<u8>().add((y * (width * 4).next_multiple_of(256) + x * 4) as usize);
                    std::ptr::copy_nonoverlapping(texels[(y * width + x) as usize].as_ptr(), at, 4);
                }
            }
            list.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.CopyTextureRegion(
                &whole(&texture),
                0,
                0,
                0,
                &rows(&upload, width, height),
                None,
            );
            list.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
        keep.push(upload);
        texture
    }

    #[test]
    fn the_presentation_cap_preserves_aspect_and_never_enlarges() {
        let cap = (1280, 720);
        assert_eq!(fit_output(1920, 1080, cap), cap);
        assert_eq!(fit_output(3840, 2160, cap), cap);
        assert_eq!(fit_output(1024, 768, cap), (960, 720));
        assert_eq!(fit_output(2560, 1080, cap), (1280, 540));
        assert_eq!(fit_output(640, 360, cap), (640, 360));
        assert_eq!(fit_output(1, 1440, cap), (1, 720));
    }

    #[test]
    fn the_720p_output_samples_the_whole_picture_and_overlay() {
        let Some(host) = crate::host::host() else { return };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let compositor = Compositor::new(&host.device).unwrap();
        let (w, h) = fit_output(1920, 1080, (1280, 720));
        let stride = (w * 4).next_multiple_of(256);
        let picture_pixel = |x: u32, y: u32| [x as u8, y as u8, 37, 77];
        let overlay_pixel = |x: u32, y: u32| [21, x as u8, y as u8, if (x + y).is_multiple_of(3) { 255 } else { 0 }];
        let upload = host.new_list().unwrap();
        let mut keep = Vec::new();
        let picture = texture(1920, 1080, &(0..1920 * 1080).map(|n| picture_pixel(n % 1920, n / 1920)).collect::<Vec<_>>(), &upload.list, &mut keep);
        let overlay = texture(3840, 2160, &(0..3840 * 2160).map(|n| overlay_pixel(n % 3840, n / 3840)).collect::<Vec<_>>(), &upload.list, &mut keep);
        upload.close();
        host.execute(&[&upload]);
        for blend in [false, true] {
            let run = host.new_list().unwrap();
            let result = result_texture(&host.device, u64::from(w), h, format).unwrap();
            compositor.record(&host.device, &run.list, (&picture, format),
                blend.then_some((&overlay, format)), &result, format);
            let back = buffer(D3D12_HEAP_TYPE_READBACK, u64::from(stride) * u64::from(h), D3D12_RESOURCE_STATE_COPY_DEST);
            // SAFETY: live resources and a destination covering the full output.
            unsafe {
                run.list.ResourceBarrier(&[transition(&result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                run.list.CopyTextureRegion(&rows(&back, w, h), 0, 0, 0, &whole(&result), None);
            }
            run.close();
            host.execute(&[&run]);
            let mut data = std::ptr::null_mut();
            // SAFETY: the copy finished and the buffer stays mapped through the checks.
            unsafe { back.Map(0, None, Some(&mut data)) }.unwrap();
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = ((2 * x + 1) * 1920 / (2 * w), (2 * y + 1) * 1080 / (2 * h));
                    let mut expected = picture_pixel(sx, sy);
                    if blend {
                        let top = overlay_pixel(sx * 2, sy * 2);
                        if top[3] == 255 { expected = top; }
                        expected[3] = 255;
                    }
                    // SAFETY: x and y are inside the mapped output footprint.
                    let actual = unsafe { *data.cast::<[u8; 4]>().add((y * stride / 4 + x) as usize) };
                    assert_eq!(actual, expected, "pixel {x},{y}, overlay {blend}");
                }
            }
            // SAFETY: this test owns the finished readback.
            unsafe { back.Unmap(0, None) };
        }
    }

    /// A 4x4 red picture under a 2x2 overlay whose top-left texel is opaque
    /// green, top-right half-transparent blue and the rest clear: each
    /// overlay texel covers a 2x2 block of the picture.
    #[test]
    fn the_overlay_is_blended_over_the_picture() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let compositor = Compositor::new(device).expect("the pass compiles");
        let run = host.new_list().unwrap();
        let list = &run.list;
        let mut keep = Vec::new();
        let picture = texture(4, 4, &[[255, 0, 0, 255]; 16], list, &mut keep);
        let clear = [9, 9, 9, 0];
        let overlay = texture(
            2,
            2,
            &[[0, 255, 0, 255], [0, 0, 255, 128], clear, clear],
            list,
            &mut keep,
        );
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let result = result_texture(device, 4, 4, format).unwrap();
        compositor.record(
            device,
            list,
            (&picture, format),
            Some((&overlay, format)),
            &result,
            format,
        );
        let back = buffer(
            D3D12_HEAP_TYPE_READBACK,
            256 * 4,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: an open list and live resources.
        unsafe {
            list.ResourceBarrier(&[transition(
                &result,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(&rows(&back, 4, 4), 0, 0, 0, &whole(&result), None);
        }
        run.close();
        host.execute(&[&run]);
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a finished readback buffer.
        unsafe { back.Map(0, None, Some(&mut data)) }.unwrap();
        // SAFETY: inside the mapped buffer: 4 rows at a 256-byte pitch.
        let texel = |x: usize, y: usize| unsafe { *data.cast::<[u8; 4]>().add(y * 64 + x) };
        for (x, y) in [(0, 0), (1, 1)] {
            assert_eq!(
                texel(x, y),
                [0, 255, 0, 255],
                "opaque overlay at ({x}, {y})"
            );
        }
        let half = texel(3, 0);
        assert!(
            half[0].abs_diff(127) <= 1 && half[1] == 0 && half[2].abs_diff(128) <= 1,
            "half blue over red: {half:?}"
        );
        for (x, y) in [(0, 2), (3, 3)] {
            assert_eq!(texel(x, y), [255, 0, 0, 255], "clear overlay at ({x}, {y})");
        }
    }
}
