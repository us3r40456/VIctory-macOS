//! A made-up frame run through the host layer, for the tests of the frame
//! recorder (`tests/replay.rs`): the host's device, lists and submissions
//! (`Host::execute`, with its waits and upkeep), its buffers the CPU maps
//! and its view slots, as a title's frame uses them, in a small program
//! whose picture depends on all of it:
//!
//! - two textures filled once, long before the frame that is captured (one
//!   made with other formats to view it in, one placed in a heap, with a
//!   second level);
//! - a colour and a depth target, cleared and drawn to in three submissions
//!   of four lists, each draw behind or in front of the others: one plain,
//!   one with indices, one with its arguments read from a buffer;
//! - two rectangles of the colour target cleared through an unordered-access
//!   view of it, one in the first submission and one in the last (as the
//!   host layer clears for a title's `ClearUnorderedAccessView`);
//! - four more draws by arguments the host layer clamps
//!   (`crate::indirect`): three in the list before theirs, two of those in
//!   one batch and one alone (as it records the clamps of a title's runs of
//!   indirect draws with `VICTORY_BATCH_CLAMPS=1`), and one in place;
//! - a buffer the CPU writes anew before every submission, read by a
//!   compute pass that writes another, which the draws read;
//! - a view slot whose view is replaced between two submissions, and one
//!   that holds no view until one is made in it between two others;
//! - the picture copied to a buffer at the frame's end;
//! - textures no command of the frame names, of the formats and shapes a
//!   title's are (compressed, float, packed, depth with stencil, levels,
//!   layers, volumes), filled once: what a capture must take of the GPU's
//!   memory, and a replay put back, whatever the host makes of each.
//!
//! Every frame is the same but for the vertices of the draws by arguments
//! (three in every fifth frame, six in the others), so what a frame leaves
//! is what a replay of one like it must leave, and arguments left from the
//! frame before are not the frame's own.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR, s};

use crate::compute::{Param, srv};
use crate::host::{HostList, host, transition};

/// The picture's width and height.
pub const SIZE: u32 = 128;
/// The textures' width and height.
const SMALL: u32 = 16;
/// Words of the buffers.
const WORDS: usize = 16;

/// The names of the resources a test asks a replay for.
pub const TARGET: &str = "replay test: picture";
pub const RESULTS: &str = "replay test: results";

const DRAW: &str = r#"
static const float2 CORNERS[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
cbuffer C : register(b0) { float4 place; float4 tint; float depth; };
Texture2D<float4> picture : register(t0);
StructuredBuffer<uint> shades : register(t1);
float4 vs(uint v : SV_VertexID) : SV_Position {
    float2 c = CORNERS[v % 6];
    return float4(place.xy + c * place.zw, depth, 1.0);
}
float4 ps(float4 at : SV_Position) : SV_Target {
    uint s = shades[(uint(at.x) / 8) % 16];
    float4 shade = float4((s & 255) / 255.0, ((s >> 8) & 255) / 255.0, 0.0, 0.0);
    return picture.Load(int3(int2(at.xy) % 16, 0)) * tint + shade * 0.25;
}
"#;

const COMPUTE: &str = r#"
cbuffer C : register(b0) { uint seed; };
StructuredBuffer<uint> source : register(t0);
RWByteAddressBuffer results : register(u0);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    results.Store(id.x * 4, source[id.x] * 3 + seed * 257 + id.x);
}
"#;

/// What the last frame left.
pub struct Made {
    /// The picture as the frame itself copied it out: rows of [`SIZE`]
    /// RGBA texels, [`SIZE`] of them.
    pub picture: Vec<u8>,
    /// The buffer the compute passes write.
    pub results: Vec<u8>,
    /// The textures of many formats: what each holds.
    pub others: Vec<Other>,
}

/// A texture no command of the frame names.
pub struct Other {
    pub name: String,
    /// By subresource: the bytes of its rows that mean something.
    pub subs: Vec<Rows>,
}

/// A subresource's rows in the layout copies use: `rows` rows `pitch`
/// bytes apart, `row` bytes of each meaning something; `bytes` holds those,
/// row after row.
pub struct Rows {
    pub pitch: usize,
    pub rows: usize,
    pub row: usize,
    pub bytes: Vec<u8>,
}

impl Rows {
    /// Whether `layout` (a subresource read back, rows `pitch` apart) holds
    /// these rows.
    pub fn are_in(&self, layout: &[u8]) -> bool {
        (0..self.rows).all(|y| {
            layout.get(y * self.pitch..y * self.pitch + self.row)
                == self.bytes.get(y * self.row..(y + 1) * self.row)
        })
    }
}

/// The formats of the textures no command names: colour ones by number
/// (float, packed, compressed), then depth ones, and what a clear of 0.75
/// and stencil 0x5a leaves is what they hold.
const COLOURS: [i32; 21] = [
    2, 10, 24, 26, 28, 29, 34, 41, 42, 49, 54, 56, 61, 67, 71, 77, 80, 83, 87, 95, 98,
];
const DEPTHS: [i32; 4] = [40, 20, 45, 55];

struct Extra {
    name: String,
    res: ID3D12Resource,
    desc: D3D12_RESOURCE_DESC,
    subs: u32,
    depth: bool,
    /// What it holds, by subresource, once filled.
    held: Vec<Rows>,
}

fn texture(
    size: u32,
    mips: u16,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    }
}

/// A texel of the made-up textures.
fn texel(which: u32, level: u32, x: u32, y: u32) -> [u8; 4] {
    let v = x * 13 + y * 29 + which * 97 + level * 53;
    [
        (v * 3) as u8,
        (v * 5 + 40) as u8,
        (v * 7 + which * 120) as u8,
        255,
    ]
}

struct Program {
    target: ID3D12Resource,
    depth: ID3D12Resource,
    first: ID3D12Resource,
    second: ID3D12Resource,
    _heap: ID3D12Heap,
    inputs: (ID3D12Resource, *mut u8),
    results: (ID3D12Resource, *mut u8),
    out: (ID3D12Resource, *mut u8),
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    views: ID3D12DescriptorHeap,
    step: usize,
    /// The colour target's unordered-access view by a CPU handle, for its
    /// clears (the same view is in slot [`CLEARED`] of `views`).
    clears: ID3D12DescriptorHeap,
    draw_root: ID3D12RootSignature,
    draw: ID3D12PipelineState,
    compute_root: ID3D12RootSignature,
    compute: ID3D12PipelineState,
    signature: ID3D12CommandSignature,
    lists: Vec<HostList>,
    extras: Vec<Extra>,
}

/// How a rectangle is drawn.
#[derive(Clone, Copy)]
enum How<'a> {
    Plain,
    /// With indices from the inputs' buffer.
    Indexed,
    /// With its arguments from the inputs' buffer.
    Indirect,
    /// With its arguments from where the host layer's clamp put them.
    Clamped(&'a ID3D12Resource, u64),
}

/// The slot of `views` with the colour target's unordered-access view.
const CLEARED: usize = 5;

/// Where the inputs' buffer holds a draw's arguments, and six indices.
const ARGUMENTS: usize = 1024;
const INDICES: usize = 2048;

fn compile(source: &str, entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names outlive the call.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("replay test"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

impl Program {
    fn new() -> windows::core::Result<Program> {
        let host = host().ok_or_else(windows::core::Error::empty)?;
        let device = &host.device;
        let default = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let committed = |desc: &D3D12_RESOURCE_DESC, state| {
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device.CreateCommittedResource(
                    &default,
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    None,
                    &mut made,
                )?
            };
            made.ok_or_else(windows::core::Error::empty)
        };
        let rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
        // A target shaders may write too: it is also cleared through an
        // unordered-access view.
        let target = committed(
            &texture(
                SIZE,
                1,
                rgba,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ),
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        crate::host::name(&target, || TARGET.into());
        let depth = committed(
            &texture(
                SIZE,
                1,
                DXGI_FORMAT_D32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ),
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        )?;
        // One with other formats to view it in, as the host layer makes the
        // title's colour textures where the host takes that.
        let small = texture(SMALL, 1, rgba, D3D12_RESOURCE_FLAG_NONE);
        let castable = device.cast::<ID3D12Device10>().ok().and_then(|device| {
            let desc1 = D3D12_RESOURCE_DESC1 {
                Dimension: small.Dimension,
                Width: small.Width,
                Height: small.Height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: rgba,
                SampleDesc: small.SampleDesc,
                ..Default::default()
            };
            let formats = [DXGI_FORMAT_R8G8B8A8_UNORM_SRGB];
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation; the formats live through the call.
            unsafe {
                device.CreateCommittedResource3(
                    &default,
                    D3D12_HEAP_FLAG_NONE,
                    &desc1,
                    D3D12_BARRIER_LAYOUT_COMMON,
                    None,
                    None::<&ID3D12ProtectedResourceSession>,
                    Some(&formats),
                    &mut made,
                )
            }
            .ok()
            .and(made)
        });
        let first = match castable {
            Some(first) => first,
            None => committed(&small, D3D12_RESOURCE_STATE_COMMON)?,
        };
        // One placed in a heap, with a second level.
        let leveled = texture(SMALL, 2, rgba, D3D12_RESOURCE_FLAG_NONE);
        let mut heap: Option<ID3D12Heap> = None;
        let mut second: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device.CreateHeap(
                &D3D12_HEAP_DESC {
                    SizeInBytes: 1 << 20,
                    Properties: default,
                    Alignment: 0,
                    Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                },
                &mut heap,
            )?;
        }
        let heap = heap.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: a range of the heap for the description.
        unsafe {
            device.CreatePlacedResource(
                &heap,
                65536,
                &leveled,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut second,
            )?;
        }
        let second = second.ok_or_else(windows::core::Error::empty)?;
        let shared = |size| {
            host.shared_buffer(size)
                .ok_or_else(windows::core::Error::empty)
        };
        let (inputs, results) = (shared(4096)?, shared(4096)?);
        crate::host::name(&results.0, || RESULTS.into());
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: outputs for one subresource.
        unsafe {
            device.GetCopyableFootprints(
                &target.GetDesc(),
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            )
        };
        let out = shared(total)?;
        // SAFETY: plain creation.
        let views: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
        };
        // SAFETY: a plain query.
        let step = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        } as usize;
        // The target's unordered-access view, by a CPU handle in a heap
        // shaders do not see and in a slot of the heap they do: a clear
        // through it takes both.
        // SAFETY: plain creation; slots of live heaps and a live texture.
        let clears: ID3D12DescriptorHeap = unsafe {
            let clears: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: rgba,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            let mut visible = views.GetCPUDescriptorHandleForHeapStart();
            visible.ptr += CLEARED * step;
            for at in [clears.GetCPUDescriptorHandleForHeapStart(), visible] {
                device.CreateUnorderedAccessView(&target, None::<&ID3D12Resource>, Some(&desc), at);
            }
            clears
        };
        let draw_params = [
            Param::Constants {
                register: 0,
                count: 9,
            },
            Param::Table(&[srv(0, 1)]),
            Param::Srv {
                register: 1,
                space: 0,
            },
        ];
        let draw_root = crate::compute::root_signature(device, &draw_params)?;
        let (vs, ps) = (
            compile(DRAW, s!("vs"), s!("vs_5_1"))?,
            compile(DRAW, s!("ps"), s!("ps_5_1"))?,
        );
        // SAFETY: the blobs' own bytes.
        let bytes = |b: &ID3DBlob| unsafe {
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            }
        };
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call.
            pRootSignature: unsafe { std::mem::transmute_copy(&draw_root) },
            VS: bytes(&vs),
            PS: bytes(&ps),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.RTVFormats[0] = rgba;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = true.into();
        desc.DepthStencilState.DepthEnable = true.into();
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        // SAFETY: a complete description.
        let draw: ID3D12PipelineState = unsafe { device.CreateGraphicsPipelineState(&desc) }?;
        crate::host::name(&draw, || "replay test: draw".into());
        let (compute_root, compute) = crate::compute::pipeline(
            COMPUTE,
            s!("replay test compute"),
            0,
            &[
                Param::Constants {
                    register: 0,
                    count: 1,
                },
                Param::Srv {
                    register: 0,
                    space: 0,
                },
                Param::Uav { register: 0 },
            ],
        )?;
        crate::host::name(&compute, || "replay test: compute".into());
        let arguments = [D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            ..Default::default()
        }];
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a description of one argument that lives through the call.
        unsafe {
            device.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 16,
                    NumArgumentDescs: 1,
                    pArgumentDescs: arguments.as_ptr(),
                    NodeMask: 0,
                },
                None::<&ID3D12RootSignature>,
                &mut signature,
            )?;
        }
        let signature = signature.ok_or_else(windows::core::Error::empty)?;
        let lists = (0..4)
            .map(|_| {
                let list = host.new_list()?;
                list.close();
                Ok(list)
            })
            .collect::<windows::core::Result<Vec<HostList>>>()?;
        // Textures no command names: every format in one shape, and one
        // format in every shape.
        let mut extras = Vec::new();
        let mut extra = |name: String, desc: D3D12_RESOURCE_DESC, depth: bool| {
            let state = if depth {
                D3D12_RESOURCE_STATE_DEPTH_WRITE
            } else {
                D3D12_RESOURCE_STATE_COMMON
            };
            // A format this host does not make is left out, and said.
            match committed(&desc, state) {
                Ok(res) => {
                    crate::host::name(&res, || name.clone());
                    let planes: u32 = if matches!(desc.Format.0, 20 | 45) { 2 } else { 1 };
                    let layers = match desc.Dimension {
                        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 1,
                        _ => u32::from(desc.DepthOrArraySize),
                    };
                    extras.push(Extra {
                        name,
                        res,
                        desc,
                        subs: u32::from(desc.MipLevels) * layers * planes,
                        depth,
                        held: Vec::new(),
                    });
                }
                Err(e) => eprintln!("no texture \"{name}\" on this host: {e}"),
            }
        };
        for format in COLOURS {
            // Every level down to one texel: the last ones of a compressed
            // format are less than a block.
            let desc = texture(64, 7, DXGI_FORMAT(format), D3D12_RESOURCE_FLAG_NONE);
            extra(format!("replay test: format {format}"), desc, false);
        }
        // Sizes that are no powers of two, nor whole blocks.
        for (name, format, width, height) in [
            ("odd blocks", DXGI_FORMAT_BC1_UNORM, 62, 30),
            ("odd blocks of 16 bytes", DXGI_FORMAT_BC7_UNORM, 150, 90),
            ("odd", rgba, 100, 50),
            ("one texel", rgba, 1, 1),
        ] {
            let mut desc = texture(width, 0, format, D3D12_RESOURCE_FLAG_NONE);
            desc.Height = height;
            desc.MipLevels = (u32::BITS - width.max(height).leading_zeros()) as u16;
            extra(format!("replay test: {name}"), desc, false);
        }
        for format in DEPTHS {
            let desc = texture(
                64,
                1,
                DXGI_FORMAT(format),
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            );
            extra(format!("replay test: depth format {format}"), desc, true);
        }
        let mut shadows = texture(
            64,
            1,
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        );
        shadows.DepthOrArraySize = 4;
        extra("replay test: depth layers".into(), shadows, true);
        let mut layers = texture(32, 2, rgba, D3D12_RESOURCE_FLAG_NONE);
        layers.DepthOrArraySize = 6;
        extra("replay test: six layers".into(), layers, false);
        let mut volume = texture(16, 2, rgba, D3D12_RESOURCE_FLAG_NONE);
        (volume.Dimension, volume.DepthOrArraySize) = (D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8);
        extra("replay test: a volume".into(), volume, false);
        let mut program = Program {
            target,
            depth,
            first,
            second,
            _heap: heap,
            inputs,
            results,
            out,
            footprint,
            views,
            step,
            clears,
            draw_root,
            draw,
            compute_root,
            compute,
            signature,
            lists,
            extras,
        };
        program.fill()?;
        program.fill_extras()?;
        Ok(program)
    }

    fn slot(&self, i: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: a plain query.
        let mut at = unsafe { self.views.GetCPUDescriptorHandleForHeapStart() };
        at.ptr += i * self.step;
        at
    }

    fn table(&self, i: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a plain query.
        let mut at = unsafe { self.views.GetGPUDescriptorHandleForHeapStart() };
        at.ptr += (i * self.step) as u64;
        at
    }

    /// Makes the view in slot `i` one of `texture`.
    fn view(&self, i: usize, texture: &ID3D12Resource) {
        let Some(host) = host() else { return };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: u32::MAX,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap and a live texture.
        unsafe {
            host.device
                .CreateShaderResourceView(texture, Some(&desc), self.slot(i))
        };
    }

    /// Makes the view in slot `i` one of nothing.
    fn no_view(&self, i: usize) {
        let Some(host) = host() else { return };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        // SAFETY: a slot of our heap; a null view has a description.
        unsafe {
            host.device.CreateShaderResourceView(
                None::<&ID3D12Resource>,
                Some(&desc),
                self.slot(i),
            )
        };
    }

    /// The textures' texels, once: by a list of the host's from one of its
    /// upload buffers.
    fn fill(&self) -> windows::core::Result<()> {
        let host = host().ok_or_else(windows::core::Error::empty)?;
        let (staging, data) = host
            .upload_buffer(1 << 16)
            .ok_or_else(windows::core::Error::empty)?;
        let list = &self.lists[0];
        list.reset();
        let mut at = 0u64;
        for (which, (texture, levels)) in [(&self.first, 1u32), (&self.second, 2)]
            .into_iter()
            .enumerate()
        {
            for level in 0..levels {
                let size = SMALL >> level;
                let pitch = (size * 4).next_multiple_of(256);
                for y in 0..size {
                    for x in 0..size {
                        let t = texel(which as u32, level, x, y);
                        // SAFETY: inside the upload buffer's mapping.
                        unsafe {
                            data.add((at + u64::from(y * pitch + x * 4)) as usize)
                                .cast::<[u8; 4]>()
                                .write(t)
                        };
                    }
                }
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: borrowed; the location never releases it.
                    pResource: unsafe { std::mem::transmute_copy(&staging) },
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: at,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: size,
                                Height: size,
                                Depth: 1,
                                RowPitch: pitch,
                            },
                        },
                    },
                };
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: as above.
                    pResource: unsafe { std::mem::transmute_copy(texture) },
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: level,
                    },
                };
                // SAFETY: an open list; live resources; the texture in the
                // common state, which a copy promotes.
                unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
                at += u64::from(pitch * size).next_multiple_of(512);
            }
        }
        // To a state shaders read them in.
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list; live textures, copy destinations by now.
        unsafe {
            list.list.ResourceBarrier(&[
                transition(&self.first, D3D12_RESOURCE_STATE_COPY_DEST, read),
                transition(&self.second, D3D12_RESOURCE_STATE_COPY_DEST, read),
            ]);
        }
        list.close();
        host.execute(&[list]);
        self.view(0, &self.first);
        self.view(1, &self.second);
        self.view(2, &self.first);
        Ok(())
    }

    /// The layout of a subresource of `desc` in a buffer: its footprint,
    /// its rows and the bytes of each that mean something.
    fn layout(desc: &D3D12_RESOURCE_DESC, sub: u32) -> (D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64, u64) {
        let mut f = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut row, mut total) = (0, 0, 0);
        if let Some(host) = host() {
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device.GetCopyableFootprints(
                    desc,
                    sub,
                    1,
                    0,
                    Some(&mut f),
                    Some(&mut rows),
                    Some(&mut row),
                    Some(&mut total),
                )
            };
        }
        (f, rows, row, total)
    }

    /// Fills the textures no command names, once: colour ones with bytes
    /// from an upload buffer, depth ones by a clear; then reads every one
    /// back: that is what they hold. (Mostly the bytes given. D3DMetal
    /// keeps the values of the two packed float formats, not their bits:
    /// what is not a number comes back as its plain form, a shared exponent
    /// as the smallest that holds the values.)
    fn fill_extras(&mut self) -> windows::core::Result<()> {
        let host = host().ok_or_else(windows::core::Error::empty)?;
        const ROOM: u64 = 32 << 20;
        let (staging, data) = host
            .upload_buffer(ROOM)
            .ok_or_else(windows::core::Error::empty)?;
        let (back, read) = host
            .shared_buffer(ROOM)
            .ok_or_else(windows::core::Error::empty)?;
        let (fill, fetch) = (&self.lists[0], &self.lists[1]);
        fill.reset();
        fetch.reset();
        let location = |res: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed; the location never releases it.
            pResource: unsafe { std::mem::transmute_copy(res) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        let placed = |res: &ID3D12Resource, mut f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, at: u64| {
            f.Offset = at;
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(res) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
            }
        };
        let mut at = 0u64;
        // For each subresource: its texture, where it is read back to, its
        // layout, and the bytes it was given (none for a depth target).
        let mut reads: Vec<(usize, u64, usize, usize, usize, Vec<u8>)> = Vec::new();
        for (index, e) in self.extras.iter().enumerate() {
            // The state it is in once filled.
            let filled = if e.depth {
                D3D12_RESOURCE_STATE_DEPTH_WRITE
            } else {
                D3D12_RESOURCE_STATE_COPY_DEST
            };
            let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
            // SAFETY: an open list; a live texture in that state by then.
            unsafe { fetch.list.ResourceBarrier(&[transition(&e.res, filled, source)]) };
            if e.depth {
                let all = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
                let flags = if e.subs > 1 { all } else { D3D12_CLEAR_FLAG_DEPTH };
                // SAFETY: an open list; a depth target in its state.
                unsafe {
                    fill.list.ClearDepthStencilView(
                        host.dsv(&e.res, None),
                        flags,
                        0.75,
                        0x5a,
                        None,
                    );
                }
            }
            for sub in 0..e.subs {
                let (f, rows, row, total) = Self::layout(&e.desc, sub);
                let (pitch, slices) = (f.Footprint.RowPitch as usize, f.Footprint.Depth.max(1));
                let (rows, row) = (rows as usize * slices as usize, row as usize);
                assert!(at + total <= ROOM);
                let mut given = Vec::new();
                if !e.depth {
                    given.reserve(rows * row);
                    for y in 0..rows {
                        for x in 0..row {
                            let v =
                                (index * 31 + sub as usize * 17 + y * 7 + x * 3 + (x >> 3)) as u8;
                            // SAFETY: inside the upload buffer's mapping.
                            unsafe { data.add(at as usize + y * pitch + x).write(v) };
                            given.push(v);
                        }
                    }
                    // SAFETY: an open list; live resources; the texture in
                    // the common state, which a copy promotes.
                    unsafe {
                        fill.list.CopyTextureRegion(
                            &location(&e.res, sub),
                            0,
                            0,
                            0,
                            &placed(&staging, f, at),
                            None,
                        );
                    }
                }
                // SAFETY: an open list; live resources.
                unsafe {
                    fetch.list.CopyTextureRegion(
                        &placed(&back, f, at),
                        0,
                        0,
                        0,
                        &location(&e.res, sub),
                        None,
                    );
                }
                reads.push((index, at, pitch, rows, row, given));
                at += total.next_multiple_of(512);
            }
            // Depth targets go back to their state; the others stay copy
            // sources.
            if e.depth {
                // SAFETY: as above.
                unsafe { fetch.list.ResourceBarrier(&[transition(&e.res, source, filled)]) };
            }
        }
        fill.close();
        fetch.close();
        host.execute(&[fill]);
        host.execute(&[fetch]);
        for (index, from, pitch, rows, row, given) in reads {
            let mut bytes = Vec::with_capacity(rows * row);
            for y in 0..rows {
                // SAFETY: a row of the copy's buffer; the copy has run.
                bytes.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(read.add(from as usize + y * pitch), row)
                });
            }
            let e = &mut self.extras[index];
            // What was given is what is held, but for the packed floats.
            let packed = matches!(e.desc.Format.0, 26 | 67);
            if !e.depth && !packed {
                assert!(bytes == given, "{} does not hold the bytes it was given", e.name);
            } else if !e.depth && bytes != given {
                eprintln!("{} holds other bits than it was given", e.name);
            }
            e.held.push(Rows {
                pitch,
                rows,
                row,
                bytes,
            });
        }
        Ok(())
    }

    /// What the CPU writes before a submission.
    fn write_inputs(&self, round: u32, number: u64) {
        for i in 0..WORDS {
            let v = (i as u32 * 7 + 1) * (round + 1) + round * 1000;
            // SAFETY: inside the buffer's mapping; no list in flight reads
            // it (every submission was waited for).
            unsafe { self.inputs.1.cast::<u32>().add(i).write(v) };
        }
        // A draw's arguments (six vertices and one instance; three in
        // every fifth frame, so that arguments left from the frame before
        // are not this frame's: tests/replay.rs records such a frame, the
        // ninth, and ends with another, as it compares with the last
        // frame's picture) and six indices.
        let vertices = if number % 5 == 4 { 3 } else { 6 };
        // SAFETY: as above.
        unsafe {
            self.inputs
                .1
                .add(ARGUMENTS)
                .cast::<[u32; 4]>()
                .write([vertices, 1, 0, 0]);
            self.inputs
                .1
                .add(INDICES)
                .cast::<[u16; 6]>()
                .write([0, 1, 2, 3, 4, 5]);
        }
    }

    /// Records a compute pass over the inputs.
    fn pass(&self, l: &ID3D12GraphicsCommandList, seed: u32) {
        // SAFETY: an open list; live objects.
        unsafe {
            l.SetComputeRootSignature(&self.compute_root);
            l.SetPipelineState(&self.compute);
            l.SetComputeRoot32BitConstant(0, seed, 0);
            l.SetComputeRootShaderResourceView(1, self.inputs.0.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, self.results.0.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::transmute_copy(&self.results.0),
                    }),
                },
            }]);
        }
    }

    /// Records a rectangle: where, its tint and depth, the view slot its
    /// texture is in, and how it is drawn.
    fn rectangle(
        &self,
        l: &ID3D12GraphicsCommandList,
        place: [f32; 4],
        tint: [f32; 4],
        depth: f32,
        slot: usize,
        how: How<'_>,
    ) {
        let Some(host) = host() else { return };
        let (rtv, dsv) = (host.rtv(&self.target, None), host.dsv(&self.depth, None));
        let mut constants = [0u32; 9];
        for (c, v) in constants.iter_mut().zip(place.iter().chain(&tint).chain([&depth])) {
            *c = v.to_bits();
        }
        // SAFETY: an open list; live objects; the targets in their states.
        unsafe {
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }]);
            l.SetDescriptorHeaps(&[Some(self.views.clone())]);
            l.SetGraphicsRootSignature(&self.draw_root);
            l.SetPipelineState(&self.draw);
            l.SetGraphicsRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
            l.SetGraphicsRootDescriptorTable(1, self.table(slot));
            l.SetGraphicsRootShaderResourceView(2, self.results.0.GetGPUVirtualAddress());
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            match how {
                How::Plain => l.DrawInstanced(6, 1, 0, 0),
                How::Indexed => {
                    l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: self.inputs.0.GetGPUVirtualAddress() + INDICES as u64,
                        SizeInBytes: 12,
                        Format: DXGI_FORMAT_R16_UINT,
                    }));
                    l.DrawIndexedInstanced(6, 1, 0, 0, 0);
                }
                How::Indirect => l.ExecuteIndirect(
                    &self.signature,
                    1,
                    &self.inputs.0,
                    ARGUMENTS as u64,
                    None::<&ID3D12Resource>,
                    0,
                ),
                How::Clamped(arguments, at) => {
                    l.ExecuteIndirect(&self.signature, 1, arguments, at, None::<&ID3D12Resource>, 0)
                }
            }
        }
    }

    /// Where the inputs' buffer holds the draw's arguments, for the host
    /// layer's clamp: the buffer's start and the address, and a plain draw
    /// of one record.
    fn arguments(&self) -> ((u64, u64), (u32, u32, u32)) {
        // SAFETY: a plain query on a live buffer.
        let base = unsafe { self.inputs.0.GetGPUVirtualAddress() };
        ((base, base + ARGUMENTS as u64), (0, 16, 1))
    }

    /// Records a clear of the rectangle `rect` of the colour target to
    /// `colour` through its unordered-access view; the target is a render
    /// target before and after.
    fn clear_unordered(&self, l: &ID3D12GraphicsCommandList, colour: [f32; 4], rect: RECT) {
        let (target, unordered) = (
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: an open list; live objects; both handles hold the view.
        unsafe {
            l.SetDescriptorHeaps(&[Some(self.views.clone())]);
            l.ResourceBarrier(&[transition(&self.target, target, unordered)]);
            l.ClearUnorderedAccessViewFloat(
                self.table(CLEARED),
                self.clears.GetCPUDescriptorHandleForHeapStart(),
                &self.target,
                &colour,
                &[rect],
            );
            l.ResourceBarrier(&[transition(&self.target, unordered, target)]);
        }
    }

    /// One frame: three submissions, then the present.
    fn frame(&self, number: u64, drawn: u64) {
        let Some(host) = host() else { return };
        let [a, b, c, d] = &self.lists[..] else { return };
        let common = D3D12_RESOURCE_STATE_COMMON;
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // The first submission: clears, a pass, a rectangle of the first
        // texture.
        self.write_inputs(0, number);
        self.view(2, &self.first);
        self.no_view(3);
        a.reset();
        // SAFETY: an open list; live objects.
        unsafe {
            let l = &a.list;
            l.ResourceBarrier(&[transition(&self.target, common, target)]);
            l.ClearRenderTargetView(host.rtv(&self.target, None), &[0.0, 0.1, 0.3, 1.0], None);
            l.ClearDepthStencilView(
                host.dsv(&self.depth, None),
                D3D12_CLEAR_FLAG_DEPTH,
                1.0,
                0,
                None,
            );
        }
        // A strip along the top, cleared through the unordered-access
        // view: under what is drawn where that reaches it.
        let strip = RECT {
            left: 0,
            top: 0,
            right: SIZE as i32,
            bottom: 12,
        };
        self.clear_unordered(&a.list, [0.9, 0.2, 0.1, 1.0], strip);
        self.pass(&a.list, 5);
        let white = [1.0, 1.0, 1.0, 1.0];
        self.rectangle(&a.list, [-0.9, -0.9, 1.2, 1.2], white, 0.6, 2, How::Plain);
        a.close();
        host.execute(&[a]);
        // The second: other inputs, a view where there was none; two lists.
        self.write_inputs(1, number);
        self.view(3, &self.second);
        b.reset();
        self.pass(&b.list, 9);
        c.reset();
        // Three draws by arguments the host layer clamps in the list before
        // their own, as it does for a title's runs of them: two in one
        // batch, then one alone.
        let (at, spec) = self.arguments();
        let hoisted: Vec<(ID3D12Resource, u64)> = [true, true, false]
            .into_iter()
            .filter_map(|batched| crate::indirect::bound_before_as(b, c, at, spec, None, batched))
            .collect();
        c.clamp_end(b);
        b.close();
        let pink = [1.0, 0.6, 0.8, 1.0];
        self.rectangle(&c.list, [-0.3, -0.5, 1.1, 1.0], pink, 0.4, 3, How::Indexed);
        let places = [
            ([-0.95, 0.3, 0.5, 0.6], [0.5, 0.7, 1.0, 1.0], 0.3),
            ([0.45, -0.95, 0.5, 0.4], [1.0, 0.6, 0.2, 1.0], 0.35),
            ([-0.2, 0.55, 0.4, 0.4], [0.8, 0.5, 1.0, 1.0], 0.25),
        ];
        for ((arguments, at), (place, tint, depth)) in hoisted.iter().zip(places) {
            self.rectangle(&c.list, place, tint, depth, 3, How::Clamped(arguments, *at));
        }
        c.close();
        host.execute(&[b, c]);
        // The third: the view the first one drew with replaced, a
        // rectangle between the two, and the picture copied out.
        self.write_inputs(2, number);
        self.view(2, &self.second);
        d.reset();
        self.pass(&d.list, 1);
        let green = [0.7, 1.0, 0.5, 1.0];
        self.rectangle(&d.list, [-0.6, -0.2, 1.5, 0.9], green, 0.5, 2, How::Indirect);
        // And one the host layer clamps in place, just before it.
        let (at, spec) = self.arguments();
        if let Some(arguments) = crate::indirect::bound(d, at.1, spec, None) {
            let yellow = [1.0, 0.9, 0.3, 1.0];
            self.rectangle(&d.list, [0.2, 0.4, 0.7, 0.5], yellow, 0.2, 2, How::Clamped(&arguments, 0));
            crate::indirect::done(d, &arguments);
        }
        // A corner cleared the same way over everything drawn, part of it
        // over the strip.
        let corner = RECT {
            left: SIZE as i32 - 20,
            top: 4,
            right: SIZE as i32,
            bottom: 28,
        };
        self.clear_unordered(&d.list, [0.1, 0.8, 0.9, 1.0], corner);
        // SAFETY: an open list; live objects.
        unsafe {
            let l = &d.list;
            let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
            l.ResourceBarrier(&[transition(&self.target, target, source)]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.out.0),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: self.footprint,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            l.ResourceBarrier(&[transition(&self.target, source, common)]);
        }
        d.close();
        host.execute(&[d]);
        crate::replay::record::present(number, Some((&self.target, common.0 as u32)), drawn);
    }

    fn made(&self) -> Made {
        let pitch = self.footprint.Footprint.RowPitch as usize;
        let mut picture = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for y in 0..SIZE as usize {
            // SAFETY: a row of the copy's buffer; the copy has run.
            picture.extend_from_slice(unsafe {
                std::slice::from_raw_parts(self.out.1.add(y * pitch), SIZE as usize * 4)
            });
        }
        // SAFETY: the buffer's mapping; its passes have run.
        let results =
            unsafe { std::slice::from_raw_parts(self.results.1, WORDS * 4) }.to_vec();
        let others = self
            .extras
            .iter()
            .map(|e| Other {
                name: e.name.clone(),
                subs: e
                    .held
                    .iter()
                    .map(|r| Rows {
                        pitch: r.pitch,
                        rows: r.rows,
                        row: r.row,
                        bytes: r.bytes.clone(),
                    })
                    .collect(),
            })
            .collect();
        Made {
            picture,
            results,
            others,
        }
    }
}

/// Runs frames `1..=frames` of the made-up program through the host layer,
/// each ended as the title's are (`record::present`); frame `world` is told
/// as the world's first (by the draws the host is said to have run). What
/// the last frame left; `None` without a host GPU.
pub fn run(frames: u64, world: u64) -> Option<Made> {
    host()?;
    let program = match Program::new() {
        Ok(program) => program,
        Err(e) => panic!("the made-up frame's objects: {e}"),
    };
    let mut drawn = 0;
    for number in 1..=frames {
        // Three draws a frame; the world's first frame runs many.
        drawn += if number == world { 600 } else { 3 };
        program.frame(number, drawn);
    }
    Some(program.made())
}
