//! What a D3D12.X list call costs on the title's thread: several threads
//! record command lists at once, the way a title's render threads do,
//! through the method slots the title calls (device and list tables), and
//! each call is timed. Draws and dispatches run real host pipelines, made
//! from hand-assembled console shaders ([`gfx10`]) that read descriptor
//! tables, so binding works as it does for a title's draws.
//!
//! Per thread and frame: the list is reset, its heaps, targets, viewport,
//! scissor, topology, pipeline and buffers set; then per draw a constant
//! buffer view is written into the shader-visible heap and bound as a
//! table, root constants and a root view are set, and an indexed draw is
//! recorded; every so often a texture view is copied into the heap and
//! bound, and a barrier pair recorded; then a compute pass (root, pipeline,
//! table, dispatch) between barriers; then the list is closed. The main
//! thread submits all lists and waits for them, as a frame does.
//!
//! Issue #2586 measured about 8,500 calls a frame on each of five title
//! threads; the defaults are close to that. The numbers are not a title's:
//! they are the same for every build, so two builds can be compared on one
//! machine.
//!
//! `cargo test --release -p victory-gpu --lib list_calls -- --ignored --nocapture`
//! (`LIST_BENCH_THREADS`, `LIST_BENCH_DRAWS`, `LIST_BENCH_FRAMES` change the
//! shape). Run it alone: other tests switch the device to the newer layouts,
//! which move list slots.

use std::time::{Duration, Instant};

use crate::object::{Iid, S_OK, This, guid, slot};

mod gfx10;

/// The render target's width and height, and the compute shader's group
/// size (the compute shader's own constants).
const SIZE: u32 = 64;
const GROUP: u32 = 8;

const QUEUE: Iid = guid(
    0x0ec870a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);
const ALLOCATOR: Iid = guid(
    0x6102dee4,
    0xaf59,
    0x4b09,
    [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
);
const LIST: Iid = guid(
    0x5b160d0f,
    0xac1b,
    0x4185,
    [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
);
const HEAP: Iid = guid(
    0x8efb471d,
    0x616c,
    0x4f49,
    [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
);
const RESOURCE: Iid = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);
const FENCE: Iid = guid(
    0x0a753dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);
const PIPELINE: Iid = guid(
    0x765a30f3,
    0xf624,
    0x4c6f,
    [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
);

// DXGI formats.
const RGBA8: u32 = 28;
const D32: u32 = 40;
const RG32F: u32 = 16;
const RGB32F: u32 = 6;
const R16U: u32 = 57;
const R32_TYPELESS: u32 = 39;

// Resource states.
const COMMON: u32 = 0;
const RENDER_TARGET: u32 = 0x4;
const UNORDERED: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const NON_PIXEL: u32 = 0x40;
const PIXEL: u32 = 0x80;
const GENERIC_READ: u32 = 0xAC3;

const VIEW: u64 = 32;
/// Views each thread owns in the shader-visible heap, from
/// `THREAD_VIEWS * (thread + 1)`: its compute table (frame, texture shown,
/// output), its ring of per-draw constant views, its ring of copied
/// texture views.
const THREAD_VIEWS: u32 = 8192;
const DRAW_RING: u32 = 4096;
const COPY_RING: u32 = 1024;
const COPIES_AT: u32 = 16 + DRAW_RING;
/// Bytes of constants per draw view.
const CONSTANTS: u64 = 256;

/// Calls `$obj`'s method at byte offset `$at`.
macro_rules! call {
    ($obj:expr, $at:expr, fn($($t:ty),* $(,)?) $(-> $r:ty)?, $($arg:expr),* $(,)?) => {{
        let obj: This = $obj;
        slot::<extern "system" fn(This $(, $t)*) $(-> $r)?>(obj, $at)(obj $(, $arg)*)
    }};
}

#[repr(C)]
struct HeapProperties {
    kind: u32,
    cpu_page: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth: u16,
    mips: u16,
    format: u32,
    samples: u32,
    quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
struct ClearValue {
    format: u32,
    value: [f32; 4],
}

#[repr(C)]
struct Barrier {
    kind: u32,
    flags: u32,
    resource: This,
    subresource: u32,
    before: u32,
    after: u32,
}

#[repr(C)]
struct BufferView {
    address: u64,
    size: u32,
    stride_or_format: u32,
}

#[repr(C)]
struct InputElement {
    name: *const u8,
    index: u32,
    format: u32,
    slot: u32,
    offset: u32,
    class: u32,
    step: u32,
}

#[repr(C)]
struct ComputeDesc {
    root: This,
    code: *const u8,
    size: usize,
}

/// A pipeline stream: each subobject a type then its data, aligned to
/// pointers.
#[derive(Default)]
struct Stream(Vec<u8>);

impl Stream {
    fn push(&mut self, kind: u32, data: &[u8], pointers: bool) {
        self.0.extend(kind.to_le_bytes());
        if pointers {
            self.0.extend([0; 4]);
        }
        self.0.extend(data);
        self.0.resize(self.0.len().next_multiple_of(8), 0);
    }
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn pointer(p: *const u8, size: usize) -> Vec<u8> {
    let mut out = (p as u64).to_le_bytes().to_vec();
    out.extend((size as u64).to_le_bytes());
    out
}

fn buffer(width: u64, flags: u32) -> ResourceDesc {
    ResourceDesc {
        dimension: 1,
        alignment: 0,
        width,
        height: 1,
        depth: 1,
        mips: 1,
        format: 0,
        samples: 1,
        quality: 0,
        layout: 1,
        flags,
    }
}

fn texture(format: u32, flags: u32) -> ResourceDesc {
    ResourceDesc {
        dimension: 3,
        alignment: 0,
        width: u64::from(SIZE),
        height: SIZE,
        depth: 1,
        mips: 1,
        format,
        samples: 1,
        quality: 0,
        layout: 0,
        flags,
    }
}

/// A committed resource: `(heap type, description, state, clear)`.
fn resource(heap: u32, desc: ResourceDesc, state: u32, clear: Option<ClearValue>) -> This {
    let device = crate::device::device();
    let props = HeapProperties {
        kind: heap,
        cpu_page: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let clear = clear.as_ref().map_or(std::ptr::null(), std::ptr::from_ref);
    let mut out = std::ptr::null();
    let hr = call!(
        device,
        0xd8,
        fn(
            *const HeapProperties,
            u32,
            *const ResourceDesc,
            u32,
            *const ClearValue,
            *const Iid,
            *mut This,
        ) -> i32,
        &props,
        0,
        &desc,
        state,
        clear,
        &RESOURCE,
        &mut out
    );
    assert_eq!(hr, S_OK, "a committed resource");
    out
}

fn address(resource: This) -> u64 {
    call!(resource, 0x58, fn() -> u64,)
}

fn heap(kind: u32, count: u32, visible: bool) -> (This, u64, u64) {
    // The description, then no placement: newer titles pass memory of
    // their own for the descriptors after it, and the device reads it.
    let desc = [kind, count, u32::from(visible), 0, 0, 0, 0, 0];
    let mut heap = std::ptr::null();
    let hr = call!(
        crate::device::device(),
        0x70,
        fn(*const u32, *const Iid, *mut This) -> i32,
        desc.as_ptr(),
        &HEAP,
        &mut heap
    );
    assert_eq!(hr, S_OK, "a descriptor heap");
    let cpu = call!(heap, 0x48, fn() -> u64,);
    let gpu = if visible {
        call!(heap, 0x50, fn() -> u64,)
    } else {
        0
    };
    (heap, cpu, gpu)
}

/// The methods timed, as the title calls them.
#[derive(Clone, Copy)]
enum M {
    Reset,
    Close,
    Heaps,
    RootSignature,
    Targets,
    Viewport,
    Scissor,
    Topology,
    Pipeline,
    VertexBuffers,
    IndexBuffer,
    Table,
    Constants,
    RootView,
    DrawIndexed,
    Dispatch,
    Barrier,
    CreateCbv,
    CopyDescriptors,
}

const METHODS: [&str; 19] = [
    "Reset",
    "Close",
    "SetDescriptorHeaps",
    "Set*RootSignature",
    "OMSetRenderTargets",
    "RSSetViewports",
    "RSSetScissorRects",
    "IASetPrimitiveTopology",
    "SetPipelineState",
    "IASetVertexBuffers",
    "IASetIndexBuffer",
    "Set*RootDescriptorTable",
    "Set*Root32BitConstants",
    "SetGraphicsRootCBV",
    "DrawIndexedInstanced",
    "Dispatch",
    "ResourceBarrier",
    "(device) CreateCBV",
    "(device) CopyDescriptorsSimple",
];

/// Time and calls per method, on one thread.
#[derive(Clone, Default)]
struct Times {
    spent: [Duration; METHODS.len()],
    calls: [u64; METHODS.len()],
}

impl Times {
    fn time<R>(&mut self, m: M, f: impl FnOnce() -> R) -> R {
        let started = Instant::now();
        let r = f();
        self.spent[m as usize] += started.elapsed();
        self.calls[m as usize] += 1;
        r
    }

    fn add(&mut self, other: &Times) {
        for i in 0..METHODS.len() {
            self.spent[i] += other.spent[i];
            self.calls[i] += other.calls[i];
        }
    }
}

/// What every thread shares.
struct Shared {
    views_heap: This,
    samplers_heap: This,
    view_cpu: u64,
    view_gpu: u64,
    sampler_gpu: u64,
    /// Non-shader-visible views of the texture, the sources of copies.
    staging_cpu: u64,
    graphics: This,
    compute: This,
    vertices: BufferView,
    indices: BufferView,
    constants: u64,
}

/// One title thread's list and resources.
struct Thread {
    n: u32,
    allocator: This,
    list: This,
    rtv: u64,
    dsv: u64,
    target: This,
    aux: This,
}

// SAFETY: plain addresses of objects that live for the whole run; each
// thread records its own list.
unsafe impl Sync for Thread {}
// SAFETY: as for `Thread`; the objects are only read.
unsafe impl Sync for Shared {}

fn env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Thread {
    fn barrier(&self, t: &mut Times, resource: This, before: u32, after: u32) {
        let b = Barrier {
            kind: 0,
            flags: 0,
            resource,
            subresource: u32::MAX,
            before,
            after,
        };
        t.time(M::Barrier, || {
            call!(self.list, 0xd0, fn(u32, *const Barrier) -> u32, 1, &b)
        });
    }

    /// Records one frame's list.
    fn record(&self, s: &Shared, frame: u32, draws: u32, t: &mut Times) {
        let device = crate::device::device();
        let list = self.list;
        let base = THREAD_VIEWS * (self.n + 1);
        let view_cpu = |i: u32| s.view_cpu + u64::from(i) * VIEW;
        let view_gpu = |i: u32| s.view_gpu + u64::from(i) * VIEW;
        call!(self.allocator, 0x40, fn() -> i32,);
        let hr = t.time(M::Reset, || {
            call!(
                list,
                0x50,
                fn(This, This) -> i32,
                self.allocator,
                std::ptr::null()
            )
        });
        assert_eq!(hr, S_OK);
        let heaps = [s.views_heap, s.samplers_heap];
        t.time(M::Heaps, || {
            call!(list, 0x100, fn(u32, *const This), 2, heaps.as_ptr())
        });
        t.time(M::RootSignature, || {
            call!(list, 0x110, fn(This), std::ptr::null())
        });
        t.time(M::Targets, || {
            call!(
                list,
                0x190,
                fn(u32, *const u64, i32, *const u64),
                1,
                &self.rtv,
                0,
                &self.dsv
            )
        });
        let viewport = [0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0f32];
        t.time(M::Viewport, || {
            call!(list, 0xa8, fn(u32, *const [f32; 6]), 1, &viewport)
        });
        let scissor = [0, 0, SIZE as i32, SIZE as i32];
        t.time(M::Scissor, || {
            call!(list, 0xb0, fn(u32, *const [i32; 4]), 1, &scissor)
        });
        t.time(M::Topology, || call!(list, 0xa0, fn(u32), 4));
        t.time(M::Pipeline, || call!(list, 0xc8, fn(This), s.graphics));
        t.time(M::VertexBuffers, || {
            call!(
                list,
                0x180,
                fn(u32, u32, *const BufferView),
                0,
                1,
                &s.vertices
            )
        });
        t.time(M::IndexBuffer, || {
            call!(list, 0x178, fn(*const BufferView), &s.indices)
        });
        t.time(M::Table, || {
            call!(list, 0x120, fn(u32, u64), 0, view_gpu(0))
        });
        t.time(M::Table, || {
            call!(list, 0x120, fn(u32, u64), 2, view_gpu(5))
        });
        t.time(M::Table, || {
            call!(list, 0x120, fn(u32, u64), 3, s.sampler_gpu)
        });
        let constants = s.constants + u64::from(self.n) * u64::from(DRAW_RING) * CONSTANTS;
        for d in 0..draws {
            let k = (frame * draws + d) % DRAW_RING;
            // The draw's constants: a view written into the heap, bound as
            // a table.
            let desc = [constants + u64::from(k) * CONSTANTS, CONSTANTS];
            let slot = base + 16 + k;
            t.time(M::CreateCbv, || {
                call!(
                    device,
                    0x88,
                    fn(*const [u64; 2], u64),
                    &desc,
                    view_cpu(slot)
                )
            });
            t.time(M::Table, || {
                call!(list, 0x120, fn(u32, u64), 1, view_gpu(slot))
            });
            let values = [d, frame, 0x3f80_0000, 0];
            t.time(M::Constants, || {
                call!(
                    list,
                    0x140,
                    fn(u32, u32, *const u32, u32),
                    4,
                    4,
                    values.as_ptr(),
                    0
                )
            });
            t.time(M::RootView, || call!(list, 0x150, fn(u32, u64), 5, desc[0]));
            if d % 32 == 0 {
                // A material's texture view copied into the heap.
                let c = base + COPIES_AT + (frame * draws / 32 + d / 32) % COPY_RING;
                t.time(M::CopyDescriptors, || {
                    call!(
                        device,
                        0xc0,
                        fn(u32, u64, u64, u32),
                        1,
                        view_cpu(c),
                        s.staging_cpu,
                        0
                    )
                });
                t.time(M::Table, || {
                    call!(list, 0x120, fn(u32, u64), 2, view_gpu(c))
                });
            }
            if d % 128 == 64 {
                self.barrier(t, self.aux, COMMON, PIXEL);
                self.barrier(t, self.aux, PIXEL, COMMON);
            }
            if d % 256 == 255 {
                t.time(M::Pipeline, || call!(list, 0xc8, fn(This), s.graphics));
                t.time(M::VertexBuffers, || {
                    call!(
                        list,
                        0x180,
                        fn(u32, u32, *const BufferView),
                        0,
                        1,
                        &s.vertices
                    )
                });
                t.time(M::IndexBuffer, || {
                    call!(list, 0x178, fn(*const BufferView), &s.indices)
                });
            }
            t.time(M::DrawIndexed, || {
                call!(list, 0x68, fn(u32, u32, u32, i32, u32), 3, 1, 0, 0, 0)
            });
        }
        // The compute pass reads the frame and writes the texture shown and
        // the output buffer.
        self.barrier(t, self.target, RENDER_TARGET, NON_PIXEL);
        t.time(M::RootSignature, || {
            call!(list, 0x108, fn(This), std::ptr::null())
        });
        t.time(M::Pipeline, || call!(list, 0xc8, fn(This), s.compute));
        for _ in 0..draws / 32 {
            t.time(M::Table, || {
                call!(list, 0x118, fn(u32, u64), 0, view_gpu(base))
            });
            let values = [frame, 1];
            t.time(M::Constants, || {
                call!(
                    list,
                    0x138,
                    fn(u32, u32, *const u32, u32),
                    1,
                    2,
                    values.as_ptr(),
                    0
                )
            });
            t.time(M::Dispatch, || {
                call!(list, 0x70, fn(u32, u32, u32), SIZE / GROUP, SIZE / GROUP, 1)
            });
        }
        self.barrier(t, self.target, NON_PIXEL, RENDER_TARGET);
        let hr = t.time(M::Close, || call!(list, 0x48, fn() -> i32,));
        assert_eq!(hr, S_OK);
    }
}

/// This thread's processor time so far.
fn thread_cpu() -> Duration {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let mut t = [FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    }; 4];
    let [created, exited, kernel, user] = &mut t;
    // SAFETY: four times of our own thread into our own variables.
    unsafe { GetThreadTimes(GetCurrentThread(), created, exited, kernel, user) };
    let v = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    Duration::from_nanos((v(user) + v(kernel)) * 100)
}

#[test]
#[ignore = "a benchmark: run alone, in release"]
fn list_calls() {
    if crate::host::host().is_none() {
        return; // no host GPU here
    }
    let threads = env("LIST_BENCH_THREADS", 5);
    let draws = env("LIST_BENCH_DRAWS", 1600);
    let frames = env("LIST_BENCH_FRAMES", 10);
    let device = crate::device::device();

    let (views_heap, view_cpu, view_gpu) = heap(0, THREAD_VIEWS * (threads + 1), true);
    let (samplers_heap, sampler_cpu, sampler_gpu) = heap(1, 16, true);
    let (_staging, staging_cpu, _) = heap(0, 16, false);
    let (_targets, rtv_cpu, _) = heap(2, threads, false);
    let (_depths, dsv_cpu, _) = heap(3, threads, false);

    let shaders = [
        gfx10::vertex_container(),
        gfx10::pixel_container(),
        gfx10::compute_container(),
    ];
    let names = [c"POSITION", c"NORMAL", c"TEXCOORD"];
    let elements: Vec<InputElement> = [(RGB32F, 0), (RGB32F, 12), (RG32F, 24)]
        .iter()
        .zip(names)
        .map(|(&(format, offset), name)| InputElement {
            name: name.as_ptr().cast(),
            index: 0,
            format,
            slot: 0,
            offset,
            class: 0,
            step: 0,
        })
        .collect();
    let mut s = Stream::default();
    s.push(0, &0u64.to_le_bytes(), true);
    s.push(1, &pointer(shaders[0].as_ptr(), shaders[0].len()), true);
    s.push(2, &pointer(shaders[1].as_ptr(), shaders[1].len()), true);
    s.push(10, &words(&[3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]), false);
    let keep = [1, 1, 1, 8];
    let mut depth = vec![1, 1, 2, 0, 0xffff];
    depth.extend(keep);
    depth.extend(keep);
    s.push(11, &words(&depth), false);
    let mut layout = pointer(elements.as_ptr().cast(), 0);
    layout[8..16].copy_from_slice(&(elements.len() as u64).to_le_bytes());
    s.push(12, &layout, true);
    s.push(14, &3u32.to_le_bytes(), false);
    s.push(15, &words(&[RGBA8, 0, 0, 0, 0, 0, 0, 0, 1]), false);
    s.push(16, &D32.to_le_bytes(), false);
    s.push(17, &words(&[1, 0]), false);
    let stream = [s.0.len() as u64, s.0.as_ptr() as u64];
    let mut graphics = std::ptr::null();
    let hr = call!(
        device,
        0x2b0,
        fn(*const [u64; 2], *const Iid, *mut This) -> i32,
        &stream,
        &PIPELINE,
        &mut graphics
    );
    assert_eq!(hr, S_OK, "the graphics pipeline");
    let desc = ComputeDesc {
        root: std::ptr::null(),
        code: shaders[2].as_ptr(),
        size: shaders[2].len(),
    };
    let mut compute = std::ptr::null();
    let hr = call!(
        device,
        0x58,
        fn(*const ComputeDesc, *const Iid, *mut This) -> i32,
        &desc,
        &PIPELINE,
        &mut compute
    );
    assert_eq!(hr, S_OK, "the compute pipeline");

    // Shared buffers and views: the scene's constants (view 0), the
    // texture (view 5, and the copies' source), a sampler.
    let vertices = resource(2, buffer(3 * 32, 0), GENERIC_READ, None);
    let indices = resource(2, buffer(8, 0), GENERIC_READ, None);
    let constants = resource(
        2,
        buffer(
            u64::from(threads) * u64::from(DRAW_RING) * CONSTANTS + CONSTANTS,
            0,
        ),
        GENERIC_READ,
        None,
    );
    let tex = resource(1, texture(RGBA8, 0), PIXEL, None);
    let scene = [
        address(constants) + u64::from(threads) * u64::from(DRAW_RING) * CONSTANTS,
        CONSTANTS,
    ];
    call!(device, 0x88, fn(*const [u64; 2], u64), &scene, view_cpu);
    let srv = [RGBA8, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
    call!(
        device,
        0x90,
        fn(This, *const [u32; 10], u64),
        tex,
        &srv,
        view_cpu + 5 * VIEW
    );
    call!(
        device,
        0x90,
        fn(This, *const [u32; 10], u64),
        tex,
        &srv,
        staging_cpu
    );
    let mut sampler = [0x15u32, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, f32::MAX.to_bits()];
    sampler[4] = 0f32.to_bits();
    call!(
        device,
        0xb0,
        fn(*const [u32; 13], u64),
        &sampler,
        sampler_cpu
    );

    let mut workers = Vec::new();
    for n in 0..threads {
        let mut allocator = std::ptr::null();
        let mut list = std::ptr::null();
        assert_eq!(
            call!(
                device,
                0x48,
                fn(u32, *const Iid, *mut This) -> i32,
                0,
                &ALLOCATOR,
                &mut allocator
            ),
            S_OK
        );
        let hr = call!(
            device,
            0x60,
            fn(u32, u32, This, This, *const Iid, *mut This) -> i32,
            0,
            0,
            allocator,
            std::ptr::null(),
            &LIST,
            &mut list
        );
        assert_eq!(hr, S_OK);
        assert_eq!(call!(list, 0x48, fn() -> i32,), S_OK);
        let clear = |format, value| Some(ClearValue { format, value });
        let target = resource(1, texture(RGBA8, 1), RENDER_TARGET, clear(RGBA8, [0.0; 4]));
        let depth = resource(
            1,
            texture(D32, 2),
            DEPTH_WRITE,
            clear(D32, [1.0, 0.0, 0.0, 0.0]),
        );
        let shown = resource(1, texture(RGBA8, 4), UNORDERED, None);
        let output = resource(1, buffer(u64::from(SIZE * SIZE * 16), 4), UNORDERED, None);
        let aux = resource(1, texture(RGBA8, 0), COMMON, None);
        let (rtv, dsv) = (rtv_cpu + u64::from(n) * 128, dsv_cpu + u64::from(n) * 128);
        call!(
            device,
            0xa0,
            fn(This, *const u8, u64),
            target,
            std::ptr::null(),
            rtv
        );
        call!(
            device,
            0xa8,
            fn(This, *const u8, u64),
            depth,
            std::ptr::null(),
            dsv
        );
        let at = view_cpu + u64::from(THREAD_VIEWS * (n + 1)) * VIEW;
        call!(
            device,
            0x90,
            fn(This, *const [u32; 10], u64),
            target,
            &srv,
            at
        );
        let uav = [RGBA8, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        call!(
            device,
            0x98,
            fn(This, This, *const [u32; 10], u64),
            shown,
            std::ptr::null(),
            &uav,
            at + VIEW
        );
        let raw = [R32_TYPELESS, 1, 0, 0, SIZE * SIZE * 4, 0, 0, 0, 1, 0];
        call!(
            device,
            0x98,
            fn(This, This, *const [u32; 10], u64),
            output,
            std::ptr::null(),
            &raw,
            at + 2 * VIEW
        );
        workers.push(Thread {
            n,
            allocator,
            list,
            rtv,
            dsv,
            target,
            aux,
        });
    }
    let shared = Shared {
        views_heap,
        samplers_heap,
        view_cpu,
        view_gpu,
        sampler_gpu,
        staging_cpu,
        graphics,
        compute,
        vertices: BufferView {
            address: address(vertices),
            size: 3 * 32,
            stride_or_format: 32,
        },
        indices: BufferView {
            address: address(indices),
            size: 6,
            stride_or_format: R16U,
        },
        constants: address(constants),
    };

    let mut queue = std::ptr::null();
    let desc = [0u32; 4];
    assert_eq!(
        call!(
            device,
            0x40,
            fn(*const u32, *const Iid, *mut This) -> i32,
            desc.as_ptr(),
            &QUEUE,
            &mut queue
        ),
        S_OK
    );
    let mut fence = std::ptr::null();
    assert_eq!(
        call!(
            device,
            0x120,
            fn(u64, u32, *const Iid, *mut This) -> i32,
            0,
            0,
            &FENCE,
            &mut fence
        ),
        S_OK
    );
    let lists: Vec<This> = workers.iter().map(|w| w.list).collect();
    let mut fence_value = 0;
    let mut submit = || {
        call!(
            queue,
            0x50,
            fn(u32, *const This),
            lists.len() as u32,
            lists.as_ptr()
        );
        fence_value += 1;
        call!(queue, 0x70, fn(This, u64) -> i32, fence, fence_value);
        call!(
            fence,
            0x48,
            fn(u64, *mut u8) -> i32,
            fence_value,
            std::ptr::null_mut()
        );
    };

    // Two frames unmeasured: the host pipelines and heaps are made.
    for frame in 0..2 {
        for w in &workers {
            w.record(&shared, frame, draws, &mut Times::default());
        }
        submit();
        let started = Instant::now();
        while crate::pipeline::host_pipeline(graphics).is_none()
            || crate::pipeline::host_pipeline(compute).is_none()
        {
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "the host pipelines are made"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    crate::draw::take_dropped();
    let (drawn, dispatched) = (crate::draw::drawn(), crate::draw::dispatched());

    let start = std::sync::Barrier::new(threads as usize + 1);
    let done = std::sync::Barrier::new(threads as usize + 1);
    let mut wall = Duration::ZERO;
    let results: Vec<(Times, Duration, Duration)> = std::thread::scope(|scope| {
        let handles: Vec<_> = workers
            .iter()
            .map(|w| {
                let (shared, start, done) = (&shared, &start, &done);
                scope.spawn(move || {
                    let mut t = Times::default();
                    let (mut busy, mut cpu) = (Duration::ZERO, Duration::ZERO);
                    for frame in 2..2 + frames {
                        start.wait();
                        let (began, cpu0) = (Instant::now(), thread_cpu());
                        w.record(shared, frame, draws, &mut t);
                        busy += began.elapsed();
                        cpu += thread_cpu() - cpu0;
                        done.wait();
                    }
                    (t, busy, cpu)
                })
            })
            .collect();
        for _ in 0..frames {
            start.wait();
            let began = Instant::now();
            done.wait();
            wall += began.elapsed();
            submit();
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let ran = crate::draw::drawn() - drawn;
    let ran_dispatches = crate::draw::dispatched() - dispatched;
    let dropped = crate::draw::take_dropped();
    let mut all = Times::default();
    for (t, _, _) in &results {
        all.add(t);
    }
    let per_frame = f64::from(threads * frames);
    let calls: u64 = all.calls.iter().sum();
    let spent: Duration = all.spent.iter().sum();
    let busy: Duration = results.iter().map(|r| r.1).sum();
    let cpu: Duration = results.iter().map(|r| r.2).sum();
    // What timing a call costs by itself.
    let mut empty = Times::default();
    for _ in 0..100_000 {
        empty.time(M::Reset, || ());
    }
    let overhead = empty.spent[0].as_nanos() as f64 / 100_000.0;
    println!(
        "list calls: {threads} threads x {:.0} calls a frame, {frames} frames ({} draws and {} dispatches run on the host, {dropped} dropped); timer overhead {overhead:.0} ns a call, not subtracted",
        calls as f64 / per_frame,
        ran,
        ran_dispatches,
    );
    println!(
        "{:<32} {:>10} {:>10} {:>12} {:>7}",
        "method", "calls/thr", "ns/call", "ms/thr/frame", "share"
    );
    let mut order: Vec<usize> = (0..METHODS.len()).filter(|&i| all.calls[i] > 0).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(all.spent[i]));
    for i in order {
        let ns = all.spent[i].as_nanos() as f64;
        println!(
            "{:<32} {:>10.0} {:>10.0} {:>12.3} {:>6.1}%",
            METHODS[i],
            all.calls[i] as f64 / per_frame,
            ns / all.calls[i] as f64,
            ns / per_frame / 1e6,
            100.0 * ns / spent.as_nanos() as f64,
        );
    }
    println!(
        "all calls: {:.0} ns a call, {:.2} ms a thread a frame in calls ({:.2} ms recording, {:.2} ms processor time); {:.2} ms a frame for all threads to record",
        spent.as_nanos() as f64 / calls as f64,
        spent.as_secs_f64() * 1e3 / per_frame,
        busy.as_secs_f64() * 1e3 / per_frame,
        cpu.as_secs_f64() * 1e3 / per_frame,
        wall.as_secs_f64() * 1e3 / f64::from(frames),
    );
    assert_eq!(
        ran,
        u64::from(threads * frames * draws),
        "every draw ran on the host"
    );
}
