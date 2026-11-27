//! What the frame recorder and replayer cost at the size of a title's frame
//! (not run by default: `cargo test --release ... --test replay_big --
//! --ignored --nocapture`): thousands of textures, gigabytes of buffers the
//! CPU maps and views name (as the host layer's copies of title memory
//! are), a view heap of a title's size, 20 submissions of 40,000 calls.
//!
//! `BIG_TEXTURES` (4000, 1.4 MB each), `BIG_SHARED_MB` (4096, in buffers of
//! 16 MB), `BIG_CARD` (300 buffers of 1 MB in the GPU's own memory),
//! `BIG_VIEWS` (2359296) and `BIG_FOLDER` (a temporary one; a capture kept
//! there can be run again with `victory replay`) change it.
#![cfg(windows)]

use std::mem::ManuallyDrop;
use std::time::Instant;

use victory_gpu::replay::play::{self, Options};
use victory_gpu::replay::record::{self, At, Settings};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

const SUBMITS: usize = 20;
const CALLS: usize = 40_000;
const SHARED: u64 = 16 << 20;

fn buffer(device: &ID3D12Device, kind: u32, size: u64) -> (ID3D12Resource, *mut u8) {
    let (heap, state, flags) = match kind {
        0 => (
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ),
        1 => (
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
        ),
        // As `Host::shared_buffer` makes the copies of title memory.
        _ => (
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ),
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation and mapping.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made)
            .unwrap();
        let made = made.unwrap();
        if kind > 0 {
            made.Map(0, None, Some(&mut data)).unwrap();
            // Not all zero, as real data is.
            for at in (0..size as usize).step_by(4096) {
                data.cast::<u8>().add(at).write(at as u8 | 1);
            }
        }
        (made, data.cast())
    }
}

fn barrier(
    res: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer never released.
                pResource: unsafe { std::mem::transmute_copy(res) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    }
}

/// A number from the environment.
fn number(name: &str, otherwise: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(otherwise)
}

struct Big {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    textures: Vec<ID3D12Resource>,
    card: Vec<ID3D12Resource>,
    shared: Vec<(ID3D12Resource, *mut u8)>,
    upload: Vec<(ID3D12Resource, *mut u8)>,
    views: ID3D12DescriptorHeap,
    target: ID3D12Resource,
}

impl Big {
    fn new(device: &ID3D12Device, textures: usize) -> Big {
        let texture = |size: u32, flags, state| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(size),
                Height: size,
                DepthOrArraySize: 1,
                MipLevels: if size == 512 { 10 } else { 1 },
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut made,
                    )
                    .unwrap()
            };
            made.unwrap()
        };
        let views = number("BIG_VIEWS", 2_359_296) as u32;
        // SAFETY: plain creation; slots of the heap just made.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let lists = (0..SUBMITS)
                .map(|_| {
                    let a: ID3D12CommandAllocator = device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .unwrap();
                    let l: ID3D12GraphicsCommandList = device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
                        .unwrap();
                    l.Close().unwrap();
                    (a, l)
                })
                .collect();
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: views,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let textures: Vec<ID3D12Resource> = (0..textures)
                .map(|_| {
                    texture(
                        512,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )
                })
                .collect();
            // A view of each texture, then the rest in runs of one view, as
            // a title's heap mostly holds null views.
            for i in 0..(views as usize).min(textures.len() * 8) {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start + i * step,
                };
                device.CreateShaderResourceView(&textures[i % textures.len()], None, at);
            }
            let shared: Vec<(ID3D12Resource, *mut u8)> =
                (0..(number("BIG_SHARED_MB", 4096) as u64 * (1 << 20) / SHARED).max(1))
                    .map(|_| buffer(device, 2, SHARED))
                    .collect();
            for (i, (b, _)) in shared.iter().enumerate() {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            NumElements: (SHARED / 4) as u32,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                device.CreateShaderResourceView(
                    b,
                    Some(&desc),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: start + i * step,
                    },
                );
            }
            Big {
                device: device.clone(),
                queue,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
                value: 0,
                lists,
                textures,
                card: (0..number("BIG_CARD", 300))
                    .map(|_| buffer(device, 0, 1 << 20).0)
                    .collect(),
                shared,
                upload: (0..500).map(|_| buffer(device, 1, 1 << 20)).collect(),
                views: heap,
                target: texture(
                    1024,
                    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            }
        }
    }

    fn wait(&mut self) {
        self.value += 1;
        // SAFETY: our queue and fence.
        unsafe {
            let event = CreateEventW(None, false, false, None).unwrap();
            self.queue.Signal(&self.fence, self.value).unwrap();
            self.fence.SetEventOnCompletion(self.value, event).unwrap();
            // The fence decides: D3DMetal does not always set the event.
            while self.fence.GetCompletedValue() < self.value {
                WaitForSingleObject(event, 1000);
            }
            let _ = windows::Win32::Foundation::CloseHandle(event);
            self.device.GetDeviceRemovedReason().unwrap();
        }
    }

    fn frame(&mut self, n: usize) {
        if n == 0 {
            // Nothing in the GPU's own memory is all zero, as in a title's:
            // its buffers and every texture's first level are written.
            // SAFETY: a list whose last run has finished; live resources.
            unsafe {
                let (a, l) = &self.lists[0];
                a.Reset().unwrap();
                l.Reset(a, None).unwrap();
                for (i, card) in self.card.iter().enumerate() {
                    l.CopyBufferRegion(card, 0, &self.upload[i % self.upload.len()].0, 0, 1 << 20);
                }
                for (i, t) in self.textures.iter().enumerate() {
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&self.upload[i % self.upload.len()].0),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 0,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: 512,
                                    Height: 500,
                                    Depth: 1,
                                    RowPitch: 2048,
                                },
                            },
                        },
                    };
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(t),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    };
                    l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
                l.Close().unwrap();
                self.queue.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            }
            self.wait();
        }
        for s in 0..SUBMITS {
            // SAFETY: lists whose last run has finished; mappings written
            // inside their bounds.
            unsafe {
                let (a, l) = &self.lists[s];
                a.Reset().unwrap();
                l.Reset(a, None).unwrap();
                l.SetDescriptorHeaps(&[Some(self.views.clone())]);
                for i in 0..CALLS / SUBMITS / 4 {
                    let card = &self.card[(i * 7 + s) % self.card.len()];
                    let from = &self.upload[(i + s * 31) % self.upload.len()].0;
                    l.CopyBufferRegion(card, (i % 1000 * 1000) as u64 & !3, from, 0, 256);
                    l.ResourceBarrier(&[barrier(
                        card,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    l.ResourceBarrier(&[barrier(
                        card,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    l.RSSetScissorRects(&[Default::default()]);
                }
                l.Close().unwrap();
                // What a title's CPU changes before a submission: a page
                // here and there.
                for i in 0..60 {
                    let at = n * 97 + s * 13 + i * 7;
                    let (shared, upload) = (
                        self.shared[at % self.shared.len()].1,
                        self.upload[at % self.upload.len()].1,
                    );
                    shared.add(i * 65536).write((n + s) as u8);
                    upload.add(i * 8192).write((n + s) as u8);
                }
                self.queue.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            }
            self.wait();
        }
        record::present(
            n as u64 + 1,
            Some((&self.target, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)),
            0,
        );
    }
}

#[test]
#[ignore = "takes gigabytes of memory and of the disk"]
fn a_frame_of_a_titles_size() {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: plain creation.
    if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
        eprintln!("skipped: no host GPU");
        return;
    }
    let device = device.unwrap();
    let textures = number("BIG_TEXTURES", 4000);
    let timed = |what: &str, began: Instant| {
        println!("{what}: {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
    };
    let began = Instant::now();
    {
        let mut big = Big::new(&device, textures);
        timed("made, without the recorder", began);
        for n in 0..4 {
            let began = Instant::now();
            big.frame(n);
            timed("frame, without the recorder", began);
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let folder = std::env::var_os("BIG_FOLDER").map_or(dir.path().to_owned(), Into::into);
    let settings = Settings {
        folder,
        at: At::Frame(7),
        frames: 1,
        always: true,
    };
    let folder = record::start(settings, &device).unwrap();
    let began = Instant::now();
    let mut big = Big::new(&device, textures);
    timed("made, with the recorder", began);
    for n in 0..8 {
        let began = Instant::now();
        big.frame(n);
        timed(
            [
                "frame",
                "frame, lists recorded",
                "frame, captured",
                "frame, after",
            ][[1, 1, 1, 1, 1, 1, 2, 3][n]],
            began,
        );
    }
    assert_eq!(record::finished(), Some(folder.clone()));
    drop(big);
    record::unhook(&device);
    println!(
        "{}",
        std::fs::read_to_string(folder.join("notes.txt")).unwrap()
    );
    for name in ["frame.bin", "contents.bin", "deltas.bin", "present.bin", "shaders.bin"] {
        let mut size = 0u64;
        if let Ok(mut f) = std::fs::File::open(folder.join(name)) {
            size = std::io::Seek::seek(&mut f, std::io::SeekFrom::End(0)).unwrap_or(0);
        }
        println!("{:>8} MB {name}", size >> 20);
    }
    let began = Instant::now();
    let reports = play::replay(
        &folder,
        &Options {
            loops: 3,
            ..Default::default()
        },
    )
    .unwrap();
    timed("loaded and replayed", began);
    for (what, r) in &reports {
        println!("{}", r.text(what));
    }
}
