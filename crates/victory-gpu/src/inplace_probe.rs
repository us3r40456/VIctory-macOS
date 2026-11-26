//! A probe of whether the host's D3D12 opens heaps over memory the process
//! already has (`ID3D12Device3::OpenExistingHeapFromAddress`,
//! `ID3D12Device13::OpenExistingHeapFromAddress1` and
//! `OpenExistingHeapFromFileMapping`), and whether a buffer placed on such
//! a heap really is that memory: what the CPU writes the GPU copies out,
//! and what the GPU copies in the CPU reads, with no map or readback.
//! That decides whether the "title memory in place" path of the Windows
//! forks can work under Wine and D3DMetal.
//!
//! Adapted from the avatarmanz fork's `crates/victory-gpu/src/inplace.rs`
//! (its `probe`, `Rig`, `private` and `section`), GPL-3.0 like this
//! repository. This version uses only copies (no `D3DCompile`), tries
//! three sizes and times the openings.
//!
//! Every test here is ignored by default and only prints; none fails on
//! what the host answers. Each check is its own test, so a call that stops
//! the process leaves the others to run on their own. Run them with:
//!
//! ```sh
//! tools/macos/test-wine.sh -p victory-gpu --lib inplace_probe -- --ignored --nocapture --test-threads=1
//! ```

use std::ffi::c_void;
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;
use windows_sys::Win32::Foundation::{GetLastError, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory as mem;

const K64: usize = 64 << 10;
/// The sizes tried: 64 KiB, 4 MiB and 256 MiB, all 64 KiB aligned.
const SIZES: [usize; 3] = [K64, 4 << 20, 256 << 20];

fn host_device() -> Option<ID3D12Device> {
    let device = crate::host::host().map(|h| h.device.clone());
    if device.is_none() {
        eprintln!("INPLACE skipped: no host GPU here");
    }
    device
}

fn line(ok: bool, what: &str, detail: &str) {
    let mark = if ok { "PASS" } else { "FAIL" };
    if detail.is_empty() {
        eprintln!("INPLACE {mark} {what}");
    } else {
        eprintln!("INPLACE {mark} {what}: {detail}");
    }
}

fn code(e: &windows::core::Error) -> String {
    format!("hr={:#010x} ({})", e.code().0 as u32, e.message().trim())
}

fn report<T>(what: &str, result: &windows::core::Result<T>) {
    match result {
        Ok(_) => line(true, what, ""),
        Err(e) => line(false, what, &code(e)),
    }
}

fn size_name(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{} MiB", bytes >> 20)
    } else {
        format!("{} KiB", bytes >> 10)
    }
}

/// What a heap is opened on.
#[derive(Clone, Copy)]
enum From {
    /// A whole `VirtualAlloc` allocation, by its base address.
    Region(usize),
    /// A committed range of one: address and bytes.
    Part(usize, usize),
    /// A pagefile-backed section, by its handle.
    Section(usize),
}

impl From {
    fn name(self) -> &'static str {
        match self {
            From::Region(_) => "OpenExistingHeapFromAddress",
            From::Part(..) => "OpenExistingHeapFromAddress1",
            From::Section(_) => "OpenExistingHeapFromFileMapping",
        }
    }
}

fn open(device: &ID3D12Device, from: From) -> windows::core::Result<ID3D12Heap> {
    // SAFETY: the calls take an address or a handle of the caller's and
    // check both themselves; a refusal is an error code.
    unsafe {
        match from {
            From::Region(at) => device
                .cast::<ID3D12Device3>()?
                .OpenExistingHeapFromAddress(at as *const c_void),
            From::Part(at, bytes) => device
                .cast::<ID3D12Device13>()?
                .OpenExistingHeapFromAddress1(at as *const c_void, bytes),
            From::Section(handle) => device
                .cast::<ID3D12Device3>()?
                .OpenExistingHeapFromFileMapping(HANDLE(handle as *mut c_void)),
        }
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
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
    }
}

/// A buffer over all of `heap`. Windows wants `ALLOW_CROSS_ADAPTER` on a
/// buffer in an opened heap (the heap is cross-adapter shared); if the
/// host refuses that, the plain buffer is tried too. The flag set used is
/// returned with it.
fn place(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    size: u64,
) -> (windows::core::Result<ID3D12Resource>, &'static str) {
    let tries = [
        (
            D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            "cross-adapter+UAV",
        ),
        (D3D12_RESOURCE_FLAG_NONE, "no flags"),
    ];
    let mut last = Err(windows::core::Error::empty());
    for (flags, name) in tries {
        let mut placed: Option<ID3D12Resource> = None;
        // SAFETY: plain creation in a live heap, inside its size.
        let made = unsafe {
            device.CreatePlacedResource(
                heap,
                0,
                &buffer_desc(size, flags),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut placed,
            )
        };
        match made.and_then(|()| placed.ok_or_else(windows::core::Error::empty)) {
            Ok(buffer) => return (Ok(buffer), name),
            Err(e) => last = Err(e),
        }
    }
    (last, "any flags")
}

/// A queue, list and fence of the probe's own (adapted from the fork's `Rig`).
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                done: 0,
                event,
            })
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    fn run(
        &mut self,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    fn committed(
        &self,
        kind: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE),
                state,
                None,
                &mut buffer,
            )?
        };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// What a GPU copy reads of `size` bytes of `resource` from `offset`.
    fn fetch(
        &mut self,
        resource: &ID3D12Resource,
        offset: u64,
        size: usize,
    ) -> windows::core::Result<Vec<u8>> {
        let back = self.committed(
            D3D12_HEAP_TYPE_READBACK,
            size as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
            back.Unmap(0, None);
            Ok(bytes)
        }
    }

    /// A GPU copy writes `bytes` at `offset` of `resource`, from an upload buffer.
    fn store(
        &mut self,
        resource: &ID3D12Resource,
        offset: u64,
        bytes: &[u8],
    ) -> windows::core::Result<()> {
        let up = self.committed(
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            up.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe {
            list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64)
        })
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed) ^ (i >> 8) as u8)
        .collect()
}

/// # Safety
/// `at` must be writable for `bytes.len()` bytes.
unsafe fn write(at: *mut u8, bytes: &[u8]) {
    // SAFETY: the caller's.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len()) };
}

/// # Safety
/// `at` must be readable for `bytes.len()` bytes.
unsafe fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the caller's.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

fn first_difference(got: &[u8], want: &[u8]) -> String {
    match got.iter().zip(want).position(|(a, b)| a != b) {
        Some(i) => format!(
            "first difference at byte {i}: got {:#04x}, want {:#04x}",
            got[i], want[i]
        ),
        None => "lengths differ".to_owned(),
    }
}

/// The aliasing checks on a buffer placed over `bytes` bytes at `memory`
/// (the CPU's address of the heap's offset 0). `pre` is what the CPU wrote
/// to the first 64 KiB before the heap was opened.
///
/// # Safety
/// `memory` must be the process's own read-write memory of `bytes` bytes.
unsafe fn alias(
    rig: &mut Rig,
    tag: &str,
    buffer: &ID3D12Resource,
    memory: *mut u8,
    bytes: usize,
    pre: &[u8],
) {
    let n = K64.min(bytes);
    // 1. What the CPU wrote before the heap was opened.
    match rig.fetch(buffer, 0, n) {
        Ok(got) => line(
            got == pre,
            &format!("{tag}: GPU copy sees CPU bytes written before open"),
            &if got == pre {
                String::new()
            } else {
                first_difference(&got, pre)
            },
        ),
        Err(e) => line(
            false,
            &format!("{tag}: GPU copy sees CPU bytes written before open"),
            &code(&e),
        ),
    }
    // 2. What the CPU writes after the heap and buffer were made, with no
    // call in between: true aliasing, not a snapshot taken at open. Both
    // the first and the last 64 KiB.
    for (where_, offset) in [("start", 0), ("end", bytes - n)] {
        let after = pattern(n, 0x77 ^ offset as u8 ^ (offset >> 16) as u8);
        // SAFETY: inside the caller's memory.
        unsafe { write(memory.add(offset), &after) };
        let what = format!(
            "{tag}: GPU copy sees CPU write made after open ({where_}, offset {offset:#x})"
        );
        match rig.fetch(buffer, offset as u64, n) {
            Ok(got) if got == after => line(true, &what, ""),
            Ok(got) if got == pre && offset == 0 => line(
                false,
                &what,
                "GPU still sees the bytes from before open (snapshot)",
            ),
            Ok(got) => line(false, &what, &first_difference(&got, &after)),
            Err(e) => line(false, &what, &code(&e)),
        }
    }
    // 3. What a GPU copy writes into the placed buffer, read at the
    // memory's own address with no map or readback.
    for (where_, offset) in [("start", 0), ("end", bytes - n)] {
        let new = pattern(n, 0x5a ^ (offset >> 16) as u8);
        let what =
            format!("{tag}: CPU sees GPU copy into placed buffer at original address ({where_})");
        match rig.store(buffer, offset as u64, &new) {
            Ok(()) => {
                // SAFETY: inside the caller's memory.
                let ok = unsafe { same(memory.add(offset), &new) };
                let detail = if ok {
                    String::new()
                } else {
                    // SAFETY: inside the caller's memory.
                    first_difference(
                        unsafe { std::slice::from_raw_parts(memory.add(offset), n) },
                        &new,
                    )
                };
                line(ok, &what, &detail);
            }
            Err(e) => line(false, &what, &code(&e)),
        }
    }
}

/// `VirtualProtect` of one 64 KiB page at `page` while the heap stands:
/// read-only and back.
///
/// # Safety
/// `page` must be the process's own committed memory of 64 KiB.
unsafe fn protect(tag: &str, page: *mut u8) {
    let mut old = 0;
    // SAFETY: the caller's memory.
    let (ro, err) = unsafe {
        let ok = mem::VirtualProtect(page.cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
        (ok, if ok { 0 } else { GetLastError() })
    };
    let mut back = true;
    if ro {
        // SAFETY: the caller's memory.
        back = unsafe { mem::VirtualProtect(page.cast(), K64, old, &mut old) } != 0;
    }
    line(
        ro && back,
        &format!("{tag}: VirtualProtect(PAGE_READONLY) inside the heap while it is open"),
        &if ro {
            if back {
                String::new()
            } else {
                "restoring failed".to_owned()
            }
        } else {
            format!("error {err}")
        },
    );
}

/// Opens a heap on fresh private memory of `bytes` bytes, both whole
/// (`OpenExistingHeapFromAddress`) and with an explicit size
/// (`OpenExistingHeapFromAddress1`), and runs the checks on each.
fn private(device: &ID3D12Device, explicit: bool) {
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return line(false, "a queue of the probe's own", &code(&e)),
    };
    for bytes in SIZES {
        // SAFETY: memory of our own, released below; every address used lies inside it.
        unsafe {
            let base = mem::VirtualAlloc(
                std::ptr::null(),
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
            .cast::<u8>();
            if base.is_null() {
                line(
                    false,
                    &format!("VirtualAlloc {}", size_name(bytes)),
                    &format!("error {}", GetLastError()),
                );
                continue;
            }
            let pre = pattern(K64.min(bytes), 0x11);
            write(base, &pre);
            let from = if explicit {
                From::Part(base as usize, bytes)
            } else {
                From::Region(base as usize)
            };
            let tag = format!("{} {}", from.name(), size_name(bytes));
            let heap = open(device, from);
            report(&format!("{tag} on VirtualAlloc'd committed memory"), &heap);
            if let Ok(heap) = &heap {
                let d = heap.GetDesc();
                eprintln!(
                    "INPLACE      heap desc: {} bytes, type {}, cpu page {}, pool {}, flags {:#x}",
                    d.SizeInBytes,
                    d.Properties.Type.0,
                    d.Properties.CPUPageProperty.0,
                    d.Properties.MemoryPoolPreference.0,
                    d.Flags.0
                );
                let (placed, flags) = place(device, heap, bytes as u64);
                report(
                    &format!("{tag}: CreatePlacedResource buffer ({flags})"),
                    &placed,
                );
                if let Ok(buffer) = &placed {
                    alias(&mut rig, &tag, buffer, base, bytes, &pre);
                }
                protect(&tag, base);
                drop(placed);
            }
            drop(heap);
            let freed = mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE) != 0;
            if !freed {
                line(
                    false,
                    &format!("{tag}: VirtualFree after the heap went"),
                    &format!("error {}", GetLastError()),
                );
            }
        }
    }
    // A part inside a bigger allocation (what the fork uses in place of
    // whole allocations): 16 pages from the 16th of 64.
    if explicit {
        // SAFETY: as above.
        unsafe {
            let size = 64 * K64;
            let base = mem::VirtualAlloc(
                std::ptr::null(),
                size,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
            .cast::<u8>();
            if base.is_null() {
                return;
            }
            let (at, bytes) = (base.add(16 * K64), 16 * K64);
            let pre = pattern(K64, 0x22);
            write(at, &pre);
            let tag = "OpenExistingHeapFromAddress1 1 MiB part at +1 MiB of a 4 MiB allocation"
                .to_owned();
            let heap = open(device, From::Part(at as usize, bytes));
            report(&tag, &heap);
            if let Ok(heap) = &heap {
                let (placed, flags) = place(device, heap, bytes as u64);
                report(
                    &format!("{tag}: CreatePlacedResource buffer ({flags})"),
                    &placed,
                );
                if let Ok(buffer) = &placed {
                    alias(&mut rig, &tag, buffer, at, bytes, &pre);
                }
            }
            drop(heap);
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }
}

/// A pagefile-backed section (`SEC_COMMIT`) of each size, mapped with
/// `MapViewOfFile`, opened with `OpenExistingHeapFromFileMapping`.
fn section(device: &ID3D12Device) {
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return line(false, "a queue of the probe's own", &code(&e)),
    };
    for bytes in SIZES {
        let tag = format!("OpenExistingHeapFromFileMapping {}", size_name(bytes));
        // SAFETY: a section and view of our own, closed below.
        unsafe {
            let handle = mem::CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_COMMIT,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            if handle.is_null() {
                line(
                    false,
                    &format!("CreateFileMapping {}", size_name(bytes)),
                    &format!("error {}", GetLastError()),
                );
                continue;
            }
            let view = mem::MapViewOfFile(
                handle,
                mem::FILE_MAP_WRITE | mem::FILE_MAP_READ,
                0,
                0,
                bytes,
            );
            let memory = view.Value.cast::<u8>();
            if memory.is_null() {
                line(
                    false,
                    &format!("MapViewOfFile {}", size_name(bytes)),
                    &format!("error {}", GetLastError()),
                );
                CloseHandle(HANDLE(handle)).ok();
                continue;
            }
            let pre = pattern(K64.min(bytes), 0x33);
            write(memory, &pre);
            let heap = open(device, From::Section(handle as usize));
            report(&format!("{tag} on a pagefile-backed section"), &heap);
            if let Ok(heap) = &heap {
                let d = heap.GetDesc();
                eprintln!(
                    "INPLACE      heap desc: {} bytes, type {}, cpu page {}, pool {}, flags {:#x}",
                    d.SizeInBytes,
                    d.Properties.Type.0,
                    d.Properties.CPUPageProperty.0,
                    d.Properties.MemoryPoolPreference.0,
                    d.Flags.0
                );
                let (placed, flags) = place(device, heap, bytes as u64);
                report(
                    &format!("{tag}: CreatePlacedResource buffer ({flags})"),
                    &placed,
                );
                if let Ok(buffer) = &placed {
                    alias(&mut rig, &tag, buffer, memory, bytes, &pre);
                }
                protect(&tag, memory);
                drop(placed);
            }
            drop(heap);
            mem::UnmapViewOfFile(view);
            CloseHandle(HANDLE(handle)).ok();
        }
    }
}

/// Mean milliseconds per call of opening a heap and of placing a buffer
/// over all of it, on fresh untouched memory each time.
fn timing(device: &ID3D12Device) {
    for bytes in SIZES {
        let runs = if bytes >= 256 << 20 { 3 } else { 10 };
        for kind in 0..3 {
            let (mut open_ms, mut place_ms, mut n, mut failed) = (0.0, 0.0, 0, None);
            let mut name = "";
            for _ in 0..runs {
                // SAFETY: memory and sections of our own, released in the loop.
                unsafe {
                    let (from, release): (From, Box<dyn FnOnce()>) = match kind {
                        0 | 1 => {
                            let base = mem::VirtualAlloc(
                                std::ptr::null(),
                                bytes,
                                mem::MEM_RESERVE | mem::MEM_COMMIT,
                                mem::PAGE_READWRITE,
                            );
                            if base.is_null() {
                                break;
                            }
                            let from = if kind == 0 {
                                From::Region(base as usize)
                            } else {
                                From::Part(base as usize, bytes)
                            };
                            (
                                from,
                                Box::new(move || {
                                    mem::VirtualFree(base, 0, mem::MEM_RELEASE);
                                }),
                            )
                        }
                        _ => {
                            let handle = mem::CreateFileMappingW(
                                INVALID_HANDLE_VALUE,
                                std::ptr::null(),
                                mem::PAGE_READWRITE | mem::SEC_COMMIT,
                                0,
                                bytes as u32,
                                std::ptr::null(),
                            );
                            if handle.is_null() {
                                break;
                            }
                            (
                                From::Section(handle as usize),
                                Box::new(move || {
                                    CloseHandle(HANDLE(handle)).ok();
                                }),
                            )
                        }
                    };
                    name = from.name();
                    let started = Instant::now();
                    let heap = open(device, from);
                    let opened = started.elapsed().as_secs_f64() * 1000.0;
                    match &heap {
                        Ok(heap) => {
                            let started = Instant::now();
                            let (placed, _) = place(device, heap, bytes as u64);
                            let took = started.elapsed().as_secs_f64() * 1000.0;
                            match placed {
                                Ok(_) => {
                                    open_ms += opened;
                                    place_ms += took;
                                    n += 1;
                                }
                                Err(e) => failed = Some(format!("place {}", code(&e))),
                            }
                        }
                        Err(e) => failed = Some(format!("open {}", code(e))),
                    }
                    drop(heap);
                    release();
                }
                if failed.is_some() {
                    break;
                }
            }
            let what = format!("timing {name} {}", size_name(bytes));
            match failed {
                Some(why) => line(false, &what, &why),
                None if n > 0 => line(
                    true,
                    &what,
                    &format!(
                        "open {:.3} ms/call, CreatePlacedResource {:.3} ms/call ({n} runs)",
                        open_ms / n as f64,
                        place_ms / n as f64
                    ),
                ),
                None => line(false, &what, "no memory to try with"),
            }
        }
    }
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn inplace_probe_1_interfaces() {
    let Some(device) = host_device() else { return };
    report(
        "QueryInterface ID3D12Device3",
        &device.cast::<ID3D12Device3>(),
    );
    report(
        "QueryInterface ID3D12Device13",
        &device.cast::<ID3D12Device13>(),
    );
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn inplace_probe_2_address() {
    let Some(device) = host_device() else { return };
    private(&device, false);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn inplace_probe_3_address1() {
    let Some(device) = host_device() else { return };
    private(&device, true);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn inplace_probe_4_file_mapping() {
    let Some(device) = host_device() else { return };
    section(&device);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn inplace_probe_5_timing() {
    let Some(device) = host_device() else { return };
    timing(&device);
}
