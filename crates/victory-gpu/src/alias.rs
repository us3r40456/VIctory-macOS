//! Zero-copy title buffers on Apple Silicon through D3DMetal (`mach_vm_remap`
//! aliasing, issue #204).
//!
//! D3DMetal refuses `OpenExistingHeapFromAddress*`, so VIctory normally keeps
//! host copies of title buffers (`crate::shadow`) and copies pages back and
//! forth. On Apple Silicon (unified memory), D3DMetal's own mapped write-back
//! L0 buffers are backed by the same physical RAM. Under Wine + Rosetta 2, raw
//! `mach_msg2_trap` MIG calls (`mach_vm_remap` with `copy = FALSE`,
//! `VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) point the title's `VirtualAlloc`
//! virtual address range directly at the D3DMetal buffer's physical pages.
//!
//! On by default on Rosetta + D3DMetal (never WARP); `VICTORY_ALIAS_BUFFERS=0`
//! keeps the host copies. The probe at start-up ([`supported`]) still falls
//! back to the copies where the remap does not work. (mac: unset is off on
//! this line, [`UNSET_HERE`].)
//!
//! mac: GPU writes into aliased memory mark the
//! other trackers over it without opening their guards
//! (`crate::watch::notify_written`), and only where the writes were not
//! marked; a commit that brings back memory the title released or
//! decommitted through the host puts its aliases right
//! (`crate::alias_plan`); and the frame line counts both paths
//! ([`frame_line`]); it needs `VICTORY_COPY_ORDER=split`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};

use crate::host::host;
use crate::watch::PAGE;

#[cfg(any(test, feature = "diagnostics"))]
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(target_arch = "x86_64")]
mod mach {
    use core::arch::asm;
    use std::sync::{Mutex, OnceLock};

    const MACH: u64 = 0x0100_0000;
    const TRAP_REPLY_PORT: u64 = 26;
    const TRAP_TASK_SELF: u64 = 28;
    /// `mach_msg2_trap` (8 arguments). `mach_msg_trap` (31) is fatal on
    /// macOS 27+ for kernel-object calls (`EXC_GUARD`).
    const TRAP_MSG2: u64 = 47;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;

    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    unsafe fn trap(n: u64, a: [u64; 6]) -> u64 {
        // SAFETY: forwarded to `trap8`.
        unsafe { trap8(n, a, 0, 0) }
    }

    unsafe fn trap8(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
        let ret: u64;
        // SAFETY: x86-64 XNU Mach trap ABI under Rosetta 2; 7th and 8th args
        // sit at `[rsp + 8]` and `[rsp + 16]`.
        unsafe {
            asm!(
                "sub rsp, 32",
                "mov qword ptr [rsp], 0",
                "mov qword ptr [rsp + 8], {a7}",
                "mov qword ptr [rsp + 16], {a8}",
                "syscall",
                "add rsp, 32",
                inlateout("rax") MACH | n => ret,
                in("rdi") a[0],
                in("rsi") a[1],
                in("rdx") a[2],
                in("r10") a[3],
                in("r8") a[4],
                in("r9") a[5],
                a7 = in(reg) a7,
                a8 = in(reg) a8,
                lateout("rcx") _,
                lateout("r11") _,
            );
        }
        ret
    }

    pub fn task_self() -> u32 {
        // SAFETY: argument-less Mach trap.
        unsafe { trap(TRAP_TASK_SELF, [0; 6]) as u32 }
    }

    fn reply_port() -> u32 {
        static PORT: OnceLock<u32> = OnceLock::new();
        *PORT.get_or_init(|| {
            // SAFETY: argument-less Mach trap.
            unsafe { trap(TRAP_REPLY_PORT, [0; 6]) as u32 }
        })
    }

    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct Header {
        bits: u32,
        size: u32,
        remote: u32,
        local: u32,
        voucher: u32,
        id: i32,
    }

    /// `mach_vm_remap` MIG request (`msgh_id = 4813`, `#pragma pack(4)`).
    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct RemapRequest {
        head: Header,
        descriptors: u32,
        src_task: u32,
        pad1: u32,
        disp_type: u32,
        ndr: [u8; 8],
        target_address: u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_address: u64,
        copy: u32,
        inheritance: u32,
    }

    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    struct RemapReply {
        head: Header,
        ndr: [u8; 8],
        ret: i32,
        target_address: u64,
        cur_protection: i32,
        max_protection: i32,
        trailer: [u8; 128],
    }

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    static REMAP_LOCK: Mutex<()> = Mutex::new(());

    /// Maps `[src, src + size)` of this task at `target`, sharing physical
    /// pages (`copy = FALSE`).
    ///
    /// # Safety
    /// Replaces whatever is mapped at `target`; both ranges must be page
    /// aligned and owned by the caller.
    pub unsafe fn remap_shared(target: u64, src: u64, size: u64, flags: i32) -> Remapped {
        let _guard = REMAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let task = task_self();
        let reply = reply_port();
        let mut msg = [0u8; size_of::<RemapReply>()];
        let req = RemapRequest {
            head: Header {
                // MACH_MSGH_BITS(COPY_SEND=19, MAKE_SEND_ONCE=21) | MACH_MSGH_BITS_COMPLEX
                bits: 19 | (21 << 8) | 0x8000_0000,
                size: size_of::<RemapRequest>() as u32,
                remote: task,
                local: reply,
                voucher: 0,
                id: 4813,
            },
            descriptors: 1,
            src_task: task,
            pad1: 0,
            disp_type: 19 << 16,
            ndr: [0, 0, 0, 0, 1, 0, 0, 0],
            target_address: target,
            size,
            mask: 0,
            flags,
            src_address: src,
            copy: 0,
            inheritance: 1,
        };
        // SAFETY: `req` and `msg` are plain-old-data structs of known size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const req).cast::<u8>(),
                msg.as_mut_ptr(),
                size_of::<RemapRequest>(),
            );
        }
        let pack = |lo: u32, hi: u32| (u64::from(hi) << 32) | u64::from(lo);
        let args = [
            msg.as_mut_ptr() as u64,
            3 | MACH64_SEND_KOBJECT_CALL,
            pack(req.head.bits, size_of::<RemapRequest>() as u32),
            pack(task, reply),
            pack(0, 4813),
            pack(1, reply),
        ];
        // SAFETY: `msg` holds the request and space for the reply and trailer.
        let r = unsafe { trap8(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
        // SAFETY: the kernel wrote at most `msg.len()` bytes into `msg`.
        let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
        Remapped {
            msg: r,
            kr: rep.ret,
            address: rep.target_address,
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod mach {
    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    pub fn task_self() -> u32 {
        0
    }

    pub unsafe fn remap_shared(_target: u64, _src: u64, _size: u64, _flags: i32) -> Remapped {
        Remapped {
            msg: 1,
            kr: -1,
            address: 0,
        }
    }
}

/// `asked`: `Some(on)` when `VICTORY_ALIAS_BUFFERS` is set (`0` is off, any
/// other value on), `None` when it is not: then on.
fn decide_enabled(asked: Option<bool>, rosetta: bool, warp: bool) -> bool {
    asked.unwrap_or(true) && rosetta && !warp
}

/// mac: what an unset `VICTORY_ALIAS_BUFFERS` asks for on this line: off.
/// VIctory-macOS has it on (#322, [`decide_enabled`]). Here the line runs
/// with fault tracking and its copy-order split, which the aliasing as
/// merged does not follow (a title copy lands in aliased memory before the
/// commands recorded ahead of it have run; an alias the title freed through
/// the host stays registered): the adaptation is prepared apart (branch
/// `prep24/zero-copy`) and the default waits for it and for an A/B run.
/// `VICTORY_ALIAS_BUFFERS=1` turns the aliasing on as the Mac port has it.
const UNSET_HERE: bool = false;

/// Whether title buffers are aliased on this host: Rosetta + D3DMetal, with
/// `VICTORY_ALIAS_BUFFERS=1` (mac: unset is off here, [`UNSET_HERE`]), and
/// the copy-order split on (mac). Asked first by the title's first
/// `XMemVirtualAlloc`, which prints the switch's one line.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_ALIAS_BUFFERS")
            .map(|_| crate::switch::on("VICTORY_ALIAS_BUFFERS"))
            .or(Some(UNSET_HERE));
        let wanted = asked == Some(true);
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        let rosetta = crate::watch::rosetta();
        // Title copies are made on the CPU while their submission is
        // prepared. Into aliased memory, every command of that submission
        // reads their bytes, also the commands before them, unless the
        // copy-order split (`VICTORY_COPY_ORDER=split`) has run those first.
        let split = crate::mac_switches::copy_order_split();
        // With flights the next submission is prepared while the one left
        // on the GPU still writes: its texture uploads can read aliased
        // memory those shaders are writing, which host copies only see when
        // the copy-back lands (`crate::host::unwaited::flights`).
        let flights = crate::host::unwaited::flights::on();
        let on = decide_enabled(asked, rosetta, warp) && split.is_some() && !flights;
        if on {
            tracing::info!(
                "VICTORY_ALIAS_BUFFERS: read-write title memory committed in 64 KiB pages shares its pages with host \
                 buffers (mach_vm_remap, VIctory-macOS #207, #322): title buffers there are neither copied nor copied \
                 back; shaders' writes there mark the other trackers without opening guards (fault tracking {}); \
                 writes the runtime withholds land in title memory there all the same; counters at the end of the \
                 frame line",
                if crate::watch::fault_tracking() { "on" } else { "off" }
            );
            if split != Some(usize::MAX) {
                tracing::warn!(
                    "VICTORY_ALIAS_BUFFERS with VICTORY_COPY_ORDER=split:<bytes>: a title copy into aliased memory that \
                     earlier commands read through a longer range is not cut, and those commands read its new bytes"
                );
            }
        } else if wanted {
            tracing::warn!(
                "VICTORY_ALIAS_BUFFERS refused: {}; title buffers keep their host copies",
                if !rosetta {
                    "not under Rosetta"
                } else if warp {
                    "WARP draws in software, its buffers are not the GPU's memory"
                } else if flights {
                    "VICTORY_FLIGHTS is on: the next submission is prepared while the one on the GPU still writes \
                     aliased memory, and its texture uploads could read those writes half done"
                } else {
                    "VICTORY_COPY_ORDER is not split: without its cuts every command of a submission would read the \
                     bytes of the submission's last title copy into aliased memory, also the commands before the copy"
                }
            );
        }
        on
    })
}

/// Counts since the last frame line, by [`C`].
static COUNTS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

/// What [`COUNTS`] holds.
#[derive(Clone, Copy)]
enum C {
    /// Bytes of aliased memory submissions asked to have up to date, each
    /// byte once per submission, and the ranges and regions they make.
    AskedBytes,
    AskedRanges,
    AskedRegions,
    /// Host copies (not aliased) the same submissions brought up to date,
    /// and the bytes copied into them.
    Copies,
    CopiedBytes,
    /// Bytes of aliased memory shaders wrote (told to the other trackers),
    /// the tracked pages that marked, and the bytes of withheld writes.
    WrittenBytes,
    MarkedPages,
    WithheldBytes,
    /// Commits that came back over a stale or decommitted alias.
    Mended,
    /// Title copies into aliased memory, their bytes, and those of them
    /// that open a host list the copy-order split cut ([`title_copy`]).
    CopiesIn,
    CopiedIn,
    CopiesCut,
}

/// Since the start: aliases made, refused by the remap, found stale, and
/// runs pointed back at their buffers.
static TOTALS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn add(c: C, n: u64) {
    COUNTS[c as usize].fetch_add(n, Ordering::Relaxed);
}

/// A submission asked for `bytes` of aliased memory in `ranges` ranges of
/// `regions` regions, and brought `copies` host copies up to date with
/// `copied` bytes (`crate::host`'s batch).
pub(crate) fn asked(regions: usize, ranges: usize, bytes: usize, copies: usize, copied: usize) {
    if !enabled() {
        return;
    }
    add(C::AskedRegions, regions as u64);
    add(C::AskedRanges, ranges as u64);
    add(C::AskedBytes, bytes as u64);
    add(C::Copies, copies as u64);
    add(C::CopiedBytes, copied as u64);
}

/// Shaders wrote `bytes` of aliased memory, which marked `pages` tracked
/// pages; `withheld`: writes that were to stay out of title memory.
pub(crate) fn noted(bytes: usize, pages: usize, withheld: bool) {
    add(C::WrittenBytes, bytes as u64);
    add(C::MarkedPages, pages as u64);
    if withheld {
        add(C::WithheldBytes, bytes as u64);
    }
}

/// A title copy (`CopyBufferRegion`, buffer `CopyResource`,
/// `CopyMemoryToMemoryX`) of `len` bytes into `dst`, made on the CPU while
/// its submission is prepared (`crate::host`, `Step::Copy`): counted where it
/// lands in aliased memory, and `cut` where it opens a host list the
/// copy-order split cut (`VICTORY_COPY_ORDER=split`). Such a copy is made
/// only once the part before it has run on the GPU, so the commands before
/// it read the old bytes in aliased memory as in a host copy. A copy the
/// split did not cut changes what every command of its submission reads,
/// also those before it, where they read its bytes through what the split
/// does not see (the view heap itself, title memory's page table).
pub(crate) fn title_copy(dst: usize, len: usize, cut: bool) {
    if enabled() && !crate::shadow::aliased_between(dst as u64, dst.saturating_add(len) as u64).is_empty() {
        add(C::CopiesIn, 1);
        add(C::CopiedIn, len as u64);
        if cut {
            add(C::CopiesCut, 1);
        }
    }
}

/// The end of the frame line (`frames` since the last one), empty with the
/// switch off: per frame, what took the zero-copy path against the copy path.
pub(crate) fn frame_line(frames: f64) -> String {
    if !enabled() {
        return String::new();
    }
    let [asked, ranges, regions, copies, copied, written, marked, withheld, mended, copies_in, copied_in, copies_cut] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map(|i| COUNTS[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let [made, failed, stale, again] = [0, 1, 2, 3].map(|i| TOTALS[i].load(Ordering::Relaxed));
    let (now, bytes) = crate::shadow::aliased_totals();
    format!(
        "; zero-copy: {:.1} MB in {ranges:.0} ranges of {regions:.0} aliased regions asked, none copied, against \
         {copies:.0} host copies refreshed with {:.1} MB copied; shaders wrote {:.1} MB in place, marking {marked:.0} \
         tracked pages, {:.1} MB of it withheld writes; {copies_in:.0} title copies ({:.1} KB) into aliased memory, \
         {copies_cut:.0} of them after a copy-order cut; {mended:.1} commits mended; aliased now {now} regions \
         {:.0} MB ({made} made, {failed} refused, {stale} stale dropped, {again} runs pointed back)",
        asked / 1e6,
        copied / 1e6,
        written / 1e6,
        withheld / 1e6,
        copied_in / 1e3,
        bytes as f64 / 1e6,
    )
}

/// The runs of `[at, at + len)` not committed now, for a commit at a given
/// address (asked before it, [`committed`]).
pub fn uncommitted(at: usize, len: usize) -> Vec<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = at.saturating_add(len);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut p = at;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) } == 0 {
            runs.push((p, end));
            break;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).clamp(p + 1, end);
        if info.State != MEM_COMMIT {
            match runs.last_mut() {
                Some((_, e)) if *e == p => *e = next,
                _ => runs.push((p, next)),
            }
        }
        p = next;
    }
    runs
}

/// After a successful `XMemVirtualAlloc` of `[at, at + len)` (switch on):
/// `fresh` for a new reservation, `eligible` for read-write memory committed
/// in whole 64 KiB pages (not a placeholder), `before` what of a commit at a
/// given address was not committed before it ([`uncommitted`]). Drops aliases
/// the reservation shows stale, points recommitted runs of aliases back at
/// their buffers, and aliases fresh memory (`crate::alias_plan`).
pub fn committed(at: usize, len: usize, fresh: bool, eligible: bool, before: Option<&[(usize, usize)]>) {
    let end = at.saturating_add(len);
    let whole = [(at, end)];
    let uncommitted = if fresh { &whole[..] } else { before.unwrap_or(&[]) };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let aliases: Vec<(usize, usize)> = crate::shadow::aliased_between(at as u64, end as u64)
        .into_iter()
        .map(|(s, e)| (s as usize, e as usize))
        .collect();
    let plan = crate::alias_plan::plan(at, len, fresh, eligible, uncommitted, &aliases);
    if !plan.stale.is_empty() || !plan.again.is_empty() {
        add(C::Mended, 1);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                at = format_args!("{at:#x}"),
                len,
                stale = plan.stale.len(),
                again = plan.again.len(),
                "VICTORY_ALIAS_BUFFERS: the title released or decommitted aliased memory through the host and committed \
                 it again; its aliases were put right (counted in the frame line, logged once)"
            );
        });
    }
    for start in plan.stale {
        if crate::shadow::unregister_aliased(start as u64) {
            TOTALS[2].fetch_add(1, Ordering::Relaxed);
        }
    }
    for (from, to) in plan.again {
        if point_back(from, to) {
            TOTALS[3].fetch_add(1, Ordering::Relaxed);
        }
    }
    if plan.new {
        alias_locked(at, len, false);
    }
}

/// Points the recommitted run `[from, to)` of an aliased region back at the
/// region's buffer, zeroed (as recommitted memory reads). The run was not
/// committed before, so no title buffer in use is there.
fn point_back(from: usize, to: usize) -> bool {
    let Some(shadow) = crate::shadow::find_aliased(from as u64, to as u64) else {
        return false;
    };
    let at = shadow.host_ptr(from as u64);
    // SAFETY: `[from, to)` is inside the region, so `at` has `to - from`
    // bytes of its mapping; both are 64 KiB aligned (`alias_plan`).
    let r = unsafe {
        std::ptr::write_bytes(at, 0, to - from);
        mach::remap_shared(
            from as u64,
            at as u64,
            (to - from) as u64,
            mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
        )
    };
    if r.msg != 0 || r.kr != 0 || r.address != from as u64 {
        tracing::warn!(
            from = format_args!("{from:#x}"),
            to = format_args!("{to:#x}"),
            ?r,
            "mach_vm_remap of a recommitted run failed: its title buffers would read stale bytes"
        );
        return false;
    }
    crate::watch::title_memory_remapped(from, to - from);
    true
}

/// Serializes aliasing and its registry changes.
static LOCK: Mutex<()> = Mutex::new(());

/// Probes once whether `mach_vm_remap` onto a D3DMetal write-back L0 buffer
/// works and whether fresh buffers arrive zero-initialized.
fn supported() -> Option<bool> {
    static PROBE: OnceLock<Option<bool>> = OnceLock::new();
    *PROBE.get_or_init(|| {
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        if !crate::watch::rosetta() || warp || mach::task_self() == 0 {
            return None;
        }
        let h = host()?;
        let (buffer, gpu_mem) = h.shared_buffer(PAGE as u64)?;
        // SAFETY: `gpu_mem` is our `PAGE`-byte mapped buffer.
        let zeroed = unsafe {
            gpu_mem.cast::<u64>().read_volatile() == 0
                && gpu_mem.add(PAGE - 8).cast::<u64>().read_volatile() == 0
        };
        // SAFETY: standalone `PAGE`-byte allocation freed below.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        if title.is_null() {
            return None;
        }
        // SAFETY: `title` and `gpu_mem` are both `PAGE`-aligned and `PAGE` bytes.
        let r = unsafe {
            mach::remap_shared(
                title as u64,
                gpu_mem as u64,
                PAGE as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        let ok = if r.msg == 0 && r.kr == 0 && r.address == title as u64 {
            // SAFETY: both pointers point to the same `PAGE` of RAM.
            unsafe {
                title.add(128).write_volatile(0x5a);
                let seen_gpu = gpu_mem.add(128).read_volatile() == 0x5a;
                gpu_mem.add(256).write_volatile(0xa5);
                let seen_title = title.add(256).read_volatile() == 0xa5;
                seen_gpu && seen_title
            }
        } else {
            false
        };
        // SAFETY: `title` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
        drop(buffer);
        if ok {
            tracing::info!(
                zeroed,
                "zero-copy title buffer aliasing active (mach_vm_remap)"
            );
            Some(zeroed)
        } else {
            tracing::warn!(
                ?r,
                "mach_vm_remap aliasing probe failed; falling back to shadow copies"
            );
            None
        }
    })
}

/// CPU-visible backing for XMem physical pages. The placed buffer keeps a
/// CPU mapping of the heap from which reserved resources can take tiles.
#[cfg(any(test, feature = "diagnostics"))]
pub struct PhysicalHeap {
    backing: ID3D12Resource,
    // A placed buffer must keep its heap alive.
    pub(crate) _heap: ID3D12Heap,
    address: usize,
    len: usize,
}

#[cfg(any(test, feature = "diagnostics"))]
impl PhysicalHeap {
    pub(crate) fn new(device: &ID3D12Device, len: usize) -> windows::core::Result<Self> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: len as u64,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut heap = None;
        let mut backing = None;
        let mut address = std::ptr::null_mut();
        // SAFETY: plain D3D12 creation with output slots on this stack.
        let (heap, backing) = unsafe {
            device.CreateHeap(&desc, &mut heap)?;
            let heap: ID3D12Heap = heap.ok_or_else(windows::core::Error::empty)?;
            device.CreatePlacedResource(
                &heap,
                0,
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut backing,
            )?;
            let backing: ID3D12Resource = backing.ok_or_else(windows::core::Error::empty)?;
            backing.Map(0, None, Some(&mut address))?;
            (heap, backing)
        };
        if address.is_null() {
            return Err(windows::core::Error::empty());
        }
        Ok(Self {
            backing,
            _heap: heap,
            address: address as usize,
            len,
        })
    }

    /// The CPU view, valid for the lifetime of this heap.
    pub fn cpu_address(&self) -> usize {
        self.address
    }

    /// Replaces a Windows file view with the corresponding heap pages.
    ///
    /// # Safety
    /// The caller owns the destination view and keeps this heap alive as
    /// long as the view is mapped.
    pub unsafe fn alias_view(&self, at: usize, offset: usize, len: usize) -> bool {
        if at == 0
            || len == 0
            || [at, offset, len].iter().any(|n| !n.is_multiple_of(PAGE))
            || offset.checked_add(len).is_none_or(|end| end > self.len)
        {
            return false;
        }
        // SAFETY: the caller owns the view; the source range is in the
        // live placed buffer and checked above.
        let remap = unsafe {
            mach::remap_shared(
                at as u64,
                (self.address + offset) as u64,
                len as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        remap.msg == 0 && remap.kr == 0 && remap.address == at as u64
    }
}

#[cfg(any(test, feature = "diagnostics"))]
impl Drop for PhysicalHeap {
    fn drop(&mut self) {
        // SAFETY: unmapped once, while the placed buffer still owns the view.
        unsafe { self.backing.Unmap(0, None) };
    }
}

/// Allocates CPU-visible physical tiles for the Rosetta diagnostics trial.
#[cfg(feature = "diagnostics")]
pub fn physical_heap(len: usize) -> Option<PhysicalHeap> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_TILES") && supported().is_some())
        || len == 0
        || !len.is_multiple_of(PAGE)
    {
        return None;
    }
    let host = host()?;
    match PhysicalHeap::new(&host.device, len) {
        Ok(heap) => Some(heap),
        Err(error) => {
            tracing::warn!(len, %error, "physical tile heap unavailable; using a file section");
            None
        }
    }
}

/// Aliases `[base, base + len)` onto a new write-back L0 D3D12 buffer so CPU
/// and GPU share physical pages with zero copy.
pub fn alias_region(base: usize, len: usize, preserve_contents: bool) -> bool {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    alias_locked(base, len, preserve_contents)
}

/// [`alias_region`], under [`LOCK`].
fn alias_locked(base: usize, len: usize, preserve_contents: bool) -> bool {
    let Some(zeroed) = supported() else {
        return false;
    };
    if base == 0 || len < PAGE || !base.is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
        return false;
    }
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    if crate::shadow::has_aliased_overlap(base as u64, end as u64) {
        return false;
    }
    let Some(h) = host() else {
        return false;
    };
    let Some((buffer, gpu_mem)) = h.shared_buffer(len as u64) else {
        TOTALS[1].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(base = format_args!("{base:#x}"), len, "no host buffer to alias title memory onto; keeping plain memory");
        return false;
    };
    // SAFETY: `base` and `gpu_mem` are `len`-byte owned mappings.
    unsafe {
        if preserve_contents {
            std::ptr::copy_nonoverlapping(base as *const u8, gpu_mem, len);
        } else if !zeroed {
            std::ptr::write_bytes(gpu_mem, 0, len);
        }
        let r = mach::remap_shared(
            base as u64,
            gpu_mem as u64,
            len as u64,
            mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
        );
        if r.msg != 0 || r.kr != 0 || r.address != base as u64 {
            tracing::warn!(
                base = format_args!("{base:#x}"),
                len,
                ?r,
                "mach_vm_remap failed; keeping plain memory"
            );
            TOTALS[1].fetch_add(1, Ordering::Relaxed);
            return false;
        }
    }
    // What the title's memory showed is gone (nothing for fresh memory):
    // guards set there are gone with it, and its trackers see it changed.
    crate::watch::title_memory_remapped(base, len);
    TOTALS[0].fetch_add(1, Ordering::Relaxed);
    crate::host::name(&buffer, || {
        format!("aliased title buffer {base:#x}+{len:#x}")
    });
    // SAFETY: `buffer` is alive.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = crate::shadow::Shadow::new_aliased(buffer, gpu_mem, gpu, base as u64, len as u64);
    crate::shadow::register_aliased(shadow);
    if len >= 16 << 20 {
        tracing::info!(
            base = format_args!("{base:#x}"),
            len,
            "large title buffer aliased zero-copy"
        );
    } else {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            "title buffer aliased zero-copy"
        );
    }
    true
}

/// Removes an aliased region registered at `base`.
pub fn unalias_region(base: usize) -> bool {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    crate::shadow::unregister_aliased(base as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, RemoveVectoredExceptionHandler, EXCEPTION_POINTERS,
    };
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{
        VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_DECOMMIT, PAGE_READONLY,
    };

    struct AliasedAlloc {
        ptr: *mut u8,
        aliased: bool,
    }

    impl AliasedAlloc {
        fn new(len: usize, init: Option<&dyn Fn(*mut u8)>, preserve: bool) -> Self {
            // SAFETY: allocating `len` bytes of RW memory for the test.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            if let Some(f) = init {
                f(ptr);
            }
            let aliased = alias_region(ptr as usize, len, preserve);
            assert!(aliased, "alias_region failed");
            Self { ptr, aliased }
        }
    }

    impl Drop for AliasedAlloc {
        fn drop(&mut self) {
            if self.aliased {
                unalias_region(self.ptr as usize);
            }
            // SAFETY: `self.ptr` was allocated by `VirtualAlloc`.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
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
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        }
    }

    fn committed(
        device: &ID3D12Device,
        heap_type: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            ..Default::default()
        };
        let mut res = None;
        // SAFETY: valid committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(size),
                    state,
                    None,
                    &mut res,
                )
                .unwrap();
        }
        res.unwrap()
    }

    fn run_copy(device: &ID3D12Device, f: impl FnOnce(&ID3D12GraphicsCommandList)) {
        // SAFETY: standalone queue/list/fence for test verification.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();
            f(&list);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = windows::core::Interface::cast(&list).unwrap();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            if fence.GetCompletedValue() < 1 {
                fence.SetEventOnCompletion(1, event).unwrap();
                WaitForSingleObject(event, 5000);
            }
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
    }

    fn gpu_read(device: &ID3D12Device, src: &ID3D12Resource, offset: u64, len: usize) -> Vec<u8> {
        let back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            len as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: `back` and `src` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(&back, 0, src, offset, len as u64)
        });
        let mut out = vec![0u8; len];
        // SAFETY: `back` is a readback buffer of `len` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            back.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), len);
            back.Unmap(0, None);
        }
        out
    }

    fn gpu_write(device: &ID3D12Device, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) {
        let up = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: `up` is an upload buffer of `bytes.len()` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: `dst` and `up` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(dst, offset, &up, 0, bytes.len() as u64)
        });
    }

    #[test]
    fn switch_respects_rosetta_and_warp() {
        assert!(!decide_enabled(Some(false), true, false));
        assert!(!decide_enabled(Some(true), false, false));
        assert!(!decide_enabled(Some(true), true, true));
        assert!(decide_enabled(Some(true), true, false));
    }

    #[test]
    fn unset_is_on_under_rosetta_only() {
        assert!(decide_enabled(None, true, false));
        assert!(!decide_enabled(None, false, false));
        assert!(!decide_enabled(None, true, true));
    }

    #[test]
    fn aliased_buffer_is_coherent_with_gpu_copies() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 4 * PAGE;
        let alloc = AliasedAlloc::new(
            len,
            Some(&|p| {
                for i in (0..len).step_by(4096) {
                    // SAFETY: `p` has `len` bytes.
                    unsafe { p.add(i).write((i >> 12) as u8) };
                }
            }),
            true,
        );
        // Initial contents survived remap.
        for i in (0..len).step_by(4096) {
            // SAFETY: `alloc.ptr` has `len` bytes.
            assert_eq!(unsafe { alloc.ptr.add(i).read_volatile() }, (i >> 12) as u8);
        }

        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(shadow.clean());
        assert_eq!(shadow.update(alloc.ptr as usize, len), 0);
        assert_eq!(shadow.copy_back(alloc.ptr as usize, len), 0);

        for round in 0..8u8 {
            let off = len - 2 * PAGE;
            for i in 0..PAGE {
                let b = (i as u8).wrapping_mul(31) ^ round;
                // SAFETY: inside `alloc`.
                unsafe { alloc.ptr.add(off + i).write_volatile(b) };
            }
            let got = gpu_read(&h.device, &shadow.buffer, off as u64, PAGE);
            for (i, &b) in got.iter().enumerate() {
                assert_eq!(b, (i as u8).wrapping_mul(31) ^ round);
            }

            let new: Vec<u8> = (0..PAGE).map(|i| (i as u8) ^ round ^ 0xc3).collect();
            gpu_write(&h.device, &shadow.buffer, (len - PAGE) as u64, &new);
            // SAFETY: inside `alloc`.
            let cpu = unsafe { std::slice::from_raw_parts(alloc.ptr.add(len - PAGE), PAGE) };
            assert_eq!(cpu, &new[..]);
        }
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn aliased_gpu_writes_notify_overlapping_texture_tracker() {
        if !crate::watch::rosetta() || !crate::watch::fault_tracking() {
            return;
        }
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 2 * PAGE;
        // SAFETY: allocating `len` bytes with `MEM_WRITE_WATCH` for tracker test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let alloc = AliasedAlloc { ptr, aliased: true };
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let base = alloc.ptr as usize;

        let tex = crate::watch::track(base, len, false).expect("texture tracker");
        let _ = tex.take(base, base + len);
        assert!(tex.clean());

        // A range whose writes are not marked (main-mac: only those are told
        // at their copy-back).
        shadow.writing(base, PAGE, false);
        assert_eq!(shadow.copy_back(base, PAGE), 0);
        assert!(tex.changed(base / PAGE));
        assert!(!tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert!(tex.clean());
        assert_eq!(shadow.copy_back_marked(&[base + PAGE]), 0);
        assert!(!tex.changed(base / PAGE));
        assert!(tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert_eq!(shadow.copy_back_page(base), 0);
        assert!(tex.changed(base / PAGE));
    }

    /// 64 KiB of `MEM_WRITE_WATCH` memory per page, aliased, with a texture
    /// tracker over it that has taken (and guarded) every page.
    fn aliased_with_tracker(pages: usize) -> Option<(AliasedAlloc, Arc<crate::shadow::Shadow>, crate::watch::Tracked)> {
        if !crate::watch::rosetta() || !crate::watch::fault_tracking() || host().is_none() {
            return None;
        }
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = pages * PAGE;
        // SAFETY: a fresh allocation for the test.
        let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }
            .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let alloc = AliasedAlloc { ptr, aliased: true };
        let shadow = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        let tex = crate::watch::track(ptr as usize, len, false).expect("texture tracker");
        let _ = tex.take(ptr as usize, ptr as usize + len);
        assert!(tex.clean());
        Some((alloc, shadow, tex))
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        assert_ne!(unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) }, 0);
        info.Protect
    }

    #[test]
    fn marked_writes_are_told_by_their_marks_alone() {
        let Some((alloc, shadow, tex)) = aliased_with_tracker(2) else {
            return;
        };
        let base = alloc.ptr as usize;
        // A heap view's writes are marked: its declared range tells nothing
        // at the copy-back (VIctory-macOS #207 told it whole, every page of
        // the title's big heaps at every submission).
        shadow.writing(base, 2 * PAGE, true);
        assert_eq!(shadow.copy_back(base, 2 * PAGE), 0);
        assert!(tex.clean(), "a marked range's copy-back marks nothing");
        // Withheld writes are in title memory all the same: told whole.
        shadow.writing(base, 2 * PAGE, true);
        shadow.keep_writes(base, 2 * PAGE);
        assert!(tex.changed(base / PAGE) && tex.changed(base / PAGE + 1));
        let _ = tex.take(base, base + 2 * PAGE);
        // A root view's writes are not marked: its range is told.
        shadow.writing(base + PAGE, PAGE, false);
        assert_eq!(shadow.copy_back(base + PAGE, PAGE), 0);
        assert!(!tex.changed(base / PAGE));
        assert!(tex.changed(base / PAGE + 1));
        // Once: the range is no longer pending.
        let _ = tex.take(base, base + 2 * PAGE);
        assert_eq!(shadow.copy_back(base + PAGE, PAGE), 0);
        assert!(tex.clean());
    }

    #[test]
    fn gpu_writes_leave_the_title_memory_guarded() {
        let Some((alloc, shadow, tex)) = aliased_with_tracker(1) else {
            return;
        };
        let base = alloc.ptr as usize;
        assert_eq!(protection(base), PAGE_READONLY, "taken: guarded");
        assert_eq!(shadow.copy_back_page(base), 0);
        assert!(tex.changed(base / PAGE));
        assert_eq!(protection(base), PAGE_READONLY, "a GPU write opens no guard");
        // Taken again without a protection change; the title's next write is
        // still seen.
        let _ = tex.take(base, base + PAGE);
        assert!(tex.clean());
        // SAFETY: inside the allocation; the guard's fault marks the page.
        unsafe { alloc.ptr.add(8).write_volatile(0x5c) };
        assert!(tex.changed(base / PAGE), "the title's write after a GPU write is seen");
        assert_eq!(shadow.read_host(base as u64 + 8, 1)[0] & 0xff, 0x5c, "and the buffer has it");
    }

    #[test]
    fn a_new_reservation_over_a_released_alias_replaces_it() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: a fresh allocation for the test.
        let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let old = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        // The title releases it through the host, which the runtime does not
        // see, and the host hands the addresses out again.
        // SAFETY: our allocation.
        assert_ne!(unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) }, 0);
        // SAFETY: the addresses just released.
        let again = unsafe { VirtualAlloc(ptr.cast(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert_eq!(again, ptr, "the same addresses again");
        // SAFETY: inside the new allocation.
        unsafe { ptr.add(64).write_volatile(0x7e) };
        let stale = unsafe { old.host_ptr(ptr as u64 + 64).read_volatile() } != 0x7e;
        eprintln!("after the host's release, the old alias's buffer misses the title's writes: {stale}");
        super::committed(ptr as usize, len, true, true, None);
        let new = crate::shadow::find(ptr as u64, len as u64).expect("aliased again");
        assert!(!Arc::ptr_eq(&old, &new), "the stale alias is dropped");
        assert!(new.is_aliased());
        // SAFETY: inside the allocation.
        unsafe { ptr.add(128).write_volatile(0x3c) };
        assert_eq!(gpu_read(&h.device, &new.buffer, 128, 1), vec![0x3c], "the new alias is the title's memory");
        drop((old, new));
        let _ = AliasedAlloc { ptr, aliased: true };
    }

    #[test]
    fn a_recommit_after_a_decommit_points_the_pages_back() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let (base, second) = (alloc.ptr as usize, alloc.ptr as usize + PAGE);
        let shadow = crate::shadow::find(base as u64, len as u64).expect("aliased shadow");
        assert!(uncommitted(base, len).is_empty());
        // SAFETY: inside the allocation.
        unsafe { (second as *mut u8).add(8).write_volatile(0x11) };
        // The title decommits the second page through the host.
        // SAFETY: our allocation.
        assert_ne!(unsafe { VirtualFree(second as *mut _, PAGE, MEM_DECOMMIT) }, 0);
        let before = uncommitted(base, len);
        assert_eq!(before, vec![(second, second + PAGE)]);
        // SAFETY: recommitting it.
        assert!(!unsafe { VirtualAlloc(second as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: inside the allocation.
        unsafe { (second as *mut u8).add(16).write_volatile(0x22) };
        let broken = unsafe { shadow.host_ptr(second as u64 + 16).read_volatile() } != 0x22;
        eprintln!("after the host's decommit and commit, the alias's buffer misses the title's writes: {broken}");
        super::committed(second, PAGE, false, true, Some(&before));
        // SAFETY: inside the allocation.
        unsafe {
            assert_eq!((second as *const u8).add(8).read_volatile(), 0, "recommitted memory reads zero");
            (second as *mut u8).add(24).write_volatile(0x33);
        }
        assert_eq!(gpu_read(&h.device, &shadow.buffer, (PAGE + 24) as u64, 1), vec![0x33], "the buffer is the title's memory again");
        assert!(crate::shadow::find(base as u64, len as u64).is_some_and(|s| Arc::ptr_eq(&s, &shadow)));
    }

    #[test]
    fn aliased_shadow_refreshes_widened_rgb32_view() {
        if !crate::watch::rosetta() {
            return;
        }
        let len = PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let widened = shadow
            .widened(alloc.ptr as u64, 48, 0x3f80_0000)
            .expect("widened view");
        assert!(!shadow.clean());

        let words = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // SAFETY: writing 48 bytes at the start of `alloc.ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), alloc.ptr.cast::<u32>(), words.len());
        }
        assert_eq!(shadow.update(alloc.ptr as usize, 48), 0);
        assert_eq!(
            widened.read_host(16),
            vec![
                1,
                2,
                3,
                0x3f80_0000,
                4,
                5,
                6,
                0x3f80_0000,
                7,
                8,
                9,
                0x3f80_0000,
                10,
                11,
                12,
                0x3f80_0000,
            ]
        );
    }

    #[test]
    fn aliased_memory_supports_virtual_query_and_write_guard() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        static GUARD_AT: AtomicUsize = AtomicUsize::new(0);
        static GUARD_HIT: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: `info` is provided by Windows/Wine exception dispatch.
            unsafe {
                let rec = &*(*info).ExceptionRecord;
                let page = GUARD_AT.load(Ordering::SeqCst);
                if rec.ExceptionCode as u32 == 0xC000_0005 && page != 0 {
                    let addr = rec.ExceptionInformation[1];
                    if addr >= page && addr < page + PAGE {
                        let mut old = 0;
                        VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old);
                        GUARD_HIT.store(true, Ordering::SeqCst);
                        return -1;
                    }
                }
                0
            }
        }

        let len = 2 * PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        // SAFETY: querying our own allocation.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let q = VirtualQuery(alloc.ptr.add(PAGE).cast(), &mut info, size_of_val(&info));
            assert_ne!(q, 0);
            assert_eq!(info.State, MEM_COMMIT);
            assert_eq!(info.Protect, PAGE_READWRITE);

            let handler = AddVectoredExceptionHandler(1, Some(on_fault));
            let page = alloc.ptr.add(PAGE);
            GUARD_AT.store(page as usize, Ordering::SeqCst);
            GUARD_HIT.store(false, Ordering::SeqCst);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), PAGE, PAGE_READONLY, &mut old),
                0
            );
            page.add(100).write_volatile(0x77);
            let hit = GUARD_HIT.load(Ordering::SeqCst);
            GUARD_AT.store(0, Ordering::SeqCst);
            RemoveVectoredExceptionHandler(handler);
            assert!(hit, "VirtualProtect write guard did not fault");
        }
        assert_eq!(
            gpu_read(&h.device, &shadow.buffer, (PAGE + 100) as u64, 1),
            vec![0x77]
        );
    }

    #[test]
    fn unalias_region_removes_shadow_and_allows_virtual_free() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: allocating `len` bytes for explicit unalias + free test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let shadow = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(unalias_region(ptr as usize));
        assert!(crate::shadow::find(ptr as u64, len as u64).is_none_or(|s| !s.is_aliased()));
        // SAFETY: `ptr` was allocated by `VirtualAlloc`.
        assert_ne!(unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) }, 0);
        drop(shadow);
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }
}
