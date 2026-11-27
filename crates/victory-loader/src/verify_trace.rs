//! Native x86-64 inline probe and live registry dumper for the Series 1.0.2.1
//! content-activation path (`image+0x1dac680` -> `image+0x1d77d90` ->
//! `image+0x292d220` -> `image+0x292d370`).
//!
//! `VICTORY_TRACE_VERIFY=1` traces every decision inside `0x1d77d90` and
//! `0x292d370` (provider readiness state, second-table membership, sub-item
//! lookup, validator `[second+0x18]`, and mount `[second+0x20]`).
//!
//! Independently, [`dump_detail`] inspects `G = *[image+0x7f00510]` at the
//! natural `+0x1dac734` assert: provider states (`[G+0x2a0]+0x1058/+0x1060`),
//! second-table header (`[G+0x2a8]`), main package entries (`[G+0x2b0]+0x30/38`),
//! and every sub-item (`[G+0x2b0]+0x10/18`) with its RAGE SSO strings.

use std::collections::BTreeMap;
use std::fmt::{self, LowerHex};
use std::fs::File;
use std::io::{self, Write};
use std::os::windows::io::AsHandle;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use windows_sys::Win32::System::Diagnostics::Debug::{FlushInstructionCache, ReadProcessMemory};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    VirtualQuery,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId};

use crate::crash::locate;

const TITLE_HASH: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const ASSERT_RVA: usize = 0x1dac734;
const SINGLETON_RVA: usize = 0x7f00510;
const POOL_SIZE: usize = 0x1000;
const TRAMP_STRIDE: usize = 256;
const MAX_SUB_ITEMS: usize = 64;
const MAX_TABLE_GROUPS: usize = 4096;

const VERIFY_ENTER: usize = 0;
const PROVIDER_OK: usize = 1;
const MOUNT_ENTER: usize = 2;
const ITEM_LOOKUP: usize = 3;
const ITEM_FOUND: usize = 4;
const ITEM_VALID_OK: usize = 5;
const ITEM_MOUNT_CALL: usize = 6;
const ITEM_MOUNT_OK: usize = 7;

struct SiteSpec {
    name: &'static str,
    rva: usize,
    original: &'static [u8],
}

/// Eight non-RIP-relative sites spanning every failure exit in `0x1d77d90` and
/// `0x292d370`.
const SITES: [SiteSpec; 8] = [
    SiteSpec {
        name: "verify_enter",
        rva: 0x1d77dab,
        // mov rcx, r14; mov rdx, rsi
        original: &[0x4c, 0x89, 0xf1, 0x48, 0x89, 0xf2],
    },
    SiteSpec {
        name: "provider_ok",
        rva: 0x1d77dba,
        // mov rbx, qword ptr [r14 + 0x2a8]
        original: &[0x49, 0x8b, 0x9e, 0xa8, 0x02, 0x00, 0x00],
    },
    SiteSpec {
        name: "mount_enter",
        rva: 0x1d77de5,
        // mov rcx, rbx; mov rdx, rsi
        original: &[0x48, 0x89, 0xd9, 0x48, 0x89, 0xf2],
    },
    SiteSpec {
        name: "item_lookup",
        rva: 0x292d423,
        // mov r8d, dword ptr [rbx]; mov ecx, dword ptr [rax + 0x10]
        original: &[0x44, 0x8b, 0x03, 0x8b, 0x48, 0x10],
    },
    SiteSpec {
        name: "item_found",
        rva: 0x292d43b,
        // mov r15, rax; mov rax, qword ptr [rsi]
        original: &[0x49, 0x89, 0xc7, 0x48, 0x8b, 0x06],
    },
    SiteSpec {
        name: "item_valid_ok",
        rva: 0x292d452,
        // inc dword ptr [rsp + 0x3c]; mov edx, dword ptr [r13]
        original: &[0xff, 0x44, 0x24, 0x3c, 0x41, 0x8b, 0x55, 0x00],
    },
    SiteSpec {
        name: "item_mount_call",
        rva: 0x292d5c2,
        // lea r8, [rsp + 0x60]; mov rax, qword ptr [rsi]
        original: &[0x4c, 0x8d, 0x44, 0x24, 0x60, 0x48, 0x8b, 0x06],
    },
    SiteSpec {
        name: "item_mount_ok",
        rva: 0x292d5d9,
        // mov qword ptr [rsp + 0x30], r12; mov qword ptr [rsp + 0x48], rbx
        original: &[0x4c, 0x89, 0x64, 0x24, 0x30, 0x48, 0x89, 0x5c, 0x24, 0x48],
    },
];

/// Neighboring call/branch bytes verified before patching.
const GUARDS: [(usize, &[u8]); 5] = [
    // 0x1d77db1: call 0x292d0b0; test al, al; je +0x50
    (0x1d77db1, &[0xe8, 0xfa, 0x52, 0xbb, 0x00, 0x84, 0xc0, 0x74, 0x50]),
    // 0x1d77ddc: call 0x292d130; test al, al; jne +0x25
    (0x1d77ddc, &[0xe8, 0x4f, 0x53, 0xbb, 0x00, 0x84, 0xc0, 0x75, 0x25]),
    // 0x1d77deb: call 0x292d220
    (0x1d77deb, &[0xe8, 0x30, 0x54, 0xbb, 0x00]),
    // 0x292d447: call qword ptr [rax + 0x18]; test al, al; je 0x292d9ae
    (0x292d447, &[0xff, 0x50, 0x18, 0x84, 0xc0, 0x0f, 0x84, 0x5c, 0x05, 0x00, 0x00]),
    // 0x292d5ca: call qword ptr [rax + 0x20]; mov [rsp+0x2f], al; test al, al; je 0x292d98f
    (0x292d5ca, &[0xff, 0x50, 0x20, 0x88, 0x44, 0x24, 0x2f, 0x84, 0xc0, 0x0f, 0x84, 0xb6, 0x03, 0x00, 0x00]),
];

#[repr(C)]
struct SavedRegs {
    rax: u64,
    rcx: u64,
    rdx: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    rbx: u64,
    rsi: u64,
    rdi: u64,
    r14: u64,
    r15: u64,
    rsp: u64,
    rflags: u64,
}

#[derive(Default)]
struct VerifyCounts {
    visits: [u64; 8],
    seq: u64,
    per_id: BTreeMap<u32, [u64; 3]>,
}

struct Trace {
    image: usize,
    output: File,
    counts: Mutex<VerifyCounts>,
    incomplete: AtomicBool,
    finished: AtomicBool,
    active: AtomicUsize,
}

static TRACE: OnceLock<Trace> = OnceLock::new();

/// Arms the native verify/mount probe when `VICTORY_TRACE_VERIFY` is set and
/// non-zero. Must run before any title thread executes.
///
/// # Safety
/// `image..image + size` must be the mapped main title image for `hash`, with
/// no title threads running yet.
pub unsafe fn start(hash: &str, image: usize, size: usize) -> Result<()> {
    let mode = std::env::var("VICTORY_TRACE_VERIFY").unwrap_or_default();
    let mode = mode.trim();
    if mode.is_empty() || mode == "0" {
        return Ok(());
    }
    ensure!(
        hash == TITLE_HASH,
        "verify_trace could not arm: unsupported title hash ({hash})"
    );
    ensure!(
        image != 0 && image.checked_add(size).is_some() && size > SINGLETON_RVA + 8,
        "verify_trace could not arm: mapped image too small ({size:#x})"
    );
    let output = File::from(io::stderr().as_handle().try_clone_to_owned()?);
    TRACE
        .set(Trace {
            image,
            output,
            counts: Mutex::default(),
            incomplete: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            active: AtomicUsize::new(0),
        })
        .map_err(|_| anyhow::anyhow!("verify_trace already started"))?;

    // SAFETY: `image..image+size` is mapped and no title thread has started.
    let pool = match unsafe { install_detours(image, size, on_site) } {
        Ok(pool) => pool,
        Err(err) => {
            log_line(format_args!("verify_trace could not arm: {err}"));
            return Err(err);
        }
    };
    log_line(format_args!(
        "verify_trace armed mode=native image={image:#x} pool={pool:#x}"
    ));
    Ok(())
}

/// Emits the live trace summary at the natural registry assert breakpoint.
pub(crate) fn on_breakpoint(address: usize) {
    if let Some(trace) = TRACE.get()
        && address == trace.image + ASSERT_RVA
    {
        finish(trace, "registry_assert");
    }
}

/// Emits the live trace summary if the title exits before the assert.
pub fn on_title_exit() {
    if let Some(trace) = TRACE.get() {
        finish(trace, "title_exit");
    }
}

fn finish(trace: &Trace, reason: &str) {
    if trace.finished.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = std::panic::catch_unwind(|| {
        let start = Instant::now();
        while trace.active.load(Ordering::SeqCst) != 0
            && start.elapsed() < Duration::from_millis(25)
        {
            std::thread::yield_now();
        }
        let incomplete =
            trace.incomplete.load(Ordering::Relaxed) || trace.active.load(Ordering::SeqCst) != 0;
        match trace.counts.try_lock() {
            Ok(counts) => {
                let [a, b, c, d, e, f, g, h] = counts.visits;
                log_line(format_args!(
                    "verify_trace summary reason={reason} verify_enter={a} provider_ok={b} \
                     mount_enter={c} item_lookup={d} item_found={e} item_valid_ok={f} \
                     item_mount_call={g} item_mount_ok={h} io_complete={}",
                    !incomplete
                ));
                for (id, [en, ok, mnt]) in &counts.per_id {
                    log_line(format_args!(
                        "verify_trace id_summary id={id:#010x} verify_enter={en} provider_ok={ok} mount_enter={mnt}"
                    ));
                }
            }
            Err(_) => {
                log_line(format_args!(
                    "verify_trace summary reason={reason} io_complete=false counts=unavailable"
                ));
            }
        }
    });
}

extern "system" fn on_site(site: u32, regs: *const SavedRegs) {
    let Some(trace) = TRACE.get() else {
        return;
    };
    if trace.finished.load(Ordering::Acquire) {
        return;
    }
    trace.active.fetch_add(1, Ordering::SeqCst);
    let ok = std::panic::catch_unwind(|| {
        if trace.finished.load(Ordering::SeqCst) {
            return;
        }
        // SAFETY: `SavedRegs` was pushed on the current thread's stack by our trampoline.
        let r = unsafe { &*regs };
        // SAFETY: plain Win32 call for the current thread ID.
        let tid = unsafe { GetCurrentThreadId() };
        let site = site as usize;
        if site >= SITES.len() {
            trace.incomplete.store(true, Ordering::Relaxed);
            return;
        }
        let mut counts = trace.counts.lock().unwrap_or_else(|e| e.into_inner());
        counts.visits[site] += 1;
        counts.seq += 1;
        let seq = counts.seq;
        let name = SITES[site].name;

        let line = match site {
            VERIFY_ENTER | PROVIDER_OK | MOUNT_ENTER => {
                let id = read_u32(r.rsi as usize, 0);
                if let Some(key) = id {
                    counts.per_id.entry(key).or_default()[site] += 1;
                }
                let g = r.r14 as usize;
                let provider = read_usize(g, 0x2a0);
                let second = read_usize(g, 0x2a8);
                let second_78 = second.and_then(|s| read_usize(s, 0x78));
                let state = provider
                    .zip(id)
                    .and_then(|(p, key)| lookup_provider_state(p, key));
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} id={} g={g:#x} \
                     provider={} prov_state={} second={} second_78={}",
                    fmt_hex32(id),
                    fmt_hex(provider),
                    state.map_or_else(|| "?".into(), |s| s.to_string()),
                    fmt_hex(second),
                    fmt_hex(second_78),
                )
            }
            ITEM_LOOKUP => {
                let sub_id = read_u32(r.rbx as usize, 0);
                let table_obj = r.rax as usize;
                let cap = read_u32(table_obj, 0x10);
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} sub_id={} second={:#x} \
                     sub_table={table_obj:#x} sub_cap={}",
                    fmt_hex32(sub_id),
                    r.rsi,
                    cap.map_or_else(|| "?".into(), |c| c.to_string()),
                )
            }
            ITEM_FOUND => {
                let sub_id = read_u32(r.rbx as usize, 0);
                let item = r.rax as usize;
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} sub_id={} {}",
                    fmt_hex32(sub_id),
                    describe_item(item),
                )
            }
            ITEM_VALID_OK => {
                let sub_id = read_u32(r.rbx as usize, 0);
                let item = r.r15 as usize;
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} sub_id={} item={item:#x}",
                    fmt_hex32(sub_id),
                )
            }
            ITEM_MOUNT_CALL => {
                let sub_id = read_u32(r.rbx as usize, 0);
                let item = r.r15 as usize;
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} sub_id={} {}",
                    fmt_hex32(sub_id),
                    describe_item(item),
                )
            }
            ITEM_MOUNT_OK => {
                let sub_id = read_u32(r.rbx as usize, 0);
                format!(
                    "verify_trace seq={seq} site={name} tid={tid} sub_id={}",
                    fmt_hex32(sub_id),
                )
            }
            _ => return,
        };
        log_line(format_args!("{line}"));
    });
    if ok.is_err() {
        trace.incomplete.store(true, Ordering::Relaxed);
    }
    trace.active.fetch_sub(1, Ordering::SeqCst);
}

/// Read-only snapshot of `G = *[image+0x7f00510]`, `[G+0x2a0]` provider states,
/// `[G+0x2a8]` second table header, and `[G+0x2b0]` main packages + sub-items.
/// Writes `verify-detail.txt` in `dir` and mirrors every line to stderr.
pub(crate) fn dump_detail(image: usize, dir: &Path) {
    let text = format_detail(image);
    for line in text.lines() {
        eprintln!("VIctory: verify_detail {line}");
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join("verify-detail.txt"), &text);
}

fn format_detail(image: usize) -> String {
    let mut out = String::new();
    let mut push = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    let Some(g) = read_usize(image, SINGLETON_RVA).filter(|&v| v != 0) else {
        push(format!("singleton *[image+{SINGLETON_RVA:#x}] null or unreadable"));
        return out;
    };
    let provider = read_usize(g, 0x2a0);
    let second = read_usize(g, 0x2a8);
    let main = read_usize(g, 0x2b0);
    push(format!(
        "G={g:#x} vtable={} provider={} second={} main={}",
        fmt_symbol(read_usize(g, 0)),
        fmt_hex(provider),
        fmt_hex(second),
        fmt_hex(main),
    ));

    if let Some(p) = provider.filter(|&v| v != 0) {
        let cap = read_u32(p, 0x1058);
        let base = read_usize(p, 0x1060);
        push(format!(
            "provider={p:#x} vtable={} table_1058_cap={} base={}",
            fmt_symbol(read_usize(p, 0)),
            cap.map_or_else(|| "?".into(), |c| c.to_string()),
            fmt_hex(base),
        ));
        if let (Some(c), Some(b)) = (cap, base) {
            let entries = dump_provider_table(c as usize, b);
            push(format!("provider_entries={}", entries.len()));
            for (key, state) in entries {
                let pass_v50 = state >= 2;
                let pass_v58 = (state & !1) == 8;
                push(format!(
                    "  provider_entry key={key:#010x} state={state} pass_v50={pass_v50} pass_v58={pass_v58}"
                ));
            }
        }
    }

    if let Some(s) = second.filter(|&v| v != 0) {
        push(format!(
            "second={s:#x} vtable={} t1_cap={} t1_base={} t2_cap={} t2_base={} second_78={}",
            fmt_symbol(read_usize(s, 0)),
            read_u32(s, 0x8).map_or_else(|| "?".into(), |c| c.to_string()),
            fmt_hex(read_usize(s, 0x10)),
            read_u32(s, 0x18).map_or_else(|| "?".into(), |c| c.to_string()),
            fmt_hex(read_usize(s, 0x20)),
            fmt_hex(read_usize(s, 0x78)),
        ));
    }

    if let Some(m) = main.filter(|&v| v != 0) {
        let sub_cap = read_u32(m, 0x10);
        let sub_base = read_usize(m, 0x18);
        let pkg_cap = read_u32(m, 0x30);
        let pkg_base = read_usize(m, 0x38);
        push(format!(
            "main={m:#x} sub_cap={} sub_base={} pkg_cap={} pkg_base={}",
            sub_cap.map_or_else(|| "?".into(), |c| c.to_string()),
            fmt_hex(sub_base),
            pkg_cap.map_or_else(|| "?".into(), |c| c.to_string()),
            fmt_hex(pkg_base),
        ));
        if let (Some(pc), Some(pb)) = (pkg_cap, pkg_base) {
            let pkgs = dump_ptr_swisstable(pc as usize, pb);
            push(format!("main_pkg_entries={}", pkgs.len()));
            for (key, entry) in pkgs {
                let field_14 = read_u32(entry, 0x14);
                let item_count = read_u32(entry, 0x24);
                let items_ptr = read_usize(entry, 0x28);
                push(format!(
                    "  pkg key={key:#010x} entry={entry:#x} vtable={} field_14={} item_count={} items_ptr={}",
                    fmt_symbol(read_usize(entry, 0)),
                    field_14.map_or_else(|| "?".into(), |v| v.to_string()),
                    item_count.map_or_else(|| "?".into(), |v| v.to_string()),
                    fmt_hex(items_ptr),
                ));
                if let (Some(cnt), Some(ptr)) = (item_count, items_ptr)
                    && ptr != 0
                {
                    let n = (cnt as usize).min(MAX_SUB_ITEMS);
                    for i in 0..n {
                        let Some(sub_id) = read_u32(ptr, i * 4) else {
                            push(format!("    sub[{i}] unreadable"));
                            break;
                        };
                        let item = sub_cap
                            .zip(sub_base)
                            .and_then(|(sc, sb)| lookup_ptr_swisstable(sc as usize, sb, sub_id));
                        match item {
                            Some(it) => push(format!(
                                "    sub[{i}] id={sub_id:#010x} {}",
                                describe_item(it)
                            )),
                            None => push(format!(
                                "    sub[{i}] id={sub_id:#010x} item=NOT_FOUND"
                            )),
                        }
                    }
                }
            }
        }
    }
    out
}

/// Looks up `key` in `[provider + 0x1058 / +0x1060]` (stride `0x310`, 16 slots
/// of `0x30` bytes, `key` at `+0x10`, `state` u32 at `+0x34`).
fn lookup_provider_state(provider: usize, key: u32) -> Option<u32> {
    let cap = read_u32(provider, 0x1058)? as usize;
    let base = read_usize(provider, 0x1060)?;
    if cap == 0 || cap > MAX_TABLE_GROUPS || base == 0 {
        return None;
    }
    let h2 = (key & 0x7f) as u8;
    let mut group = ((key >> 7) as usize) & (cap - 1);
    for _ in 0..cap {
        let gaddr = base.checked_add(group.checked_mul(0x310)?)?;
        let ctrl = read_bytes::<16>(gaddr)?;
        for (slot, &c) in ctrl.iter().enumerate() {
            if c == h2 {
                let slot_addr = gaddr + 0x10 + slot * 0x30;
                if read_u32(slot_addr, 0) == Some(key) {
                    return read_u32(slot_addr, 0x24);
                }
            }
        }
        if ctrl.iter().any(|&c| (c as i8) < 0) {
            return None;
        }
        group = (group + 1) & (cap - 1);
    }
    None
}

fn dump_provider_table(cap: usize, base: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if cap == 0 || base == 0 {
        return out;
    }
    for g in 0..cap.min(MAX_TABLE_GROUPS) {
        let Some(gaddr) = base.checked_add(g.saturating_mul(0x310)) else {
            break;
        };
        let Some(ctrl) = read_bytes::<16>(gaddr) else {
            break;
        };
        for (slot, &c) in ctrl.iter().enumerate() {
            if (c & 0x80) == 0 {
                let saddr = gaddr + 0x10 + slot * 0x30;
                if let (Some(key), Some(state)) = (read_u32(saddr, 0), read_u32(saddr, 0x24)) {
                    out.push((key, state));
                }
            }
        }
    }
    out
}

/// Scans a `0x110`-stride SwissTable (`0x10` control bytes + 16 × `(u32 key, u32 pad, usize ptr)`).
fn dump_ptr_swisstable(cap: usize, base: usize) -> Vec<(u32, usize)> {
    let mut out = Vec::new();
    if cap == 0 || base == 0 {
        return out;
    }
    for g in 0..cap.min(MAX_TABLE_GROUPS) {
        let Some(gaddr) = base.checked_add(g.saturating_mul(0x110)) else {
            break;
        };
        let Some(ctrl) = read_bytes::<16>(gaddr) else {
            break;
        };
        for (slot, &c) in ctrl.iter().enumerate() {
            if (c & 0x80) == 0 {
                let saddr = gaddr + 0x10 + slot * 0x10;
                if let (Some(key), Some(ptr)) = (read_u32(saddr, 0), read_usize(saddr, 8)) {
                    out.push((key, ptr));
                }
            }
        }
    }
    out
}

fn lookup_ptr_swisstable(cap: usize, base: usize, key: u32) -> Option<usize> {
    if cap == 0 || cap > MAX_TABLE_GROUPS || base == 0 {
        return None;
    }
    let h2 = (key & 0x7f) as u8;
    let mut group = ((key >> 7) as usize) & (cap - 1);
    for _ in 0..cap {
        let gaddr = base.checked_add(group.checked_mul(0x110)?)?;
        let ctrl = read_bytes::<16>(gaddr)?;
        for (slot, &c) in ctrl.iter().enumerate() {
            if c == h2 {
                let saddr = gaddr + 0x10 + slot * 0x10;
                if read_u32(saddr, 0) == Some(key) {
                    return read_usize(saddr, 8);
                }
            }
        }
        if ctrl.iter().any(|&c| (c as i8) < 0) {
            return None;
        }
        group = (group + 1) & (cap - 1);
    }
    None
}

fn describe_item(item: usize) -> String {
    if item == 0 {
        return "item=0x0".into();
    }
    let vtable = fmt_symbol(read_usize(item, 0));
    let c0 = read_rage_sso(item, 0x08);
    let c1 = read_rage_sso(item, 0x18);
    let c2 = read_rage_sso(item, 0x28);
    let c3 = read_rage_sso(item, 0x38);
    let mount = read_rage_sso(item, 0x70);
    let path = read_rage_sso(item, 0x80);
    format!(
        "item={item:#x} vtable={vtable} cond=[{c0:?},{c1:?},{c2:?},{c3:?}] mount={mount:?} path={path:?}"
    )
}

/// Reads a 16-byte RAGE SSO string at `base + offset`:
/// - `raw[15] & 0x80 != 0`: inline buffer `raw[..len]`, `len = (raw[15] & 0x7f).min(15)`.
/// - else: heap pointer at `raw[0..8]`, length `u16` at `raw[8..10]`.
fn read_rage_sso(base: usize, offset: usize) -> String {
    let Some(raw) = read_field::<16>(base, offset) else {
        return "?".into();
    };
    let bytes: Vec<u8> = if (raw[15] & 0x80) != 0 {
        let len = usize::from(raw[15] & 0x7f).min(15);
        raw[..len].to_vec()
    } else {
        let ptr = usize::from_le_bytes(raw[0..8].try_into().unwrap());
        let len = usize::from(u16::from_le_bytes(raw[8..10].try_into().unwrap()));
        if ptr == 0 || len == 0 {
            return String::new();
        }
        if len > 512 {
            return format!("<len={len}>");
        }
        let Some(buf) = read_slice(ptr, len.min(256)) else {
            return format!("<unreadable {ptr:#x} len={len}>");
        };
        buf
    };
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let slice = &bytes[..nul];
    if slice.iter().all(|&b| (0x20..=0x7e).contains(&b)) {
        String::from_utf8_lossy(slice).into_owned()
    } else {
        format!("<non-ascii len={}>", slice.len())
    }
}

fn log_line(args: fmt::Arguments<'_>) {
    if let Some(trace) = TRACE.get()
        && (&trace.output)
            .write_all(format!("VIctory: {args}\n").as_bytes())
            .is_err()
    {
        trace.incomplete.store(true, Ordering::Relaxed);
    }
}

fn fmt_hex<T: LowerHex>(value: Option<T>) -> String {
    value.map_or_else(|| "?".into(), |v| format!("{v:#x}"))
}

fn fmt_hex32(value: Option<u32>) -> String {
    value.map_or_else(|| "?".into(), |v| format!("{v:#010x}"))
}

fn fmt_symbol(value: Option<usize>) -> String {
    match value {
        None => "?".into(),
        Some(0) => "0x0".into(),
        Some(p) => locate(p),
    }
}

fn read_slice(address: usize, len: usize) -> Option<Vec<u8>> {
    if address == 0 || len == 0 {
        return None;
    }
    let end = address.checked_add(len - 1)?;
    let mut page = address & !0xfff;
    let end_page = end & !0xfff;
    loop {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries virtual memory metadata without dereferencing `page`.
        let found = unsafe { VirtualQuery(page as *const _, &mut info, size_of_val(&info)) };
        if found == 0
            || info.State != MEM_COMMIT
            || info.Protect & PAGE_GUARD != 0
            || info.Protect & 0xff == PAGE_NOACCESS
        {
            return None;
        }
        if page == end_page {
            break;
        }
        page = page.checked_add(0x1000)?;
    }
    let mut data = vec![0u8; len];
    let mut count = 0;
    // SAFETY: `ReadProcessMemory` safely copies into `data` without faulting on concurrent unmap.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            address as *const _,
            data.as_mut_ptr().cast(),
            len,
            &mut count,
        )
    };
    (ok != 0 && count == len).then_some(data)
}

fn read_bytes<const N: usize>(address: usize) -> Option<[u8; N]> {
    let v = read_slice(address, N)?;
    v.try_into().ok()
}

fn read_field<const N: usize>(base: usize, offset: usize) -> Option<[u8; N]> {
    if base == 0 {
        return None;
    }
    read_bytes(base.checked_add(offset)?)
}

fn read_u32(base: usize, offset: usize) -> Option<u32> {
    read_field::<4>(base, offset).map(u32::from_le_bytes)
}

fn read_usize(base: usize, offset: usize) -> Option<usize> {
    read_field::<8>(base, offset).map(usize::from_le_bytes)
}

fn emit_trampoline(
    site_index: u32,
    original: &[u8],
    return_to: usize,
    tramp_addr: usize,
    handler: usize,
) -> Option<Vec<u8>> {
    let mut c = Vec::with_capacity(240);
    // Preserve RFLAGS first, anchor RSP via RBP, 16-byte align RSP.
    c.push(0x9c); // pushfq
    c.push(0x55); // push rbp
    c.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
    c.extend_from_slice(&[0x48, 0x83, 0xe4, 0xf0]); // and rsp, -16
    // 0x20 shadow + 0x70 SavedRegs (14*8=0x70) + 0x10 pad + 0x60 XMM0..5 = 0x100.
    c.extend_from_slice(&[0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00]); // sub rsp, 0x100

    // Save SavedRegs at [rsp + 0x20..0x90].
    c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x20]); // rax
    c.extend_from_slice(&[0x48, 0x89, 0x4c, 0x24, 0x28]); // rcx
    c.extend_from_slice(&[0x48, 0x89, 0x54, 0x24, 0x30]); // rdx
    c.extend_from_slice(&[0x4c, 0x89, 0x44, 0x24, 0x38]); // r8
    c.extend_from_slice(&[0x4c, 0x89, 0x4c, 0x24, 0x40]); // r9
    c.extend_from_slice(&[0x4c, 0x89, 0x54, 0x24, 0x48]); // r10
    c.extend_from_slice(&[0x4c, 0x89, 0x5c, 0x24, 0x50]); // r11
    c.extend_from_slice(&[0x48, 0x89, 0x5c, 0x24, 0x58]); // rbx
    c.extend_from_slice(&[0x48, 0x89, 0x74, 0x24, 0x60]); // rsi
    c.extend_from_slice(&[0x48, 0x89, 0x7c, 0x24, 0x68]); // rdi
    c.extend_from_slice(&[0x4c, 0x89, 0x74, 0x24, 0x70]); // r14
    c.extend_from_slice(&[0x4c, 0x89, 0x7c, 0x24, 0x78]); // r15
    c.extend_from_slice(&[0x48, 0x8d, 0x45, 0x10]); // lea rax, [rbp+0x10] (orig rsp)
    c.extend_from_slice(&[0x48, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]); // mov [rsp+0x80], rax
    c.extend_from_slice(&[0x48, 0x8b, 0x45, 0x08]); // mov rax, [rbp+0x08] (orig rflags)
    c.extend_from_slice(&[0x48, 0x89, 0x84, 0x24, 0x88, 0x00, 0x00, 0x00]); // mov [rsp+0x88], rax

    // Save XMM0..XMM5 at [rsp+0xa0..0x100].
    for i in 0u8..6 {
        let modrm = 0x84 | (i << 3);
        let disp = (0xa0u32 + u32::from(i) * 16).to_le_bytes();
        c.extend_from_slice(&[0x0f, 0x11, modrm, 0x24]);
        c.extend_from_slice(&disp);
    }

    // Call handler(ecx = site_index, rdx = &SavedRegs).
    c.push(0xfc); // cld
    c.push(0xb9); // mov ecx, imm32
    c.extend_from_slice(&site_index.to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8d, 0x54, 0x24, 0x20]); // lea rdx, [rsp+0x20]
    c.extend_from_slice(&[0x48, 0xb8]); // mov rax, imm64
    c.extend_from_slice(&(handler as u64).to_le_bytes());
    c.extend_from_slice(&[0xff, 0xd0]); // call rax

    // Restore XMM0..XMM5.
    for i in 0u8..6 {
        let modrm = 0x84 | (i << 3);
        let disp = (0xa0u32 + u32::from(i) * 16).to_le_bytes();
        c.extend_from_slice(&[0x0f, 0x10, modrm, 0x24]);
        c.extend_from_slice(&disp);
    }

    // Restore volatile GPRs.
    c.extend_from_slice(&[0x4c, 0x8b, 0x5c, 0x24, 0x50]); // r11
    c.extend_from_slice(&[0x4c, 0x8b, 0x54, 0x24, 0x48]); // r10
    c.extend_from_slice(&[0x4c, 0x8b, 0x4c, 0x24, 0x40]); // r9
    c.extend_from_slice(&[0x4c, 0x8b, 0x44, 0x24, 0x38]); // r8
    c.extend_from_slice(&[0x48, 0x8b, 0x54, 0x24, 0x30]); // rdx
    c.extend_from_slice(&[0x48, 0x8b, 0x4c, 0x24, 0x28]); // rcx
    c.extend_from_slice(&[0x48, 0x8b, 0x44, 0x24, 0x20]); // rax

    // Restore RSP, RBP, RFLAGS.
    c.extend_from_slice(&[0x48, 0x89, 0xec]); // mov rsp, rbp
    c.push(0x5d); // pop rbp
    c.push(0x9d); // popfq

    // Original instructions + jump back.
    c.extend_from_slice(original);
    let jmp_end = tramp_addr.checked_add(c.len() + 5)?;
    let rel = i32::try_from((return_to as i128) - (jmp_end as i128)).ok()?;
    c.push(0xe9);
    c.extend_from_slice(&rel.to_le_bytes());
    Some(c)
}

fn emit_jump(from: usize, len: usize, to: usize) -> Option<Vec<u8>> {
    if len < 5 {
        return None;
    }
    let rel = i32::try_from((to as i128) - (from.checked_add(5)? as i128)).ok()?;
    let mut patch = vec![0x90u8; len];
    patch[0] = 0xe9;
    patch[1..5].copy_from_slice(&rel.to_le_bytes());
    Some(patch)
}

fn alloc_near(image: usize, image_size: usize) -> Option<*mut u8> {
    let min_site = image + SITES[0].rva;
    let max_site = image + SITES[SITES.len() - 1].rva + 16;
    let try_hint = |hint: usize| -> Option<*mut u8> {
        if hint == 0 {
            return None;
        }
        // SAFETY: reserving a fresh anonymous page at a hint; failure returns null.
        let ptr = unsafe {
            VirtualAlloc(
                hint as *const _,
                POOL_SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as *mut u8;
        if ptr.is_null() {
            return None;
        }
        let addr = ptr as usize;
        let d_lo = (addr as i128 - min_site as i128).abs();
        let d_hi = ((addr + POOL_SIZE) as i128 - max_site as i128).abs();
        if d_lo < (i32::MAX as i128 - 0x10000) && d_hi < (i32::MAX as i128 - 0x10000) {
            Some(ptr)
        } else {
            // SAFETY: freeing our own out-of-range allocation.
            unsafe {
                VirtualFree(
                    ptr.cast(),
                    0,
                    windows_sys::Win32::System::Memory::MEM_RELEASE,
                );
            }
            None
        }
    };
    for step in 1..2048usize {
        if let Some(hint) = image.checked_sub(step << 20)
            && let Some(ptr) = try_hint(hint)
        {
            return Some(ptr);
        }
    }
    let after = image
        .saturating_add(image_size)
        .next_multiple_of(1 << 20);
    for step in 0..2048usize {
        if let Some(hint) = after.checked_add(step << 20)
            && let Some(ptr) = try_hint(hint)
        {
            return Some(ptr);
        }
    }
    None
}

unsafe fn install_detours(
    image: usize,
    image_size: usize,
    handler: extern "system" fn(u32, *const SavedRegs),
) -> Result<usize> {
    for spec in &SITES {
        ensure!(
            spec.rva + spec.original.len() <= image_size,
            "site {} at image+{:#x} exceeds image bounds",
            spec.name,
            spec.rva
        );
        // SAFETY: `spec.rva..spec.rva+len` lies inside the mapped image.
        let found = unsafe {
            std::slice::from_raw_parts((image + spec.rva) as *const u8, spec.original.len())
        };
        if found != spec.original {
            bail!("unexpected bytes at image+{:#x} ({})", spec.rva, spec.name);
        }
    }
    for (rva, expected) in &GUARDS {
        ensure!(
            rva + expected.len() <= image_size,
            "guard at image+{rva:#x} exceeds image bounds"
        );
        // SAFETY: `rva..rva+len` lies inside the mapped image.
        let found = unsafe { std::slice::from_raw_parts((image + rva) as *const u8, expected.len()) };
        if found != *expected {
            bail!("unexpected guard bytes at image+{rva:#x}");
        }
    }

    let pool = alloc_near(image, image_size)
        .ok_or_else(|| anyhow::anyhow!("no memory within rel32 reach of {image:#x}"))?;
    let pool_addr = pool as usize;

    let mut patches = Vec::with_capacity(SITES.len());
    for (index, spec) in SITES.iter().enumerate() {
        let site_addr = image + spec.rva;
        let return_to = site_addr + spec.original.len();
        let tramp_offset = index * TRAMP_STRIDE;
        let tramp_addr = pool_addr + tramp_offset;
        let tramp = emit_trampoline(
            index as u32,
            spec.original,
            return_to,
            tramp_addr,
            handler as *const () as usize,
        )
        .ok_or_else(|| anyhow::anyhow!("trampoline out of reach for {}", spec.name))?;
        ensure!(
            tramp.len() <= TRAMP_STRIDE && tramp_offset + tramp.len() <= POOL_SIZE,
            "trampoline overflow for {}",
            spec.name
        );
        let jump = emit_jump(site_addr, spec.original.len(), tramp_addr)
            .ok_or_else(|| anyhow::anyhow!("jump out of reach for {}", spec.name))?;
        // SAFETY: writing into our freshly allocated writable trampoline pool.
        unsafe {
            std::ptr::copy_nonoverlapping(tramp.as_ptr(), pool.add(tramp_offset), tramp.len());
        }
        patches.push((site_addr, jump));
    }

    let mut old = 0;
    // SAFETY: sealing our trampoline pool RX before routing execution to it.
    let sealed = unsafe { VirtualProtect(pool.cast(), POOL_SIZE, PAGE_EXECUTE_READ, &mut old) };
    ensure!(sealed != 0, "VirtualProtect(RX) on verify trampoline pool failed");

    for (site_addr, jump) in &patches {
        let mut prev = 0;
        // SAFETY: verified code site in `image`; no title threads are running yet.
        unsafe {
            if VirtualProtect(
                *site_addr as *const _,
                jump.len(),
                PAGE_EXECUTE_READWRITE,
                &mut prev,
            ) == 0
            {
                bail!("VirtualProtect(RWX) failed at {site_addr:#x}");
            }
            std::ptr::copy_nonoverlapping(jump.as_ptr(), *site_addr as *mut u8, jump.len());
            VirtualProtect(*site_addr as *const _, jump.len(), prev, &mut prev);
        }
    }
    // SAFETY: flushing the instruction cache after patching code.
    unsafe { FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0) };
    Ok(pool_addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rage_sso_and_swisstable_readers_decode_inline_heap_and_provider_state() {
        // Inline SSO string: "data:/" (len 6, high bit set -> 0x86).
        let mut inline_buf = [0u8; 0x90];
        inline_buf[0x70..0x76].copy_from_slice(b"data:/");
        inline_buf[0x7f] = 0x80 | 6;
        // Heap SSO string at +0x80: "platform:/levels/gta6/test.rpf".
        let heap_str = b"platform:/levels/gta6/test.rpf\0";
        inline_buf[0x80..0x88].copy_from_slice(&(heap_str.as_ptr() as usize).to_le_bytes());
        inline_buf[0x88..0x8a].copy_from_slice(&30u16.to_le_bytes());
        let item_ptr = std::hint::black_box(&inline_buf).as_ptr() as usize;
        assert_eq!(read_rage_sso(item_ptr, 0x70), "data:/");
        assert_eq!(
            read_rage_sso(item_ptr, 0x80),
            "platform:/levels/gta6/test.rpf"
        );

        // Synthetic provider SwissTable (1 group = 0x310 bytes) with 0xc1fa30c2 state=5.
        let mut prov_table = vec![0x80u8; 0x310];
        let key: u32 = 0xc1fa30c2;
        prov_table[3] = (key & 0x7f) as u8;
        let slot3 = 0x10 + 3 * 0x30;
        prov_table[slot3..slot3 + 4].copy_from_slice(&key.to_le_bytes());
        prov_table[slot3 + 0x24..slot3 + 0x28].copy_from_slice(&5u32.to_le_bytes());

        let mut prov = vec![0u8; 0x1070];
        prov[0x1058..0x105c].copy_from_slice(&1u32.to_le_bytes());
        prov[0x1060..0x1068].copy_from_slice(&(prov_table.as_ptr() as usize).to_le_bytes());
        assert_eq!(lookup_provider_state(prov.as_ptr() as usize, key), Some(5));
        assert_eq!(
            lookup_provider_state(prov.as_ptr() as usize, 0x02e801f9),
            None
        );
        assert_eq!(dump_provider_table(1, prov_table.as_ptr() as usize), vec![(key, 5)]);

        // Synthetic ptr SwissTable (1 group = 0x110 bytes) mapping sub_id -> item_ptr.
        let mut sub_table = vec![0x80u8; 0x110];
        let sub_id: u32 = 0x12345678;
        sub_table[5] = (sub_id & 0x7f) as u8;
        let s5 = 0x10 + 5 * 0x10;
        sub_table[s5..s5 + 4].copy_from_slice(&sub_id.to_le_bytes());
        sub_table[s5 + 8..s5 + 16].copy_from_slice(&item_ptr.to_le_bytes());
        assert_eq!(
            lookup_ptr_swisstable(1, sub_table.as_ptr() as usize, sub_id),
            Some(item_ptr)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn trampoline_captures_rsi_r14_r15_and_preserves_all_registers() {
        static GOT: OnceLock<Mutex<(u64, u64, u64, u64)>> = OnceLock::new();
        extern "system" fn capture(_site: u32, regs: *const SavedRegs) {
            // SAFETY: `regs` is built on the current thread's stack by the trampoline.
            let r = unsafe { &*regs };
            let _ = GOT.set(Mutex::new((r.rax, r.rsi, r.r14, r.r15)));
        }

        // Allocate a tiny RWX page containing `mov r15, rax; mov rax, [rsi]; ret`,
        // patch it with our trampoline, call it, and confirm return & registers match.
        // SAFETY: our own anonymous RWX allocation, freed before return.
        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                0x2000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            assert!(!page.is_null());
            // Wrapper at +0x00:
            // push r14; push r15; push rsi; mov rsi, rcx; mov rax, rdx; mov r14, r8; mov r15, r9;
            // [SITE +0x16]: mov r15, rax; mov rax, [rsi] (6 bytes = SITES[ITEM_FOUND].original!)
            // add rax, r15; pop rsi; pop r15; pop r14; ret
            let mut fn_code: Vec<u8> = vec![
                0x41, 0x56, // push r14
                0x41, 0x57, // push r15
                0x56, // push rsi
                0x48, 0x89, 0xce, // mov rsi, rcx
                0x48, 0x89, 0xd0, // mov rax, rdx
                0x4d, 0x89, 0xc6, // mov r14, r8
                0x4d, 0x89, 0xcf, // mov r15, r9
            ];
            let site_off = fn_code.len();
            fn_code.extend_from_slice(SITES[ITEM_FOUND].original);
            fn_code.extend_from_slice(&[
                0x4c, 0x01, 0xf8, // add rax, r15
                0x5e, // pop rsi
                0x41, 0x5f, // pop r15
                0x41, 0x5e, // pop r14
                0xc3, // ret
            ]);
            std::ptr::copy_nonoverlapping(fn_code.as_ptr(), page, fn_code.len());

            let site_addr = page as usize + site_off;
            let tramp_addr = page as usize + 0x1000;
            let tramp = emit_trampoline(
                ITEM_FOUND as u32,
                SITES[ITEM_FOUND].original,
                site_addr + SITES[ITEM_FOUND].original.len(),
                tramp_addr,
                capture as *const () as usize,
            )
            .unwrap();
            std::ptr::copy_nonoverlapping(tramp.as_ptr(), tramp_addr as *mut u8, tramp.len());
            let jmp = emit_jump(site_addr, SITES[ITEM_FOUND].original.len(), tramp_addr).unwrap();
            std::ptr::copy_nonoverlapping(jmp.as_ptr(), site_addr as *mut u8, jmp.len());
            FlushInstructionCache(GetCurrentProcess(), page.cast(), 0x2000);

            let val: u64 = 0x1000;
            let f: extern "win64" fn(*const u64, u64, u64, u64) -> u64 = std::mem::transmute(page);
            let res = f(&val, 0x234, 0xaaaabbbb, 0xccccdddd);
            assert_eq!(res, 0x1234);
            let got = *GOT.get().unwrap().lock().unwrap();
            assert_eq!(got, (0x234, (&val as *const u64) as u64, 0xaaaabbbb, 0xccccdddd));

            VirtualFree(
                page.cast(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
        }
    }
}
