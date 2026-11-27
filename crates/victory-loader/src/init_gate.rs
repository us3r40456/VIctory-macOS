//! Native x86-64 inline probe for the Series 1.0.2.1 initialization gate
//! (`image+0x63db9..0x63ee7`).
//!
//! Replaces Frida Gadget / QuickJS with direct per-site trampolines so concurrent
//! worker threads under Wine/Rosetta never collide on shared JS/TLS state or
//! corrupt title registers. Every title-memory read uses `VirtualQuery` plus
//! `ReadProcessMemory`, and `GetCurrentThreadId` supplies the real Windows TID.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, LowerHex};
use std::fs::File;
use std::io::{self, Write};
use std::os::windows::io::AsHandle;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use windows_sys::Win32::System::Diagnostics::Debug::{FlushInstructionCache, ReadProcessMemory};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId};

use crate::crash::locate;

const TITLE_HASH: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const ASSERT_RVA: usize = 0x1dac734;
const SINGLETON_RVA: usize = 0x7f00510;
const GATE_END_RVA: usize = 0x64009;
const POOL_SIZE: usize = 0x1000;
const TRAMP_STRIDE: usize = 256;

const WORD: usize = 0;
const BYTE: usize = 1;
const OPTIONAL_CHECK: usize = 2;
const PREDICATE_CALL: usize = 3;
const PREDICATE_RETURN: usize = 4;
const ACCEPTED: usize = 5;
const SKIP: usize = 6;

const DEFAULT_INTERMEDIATE_LOG_LIMIT: u64 = 64;
const DEFAULT_ACCEPTED_LOG_LIMIT: u64 = 2048;

struct SiteSpec {
    name: &'static str,
    rva: usize,
    original: &'static [u8],
}

/// Seven non-RIP-relative sites spanning the three rejection paths and the
/// accepted / skipped destinations in `[0x63cd0, 0x64009)`.
const SITES: [SiteSpec; 7] = [
    SiteSpec {
        name: "word",
        rva: 0x63db9,
        // movzx eax, word ptr [rbx + 0xe4]
        original: &[0x0f, 0xb7, 0x83, 0xe4, 0, 0, 0],
    },
    SiteSpec {
        name: "byte",
        rva: 0x63dd1,
        // mov al, byte ptr [rbx + 0xe1]
        original: &[0x8a, 0x83, 0xe1, 0, 0, 0],
    },
    SiteSpec {
        name: "optional_check",
        rva: 0x63de8,
        // mov rax, qword ptr [rbx + 0xe8]
        original: &[0x48, 0x8b, 0x83, 0xe8, 0, 0, 0],
    },
    SiteSpec {
        name: "predicate_call",
        rva: 0x63e06,
        // lea rcx, [rax + 0xf]; mov rdx, rbx
        original: &[0x48, 0x8d, 0x48, 0x0f, 0x48, 0x89, 0xda],
    },
    SiteSpec {
        name: "predicate_return",
        rva: 0x63e14,
        // mov byte ptr [rbx + 0xf7], 0x1
        original: &[0xc6, 0x83, 0xf7, 0, 0, 0, 1],
    },
    SiteSpec {
        name: "accepted",
        rva: 0x63e23,
        // mov rax, qword ptr [r12]; mov rcx, r12
        original: &[0x49, 0x8b, 0x04, 0x24, 0x4c, 0x89, 0xe1],
    },
    SiteSpec {
        name: "skip_store",
        rva: 0x63edd,
        // mov dword ptr [rbx + 0xf0], 0x3
        original: &[0xc7, 0x83, 0xf0, 0, 0, 0, 3, 0, 0, 0],
    },
];

/// Control-flow bytes verified before patching so we only instrument the exact
/// gate layout.
const GUARDS: [(usize, &[u8]); 5] = [
    // 0x63dc0: test ax, ax; je +0x63dd1
    (0x63dc0, &[0x66, 0x85, 0xc0, 0x74, 0x0c]),
    // 0x63dcb: je +0x63edd
    (0x63dcb, &[0x0f, 0x84, 0x0c, 0x01, 0x00, 0x00]),
    // 0x63de2: jne +0x63edd
    (0x63de2, &[0x0f, 0x85, 0xf5, 0x00, 0x00, 0x00]),
    // 0x63e0d: and rcx, -0x10; call qword ptr [rax + 0x50]
    (0x63e0d, &[0x48, 0x83, 0xe1, 0xf0, 0xff, 0x50, 0x50]),
    // 0x63e1b: test al, al; je +0x63edd
    (0x63e1b, &[0x84, 0xc0, 0x0f, 0x84, 0xba, 0x00, 0x00, 0x00]),
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
    r12: u64,
    rsp: u64,
    rflags: u64,
}

#[derive(Default)]
struct Passage {
    number: u64,
    last_site: usize,
    predicate_target: Option<usize>,
    returned: Option<u8>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RejectSignature {
    cause: &'static str,
    callback: String,
    node_e4: Option<u16>,
    manager_5e: Option<u16>,
    node_e1: Option<u8>,
    manager_70: Option<u8>,
    node_e2: Option<u8>,
    predicate_target: String,
    predicate_al: Option<u8>,
}

#[derive(Default)]
struct ManagerStats {
    hits: u64,
    same_yes: u64,
    same_no: u64,
    last_5e: Option<u16>,
    last_70: Option<u8>,
}

#[derive(Default)]
struct GateCounts {
    visits: [u64; 7],
    rejects: [u64; 4],
    nodes: BTreeSet<usize>,
    managers: BTreeMap<usize, ManagerStats>,
    tids: BTreeSet<u32>,
    pending: BTreeMap<(u32, usize, usize, usize), Passage>,
    reject_sigs: BTreeMap<RejectSignature, u64>,
    replaced: u64,
    suppressed_lines: u64,
    intermediate_logged: u64,
    accepted_logged: u64,
}

struct Trace {
    image: usize,
    verbose: bool,
    output: File,
    counts: Mutex<GateCounts>,
    incomplete: AtomicBool,
    finished: AtomicBool,
    active: AtomicUsize,
}

static TRACE: OnceLock<Trace> = OnceLock::new();

/// Arms the native init-gate probe when `VICTORY_TRACE_INIT` is enabled (`1`,
/// `all`, or `verbose`). Must run after mapping and before any title thread
/// executes.
///
/// # Safety
/// `image..image + size` must be the mapped main title image for `hash`, with
/// no title threads running yet.
pub unsafe fn start(hash: &str, image: usize, size: usize) -> Result<()> {
    let mode = std::env::var("VICTORY_TRACE_INIT").unwrap_or_default();
    let mode = mode.trim();
    if mode.is_empty() || mode == "0" {
        return Ok(());
    }
    let verbose = matches!(mode, "all" | "verbose" | "2");
    ensure!(
        hash == TITLE_HASH,
        "init_gate trace could not arm: unsupported title hash ({hash})"
    );
    ensure!(
        image != 0 && image.checked_add(size).is_some() && size > SINGLETON_RVA + 8 && size > GATE_END_RVA,
        "init_gate trace could not arm: mapped image too small ({size:#x})"
    );
    let output = File::from(io::stderr().as_handle().try_clone_to_owned()?);
    TRACE
        .set(Trace {
            image,
            verbose,
            output,
            counts: Mutex::default(),
            incomplete: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            active: AtomicUsize::new(0),
        })
        .map_err(|_| anyhow::anyhow!("init_gate tracer already started"))?;

    // SAFETY: `image..image+size` is mapped and no title thread has started.
    let pool = match unsafe { install_native_detours(image, size, on_site) } {
        Ok(pool) => pool,
        Err(err) => {
            log_line(format_args!("init_gate trace could not arm: {err}"));
            return Err(err);
        }
    };
    log_line(format_args!(
        "init_gate armed mode=native verbose={verbose} image={image:#x} pool={pool:#x}"
    ));
    Ok(())
}

/// Summarizes the gate trace on the natural registry assert breakpoint.
pub(crate) fn on_breakpoint(address: usize) {
    if let Some(trace) = TRACE.get()
        && address == trace.image + ASSERT_RVA
    {
        finish(trace, "registry_assert");
    }
}

/// Summarizes the gate trace if the title exits before reaching the assert.
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
        log_line(format_args!(
            "init_gate final_table reason={reason} {}",
            snapshot_registry(trace.image)
        ));
        let start = Instant::now();
        while trace.active.load(Ordering::SeqCst) != 0
            && start.elapsed() < Duration::from_millis(25)
        {
            std::thread::yield_now();
        }
        let incomplete =
            trace.incomplete.load(Ordering::Relaxed) || trace.active.load(Ordering::SeqCst) != 0;
        match trace.counts.try_lock() {
            Ok(counts) => counts.emit_summary(reason, incomplete),
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner().emit_summary(reason, true),
            Err(std::sync::TryLockError::WouldBlock) => {
                log_line(format_args!(
                    "init_gate summary reason={reason} io_complete=false counts=unavailable"
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
        // SAFETY: the trampoline builds `SavedRegs` on the current thread's stack.
        let r = unsafe { &*regs };
        // SAFETY: plain Win32 call returning the current Windows thread ID.
        let tid = unsafe { GetCurrentThreadId() };
        let site = site as usize;
        if site >= SITES.len() {
            trace.incomplete.store(true, Ordering::Relaxed);
            return;
        }
        let mut counts = trace.counts.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(line) = counts.visit(
            site,
            tid,
            r.rsp as usize,
            r.rbx as usize,
            r.r12 as usize,
            r.rax as usize,
            trace.verbose,
        ) {
            log_line(format_args!("{line}"));
        }
    });
    if ok.is_err() {
        trace.incomplete.store(true, Ordering::Relaxed);
    }
    trace.active.fetch_sub(1, Ordering::SeqCst);
}

impl GateCounts {
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        site: usize,
        tid: u32,
        sp: usize,
        node: usize,
        manager: usize,
        rax: usize,
        verbose: bool,
    ) -> Option<String> {
        self.visits[site] += 1;
        self.tids.insert(tid);
        self.nodes.insert(node);
        let seq: u64 = self.visits.iter().sum();

        let node_80 = read_usize(node, 0x80);
        let same_manager = match node_80 {
            Some(p) if p != 0 && p == manager => "yes",
            Some(_) => "no",
            None => "?",
        };
        let callback_ptr = read_usize(node, 0x50);
        let callback = fmt_ptr_or_symbol(callback_ptr);
        let node_e4 = read_u16(node, 0xe4);
        let manager_5e = read_u16(manager, 0x5e);
        let node_e1 = read_u8(node, 0xe1);
        let manager_70 = read_u8(manager, 0x70);
        let node_e2 = read_u8(node, 0xe2);
        let node_e8 = read_usize(node, 0xe8);
        let node_f7 = read_u8(node, 0xf7);
        let node_d8 = read_u8(node, 0xd8);

        let mstat = self.managers.entry(manager).or_default();
        mstat.hits += 1;
        if same_manager == "yes" {
            mstat.same_yes += 1;
        } else if same_manager == "no" {
            mstat.same_no += 1;
        }
        if manager_5e.is_some() {
            mstat.last_5e = manager_5e;
        }
        if manager_70.is_some() {
            mstat.last_70 = manager_70;
        }

        let key = (tid, sp, node, manager);
        if site == WORD
            && self
                .pending
                .insert(
                    key,
                    Passage {
                        number: seq,
                        ..Default::default()
                    },
                )
                .is_some()
        {
            self.replaced += 1;
        }

        let mut outcome = "?";
        let mut passage = self.pending.get_mut(&key);
        if let Some(p) = passage.as_deref_mut() {
            match site {
                PREDICATE_CALL => {
                    p.predicate_target = read_usize(rax, 0x50);
                }
                PREDICATE_RETURN => {
                    p.returned = (p.last_site == PREDICATE_CALL).then_some(rax as u8);
                }
                SKIP => {
                    let (index, label) = match p.last_site {
                        WORD => (0, "word"),
                        BYTE => (1, "byte"),
                        PREDICATE_RETURN if p.returned == Some(0) => (2, "predicate"),
                        _ => (3, "unknown"),
                    };
                    self.rejects[index] += 1;
                    outcome = label;
                }
                _ => {}
            }
            p.last_site = site;
        } else if site == SKIP {
            self.rejects[3] += 1;
            outcome = "unknown";
        }

        let pass = passage
            .as_ref()
            .map(|p| p.number.to_string())
            .unwrap_or_else(|| "?".into());
        let target_ptr = passage
            .as_ref()
            .and_then(|p| p.predicate_target)
            .or_else(|| {
                if matches!(site, OPTIONAL_CHECK | SKIP) {
                    node_e8.filter(|&p| p != 0).and_then(|p| read_usize(p, 0x50))
                } else {
                    None
                }
            });
        let target = fmt_ptr_or_symbol(target_ptr);
        let returned_byte = passage.as_ref().and_then(|p| p.returned);
        let returned = fmt_hex(returned_byte);

        if site == SKIP {
            let sig = RejectSignature {
                cause: outcome,
                callback: callback.clone(),
                node_e4,
                manager_5e,
                node_e1,
                manager_70,
                node_e2,
                predicate_target: target.clone(),
                predicate_al: returned_byte,
            };
            *self.reject_sigs.entry(sig).or_default() += 1;
        }

        if matches!(site, ACCEPTED | SKIP) {
            self.pending.remove(&key);
        }

        let always_interesting = matches!(site, SKIP | PREDICATE_CALL | PREDICATE_RETURN)
            || same_manager == "no"
            || (site == OPTIONAL_CHECK && node_e8.is_some_and(|p| p != 0));
        let should_log = if verbose || always_interesting {
            true
        } else if site == ACCEPTED {
            if self.accepted_logged < DEFAULT_ACCEPTED_LOG_LIMIT {
                self.accepted_logged += 1;
                true
            } else {
                self.suppressed_lines += 1;
                false
            }
        } else if self.intermediate_logged < DEFAULT_INTERMEDIATE_LOG_LIMIT {
            self.intermediate_logged += 1;
            true
        } else {
            self.suppressed_lines += 1;
            false
        };

        if !should_log {
            return None;
        }

        let label = SITES[site].name;
        Some(format!(
            "init_gate seq={seq} pass={pass} site={label} tid={tid} sp={sp:#x} \
             node={node:#x} manager={manager:#x} node_80={} same_manager={same_manager} \
             callback={callback} node_e4={} manager_5e={} node_e1={} manager_70={} node_e2={} \
             node_e8={} node_f7={} node_d8={} reject={outcome} predicate_target={target} predicate_al={returned}",
            fmt_hex(node_80),
            fmt_hex(node_e4),
            fmt_hex(manager_5e),
            fmt_hex(node_e1),
            fmt_hex(manager_70),
            fmt_hex(node_e2),
            fmt_hex(node_e8),
            fmt_hex(node_f7),
            fmt_hex(node_d8),
        ))
    }

    fn emit_summary(&self, reason: &str, incomplete: bool) {
        let [
            word,
            byte,
            optional_check,
            optional_calls,
            optional_returns,
            accepted,
            skipped,
        ] = self.visits;
        let [r1, r2, r3, unknown] = self.rejects;
        log_line(format_args!(
            "init_gate summary reason={reason} word={word} byte={byte} optional_check={optional_check} \
             predicate_calls={optional_calls} predicate_returns={optional_returns} \
             accepted={accepted} skip_store={skipped} \
             reject_word={r1} reject_byte={r2} reject_predicate={r3} reject_unknown={unknown} \
             distinct_nodes={} distinct_managers={} distinct_tids={} open={} replaced={} \
             suppressed={} io_complete={}",
            self.nodes.len(),
            self.managers.len(),
            self.tids.len(),
            self.pending.len(),
            self.replaced,
            self.suppressed_lines,
            !incomplete,
        ));
        for (mgr, stat) in &self.managers {
            log_line(format_args!(
                "init_gate manager addr={mgr:#x} hits={} same_yes={} same_no={} manager_5e={} manager_70={}",
                stat.hits,
                stat.same_yes,
                stat.same_no,
                fmt_hex(stat.last_5e),
                fmt_hex(stat.last_70),
            ));
        }
        for (sig, count) in &self.reject_sigs {
            log_line(format_args!(
                "init_gate reject_group count={count} reject={} callback={} \
                 node_e4={} manager_5e={} node_e1={} manager_70={} node_e2={} \
                 predicate_target={} predicate_al={}",
                sig.cause,
                sig.callback,
                fmt_hex(sig.node_e4),
                fmt_hex(sig.manager_5e),
                fmt_hex(sig.node_e1),
                fmt_hex(sig.manager_70),
                fmt_hex(sig.node_e2),
                sig.predicate_target,
                fmt_hex(sig.predicate_al),
            ));
        }
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

fn fmt_ptr_or_symbol(value: Option<usize>) -> String {
    match value {
        None => "?".into(),
        Some(0) => "0x0".into(),
        Some(p) => locate(p),
    }
}

fn read_bytes<const N: usize>(address: usize) -> Option<[u8; N]> {
    if address == 0 {
        return None;
    }
    let end = address.checked_add(N - 1)?;
    for page in [address, end] {
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
    }
    let mut data = [0u8; N];
    let mut count = 0;
    // SAFETY: `ReadProcessMemory` safely copies into `data` and handles
    // concurrent unmaps/protection changes without faulting.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            address as *const _,
            data.as_mut_ptr().cast(),
            N,
            &mut count,
        )
    };
    (ok != 0 && count == N).then_some(data)
}

fn read_field<const N: usize>(base: usize, offset: usize) -> Option<[u8; N]> {
    if base == 0 {
        return None;
    }
    read_bytes(base.checked_add(offset)?)
}

fn read_u8(base: usize, offset: usize) -> Option<u8> {
    read_field::<1>(base, offset).map(|b| b[0])
}

fn read_u16(base: usize, offset: usize) -> Option<u16> {
    read_field::<2>(base, offset).map(u16::from_le_bytes)
}

fn read_usize(base: usize, offset: usize) -> Option<usize> {
    read_field::<8>(base, offset).map(usize::from_le_bytes)
}

fn snapshot_registry(image: usize) -> String {
    let singleton = read_usize(image, SINGLETON_RVA);
    let descriptor = singleton
        .filter(|&g| g != 0)
        .and_then(|g| read_usize(g, 0x2a8));
    let cap = descriptor
        .filter(|&d| d != 0)
        .and_then(|d| read_field::<4>(d, 8))
        .map(u32::from_le_bytes);
    let base = descriptor
        .filter(|&d| d != 0)
        .and_then(|d| read_usize(d, 0x10));
    format!(
        "g={} descriptor={} cap={} base={}",
        fmt_hex(singleton),
        fmt_hex(descriptor),
        cap.map_or_else(|| "?".into(), |n| n.to_string()),
        fmt_hex(base),
    )
}

/// Emits one native x86-64 trampoline that preserves every volatile register,
/// `RFLAGS`, and `XMM0..XMM5`, calls `handler(site_index, &SavedRegs)` with a
/// 16-byte aligned Win64 stack frame, runs `original` verbatim, and jumps to
/// `return_to`.
fn emit_trampoline(
    site_index: u32,
    original: &[u8],
    return_to: usize,
    tramp_addr: usize,
    handler: usize,
) -> Option<Vec<u8>> {
    let mut c = Vec::with_capacity(224);
    // Save RFLAGS first (before `and`/`sub` modify flags), then anchor RSP via RBP.
    c.push(0x9c); // pushfq
    c.push(0x55); // push rbp
    c.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
    c.extend_from_slice(&[0x48, 0x83, 0xe4, 0xf0]); // and rsp, -16
    // 0x20 shadow space + 0x60 SavedRegs + 0x60 XMM0..XMM5 = 0xe0 (16-byte aligned).
    c.extend_from_slice(&[0x48, 0x81, 0xec, 0xe0, 0x00, 0x00, 0x00]); // sub rsp, 0xe0

    // Populate SavedRegs at [rsp + 0x20].
    c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x20]); // mov [rsp+0x20], rax
    c.extend_from_slice(&[0x48, 0x89, 0x4c, 0x24, 0x28]); // mov [rsp+0x28], rcx
    c.extend_from_slice(&[0x48, 0x89, 0x54, 0x24, 0x30]); // mov [rsp+0x30], rdx
    c.extend_from_slice(&[0x4c, 0x89, 0x44, 0x24, 0x38]); // mov [rsp+0x38], r8
    c.extend_from_slice(&[0x4c, 0x89, 0x4c, 0x24, 0x40]); // mov [rsp+0x40], r9
    c.extend_from_slice(&[0x4c, 0x89, 0x54, 0x24, 0x48]); // mov [rsp+0x48], r10
    c.extend_from_slice(&[0x4c, 0x89, 0x5c, 0x24, 0x50]); // mov [rsp+0x50], r11
    c.extend_from_slice(&[0x48, 0x89, 0x5c, 0x24, 0x58]); // mov [rsp+0x58], rbx
    c.extend_from_slice(&[0x4c, 0x89, 0x64, 0x24, 0x60]); // mov [rsp+0x60], r12
    c.extend_from_slice(&[0x48, 0x8d, 0x45, 0x10]); // lea rax, [rbp+0x10] (orig rsp)
    c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x68]); // mov [rsp+0x68], rax
    c.extend_from_slice(&[0x48, 0x8b, 0x45, 0x08]); // mov rax, [rbp+0x08] (orig rflags)
    c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x70]); // mov [rsp+0x70], rax

    // Save volatile XMM0..XMM5 at [rsp+0x80..0xe0].
    for i in 0u8..6 {
        let modrm = 0x84 | (i << 3);
        let disp = (0x80u32 + u32::from(i) * 16).to_le_bytes();
        c.extend_from_slice(&[0x0f, 0x11, modrm, 0x24]);
        c.extend_from_slice(&disp);
    }

    // Win64 call: DF=0, ecx = site_index, rdx = &SavedRegs, call handler.
    c.push(0xfc); // cld
    c.push(0xb9); // mov ecx, imm32
    c.extend_from_slice(&site_index.to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8d, 0x54, 0x24, 0x20]); // lea rdx, [rsp+0x20]
    c.extend_from_slice(&[0x48, 0xb8]); // mov rax, imm64
    c.extend_from_slice(&(handler as u64).to_le_bytes());
    c.extend_from_slice(&[0xff, 0xd0]); // call rax

    // Restore volatile XMM0..XMM5.
    for i in 0u8..6 {
        let modrm = 0x84 | (i << 3);
        let disp = (0x80u32 + u32::from(i) * 16).to_le_bytes();
        c.extend_from_slice(&[0x0f, 0x10, modrm, 0x24]);
        c.extend_from_slice(&disp);
    }

    // Restore volatile GPRs. Non-volatile registers (rbx, r12, etc.) were
    // preserved by the `extern "system"` handler.
    c.extend_from_slice(&[0x4c, 0x8b, 0x5c, 0x24, 0x50]); // mov r11, [rsp+0x50]
    c.extend_from_slice(&[0x4c, 0x8b, 0x54, 0x24, 0x48]); // mov r10, [rsp+0x48]
    c.extend_from_slice(&[0x4c, 0x8b, 0x4c, 0x24, 0x40]); // mov r9,  [rsp+0x40]
    c.extend_from_slice(&[0x4c, 0x8b, 0x44, 0x24, 0x38]); // mov r8,  [rsp+0x38]
    c.extend_from_slice(&[0x48, 0x8b, 0x54, 0x24, 0x30]); // mov rdx, [rsp+0x30]
    c.extend_from_slice(&[0x48, 0x8b, 0x4c, 0x24, 0x28]); // mov rcx, [rsp+0x28]
    c.extend_from_slice(&[0x48, 0x8b, 0x44, 0x24, 0x20]); // mov rax, [rsp+0x20]

    // Restore RSP, RBP, and RFLAGS.
    c.extend_from_slice(&[0x48, 0x89, 0xec]); // mov rsp, rbp
    c.push(0x5d); // pop rbp
    c.push(0x9d); // popfq

    // Execute the exact relocated original instruction(s).
    c.extend_from_slice(original);

    // Relative jump back to `return_to`.
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
    let try_hint = |hint: usize| -> Option<*mut u8> {
        if hint == 0 {
            return None;
        }
        // SAFETY: reserving a fresh anonymous page at a hint; returns null if unavailable.
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
        // Verify every site and trampoline endpoint stays comfortably within rel32 reach.
        let gate_lo = image + SITES[0].rva;
        let gate_hi = image + SITES[SITES.len() - 1].rva + 16;
        let dist_lo = (addr as i128 - gate_lo as i128).abs();
        let dist_hi = ((addr + POOL_SIZE) as i128 - gate_hi as i128).abs();
        if dist_lo < (i32::MAX as i128 - 0x10000) && dist_hi < (i32::MAX as i128 - 0x10000) {
            Some(ptr)
        } else {
            // SAFETY: freeing the allocation we just created because it fell out of rel32 reach.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualFree(
                    ptr.cast(),
                    0,
                    windows_sys::Win32::System::Memory::MEM_RELEASE,
                );
            }
            None
        }
    };

    // Try below the image first (same strategy as `masked.rs`), then above.
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

/// Verifies all gate site and guard bytes, builds the 7 native trampolines,
/// and patches the 7 sites in place.
///
/// # Safety
/// `image..image+image_size` must be a mapped code region containing the
/// verified gate layout, with no threads executing inside it.
unsafe fn install_native_detours(
    image: usize,
    image_size: usize,
    handler: extern "system" fn(u32, *const SavedRegs),
) -> Result<usize> {
    // 1. Verify every site and guard byte BEFORE modifying anything.
    for spec in &SITES {
        ensure!(
            spec.rva + spec.original.len() <= image_size,
            "site {} at image+{:#x} exceeds image bounds",
            spec.name,
            spec.rva
        );
        // SAFETY: `spec.rva..spec.rva + len` lies within `image..image+image_size`.
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
        // SAFETY: `rva..rva + len` lies within `image..image+image_size`.
        let found = unsafe { std::slice::from_raw_parts((image + rva) as *const u8, expected.len()) };
        if found != *expected {
            bail!("unexpected guard bytes at image+{rva:#x}");
        }
    }

    // 2. Allocate trampoline pool within rel32 reach of the gate.
    let pool = alloc_near(image, image_size)
        .ok_or_else(|| anyhow::anyhow!("no executable memory within rel32 reach of {image:#x}"))?;
    let pool_addr = pool as usize;

    // 3. Emit all 7 trampolines and compute all 7 jump patches before patching title code.
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
        // SAFETY: `tramp_offset + tramp.len() <= POOL_SIZE` in our freshly allocated writable pool.
        unsafe {
            std::ptr::copy_nonoverlapping(tramp.as_ptr(), pool.add(tramp_offset), tramp.len());
        }
        patches.push((site_addr, jump));
    }

    // 4. Seal trampoline pool RX before routing execution to it.
    let mut old = 0;
    // SAFETY: `pool..pool+POOL_SIZE` is our allocation from `alloc_near`.
    let sealed = unsafe { VirtualProtect(pool.cast(), POOL_SIZE, PAGE_EXECUTE_READ, &mut old) };
    ensure!(sealed != 0, "VirtualProtect(RX) on trampoline pool failed");

    // 5. Patch all 7 sites in the title image.
    for (site_addr, jump) in &patches {
        let mut prev = 0;
        // SAFETY: verified code site in `image`; no title threads are executing yet.
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
    // SAFETY: flushing the process instruction cache after patching code.
    unsafe { FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0) };
    Ok(pool_addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn gate_counts_classifies_all_three_rejections_and_preserves_missing_fields() {
        let mut counts = GateCounts::default();
        let mut manager = [0u8; 0x80];
        manager[0x5e..0x60].copy_from_slice(&0x000fu16.to_le_bytes());
        manager[0x70] = 0x00;
        let mp = std::hint::black_box(&manager).as_ptr() as usize;

        let mut node = [0u8; 0x100];
        node[0x80..0x88].copy_from_slice(&mp.to_le_bytes());
        node[0xe4..0xe6].copy_from_slice(&0x0010u16.to_le_bytes());
        node[0xe1] = 0x01;
        node[0xe2] = 0x00;
        let np = std::hint::black_box(&node).as_ptr() as usize;

        // Word rejection.
        counts.visit(WORD, 10, 0x1000, np, mp, 0, true);
        let line = counts.visit(SKIP, 10, 0x1000, np, mp, 0, true).unwrap();
        assert!(line.contains("reject=word"), "{line}");
        assert!(line.contains("same_manager=yes"), "{line}");

        // Byte rejection.
        counts.visit(WORD, 11, 0x2000, np, mp, 0, true);
        counts.visit(BYTE, 11, 0x2000, np, mp, 0, true);
        let line = counts.visit(SKIP, 11, 0x2000, np, mp, 0, true).unwrap();
        assert!(line.contains("reject=byte"), "{line}");

        // Predicate rejection with real method pointer at [obj + 0x50].
        let mut pred = [0usize; 12];
        pred[10] = locate as *const () as usize;
        let pred_ptr = std::hint::black_box(&pred).as_ptr() as usize;
        node[0xe8..0xf0].copy_from_slice(&pred_ptr.to_le_bytes());
        counts.visit(WORD, 12, 0x3000, np, mp, 0, true);
        counts.visit(BYTE, 12, 0x3000, np, mp, 0, true);
        counts.visit(OPTIONAL_CHECK, 12, 0x3000, np, mp, 0, true);
        counts.visit(PREDICATE_CALL, 12, 0x3000, np, mp, pred_ptr, true);
        counts.visit(PREDICATE_RETURN, 12, 0x3000, np, mp, 0x100, true);
        let line = counts.visit(SKIP, 12, 0x3000, np, mp, 0, true).unwrap();
        assert!(line.contains("reject=predicate"), "{line}");
        assert!(line.contains("predicate_al=0x0"), "{line}");
        assert!(line.contains(&format!("predicate_target={}", locate(pred[10]))), "{line}");

        // Unreadable / wild manager pointer must never fault and must format as `?`.
        let bad = counts.visit(WORD, 13, 0x4000, np, 0x30, 0, true).unwrap();
        assert!(bad.contains("same_manager=no"), "{bad}");
        assert!(bad.contains("manager_5e=?"), "{bad}");
        assert!(bad.contains("manager_70=?"), "{bad}");

        assert_eq!(counts.rejects, [1, 1, 1, 0]);
        assert_eq!(counts.reject_sigs.len(), 3);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_trampolines_run_safely_across_16_concurrent_threads() {
        static TEST_COUNTS: OnceLock<Mutex<GateCounts>> = OnceLock::new();
        extern "system" fn test_handler(site: u32, regs: *const SavedRegs) {
            // SAFETY: `regs` is built on the current thread's stack by the trampoline.
            let r = unsafe { &*regs };
            // SAFETY: returns current thread ID.
            let tid = unsafe { GetCurrentThreadId() };
            let mut c = TEST_COUNTS.get().unwrap().lock().unwrap();
            c.visit(
                site as usize,
                tid,
                r.rsp as usize,
                r.rbx as usize,
                r.r12 as usize,
                r.rax as usize,
                false,
            );
        }
        extern "system" fn pred_false(_obj: usize, _node: usize) -> u8 {
            0
        }
        extern "system" fn pred_true(_obj: usize, _node: usize) -> u8 {
            1
        }
        extern "system" fn manager_register(_mgr: usize, node: usize, _flag: u8) {
            // Mark accepted state = 4 at [node + 0xf0] (mirrors the real accepted path).
            // SAFETY: `node` points to our test node buffer of 0x100 bytes.
            unsafe { ((node + 0xf0) as *mut u32).write(4) };
        }

        let _ = TEST_COUNTS.set(Mutex::default());

        // Allocate a synthetic image region covering up to GATE_END_RVA + 0x100.
        let image_size = 0x70000usize;
        // SAFETY: fresh anonymous RWX allocation for the synthetic gate function.
        let image = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                image_size,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
        } as *mut u8;
        assert!(!image.is_null());
        let base = image as usize;

        // Place a clean Win64 wrapper at 0x63d90 that sets rbx=rcx (node), r12=rdx (manager),
        // then falls straight into the exact Series 1.0.2.1 gate bytes at 0x63db9..0x63ef0.
        // SAFETY: writing inside our own RWX allocation.
        unsafe {
            let entry = 0x63da0usize;
            // push rbx; push r12; sub rsp, 0x28; mov rbx, rcx; mov r12, rdx; jmp 0x63db9
            let prolog: [u8; 16] = [
                0x53,                   // push rbx
                0x41, 0x54,             // push r12
                0x48, 0x83, 0xec, 0x28, // sub rsp, 0x28
                0x48, 0x89, 0xcb,       // mov rbx, rcx
                0x49, 0x89, 0xd4,       // mov r12, rdx
                0xeb, 0x09,             // jmp +9 -> 0x63db9 (entry 0x63da0 + 15 + 9 = 0x63db8? wait: calc below!)
                0x90,
            ];
            std::ptr::copy_nonoverlapping(prolog.as_ptr(), image.add(entry), prolog.len());
            // Fix exact short jmp offset from `entry + 15` to `0x63db9`.
            let disp = (0x63db9isize - (entry + 15) as isize) as i8;
            *image.add(entry + 14) = disp as u8;

            // Exact bytes of `0x63db9..0x63e33` from the real title binary:
            let gate_body: &[u8] = &[
                // 0x63db9: movzx eax, word ptr [rbx+0xe4]
                0x0f, 0xb7, 0x83, 0xe4, 0x00, 0x00, 0x00,
                // 0x63dc0: test ax, ax; je 0x63dd1
                0x66, 0x85, 0xc0, 0x74, 0x0c,
                // 0x63dc5: and ax, word ptr [r12+0x5e]
                0x66, 0x41, 0x23, 0x44, 0x24, 0x5e,
                // 0x63dcb: je 0x63edd
                0x0f, 0x84, 0x0c, 0x01, 0x00, 0x00,
                // 0x63dd1: mov al, byte ptr [rbx+0xe1]
                0x8a, 0x83, 0xe1, 0x00, 0x00, 0x00,
                // 0x63dd7: and al, byte ptr [r12+0x70]
                0x41, 0x22, 0x44, 0x24, 0x70,
                // 0x63ddc: cmp al, byte ptr [rbx+0xe2]
                0x3a, 0x83, 0xe2, 0x00, 0x00, 0x00,
                // 0x63de2: jne 0x63edd
                0x0f, 0x85, 0xf5, 0x00, 0x00, 0x00,
                // 0x63de8: mov rax, qword ptr [rbx+0xe8]
                0x48, 0x8b, 0x83, 0xe8, 0x00, 0x00, 0x00,
                // 0x63def: test rax, rax; je 0x63e23
                0x48, 0x85, 0xc0, 0x74, 0x2f,
                // 0x63df4: cmp byte ptr [rbx+0xf7], 1; jne 0x63e06
                0x80, 0xbb, 0xf7, 0x00, 0x00, 0x00, 0x01, 0x75, 0x09,
                // 0x63dfd: test byte ptr [rbx+0xd8], 0x10; je 0x63e23
                0xf6, 0x83, 0xd8, 0x00, 0x00, 0x00, 0x10, 0x74, 0x1d,
                // 0x63e06: lea rcx, [rax+0xf]; mov rdx, rbx; and rcx, -0x10; call [rax+0x50]
                0x48, 0x8d, 0x48, 0x0f,
                0x48, 0x89, 0xda,
                0x48, 0x83, 0xe1, 0xf0,
                0xff, 0x50, 0x50,
                // 0x63e14: mov byte ptr [rbx+0xf7], 1
                0xc6, 0x83, 0xf7, 0x00, 0x00, 0x00, 0x01,
                // 0x63e1b: test al, al; je 0x63edd
                0x84, 0xc0, 0x0f, 0x84, 0xba, 0x00, 0x00, 0x00,
                // 0x63e23: mov rax, [r12]; mov rcx, r12; mov rdx, rbx; mov r8b, 1; call [rax+0x20]
                0x49, 0x8b, 0x04, 0x24,
                0x4c, 0x89, 0xe1,
                0x48, 0x89, 0xda,
                0x41, 0xb0, 0x01,
                0xff, 0x50, 0x20,
                // Epilogue at 0x63e33: mov eax, [rbx+0xf0]; add rsp, 0x28; pop r12; pop rbx; ret
                0x8b, 0x83, 0xf0, 0x00, 0x00, 0x00,
                0x48, 0x83, 0xc4, 0x28,
                0x41, 0x5c,
                0x5b,
                0xc3,
            ];
            std::ptr::copy_nonoverlapping(gate_body.as_ptr(), image.add(0x63db9), gate_body.len());

            // Exact bytes at 0x63edd (skip_store) + epilogue:
            let skip_body: &[u8] = &[
                // 0x63edd: mov dword ptr [rbx+0xf0], 3
                0xc7, 0x83, 0xf0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
                // 0x63ee7: mov eax, [rbx+0xf0]; add rsp, 0x28; pop r12; pop rbx; ret
                0x8b, 0x83, 0xf0, 0x00, 0x00, 0x00,
                0x48, 0x83, 0xc4, 0x28,
                0x41, 0x5c,
                0x5b,
                0xc3,
            ];
            std::ptr::copy_nonoverlapping(skip_body.as_ptr(), image.add(0x63edd), skip_body.len());

            // First confirm mismatched guard bytes fail cleanly without patching.
            *image.add(0x63dc0) ^= 0xff;
            assert!(install_native_detours(base, image_size, test_handler).is_err());
            assert_eq!(*image.add(0x63db9), SITES[0].original[0]);
            *image.add(0x63dc0) ^= 0xff;

            // Now arm the real native detours.
            let pool = install_native_detours(base, image_size, test_handler).unwrap();
            assert_ne!(pool, 0);

            let gate_fn: extern "win64" fn(*mut u8, *mut u8) -> u32 =
                std::mem::transmute(image.add(entry));

            // Shared manager vtable & manager object (mgr_5e = 0x000f, mgr_70 = 0x00).
            let mut vtable = [0usize; 8];
            vtable[4] = manager_register as *const () as usize; // slot +0x20
            let mut mgr = [0u8; 0x80];
            mgr[0..8].copy_from_slice(&(vtable.as_ptr() as usize).to_le_bytes());
            mgr[0x5e..0x60].copy_from_slice(&0x000fu16.to_le_bytes());
            mgr[0x70] = 0x00;

            let mut pred_obj_false = [0usize; 12];
            pred_obj_false[10] = pred_false as *const () as usize; // slot +0x50
            let mut pred_obj_true = [0usize; 12];
            pred_obj_true[10] = pred_true as *const () as usize; // slot +0x50

            let mgr_addr = Arc::new(mgr.as_ptr() as usize);
            let pf_addr = pred_obj_false.as_ptr() as usize;
            let pt_addr = pred_obj_true.as_ptr() as usize;

            const THREADS: usize = 16;
            const ROUNDS: usize = 100;
            let mut handles = Vec::with_capacity(THREADS);
            for _ in 0..THREADS {
                let mgr_a = Arc::clone(&mgr_addr);
                handles.push(std::thread::spawn(move || {
                    for _ in 0..ROUNDS {
                        let run_case = |e4: u16, e1: u8, e2: u8, e8: usize, f7: u8| -> u32 {
                            let mut node = [0u8; 0x100];
                            node[0x80..0x88].copy_from_slice(&(*mgr_a).to_le_bytes());
                            node[0xe4..0xe6].copy_from_slice(&e4.to_le_bytes());
                            node[0xe1] = e1;
                            node[0xe2] = e2;
                            node[0xe8..0xf0].copy_from_slice(&e8.to_le_bytes());
                            node[0xf7] = f7;
                            gate_fn(node.as_mut_ptr(), *mgr_a as *mut u8)
                        };
                        // 1. reject=word
                        assert_eq!(run_case(0x0010, 1, 0, 0, 0), 3);
                        // 2. reject=byte
                        assert_eq!(run_case(0x0000, 1, 1, 0, 0), 3);
                        // 3. reject=predicate
                        assert_eq!(run_case(0x0000, 1, 0, pf_addr, 0), 3);
                        // 4. accepted via predicate returning 1
                        assert_eq!(run_case(0x0000, 1, 0, pt_addr, 0), 4);
                        // 5. accepted with null predicate
                        assert_eq!(run_case(0x0000, 1, 0, 0, 0), 4);
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }

            let c = TEST_COUNTS.get().unwrap().lock().unwrap();
            let n = (THREADS * ROUNDS) as u64;
            assert_eq!(c.visits[WORD], 5 * n);
            assert_eq!(c.visits[BYTE], 4 * n);
            assert_eq!(c.visits[OPTIONAL_CHECK], 3 * n);
            assert_eq!(c.visits[PREDICATE_CALL], 2 * n);
            assert_eq!(c.visits[PREDICATE_RETURN], 2 * n);
            assert_eq!(c.visits[ACCEPTED], 2 * n);
            assert_eq!(c.visits[SKIP], 3 * n);
            assert_eq!(c.rejects, [n, n, n, 0]);
            assert_eq!(c.pending.len(), 0);
            assert_eq!(c.replaced, 0);
            assert!(c.tids.len() >= 2 && !c.tids.contains(&0), "tids={:?}", c.tids);
            assert_eq!(c.managers.len(), 1);
            assert_eq!(c.managers[&*mgr_addr].same_no, 0);

            windows_sys::Win32::System::Memory::VirtualFree(
                image.cast(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
            windows_sys::Win32::System::Memory::VirtualFree(
                pool as *mut _,
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
        }
    }
}
