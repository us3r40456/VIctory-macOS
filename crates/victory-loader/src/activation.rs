//! Observes the registration and delivery of one title event on Series 1.0.2.1.
//! The embedded diagnostic script installs both listeners before any title code runs.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsHandle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows_sys::Win32::System::LibraryLoader::{LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId};

#[path = "activation/events.rs"]
mod events;
#[path = "activation/init_gate.rs"]
mod init_gate;

const TITLE_HASH: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const GADGET_HASH: &str = "2933e407ac763473532c2ca7f3b0ae5f90678bba9392bbd0365ee1c558836a2d";
const ASSERT_RVA: usize = 0x1dac734;
const SINGLETON_RVA: usize = 0x7f00510;
const ARM_TIMEOUT: Duration = Duration::from_secs(60);

struct Trace {
    image: usize,
    size: usize,
    output: File,
    events: Mutex<events::Events>,
    gate: Option<Mutex<init_gate::GateCounts>>,
    incomplete: AtomicBool,
    finished: AtomicBool,
    active: AtomicUsize,
}

static TRACE: OnceLock<Trace> = OnceLock::new();
static ARMED: AtomicI8 = AtomicI8::new(0);

/// Start the opt-in trace after mapping the verified image, before title execution.
///
/// # Safety
/// `hash` must identify the original executable for the mapped range. Its code must
/// remain loaded for the lifetime of the process, with no title thread running yet.
pub unsafe fn start(hash: &str, image: usize, size: usize) -> Result<()> {
    let gate = std::env::var("VICTORY_TRACE_INIT").as_deref() == Ok("1");
    if !gate && std::env::var("VICTORY_TRACE_ACTIVATION").as_deref() != Ok("1") {
        return Ok(());
    }
    ensure!(
        hash == TITLE_HASH,
        "unsupported activation trace image ({hash})"
    );
    ensure!(
        image != 0 && image.checked_add(size).is_some() && size >= SINGLETON_RVA + 8,
        "activation trace RVAs fall outside the mapped title"
    );
    ensure!(
        std::env::var_os("VICTORY_SKIP_TITLE_INT3").is_none_or(|v| v.is_empty()),
        "unset VICTORY_SKIP_TITLE_INT3 to observe the natural title assert"
    );
    let dir = PathBuf::from(std::env::var_os("VICTORY_ACTIVATION_SESSION").context(
        "activation trace requires a fresh VICTORY_ACTIVATION_SESSION from the launcher",
    )?);
    ensure!(
        dir.is_absolute() && fs::symlink_metadata(&dir).is_ok_and(|m| m.is_dir()),
        "expected an absolute activation session directory: {}",
        dir.display()
    );
    let dll = dir.join("frida-gadget.dll");
    ensure!(
        fs::symlink_metadata(&dll)?.is_file(),
        "expected a regular frida-gadget.dll"
    );
    // Keep the verified DLL open with FILE_SHARE_READ, preventing replacement during loading.
    let mut pinned = File::options()
        .read(true)
        .share_mode(1)
        .open(&dll)
        .with_context(|| format!("opening {}", dll.display()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(&mut pinned)?;
    ensure!(
        hasher.finalize().to_hex().as_str() == GADGET_HASH,
        "activation Gadget DLL hash mismatch"
    );

    // Each mode stages a fixed script. Gate mode samples only the init routine.
    let (script, callback): (&[u8], usize) = if gate {
        (
            include_bytes!("init-gate.js"),
            init_gate::record as *const () as usize,
        )
    } else {
        (
            include_bytes!("activation.js"),
            record as *const () as usize,
        )
    };
    let config = format!(
        r#"{{"interaction":{{"type":"script","path":"activation.js","parameters":{{"image":"{image:#x}","record":"{:#x}","ready":"{:#x}"}}}},"runtime":"qjs"}}"#,
        callback, ready as *const () as usize
    );
    write_new(&dir.join("activation.js"), script)?;
    write_new(&dir.join("frida-gadget.config"), config.as_bytes())?;
    let output = File::from(io::stderr().as_handle().try_clone_to_owned()?);
    TRACE
        .set(Trace {
            image,
            size,
            output,
            events: Mutex::default(),
            gate: gate.then(Mutex::default),
            incomplete: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            active: AtomicUsize::new(0),
        })
        .map_err(|_| anyhow::anyhow!("activation tracer already started"))?;

    // Loading the DLL can itself block during startup. Bound both the loader and
    // the script callback; retaining the module reference keeps the listeners live.
    let (sender, receiver) = mpsc::channel();
    let start = Instant::now();
    std::thread::Builder::new()
        .name("activation-init".into())
        .spawn(move || {
            let wide: Vec<u16> = dll.as_os_str().encode_wide().chain(Some(0)).collect();
            // SAFETY: absolute NUL-terminated path to the verified, pinned library.
            let module = unsafe {
                LoadLibraryExW(
                    wide.as_ptr(),
                    std::ptr::null_mut(),
                    LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
            };
            let result = if module.is_null() {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            };
            drop(pinned);
            let _ = sender.send(result);
        })?;
    receiver
        .recv_timeout(ARM_TIMEOUT)
        .context("activation Gadget did not finish loading")?
        .context("loading the verified activation Gadget")?;
    loop {
        match ARMED.load(Ordering::Acquire) {
            1 => break,
            -1 => anyhow::bail!("diagnostic script failed to install its listeners"),
            _ => ensure!(
                start.elapsed() < ARM_TIMEOUT,
                "activation script timed out before arming"
            ),
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if gate {
        log_line(format_args!("init_gate armed image={image:#x}"));
    } else {
        log_line(format_args!("armed image={image:#x}"));
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    File::create_new(path)
        .and_then(|mut f| f.write_all(bytes))
        .with_context(|| format!("writing {} in a fresh activation session", path.display()))
}

extern "system" fn ready(success: i32) {
    ARMED.store(if success == 1 { 1 } else { -1 }, Ordering::Release);
}

extern "system" fn record(kind: u32, code: u32, caller: usize, entry: u64) -> u64 {
    let Some(trace) = TRACE.get() else { return 0 };
    trace.active.fetch_add(1, Ordering::SeqCst);
    let id = std::panic::catch_unwind(|| {
        if trace.finished.load(Ordering::SeqCst) {
            return 0;
        }
        // SAFETY: this synchronous callback runs on the intercepted Windows thread.
        let tid = unsafe { GetCurrentThreadId() };
        let mut events = trace.events.lock().unwrap_or_else(|e| e.into_inner());
        events.record(trace.image, trace.size, kind, tid, code, caller, entry)
    })
    .unwrap_or_else(|_| {
        trace.incomplete.store(true, Ordering::Relaxed);
        0
    });
    trace.active.fetch_sub(1, Ordering::SeqCst);
    id
}

/// On a natural first-chance breakpoint, summarize at the known assert only.
pub(crate) fn on_breakpoint(address: usize) {
    if let Some(trace) = TRACE.get()
        && address == trace.image + ASSERT_RVA
    {
        finish(trace, "registry_assert");
    }
}

/// Also report a normally returning title that did not reach the registry assert.
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
            "final_table reason={reason} {}",
            snapshot(trace.image)
        ));
        // An already admitted callback may be committing its last entry on another
        // thread. Give it a bounded grace period; never block the exception on a lock.
        let start = Instant::now();
        while trace.active.load(Ordering::SeqCst) != 0
            && start.elapsed() < Duration::from_millis(25)
        {
            std::thread::yield_now();
        }
        let incomplete =
            trace.incomplete.load(Ordering::Relaxed) || trace.active.load(Ordering::SeqCst) != 0;
        if let Some(counts) = &trace.gate {
            match counts.try_lock() {
                Ok(c) => c.summary(reason, incomplete),
                Err(_) => log_line(format_args!(
                    "init_gate summary reason={reason} complete=false counts=unavailable"
                )),
            }
            return;
        }
        match trace.events.try_lock() {
            Ok(events) => events.summary(reason, incomplete),
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner().summary(reason, true),
            Err(std::sync::TryLockError::WouldBlock) => {
                log_line(format_args!(
                    "summary reason={reason} complete=false counts=unavailable (event in progress)"
                ));
            }
        }
    });
}

fn log_line(args: fmt::Arguments<'_>) {
    if let Some(trace) = TRACE.get() {
        // An independent stderr handle avoids taking the Rust stderr lock in the VEH.
        if (&trace.output)
            .write_all(format!("VIctory: activation {args}\n").as_bytes())
            .is_err()
        {
            trace.incomplete.store(true, Ordering::Relaxed);
        }
    }
}

struct Snapshot {
    singleton: Option<usize>,
    descriptor: Option<usize>,
    cap: Option<u32>,
    base: Option<usize>,
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pointer = |p: Option<usize>| p.map_or_else(|| "?".into(), |v| format!("{v:#x}"));
        let cap = self.cap.map_or_else(|| "?".into(), |n| n.to_string());
        write!(
            f,
            "g={} descriptor={} cap={cap} base={}",
            pointer(self.singleton),
            pointer(self.descriptor),
            pointer(self.base)
        )
    }
}

fn snapshot(image: usize) -> Snapshot {
    let ptr = |address| read_memory(address).map(usize::from_le_bytes);
    let field = |p: Option<usize>, offset| p.filter(|&p| p != 0)?.checked_add(offset);
    let singleton = ptr(image + SINGLETON_RVA);
    let descriptor = field(singleton, 0x2a8).and_then(ptr);
    let cap = field(descriptor, 8)
        .and_then(read_memory)
        .map(u32::from_le_bytes);
    let base = field(descriptor, 0x10).and_then(ptr);
    Snapshot {
        singleton,
        descriptor,
        cap,
        base,
    }
}

fn read_memory<const N: usize>(address: usize) -> Option<[u8; N]> {
    let mut region = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: query metadata for the address into a correctly sized local buffer.
    let found = unsafe { VirtualQuery(address as _, &mut region, size_of_val(&region)) };
    if found == 0
        || region.State != MEM_COMMIT
        || region.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0
        || address.checked_add(N)? > (region.BaseAddress as usize).checked_add(region.RegionSize)?
    {
        return None;
    }
    let mut bytes = [0; N];
    let mut count = 0;
    // SAFETY: Windows copies into our array and diagnoses a concurrent free or
    // protection change. No Rust reference to mutable title memory is formed.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            address as _,
            bytes.as_mut_ptr().cast(),
            N,
            &mut count,
        )
    };
    (ok != 0 && count == N).then_some(bytes)
}
