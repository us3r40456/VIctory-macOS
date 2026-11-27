//! The sampler (`super`): a thread of ours that looks at the watched
//! threads every period, and writes the report.
//!
//! Safety of a look, under Wine on Rosetta too: `SuspendThread` asks the
//! Wine server to stop the thread, which happens in Wine's own signal
//! handler at the next point the thread can take a signal (a thread in a
//! system call or waiting is stopped there; one inside Wine's fault
//! handling only once that handler is done, as Wine blocks the signal
//! meanwhile). `GetThreadContext` waits for that and returns the
//! registers as the thread's Windows code had them. Until `ResumeThread`
//! the sampler only reads the thread's block and stack: no allocation, no
//! lock, no loader or unwinder call, no system call but those three. The
//! stack is copied from its pointer up to the base the thread's block
//! gives, never beyond it, and only when the pointer lies inside that
//! stack (a thread on another stack is taken by its instruction pointer
//! alone).

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::{Mutex, OnceLock, TryLockError};
use std::time::{Duration, Instant};

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, ReadProcessMemory,
    RtlLookupFunctionEntry,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
    TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId, GetExitCodeThread,
    GetThreadDescription, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use super::report::EXPOSURE_US;
use super::report::{
    self, Header, Kind, Mark, Module, Modules, SKIPPED, Sample, ThreadView, Totals,
};
use super::walk::{self, Copied, SLACK};
use super::{
    Config, EXCEPTIONS, MAX_DEPTH, ON, PLACES, QUIET_ROUNDS, ROUND, STARTS, Started, marked, place,
};

/// What a look at a thread needs of its handle.
const ACCESS: u32 = THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION;
/// `GetExitCodeThread` of a thread still running.
const STILL_ACTIVE: u32 = 259;
/// Samples in a row in a wait that make a thread a waiting one.
const WAITING_AFTER: u32 = 4;
/// A gap between a thread's samples longer than this is not counted (the
/// sampler stood still: it wrote its report, say).
const MAX_GAP_US: u64 = 200_000;
/// Presents kept for the report, the latest.
const MARKS: usize = 5000;
/// How often the list of threads is taken again.
const LIST_EVERY: Duration = Duration::from_secs(1);
/// Kernel calls that wait: a thread whose instruction pointer is in one of
/// them is waiting (Wine's system call stubs are 32 bytes; a thread in a
/// call stands at their end).
const WAITS: [(&str, &[&str]); 2] = [
    (
        "ntdll.dll",
        &[
            "NtWaitForSingleObject",
            "NtWaitForMultipleObjects",
            "NtDelayExecution",
            "NtRemoveIoCompletion",
            "NtRemoveIoCompletionEx",
            "NtWaitForAlertByThreadId",
            "NtSignalAndWaitForSingleObject",
            "NtWaitForKeyedEvent",
            "NtYieldExecution",
            "NtReplyWaitReceivePort",
            "NtWaitForDebugEvent",
            "NtWaitForWorkViaWorkerFactory",
            // mac: a registry watcher waits in these (Wine's notification
            // thread, always "running" in the first profile).
            "NtNotifyChangeKey",
            "NtNotifyChangeMultipleKeys",
        ],
    ),
    (
        "win32u.dll",
        &[
            "NtUserMsgWaitForMultipleObjectsEx",
            "NtUserGetMessage",
            "NtUserWaitMessage",
            "NtUserWaitForInputIdle",
        ],
    ),
];
/// mac: kernel calls that read files: a thread in one is in I/O, neither
/// running nor waiting (the streaming workers spent 261 ms a frame there in
/// the first profile).
const IO: [(&str, &[&str]); 1] = [("ntdll.dll", &["NtReadFile", "NtReadFileScatter"])];
const STUB: u64 = 0x20;
/// mac: threads that live in one call into Wine's Unix side and wait there
/// nearly all the time; a sample of theirs in the image their thread
/// started in is a wait (mmdevapi's `timer_loop_func`, which calls the
/// Unix `timer_loop` once; 5 threads and 1,950 ms a frame of "running" in
/// the first profile).
const UNIX_CALL_THREADS: [&str; 1] = ["audio_client_timer"];
/// mac: looks in a row at the same address outside every PE image that make
/// it a wait (a thread blocked in Unix code, such as D3DMetal's waiter in
/// `__ulock_wait2`, stands still there; running code does not).
const SAME_AFTER: u32 = 3;
/// mac: Unix-side wait addresses learned at most.
const LEARNED_MOST: usize = 256;
/// By thread (its id's place): the id of the last thread there that ended
/// the normal way, through the image's thread detach.
static DETACHED: [std::sync::atomic::AtomicU32; PLACES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; PLACES];

/// The loader calls the image's TLS callbacks when a thread ends the
/// normal way; Wine ends a thread after a nested exception without that.
#[unsafe(link_section = ".CRT$XLAW")]
static ON_THREAD: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = on_thread;

unsafe extern "system" fn on_thread(_: *mut c_void, reason: u32, _: *mut c_void) {
    const DLL_THREAD_DETACH: u32 = 3;
    if reason == DLL_THREAD_DETACH && ON.load(std::sync::atomic::Ordering::Relaxed) {
        // SAFETY: a plain query.
        let id = unsafe { GetCurrentThreadId() };
        DETACHED[place(id)].store(id, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// A thread being sampled.
struct Watched {
    id: u32,
    handle: HANDLE,
    /// Its thread block: its stack's base and limit are there.
    teb: usize,
    name: String,
    start: u64,
    /// Samples in a row in a wait.
    waits_in_row: u32,
    /// mac: its last look was left out (an exception): what it does now is
    /// not known, so a round it is not due in does not count it running.
    unseen_last: bool,
    /// mac: its last instruction pointer, and in how many looks in a row
    /// before this one it stood there.
    last_rip: u64,
    same_rip: u32,
    /// Its previous sample (microseconds since the sampler started; 0: none).
    last_us: u64,
    failures: u32,
    ended: Option<u32>,
    /// Its exception starts when it began to be watched, at the last
    /// measure, and their rate a second (quick to rise, slow to fall).
    starts_from: u32,
    starts_then: u32,
    rate: f64,
    /// Its starts as last seen in a round, and when.
    starts_seen: u32,
    seen_us: u64,
    /// Its last look, and no look before `next_us` (microseconds since the
    /// sampler started).
    looked_us: u64,
    next_us: u64,
    looks: u64,
    totals: Totals,
}

/// mac: what a sampled thread was doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Running,
    Waiting,
    /// In file I/O.
    Io,
    /// Not looked at (it had just taken an exception).
    Unseen,
}

/// A sample of this round, before it is known whether its thread ran alone.
struct Pending {
    thread: usize,
    us: u64,
    state: State,
    function: u64,
    depth: usize,
    frames: [u64; MAX_DEPTH],
}

pub(super) struct Sampler {
    config: Config,
    /// The sampling thread, not sampled.
    me: u32,
    process: u32,
    threads: Vec<Watched>,
    /// Every thread id met, watched or not.
    seen: HashSet<u32>,
    clock: Instant,
    /// When the window began (microseconds since `clock`).
    from_us: Option<u64>,
    period_us: u64,
    context: Box<Aligned>,
    /// The stack copy: `room` bytes, zeroes after the last copy.
    copy: *mut u8,
    room: usize,
    dirty: usize,
    /// Of the last look: the stack's bounds and where the copy starts.
    low: usize,
    high: usize,
    offset: usize,
    frames: [u64; MAX_DEPTH],
    pending: Vec<Pending>,
    rounds: u64,
    late: u64,
    waits: Vec<(u64, u64)>,
    /// mac: the file-read calls' stubs.
    io: Vec<(u64, u64)>,
    /// mac: wait addresses learned outside every PE image (`SAME_AFTER`).
    learned: HashSet<u64>,
    /// Function starts by instruction pointer.
    functions: HashMap<u64, u64>,
    all: Totals,
    marks: VecDeque<Mark>,
    presented_from: u64,
    presented_then: u64,
    /// The title's images: file, base, size.
    title: Vec<(String, u64, u64)>,
    modules: Modules,
    modules_at: Option<Instant>,
    exports: HashMap<u64, Vec<(u32, String)>>,
    suspensions: u64,
    failed: u64,
    /// Looks left out: the thread had just taken an exception.
    skipped: u64,
    quiet_rounds: u64,
    /// Watched threads that ended as Wine ends one after a nested exception.
    lost: Vec<u32>,
    /// Threads expected lost by the looks so far.
    risked: f64,
    /// Since when exceptions are marked; the last measure of the rates.
    marks_since: Option<Instant>,
    rates_at: Instant,
    held_ns: u64,
    held_max_ns: u64,
    /// The time looks were held lately (microseconds, a running mean).
    held_recent_us: f64,
    running_rounds: [u64; 9],
    listed_at: Option<Instant>,
    written_at: Instant,
    write_failed: bool,
    done: bool,
}

// SAFETY: the handles and the stack copy belong to the process, not to a
// thread; one thread at a time uses the sampler (behind `SAMPLER`'s lock).
unsafe impl Send for Sampler {}

static SAMPLER: OnceLock<Mutex<Sampler>> = OnceLock::new();

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

/// Starts the sampling thread.
pub(super) fn start(config: Config, title: Vec<(String, u64, u64)>) -> Result<(), String> {
    let sampler = Sampler::new(config, title)?;
    SAMPLER
        .set(Mutex::new(sampler))
        .map_err(|_| "already started".to_string())?;
    std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(run)
        .map_err(|e| format!("no thread: {e}"))?;
    // SAFETY: registers a function with the C runtime's exit table, and a
    // vectored handler that only marks the thread (the runtime's change
    // tracking marks its own faults, ahead of it).
    unsafe {
        atexit(at_exit);
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(
            1,
            Some(on_exception),
        );
    }
    Ok(())
}

/// Marks every exception dispatched to the vectored handlers that reaches it.
unsafe extern "system" fn on_exception(
    _: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    super::excepted();
    // EXCEPTION_CONTINUE_SEARCH: the next handler decides.
    0
}

fn run() {
    let Some(lock) = SAMPLER.get() else {
        return;
    };
    let period = {
        let s = lock.lock().unwrap_or_else(|e| e.into_inner());
        Duration::from_micros(s.period_us)
    };
    loop {
        let began = Instant::now();
        {
            let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
            if !s.step() {
                return;
            }
            if began.elapsed() > period {
                s.late += 1;
            }
        }
        if let Some(rest) = period.checked_sub(began.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

/// At exit: the report as it stands, unless the sampler cannot be had soon.
extern "C" fn at_exit() {
    let Some(lock) = SAMPLER.get() else {
        return;
    };
    for _ in 0..300 {
        let mut s = match lock.try_lock() {
            Ok(s) => s,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
        };
        if !s.done && s.from_us.is_some() {
            s.write(true);
        }
        s.done = true;
        return;
    }
}

impl Sampler {
    pub(super) fn new(config: Config, title: Vec<(String, u64, u64)>) -> Result<Sampler, String> {
        let room = config.stack + 16 + SLACK;
        // SAFETY: fresh zeroed memory of ours, freed on drop.
        let copy = unsafe { VirtualAlloc(null(), room, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        if copy.is_null() {
            return Err("no memory for the stack copy".into());
        }
        let period_us = ((config.period_ms * 1000.0) as u64).max(1);
        let quiet_rounds = (config.quiet_ms / config.period_ms).ceil() as u64;
        // SAFETY: reads our own static, so the linker keeps it (and its section).
        unsafe { std::ptr::from_ref(&ON_THREAD).read_volatile() };
        Ok(Sampler {
            config,
            me: 0,
            // SAFETY: a plain query.
            process: unsafe { GetCurrentProcessId() },
            threads: Vec::new(),
            seen: HashSet::new(),
            clock: Instant::now(),
            from_us: None,
            period_us,
            // SAFETY: a zeroed CONTEXT is a valid one.
            context: Box::new(Aligned(unsafe { std::mem::zeroed() })),
            copy: copy.cast(),
            room,
            dirty: 0,
            low: 0,
            high: 0,
            offset: 0,
            frames: [0; MAX_DEPTH],
            pending: Vec::new(),
            rounds: 0,
            late: 0,
            waits: stubs(&WAITS),
            io: stubs(&IO),
            learned: HashSet::new(),
            functions: HashMap::new(),
            all: Totals::with_callers(),
            marks: VecDeque::new(),
            presented_from: 0,
            presented_then: 0,
            title,
            modules: Modules::default(),
            modules_at: None,
            exports: HashMap::new(),
            suspensions: 0,
            failed: 0,
            skipped: 0,
            quiet_rounds,
            lost: Vec::new(),
            risked: 0.0,
            marks_since: None,
            rates_at: Instant::now(),
            held_ns: 0,
            held_max_ns: 0,
            held_recent_us: 0.0,
            running_rounds: [0; 9],
            listed_at: None,
            written_at: Instant::now(),
            write_failed: false,
            done: false,
        })
    }

    fn us(&self) -> u64 {
        self.clock.elapsed().as_micros() as u64
    }

    /// One period's work: a round once the window is open, the list of
    /// threads and the report when due. False once the window has closed.
    pub(super) fn step(&mut self) -> bool {
        if self.done {
            return false;
        }
        if self.me == 0 {
            // SAFETY: a plain query.
            self.me = unsafe { GetCurrentThreadId() };
            // Exceptions are marked from now on, before the window opens,
            // so its first rounds know which threads are taking them.
            QUIET_ROUNDS.store(self.quiet_rounds, std::sync::atomic::Ordering::Relaxed);
            ON.store(true, std::sync::atomic::Ordering::Relaxed);
            self.marks_since = Some(Instant::now());
        }
        let now_s = self.clock.elapsed().as_secs_f64();
        if now_s < self.config.after_s {
            return true;
        }
        if self.from_us.is_none() {
            self.from_us = Some(self.us());
            self.presented_from = crate::queue::presented();
            self.presented_then = self.presented_from;
            self.list_modules();
            self.written_at = Instant::now();
        }
        if self.listed_at.is_none_or(|at| at.elapsed() >= LIST_EVERY) {
            self.list_threads();
            self.listed_at = Some(Instant::now());
        }
        self.round();
        // A thread lost as Wine loses one after a look met an exception:
        // no more looks, the report says so.
        let over = !self.lost.is_empty()
            || self.config.for_s > 0.0 && now_s >= self.config.after_s + self.config.for_s;
        if over || self.written_at.elapsed().as_secs_f64() >= self.config.write_s {
            self.write(over);
            self.written_at = Instant::now();
            // The time the report took is no one's.
            for t in &mut self.threads {
                t.last_us = 0;
            }
        }
        if over {
            self.finish();
            return false;
        }
        true
    }

    /// Looks at every watched thread that is due.
    fn round(&mut self) {
        self.rounds += 1;
        ROUND.store(self.rounds, std::sync::atomic::Ordering::Relaxed);
        let presented = crate::queue::presented();
        if presented != self.presented_then {
            self.presented_then = presented;
            if self.marks.len() == MARKS {
                self.marks.pop_front();
            }
            let acc = self.all.acc;
            self.marks.push_back(Mark {
                frame: presented,
                ms: self.us() as f64 / 1000.0,
                samples: acc.all.samples,
                running: acc.running.samples,
                alone: acc.alone.samples,
            });
        }
        let mut pending = std::mem::take(&mut self.pending);
        pending.clear();
        let idle = u64::from(self.config.idle_every.max(1));
        let depth = self.config.depth.clamp(1, MAX_DEPTH);
        let mut running = 0;
        for i in 0..self.threads.len() {
            let t = &self.threads[i];
            if t.ended.is_some() {
                continue;
            }
            let waiting = t.waits_in_row >= WAITING_AFTER;
            if waiting && idle > 1 && !(self.rounds + i as u64).is_multiple_of(idle) {
                continue;
            }
            if self.config.risk > 0.0 {
                // New starts since the last round: the rate goes up at once
                // (over at least 100 ms), and the next look moves out.
                let now = self.us();
                let starts = STARTS[place(t.id)].load(std::sync::atomic::Ordering::Relaxed);
                let (risk, exposure) = (self.config.risk, self.exposure_us());
                let t = &mut self.threads[i];
                if starts != t.starts_seen {
                    let gap = now.saturating_sub(t.seen_us).max(100_000) as f64 / 1e6;
                    t.rate = t
                        .rate
                        .max(f64::from(starts.wrapping_sub(t.starts_seen)) / gap);
                    (t.starts_seen, t.seen_us) = (starts, now);
                    t.next_us = t
                        .next_us
                        .max(t.looked_us + interval_us(t.rate, risk, exposure));
                }
                if now < t.next_us {
                    // Not due: as it was at its last look (not known after
                    // a look left out).
                    running += usize::from(t.waits_in_row == 0 && !t.unseen_last);
                    continue;
                }
            }
            let t = &self.threads[i];
            if self.quiet_rounds > 0 && marked(place(t.id), self.rounds + 1, self.quiet_rounds) {
                // Just took an exception: not looked at. mac: what it does
                // meanwhile is not known, so the sample is neither running
                // nor waiting, and the round's count of running threads
                // (for "alone") leaves it out.
                self.skipped += 1;
                let at = self.us();
                let t = &mut self.threads[i];
                let us = match at.checked_sub(t.last_us) {
                    Some(gap) if t.last_us != 0 && gap <= MAX_GAP_US => gap,
                    _ => self.period_us,
                };
                t.last_us = at;
                t.waits_in_row = 0;
                t.unseen_last = true;
                let mut frames = [0; MAX_DEPTH];
                frames[0] = SKIPPED;
                pending.push(Pending {
                    thread: i,
                    us,
                    state: State::Unseen,
                    function: 0,
                    depth: 1,
                    frames,
                });
                continue;
            }
            let Some(got) = self.look(i) else {
                self.failed += 1;
                self.threads[i].failures += 1;
                if self.threads[i].failures >= 3 {
                    self.check_ended(i, false);
                }
                continue;
            };
            let at = self.us();
            let copied = Copied {
                base: self.copy as usize,
                room: self.room,
                offset: self.offset,
                got,
            };
            // SAFETY: the copy is ours, with zeroes after what was copied.
            let n = unsafe {
                walk::walk(
                    &mut self.context.0,
                    &copied,
                    self.low,
                    self.high,
                    &mut self.frames[..depth],
                )
            };
            let rip = self.frames[0];
            let function = self.function_of(rip);
            let mut state = self.state_of(rip);
            if state == State::Running && self.unix_wait(i, rip) {
                state = State::Waiting;
            }
            let t = &mut self.threads[i];
            t.same_rip = if rip == t.last_rip { t.same_rip.saturating_add(1) } else { 0 };
            t.last_rip = rip;
            let is_running = state == State::Running;
            let fallback = self.period_us * if waiting { idle } else { 1 };
            let exposure = self.exposure_us();
            let t = &mut self.threads[i];
            t.failures = 0;
            t.unseen_last = false;
            t.looks += 1;
            t.looked_us = at;
            t.next_us = at + interval_us(t.rate, self.config.risk, exposure);
            self.risked += t.rate * exposure / 1e6;
            let us = match at.checked_sub(t.last_us) {
                Some(gap) if t.last_us != 0 && gap <= MAX_GAP_US => gap,
                _ => fallback,
            };
            t.last_us = at;
            // A thread in file I/O is looked at as often as a running one.
            t.waits_in_row = if state == State::Waiting {
                t.waits_in_row.saturating_add(1)
            } else {
                0
            };
            running += usize::from(is_running);
            let mut frames = [0; MAX_DEPTH];
            frames[..n].copy_from_slice(&self.frames[..n]);
            pending.push(Pending {
                thread: i,
                us,
                state,
                function,
                depth: n,
                frames,
            });
        }
        self.running_rounds[running.min(self.running_rounds.len() - 1)] += 1;
        for p in &pending {
            let is_running = p.state == State::Running;
            let sample = Sample {
                frames: &p.frames[..p.depth],
                function: p.function,
                us: p.us,
                running: is_running,
                alone: is_running && running == 1,
                io: p.state == State::Io,
                unseen: p.state == State::Unseen,
            };
            self.all.add(&sample);
            self.threads[p.thread].totals.add(&sample);
        }
        self.pending = pending;
    }

    /// Stops thread `i`, takes its registers and a copy of its stack, and
    /// lets it go: the bytes copied, or `None` if it could not be looked at.
    fn look(&mut self, i: usize) -> Option<usize> {
        let (handle, teb) = (self.threads[i].handle, self.threads[i].teb);
        let context = &mut self.context.0;
        context.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
        let (limit, copy) = (self.config.stack, self.copy);
        let began = Instant::now();
        // SAFETY: a thread of this process that is not this one; the
        // context is aligned and ours; the copy has room for `limit` bytes
        // after 15 of offset. While the thread stands still this reads its
        // thread block (which lives while the thread does, and it cannot
        // end while stopped) and its stack between the stack pointer and
        // the stack's base, which is in use and so committed; nothing here
        // allocates or takes a lock.
        let (read, low, high, offset, got) = unsafe {
            if SuspendThread(handle) == u32::MAX {
                return None;
            }
            let read = GetThreadContext(handle, context) != 0;
            let (mut low, mut high, mut offset, mut got) = (0, 0, 0, 0);
            if read && teb != 0 {
                let tib = teb as *const usize;
                high = tib.add(1).read_volatile();
                low = tib.add(2).read_volatile();
                let rsp = context.Rsp as usize;
                if low < high && rsp >= low && rsp < high {
                    got = (high - rsp).min(limit);
                    offset = rsp & 15;
                    std::ptr::copy_nonoverlapping(rsp as *const u8, copy.add(offset), got);
                }
            }
            ResumeThread(handle);
            (read, low, high, offset, got)
        };
        let held = began.elapsed().as_nanos() as u64;
        self.suspensions += 1;
        self.held_ns += held;
        self.held_max_ns = self.held_max_ns.max(held);
        self.held_recent_us = 0.9 * self.held_recent_us + 0.1 * held as f64 / 1000.0;
        if !read {
            return None;
        }
        // What an earlier, longer copy left behind it.
        let end = offset + got;
        if end < self.dirty {
            // SAFETY: inside the copy.
            unsafe { self.copy.add(end).write_bytes(0, self.dirty - end) };
        }
        self.dirty = end;
        (self.low, self.high, self.offset) = (low, high, offset);
        Some(got)
    }

    /// The exposure of a look (`EXPOSURE_US`, measured with looks held
    /// about 150 us), taken to grow with the time looks are held lately:
    /// a loaded machine delivers the stop later (a guess, on the safe side).
    fn exposure_us(&self) -> f64 {
        EXPOSURE_US * (self.held_recent_us / 150.0).max(1.0)
    }

    fn is_wait(&self, rip: u64) -> bool {
        self.waits.iter().any(|&(from, to)| rip >= from && rip < to)
    }

    /// mac: whether thread `i`, at `rip` and not in a known wait, waits in
    /// Wine's Unix side: a thread of [`UNIX_CALL_THREADS`] in the image it
    /// started in, or an address outside every image where a thread stood
    /// [`SAME_AFTER`] looks in a row before (learned once for all threads).
    fn unix_wait(&mut self, i: usize, rip: u64) -> bool {
        if self.learned.contains(&rip) {
            return true;
        }
        let t = &self.threads[i];
        if UNIX_CALL_THREADS.contains(&t.name.as_str()) {
            let image = |at| self.modules.find(at).map(|m| m.base);
            if let Some(base) = image(rip)
                && image(t.start) == Some(base)
            {
                return true;
            }
        }
        let outside = self.modules.find(rip).is_none();
        if stands_still(t.last_rip, t.same_rip, rip, outside) && self.learned.len() < LEARNED_MOST {
            self.learned.insert(rip);
            return true;
        }
        false
    }

    /// What a thread standing at `rip` is doing.
    fn state_of(&self, rip: u64) -> State {
        if self.is_wait(rip) {
            State::Waiting
        } else if self.io.iter().any(|&(from, to)| rip >= from && rip < to) {
            State::Io
        } else {
            State::Running
        }
    }

    /// Where the function holding `rip` starts, by unwind data; 0 if none.
    fn function_of(&mut self, rip: u64) -> u64 {
        if let Some(&start) = self.functions.get(&rip) {
            return start;
        }
        let mut image = 0;
        // SAFETY: a lookup; no thread of ours is stopped now.
        let entry = unsafe { RtlLookupFunctionEntry(rip, &mut image, null_mut()) };
        let start = if entry.is_null() {
            0
        } else {
            // SAFETY: the image's own table entry.
            image + u64::from(unsafe { (*entry).BeginAddress })
        };
        if self.functions.len() > 1 << 20 {
            self.functions.clear();
        }
        self.functions.insert(rip, start);
        start
    }

    /// Takes the list of threads again: watches the new ones the filter
    /// picks, and marks the watched ones that are gone.
    fn list_threads(&mut self) {
        let mut alive = HashSet::new();
        // SAFETY: a snapshot of this process's threads, read with a record
        // of the size it says, and closed.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return;
            }
            let mut entry: THREADENTRY32 = std::mem::zeroed();
            entry.dwSize = size_of::<THREADENTRY32>() as u32;
            let mut more = Thread32First(snapshot, &mut entry) != 0;
            while more {
                if entry.th32OwnerProcessID == self.process && entry.th32ThreadID != self.me {
                    alive.insert(entry.th32ThreadID);
                }
                more = Thread32Next(snapshot, &mut entry) != 0;
            }
            CloseHandle(snapshot);
        }
        let mut new: Vec<u32> = alive
            .iter()
            .copied()
            .filter(|id| !self.seen.contains(id))
            .collect();
        new.sort_unstable();
        for id in new {
            self.seen.insert(id);
            self.consider(id);
        }
        for i in 0..self.threads.len() {
            if self.threads[i].ended.is_none() && !alive.contains(&self.threads[i].id) {
                self.check_ended(i, true);
            }
        }
        // The rates of starts since the last measure; they fall slowly.
        let elapsed = self.rates_at.elapsed().as_secs_f64().max(0.1);
        self.rates_at = Instant::now();
        for t in &mut self.threads {
            let starts = STARTS[place(t.id)].load(std::sync::atomic::Ordering::Relaxed);
            let rate = f64::from(starts.wrapping_sub(t.starts_then)) / elapsed;
            t.starts_then = starts;
            t.rate = rate.max(t.rate * 0.7);
            if t.rate < 0.01 {
                t.rate = 0.0;
            }
        }
    }

    /// Opens thread `id` and watches it if the filter picks it.
    fn consider(&mut self, id: u32) {
        // SAFETY: opens a thread of this process.
        let handle = unsafe { OpenThread(ACCESS, 0, id) };
        if handle.is_null() {
            return;
        }
        let mut info = BasicInfo {
            exit_status: 0,
            teb: 0,
            client_id: [0; 2],
            affinity: 0,
            priority: 0,
            base_priority: 0,
        };
        let mut start = 0usize;
        // SAFETY: writes records of the system's layout and size.
        unsafe {
            if NtQueryInformationThread(
                handle,
                ThreadBasicInformation,
                (&raw mut info).cast(),
                size_of::<BasicInfo>() as u32,
                null_mut(),
            ) != 0
            {
                info.teb = 0;
            }
            NtQueryInformationThread(
                handle,
                ThreadQuerySetWin32StartAddress,
                (&raw mut start).cast(),
                size_of::<usize>() as u32,
                null_mut(),
            );
        }
        let start = start as u64;
        let name = thread_name(handle);
        if self.modules.find(start).is_none() && self.modules_stale() {
            self.list_modules();
        }
        let started = match self.modules.find(start).map(|m| m.kind) {
            Some(Kind::Title) => Started::Title,
            Some(Kind::Runtime) => Started::Runtime,
            Some(Kind::System) => Started::System,
            None => Started::Unknown,
        };
        if !self.config.threads.picks(id, &name, started) {
            // SAFETY: opened above.
            unsafe { CloseHandle(handle) };
            return;
        }
        // A thread before it with the same id (Wine uses ids again) may have
        // left its detach in the place.
        let _ = DETACHED[place(id)].compare_exchange(
            id,
            0,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Its rate of starts so far: since exceptions are marked.
        let starts = STARTS[place(id)].load(std::sync::atomic::Ordering::Relaxed);
        let marked_for = self
            .marks_since
            .map_or(1.0, |at| at.elapsed().as_secs_f64().max(1.0));
        let rate = f64::from(starts) / marked_for;
        let now = self.us();
        self.threads.push(Watched {
            id,
            handle,
            teb: info.teb,
            name,
            start,
            waits_in_row: 0,
            unseen_last: false,
            last_rip: 0,
            same_rip: 0,
            last_us: 0,
            failures: 0,
            ended: None,
            starts_from: starts,
            starts_then: starts,
            rate,
            starts_seen: starts,
            seen_us: now,
            looked_us: now,
            next_us: now + interval_us(rate, self.config.risk, self.exposure_us()),
            looks: 0,
            totals: Totals::default(),
        });
        self.pending.reserve(self.threads.len());
    }

    /// Thread `i` ended if its exit code says so: then it is let go.
    /// `gone`: the thread list no longer has it (else looks at it failed).
    fn check_ended(&mut self, i: usize, gone: bool) {
        let t = &mut self.threads[i];
        let mut code = 0;
        // SAFETY: our handle.
        let read = unsafe { GetExitCodeThread(t.handle, &mut code) } != 0;
        if read && code == STILL_ACTIVE || !read && !gone {
            return;
        }
        t.ended = Some(code);
        // SAFETY: ours, not used again.
        unsafe { CloseHandle(t.handle) };
        t.handle = null_mut();
        // Ended without its detach (two threads sharing a place can make a
        // normal end look like this: then the sampler stops for nothing).
        let detached = DETACHED[place(t.id)].load(std::sync::atomic::Ordering::Relaxed) == t.id;
        if !detached {
            self.lost.push(t.id);
            tracing::warn!(
                thread = t.id,
                name = t.name.as_str(),
                samples = t.totals.acc.all.samples,
                code,
                "profile: a watched thread ended without the normal thread exit, as Wine ends one after a nested \
                 exception (a look that met an exception under Rosetta); the sampler stops"
            );
        } else {
            tracing::info!(
                thread = t.id,
                name = t.name.as_str(),
                code,
                samples = t.totals.acc.all.samples,
                "profile: a watched thread ended"
            );
        }
    }

    fn modules_stale(&self) -> bool {
        self.modules_at
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(2))
    }

    /// The loaded images, from a snapshot.
    fn list_modules(&mut self) {
        // SAFETY: plain queries; a snapshot of this process's modules read
        // with a record of the size it says, and closed.
        let list = unsafe {
            let exe = GetModuleHandleW(null()) as u64;
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return;
            }
            let mut list = Vec::new();
            let mut entry: MODULEENTRY32W = std::mem::zeroed();
            entry.dwSize = size_of::<MODULEENTRY32W>() as u32;
            let mut more = Module32FirstW(snapshot, &mut entry) != 0;
            while more {
                let len = entry
                    .szModule
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(entry.szModule.len());
                let name = String::from_utf16_lossy(&entry.szModule[..len]);
                let base = entry.modBaseAddr as u64;
                let titled = self.title.iter().find(|(_, b, _)| *b == base);
                let kind = if base == exe {
                    Kind::Runtime
                } else if titled.is_some()
                    || name.to_ascii_lowercase().starts_with("victory-image-")
                {
                    Kind::Title
                } else {
                    Kind::System
                };
                list.push(Module {
                    name,
                    base,
                    size: u64::from(entry.modBaseSize),
                    kind,
                    note: titled.map(|(file, _, _)| file.clone()).unwrap_or_default(),
                });
                more = Module32NextW(snapshot, &mut entry) != 0;
            }
            CloseHandle(snapshot);
            list
        };
        self.modules = Modules::new(list);
        self.modules_at = Some(Instant::now());
    }

    /// `module+0xrva`, with the nearest export before it for system DLLs.
    fn name(&self, at: u64) -> String {
        if at == SKIPPED {
            return format!(
                "(not looked at: an exception within {} ms)",
                self.config.quiet_ms
            );
        }
        let Some(m) = self.modules.find(at) else {
            return format!("{at:#x}");
        };
        let rva = at - m.base;
        let mut text = format!("{}+{rva:#x}", m.name);
        if let Some(list) = self.exports.get(&m.base) {
            let i = list.partition_point(|e| u64::from(e.0) <= rva);
            if i > 0 {
                let (from, export) = &list[i - 1];
                let off = rva - u64::from(*from);
                if off < 0x1_0000 {
                    text.push_str(&format!(" {export}+{off:#x}"));
                }
            }
        }
        text
    }

    /// Writes the report, through a temporary file.
    pub(super) fn write(&mut self, done: bool) {
        if self.modules_stale()
            && self
                .all
                .buckets
                .keys()
                .any(|&at| self.modules.find(at).is_none())
        {
            self.list_modules();
        }
        let mut sampled: HashSet<u64> = HashSet::new();
        for &at in self.all.buckets.keys() {
            if let Some(m) = self.modules.find(at)
                && m.kind == Kind::System
            {
                sampled.insert(m.base);
            }
        }
        for m in self.modules.iter() {
            if sampled.contains(&m.base) && !self.exports.contains_key(&m.base) {
                self.exports.insert(m.base, exports(m.base, m.size));
            }
        }
        let text = self.render(done);
        let part = format!("{}.part", self.config.file);
        let written =
            std::fs::write(&part, text).and_then(|()| std::fs::rename(&part, &self.config.file));
        if let Err(why) = written
            && !self.write_failed
        {
            self.write_failed = true;
            tracing::warn!(file = self.config.file.as_str(), %why, "profile: the report could not be written");
        }
    }

    fn render(&self, done: bool) -> String {
        let from = self.from_us.unwrap_or(0);
        let header = Header {
            file: self.config.file.clone(),
            from_s: from as f64 / 1e6,
            seconds: self.us().saturating_sub(from) as f64 / 1e6,
            period_ms: self.config.period_ms,
            rounds: self.rounds,
            late: self.late,
            suspensions: self.suspensions,
            failed: self.failed,
            held_us: self.held_ns / 1000,
            held_max_us: self.held_max_ns / 1000,
            threads_seen: self.seen.len(),
            threads_watched: self.threads.len(),
            filter: self.config.threads.text().to_string(),
            depth: self.config.depth,
            stack_kb: self.config.stack / 1024,
            idle_every: self.config.idle_every,
            frames: self.presented_then - self.presented_from,
            running_rounds: self.running_rounds.to_vec(),
            waits_known: self.waits.len(),
            learned: {
                let mut l: Vec<u64> = self.learned.iter().copied().collect();
                l.sort_unstable();
                l
            },
            skipped: self.skipped,
            quiet_ms: self.config.quiet_ms,
            starts: self
                .threads
                .iter()
                .map(|t| u64::from(self.starts_of(t)))
                .sum(),
            risked: self.risked,
            risk: self.config.risk,
            lost: self.lost.clone(),
            done,
        };
        let views: Vec<ThreadView> = self
            .threads
            .iter()
            .map(|t| ThreadView {
                id: t.id,
                name: &t.name,
                start: t.start,
                ended: t.ended,
                exceptions: EXCEPTIONS[place(t.id)].load(std::sync::atomic::Ordering::Relaxed),
                starts: self.starts_of(t),
                looks: t.looks,
                rate: t.rate,
                totals: &t.totals,
            })
            .collect();
        let marks: Vec<Mark> = self.marks.iter().copied().collect();
        report::render(&header, &self.modules, &self.all, &views, &marks, &|at| {
            self.name(at)
        })
    }

    /// Exception starts of thread `t` since it was first watched.
    fn starts_of(&self, t: &Watched) -> u32 {
        STARTS[place(t.id)]
            .load(std::sync::atomic::Ordering::Relaxed)
            .wrapping_sub(t.starts_from)
    }

    /// The window is over: every thread is let go.
    fn finish(&mut self) {
        for t in &mut self.threads {
            if !t.handle.is_null() {
                // SAFETY: ours, not used again.
                unsafe { CloseHandle(t.handle) };
                t.handle = null_mut();
                t.ended.get_or_insert(STILL_ACTIVE);
            }
        }
        self.done = true;
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        for t in &self.threads {
            if !t.handle.is_null() {
                // SAFETY: ours.
                unsafe { CloseHandle(t.handle) };
            }
        }
        // SAFETY: allocated in `new`, not used again.
        unsafe { VirtualFree(self.copy.cast(), 0, MEM_RELEASE) };
    }
}

/// The time between looks at a thread that starts exceptions `rate` times
/// a second, so that its expected loss stays under `risk` threads a
/// minute (microseconds; 0 when it needs no spacing).
fn interval_us(rate: f64, risk: f64, exposure_us: f64) -> u64 {
    if rate <= 0.0 || risk <= 0.0 {
        return 0;
    }
    (60.0 * rate * exposure_us / risk) as u64
}

/// The address ranges of the wait functions' stubs.
/// mac: whether a thread now at `rip`, last at `last` for `same` looks in a
/// row before that, stands still outside every image (`outside`) long
/// enough for `rip` to be a wait.
fn stands_still(last: u64, same: u32, rip: u64, outside: bool) -> bool {
    outside && rip != 0 && rip == last && same + 1 >= SAME_AFTER
}

/// The stubs of the kernel calls `calls` names, by module.
fn stubs(calls: &[(&str, &[&str])]) -> Vec<(u64, u64)> {
    let mut list = Vec::new();
    for &(dll, names) in calls {
        let wide: Vec<u16> = dll.encode_utf16().chain([0]).collect();
        // SAFETY: a plain query for a loaded module.
        let module = unsafe { GetModuleHandleW(wide.as_ptr()) };
        if module.is_null() {
            continue;
        }
        for name in names {
            let c = format!("{name}\0");
            // SAFETY: a loaded module and a terminated name.
            if let Some(f) = unsafe { GetProcAddress(module, c.as_ptr()) } {
                let at = f as usize as u64;
                list.push((at, at + STUB));
            }
        }
    }
    list
}

fn thread_name(handle: HANDLE) -> String {
    let mut name = null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(handle, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..512).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

/// Bytes of this process at `at`, through the kernel (a read that fails
/// is no fault).
fn read(at: u64, len: usize) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    let mut got = 0;
    // SAFETY: the kernel copies into our buffer of `len` bytes.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as *const c_void,
            bytes.as_mut_ptr().cast(),
            len,
            &mut got,
        )
    };
    (ok != 0 && got == len).then_some(bytes)
}

/// The named exports of the image at `base` (`size` bytes) by address,
/// read through the kernel; empty if it has none or they cannot be read.
fn exports(base: u64, size: u64) -> Vec<(u32, String)> {
    let u32_at = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
    };
    let u16_at = |b: &[u8], at: usize| {
        b.get(at..at + 2)
            .map(|s| u16::from_le_bytes(s.try_into().expect("2 bytes")))
    };
    let list = || -> Option<Vec<(u32, String)>> {
        let head = read(base, 0x400.min(size as usize))?;
        let pe = u32_at(&head, 0x3c)? as usize;
        if head.get(..2)? != b"MZ"
            || u32_at(&head, pe)? != 0x4550
            || u16_at(&head, pe + 24)? != 0x20b
        {
            return None;
        }
        // The export directory: the first data directory of a PE32+ header.
        let (rva, len) = (u32_at(&head, pe + 24 + 112)?, u32_at(&head, pe + 24 + 116)?);
        if rva == 0 || len < 40 || u64::from(rva) + u64::from(len) > size || len > 16 << 20 {
            return None;
        }
        let dir = read(base + u64::from(rva), len as usize)?;
        let inside = |at: u32| {
            at.checked_sub(rva)
                .map(|o| o as usize)
                .filter(|&o| o < dir.len())
        };
        let (count, functions, names, ordinals) = (
            u32_at(&dir, 0x18)?,
            u32_at(&dir, 0x1c)?,
            u32_at(&dir, 0x20)?,
            u32_at(&dir, 0x24)?,
        );
        let all = read(
            base + u64::from(functions),
            4 * u32_at(&dir, 0x14)? as usize,
        )?;
        let mut out = Vec::new();
        for i in 0..count as usize {
            let Some(name) = inside(u32_at(&dir, inside(names)? + 4 * i)?) else {
                continue;
            };
            let ordinal = usize::from(u16_at(&dir, inside(ordinals)? + 2 * i)?);
            let Some(at) = u32_at(&all, 4 * ordinal) else {
                continue;
            };
            // Forwarded exports point into the directory: no code.
            if inside(at).is_some() {
                continue;
            }
            let end = dir[name..]
                .iter()
                .position(|&c| c == 0)
                .map_or(dir.len(), |e| name + e);
            out.push((at, String::from_utf8_lossy(&dir[name..end]).into_owned()));
        }
        // One name an address: `Nt` rather than its `Zw` twin, else the shortest.
        out.sort_unstable_by(|a, b| {
            let rank = |name: &str| (name.starts_with("Zw"), name.len());
            a.0.cmp(&b.0)
                .then(rank(&a.1).cmp(&rank(&b.1)))
                .then(a.1.cmp(&b.1))
        });
        out.dedup_by_key(|e| e.0);
        Some(out)
    };
    list().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Filter;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};

    fn config(threads: &str, file: &str) -> Config {
        Config {
            file: file.into(),
            period_ms: 1.0,
            after_s: 0.0,
            for_s: 0.0,
            write_s: 3600.0,
            threads: Filter::parse(threads),
            depth: 8,
            stack: 16 << 10,
            idle_every: 1,
            quiet_ms: 20.0,
            risk: 0.02,
        }
    }

    fn run_for(s: &mut Sampler, time: Duration) {
        let began = Instant::now();
        while began.elapsed() < time {
            let round = Instant::now();
            assert!(s.step());
            if let Some(rest) = Duration::from_micros(s.period_us).checked_sub(round.elapsed()) {
                std::thread::sleep(rest);
            }
        }
    }

    /// The image range of the function at `f`, by its unwind data.
    fn range_of(f: usize) -> (u64, u64) {
        let mut image = 0;
        // SAFETY: a lookup of our own code.
        let entry = unsafe { RtlLookupFunctionEntry(f as u64, &mut image, null_mut()) };
        assert!(!entry.is_null());
        // SAFETY: the table entry just found.
        let (begin, end) = unsafe { ((*entry).BeginAddress, (*entry).EndAddress) };
        (image + u64::from(begin), image + u64::from(end))
    }

    fn stats(s: &Sampler) -> String {
        format!(
            "{} rounds, {} suspensions ({} failed, {} left out), held {:.1} us on average, {} us at most",
            s.rounds,
            s.suspensions,
            s.failed,
            s.skipped,
            s.held_ns as f64 / 1000.0 / s.suspensions.max(1) as f64,
            s.held_max_ns / 1000
        )
    }

    /// Spins until `stop`, in a loop of its own instructions (no calls,
    /// even in a debug build).
    #[inline(never)]
    fn spin(stop: &AtomicBool) -> u64 {
        let mut n = 0u64;
        while !stop.load(Relaxed) {
            let mut left = 200_000u64;
            // SAFETY: counts a register down; touches nothing else.
            unsafe {
                std::arch::asm!("2:", "dec {0}", "jnz 2b", inout(reg) left, options(nomem, nostack))
            };
            n = n.wrapping_add(1 + left);
        }
        n
    }

    #[inline(never)]
    fn calls_spin(stop: &AtomicBool) -> u64 {
        std::hint::black_box(spin(stop)) ^ 1
    }

    fn spinner() -> (u32, Arc<AtomicBool>, std::thread::JoinHandle<u64>) {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn({
            let stop = stop.clone();
            move || {
                // SAFETY: a plain query.
                tx.send(unsafe { GetCurrentThreadId() }).unwrap();
                calls_spin(&stop)
            }
        });
        (rx.recv().unwrap(), stop, worker)
    }

    /// A thread spinning in a known function is found there: the hottest
    /// 64-byte bucket lies in that function, nearly every sample is in it
    /// and running, and the walk finds the function that called it.
    #[test]
    fn a_spinning_thread_is_found_in_its_function() {
        let (id, stop, worker) = spinner();
        let mut s = Sampler::new(config(&id.to_string(), "NUL"), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(1500));
        stop.store(true, Relaxed);
        worker.join().unwrap();
        eprintln!("spinning thread: {}", stats(&s));
        assert_eq!(s.threads.len(), 1, "only the spinning thread is watched");
        let t = &s.threads[0];
        assert_eq!(t.id, id);
        let samples = t.totals.acc.all.samples;
        assert!(samples >= 200, "{samples} samples: {}", stats(&s));
        let (begin, end) = range_of(spin as fn(&AtomicBool) -> u64 as usize);
        let (top, top_acc) = t
            .totals
            .buckets
            .iter()
            .max_by_key(|(at, a)| (a.running.us, **at))
            .unwrap();
        assert!(
            *top + 64 > begin && *top < end,
            "hottest bucket {top:#x} outside spin {begin:#x}..{end:#x}"
        );
        let inside: u64 = t
            .totals
            .buckets
            .iter()
            .filter(|(at, _)| **at + 64 > begin && **at < end)
            .map(|(_, a)| a.running.samples)
            .sum();
        assert!(
            inside * 100 >= samples * 95,
            "{inside} of {samples} samples in spin, top {top_acc:?}"
        );
        assert!(
            t.totals.acc.running.samples * 100 >= samples * 95,
            "{:?}",
            t.totals.acc
        );
        // A single watched thread runs alone whenever it runs.
        assert_eq!(t.totals.acc.alone.samples, t.totals.acc.running.samples);
        // The walk: the caller is `calls_spin`.
        let (from, to) = range_of(calls_spin as fn(&AtomicBool) -> u64 as usize);
        let chains = s.all.chains.as_ref().unwrap();
        let called: u64 = chains
            .iter()
            .filter(|(key, _)| key[1] >= from && key[1] < to)
            .map(|(_, c)| c.samples)
            .sum();
        assert!(
            called * 100 >= samples * 80,
            "{called} of {samples} samples walked to calls_spin"
        );
        let text = s.render(true);
        assert!(
            text.contains("== 64-byte buckets, running samples"),
            "{text}"
        );
    }

    /// Threads blocked in a channel, a sleep and an event wait read as waiting.
    #[test]
    fn waiting_threads_read_as_waiting() {
        use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent, WaitForSingleObject};
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        let stop = Arc::new(AtomicBool::new(false));
        // SAFETY: a fresh unnamed manual-reset event.
        let event = unsafe { CreateEventW(null(), 1, 0, null()) } as usize;
        let mut workers = Vec::new();
        for kind in 0..3 {
            let tx = tx.clone();
            let stop = stop.clone();
            workers.push(std::thread::spawn(move || {
                // SAFETY: a plain query.
                tx.send(unsafe { GetCurrentThreadId() }).unwrap();
                match kind {
                    0 => {
                        while !stop.load(Relaxed) {
                            std::thread::sleep(Duration::from_millis(50));
                        }
                    }
                    // SAFETY: the event lives until after the join.
                    1 => unsafe {
                        WaitForSingleObject(event as HANDLE, u32::MAX);
                    },
                    _ => {
                        while !stop.load(Relaxed) {
                            std::thread::park_timeout(Duration::from_millis(50));
                        }
                    }
                }
            }));
        }
        let ids: Vec<u32> = (0..3).map(|_| rx.recv().unwrap()).collect();
        std::thread::sleep(Duration::from_millis(20));
        let filter = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut s = Sampler::new(config(&filter, "NUL"), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(600));
        stop.store(true, Relaxed);
        // SAFETY: the event made above.
        unsafe {
            SetEvent(event as HANDLE);
        }
        for w in workers {
            w.join().unwrap();
        }
        // SAFETY: the event made above, not used again.
        unsafe { CloseHandle(event as HANDLE) };
        s.list_modules();
        eprintln!(
            "waiting threads: {}; {} wait functions known",
            stats(&s),
            s.waits.len()
        );
        assert!(s.waits.len() >= 8, "{} wait functions found", s.waits.len());
        assert_eq!(s.threads.len(), 3);
        for t in &s.threads {
            let (top, _) = t
                .totals
                .buckets
                .iter()
                .max_by_key(|(at, a)| (a.all.samples, **at))
                .unwrap();
            let acc = t.totals.acc;
            eprintln!(
                "  thread {}: {} samples, {} running, mostly at {}",
                t.id,
                acc.all.samples,
                acc.running.samples,
                s.name(*top)
            );
            assert!(acc.all.samples >= 50, "{acc:?}");
            assert!(
                acc.running.samples * 10 <= acc.all.samples,
                "thread {} read as running: {acc:?}",
                t.id
            );
        }
    }

    /// mac: an address outside every image is a wait once a thread stood
    /// there in 4 looks in a row, never inside an image, never at 0.
    #[test]
    fn a_still_unix_address_becomes_a_wait() {
        let at = 0x7ff8_0798_7580;
        assert!(!stands_still(0, 0, at, true), "first look");
        assert!(!stands_still(at, 0, at, true), "second look");
        assert!(!stands_still(at, 1, at, true), "third look");
        assert!(stands_still(at, 2, at, true), "fourth look");
        assert!(!stands_still(at, 2, at, false), "inside an image");
        assert!(!stands_still(at + 4, 2, at, true), "moved");
        assert!(!stands_still(0, 5, 0, true), "no address");
    }

    /// mac: a thread named as one of `UNIX_CALL_THREADS` counts as waiting
    /// in the image it started in; another thread doing the same runs.
    #[test]
    fn audio_timer_threads_count_as_waiting() {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<(usize, u32)>();
        let workers: Vec<_> = (0..2)
            .map(|which| {
                let (stop, tx) = (stop.clone(), tx.clone());
                let name = if which == 0 { UNIX_CALL_THREADS[0] } else { "plain spinner" };
                std::thread::Builder::new()
                    .name(name.into())
                    .spawn(move || {
                        // SAFETY: a plain query.
                        tx.send((which, unsafe { GetCurrentThreadId() })).unwrap();
                        calls_spin(&stop)
                    })
                    .unwrap()
            })
            .collect();
        let mut ids = [0u32; 2];
        for _ in 0..2 {
            let (which, id) = rx.recv().unwrap();
            ids[which] = id;
        }
        let filter = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut s = Sampler::new(config(&filter, "NUL"), Vec::new()).unwrap();
        s.list_modules();
        run_for(&mut s, Duration::from_millis(600));
        stop.store(true, Relaxed);
        for w in workers {
            w.join().unwrap();
        }
        eprintln!("audio timer and spinner: {}", stats(&s));
        assert_eq!(s.threads.len(), 2);
        for t in &s.threads {
            let acc = t.totals.acc;
            eprintln!("  thread {} \"{}\": {acc:?}", t.id, t.name);
            assert!(acc.all.samples >= 50, "{acc:?}");
            if t.id == ids[0] {
                assert_eq!(t.name, UNIX_CALL_THREADS[0]);
                assert!(acc.running.samples * 10 <= acc.all.samples, "the timer thread runs: {acc:?}");
            } else {
                assert!(acc.running.samples * 10 >= acc.all.samples * 9, "the spinner waits: {acc:?}");
            }
        }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreatePipe(read: *mut HANDLE, write: *mut HANDLE, attributes: *const c_void, size: u32) -> i32;
        fn ReadFile(file: HANDLE, buffer: *mut u8, len: u32, read: *mut u32, overlapped: *mut c_void) -> i32;
        fn WriteFile(file: HANDLE, buffer: *const u8, len: u32, written: *mut u32, overlapped: *mut c_void) -> i32;
    }

    /// mac: a thread blocked in a file read reads as in I/O, and one
    /// blocked in a registry watch as waiting; neither as running.
    #[test]
    fn file_reads_and_registry_watches_are_not_running() {
        use windows_sys::Win32::System::Registry::{
            HKEY, HKEY_CURRENT_USER, KEY_ALL_ACCESS, REG_NOTIFY_CHANGE_LAST_SET, REG_OPTION_VOLATILE, REG_SZ,
            RegCloseKey, RegCreateKeyExW, RegDeleteKeyW, RegNotifyChangeKeyValue, RegSetValueExW,
        };
        let (mut read, mut write): (HANDLE, HANDLE) = (null_mut(), null_mut());
        // SAFETY: out pointers to two handles.
        assert_ne!(unsafe { CreatePipe(&mut read, &mut write, null(), 0) }, 0);
        let name: Vec<u16> = "Software\\victory-profile-test".encode_utf16().chain([0]).collect();
        let mut key: HKEY = null_mut();
        // SAFETY: a volatile key of our own under the test prefix's user hive.
        assert_eq!(
            unsafe {
                RegCreateKeyExW(
                    HKEY_CURRENT_USER,
                    name.as_ptr(),
                    0,
                    null(),
                    REG_OPTION_VOLATILE,
                    KEY_ALL_ACCESS,
                    null(),
                    &mut key,
                    null_mut(),
                )
            },
            0
        );
        let (tx, rx) = std::sync::mpsc::channel::<(usize, u32)>();
        let (read_at, key_at) = (read as usize, key as usize);
        let reader = std::thread::spawn({
            let tx = tx.clone();
            move || {
                // SAFETY: a plain query.
                tx.send((0, unsafe { GetCurrentThreadId() })).unwrap();
                let (mut byte, mut got) = (0u8, 0u32);
                // SAFETY: the pipe lives until after the join; one byte of room.
                unsafe { ReadFile(read_at as HANDLE, &mut byte, 1, &mut got, null_mut()) };
                got
            }
        });
        let watcher = std::thread::spawn(move || {
            // SAFETY: a plain query.
            tx.send((1, unsafe { GetCurrentThreadId() })).unwrap();
            // SAFETY: the key lives until after the join; a synchronous watch.
            unsafe { RegNotifyChangeKeyValue(key_at as HKEY, 0, REG_NOTIFY_CHANGE_LAST_SET, null_mut(), 0) }
        });
        let mut ids = [0u32; 2];
        for _ in 0..2 {
            let (which, id) = rx.recv().unwrap();
            ids[which] = id;
        }
        std::thread::sleep(Duration::from_millis(50));
        let filter = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut s = Sampler::new(config(&filter, "NUL"), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(600));
        let value: Vec<u16> = "x".encode_utf16().chain([0]).collect();
        // SAFETY: our pipe and key; a value of two UTF-16 units.
        unsafe {
            let mut written = 0;
            WriteFile(write, [1u8].as_ptr(), 1, &mut written, null_mut());
            RegSetValueExW(key, value.as_ptr(), 0, REG_SZ, value.as_ptr().cast(), 4);
        }
        assert_eq!(reader.join().unwrap(), 1);
        assert_eq!(watcher.join().unwrap(), 0);
        // SAFETY: handles and key made above, not used again.
        unsafe {
            CloseHandle(read);
            CloseHandle(write);
            RegCloseKey(key);
            RegDeleteKeyW(HKEY_CURRENT_USER, name.as_ptr());
        }
        eprintln!("file read and registry watch: {}; {} I/O calls known", stats(&s), s.io.len());
        assert!(!s.io.is_empty());
        for t in &s.threads {
            let acc = t.totals.acc;
            eprintln!("  thread {}: {acc:?}", t.id);
            assert!(acc.all.samples >= 50, "{acc:?}");
            assert!(acc.running.samples * 10 <= acc.all.samples, "thread {} read as running: {acc:?}", t.id);
            if t.id == ids[0] {
                assert!(acc.io.samples * 10 >= acc.all.samples * 9, "the reader is not in I/O: {acc:?}");
            } else {
                assert_eq!(acc.io.samples, 0, "the watcher is in I/O: {acc:?}");
            }
        }
    }

    static GUARDED: AtomicUsize = AtomicUsize::new(0);
    static FAULTS: AtomicU64 = AtomicU64::new(0);

    /// Gives our guarded page its write access back, as the runtime's
    /// change tracking does (`crate::watch`).
    unsafe extern "system" fn on_fault(
        info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, VirtualProtect};
        // SAFETY: the system passes valid exception pointers.
        let record = unsafe { &*(*info).ExceptionRecord };
        let page = GUARDED.load(Relaxed);
        if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || page == 0 {
            return 0;
        }
        let at = record.ExceptionInformation[1];
        if at < page || at >= page + 4096 {
            return 0;
        }
        // As the runtime's change tracking does.
        crate::profile::excepted();
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        // SAFETY: our own page.
        unsafe { VirtualProtect(page as *const c_void, 4096, PAGE_READWRITE, &mut old) };
        FAULTS.fetch_add(1, Relaxed);
        -1
    }

    /// Sampling every half millisecond, threads that fault and recover
    /// (a guarded page given back by a vectored handler, as change
    /// tracking does; Wine's write watch, handled inside Wine), that call
    /// the system all the time, and that spin, all go on and end normally.
    fn survive(kinds: &[u32], time: Duration) {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_PROTECTION_FLAGS, PAGE_READONLY, ResetWriteWatch,
            VirtualProtect, VirtualQuery,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        use windows_sys::Win32::System::Threading::SwitchToThread;
        // SAFETY: our handler, removed below.
        let handler = unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
        assert!(!handler.is_null());
        // SAFETY: fresh pages of ours.
        let guarded =
            unsafe { VirtualAlloc(null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }
                as usize;
        // SAFETY: as above.
        let watched = unsafe {
            VirtualAlloc(
                null(),
                16 * 4096,
                MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(guarded != 0 && watched != 0);
        GUARDED.store(guarded, Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut workers = Vec::new();
        for &kind in kinds {
            let (stop, tx) = (stop.clone(), tx.clone());
            workers.push(std::thread::spawn(move || {
                // SAFETY: a plain query.
                tx.send((unsafe { GetCurrentThreadId() }, kind)).unwrap();
                let mut n = 0u64;
                while !stop.load(Relaxed) {
                    // SAFETY: our pages; the handler gives the guarded one
                    // back; every call is on our own memory.
                    unsafe {
                        match kind {
                            0 => {
                                let mut old: PAGE_PROTECTION_FLAGS = 0;
                                VirtualProtect(
                                    guarded as *const c_void,
                                    4096,
                                    PAGE_READONLY,
                                    &mut old,
                                );
                                (guarded as *mut u64).write_volatile(n);
                            }
                            1 => {
                                ResetWriteWatch(watched as *const c_void, 16 * 4096);
                                for p in 0..16 {
                                    ((watched + p * 4096) as *mut u64).write_volatile(n);
                                }
                            }
                            2 => {
                                SwitchToThread();
                                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                                VirtualQuery(
                                    watched as *const c_void,
                                    &mut info,
                                    size_of::<MEMORY_BASIC_INFORMATION>(),
                                );
                                std::hint::black_box(read(guarded as u64, 64));
                            }
                            _ => {
                                std::hint::black_box(spin(&AtomicBool::new(true)));
                                for _ in 0..1000 {
                                    n = std::hint::black_box(n.wrapping_mul(31).wrapping_add(7));
                                }
                            }
                        }
                    }
                    n += 1;
                }
                n
            }));
        }
        drop(tx);
        let ids: HashMap<u32, u32> = kinds.iter().map(|_| rx.recv().unwrap()).collect();
        let filter = ids.keys().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut c = config(&filter, "NUL");
        c.period_ms = 0.5;
        let mut s = Sampler::new(c, Vec::new()).unwrap();
        let faults_before = FAULTS.load(Relaxed);
        run_for(&mut s, time);
        stop.store(true, Relaxed);
        let counts: Vec<u64> = workers
            .into_iter()
            .map(|w| w.join().expect("a worker ended early"))
            .collect();
        let faults = FAULTS.load(Relaxed) - faults_before;
        GUARDED.store(0, Relaxed);
        // SAFETY: ours, not used again.
        unsafe {
            RemoveVectoredExceptionHandler(handler);
            VirtualFree(guarded as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(watched as *mut c_void, 0, MEM_RELEASE);
        }
        eprintln!(
            "faulting, calling and spinning threads: {}; {faults} guard faults; loops {counts:?}; samples per thread {:?}",
            stats(&s),
            s.threads
                .iter()
                .map(|t| t.totals.acc.all.samples)
                .collect::<Vec<_>>()
        );
        assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
        assert!(!kinds.contains(&0) || faults > 100, "{faults} faults");
        assert_eq!(s.failed, 0, "{}", stats(&s));
        assert!(s.threads.iter().all(|t| t.ended.is_none()) && s.lost.is_empty());
        for t in &s.threads {
            let kind = ids[&t.id];
            if kind == 0 {
                // It takes exceptions all the time: hardly ever looked at.
                assert!(
                    t.looks * 20 < s.rounds,
                    "{} looks in {} rounds",
                    t.looks,
                    s.rounds
                );
            } else {
                assert!(t.totals.acc.all.samples > 100, "kind {kind}: {}", stats(&s));
            }
        }
    }

    /// mac: "alone" is not zero where one real thread runs: a spinner
    /// beside threads that wait, read a file or fault all the time (left
    /// out of looks) is alone in nearly all its running samples, and the
    /// faulting thread's left-out looks are never running.
    #[test]
    fn a_thread_running_beside_waiting_reading_and_faulting_ones_is_alone() {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, PAGE_READONLY, VirtualProtect};
        // SAFETY: our handler, removed below.
        let handler = unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
        assert!(!handler.is_null());
        // SAFETY: a fresh page of ours.
        let guarded = unsafe { VirtualAlloc(null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(guarded, 0);
        GUARDED.store(guarded, Relaxed);
        let (mut read_end, mut write_end): (HANDLE, HANDLE) = (null_mut(), null_mut());
        // SAFETY: out pointers to two handles.
        assert_ne!(unsafe { CreatePipe(&mut read_end, &mut write_end, null(), 0) }, 0);
        let read_at = read_end as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<(usize, u32)>();
        // 0 spins, 1 sleeps, 2 reads a pipe, 3 faults on a guarded page.
        let workers: Vec<_> = (0..4)
            .map(|kind| {
                let (stop, tx) = (stop.clone(), tx.clone());
                std::thread::spawn(move || {
                    // SAFETY: a plain query.
                    tx.send((kind, unsafe { GetCurrentThreadId() })).unwrap();
                    let mut n = 0u64;
                    match kind {
                        0 => n = calls_spin(&stop),
                        1 => {
                            while !stop.load(Relaxed) {
                                std::thread::sleep(Duration::from_millis(20));
                            }
                        }
                        2 => {
                            let (mut byte, mut got) = (0u8, 0u32);
                            // SAFETY: the pipe lives until after the join.
                            unsafe { ReadFile(read_at as HANDLE, &mut byte, 1, &mut got, null_mut()) };
                        }
                        _ => {
                            while !stop.load(Relaxed) {
                                let mut old: PAGE_PROTECTION_FLAGS = 0;
                                // SAFETY: our page; the handler gives it back.
                                unsafe {
                                    VirtualProtect(guarded as *const c_void, 4096, PAGE_READONLY, &mut old);
                                    (guarded as *mut u64).write_volatile(n);
                                }
                                n += 1;
                            }
                        }
                    }
                    n
                })
            })
            .collect();
        let mut ids = [0u32; 4];
        for _ in 0..4 {
            let (kind, id) = rx.recv().unwrap();
            ids[kind] = id;
        }
        let filter = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let mut s = Sampler::new(config(&filter, "NUL"), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(1500));
        stop.store(true, Relaxed);
        // SAFETY: our pipe.
        unsafe {
            let mut written = 0;
            WriteFile(write_end, [1u8].as_ptr(), 1, &mut written, null_mut());
        }
        for w in workers {
            w.join().unwrap();
        }
        GUARDED.store(0, Relaxed);
        // SAFETY: ours, not used again.
        unsafe {
            RemoveVectoredExceptionHandler(handler);
            VirtualFree(guarded as *mut c_void, 0, MEM_RELEASE);
            CloseHandle(read_end);
            CloseHandle(write_end);
        }
        eprintln!("spinner beside waiting, reading and faulting threads: {}", stats(&s));
        let acc_of = |id: u32| s.threads.iter().find(|t| t.id == id).unwrap().totals.acc;
        for (kind, &id) in ids.iter().enumerate() {
            eprintln!("  kind {kind}, thread {id}: {:?}", acc_of(id));
        }
        let spinner = acc_of(ids[0]);
        assert!(spinner.running.samples >= 200, "{spinner:?}");
        assert!(
            spinner.alone.samples * 10 >= spinner.running.samples * 8,
            "the spinner is seldom alone: {spinner:?}"
        );
        let faulting = acc_of(ids[3]);
        assert!(s.skipped > 0, "no look was left out: {}", stats(&s));
        assert!(faulting.unseen.samples > 0, "{faulting:?}");
        assert!(acc_of(ids[1]).running.samples * 10 <= acc_of(ids[1]).all.samples);
        assert!(acc_of(ids[2]).io.samples * 10 >= acc_of(ids[2]).all.samples * 9);
        // The whole report: alone time is there, the left-out mark is not
        // in a running table.
        let text = s.render(true);
        assert!(!text.contains("not looked at (an exception)"), "{text}");
        assert!(s.all.acc.alone.samples > 0);
    }

    #[test]
    fn threads_that_fault_and_call_the_system_survive_sampling() {
        let kinds: Vec<u32> = std::env::var("PROFILE_KINDS").map_or(vec![0, 1, 2, 3], |v| {
            v.split(',').filter_map(|k| k.parse().ok()).collect()
        });
        survive(&kinds, Duration::from_secs(3));
    }

    /// The same for a minute (`--ignored`): a stronger check before a game run.
    #[test]
    #[ignore = "a minute long; run by hand"]
    fn threads_survive_a_minute_of_sampling() {
        survive(&[0, 1, 2, 3], Duration::from_secs(60));
    }

    static TRAPS: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn on_trap(
        info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use windows_sys::Win32::Foundation::{EXCEPTION_BREAKPOINT, EXCEPTION_ILLEGAL_INSTRUCTION};
        // SAFETY: the system passes valid exception pointers.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        let at = context.Rip as usize;
        let spot = trap_spot as fn(u32) as usize;
        if at < spot || at > spot + 64 {
            return 0;
        }
        match record.ExceptionCode {
            EXCEPTION_BREAKPOINT => {
                if (context.Rip as *const u8).cast::<u8>() as usize == at {
                    // SAFETY: our own code.
                    if unsafe { (at as *const u8).read() } == 0xcc {
                        context.Rip += 1;
                    }
                }
            }
            EXCEPTION_ILLEGAL_INSTRUCTION => context.Rip += 2,
            _ => return 0,
        }
        crate::profile::excepted();
        TRAPS.fetch_add(1, Relaxed);
        -1
    }

    #[inline(never)]
    fn trap_spot(how: u32) {
        // SAFETY: the handler steps over the trap.
        unsafe {
            if how == 0 {
                std::arch::asm!("int3");
            } else {
                std::arch::asm!("ud2");
            }
        }
    }

    /// How soon a thread that faults ends under sampling (`--ignored`,
    /// by hand): `PROFILE_MODE` veh (a guarded page given back by a
    /// vectored handler), watch (Wine's write watch), int3 or ud2 (traps
    /// stepped over by a vectored handler); `PROFILE_GAP_US` of spinning
    /// between faults; `PROFILE_MS` the period; `PROFILE_SECS` at most.
    #[test]
    #[ignore = "an experiment; run by hand"]
    fn fault_hazard() {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Memory::{
            PAGE_PROTECTION_FLAGS, PAGE_READONLY, ResetWriteWatch, VirtualProtect,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let env = |name: &str, default: f64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let mode = std::env::var("PROFILE_MODE").unwrap_or_else(|_| "veh".into());
        let (gap_us, period_ms, secs) = (
            env("PROFILE_GAP_US", 0.0),
            env("PROFILE_MS", 0.5),
            env("PROFILE_SECS", 10.0),
        );
        // SAFETY: our handlers, removed below.
        let handlers = unsafe {
            [
                AddVectoredExceptionHandler(1, Some(on_fault)),
                AddVectoredExceptionHandler(1, Some(on_trap)),
            ]
        };
        // SAFETY: fresh pages of ours.
        let guarded =
            unsafe { VirtualAlloc(null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }
                as usize;
        // SAFETY: as above.
        let watched = unsafe {
            VirtualAlloc(
                null(),
                4096,
                MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        GUARDED.store(guarded, Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let beat = Arc::new(AtomicU64::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn({
            let (stop, mode, beat) = (stop.clone(), mode.clone(), beat.clone());
            move || {
                // SAFETY: a plain query.
                tx.send(unsafe { GetCurrentThreadId() }).unwrap();
                let mut n = 0u64;
                // Plain instructions between faults, no system call (about
                // a thousand loops a microsecond under Rosetta: measured
                // from the fault rate).
                let loops = (gap_us * 1000.0) as u64;
                while !stop.load(Relaxed) {
                    let mut left = loops.max(1);
                    // SAFETY: counts a register down; touches nothing else.
                    unsafe {
                        std::arch::asm!("2:", "dec {0}", "jnz 2b", inout(reg) left, options(nomem, nostack))
                    };
                    std::hint::black_box(left);
                    // SAFETY: our pages and traps, handled above.
                    unsafe {
                        match mode.as_str() {
                            "watch" => {
                                ResetWriteWatch(watched as *const c_void, 4096);
                                (watched as *mut u64).write_volatile(n);
                            }
                            "int3" => trap_spot(0),
                            "ud2" => trap_spot(1),
                            "none" => {}
                            "burst" => {
                                for _ in 0..20 {
                                    let mut old: PAGE_PROTECTION_FLAGS = 0;
                                    VirtualProtect(
                                        guarded as *const c_void,
                                        4096,
                                        PAGE_READONLY,
                                        &mut old,
                                    );
                                    (guarded as *mut u64).write_volatile(n);
                                }
                            }
                            _ => {
                                let mut old: PAGE_PROTECTION_FLAGS = 0;
                                VirtualProtect(
                                    guarded as *const c_void,
                                    4096,
                                    PAGE_READONLY,
                                    &mut old,
                                );
                                (guarded as *mut u64).write_volatile(n);
                            }
                        }
                    }
                    n += 1;
                    beat.store(n, Relaxed);
                }
                n
            }
        });
        let id = rx.recv().unwrap();
        let mut c = config(&id.to_string(), "NUL");
        c.period_ms = period_ms;
        c.quiet_ms = env("PROFILE_QUIET_MS", 20.0);
        c.risk = env("PROFILE_RISK", 0.02);
        let mut s = Sampler::new(c, Vec::new()).unwrap();
        let (faults0, traps0) = (FAULTS.load(Relaxed), TRAPS.load(Relaxed));
        let began = Instant::now();
        let (mut last_beat, mut beat_at) = (0, Instant::now());
        let mut died = false;
        while began.elapsed().as_secs_f64() < secs {
            let round = Instant::now();
            s.step();
            let now = beat.load(Relaxed);
            if now != last_beat {
                (last_beat, beat_at) = (now, Instant::now());
            } else if beat_at.elapsed()
                > Duration::from_millis(300) + Duration::from_secs_f64(gap_us / 1e6)
            {
                died = true;
                break;
            }
            if let Some(rest) = Duration::from_micros(s.period_us).checked_sub(round.elapsed()) {
                std::thread::sleep(rest);
            }
        }
        let lived = if died {
            beat_at - began
        } else {
            began.elapsed()
        }
        .as_secs_f64();
        s.list_modules();
        let outside: Vec<String> = s.threads[0]
            .totals
            .buckets
            .iter()
            .filter(|(at, _)| s.modules.find(**at).is_none())
            .map(|(at, a)| format!("{at:#x} x{}", a.all.samples))
            .collect();
        let top: Vec<String> = {
            let mut b: Vec<_> = s.threads[0].totals.buckets.iter().collect();
            b.sort_by_key(|(_, a)| std::cmp::Reverse(a.all.samples));
            b.iter()
                .take(6)
                .map(|(at, a)| format!("{} x{}", s.name(**at), a.all.samples))
                .collect()
        };
        eprintln!("HAZARD samples outside images: {outside:?}; top {top:?}");
        stop.store(true, Relaxed);
        let faults = FAULTS.load(Relaxed) - faults0 + TRAPS.load(Relaxed) - traps0;
        // The sampler's own view of the loss, as in a run.
        if died {
            for _ in 0..5 {
                s.step();
                std::thread::sleep(Duration::from_millis(300));
            }
        }
        eprintln!(
            "HAZARD mode {mode} gap {gap_us} us period {period_ms} ms quiet {} ms: {} after {lived:.2} s; {}; {faults} faults ({:.0}/s), loops {last_beat}; sampler lost {:?}, ended {:?}; looks {}, risked {:.3}",
            s.config.quiet_ms,
            if died { "DIED" } else { "survived" },
            stats(&s),
            faults as f64 / lived,
            s.lost,
            s.threads.first().and_then(|t| t.ended),
            s.threads.first().map_or(0, |t| t.looks),
            s.risked
        );
        if died {
            // Its join would panic: it never finished.
            std::mem::forget(worker);
        } else {
            worker.join().unwrap();
        }
        GUARDED.store(0, Relaxed);
        // SAFETY: ours, not used again.
        unsafe {
            for h in handlers {
                RemoveVectoredExceptionHandler(h);
            }
        }
    }

    /// The report goes to its file, through a temporary one, and names
    /// this test's executable as the runtime's image.
    #[test]
    fn the_report_is_written_with_the_images() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("profile.txt");
        let (id, stop, worker) = spinner();
        let mut s =
            Sampler::new(config(&id.to_string(), file.to_str().unwrap()), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(300));
        stop.store(true, Relaxed);
        worker.join().unwrap();
        // It ended the normal way, through its thread detach: not lost.
        s.list_threads();
        assert_eq!(s.threads[0].ended, Some(0));
        assert!(s.lost.is_empty(), "{:?}", s.lost);
        s.write(true);
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(!dir.path().join("profile.txt.part").exists());
        let exe = std::env::current_exe().unwrap();
        let exe = exe.file_name().unwrap().to_string_lossy();
        assert!(
            text.starts_with("VIctory profile (VICTORY_PROFILE="),
            "{text}"
        );
        assert!(text.contains(", final\n"), "{text}");
        assert!(text.contains(&format!("{exe} at 0x")), "{text}");
        assert!(text.contains("(runtime)"), "{text}");
        assert!(text.contains(&format!("thread {id} ")), "{text}");
    }

    /// A report of a spinning thread to `PROFILE_REPORT` (`--ignored`, by
    /// hand), to try `tools/macos/symbolize-profile.sh` on.
    #[test]
    #[ignore = "writes a report for the symbolizer; run by hand"]
    fn a_report_for_the_symbolizer() {
        let file = std::env::var("PROFILE_REPORT").expect("PROFILE_REPORT names the file");
        let (id, stop, worker) = spinner();
        let mut s = Sampler::new(config(&id.to_string(), &file), Vec::new()).unwrap();
        run_for(&mut s, Duration::from_millis(500));
        stop.store(true, Relaxed);
        worker.join().unwrap();
        s.write(true);
    }

    /// The exports of a system DLL name its addresses.
    #[test]
    fn system_addresses_are_named_by_export() {
        let mut s = Sampler::new(config("0", "NUL"), Vec::new()).unwrap();
        s.list_modules();
        let ntdll = s
            .modules
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("ntdll.dll"))
            .unwrap()
            .clone();
        s.exports
            .insert(ntdll.base, exports(ntdll.base, ntdll.size));
        let (wait, _) = s.waits[0];
        let name = s.name(wait + 0x14);
        assert!(name.contains("NtWaitForSingleObject+0x14"), "{name}");
        assert!(name.starts_with("ntdll.dll+0x"), "{name}");
    }
}
