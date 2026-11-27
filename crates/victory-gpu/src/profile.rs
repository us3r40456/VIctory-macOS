//! mac: where the process's threads spend their time
//! (`VICTORY_PROFILE=<file>`), from inside the process.
//!
//! Under Rosetta every Windows instruction runs as translated code: a Mac
//! sampler (Instruments, `sample`) sees the ARM addresses of the
//! translation and cannot tell the title's code from ours. A thread of ours
//! inside the process sees each thread's x86 registers instead.
//!
//! Ported from Belobug's sampler of the submitting thread (belobug
//! ffedb7da, `crates/victory-gpu/src/profile.rs` there) and extended:
//! every thread of the process by default (`VICTORY_PROFILE_THREADS` picks
//! some by name or kind), the instruction pointer and a few return
//! addresses, sums by module, function, 4 KiB page and 64-byte bucket, the
//! samples taken while one sampled thread ran alone, and the presents.
//!
//! Every `VICTORY_PROFILE_MS` milliseconds (1) a round looks at each
//! watched thread (`sampler`): it suspends the thread, reads its registers
//! and copies up to `VICTORY_PROFILE_STACK_KB` (16) of its stack, and
//! resumes it. Between the suspension and the resumption nothing of ours
//! allocates, takes a lock or calls the loader, any of which the stopped
//! thread may hold. The copy is walked afterwards by each image's unwind
//! data, every read checked to stay inside the copy (`walk`, from
//! hyphennn f85101b). A thread found waiting in its last four samples is
//! looked at only every `VICTORY_PROFILE_IDLE` rounds (16); each sample
//! stands for the time since that thread's previous one.
//!
//! The report (`report`) is written every `VICTORY_PROFILE_WRITE` seconds
//! (10), through a temporary file, so a run that is killed keeps what it
//! had, and at exit. `VICTORY_PROFILE_AFTER` and `VICTORY_PROFILE_FOR`
//! (seconds) set the window. Off unless `VICTORY_PROFILE` names a file:
//! then nothing here runs, and nothing else changes.
//!
//! Under Wine on Rosetta a suspension that meets a thread's exception (a
//! fault or trap dispatched to the vectored handlers, such as the change
//! tracking's guard faults under `VICTORY_FAULT_TRACKING=1`) ends that
//! thread: Wine logs `nested exception on signal stack` at `_sigtramp` and
//! lets the thread go (measured: about 0.35 us of exposure per exception
//! per look; Wine's own write-watch faults, system calls and waits are
//! safe). So the runtime's handlers mark the thread taking an exception
//! ([`excepted`]), and a thread marked within the last
//! `VICTORY_PROFILE_QUIET_MS` (20) is not looked at: its time is counted
//! as "not looked at". What is left is an exception that starts just as a
//! look does (measured: a look at a thread that starts exceptions at a
//! rate r loses it with a chance of about r * 10 us). So each thread's
//! looks are spaced by its measured rate of starts, to keep the expected
//! loss under `VICTORY_PROFILE_RISK` (0.02) threads a minute for each
//! such thread; the report gives what was risked. A thread lost that way
//! ends without the normal thread exit: the sampler sees that, stops and
//! says so in the report.

#![cfg_attr(not(windows), allow(dead_code))]

// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod report;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
mod walk;

/// The switches, read once.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config {
    /// Where the report goes, as the process names files (a Windows path).
    pub(crate) file: String,
    /// The time between rounds.
    pub(crate) period_ms: f64,
    /// Seconds from the start before the first round.
    pub(crate) after_s: f64,
    /// Seconds of sampling; 0: until the process ends.
    pub(crate) for_s: f64,
    /// Seconds between reports.
    pub(crate) write_s: f64,
    /// Which threads are watched.
    pub(crate) threads: Filter,
    /// Addresses kept of a sample: the instruction pointer and return addresses.
    pub(crate) depth: usize,
    /// Bytes of a stack copied at most.
    pub(crate) stack: usize,
    /// A waiting thread is looked at every this many rounds.
    pub(crate) idle_every: u32,
    /// A thread that took an exception this recently is not looked at.
    pub(crate) quiet_ms: f64,
    /// Threads expected lost a minute, for each thread that starts
    /// exceptions, that the looks may risk: they are spaced to stay under.
    pub(crate) risk: f64,
}

/// The most addresses a sample keeps.
pub(crate) const MAX_DEPTH: usize = 32;

impl Config {
    /// From `get` (the environment); `None` unless `VICTORY_PROFILE` names a file.
    pub(crate) fn from(get: impl Fn(&str) -> Option<String>) -> Option<Config> {
        let file = get("VICTORY_PROFILE")?.trim().to_string();
        if file.is_empty() || file == "0" {
            return None;
        }
        let number = |name: &str, default: f64, low: f64, high: f64| {
            get(name)
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .map_or(default, |v| v.clamp(low, high))
        };
        Some(Config {
            file,
            period_ms: number("VICTORY_PROFILE_MS", 1.0, 0.1, 1000.0),
            after_s: number("VICTORY_PROFILE_AFTER", 0.0, 0.0, 1e6),
            for_s: number("VICTORY_PROFILE_FOR", 0.0, 0.0, 1e6),
            write_s: number("VICTORY_PROFILE_WRITE", 10.0, 1.0, 3600.0),
            threads: Filter::parse(&get("VICTORY_PROFILE_THREADS").unwrap_or_default()),
            depth: number("VICTORY_PROFILE_DEPTH", 8.0, 1.0, MAX_DEPTH as f64) as usize,
            stack: number("VICTORY_PROFILE_STACK_KB", 16.0, 0.0, 256.0) as usize * 1024,
            idle_every: number("VICTORY_PROFILE_IDLE", 16.0, 1.0, 1000.0) as u32,
            quiet_ms: number("VICTORY_PROFILE_QUIET_MS", 20.0, 0.0, 10_000.0),
            risk: number("VICTORY_PROFILE_RISK", 0.02, 0.0, 100.0),
        })
    }

    /// The line the launcher prints.
    pub(crate) fn describe(&self) -> String {
        format!(
            "on: every {} ms, {} from {} s{}, threads {}, none within {} ms of an exception, report every {} s to {}",
            self.period_ms,
            if self.for_s > 0.0 {
                format!("for {} s", self.for_s)
            } else {
                "until the end".into()
            },
            self.after_s,
            if self.idle_every > 1 {
                format!(" (waiting threads every {} rounds)", self.idle_every)
            } else {
                String::new()
            },
            if self.threads.is_all() {
                "all".to_string()
            } else {
                self.threads.text.clone()
            },
            self.quiet_ms,
            self.write_s,
            self.file
        )
    }
}

/// Which threads are watched (`VICTORY_PROFILE_THREADS`): a comma list of
/// words, any of which picks a thread. `title`, `runtime` and `system`:
/// threads that started in the title's images, in victory.exe, or
/// elsewhere; `unnamed`: threads without a name; a number: that thread id;
/// any other word: threads whose name holds it (any case). Empty or `all`:
/// every thread.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Filter {
    words: Vec<Word>,
    text: String,
}

#[derive(Clone, Debug, PartialEq)]
enum Word {
    Title,
    Runtime,
    System,
    Unnamed,
    Id(u32),
    Name(String),
}

/// Where a thread started, for [`Filter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Started {
    Title,
    Runtime,
    System,
    Unknown,
}

impl Filter {
    pub(crate) fn parse(text: &str) -> Filter {
        let mut words = Vec::new();
        for word in text.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            let lower = word.to_ascii_lowercase();
            words.push(match lower.as_str() {
                "all" => return Filter::default(),
                "title" => Word::Title,
                "runtime" => Word::Runtime,
                "system" => Word::System,
                "unnamed" => Word::Unnamed,
                _ => match lower.parse::<u32>() {
                    Ok(id) => Word::Id(id),
                    Err(_) => Word::Name(lower),
                },
            });
        }
        Filter {
            text: if words.is_empty() {
                String::new()
            } else {
                text.trim().to_string()
            },
            words,
        }
    }

    pub(crate) fn is_all(&self) -> bool {
        self.words.is_empty()
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn picks(&self, id: u32, name: &str, started: Started) -> bool {
        if self.words.is_empty() {
            return true;
        }
        let lower = name.to_ascii_lowercase();
        self.words.iter().any(|w| match w {
            Word::Title => started == Started::Title,
            Word::Runtime => started == Started::Runtime,
            Word::System => started == Started::System,
            Word::Unnamed => name.trim().is_empty(),
            Word::Id(n) => *n == id,
            Word::Name(part) => lower.contains(part.as_str()),
        })
    }
}

/// Places in the tables of exceptions taken, by thread id.
pub(crate) const PLACES: usize = 4096;

/// Whether the sampler runs: only then do the handlers mark exceptions.
static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The sampler's round, which the marks name.
static ROUND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// By thread (its id's place): the round of its last exception.
static LAST: [std::sync::atomic::AtomicU64; PLACES] =
    [const { std::sync::atomic::AtomicU64::new(0) }; PLACES];
/// By thread: exceptions, and those after a quiet time (starts).
static EXCEPTIONS: [std::sync::atomic::AtomicU32; PLACES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; PLACES];
static STARTS: [std::sync::atomic::AtomicU32; PLACES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; PLACES];
/// Rounds of quiet after which an exception is a start.
static QUIET_ROUNDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The place of thread `id` in the tables (Wine's thread ids are
/// multiples of four). Two threads may share one: then each counts as
/// marked when the other is, which errs on the safe side.
pub(crate) fn place(id: u32) -> usize {
    (id as usize / 4) % PLACES
}

/// mac: the current thread is taking an exception (called first thing by
/// the runtime's vectored handlers): the sampler leaves it alone for a
/// while. Two loads when the sampler is off; no lock, no system call.
#[inline]
pub(crate) fn excepted() {
    use std::sync::atomic::Ordering::Relaxed;
    if !ON.load(Relaxed) {
        return;
    }
    #[cfg(windows)]
    {
        // SAFETY: a plain query (it reads the thread's own block).
        let at = place(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() });
        mark(at, ROUND.load(Relaxed) + 1, QUIET_ROUNDS.load(Relaxed));
    }
}

/// Marks place `at` as having taken an exception in `round` (the
/// sampler's round plus one: 0 is never), a start if none came in the
/// `quiet` rounds before.
fn mark(at: usize, round: u64, quiet: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let last = LAST[at].swap(round, Relaxed);
    EXCEPTIONS[at].fetch_add(1, Relaxed);
    if last == 0 || round.saturating_sub(last) > quiet {
        STARTS[at].fetch_add(1, Relaxed);
    }
}

/// Whether place `at` took an exception within the `quiet` rounds before
/// `round` (counted as [`mark`] counts them).
fn marked(at: usize, round: u64, quiet: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let last = LAST[at].load(Relaxed);
    last != 0 && round.saturating_sub(last) <= quiet
}

/// Starts the sampler if `VICTORY_PROFILE` asks for one, with the title's
/// images (`(file, base, size)`) named in its report; the line to print, or
/// `None` when it is off.
pub fn start<'a>(title: impl IntoIterator<Item = (&'a str, usize, usize)>) -> Option<String> {
    let config = Config::from(|name| std::env::var(name).ok())?;
    #[cfg(windows)]
    {
        let title: Vec<(String, u64, u64)> = title
            .into_iter()
            .map(|(name, base, size)| (name.to_string(), base as u64, size as u64))
            .collect();
        Some(match sampler::start(config.clone(), title) {
            Ok(()) => config.describe(),
            Err(why) => format!("not started: {why}"),
        })
    }
    #[cfg(not(windows))]
    {
        let _ = (config, title.into_iter().count());
        Some("not started: Windows only".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pairs: &[(&str, &str)]) -> Option<Config> {
        Config::from(|name| {
            pairs
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn off_unless_a_file_is_named() {
        assert_eq!(config(&[]), None);
        assert_eq!(config(&[("VICTORY_PROFILE", "")]), None);
        assert_eq!(config(&[("VICTORY_PROFILE", " 0 ")]), None);
        let c = config(&[("VICTORY_PROFILE", "Z:\\tmp\\p.txt")]).unwrap();
        assert_eq!(
            c,
            Config {
                file: "Z:\\tmp\\p.txt".into(),
                period_ms: 1.0,
                after_s: 0.0,
                for_s: 0.0,
                write_s: 10.0,
                threads: Filter::default(),
                depth: 8,
                stack: 16 * 1024,
                idle_every: 16,
                quiet_ms: 20.0,
                risk: 0.02,
            }
        );
        assert!(
            c.describe().contains("every 1 ms, until the end from 0 s"),
            "{}",
            c.describe()
        );
    }

    #[test]
    fn numbers_are_read_and_kept_in_range() {
        let c = config(&[
            ("VICTORY_PROFILE", "p.txt"),
            ("VICTORY_PROFILE_MS", "0.5"),
            ("VICTORY_PROFILE_AFTER", "110"),
            ("VICTORY_PROFILE_FOR", "60"),
            ("VICTORY_PROFILE_WRITE", "0"),
            ("VICTORY_PROFILE_DEPTH", "500"),
            ("VICTORY_PROFILE_STACK_KB", "4"),
            ("VICTORY_PROFILE_IDLE", "x"),
        ])
        .unwrap();
        assert_eq!(c.period_ms, 0.5);
        assert_eq!((c.after_s, c.for_s), (110.0, 60.0));
        assert_eq!(c.write_s, 1.0);
        assert_eq!(c.depth, MAX_DEPTH);
        assert_eq!(c.stack, 4096);
        assert_eq!(c.idle_every, 16);
        let c = config(&[("VICTORY_PROFILE", "p.txt"), ("VICTORY_PROFILE_MS", "inf")]).unwrap();
        assert_eq!(c.period_ms, 1.0);
    }

    #[test]
    fn exceptions_mark_a_place_for_some_rounds_and_count_starts() {
        use std::sync::atomic::Ordering::Relaxed;
        // A place no thread of the test process has.
        let at = place(4 * 4000 + 8 + 4 * PLACES as u32);
        assert_eq!(at, 4002);
        assert!(!marked(at, 10, 2));
        mark(at, 10, 2);
        mark(at, 10, 2);
        assert!(marked(at, 10, 2));
        assert!(marked(at, 12, 2));
        assert!(!marked(at, 13, 2));
        assert_eq!(
            (EXCEPTIONS[at].load(Relaxed), STARTS[at].load(Relaxed)),
            (2, 1)
        );
        mark(at, 12, 2);
        assert_eq!(
            (EXCEPTIONS[at].load(Relaxed), STARTS[at].load(Relaxed)),
            (3, 1)
        );
        mark(at, 20, 2);
        assert_eq!(
            (EXCEPTIONS[at].load(Relaxed), STARTS[at].load(Relaxed)),
            (4, 2)
        );
    }

    #[test]
    fn threads_are_picked_by_kind_name_or_id() {
        let all = Filter::parse(" all ");
        assert!(all.is_all());
        assert!(all.picks(1, "", Started::Unknown));
        let f = Filter::parse("title, heap-sync,4242,unnamed");
        assert_eq!(f.text(), "title, heap-sync,4242,unnamed");
        assert!(f.picks(7, "x", Started::Title));
        assert!(f.picks(7, "victory-HEAP-SYNC-2", Started::Runtime));
        assert!(f.picks(4242, "dstorage 1", Started::Runtime));
        assert!(f.picks(8, " ", Started::System));
        assert!(!f.picks(8, "victory-window", Started::Runtime));
        let f = Filter::parse("runtime");
        assert!(f.picks(1, "a", Started::Runtime));
        assert!(!f.picks(1, "a", Started::System));
        assert!(Filter::parse("system").picks(1, "", Started::System));
    }
}
