//! The profile's sums and its report text (`super`): plain std, so its
//! tests run on any host.
//!
//! A sample is a thread's instruction pointer and the return addresses
//! above it, with the wall time it stands for (the time since that
//! thread's previous sample), whether the thread was running (not in a
//! wait) and whether it was the only sampled thread running in its round.
//! Sums are kept by absolute address; modules are put to them when the
//! report is written, from the module list of that moment.

use std::collections::HashMap;
use std::fmt::Write as _;

/// Return addresses kept with a bucket for the callers' table.
pub(crate) const CALLERS: usize = 3;

/// Caller chains kept at most (each 40 bytes and a count).
const MAX_CHAINS: usize = 1 << 18;

/// The address of a look left out (the thread had just taken an exception).
pub(crate) const SKIPPED: u64 = 1;

/// Exposure of a look to an exception starting in its thread, measured
/// under Wine on Rosetta (`sampler`'s `fault_hazard`, compute between
/// exceptions: 3 threads lost in 11,749 looks at about 30 starts a second,
/// 7 to 20 us): a look at a thread that starts exceptions at a rate r
/// loses it with a chance of about r * this.
pub(crate) const EXPOSURE_US: f64 = 10.0;

/// What a loaded image is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The runtime's own executable (`victory.exe`).
    Runtime,
    /// An image of the title (the loader's prepared copies).
    Title,
    /// Anything else: Wine's DLLs, the Windows side of D3DMetal.
    System,
}

impl Kind {
    pub(crate) fn word(self) -> &'static str {
        match self {
            Kind::Runtime => "runtime",
            Kind::Title => "title",
            Kind::System => "system",
        }
    }
}

/// A loaded image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Module {
    pub(crate) name: String,
    pub(crate) base: u64,
    pub(crate) size: u64,
    pub(crate) kind: Kind,
    /// What the image is to us, if known (the title file it was made from).
    pub(crate) note: String,
}

/// The loaded images, by address.
#[derive(Clone, Debug, Default)]
pub(crate) struct Modules {
    list: Vec<Module>,
}

impl Modules {
    /// Sorted by base; of images listed over each other the first stays.
    pub(crate) fn new(mut list: Vec<Module>) -> Modules {
        list.retain(|m| m.size > 0);
        list.sort_by_key(|m| m.base);
        let mut kept: Vec<Module> = Vec::with_capacity(list.len());
        for m in list {
            if kept
                .last()
                .is_some_and(|k| m.base < k.base.saturating_add(k.size))
            {
                continue;
            }
            kept.push(m);
        }
        Modules { list: kept }
    }

    /// The image holding `at`.
    pub(crate) fn find(&self, at: u64) -> Option<&Module> {
        let i = self.list.partition_point(|m| m.base <= at);
        let m = self.list.get(i.checked_sub(1)?)?;
        (at - m.base < m.size).then_some(m)
    }

    /// `module+0xrva`, or the bare address outside every image.
    pub(crate) fn place(&self, at: u64) -> String {
        match self.find(at) {
            Some(m) => format!("{}+{:#x}", m.name, at - m.base),
            None => format!("{at:#x}"),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Module> {
        self.list.iter()
    }
}

/// The 64-byte bucket of an address.
pub(crate) fn bucket(at: u64) -> u64 {
    at & !63
}

/// The 4 KiB page of an address.
pub(crate) fn page(at: u64) -> u64 {
    at & !0xfff
}

/// Samples and the wall time they stand for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Count {
    pub(crate) samples: u64,
    pub(crate) us: u64,
}

impl Count {
    fn add(&mut self, us: u64) {
        self.samples += 1;
        self.us += us;
    }
}

impl std::ops::AddAssign for Count {
    fn add_assign(&mut self, other: Count) {
        self.samples += other.samples;
        self.us += other.us;
    }
}

/// All samples of something, those running, those alone, those in file
/// I/O and those not looked at; the rest waited.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Acc {
    pub(crate) all: Count,
    pub(crate) running: Count,
    pub(crate) alone: Count,
    /// mac: in a file read (`NtReadFile`): neither running nor waiting.
    pub(crate) io: Count,
    /// mac: not looked at (the thread had just taken an exception): what
    /// it did is not known, so it is neither running nor waiting.
    pub(crate) unseen: Count,
}

impl Acc {
    fn add(&mut self, s: &Sample) {
        self.all.add(s.us);
        if s.running {
            self.running.add(s.us);
        }
        if s.alone {
            self.alone.add(s.us);
        }
        if s.io {
            self.io.add(s.us);
        }
        if s.unseen {
            self.unseen.add(s.us);
        }
    }

    /// The time in waits: what is neither running, nor in I/O, nor unseen.
    pub(crate) fn waiting_us(&self) -> u64 {
        self.all.us - self.running.us - self.io.us - self.unseen.us
    }
}

impl std::ops::AddAssign for Acc {
    fn add_assign(&mut self, other: Acc) {
        self.all += other.all;
        self.running += other.running;
        self.alone += other.alone;
        self.io += other.io;
        self.unseen += other.unseen;
    }
}

/// One sample, as the sampler hands it over.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample<'a> {
    /// The instruction pointer, then return addresses, each minus one
    /// (inside its call instruction).
    pub(crate) frames: &'a [u64],
    /// Where the function holding the instruction pointer starts, by its
    /// image's unwind data; 0 for code without it (leaf functions).
    pub(crate) function: u64,
    /// The wall time the sample stands for.
    pub(crate) us: u64,
    /// Looked at, and in neither a wait nor file I/O.
    pub(crate) running: bool,
    /// The only sampled thread seen running in its round.
    pub(crate) alone: bool,
    /// mac: in a file read.
    pub(crate) io: bool,
    /// mac: not looked at (the [`SKIPPED`] mark).
    pub(crate) unseen: bool,
}

/// The sums of a set of samples (a thread's, or all).
#[derive(Debug, Default)]
pub(crate) struct Totals {
    pub(crate) acc: Acc,
    /// Samples with at least one return address.
    pub(crate) walked: u64,
    /// By the instruction pointer's 64-byte bucket.
    pub(crate) buckets: HashMap<u64, Acc>,
    /// By function start; code without unwind data by its bucket with the
    /// lowest bit set.
    pub(crate) functions: HashMap<u64, Acc>,
    /// Running samples by bucket and [`CALLERS`] return addresses, if kept.
    pub(crate) chains: Option<HashMap<[u64; CALLERS + 1], Count>>,
    /// Running samples left out of `chains` for want of room.
    pub(crate) chains_dropped: u64,
}

impl Totals {
    /// Sums that also keep caller chains.
    pub(crate) fn with_callers() -> Totals {
        Totals {
            chains: Some(HashMap::new()),
            ..Totals::default()
        }
    }

    pub(crate) fn add(&mut self, s: &Sample) {
        let Some(&rip) = s.frames.first() else {
            return;
        };
        self.acc.add(s);
        if s.frames.len() > 1 {
            self.walked += 1;
        }
        self.buckets.entry(bucket(rip)).or_default().add(s);
        let function = if s.function != 0 {
            s.function
        } else {
            bucket(rip) | 1
        };
        self.functions.entry(function).or_default().add(s);
        if s.running
            && let Some(chains) = &mut self.chains
        {
            let mut key = [0; CALLERS + 1];
            key[0] = bucket(rip);
            for (k, &f) in key[1..].iter_mut().zip(&s.frames[1..]) {
                *k = f;
            }
            let full = chains.len() >= MAX_CHAINS;
            match chains.get_mut(&key) {
                Some(c) => c.add(s.us),
                None if !full => {
                    chains.insert(
                        key,
                        Count {
                            samples: 1,
                            us: s.us,
                        },
                    );
                }
                None => self.chains_dropped += 1,
            }
        }
    }

    /// The bucket sums put together by `key` (a page, a module).
    fn by(&self, key: impl Fn(u64) -> u64) -> Vec<(u64, Acc)> {
        let mut sums: HashMap<u64, Acc> = HashMap::new();
        for (&at, &acc) in &self.buckets {
            *sums.entry(key(at)).or_default() += acc;
        }
        sums.into_iter().collect()
    }
}

/// How times are shown: per presented frame if frames were, else per second.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scale {
    pub(crate) frames: u64,
    pub(crate) seconds: f64,
}

impl Scale {
    fn per(self, us: u64) -> f64 {
        if self.frames > 0 {
            us as f64 / 1000.0 / self.frames as f64
        } else {
            us as f64 / 1000.0 / self.seconds.max(1e-9)
        }
    }

    fn unit(self) -> &'static str {
        if self.frames > 0 { "ms/f" } else { "ms/s" }
    }
}

/// A present seen while sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Mark {
    /// Frames presented since the process started.
    pub(crate) frame: u64,
    /// Since the sampler started.
    pub(crate) ms: f64,
    /// Samples, running samples and alone samples taken until then.
    pub(crate) samples: u64,
    pub(crate) running: u64,
    pub(crate) alone: u64,
}

/// What the report says about the run as a whole.
#[derive(Clone, Debug, Default)]
pub(crate) struct Header {
    pub(crate) file: String,
    /// The sampling window: from, and how long, in seconds since start.
    pub(crate) from_s: f64,
    pub(crate) seconds: f64,
    pub(crate) period_ms: f64,
    pub(crate) rounds: u64,
    /// Rounds that took longer than the period.
    pub(crate) late: u64,
    pub(crate) suspensions: u64,
    /// Looks that failed (the thread ended, or its context was refused).
    pub(crate) failed: u64,
    pub(crate) held_us: u64,
    pub(crate) held_max_us: u64,
    pub(crate) threads_seen: usize,
    pub(crate) threads_watched: usize,
    pub(crate) filter: String,
    pub(crate) depth: usize,
    pub(crate) stack_kb: usize,
    pub(crate) idle_every: u32,
    /// Frames presented in the window.
    pub(crate) frames: u64,
    /// Rounds by the number of sampled threads running in them (the last
    /// place: that many or more).
    pub(crate) running_rounds: Vec<u64>,
    /// Waits recognised (wait functions found in ntdll and win32u).
    pub(crate) waits_known: usize,
    /// mac: wait addresses learned outside every image (a thread stood
    /// still there).
    pub(crate) learned: Vec<u64>,
    /// Looks left out, and the quiet time after an exception that does it.
    pub(crate) skipped: u64,
    pub(crate) quiet_ms: f64,
    /// Exception starts (after a quiet time) in watched threads.
    pub(crate) starts: u64,
    /// Threads expected lost by the looks so far, and the budget a minute
    /// for each thread that starts exceptions.
    pub(crate) risked: f64,
    pub(crate) risk: f64,
    /// Watched threads that ended with exit code 1.
    pub(crate) lost: Vec<u32>,
    pub(crate) done: bool,
}

/// One thread in the report.
pub(crate) struct ThreadView<'a> {
    pub(crate) id: u32,
    pub(crate) name: &'a str,
    pub(crate) start: u64,
    /// Its exit code, if it ended while watched.
    pub(crate) ended: Option<u32>,
    /// Exceptions it took since the process started (by its place), and
    /// starts since it was watched.
    pub(crate) exceptions: u32,
    pub(crate) starts: u32,
    /// Looks at it, and its rate of exception starts as last measured.
    pub(crate) looks: u64,
    pub(crate) rate: f64,
    pub(crate) totals: &'a Totals,
}

fn share(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

/// Rows of `rows` with the most of `pick`, as a table.
#[allow(clippy::too_many_arguments)]
fn table(
    out: &mut String,
    title: &str,
    rows: &[(u64, Acc)],
    pick: fn(&Acc) -> Count,
    whole: u64,
    scale: Scale,
    most: usize,
    name: &dyn Fn(u64) -> String,
) {
    let mut sorted: Vec<&(u64, Acc)> = rows.iter().filter(|r| pick(&r.1).samples > 0).collect();
    if sorted.is_empty() {
        return;
    }
    sorted.sort_unstable_by(|a, b| pick(&b.1).us.cmp(&pick(&a.1).us).then(a.0.cmp(&b.0)));
    let _ = writeln!(out, "{title}");
    let _ = writeln!(
        out,
        "  {:>8} {:>6} {:>8} {:>8}  where",
        scale.unit(),
        "share",
        "alone",
        "samples"
    );
    for (key, acc) in sorted.into_iter().take(most) {
        let c = pick(acc);
        let _ = writeln!(
            out,
            "  {:8.2} {:5.1}% {:8.2} {:8}  {}",
            scale.per(c.us),
            share(c.us, whole),
            scale.per(acc.alone.us),
            c.samples,
            name(*key)
        );
    }
}

fn running(acc: &Acc) -> Count {
    acc.running
}

fn alone(acc: &Acc) -> Count {
    acc.alone
}

/// The whole report.
pub(crate) fn render(
    header: &Header,
    modules: &Modules,
    all: &Totals,
    threads: &[ThreadView],
    marks: &[Mark],
    name: &dyn Fn(u64) -> String,
) -> String {
    let mut out = String::new();
    let h = header;
    let scale = Scale {
        frames: h.frames,
        seconds: h.seconds,
    };
    let _ = writeln!(
        out,
        "VIctory profile (VICTORY_PROFILE={}){}",
        h.file,
        if h.done { ", final" } else { ", so far" }
    );
    let _ = writeln!(
        out,
        "window: {:.1} s from {:.1} s after the sampler started; a round every {} ms: {} rounds ({:.0} a second, {} late)",
        h.seconds,
        h.from_s,
        h.period_ms,
        h.rounds,
        h.rounds as f64 / h.seconds.max(1e-9),
        h.late
    );
    let _ = writeln!(
        out,
        "threads: {} watched of {} seen (VICTORY_PROFILE_THREADS={}); a thread waiting in its last 4 samples is looked at every {} rounds; up to {} frames from {} KiB of stack",
        h.threads_watched,
        h.threads_seen,
        if h.filter.is_empty() {
            "all"
        } else {
            &h.filter
        },
        h.idle_every,
        h.depth,
        h.stack_kb
    );
    if !h.lost.is_empty() {
        let ids: Vec<String> = h.lost.iter().map(u32::to_string).collect();
        let _ = writeln!(
            out,
            "LOST: watched thread {} ended without the normal thread exit, as Wine ends one after a nested \
             exception (a look met an exception under Rosetta); the sampler stopped. The game run after this \
             is not a normal one.",
            ids.join(", ")
        );
    }
    let mean = if h.suspensions > 0 {
        h.held_us as f64 / h.suspensions as f64
    } else {
        0.0
    };
    let _ = writeln!(
        out,
        "suspensions: {} ({} looks failed), held {:.1} us on average, {} us at most; {} looks left out \
         (the thread had taken an exception within {} ms)",
        h.suspensions, h.failed, mean, h.held_max_us, h.skipped, h.quiet_ms
    );
    let _ = writeln!(
        out,
        "exception starts in watched threads: {}; looks spaced to risk {} threads a minute each: {:.3} threads \
         expected lost so far (a look risks the thread's starts a second times {} us, measured under Rosetta, more when looks are held long)",
        h.starts, h.risk, h.risked, EXPOSURE_US
    );
    if h.frames > 0 {
        let _ = writeln!(
            out,
            "frames: {} presented in the window, {:.1} ms a frame",
            h.frames,
            h.seconds * 1000.0 / h.frames as f64
        );
    } else {
        let _ = writeln!(
            out,
            "frames: none presented in the window; times are per second"
        );
    }
    let a = all.acc;
    let _ = writeln!(
        out,
        "samples: {} ({} running, {} alone, {} in I/O, {} not looked at, {:.0}% walked); running {:.1} {unit}, \
         waiting {:.1}, in I/O {:.1}, not looked at {:.1}, alone {:.1}",
        a.all.samples,
        a.running.samples,
        a.alone.samples,
        a.io.samples,
        a.unseen.samples,
        share(all.walked, a.all.samples),
        scale.per(a.running.us),
        scale.per(a.waiting_us()),
        scale.per(a.io.us),
        scale.per(a.unseen.us),
        scale.per(a.alone.us),
        unit = scale.unit()
    );
    if !h.running_rounds.is_empty() {
        let total: u64 = h.running_rounds.iter().sum();
        let last = h.running_rounds.len() - 1;
        let parts: Vec<String> = h
            .running_rounds
            .iter()
            .enumerate()
            .map(|(n, &r)| {
                format!(
                    "{n}{}: {:.1}%",
                    if n == last { "+" } else { "" },
                    share(r, total)
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "rounds by sampled threads running: {}",
            parts.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "running = looked at and not in one of the {} waits known nor in file I/O; alone = the only sampled thread \
         seen running in its round (threads not looked at that round do not count); not looked at = the thread had \
         just taken an exception; {} = thread milliseconds{}; each sample stands for the time since that thread's previous one",
        h.waits_known,
        scale.unit(),
        if h.frames > 0 {
            " per presented frame"
        } else {
            " per second"
        }
    );
    if !h.learned.is_empty() {
        let names: Vec<String> = h.learned.iter().map(|&at| name(at)).collect();
        let _ = writeln!(
            out,
            "waits learned outside every image (a thread stood still there in 4 looks in a row): {}",
            names.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "threads named audio_client_timer count as waiting in the image they started in (one Unix call for their life)"
    );
    let whole = a.running.us;

    // Kinds of code.
    let kind_of = |at: u64| match modules.find(at).map(|m| m.kind) {
        _ if at == SKIPPED => 5,
        Some(Kind::Runtime) => 1,
        Some(Kind::Title) => 2,
        Some(Kind::System) => 3,
        None => 4,
    };
    let kinds = all.by(kind_of);
    table(
        &mut out,
        "\n== kinds of code, running samples",
        &kinds,
        running,
        whole,
        scale,
        5,
        &|k| {
            [
                "",
                "runtime (victory.exe)",
                "title",
                "system DLLs",
                "outside every PE image (Wine's Unix side)",
                "not looked at (an exception)",
            ][k as usize]
                .to_string()
        },
    );

    // Modules.
    let by_module = all.by(|at| {
        if at == SKIPPED {
            SKIPPED
        } else {
            modules.find(at).map_or(0, |m| m.base)
        }
    });
    let module_name = |base: u64| match modules.find(base) {
        _ if base == SKIPPED => name(SKIPPED),
        Some(m) if m.base == base => format!(
            "{} at {:#x} ({}{}{})",
            m.name,
            m.base,
            m.kind.word(),
            if m.note.is_empty() { "" } else { ": " },
            m.note
        ),
        _ => "(outside every image)".to_string(),
    };
    table(
        &mut out,
        "\n== modules, running samples",
        &by_module,
        running,
        whole,
        scale,
        25,
        &module_name,
    );

    let functions: Vec<(u64, Acc)> = all.functions.iter().map(|(&k, &v)| (k, v)).collect();
    let function_name = |key: u64| {
        if key & 1 != 0 {
            format!("{} (64 B; no unwind data)", name(key & !1))
        } else {
            format!("{} (function start)", name(key))
        }
    };
    table(
        &mut out,
        "\n== functions (start by unwind data), running samples",
        &functions,
        running,
        whole,
        scale,
        40,
        &function_name,
    );

    let pages = all.by(page);
    table(
        &mut out,
        "\n== 4 KiB pages, running samples",
        &pages,
        running,
        whole,
        scale,
        30,
        name,
    );

    let buckets: Vec<(u64, Acc)> = all.buckets.iter().map(|(&k, &v)| (k, v)).collect();
    table(
        &mut out,
        "\n== 64-byte buckets, running samples",
        &buckets,
        running,
        whole,
        scale,
        40,
        name,
    );
    table(
        &mut out,
        "\n== 64-byte buckets, alone samples (no other sampled thread running in the round)",
        &buckets,
        alone,
        a.alone.us,
        scale,
        25,
        name,
    );
    table(
        &mut out,
        "\n== functions, alone samples",
        &functions,
        alone,
        a.alone.us,
        scale,
        20,
        &function_name,
    );

    // Callers of the hottest buckets.
    if let Some(chains) = &all.chains {
        let mut hot = buckets.clone();
        hot.retain(|b| b.1.running.samples > 0);
        hot.sort_unstable_by(|a, b| b.1.running.us.cmp(&a.1.running.us).then(a.0.cmp(&b.0)));
        let mut by_bucket: HashMap<u64, Vec<(&[u64; CALLERS + 1], Count)>> = HashMap::new();
        for (key, &count) in chains {
            by_bucket.entry(key[0]).or_default().push((key, count));
        }
        let _ = writeln!(
            out,
            "\n== callers of the hottest 64-byte buckets (running samples; up to {CALLERS} return addresses; {} samples had no room)",
            all.chains_dropped
        );
        for (at, acc) in hot.iter().take(15) {
            let _ = writeln!(
                out,
                "  {:8.2} {} {}",
                scale.per(acc.running.us),
                scale.unit(),
                name(*at)
            );
            let mut list = by_bucket.remove(at).unwrap_or_default();
            list.sort_unstable_by(|a, b| b.1.us.cmp(&a.1.us).then(a.0.cmp(b.0)));
            for (key, count) in list.into_iter().take(5) {
                let callers: Vec<String> = key[1..]
                    .iter()
                    .take_while(|&&f| f != 0)
                    .map(|&f| name(f))
                    .collect();
                let _ = writeln!(
                    out,
                    "      {:5.1}%  <- {}",
                    share(count.us, acc.running.us),
                    if callers.is_empty() {
                        "(no return address)".to_string()
                    } else {
                        callers.join(" <- ")
                    }
                );
            }
        }
    }

    // Threads.
    let mut order: Vec<&ThreadView> = threads
        .iter()
        .filter(|t| t.totals.acc.all.samples > 0)
        .collect();
    order.sort_unstable_by(|a, b| {
        b.totals
            .acc
            .running
            .us
            .cmp(&a.totals.acc.running.us)
            .then(a.id.cmp(&b.id))
    });
    let _ = writeln!(out, "\n== threads, most running first");
    let mut quiet = Vec::new();
    for t in order {
        let acc = t.totals.acc;
        let ended = match t.ended {
            Some(code) => format!(", ended (exit code {code})"),
            None => String::new(),
        };
        if acc.running.samples == 0 && acc.io.samples == 0 && acc.unseen.samples == 0 {
            quiet.push(format!("{} \"{}\"{ended}", t.id, t.name));
            continue;
        }
        let _ = writeln!(
            out,
            "thread {} \"{}\" started at {}{ended}: running {:.2} {unit}, waiting {:.2}, in I/O {:.2}, not looked at {:.2}, \
             alone {:.2}; {} samples, {:.0}% walked; {} looks; {} exceptions, {} starts while watched ({:.1} a second lately)",
            t.id,
            t.name,
            modules.place(t.start),
            scale.per(acc.running.us),
            scale.per(acc.waiting_us()),
            scale.per(acc.io.us),
            scale.per(acc.unseen.us),
            scale.per(acc.alone.us),
            acc.all.samples,
            share(t.totals.walked, acc.all.samples),
            t.looks,
            t.exceptions,
            t.starts,
            t.rate,
            unit = scale.unit()
        );
        let functions: Vec<(u64, Acc)> = t.totals.functions.iter().map(|(&k, &v)| (k, v)).collect();
        table(
            &mut out,
            "  functions",
            &functions,
            running,
            acc.running.us,
            scale,
            8,
            &function_name,
        );
        let buckets: Vec<(u64, Acc)> = t.totals.buckets.iter().map(|(&k, &v)| (k, v)).collect();
        table(
            &mut out,
            "  64-byte buckets",
            &buckets,
            running,
            acc.running.us,
            scale,
            5,
            name,
        );
    }
    if !quiet.is_empty() {
        let _ = writeln!(out, "waiting in every sample: {}", quiet.join(", "));
    }

    // Images: the runtime's, the title's and every one with samples.
    let _ = writeln!(
        out,
        "\n== images (the runtime, the title, and those with samples)"
    );
    for m in modules.iter() {
        let sampled = by_module
            .iter()
            .any(|(b, acc)| *b == m.base && acc.all.samples > 0);
        if sampled || m.kind != Kind::System {
            let _ = writeln!(
                out,
                "  {:#014x} {:#010x} {:<8} {}{}{}",
                m.base,
                m.size,
                m.kind.word(),
                m.name,
                if m.note.is_empty() { "" } else { ": " },
                m.note
            );
        }
    }

    // Frames.
    if !marks.is_empty() {
        let _ = writeln!(
            out,
            "\n== presents seen (frame number, ms since the sampler started, samples, running and alone samples taken until then)"
        );
        for m in marks {
            let _ = writeln!(
                out,
                "  frame {} at {:.1} ms: {} {} {}",
                m.frame, m.ms, m.samples, m.running, m.alone
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(name: &str, base: u64, size: u64, kind: Kind) -> Module {
        Module {
            name: name.into(),
            base,
            size,
            kind,
            note: String::new(),
        }
    }

    fn modules() -> Modules {
        Modules::new(vec![
            module("ntdll.dll", 0x6fff_ffc0_0000, 0x20_0000, Kind::System),
            module("victory.exe", 0x1_4000_0000, 0x300_0000, Kind::Runtime),
            module(
                "victory-image-b99f876b74d4a77f.dll",
                0x6fff_ed55_0000,
                0x800_0000,
                Kind::Title,
            ),
            // Listed twice (a second snapshot), and empty: both left out.
            module("victory.exe", 0x1_4000_0000, 0x300_0000, Kind::Runtime),
            module("empty.dll", 0x10_0000, 0, Kind::System),
        ])
    }

    #[test]
    fn modules_are_found_by_address_with_their_ends_exclusive() {
        let m = modules();
        assert_eq!(m.iter().count(), 3);
        assert_eq!(m.find(0x1_4000_0000).unwrap().name, "victory.exe");
        assert_eq!(m.find(0x1_42ff_ffff).unwrap().name, "victory.exe");
        assert!(m.find(0x1_4300_0000).is_none());
        assert!(m.find(0x1_3fff_ffff).is_none());
        assert!(m.find(0).is_none());
        assert_eq!(m.find(0x6fff_ed55_1234).unwrap().kind, Kind::Title);
        assert_eq!(m.place(0x1_4012_3456), "victory.exe+0x123456");
        assert_eq!(m.place(0x7ff8_1d76_b400), "0x7ff81d76b400");
    }

    #[test]
    fn overlapping_images_keep_the_first() {
        let m = Modules::new(vec![
            module("a.dll", 0x1000, 0x2000, Kind::System),
            module("b.dll", 0x2000, 0x2000, Kind::System),
            module("c.dll", 0x3000, 0x1000, Kind::System),
        ]);
        let names: Vec<_> = m.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["a.dll", "c.dll"]);
    }

    #[test]
    fn buckets_and_pages_round_down() {
        assert_eq!(bucket(0x1_4000_107f), 0x1_4000_1040);
        assert_eq!(bucket(0x1_4000_1040), 0x1_4000_1040);
        assert_eq!(page(0x1_4000_1fff), 0x1_4000_1000);
    }

    fn sample(frames: &[u64], function: u64, us: u64, running: bool, alone: bool) -> Sample<'_> {
        Sample {
            frames,
            function,
            us,
            running,
            alone,
            io: false,
            unseen: false,
        }
    }

    /// mac: samples not looked at and samples in file I/O are counted on
    /// their own: never running, never in the running tables, and not
    /// waiting either.
    #[test]
    fn unseen_and_io_samples_are_neither_running_nor_waiting() {
        let mut t = Totals::with_callers();
        t.add(&sample(&[0x1_4000_1010], 0x1_4000_1000, 1000, true, true));
        t.add(&sample(&[0x1_4000_1010], 0x1_4000_1000, 400, false, false));
        t.add(&Sample {
            io: true,
            ..sample(&[0x6fff_ffc0_ea40], 0, 300, false, false)
        });
        t.add(&Sample {
            unseen: true,
            ..sample(&[SKIPPED], 0, 200, false, false)
        });
        let a = t.acc;
        assert_eq!(a.all, Count { samples: 4, us: 1900 });
        assert_eq!(a.running, Count { samples: 1, us: 1000 });
        assert_eq!(a.alone, Count { samples: 1, us: 1000 });
        assert_eq!(a.io, Count { samples: 1, us: 300 });
        assert_eq!(a.unseen, Count { samples: 1, us: 200 });
        assert_eq!(a.waiting_us(), 400);
        // The mark of a look left out has no running time in any table.
        assert_eq!(t.buckets[&bucket(SKIPPED)].running.samples, 0);
        assert_eq!(t.chains.as_ref().unwrap().values().map(|c| c.samples).sum::<u64>(), 1);
        let header = Header {
            seconds: 1.0,
            frames: 1,
            ..Header::default()
        };
        let threads = [ThreadView {
            id: 7,
            name: "t",
            start: 0,
            ended: None,
            exceptions: 0,
            starts: 0,
            looks: 3,
            rate: 0.0,
            totals: &t,
        }];
        let text = render(&header, &modules(), &t, &threads, &[], &|at| format!("{at:#x}"));
        assert!(text.contains("1 in I/O, 1 not looked at"), "{text}");
        assert!(text.contains("waiting 0.40, in I/O 0.30, not looked at 0.20, alone 1.00"), "{text}");
        assert!(!text.contains("not looked at (an exception)"), "a running table shows the left-out mark:\n{text}");
    }

    #[test]
    fn sums_go_by_bucket_function_and_caller_chain() {
        let mut t = Totals::with_callers();
        t.add(&sample(
            &[0x1_4000_1041, 0x1_4000_9000],
            0x1_4000_1000,
            1000,
            true,
            true,
        ));
        t.add(&sample(
            &[0x1_4000_107f, 0x1_4000_9000],
            0x1_4000_1000,
            2000,
            true,
            false,
        ));
        t.add(&sample(&[0x1_4000_1080], 0, 500, false, false));
        t.add(&sample(&[], 0, 500, true, false));
        assert_eq!(
            t.acc.all,
            Count {
                samples: 3,
                us: 3500
            }
        );
        assert_eq!(
            t.acc.running,
            Count {
                samples: 2,
                us: 3000
            }
        );
        assert_eq!(
            t.acc.alone,
            Count {
                samples: 1,
                us: 1000
            }
        );
        assert_eq!(t.walked, 2);
        assert_eq!(
            t.buckets[&0x1_4000_1040].running,
            Count {
                samples: 2,
                us: 3000
            }
        );
        assert_eq!(
            t.buckets[&0x1_4000_1080].all,
            Count {
                samples: 1,
                us: 500
            }
        );
        assert_eq!(t.functions[&0x1_4000_1000].all.samples, 2);
        // No unwind data: by bucket, marked.
        assert_eq!(t.functions[&0x1_4000_1081].all.samples, 1);
        let chains = t.chains.as_ref().unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(
            chains[&[0x1_4000_1040, 0x1_4000_9000, 0, 0]],
            Count {
                samples: 2,
                us: 3000
            }
        );
        let pages = t.by(page);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].1.all.samples, 3);
    }

    #[test]
    fn the_report_names_modules_buckets_callers_and_frames() {
        let m = modules();
        let mut all = Totals::with_callers();
        let mut thread = Totals::default();
        // A hot title loop, alone; a runtime function; a wait in ntdll.
        for i in 0..50 {
            let frames = [0x6fff_ed55_0000 + 0x221_5c35 + (i % 3), 0x1_4012_3455];
            let s = sample(&frames, 0x6fff_ed55_0000 + 0x221_5c00, 1000, true, true);
            all.add(&s);
            thread.add(&s);
        }
        for _ in 0..20 {
            all.add(&sample(
                &[0x1_4012_3456, 0x1_4000_2000],
                0x1_4012_3400,
                1000,
                true,
                false,
            ));
        }
        for _ in 0..30 {
            all.add(&sample(&[0x6fff_ffc0_1014], 0, 1000, false, false));
        }
        let header = Header {
            file: "Z:\\tmp\\profile.txt".into(),
            seconds: 0.1,
            period_ms: 1.0,
            rounds: 100,
            suspensions: 100,
            held_us: 4000,
            held_max_us: 90,
            frames: 2,
            running_rounds: vec![30, 50, 20],
            waits_known: 12,
            depth: 8,
            stack_kb: 16,
            idle_every: 8,
            ..Header::default()
        };
        let threads = [ThreadView {
            id: 1234,
            name: "",
            start: 0x6fff_ed55_0100,
            ended: None,
            exceptions: 3,
            starts: 1,
            looks: 50,
            rate: 0.5,
            totals: &thread,
        }];
        let marks = [Mark {
            frame: 7,
            ms: 33.5,
            samples: 40,
            running: 30,
            alone: 20,
        }];
        let text = render(&header, &m, &all, &threads, &marks, &|at| m.place(at));
        // 50 ms of the title's loop over two frames.
        assert!(
            text.contains("     25.00  71.4%    25.00       50  victory-image-b99f876b74d4a77f.dll+0x2215c00 (function start)"),
            "{text}"
        );
        assert!(
            text.contains("victory-image-b99f876b74d4a77f.dll+0x2215c00\n"),
            "{text}"
        );
        assert!(text.contains("<- victory.exe+0x123455"), "{text}");
        assert!(text.contains("runtime (victory.exe)"), "{text}");
        assert!(text.contains("== 4 KiB pages"), "{text}");
        assert!(
            text.contains("victory-image-b99f876b74d4a77f.dll+0x2215000\n"),
            "{text}"
        );
        assert!(
            text.contains("thread 1234 \"\" started at victory-image-b99f876b74d4a77f.dll+0x100"),
            "{text}"
        );
        assert!(text.contains("frame 7 at 33.5 ms: 40 30 20"), "{text}");
        assert!(
            text.contains("rounds by sampled threads running: 0: 30.0%, 1: 50.0%, 2+: 20.0%"),
            "{text}"
        );
        assert!(
            text.contains("frames: 2 presented in the window, 50.0 ms a frame"),
            "{text}"
        );
        assert!(
            text.contains("50 looks; 3 exceptions, 1 starts while watched (0.5 a second lately)"),
            "{text}"
        );
        assert!(
            text.contains("exception starts in watched threads: 0;"),
            "{text}"
        );
        assert!(!text.contains("LOST"), "{text}");
        // The waits are not running time.
        assert!(!text.contains("ntdll.dll+0x1014\n"), "{text}");
    }
}
