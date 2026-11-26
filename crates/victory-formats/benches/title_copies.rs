//! Time what a title copy walks of the copies a submission flushed:
//! every copy, as `host`'s `title_copied` walks them today, and only
//! the copies its bytes can be in, as the extent index walks them. The
//! copies need no title files or GPU: what the walk costs is where they
//! sit in memory and what each takes out.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use victory_formats::extents::Extents;

/// A world's copies, ~9,000 over 40 regions, every list asking for each.
const COPIES: usize = 9_000;
/// The regions copies sit in: 40 of 256 MiB, below 2^40.
const REGIONS: u64 = 40;
/// The title copies into buffers a submission runs between its lists.
const TITLE_COPIES: usize = 80;
/// Submissions a sample times.
const BATCH: u32 = 8;

/// A deterministic generator, so every run builds the same copies.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Copies clustered as title memory, in `regions` distinct 256 MiB
/// regions below 2^40: 8 streaming buffers (64–256 MiB), 200 heaps
/// (1–16 MiB), 1,000 per-draw scratch areas (256 KiB–1 MiB), the rest
/// ring buffers (64–128 KiB), until `count`.
fn copies(count: usize, regions: u64, lcg: &mut Lcg) -> Vec<(u64, u64)> {
    let mut copies = Vec::with_capacity(count);
    let place = |lcg: &mut Lcg, len: u64| {
        let region = lcg.next() % regions;
        let pages = len.div_ceil(1 << 16);
        let start = (region << 28) + ((lcg.next() % (4096 - pages.min(4096))) << 16);
        (start, len)
    };
    for _ in 0..8 {
        let len = ((lcg.next() % 192) + 64) << 20;
        copies.push(place(lcg, len));
    }
    for _ in 0..200 {
        let len = ((lcg.next() % 15) + 1) << 20;
        copies.push(place(lcg, len));
    }
    for _ in 0..1_000 {
        let len = ((lcg.next() % 768) + 256) << 10;
        copies.push(place(lcg, len));
    }
    while copies.len() < count {
        let len = ((lcg.next() % 64) + 64) << 10;
        copies.push(place(lcg, len));
    }
    copies
}

/// The small copies a submission's title steps write into title memory:
/// few to 64 KiB, wherever the title's buffers sit in it.
fn windows(lcg: &mut Lcg) -> Vec<(usize, usize)> {
    (0..TITLE_COPIES)
        .map(|_| {
            let at = (lcg.next() % (REGIONS << 28)) as usize;
            let len = (((lcg.next() % 61) + 4) << 10) as usize;
            (at, at + len)
        })
        .collect()
}

/// victory-gpu's `Ranges` (private there): byte ranges `[from, to)`, in
/// order and apart, the same shape per entry. What a title copy does to
/// each is `Ranges::remove`, so it is here with the same searches.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Ranges(Vec<(usize, usize)>);

impl Ranges {
    /// `[from, to)` alone, as a flush leaves a copy asked for whole.
    fn whole(from: usize, to: usize) -> Self {
        Self(vec![(from, to)])
    }

    /// Takes `[from, to)` out (a range around it falls in two): the two
    /// searches and their nothing-found return are what a no-op costs.
    fn remove(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let first = self.0.partition_point(|&(_, e)| e <= from);
        let last = first + self.0[first..].partition_point(|&(s, _)| s < to);
        if first == last {
            return;
        }
        let (s, e) = (self.0[first].0, self.0[last - 1].1);
        let left = (s < from).then_some((s, from));
        let right = (to < e).then_some((to, e));
        self.0.splice(first..last, left.into_iter().chain(right));
    }
}

/// One submission's title copies, one way or the other: 9 samples of
/// BATCH submissions, µs each.
fn run(
    scan: bool,
    world: &[(u64, u64)],
    current: &mut HashMap<usize, (usize, Ranges)>,
    lcg: &mut Lcg,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(9);
    let mut extents = Extents::default();
    for _ in 0..9 {
        let mut took = 0u128;
        for _ in 0..BATCH {
            // As a submission's flush leaves it, before the title steps:
            // every copy current whole (the flush's own copies are not
            // what is timed here, either way).
            for (key, (_, have)) in current.iter_mut() {
                let &(start, len) = &world[*key];
                *have = Ranges::whole(start as usize, (start + len) as usize);
            }
            let windows = windows(lcg);
            let started = Instant::now();
            if scan {
                // Today: every copy, whatever the bytes can be in.
                for &(from, to) in &windows {
                    for (_, have) in current.values_mut() {
                        have.remove(from, to);
                    }
                }
            } else {
                // The fix's way: each copy's extent filed as its flush
                // files it (charged to this way), a title copy then
                // takes out of only the copies that can hold its bytes.
                extents.clear();
                for (key, &(start, len)) in world.iter().enumerate() {
                    extents.flushed(key, start as usize, (start + len) as usize);
                }
                for &(from, to) in &windows {
                    for key in extents.overlapping(from, to) {
                        if let Some((_, have)) = current.get_mut(&key) {
                            have.remove(from, to);
                        }
                    }
                }
            }
            took += started.elapsed().as_micros();
        }
        samples.push(took as f64 / f64::from(BATCH));
    }
    samples
}

fn main() {
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let world = copies(COPIES, REGIONS, &mut lcg);
    println!(
        "bench_title_copies_shape {} copies {} regions",
        world.len(),
        REGIONS
    );

    // As a world submission leaves `host`'s `current` after its lists
    // asked for every copy whole: each copy's memory in one range.
    let mut current: HashMap<usize, (usize, Ranges)> = HashMap::with_capacity(COPIES);
    for (key, &(start, len)) in world.iter().enumerate() {
        current.insert(key, (key, Ranges::whole(start as usize, (start + len) as usize)));
    }

    // One submission both ways on the same state, untimed: what they
    // take out is the same, and the visits are what the index saves.
    let mut scan_map = current.clone();
    let mut index_map = current.clone();
    let mut extents = Extents::default();
    for (key, &(start, len)) in world.iter().enumerate() {
        extents.flushed(key, start as usize, (start + len) as usize);
    }
    let checked = windows(&mut lcg);
    let mut visits = 0usize;
    for &(from, to) in &checked {
        for (_, have) in scan_map.values_mut() {
            have.remove(from, to);
        }
        for key in extents.overlapping(from, to) {
            visits += 1;
            if let Some((_, have)) = index_map.get_mut(&key) {
                have.remove(from, to);
            }
        }
    }
    assert_eq!(scan_map, index_map, "the same taken out both ways");
    println!("bench_title_copied_visits_scan {}", TITLE_COPIES * COPIES);
    println!("bench_title_copied_visits_index {visits}");

    for (name, scan) in [("scan", true), ("index", false)] {
        let mut samples = run(scan, &world, &mut current, &mut lcg);
        samples.sort_by(|a, b| a.total_cmp(b));
        let median = samples[samples.len() / 2];
        let all = samples
            .iter()
            .map(|s| format!("{s:.0}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("bench_title_copied_{name}_us {median:.0} us (all: {all})");
        black_box(&samples);
    }
}
