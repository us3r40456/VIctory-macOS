//! Time a page-table rebuild alone; the synthetic copies need no title
//! files or GPU. Two behaviors: the buffers a rebuild is built in made
//! fresh each time (the table's rewrite today), and kept between
//! rebuilds (what the fix keeps).

use std::hint::black_box;
use std::time::Instant;

use victory_formats::pagetable::{CAPACITY, Outputs, Scratch};

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

/// A submission that changes the copy set: 4 copies released and made.
fn churn(lcg: &mut Lcg, copies: &mut [(u64, u64)], regions: u64) {
    for _ in 0..4 {
        let k = lcg.next() as usize % copies.len();
        let len = ((lcg.next() % 960) + 64) << 10;
        let start = ((lcg.next() % regions) << 28) + ((lcg.next() % 4096) << 16);
        copies[k] = (start, len);
    }
}

fn main() {
    const BATCH: u32 = 32;
    // The world's list: ~4,100 copies in a few dozen regions, its table
    // rewritten per submission that changes the copy set; and past the
    // regions a table has room for: one copy per region, the overflow
    // path.
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let stress = |lcg: &mut Lcg| {
        (0..600)
            .map(|r| {
                let len = ((lcg.next() % 64) + 64) << 10;
                ((r << 28) + ((lcg.next() % 4000) << 16), len)
            })
            .collect::<Vec<_>>()
    };
    for (label, mut world, regions) in [
        ("rebuild", copies(4_100, 40, &mut lcg), 40u64),
        ("capacity", stress(&mut lcg), 600u64),
    ] {
        let pairs: Vec<u32> = (0..world.len() as u32).map(|k| (1 << 18) | (2 * k)).collect();
        let mut warm = Scratch::default();
        let mut out = Outputs::default();
        let built = warm.build(&world, &pairs, 0, &mut out);
        let (bytes, blocks) = (built.dwords.len() * 4, built.blocks.len());
        assert!(blocks <= CAPACITY + 1, "the table fits its buffer");
        println!("bench_table_{label}_bytes {bytes} blocks {blocks}");
        let mut run = |fresh: bool| -> Vec<f64> {
            // One build before timing: the buffers exist either way.
            let mut kept = (Scratch::default(), Outputs::default());
            black_box(kept.0.build(&world, &pairs, 0, &mut kept.1));
            let mut samples = Vec::with_capacity(9);
            for _ in 0..9 {
                let start = Instant::now();
                for _ in 0..BATCH {
                    churn(&mut lcg, &mut world, regions);
                    if fresh {
                        // Made again each rebuild, as the rewrite's are.
                        kept = (Scratch::default(), Outputs::default());
                    }
                    let built = kept.0.build(&world, &pairs, 0, &mut kept.1);
                    black_box((built.dwords.len(), built.blocks.len(), built.full));
                }
                samples.push(start.elapsed().as_micros() as f64 / f64::from(BATCH));
            }
            samples
        };
        for (name, fresh) in [("alloc", true), ("reuse", false)] {
            let mut samples = run(fresh);
            samples.sort_by(|a, b| a.total_cmp(b));
            let all = samples
                .iter()
                .map(|s| format!("{s:.1}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "bench_table_{label}_us_{name} {:.1} us (all: {all})",
                samples[samples.len() / 2]
            );
        }
    }
}
