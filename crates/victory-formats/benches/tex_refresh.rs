//! Detiling a sampled texture's refresh under Rosetta, where no memory is
//! tracked (VIctory-macOS issue #1): the refresh must still read and compare
//! the whole memory (the compare is the change detector), but does it then
//! detile every subresource, as it does today, or only the subresources the
//! changed bytes touch? Streamed textures change mip by mip, so each
//! submission changes one 64 KiB page, cycling across the subresources.
//! Run with `cargo bench -p victory-formats --bench tex_refresh`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

/// One page a submission changes.
const PAGE: usize = 64 * 1024;
/// Two rounds over a 13-mip texture's subresources; the median is over all.
const SUBMISSIONS: usize = 26;

/// The bytes of the texture's memory subresource `i` lies in, as the GPU's
/// staging computes them from the layout (rounded out to its blocks).
fn span(layout: &Layout, i: u32, mips: u32) -> (usize, usize) {
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    let (mip, slice) = (i % mips, i / mips);
    let m = &layout.mips[mip as usize];
    let (z0, z1) = if d.dimension == Dimension::Tex3d {
        (0, m.depth.max(1) - 1)
    } else {
        (slice, slice)
    };
    let first = layout.offset(mip, 0, 0, z0).unwrap();
    let last = layout
        .offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1)
        .unwrap()
        + u64::from(d.element_bytes);
    let (from, to) = (first / block * block, last.div_ceil(block) * block);
    (from as usize, to.min(layout.size) as usize)
}

/// Detiles `subs` of `tiled` into `linear` rows, as a refresh does.
fn detile(layout: &Layout, linear: &mut [Vec<u8>], tiled: &[u8], subs: &[u32], mips: u32) {
    for &i in subs {
        let (mip, slice) = (i % mips, i / mips);
        let m = &layout.mips[mip as usize];
        let pitch = m.pitch as usize * layout.desc.element_bytes as usize;
        layout
            .detile(
                tiled,
                mip,
                slice,
                &mut linear[i as usize],
                pitch,
                pitch * m.rows as usize,
            )
            .unwrap();
    }
}

/// Writes one changed page into `tiled`, returning where it lies.
fn write_page(tiled: &mut [u8], at: usize) -> (usize, usize) {
    let len = PAGE.min(tiled.len() - at);
    for b in &mut tiled[at..at + len] {
        *b = b.wrapping_add(1);
    }
    (at, len)
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let cases = [
        (
            "bc7",
            Desc {
                dimension: Dimension::Tex2d,
                width: 4096,
                height: 4096,
                depth: 1,
                mips: 13,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
        (
            "rgba8",
            Desc {
                dimension: Dimension::Tex2d,
                width: 4096,
                height: 4096,
                depth: 1,
                mips: 13,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
    ];
    for (name, desc) in cases {
        let layout = Layout::new(&desc).unwrap();
        let size = layout.size as usize;
        let mips = layout.mips.len() as u32;
        let count = mips * layout.slices;
        let spans: Vec<(usize, usize)> = (0..count).map(|i| span(&layout, i, mips)).collect();
        let mut linear: Vec<Vec<u8>> = (0..count)
            .map(|i| {
                let m = &layout.mips[(i % mips) as usize];
                let pitch = m.pitch as usize * desc.element_bytes as usize;
                vec![0u8; pitch * m.rows as usize]
            })
            .collect();
        let mut tiled = vec![0u8; size];
        for (i, b) in tiled.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let every: Vec<u32> = (0..count).collect();
        let mut checksum = 0u64;
        // One 64 KiB page changes per submission, cycling across the
        // subresources (streamed textures change mip by mip).
        let pages: Vec<usize> = (0..SUBMISSIONS)
            .map(|i| spans[i % count as usize].0.min(size - 1))
            .collect();
        // Today's untracked behavior: a differing byte anywhere re-detiles
        // every subresource.
        let mut full = Vec::new();
        for &at in &pages {
            let (_, len) = write_page(&mut tiled, at);
            let start = Instant::now();
            detile(&layout, &mut linear, black_box(&tiled), &every, mips);
            full.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        // The fix: only the subresources the changed page touches.
        let mut partial = Vec::new();
        for &at in &pages {
            let (at, len) = write_page(&mut tiled, at);
            let fresh: Vec<u32> = (0..count)
                .filter(|&i| at < spans[i as usize].1 && spans[i as usize].0 < at + len)
                .collect();
            let start = Instant::now();
            detile(&layout, &mut linear, black_box(&tiled), &fresh, mips);
            partial.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        black_box(checksum);
        println!(
            "bench_tex_refresh_us_full_{name} {:.3}",
            median(full).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_partial_{name} {:.3}",
            median(partial).as_secs_f64() * 1e6
        );
    }
}
