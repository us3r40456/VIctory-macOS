//! mac: how much of the refreshed title memory lies in runs of consecutive
//! pages of one physical section (`VICTORY_PHYSICAL_RUNS=1`, a count only,
//! off by default): what aliasing the sections' pages (a `VICTORY_ALIAS_PHYSICAL`
//! not built yet, `notes/candidates-3.md`, "Physical-page mappings") could
//! take off the copies. `victory_hle`'s `xmem` tells each view of physical
//! pages it maps and unmaps; a refreshed host copy is counted by whether its
//! title buffer lies in one run of consecutive pages of one section (one view,
//! or views side by side that continue each other), in physical pages that are
//! not one run, or elsewhere.
//!
//! Plain std, so its tests run everywhere.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The 64 KiB page physical pages come in.
const PAGE: usize = 64 << 10;

/// Whether the count is on (`VICTORY_PHYSICAL_RUNS=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_RUNS"))
}

/// Views of physical pages by title address: their length, section and the
/// index of their first page in it.
static VIEWS: Mutex<BTreeMap<usize, (usize, usize, usize)>> = Mutex::new(BTreeMap::new());

/// `xmem` mapped `len` bytes of `section` from page `index` at `at`.
pub fn mapped(at: usize, len: usize, section: usize, index: usize) {
    if on() {
        VIEWS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(at, (len, section, index));
    }
}

/// `xmem` unmapped the view at `at`.
pub fn unmapped(at: usize) {
    if on() {
        VIEWS.lock().unwrap_or_else(|e| e.into_inner()).remove(&at);
    }
}

/// Where `[from, to)` lies: 0 in one run of consecutive pages of one section,
/// 1 in physical pages that are not one run, 2 elsewhere.
fn class(views: &BTreeMap<usize, (usize, usize, usize)>, from: usize, to: usize) -> usize {
    let Some((&base, &(len, section, index))) = views
        .range(..=from)
        .next_back()
        .filter(|(b, v)| **b + v.0 > from)
    else {
        return 2;
    };
    let (mut end, mut next) = (base + len, index + len / PAGE);
    while end < to {
        match views.get(&end) {
            Some(&(len, s, i)) if s == section && i == next => {
                (end, next) = (end + len, next + len / PAGE)
            }
            _ => return 1,
        }
    }
    0
}

/// Per class: copies refreshed, bytes copied, microseconds.
static COUNTS: [[AtomicU64; 3]; 3] = [const { [const { AtomicU64::new(0) }; 3] }; 3];

/// A refresh copied `bytes` into the host copy of `[start, start + len)` in
/// `us` microseconds.
pub(crate) fn refreshed(start: usize, len: usize, bytes: usize, us: u64) {
    let k = class(
        &VIEWS.lock().unwrap_or_else(|e| e.into_inner()),
        start,
        start + len,
    );
    for (i, n) in [1, bytes as u64, us].into_iter().enumerate() {
        COUNTS[k][i].fetch_add(n, Relaxed);
    }
}

/// The line logged with the frame line, per frame over `frames` frames.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let [run, scattered, other] = COUNTS.each_ref().map(|c| {
        c.each_ref()
            .map(|n| n.swap(0, Relaxed) as f64 / frames.max(1.0))
    });
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "physical runs (VICTORY_PHYSICAL_RUNS), per frame: refreshes of copies in one run of one section's pages \
         {:.0} copies, {:.1} MB in {:.1} ms; in physical pages not one run {:.0} copies, {:.1} MB in {:.1} ms; \
         elsewhere {:.0} copies, {:.1} MB in {:.1} ms; {views} views mapped",
        run[0],
        run[1] / 1e6,
        run[2] / 1e3,
        scattered[0],
        scattered[1] / 1e6,
        scattered[2] / 1e3,
        other[0],
        other[1] / 1e6,
        other[2] / 1e3,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_is_counted_by_where_its_pages_are() {
        let a = 0x1_0000_0000;
        let mut views = BTreeMap::new();
        // Section 7, pages 10 and 11 at `a`, page 12 at `a + 2 pages`: one run.
        views.insert(a, (2 * PAGE, 7, 10));
        views.insert(a + 2 * PAGE, (PAGE, 7, 12));
        // Then page 40 of it, and a page of section 8.
        views.insert(a + 3 * PAGE, (PAGE, 7, 40));
        views.insert(a + 4 * PAGE, (PAGE, 8, 41));
        assert_eq!(
            class(&views, a + 100, a + 3 * PAGE),
            0,
            "three pages in order"
        );
        assert_eq!(class(&views, a + PAGE, a + PAGE + 8), 0, "inside one view");
        assert_eq!(
            class(&views, a, a + 4 * PAGE),
            1,
            "the run breaks at page 40"
        );
        assert_eq!(class(&views, a + 3 * PAGE, a + 5 * PAGE), 1, "two sections");
        assert_eq!(
            class(&views, a + 4 * PAGE, a + 6 * PAGE),
            1,
            "runs off the views"
        );
        assert_eq!(class(&views, a - PAGE, a + PAGE), 2, "starts outside");
        assert_eq!(class(&views, a + 9 * PAGE, a + 10 * PAGE), 2);
    }
}
