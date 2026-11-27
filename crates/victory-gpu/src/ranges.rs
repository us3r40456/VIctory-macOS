//! Sets of byte ranges, for the memory a submission has brought up to date
//! and what it still has to (`crate::host`'s batch of refreshes).

#[cfg(any(feature = "diagnostics", test))]
use std::cmp::Reverse;
#[cfg(any(feature = "diagnostics", test))]
use std::collections::BinaryHeap;
#[cfg(any(feature = "diagnostics", test))]
use std::fmt;

/// Byte ranges `[from, to)`, in order and apart: ranges that touch or
/// overlap are one. A few per host copy, so a vector is the cheapest set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ranges(Vec<(usize, usize)>);

impl Ranges {
    /// The ranges, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0.iter().copied()
    }

    /// Adds `[from, to)`, joined with the ranges that overlap it or lie
    /// within `gap` bytes of it (the bytes between are then in the set).
    pub(crate) fn insert(&mut self, from: usize, to: usize, gap: usize) {
        if from >= to {
            return;
        }
        // The ranges to join: those ending within `gap` of `from` on, and
        // starting within `gap` of `to` before.
        let first = self
            .0
            .partition_point(|&(_, e)| e.saturating_add(gap) < from);
        let last = first + self.0[first..].partition_point(|&(s, _)| s <= to.saturating_add(gap));
        if first == last {
            self.0.insert(first, (from, to));
            return;
        }
        self.0[first] = (self.0[first].0.min(from), self.0[last - 1].1.max(to));
        self.0.drain(first + 1..last);
    }

    /// Takes `[from, to)` out (a range around it falls in two).
    pub(crate) fn remove(&mut self, from: usize, to: usize) {
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

    /// Whether any byte of `[from, to)` is in the set.
    pub(crate) fn overlaps(&self, from: usize, to: usize) -> bool {
        from < to
            && self
                .0
                .get(self.0.partition_point(|&(_, e)| e <= from))
                .is_some_and(|&(s, _)| s < to)
    }

    /// The parts of `[from, to)` not in the set, in order.
    pub(crate) fn missing(
        &self,
        from: usize,
        to: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let first = self.0.partition_point(|&(_, end)| end <= from);
        let mut at = from;
        self.0[first..]
            .iter()
            .copied()
            .take_while(move |&(start, _)| start < to)
            // The requested end closes the last uncovered interval.
            .chain(std::iter::once((to, to)))
            .filter_map(move |(start, end)| {
                let gap = (at, start);
                at = at.max(end);
                (gap.0 < gap.1).then_some(gap)
            })
    }
}

/// Where a queue's piece (host lists in the order they run, each with the
/// buffer ranges its commands read, the ranges its title copies write, and
/// whether it starts at a title copy that changes what the list before it
/// reads) is to be split into host submissions, so that no title copy
/// changes what a list earlier in the same submission reads
/// (`VICTORY_COPY_ORDER=split`; `crate::host::copy_order_parts`): the
/// indices of the lists that start a new submission, in order.
///
/// A submission's copies are all done before its first GPU command; a
/// copy and the commands of an earlier list keep the title's order only
/// in different submissions. A copy that changes what its own list read
/// before it is not seen here: the list is cut there when it is recorded
/// (`crate::list`), and the second part says so (`after_copy`).
pub(crate) fn copy_order_splits<'a>(
    lists: impl IntoIterator<Item = (&'a Ranges, &'a [(usize, usize)], bool)>,
) -> Vec<usize> {
    let mut splits = Vec::new();
    // The reads of the lists of the submission being gathered.
    let mut reads: Vec<&Ranges> = Vec::new();
    for (i, (read, copied, after_copy)) in lists.into_iter().enumerate() {
        let changes = |reads: &[&Ranges]| {
            copied
                .iter()
                .any(|&(at, len)| reads.iter().any(|r| r.overlaps(at, at.saturating_add(len))))
        };
        if !reads.is_empty() && (after_copy || changes(&reads)) {
            splits.push(i);
            reads.clear();
        }
        reads.push(read);
    }
    splits
}

/// How many of a submission's largest refresh ranges are logged.
#[cfg(any(feature = "diagnostics", test))]
const REFRESH_TOP: usize = 8;

/// A bounded summary of the ranges a submission refreshed into host shadows.
/// The ranges are attributed only to their host copy: a batch combines
/// requests from lists, view heaps and page tables, and does not retain a
/// reliable descriptor or view owner for each range.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Default)]
pub(crate) struct RefreshSummary {
    bytes: u64,
    ranges: u64,
    largest: BinaryHeap<Reverse<RefreshedRange>>,
}

/// One refreshed range, ordered largest first and then by title address.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RefreshedRange {
    len: u64,
    title: u64,
    host_gpu: u64,
}

#[cfg(any(feature = "diagnostics", test))]
impl Ord for RefreshedRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len
            .cmp(&other.len)
            .then_with(|| other.title.cmp(&self.title))
            .then_with(|| other.host_gpu.cmp(&self.host_gpu))
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl PartialOrd for RefreshedRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl RefreshSummary {
    /// Counts a range of `len` title bytes and where it lands in host memory.
    pub(crate) fn record(&mut self, title: u64, len: u64, host_gpu: u64) {
        self.bytes = self.bytes.saturating_add(len);
        self.ranges = self.ranges.saturating_add(1);
        if len == 0 {
            return;
        }
        let range = RefreshedRange {
            len,
            title,
            host_gpu,
        };
        if self.largest.len() < REFRESH_TOP {
            self.largest.push(Reverse(range));
        } else if self.largest.peek().is_some_and(|worst| range > worst.0) {
            self.largest.pop();
            self.largest.push(Reverse(range));
        }
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn is_empty(&self) -> bool {
        self.ranges == 0
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl fmt::Display for RefreshSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ranges, {} bytes", self.ranges, self.bytes)?;
        let mut largest: Vec<_> = self.largest.iter().copied().collect();
        largest.sort_unstable();
        for Reverse(range) in largest {
            write!(
                f,
                "; title {:#x}+{:#x} host_gpu {:#x}",
                range.title, range.len, range.host_gpu
            )?;
        }
        Ok(())
    }
}

/// The small buffer ranges (constant buffers and other views of at most
/// [`SMALL_READ`] bytes) that the commands of a submission read through
/// their tables, in the order its steps name them, to tell when a title
/// copy later in the same submission writes one (`crate::host`,
/// `Batch::rewritten`; diagnostics).
#[cfg(any(feature = "diagnostics", test))]
#[derive(Default)]
pub(crate) struct SmallReads(Vec<(usize, usize)>);

/// The longest range [`SmallReads`] keeps.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) const SMALL_READ: usize = 256;

#[cfg(any(feature = "diagnostics", test))]
impl SmallReads {
    /// A command reads `[at, at + len)`.
    pub(crate) fn read(&mut self, at: usize, len: usize) {
        let range = (at, at.saturating_add(len));
        // Kept short: a world submission names tens of thousands, most of
        // them the ranges just named.
        if len == 0 || len > SMALL_READ || self.0.len() >= 16384 || self.0.iter().rev().take(64).any(|r| *r == range) {
            return;
        }
        self.0.push(range);
    }

    /// What a copy into `[at, at + len)` writes of the ranges read so far:
    /// each part as (from, to), and how many of the ranges hold it.
    pub(crate) fn written(&self, at: usize, len: usize) -> Vec<(usize, usize, usize)> {
        let end = at.saturating_add(len);
        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        for &(from, to) in &self.0 {
            let (a, b) = (from.max(at), to.min(end));
            if a >= b {
                continue;
            }
            match out.iter_mut().find(|o| (o.0, o.1) == (a, b)) {
                Some(o) => o.2 += 1,
                None => out.push((a, b, 1)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{REFRESH_TOP, Ranges, RefreshSummary, SMALL_READ, SmallReads, copy_order_splits};

    #[test]
    fn overlaps_is_true_for_any_shared_byte() {
        let mut r = Ranges::default();
        assert!(!r.overlaps(0, usize::MAX));
        r.insert(0x100, 0x120, 0);
        r.insert(0x400, 0x500, 0);
        for (from, to, want) in [
            (0x100, 0x120, true),
            (0x0, 0x101, true),
            (0x11f, 0x200, true),
            (0x0, 0x100, false),
            (0x120, 0x400, false),
            (0x110, 0x110, false),
            (0x4ff, 0x500, true),
            (0x500, 0x600, false),
            (0x0, 0x1000, true),
        ] {
            assert_eq!(r.overlaps(from, to), want, "{from:#x}..{to:#x}");
        }
    }

    /// A piece is split before the list whose copy changes what an earlier
    /// list of the same part reads, and before a list cut at such a copy;
    /// the lists after a split are looked at against the new part only.
    #[test]
    fn a_piece_is_split_where_a_copy_changes_what_an_earlier_list_reads() {
        let reads = |r: &[(usize, usize)]| {
            let mut set = Ranges::default();
            for &(from, to) in r {
                set.insert(from, to, 0);
            }
            set
        };
        let (x, other, none) = (reads(&[(0x1fe0, 0x2000)]), reads(&[(0x9000, 0x9040)]), reads(&[]));
        let splits = |lists: &[(&Ranges, &[(usize, usize)], bool)]| copy_order_splits(lists.iter().copied());
        // Nothing to split: no copies, a copy elsewhere, a copy before any
        // read, a copy into what only a later list reads.
        assert!(splits(&[]).is_empty());
        assert!(splits(&[(&x, &[], false), (&x, &[], false)]).is_empty());
        assert!(splits(&[(&x, &[], false), (&other, &[(0x3000, 32)], false)]).is_empty());
        assert!(splits(&[(&none, &[(0x1fe0, 32)], false), (&x, &[], false)]).is_empty());
        // A list's own reads do not split it from itself (the cut does).
        assert!(splits(&[(&x, &[(0x1fe0, 32)], false)]).is_empty());
        // The title's TAA case across two lists, a byte's overlap, and a
        // reader two lists back.
        assert_eq!(splits(&[(&x, &[], false), (&x, &[(0x1fe0, 32)], false)]), [1]);
        assert_eq!(splits(&[(&x, &[], false), (&none, &[(0x1fff, 0x1000)], false)]), [1]);
        assert_eq!(splits(&[(&x, &[], false), (&other, &[], false), (&none, &[(0x1ff0, 4)], false)]), [2]);
        // A list cut at a copy starts a submission unless it is the first
        // of the piece (the part before it ran in an earlier piece).
        assert!(splits(&[(&none, &[(0x1fe0, 32)], true)]).is_empty());
        assert_eq!(splits(&[(&x, &[], false), (&x, &[(0x1fe0, 32)], true)]), [1]);
        assert_eq!(splits(&[(&none, &[], false), (&none, &[], true)]), [1]);
        // After a split only the new part's reads count: the third list's
        // copy changes what the second read, the fourth's what the first
        // read, which ran two submissions before.
        assert_eq!(
            splits(&[
                (&x, &[], false),
                (&other, &[(0x1fe0, 32)], false),
                (&none, &[(0x9000, 8)], false),
                (&none, &[(0x1fe0, 32)], false),
            ]),
            [1, 2]
        );
    }

    /// A copy writes the small ranges read before it where they overlap,
    /// counted per range; long ranges and ranges read after it are not its.
    #[test]
    fn a_copy_writes_the_small_ranges_read_before_it() {
        let mut reads = SmallReads::default();
        assert!(reads.written(0x1000, 0x100).is_empty());
        reads.read(0x1fe0, 32);
        reads.read(0x1fe0, 32);
        reads.read(0x1cd0, 48);
        reads.read(0x1fe0, 32);
        reads.read(0x4000, SMALL_READ + 1);
        reads.read(0x5000, 0);
        // The same range named three times is kept once.
        assert_eq!(reads.written(0x1fe0, 32), vec![(0x1fe0, 0x2000, 1)]);
        assert_eq!(reads.written(0x1ff0, 0x100), vec![(0x1ff0, 0x2000, 1)]);
        assert_eq!(
            reads.written(0x1000, 0x1000),
            vec![(0x1fe0, 0x2000, 1), (0x1cd0, 0x1d00, 1)]
        );
        assert!(reads.written(0x2000, 0x1000).is_empty());
        assert!(reads.written(0x4000, 0x200).is_empty());
        // Two views of different lengths at one address are two ranges.
        reads.read(0x1fe0, 16);
        assert_eq!(
            reads.written(0x1fe0, 16),
            vec![(0x1fe0, 0x1ff0, 2)]
        );
    }

    /// Synthetic ranges arrive out of order: totals cover all of them, while
    /// the formatted largest are bounded and ties have a stable order.
    #[test]
    fn refresh_summary_orders_caps_and_formats_ranges() {
        let ranges = [
            (0x1000, 0x20, 0x2000),
            (0x5000, 0x100, 0x9000),
            (0x3000, 0x40, 0x5000),
            (0x8000, 0xc0, 0xd000),
            (0x2000, 0x100, 0x4000),
            (0x6000, 0x10, 0xa000),
            (0x7000, 0x200, 0xb000),
            (0x9000, 0x60, 0xe000),
            (0x4000, 0x80, 0x6000),
            (0xa000, 0x8, 0xf000),
        ];
        let mut summary = RefreshSummary::default();
        for &(title, len, host_gpu) in &ranges {
            summary.record(title, len, host_gpu);
        }
        assert_eq!(summary.ranges, ranges.len() as u64);
        assert_eq!(summary.bytes, ranges.iter().map(|r| r.1).sum());
        assert_eq!(
            summary.to_string(),
            concat!(
                "10 ranges, 1560 bytes",
                "; title 0x7000+0x200 host_gpu 0xb000",
                "; title 0x2000+0x100 host_gpu 0x4000",
                "; title 0x5000+0x100 host_gpu 0x9000",
                "; title 0x8000+0xc0 host_gpu 0xd000",
                "; title 0x4000+0x80 host_gpu 0x6000",
                "; title 0x9000+0x60 host_gpu 0xe000",
                "; title 0x3000+0x40 host_gpu 0x5000",
                "; title 0x1000+0x20 host_gpu 0x2000",
            ),
            "only the top {REFRESH_TOP} are logged, largest first"
        );
        assert!(!summary.to_string().contains("0x6000+0x10"));
        assert!(!summary.to_string().contains("0xa000+0x8"));
    }

    fn set(ranges: &[(usize, usize)]) -> Ranges {
        let mut r = Ranges::default();
        for &(a, b) in ranges {
            r.insert(a, b, 0);
        }
        r
    }

    fn all(r: &Ranges) -> Vec<(usize, usize)> {
        r.iter().collect()
    }

    #[test]
    fn ranges_that_touch_or_overlap_are_one() {
        let mut r = set(&[(10, 20), (30, 40)]);
        assert_eq!(all(&r), [(10, 20), (30, 40)]);
        r.insert(20, 30, 0);
        assert_eq!(all(&r), [(10, 40)], "touching both");
        r.insert(5, 12, 0);
        r.insert(38, 50, 0);
        assert_eq!(all(&r), [(5, 50)], "overlapping");
        r.insert(60, 60, 0);
        assert_eq!(all(&r), [(5, 50)], "nothing added");
    }

    #[test]
    fn a_gap_joins_ranges_close_together() {
        let mut r = Ranges::default();
        r.insert(0, 10, 5);
        r.insert(15, 20, 5);
        assert_eq!(
            all(&r),
            [(0, 20)],
            "5 bytes apart, joined with the bytes between"
        );
        r.insert(26, 30, 5);
        assert_eq!(all(&r), [(0, 20), (26, 30)], "6 bytes apart");
        r.insert(22, 24, 5);
        assert_eq!(all(&r), [(0, 30)], "within 5 of both");
        r.insert(usize::MAX - 10, usize::MAX, usize::MAX);
        assert_eq!(all(&r), [(0, usize::MAX)], "no overflow");
    }

    #[test]
    fn removing_cuts_ranges_and_splits_one_it_is_inside() {
        let mut r = set(&[(0, 100)]);
        r.remove(40, 60);
        assert_eq!(all(&r), [(0, 40), (60, 100)], "split");
        r.remove(0, 10);
        r.remove(90, 200);
        assert_eq!(all(&r), [(10, 40), (60, 90)], "ends cut");
        r.remove(30, 70);
        assert_eq!(
            all(&r),
            [(10, 30), (70, 90)],
            "a range across two, and what is between"
        );
        r.remove(0, 1000);
        assert_eq!(all(&r), [], "all of it");
        let mut r = set(&[(0, 10)]);
        r.remove(5, 5);
        r.remove(7, 3);
        assert_eq!(all(&r), [(0, 10)], "nothing taken out");
    }

    #[test]
    fn missing_lists_the_parts_not_in_the_set() {
        let r = set(&[(10, 20), (30, 40)]);
        let missing = |from, to| r.missing(from, to).collect::<Vec<_>>();
        assert_eq!(missing(0, 50), [(0, 10), (20, 30), (40, 50)]);
        assert_eq!(missing(12, 18), []);
        assert_eq!(missing(15, 35), [(20, 30)]);
        assert_eq!(missing(5, 12), [(5, 10)]);
        assert_eq!(missing(38, 45), [(40, 45)]);
        assert_eq!(missing(50, 60), [(50, 60)]);
        assert_eq!(missing(25, 25), []);
        assert_eq!(
            Ranges::default().missing(3, 9).collect::<Vec<_>>(),
            [(3, 9)]
        );
    }

    /// Every operation against a bitmap, on a long run of random ones.
    #[test]
    fn ranges_match_a_bitmap() {
        const N: usize = 200;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut r = Ranges::default();
        let mut bits = [false; N];
        for step in 0..20_000 {
            let (a, b) = (next(N + 1), next(N + 1));
            let (from, to) = (a.min(b), a.max(b));
            match next(4) {
                0 | 1 => {
                    r.insert(from, to, 0);
                    bits[from..to].fill(true);
                }
                2 => {
                    r.remove(from, to);
                    bits[from..to].fill(false);
                }
                _ => {
                    let mut gaps = [false; N];
                    for (s, e) in r.missing(from, to) {
                        assert!(
                            from <= s && s < e && e <= to,
                            "step {step}: a gap inside the question"
                        );
                        gaps[s..e].fill(true);
                    }
                    for i in 0..N {
                        assert_eq!(
                            gaps[i],
                            (from..to).contains(&i) && !bits[i],
                            "step {step}: byte {i}"
                        );
                    }
                }
            }
            let mut rebuilt = [false; N];
            let mut last_end = None;
            for (s, e) in r.iter() {
                assert!(
                    s < e && last_end.is_none_or(|l| l < s),
                    "step {step}: in order, apart, not empty"
                );
                rebuilt[s..e].fill(true);
                last_end = Some(e);
            }
            assert_eq!(rebuilt, bits, "step {step}");
        }
    }
}
