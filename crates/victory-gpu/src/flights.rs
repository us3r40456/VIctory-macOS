//! mac: the plain arithmetic of the flights (`crate::host::unwaited::flights`,
//! `VICTORY_FLIGHTS=1`): plain std, so its tests run everywhere.

/// Title memory as ranges `[from, to)`: after [`Spans::join`], in order,
/// none touching the next.
#[derive(Debug, Default, Clone)]
pub(crate) struct Spans(Vec<(usize, usize)>);

impl Spans {
    /// Adds `[from, to)` (nothing if empty); [`Spans::join`] after the last.
    pub(crate) fn add(&mut self, from: usize, to: usize) {
        if from < to {
            self.0.push((from, to));
        }
    }

    /// Puts the ranges in order, those that overlap or touch as one.
    pub(crate) fn join(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(self.0.len());
        for &(from, to) in &self.0 {
            match out.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => out.push((from, to)),
            }
        }
        self.0 = out;
    }

    /// Whether `[from, from + len)` shares a byte with one of them (joined).
    pub(crate) fn touches(&self, from: usize, len: usize) -> bool {
        let to = from.saturating_add(len);
        if from >= to {
            return false;
        }
        let at = self.0.partition_point(|r| r.1 <= from);
        self.0.get(at).is_some_and(|r| r.0 < to)
    }

    #[cfg(test)]
    fn ranges(&self) -> &[(usize, usize)] {
        &self.0
    }
}

/// What the `bytes` (1 to 8) at `address` will hold once `writes` are made
/// in their order (each: where, the value it stores if that is known now,
/// how many of its low bytes it stores, 4 or 8, little-endian):
///
/// - `None`: no write reaches those bytes, memory decides;
/// - `Some(Some(value))`: the last write that reaches them stores all of
///   them, and its value is known;
/// - `Some(None)`: a write reaches them that cannot be told now (a time
///   stamp, or one that stores only a part of them and leaves the rest to
///   memory or to an earlier write).
pub(crate) fn after_writes(writes: &[(usize, Option<u64>, usize)], address: usize, bytes: usize) -> Option<Option<u64>> {
    let end = address.checked_add(bytes)?;
    if bytes == 0 || bytes > 8 {
        return None;
    }
    let mut seen = None;
    for &(at, value, n) in writes {
        let Some(stop) = at.checked_add(n) else { continue };
        if stop <= address || end <= at {
            continue;
        }
        seen = Some(match value {
            Some(v) if at <= address && end <= stop => {
                // The low `n` bytes of `v`, then the ones asked for.
                let stored = if n >= 8 { v } else { v & ((1u64 << (8 * n)) - 1) };
                let shifted = stored >> (8 * (address - at));
                Some(if bytes >= 8 { shifted } else { shifted & ((1u64 << (8 * bytes)) - 1) })
            }
            _ => None,
        });
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_join_where_they_touch_and_find_what_they_share() {
        let mut s = Spans::default();
        s.add(100, 200);
        s.add(300, 400);
        s.add(200, 250);
        s.add(500, 500);
        s.add(350, 360);
        s.join();
        assert_eq!(s.ranges(), &[(100, 250), (300, 400)]);
        assert!(s.touches(249, 1));
        assert!(!s.touches(250, 50));
        assert!(s.touches(250, 51));
        assert!(!s.touches(0, 100));
        assert!(s.touches(0, 101));
        assert!(!s.touches(400, 1000));
        assert!(!s.touches(120, 0));
        assert!(!Spans::default().touches(0, usize::MAX));
    }

    #[test]
    fn a_wait_reads_what_the_last_write_over_it_stores() {
        // Nothing there: memory decides.
        assert_eq!(after_writes(&[(0x100, Some(7), 8)], 0x200, 4), None);
        // Whole: an 8-byte value read as 8 and as its two halves.
        let w = [(0x100, Some(0x1111_2222_3333_4444), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(0x1111_2222_3333_4444)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(0x3333_4444)));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(0x1111_2222)));
        // A 4-byte write stores the low half of its value only.
        let w = [(0x100, Some(0xdead_beef_0000_0005), 4)];
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(5)));
        // Read wider than it stores: the rest is memory's, not known.
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        // A time stamp: not known.
        assert_eq!(after_writes(&[(0x100, None, 8)], 0x100, 8), Some(None));
        // In order: the later write decides, known or not.
        let w = [(0x100, Some(1), 8), (0x100, Some(2), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(2)));
        let w = [(0x100, Some(1), 8), (0x100, None, 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        let w = [(0x100, None, 8), (0x100, Some(3), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(3)));
        // A later write over a part of it: not known.
        let w = [(0x100, Some(1), 8), (0x104, Some(9), 4)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(9)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(1)));
        // Nothing asked, or more than a value holds.
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 0), None);
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 9), None);
    }
}
