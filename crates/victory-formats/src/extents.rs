//! Where the copies a submission flushed sit in memory: a title copy
//! that takes bytes out of what each copy holds (victory-gpu's `host`)
//! reaches by its bytes only the copies that can hold them, not every
//! copy flushed.

/// The memory extents of copies, each named by the key its holder is
/// filed under: sorted by start, with the largest end up to each entry
/// beside it, a window `[from, to)` walks only the entries that can
/// overlap it and skips the rest without a look.
#[derive(Default)]
pub struct Extents {
    /// `(start, end, key)`, in start order once [`Extents::overlapping`]
    /// has built it.
    entries: Vec<(usize, usize, usize)>,
    /// The largest end in `entries[..=i]`, aligned with it: past the
    /// first entry that reaches a window's `from`, nothing before it
    /// reaches either.
    ends: Vec<usize>,
    /// Whether `entries` is in start order and `ends` built.
    built: bool,
}

impl Extents {
    /// A copy that flushed, its memory `[start, end)` filed under `key`.
    pub fn flushed(&mut self, key: usize, start: usize, end: usize) {
        self.entries.push((start, end, key));
        self.built = false;
    }

    /// Nothing is held any more: as the holders are cleared.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.ends.clear();
        self.built = true;
    }

    /// The keys whose memory can overlap `[from, to)`, in no particular
    /// order: the entries that start before `to`, walked back while any
    /// of them can reach `from` (the largest end so far says whether)
    /// and kept where the entry itself reaches it. The rest hold none of
    /// the window's bytes, whatever is taken out of the ones named.
    pub fn overlapping(&mut self, from: usize, to: usize) -> impl Iterator<Item = usize> + '_ {
        if !self.built {
            self.entries.sort_unstable();
            self.ends.clear();
            let mut max = 0;
            for &(_, end, _) in &self.entries {
                max = max.max(end);
                self.ends.push(max);
            }
            self.built = true;
        }
        let Extents { entries, ends, .. } = &*self;
        let last = entries.partition_point(|&(start, ..)| start < to);
        let empty = from >= to;
        (0..last)
            .rev()
            .take_while(move |&i| !empty && ends[i] > from)
            .filter(move |&i| entries[i].1 > from)
            .map(move |i| entries[i].2)
    }
}

#[cfg(test)]
mod tests {
    use super::Extents;

    /// The keys a window reaches, in pushed order.
    fn keys(extents: &mut Extents, from: usize, to: usize) -> Vec<usize> {
        let mut out: Vec<_> = extents.overlapping(from, to).collect();
        out.sort_unstable();
        out
    }

    /// Only the copies whose memory a window's bytes can be in are
    /// reached: two copies sharing memory are both named for a title
    /// copy into it, the copies before and after are left alone.
    #[test]
    fn a_window_reaches_only_the_copies_its_bytes_can_be_in() {
        let mut extents = Extents::default();
        extents.flushed(1, 0x1_0000, 0x2_0000);
        // Shares 0x1_8000..0x2_0000 with 1.
        extents.flushed(2, 0x1_8000, 0x3_0000);
        extents.flushed(3, 0x5_0000, 0x6_0000);
        extents.flushed(4, 0, 0x100);
        assert_eq!(keys(&mut extents, 0x1_9000, 0x1_a000), vec![1, 2]);
        assert_eq!(keys(&mut extents, 0x2_8000, 0x2_9000), vec![2]);
        assert_eq!(keys(&mut extents, 0x4_0000, 0x4_1000), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0x5_8000, 0x6_0000), vec![3]);
        assert_eq!(keys(&mut extents, 0x6_0000, 0x7_0000), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0, 0x100), vec![4]);
        assert_eq!(keys(&mut extents, 0x40, 0x40), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0x2_0000, 0), Vec::<usize>::new());
    }

    /// Pushed after a lookup and cleared, the extents answer from what
    /// they hold then (`host` flushes between title copies and clears
    /// with its holders).
    #[test]
    fn pushed_after_a_lookup_and_cleared() {
        let mut extents = Extents::default();
        extents.flushed(7, 0x100, 0x200);
        assert_eq!(keys(&mut extents, 0x180, 0x1c0), vec![7]);
        extents.clear();
        assert_eq!(keys(&mut extents, 0x180, 0x1c0), Vec::<usize>::new());
        extents.flushed(8, 0x300, 0x400);
        extents.flushed(9, 0x180, 0x280);
        assert_eq!(keys(&mut extents, 0x190, 0x1a0), vec![9]);
        assert_eq!(keys(&mut extents, 0x380, 0x3c0), vec![8]);
    }

    /// Every window of a pile of copies reaches the same keys a walk of
    /// all of them would (what `host`'s title copies keep is decided
    /// entry by entry, so this is the fix's equivalence, in the small).
    #[test]
    fn every_window_reaches_what_a_walk_of_all_would() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut extents = Extents::default();
        let mut all = Vec::new();
        for key in 0..500usize {
            // Small extents in a small span, so they crowd and overlap.
            let start = (next() % 0x100_0000) as usize;
            let end = start + (next() % 0x10_0000) as usize + 1;
            all.push((start, end, key));
            extents.flushed(key, start, end);
        }
        for _ in 0..500 {
            let from = (next() % 0x110_0000) as usize;
            let to = from + (next() % 0x8_0000) as usize + 1;
            let mut reached: Vec<_> = extents.overlapping(from, to).collect();
            reached.sort_unstable();
            let mut walked: Vec<_> = all
                .iter()
                .filter(|&&(start, end, _)| start < to && end > from)
                .map(|&(_, _, key)| key)
                .collect();
            walked.sort_unstable();
            assert_eq!(reached, walked, "the window {from:#x}..{to:#x}");
        }
    }
}
