//! The clamps of many indirect executions in one dispatch
//! (`crate::indirect`, "batches"; from 64bitMango 131721b7): the table the
//! batch's program reads, and which clamps one table can hold. Plain std,
//! so its tests run everywhere; with them, a model of the clamp's program
//! and of the batch's, which the host GPU's results are compared with
//! where there is one.

// Used by `crate::indirect`, which is only built for Windows.
#![cfg_attr(not(windows), allow(dead_code))]

/// Words of a clamp in a batch's table (the program's `ENTRY`).
pub(crate) const ENTRY: usize = 10;
/// Buffers a batch's clamps read their records and counts from.
pub(crate) const SOURCES: usize = 4;

/// One clamp of a batch: what one dispatch of the clamp would be given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
    /// Bit 0: the records run under a count.
    pub(crate) flags: u32,
    /// The source buffer (of [`SOURCES`]) and byte offset of the records,
    /// and of the count.
    pub(crate) from: (u32, u32),
    pub(crate) count: (u32, u32),
    /// The byte offset of the clamped records in the destination.
    pub(crate) to: u32,
}

impl Entry {
    /// Its words in the table.
    pub(crate) fn words(&self) -> [u32; ENTRY] {
        [
            self.kind,
            self.stride,
            self.records,
            self.flags,
            self.from.0,
            self.from.1,
            self.count.0,
            self.count.1,
            self.to,
            0,
        ]
    }
}

/// Where `address` lies in the buffer that starts at `base`, as a batch's
/// program reads it: a byte offset of whole words that fits 32 bits. `None`
/// for any other: that clamp is recorded alone.
pub(crate) fn offset(base: u64, address: u64) -> Option<u32> {
    let offset = u32::try_from(address.checked_sub(base)?).ok()?;
    offset.is_multiple_of(4).then_some(offset)
}

/// The clamps of one batch, in the order they are walked (the order their
/// executions were recorded in), and the start of each buffer they read.
#[derive(Default, Debug)]
pub(crate) struct Table {
    pub(crate) entries: Vec<Entry>,
    pub(crate) sources: Vec<u64>,
}

impl Table {
    /// Adds the clamp of `records` records of `kind`, `stride` bytes apart
    /// at `from` (the start of their buffer and their offset in it), under
    /// the count at `count` if any, into the destination at `to`. False,
    /// with nothing added, if its buffers would be more than a batch
    /// reads: the caller records this batch and starts another.
    pub(crate) fn add(
        &mut self,
        (kind, stride, records): (u32, u32, u32),
        from: (u64, u32),
        count: Option<(u64, u32)>,
        to: u32,
    ) -> bool {
        let mut sources = self.sources.clone();
        for base in [Some(from.0), count.map(|c| c.0)].into_iter().flatten() {
            if !sources.contains(&base) {
                sources.push(base);
            }
        }
        if sources.len() > SOURCES {
            return false;
        }
        self.sources = sources;
        let slot = |base: u64| self.sources.iter().position(|s| *s == base).unwrap_or(0) as u32;
        let from = (slot(from.0), from.1);
        self.entries.push(Entry {
            kind,
            stride,
            records,
            flags: u32::from(count.is_some()),
            from,
            // Not read without a count; a place that can be.
            count: count.map_or(from, |(base, at)| (slot(base), at)),
            to,
        });
        true
    }

    /// The table as the program reads it.
    pub(crate) fn words(&self) -> Vec<u32> {
        self.entries.iter().flat_map(Entry::words).collect()
    }

    /// The buffers bound for the program, the unused ones as the first
    /// (a root view has to be set). `None` for an empty table.
    pub(crate) fn bound(&self) -> Option<[u64; SOURCES]> {
        let mut bound = [*self.sources.first()?; SOURCES];
        bound[..self.sources.len()].copy_from_slice(&self.sources);
        Some(bound)
    }
}

/// A model of the clamp's programs, word for word what
/// `indirect::WORDS_SOURCE` and `indirect::BATCH_SOURCE` do to memory, for
/// tests: buffers are words, offsets are in bytes as the programs take them.
#[cfg(test)]
pub(crate) mod model {
    use super::{ENTRY, SOURCES};

    pub(crate) const BUDGET: u32 = 1 << 27;
    const GROUPS: u32 = 1 << 20;

    /// The programs' `clamped`: the record at byte `at` of `src` clamped
    /// into `dest` at byte `put`; its cost.
    fn clamped(kind: u32, widest: u32, src: &[u32], at: u32, dest: &mut [u32], put: u32) -> u32 {
        let w = |i: u32| src[(at / 4 + i) as usize];
        let put = (put / 4) as usize;
        if kind <= 1 {
            let count = w(0).min(1 << 24);
            let instances = w(1).min((1 << 26) / count.max(1));
            dest[put] = count;
            dest[put + 1] = instances;
            for i in 2..if kind == 1 { 5 } else { 4 } {
                dest[put + i] = w(i as u32);
            }
            count.wrapping_mul(instances)
        } else if kind == 2 {
            let mut g = [if w(0) <= widest { w(0) } else { 65_535 }, w(1).min(65_535), w(2).min(65_535)];
            if (g[1] != 0 && g[0] > GROUPS / g[1]) || (g[2] != 0 && g[0].wrapping_mul(g[1]) > GROUPS / g[2]) {
                g = [0; 3];
            }
            dest[put..put + 3].copy_from_slice(&g);
            g[0].wrapping_mul(g[1]).wrapping_mul(g[2]).wrapping_mul(64)
        } else {
            let g = [w(0).min(65_535), w(1).min(1), w(2).min(1)];
            dest[put..put + 3].copy_from_slice(&g);
            g[0] * g[1] * g[2] * 128
        }
    }

    /// The walk both programs make over one clamp's records, charging
    /// `left`.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        left: &mut u32,
        (kind, stride, walked): (u32, u32, u32),
        widest: u32,
        src: &[u32],
        from: u32,
        dest: &mut [u32],
        to: u32,
    ) {
        for record in 0..walked {
            let (at, put) = (from + record * stride, to + record * stride);
            let cost = clamped(kind, widest, src, at, dest, put);
            if cost > *left {
                // The first record that does not fit takes what is left.
                dest[(put / 4) as usize] = 0;
                *left = 0;
            } else {
                *left -= cost;
            }
        }
    }

    /// One dispatch of the clamp alone: `records` records of `kind`,
    /// `stride` bytes apart at byte `from` of `src`, under `count` if any,
    /// into `dest` at byte `to`, charging the budget word `spent`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alone(
        spent: &mut u32,
        (kind, stride, records): (u32, u32, u32),
        widest: u32,
        src: &[u32],
        from: u32,
        count: Option<u32>,
        dest: &mut [u32],
        to: u32,
    ) {
        let mut left = BUDGET - (*spent).min(BUDGET);
        let walked = count.map_or(records, |c| records.min(c));
        walk(&mut left, (kind, stride, walked), widest, src, from, dest, to);
        *spent = BUDGET - left;
    }

    /// One dispatch of the batch's program over the first `entries` clamps
    /// of `table`, reading `sources`.
    pub(crate) fn batch(
        spent: &mut u32,
        table: &[u32],
        entries: u32,
        widest: u32,
        sources: [&[u32]; SOURCES],
        dest: &mut [u32],
    ) {
        let mut left = BUDGET - (*spent).min(BUDGET);
        for e in table.chunks(ENTRY).take(entries as usize) {
            let (kind, stride, records, flags) = (e[0], e[1], e[2], e[3]);
            let mut walked = records;
            if flags & 1 != 0 {
                walked = records.min(sources[e[6] as usize][(e[7] / 4) as usize]);
            }
            walk(&mut left, (kind, stride, walked), widest, sources[e[4] as usize], e[5], dest, e[8]);
        }
        *spent = BUDGET - left;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::model::{self, BUDGET};
    use super::{SOURCES, Table, offset};

    /// One indirect execution's clamp, for the tests here and those on the
    /// host GPU: `words` are its records (`stride` bytes each), which lie in
    /// buffer `source`; its count, if any, lies in buffer `counter`.
    #[derive(Clone, Debug)]
    pub(crate) struct Spec {
        pub(crate) kind: u32,
        pub(crate) stride: u32,
        pub(crate) records: u32,
        pub(crate) words: Vec<u32>,
        pub(crate) count: Option<u32>,
        pub(crate) source: usize,
        pub(crate) counter: usize,
    }

    /// Buffers of records the specs choose from, and of counts.
    pub(crate) const RECORD_BUFFERS: usize = 3;
    pub(crate) const COUNT_BUFFERS: usize = 2;

    /// Where a spec lies: the byte offset of its records, of its count, and
    /// of its clamped records in the destination.
    pub(crate) type Place = (u32, u32, u32);

    /// The specs laid out in buffers: the buffers' words (records' buffers
    /// first, then counts'), and for each spec the byte offset of its
    /// records, of its count, and of its clamped records in the one
    /// destination, with the destination's words.
    pub(crate) fn laid_out(specs: &[Spec]) -> (Vec<Vec<u32>>, Vec<Place>, usize) {
        let mut buffers = vec![Vec::new(); RECORD_BUFFERS + COUNT_BUFFERS];
        let mut places = Vec::new();
        let mut to = 0;
        for s in specs {
            let from = buffers[s.source].len() as u32 * 4;
            buffers[s.source].extend(&s.words);
            let counter = &mut buffers[RECORD_BUFFERS + s.counter];
            let count_at = counter.len() as u32 * 4;
            counter.push(s.count.unwrap_or(0));
            places.push((from, count_at, to));
            to += (s.stride * s.records.max(1)).next_multiple_of(16);
        }
        for b in &mut buffers {
            if b.is_empty() {
                b.push(0);
            }
        }
        (buffers, places, to as usize / 4)
    }

    /// What the title list's executions give when each is clamped alone,
    /// one after the other in the order given, charging one budget from
    /// `spent` on: the destination's words (unwritten ones `FRESH`) and the
    /// budget word after. This is what the clamps recorded each before its
    /// draw do.
    pub(crate) fn alone(specs: &[Spec], spent: u32, widest: u32) -> (Vec<u32>, u32) {
        let (buffers, places, words) = laid_out(specs);
        let (mut dest, mut spent) = (vec![FRESH; words], spent);
        for (s, &(from, count_at, to)) in specs.iter().zip(&places) {
            let count = s.count.map(|_| buffers[RECORD_BUFFERS + s.counter][count_at as usize / 4]);
            let spec = (s.kind, s.stride, s.records);
            model::alone(&mut spent, spec, widest, &buffers[s.source], from, count, &mut dest, to);
        }
        (dest, spent)
    }

    /// The same executions through batches: each clamp added to a table in
    /// order, the table run and started again where a clamp does not fit
    /// it. Also how many batches ran.
    pub(crate) fn batched(specs: &[Spec], spent: u32, widest: u32) -> (Vec<u32>, u32, usize) {
        let (buffers, places, words) = laid_out(specs);
        let (mut dest, mut spent, mut batches) = (vec![FRESH; words], spent, 0);
        // A buffer's start, as a made-up address.
        let base = |i: usize| 0x1_0000_0000 * (i as u64 + 1);
        let mut run = |table: &Table, dest: &mut Vec<u32>, spent: &mut u32| {
            let Some(bound) = table.bound() else { return };
            let sources = bound.map(|b| &buffers[(b / 0x1_0000_0000 - 1) as usize][..]);
            model::batch(spent, &table.words(), table.entries.len() as u32, widest, sources, dest);
            batches += 1;
        };
        let mut table = Table::default();
        for (s, &(from, count_at, to)) in specs.iter().zip(&places) {
            let spec = (s.kind, s.stride, s.records);
            let from = (base(s.source), from);
            let count = s.count.map(|_| (base(RECORD_BUFFERS + s.counter), count_at));
            if !table.add(spec, from, count, to) {
                run(&table, &mut dest, &mut spent);
                table = Table::default();
                assert!(table.add(spec, from, count, to), "a clamp alone fits a batch");
            }
        }
        run(&table, &mut dest, &mut spent);
        (dest, spent, batches)
    }

    /// What a destination word nothing wrote holds in the model.
    pub(crate) const FRESH: u32 = 0xfeed_f00d;

    /// A draw of `records` records of `count` vertices or indices and
    /// `instances` instances each.
    pub(crate) fn draws(kind: u32, records: u32, count: u32, instances: u32, source: usize) -> Spec {
        let per = if kind == 1 { 5 } else { 4 };
        let mut words = Vec::new();
        for _ in 0..records {
            words.extend([count, instances, 7, 8, 9].iter().take(per));
        }
        Spec { kind, stride: per as u32 * 4, records, words, count: None, source, counter: 0 }
    }

    /// Every order of `items`.
    pub(crate) fn orders<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut all = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let first = rest.remove(i);
            for mut order in orders(&rest) {
                order.insert(0, first.clone());
                all.push(order);
            }
        }
        all
    }

    /// Random specs of every kind, from records clamps cut or empty to
    /// records that run as they are, with and without counts, strides at
    /// and past the record's size, from all the buffers.
    pub(crate) fn random(next: &mut dyn FnMut() -> u64, most: u64) -> Vec<Spec> {
        let value = |next: &mut dyn FnMut() -> u64| -> u32 {
            match next() % 8 {
                0 => 0,
                1 => u32::MAX,
                2 => 1 << 24,
                3 => (1 << 23) + (next() % 1000) as u32,
                4 => 65_535 + (next() % 3) as u32,
                5 => (next() % 70_000) as u32,
                _ => (next() % 300) as u32,
            }
        };
        let n = 1 + (next() % most) as usize;
        (0..n)
            .map(|_| {
                let kind = [0, 1, 2, 10][(next() % 4) as usize];
                let size = match kind {
                    0 => 4,
                    1 => 5,
                    _ => 3,
                };
                let stride = 4 * (size + (next() % 3) as u32);
                let records = 1 + (next() % 40) as u32;
                let words = (0..stride / 4 * records).map(|_| value(&mut *next)).collect();
                let count = next().is_multiple_of(3).then(|| (next() % (u64::from(records) + 4)) as u32);
                Spec {
                    kind,
                    stride,
                    records,
                    words,
                    count,
                    source: (next() % RECORD_BUFFERS as u64) as usize,
                    counter: (next() % COUNT_BUFFERS as u64) as usize,
                }
            })
            .collect()
    }

    /// A small generator of numbers (xorshift) from `seed`.
    pub(crate) fn generator(mut seed: u64) -> impl FnMut() -> u64 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        }
    }

    /// A table gives each buffer its clamps read one slot, in the order
    /// they first appear, and takes no clamp that needs a fifth; an entry's
    /// words are in the order the program reads them.
    #[test]
    fn a_table_holds_clamps_from_four_buffers() {
        let mut table = Table::default();
        assert!(table.bound().is_none());
        assert!(table.add((0, 16, 3), (0x1000, 32), None, 0));
        assert!(table.add((1, 20, 2), (0x2000, 4), Some((0x3000, 8)), 48));
        assert!(table.add((10, 12, 1), (0x1000, 64), Some((0x1000, 96)), 96));
        assert_eq!(table.sources, [0x1000, 0x2000, 0x3000]);
        assert_eq!(table.words()[..10], [0, 16, 3, 0, 0, 32, 0, 32, 0, 0]);
        assert_eq!(table.words()[10..20], [1, 20, 2, 1, 1, 4, 2, 8, 48, 0]);
        assert_eq!(table.words()[20..], [10, 12, 1, 1, 0, 64, 0, 96, 96, 0]);
        // A fourth buffer fits, as records or as a count; a fifth does not,
        // and leaves the table as it was.
        assert!(!table.add((0, 16, 1), (0x4000, 0), Some((0x5000, 0)), 112));
        assert_eq!((table.entries.len(), table.sources.len()), (3, 3));
        assert!(table.add((0, 16, 1), (0x2000, 0), Some((0x4000, 0)), 112));
        assert!(!table.add((0, 16, 1), (0x5000, 0), None, 128));
        assert!(table.add((0, 16, 1), (0x3000, 0), None, 128));
        assert_eq!(table.sources.len(), SOURCES);
        assert_eq!(table.bound(), Some([0x1000, 0x2000, 0x3000, 0x4000]));
        let mut one = Table::default();
        assert!(one.add((2, 12, 1), (0x9000, 0), None, 0));
        assert_eq!(one.bound(), Some([0x9000; 4]));
    }

    /// Offsets a batch's program can read: whole words, within 32 bits.
    #[test]
    fn a_batch_reads_at_word_offsets_within_32_bits() {
        assert_eq!(offset(0x1000, 0x1000), Some(0));
        assert_eq!(offset(0x1000, 0x1ffc), Some(0xffc));
        assert_eq!(offset(0x1000, 0x1002), None);
        assert_eq!(offset(0x1000, 0xfff), None);
        assert_eq!(offset(0, 0xffff_fffc), Some(0xffff_fffc));
        assert_eq!(offset(0, 0x1_0000_0000), None);
    }

    /// The budget order, in every order: six executions that between them
    /// take more than a budget (large draws, a garbage draw, small draws,
    /// a dispatch, mesh dispatches under a count), from more buffers than
    /// one batch reads (three of records, two of counts). In each of their
    /// 720 orders the batches cut the
    /// same records, write the same words and leave the same budget as the
    /// clamps recorded one before each draw; and the order is what decides
    /// which records are cut.
    #[test]
    fn batches_cut_the_same_records_in_every_order() {
        let large = |source| draws(0, 3, 3 << 18, 40, source);
        let executions = [
            large(0),
            draws(1, 2, u32::MAX, u32::MAX, 1),
            Spec { count: Some(8), ..draws(0, 8, 300, 2, 2) },
            Spec { kind: 2, stride: 12, records: 2, words: vec![64, 64, 4, 1 << 21, 1, 1], count: None, source: 0, counter: 0 },
            Spec { kind: 10, stride: 12, records: 6, words: [70_000, 1, 1].repeat(6), count: Some(4), source: 2, counter: 1 },
            large(1),
        ];
        let mut outcomes = std::collections::HashSet::new();
        let mut split = 0;
        for order in orders(&executions) {
            for spent in [0, BUDGET / 2, BUDGET] {
                let each = alone(&order, spent, 1 << 20);
                let (words, left, batches) = batched(&order, spent, 1 << 20);
                assert_eq!((&words, left), (&each.0, each.1));
                split += usize::from(batches > 1);
                if spent == 0 {
                    assert_eq!(left, BUDGET, "they take more than a budget");
                    // Which executions have a record that ran empty.
                    let (_, places, _) = laid_out(&order);
                    let cut: Vec<u32> = order
                        .iter()
                        .zip(&places)
                        .filter(|(s, p)| (0..s.records).any(|r| words[((p.2 + r * s.stride) / 4) as usize] == 0))
                        .map(|(s, _)| s.words[0])
                        .collect();
                    outcomes.insert(cut);
                }
            }
        }
        assert!(outcomes.len() > 10, "the order decides what is cut: {}", outcomes.len());
        assert!(split > 0, "some orders need a second batch");
    }

    /// The same over random runs of clamps of every kind.
    #[test]
    fn a_batch_clamps_as_each_alone_does() {
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        let mut spent_all = 0;
        for trial in 0..2000 {
            let specs = random(&mut next, 24);
            let spent = [0, 0, BUDGET - 5000, BUDGET][trial % 4];
            let widest = [65_535, 1 << 20][trial % 2];
            let each = alone(&specs, spent, widest);
            let (words, left, _) = batched(&specs, spent, widest);
            assert_eq!((&words, left), (&each.0, each.1), "trial {trial}");
            spent_all += usize::from(left == BUDGET && spent == 0);
        }
        assert!(spent_all > 0, "some runs spend the whole budget");
    }

    /// A walk in another order is seen by these comparisons: two clamps of
    /// two large draws each, of which a budget runs two.
    #[test]
    fn a_batch_in_another_order_is_seen() {
        let large = |source| draws(0, 2, 1 << 24, 4, source);
        let specs = [large(0), large(1)];
        let each = alone(&specs, 0, 65_535);
        assert_eq!(each.0[..8], [1 << 24, 4, 7, 8, 1 << 24, 4, 7, 8], "the first clamp's draws run");
        assert_eq!((each.0[8], each.0[12]), (0, 0), "the second's run empty");
        // The second execution's records walked first.
        let (buffers, places, words) = laid_out(&specs);
        let mut table = Table::default();
        for i in [1, 0] {
            assert!(table.add((0, 16, 2), (i as u64 + 1, places[i].0), None, places[i].2));
        }
        let (mut dest, mut spent) = (vec![FRESH; words], 0);
        let sources = table.bound().unwrap().map(|b| &buffers[b as usize - 1][..]);
        model::batch(&mut spent, &table.words(), 2, 65_535, sources, &mut dest);
        assert_ne!(dest, each.0);
        assert_eq!(spent, each.1, "the same budget spent either way");
    }

    /// The clamp's bounds, in the model as in the program (the host GPU
    /// tests of `crate::indirect` have the same numbers).
    #[test]
    fn the_model_clamps_as_the_program_does() {
        let one = |kind: u32, words: &[u32], widest: u32| {
            let spec = Spec { kind, stride: words.len() as u32 * 4, records: 1, words: words.to_vec(), count: None, source: 0, counter: 0 };
            alone(&[spec], 0, widest)
        };
        assert_eq!(one(0, &[u32::MAX; 4], 65_535), (vec![1 << 24, 4, u32::MAX, u32::MAX], 1 << 26));
        assert_eq!(one(1, &[6, 2, 1, 2, 3], 65_535), (vec![6, 2, 1, 2, 3, FRESH, FRESH, FRESH], 12));
        assert_eq!(one(2, &[70_000, 1, 1], 65_535).0[..3], [65_535, 1, 1]);
        assert_eq!(one(2, &[70_000, 1, 1], 1 << 20).0[..3], [70_000, 1, 1]);
        assert_eq!(one(2, &[65_535, 17, 1], 65_535).0[..3], [0, 0, 0]);
        assert_eq!(one(2, &[1 << 20, 4_096, 1], 1 << 20).0[..3], [0, 0, 0]);
        assert_eq!(one(2, &[8, 8, 0], 65_535), (vec![8, 8, 0, FRESH], 0));
        assert_eq!(one(10, &[70_000, 5, 9], 65_535), (vec![65_535, 1, 1, FRESH], 65_535 * 128));
        // A garbage draw takes what is left of the budget; under a count
        // the records past it are not read.
        let garbage = draws(0, 128, u32::MAX, u32::MAX, 0);
        let all = alone(std::slice::from_ref(&garbage), 0, 65_535);
        assert_eq!(all.0.iter().step_by(4).filter(|&&c| c != 0).count(), 2);
        let under = alone(&[Spec { count: Some(3), ..garbage }], 0, 65_535);
        assert_eq!(under.0[..12].iter().step_by(4).copied().collect::<Vec<_>>(), [1 << 24, 1 << 24, 0]);
        assert!(under.0[12..].iter().all(|&w| w == FRESH));
    }
}
