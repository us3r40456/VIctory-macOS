//! mac: the maps behind a view heap's census (`crate::bindless::Viewed`: what
//! each title slot views, and how many slots view each range of a host
//! copy), in two forms chosen when a heap is made
//! (`crate::mac_switches::quick_sync`, `VICTORY_QUICK_SYNC`).
//!
//! The census is entered for every slot a heap sync makes again: the old
//! entry out, the new one in, 25,000 to 40,000 times a world frame on the
//! submitting thread, in maps of 640,000 slots and 150,000 ranges. As they
//! were (a hash map with the standard hasher for the slots, a B-tree for the
//! ranges) that is some fifteen lookups in memory the cache does not hold;
//! with hash maps and a multiply-and-fold hasher it is five. Both forms hold
//! the same entries and answer every question the same way: nothing reads
//! the ranges in order but a diagnostics count, which sorts them itself.
//!
//! Here, outside the Windows-only census, so that the tests run on any host.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash, Hasher};

/// How the census's hash maps hash: as `HashMap` does by default, or by
/// multiply-and-fold (`quick`).
#[derive(Clone)]
pub(crate) enum Hashing {
    Standard(std::hash::RandomState),
    Fold,
}

impl Hashing {
    pub(crate) fn new(quick: bool) -> Self {
        if quick {
            Hashing::Fold
        } else {
            Hashing::Standard(std::hash::RandomState::new())
        }
    }
}

impl BuildHasher for Hashing {
    type Hasher = Hashed;

    fn build_hasher(&self) -> Hashed {
        match self {
            Hashing::Standard(state) => Hashed::Standard(state.build_hasher()),
            Hashing::Fold => Hashed::Fold(0),
        }
    }
}

/// [`Hashing`]'s hasher.
pub(crate) enum Hashed {
    Standard(std::hash::DefaultHasher),
    /// The state: each word is mixed in by a 128-bit multiply whose halves
    /// are folded together. The keys are slot indices and (copy, start, end,
    /// kind) tuples of ours, not text anyone chooses, so the standard
    /// hasher's protection against chosen keys buys nothing here.
    Fold(u64),
}

impl Hashed {
    fn fold(state: &mut u64, word: u64) {
        let m = u128::from(*state ^ word).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
        *state = m as u64 ^ (m >> 64) as u64;
    }
}

impl Hasher for Hashed {
    fn finish(&self) -> u64 {
        match self {
            Hashed::Standard(h) => h.finish(),
            Hashed::Fold(state) => *state,
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        match self {
            // Every integer reaches the standard hasher as its bytes, as it
            // does in a map that uses that hasher directly.
            Hashed::Standard(h) => h.write(bytes),
            Hashed::Fold(state) => {
                for chunk in bytes.chunks(8) {
                    let mut word = [0u8; 8];
                    word[..chunk.len()].copy_from_slice(chunk);
                    Self::fold(state, u64::from_le_bytes(word));
                }
                // So that a key and the same key with zero bytes after it differ.
                Self::fold(state, bytes.len() as u64);
            }
        }
    }

    fn write_u8(&mut self, n: u8) {
        match self {
            Hashed::Standard(h) => h.write_u8(n),
            Hashed::Fold(state) => Self::fold(state, u64::from(n)),
        }
    }

    fn write_u32(&mut self, n: u32) {
        match self {
            Hashed::Standard(h) => h.write_u32(n),
            Hashed::Fold(state) => Self::fold(state, u64::from(n)),
        }
    }

    fn write_u64(&mut self, n: u64) {
        match self {
            Hashed::Standard(h) => h.write_u64(n),
            Hashed::Fold(state) => Self::fold(state, n),
        }
    }

    fn write_usize(&mut self, n: usize) {
        match self {
            Hashed::Standard(h) => h.write_usize(n),
            Hashed::Fold(state) => Self::fold(state, n as u64),
        }
    }
}

/// One value per key (the census's count of slots per range): a B-tree as
/// it was, or a hash map (`quick`).
pub(crate) enum Counts<K, V> {
    Tree(BTreeMap<K, V>),
    Hash(HashMap<K, V, Hashing>),
}

impl<K: Ord + Hash + Copy, V> Counts<K, V> {
    pub(crate) fn new(quick: bool) -> Self {
        if quick {
            Counts::Hash(HashMap::with_hasher(Hashing::Fold))
        } else {
            Counts::Tree(BTreeMap::new())
        }
    }

    /// The value of `key`, entered as `new()` if it has none.
    pub(crate) fn entry_or(&mut self, key: K, new: impl FnOnce() -> V) -> &mut V {
        match self {
            Counts::Tree(map) => map.entry(key).or_insert_with(new),
            Counts::Hash(map) => map.entry(key).or_insert_with(new),
        }
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            Counts::Tree(map) => map.remove(key),
            Counts::Hash(map) => map.remove(key),
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn len(&self) -> usize {
        match self {
            Counts::Tree(map) => map.len(),
            Counts::Hash(map) => map.len(),
        }
    }

    /// Every key with its value, in no order that both forms share (the
    /// ranges unordered views write, for the widened views of VIctory-macOS
    /// #222, `crate::bindless::widened_writes`).
    pub(crate) fn iter(&self) -> Box<dyn Iterator<Item = (&K, &V)> + '_> {
        match self {
            Counts::Tree(map) => Box::new(map.iter()),
            Counts::Hash(map) => Box::new(map.iter()),
        }
    }

    /// The keys, in no order that both forms share.
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn keys(&self) -> Box<dyn Iterator<Item = &K> + '_> {
        match self {
            Counts::Tree(map) => Box::new(map.keys()),
            Counts::Hash(map) => Box::new(map.keys()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range key as the census has them: copy, start, end, unordered.
    type Range = (usize, u64, u64, bool);

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn range(&mut self, copies: u64, starts: u64) -> Range {
            let copy = 0x7000_0000_0000 + (self.next() % copies) as usize * 0x140;
            let start = 0x1_0000_0000 + (self.next() % starts) * 16;
            (copy, start, start + 16 + self.next() % 4096, self.next() % 8 == 0)
        }
    }

    /// The census's counting (`bindless::Inner::count`, without the hulls)
    /// over `counts`: one more or one fewer slot views `key`; a range no slot
    /// views has no entry.
    fn count(counts: &mut Counts<Range, u32>, key: Range, add: bool) {
        let n = counts.entry_or(key, || 0);
        *n = if add { *n + 1 } else { n.saturating_sub(1) };
        if *n == 0 {
            counts.remove(&key);
        }
    }

    /// Slots rewritten at random hold the same ranges, counted the same, in
    /// both forms.
    #[test]
    fn both_forms_hold_the_same_entries() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut forms = [false, true].map(|quick| {
            (HashMap::<u32, Range, Hashing>::with_hasher(Hashing::new(quick)), Counts::<Range, u32>::new(quick))
        });
        assert!(matches!(forms[0].1, Counts::Tree(_)) && matches!(forms[1].1, Counts::Hash(_)));
        for step in 0..200_000 {
            let slot = (rng.next() % 5_000) as u32;
            // Few ranges, so that counts rise, fall and reach zero.
            let new = (step % 7 != 0).then(|| rng.range(40, 60));
            for (slots, counts) in &mut forms {
                if let Some(old) = slots.remove(&slot) {
                    count(counts, old, false);
                }
                if let Some(new) = new {
                    count(counts, new, true);
                    slots.insert(slot, new);
                }
            }
        }
        let sorted = |counts: &Counts<Range, u32>| {
            let mut keys: Vec<Range> = counts.keys().copied().collect();
            keys.sort_unstable();
            keys
        };
        let held = |slots: &HashMap<u32, Range, Hashing>| {
            let mut held: Vec<(u32, Range)> = slots.iter().map(|(s, r)| (*s, *r)).collect();
            held.sort_unstable();
            held
        };
        let [(tree_slots, tree), (hash_slots, hash)] = &mut forms;
        assert_eq!(tree.len(), hash.len());
        assert!(tree.len() > 1_000, "the test's ranges are shared and distinct both");
        let keys = sorted(tree);
        assert_eq!(keys, sorted(hash));
        let slots = held(tree_slots);
        assert_eq!(slots, held(hash_slots));
        // And each count is the number of slots that hold the range.
        let mut per_range = std::collections::BTreeMap::new();
        for (_, range) in &slots {
            *per_range.entry(*range).or_insert(0u32) += 1;
        }
        assert_eq!(per_range.len(), keys.len());
        for key in keys {
            assert_eq!(*tree.entry_or(key, || 0), per_range[&key]);
            assert_eq!(*hash.entry_or(key, || 0), per_range[&key]);
        }
    }

    /// Off, the maps hash exactly as a map with the standard hasher does.
    #[test]
    fn the_standard_form_hashes_as_the_standard_hasher() {
        let state = std::hash::RandomState::new();
        let ours = Hashing::Standard(state.clone());
        for slot in [0u32, 1, 127, 128, 639_999, u32::MAX] {
            assert_eq!(ours.hash_one(slot), state.hash_one(slot));
        }
        let range: Range = (0x7000_1234_5678, 0x1_0000_0040, 0x1_0000_0140, true);
        assert_eq!(ours.hash_one(range), state.hash_one(range));
        assert_eq!(ours.hash_one("text"), state.hash_one("text"));
    }

    /// The fold spreads the keys the census has: slot indices in a row, and
    /// ranges that differ in one field only, over the low bits a hash map
    /// takes its bucket from and the high bits it tells entries apart by.
    #[test]
    fn the_fold_spreads_slot_indices_and_ranges() {
        let fold = Hashing::Fold;
        let spread = |hashes: &[u64]| {
            let low: std::collections::HashSet<u64> = hashes.iter().map(|h| h & 0xFFFF).collect();
            let high: std::collections::HashSet<u64> = hashes.iter().map(|h| h >> 57).collect();
            (low.len(), high.len())
        };
        // 65,536 keys into 65,536 buckets: a random spread fills about 63 %.
        let slots: Vec<u64> = (0..1u32 << 16).map(|s| fold.hash_one(s)).collect();
        let (low, high) = spread(&slots);
        assert!(low > 38_000 && high == 128, "slot indices: {low} buckets, {high} tags");
        let copy = 0x7000_1234_5000usize;
        let starts: Vec<u64> =
            (0..1u64 << 16).map(|i| fold.hash_one((copy, 0x1_0000_0000 + 16 * i, 0x1_0000_0100 + 16 * i, false))).collect();
        let (low, high) = spread(&starts);
        assert!(low > 38_000 && high == 128, "ranges in a row: {low} buckets, {high} tags");
        let copies: Vec<u64> = (0..1usize << 16).map(|i| fold.hash_one((copy + 0x140 * i, 0x1_0000_0000u64, 0x1_0000_0100u64, false))).collect();
        let (low, high) = spread(&copies);
        assert!(low > 38_000 && high == 128, "copies in a row: {low} buckets, {high} tags");
        // The fields are not interchangeable.
        assert_ne!(fold.hash_one((1usize, 2u64, 3u64, false)), fold.hash_one((2usize, 1u64, 3u64, false)));
        assert_ne!(fold.hash_one((1usize, 2u64, 3u64, false)), fold.hash_one((1usize, 2u64, 3u64, true)));
        assert_ne!(fold.hash_one([1u8, 2].as_slice()), fold.hash_one([1u8, 2, 0].as_slice()));
    }

    /// What one slot made again costs the census in each form, at the
    /// world's sizes (640,000 slots viewing 153,000 ranges of 8,600 copies;
    /// 35,000 slots a frame get another range).
    /// `cargo test --release -p victory-gpu --lib census_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn census_cost() {
        for quick in [false, true] {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
            let ranges: Vec<Range> = (0..153_000).map(|_| rng.range(8_600, 1 << 30)).collect();
            let mut slots = HashMap::<u32, Range, Hashing>::with_hasher(Hashing::new(quick));
            let mut counts = Counts::<Range, u32>::new(quick);
            let mut set = |slot: u32, new: Range| {
                if let Some(old) = slots.remove(&slot) {
                    count(&mut counts, old, false);
                }
                count(&mut counts, new, true);
                slots.insert(slot, new);
            };
            for slot in 0..640_000 {
                set(slot, ranges[(rng.next() % 153_000) as usize]);
            }
            // Tables of 16 slots at ring positions, as the title writes them.
            let n = 20 * 35_000;
            let writes: Vec<(u32, Range)> = (0..n)
                .map(|k| ((rng.next() % 40_000) as u32 * 16 + k % 16, ranges[(rng.next() % 153_000) as usize]))
                .collect();
            let started = std::time::Instant::now();
            for (slot, range) in writes {
                set(slot, range);
            }
            let each = started.elapsed().as_nanos() as f64 / f64::from(n);
            println!("census, quick {quick}: {each:.0} ns a slot made again");
        }
    }
}
