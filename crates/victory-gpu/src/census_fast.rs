//! mac: `VICTORY_CENSUS_FAST=1` (`crate::mac_switches::census_fast`, off by
//! default): a view heap's census (`crate::bindless::Viewed`) held under its
//! lock for less time and taken less often.
//!
//! The census lock was the largest serial resource of a frame at the prison
//! cell (`notes/profile-first-run.md`: per frame 1,025 waits for 243.9 ms by
//! the title's threads, 1,405 for 117.5 ms by heap syncs). Two users held or
//! took it most:
//!
//! 1. The ranges a heap's unordered views write (`bindless::widened_writes`)
//!    were found by walking every range of the census (130,000 to 150,000
//!    in the world, five in six of them not unordered) under the lock, once
//!    for every heap-writing draw of a list at its next barrier. Here the
//!    census keeps its unordered ranges apart ([`Unordered`], changed
//!    exactly where its ranges come and go) and the last answer
//!    ([`Fast::kept`]), valid while the census's unordered ranges stay as
//!    they were: a generation counted up under the lock at each change of
//!    them ([`Fast::changed`]). A query whose generation is the kept
//!    answer's does not take the census lock at all.
//! 2. A heap sync entered every slot it made under the census lock, one
//!    lock a slot (25,000 to 35,000 a frame). Here the slots of a pass made
//!    the quick way are entered [`BATCH`] at a time
//!    (`bindless::Batch`). A slot's lock is then let go before its census
//!    entry is in, so another thread can make the slot again in between;
//!    every census update therefore carries a ticket drawn where the slot
//!    was made ([`Fast::ticket`], under the slot's lock), and an update older
//!    than the one a slot's entry is from is dropped ([`Tickets::newer`]).
//!    The census so ends with what the slot was made from last, as it did
//!    when the update went in under the slot's lock.
//!
//! Plain std, outside the Windows-only census, so that the tests run on any
//! host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;
use std::sync::atomic::{
    AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::{Arc, Mutex};

/// A range key as the census has them: copy, start, end, unordered access.
pub(crate) type Range = (usize, u64, u64, bool);

/// A range unordered views write: the copy, the title address, the bytes
/// (what `bindless::widened_writes` gives).
pub(crate) type Write<S> = (Arc<S>, usize, usize);

/// Census updates of a heap sync pass entered under one hold of the lock.
/// A pass is up to 2,048 slots; a hold of this many keeps the title's
/// threads waiting for it some tens of microseconds at most.
pub(crate) const BATCH: usize = 256;

/// The census's unordered ranges alone, with their copies, kept under the
/// census lock where its range counts change (`bindless::Inner::count`).
pub(crate) struct Unordered<S> {
    ranges: BTreeMap<Range, Arc<S>>,
}

impl<S> Unordered<S> {
    pub(crate) fn new() -> Self {
        Unordered { ranges: BTreeMap::new() }
    }

    /// Unordered range `key` of `copy` is viewed now (it was not).
    pub(crate) fn came(&mut self, key: Range, copy: &Arc<S>) {
        self.ranges.insert(key, copy.clone());
    }

    /// Unordered range `key` is viewed by no slot any more.
    pub(crate) fn went(&mut self, key: &Range) {
        self.ranges.remove(key);
    }

    /// What `bindless::widened_writes` finds by walking every range: the
    /// unordered ones that are not empty, as copy, start and bytes. In key
    /// order, the order the census's B-tree form (`VICTORY_QUICK_SYNC=0`)
    /// gives too; its hash form gives the same set in no set order.
    pub(crate) fn writes(&self) -> Vec<Write<S>> {
        self.ranges
            .iter()
            .filter(|((_, address, end, _), _)| address < end)
            .map(|(&(_, address, end, _), copy)| (copy.clone(), address as usize, (end - address) as usize))
            .collect()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

/// The parts of a census outside its lock.
pub(crate) struct Fast<S> {
    /// Counted up, under the census lock, at every change of the census's
    /// unordered ranges ([`Unordered`]).
    generation: AtomicU64,
    /// The last answer of [`Fast::answer`] and the generation it is of;
    /// none since the last change.
    kept: Mutex<Option<(u64, Arc<[Write<S>]>)>>,
    /// The last ticket drawn ([`Fast::ticket`]).
    tickets: AtomicU64,
}

impl<S> Default for Fast<S> {
    fn default() -> Self {
        Fast { generation: AtomicU64::new(0), kept: Mutex::new(None), tickets: AtomicU64::new(0) }
    }
}

impl<S> Fast<S> {
    /// Under the census lock: its unordered ranges changed. Every change of
    /// them must be told before the lock is let go, or [`Fast::kept`] goes
    /// on answering what they were. The kept answer goes with it, so that
    /// it holds no copy the census let go of (a copy no slot views any more
    /// is given back when its last holder goes, `crate::shadow`).
    pub(crate) fn changed(&self) {
        self.generation.fetch_add(1, Release);
        *self.kept.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The last answer, if the census's unordered ranges are still what it
    /// was made from. Takes no census lock: the generation is read once,
    /// and the answer kept for it is what the census held at it (a change
    /// told after the read is one this query came before).
    pub(crate) fn kept(&self) -> Option<Arc<[Write<S>]>> {
        let now = self.generation.load(Acquire);
        match &*self.kept.lock().unwrap_or_else(|e| e.into_inner()) {
            Some((of, writes)) if *of == now => Some(writes.clone()),
            _ => None,
        }
    }

    /// Under the census lock, whose unordered ranges are `unordered`: the
    /// answer, kept for the queries after. (All changes are told under the
    /// lock, so the generation read here is exact, and a kept answer is of
    /// it or older.)
    pub(crate) fn answer(&self, unordered: &Unordered<S>) -> Arc<[Write<S>]> {
        let now = self.generation.load(Relaxed);
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((of, writes)) = &*kept
            && *of == now
        {
            return writes.clone();
        }
        let writes: Arc<[Write<S>]> = unordered.writes().into();
        *kept = Some((now, writes.clone()));
        writes
    }

    /// A ticket for a census update of a slot, drawn under the slot's lock
    /// where the slot is made: of two makings of a slot, the later one has
    /// the larger ticket (the slot's lock orders them, and every ticket is
    /// drawn from this one counter).
    pub(crate) fn ticket(&self) -> u64 {
        self.tickets.fetch_add(1, Relaxed) + 1
    }
}

/// Per slot, the ticket of the update its census entry is from (0: none
/// yet). Kept under the census lock.
#[derive(Default)]
pub(crate) struct Tickets(Vec<u64>);

impl Tickets {
    /// Whether an update of slot `index` with `ticket` is later than the one
    /// the slot's entry is from; if it is, it becomes the slot's. An update
    /// that is not is dropped by the caller: the slot was made again since,
    /// and that making's update is in or on its way.
    pub(crate) fn newer(&mut self, index: u32, ticket: u64) -> bool {
        let i = index as usize;
        if self.0.len() <= i {
            self.0.resize(i + 1, 0);
        }
        if ticket <= self.0[i] {
            return false;
        }
        self.0[i] = ticket;
        true
    }
}

/// For each of a barrier's events (`Some(heap)` for a heap-writing draw,
/// none for a write of its own): whether it is the last of its heap's.
/// `crate::host::HostList::sync_widened` puts a heap's ranges in once, where
/// its last draw stood: every draw of a heap before a barrier stands for the
/// same ranges (they are read at the barrier), and of equal writes only the
/// last one decides what a widened view gets (`crate::widened::sync`: a
/// patch's later segments win; a direct copy is chosen when all writes over
/// a view are equal).
pub(crate) fn last_of_each(heaps: impl IntoIterator<Item = Option<usize>>) -> Vec<bool> {
    let heaps: Vec<Option<usize>> = heaps.into_iter().collect();
    let mut last = vec![false; heaps.len()];
    let mut met: Vec<usize> = Vec::new();
    for (i, heap) in heaps.iter().enumerate().rev() {
        if let Some(heap) = heap
            && !met.contains(heap)
        {
            met.push(*heap);
            last[i] = true;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A host copy, as far as the census cares.
    #[derive(Debug)]
    struct Copy(#[allow(dead_code)] u32);

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A census as `bindless::Inner` counts it, with the fast parts: the
    /// slots, how many slots view each range (and its copy), the unordered
    /// ranges apart, the tickets.
    struct Census {
        slots: HashMap<u32, (Range, Arc<Copy>)>,
        counts: HashMap<Range, (u32, Arc<Copy>)>,
        unordered: Unordered<Copy>,
        tickets: Tickets,
        /// Whether a change of the unordered ranges is told (false: the
        /// negative control of the tests below).
        tell: bool,
    }

    impl Census {
        fn new() -> Self {
            Census {
                slots: HashMap::new(),
                counts: HashMap::new(),
                unordered: Unordered::new(),
                tickets: Tickets::default(),
                tell: true,
            }
        }

        /// `bindless::Inner::count`: whether the unordered ranges changed.
        fn count(&mut self, key: Range, copy: &Arc<Copy>, add: bool) -> bool {
            let e = self.counts.entry(key).or_insert_with(|| (0, copy.clone()));
            let was = e.0;
            e.0 = if add { e.0 + 1 } else { e.0.saturating_sub(1) };
            let now = e.0;
            if now == 0 {
                self.counts.remove(&key);
            }
            if (was == 0) != (now == 0) && key.3 {
                if now > 0 {
                    self.unordered.came(key, copy);
                } else {
                    self.unordered.went(&key);
                }
                return true;
            }
            false
        }

        /// `bindless::Viewed::set_seen` (with the switch, ticket `ticket`).
        fn set(&mut self, fast: &Fast<Copy>, index: u32, ticket: u64, seen: Option<(Range, Arc<Copy>)>) {
            if !self.tickets.newer(index, ticket) {
                return;
            }
            let mut changed = false;
            if let Some((key, copy)) = self.slots.remove(&index) {
                changed |= self.count(key, &copy, false);
            }
            if let Some((key, copy)) = seen {
                changed |= self.count(key, &copy, true);
                self.slots.insert(index, (key, copy));
            }
            if changed && self.tell {
                fast.changed();
            }
        }

        /// The old walk of every range (`bindless::widened_writes`), sorted.
        fn walked(&self) -> Vec<(usize, usize, usize)> {
            let mut w: Vec<_> = self
                .counts
                .iter()
                .filter(|((_, address, end, uav), _)| *uav && address < end)
                .map(|((_, address, end, _), (_, copy))| (Arc::as_ptr(copy) as usize, *address as usize, (end - address) as usize))
                .collect();
            w.sort_unstable();
            w
        }
    }

    /// An answer as (copy, start, bytes), sorted.
    fn plain(writes: &[Write<Copy>]) -> Vec<(usize, usize, usize)> {
        let mut w: Vec<_> = writes.iter().map(|(c, a, l)| (Arc::as_ptr(c) as usize, *a, *l)).collect();
        w.sort_unstable();
        w
    }

    /// A query as `bindless::unordered_writes` makes it: the kept answer, or
    /// one made under the lock. Whether it was kept.
    fn query(census: &Mutex<Census>, fast: &Fast<Copy>) -> (Arc<[Write<Copy>]>, bool) {
        if let Some(kept) = fast.kept() {
            return (kept, true);
        }
        let census = census.lock().unwrap();
        (fast.answer(&census.unordered), false)
    }

    /// Slots rewritten at random (the same range again, another range, an
    /// unordered one, an empty one, none): after every change a query
    /// answers what the old walk finds, kept or made anew; a change that
    /// leaves the unordered ranges as they were keeps the answer.
    fn random_rewrites(tell: bool) -> Result<(usize, usize), String> {
        let copies: Vec<Arc<Copy>> = (0..5).map(|i| Arc::new(Copy(i))).collect();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let census = Mutex::new(Census::new());
        census.lock().unwrap().tell = tell;
        let fast = Fast::default();
        let (mut kept, mut made) = (0, 0);
        for step in 0..60_000u64 {
            let slot = (rng.next() % 300) as u32;
            let seen = (step % 9 != 0).then(|| {
                let c = (rng.next() % 5) as usize;
                let start = 0x1000 + (rng.next() % 40) * 16;
                // Now and then an empty range (end at the start).
                let len = if rng.next() % 50 == 0 { 0 } else { 16 + rng.next() % 64 };
                let key = (Arc::as_ptr(&copies[c]) as usize, start, start + len, rng.next() % 3 == 0);
                (key, copies[c].clone())
            });
            {
                let mut c = census.lock().unwrap();
                let ticket = fast.ticket();
                c.set(&fast, slot, ticket, seen);
            }
            // Queries between some changes only, two in a row now and then.
            for _ in 0..(step % 3) {
                let (answer, was_kept) = query(&census, &fast);
                let walked = census.lock().unwrap().walked();
                if plain(&answer) != walked {
                    return Err(format!("step {step}: answered {} ranges, the walk finds {}", answer.len(), walked.len()));
                }
                if was_kept {
                    kept += 1;
                } else {
                    made += 1;
                }
            }
        }
        Ok((kept, made))
    }

    #[test]
    fn answers_follow_every_change_of_the_unordered_ranges() {
        let (kept, made) = random_rewrites(true).unwrap();
        assert!(kept > 1_000 && made > 1_000, "both kept and new answers are tested: {kept} kept, {made} made");
    }

    /// The negative control: if a change of the unordered ranges is not
    /// told, the kept answer goes stale and the test above sees it.
    #[test]
    fn an_untold_change_gives_a_stale_answer() {
        assert!(random_rewrites(false).is_err());
    }

    /// The cases one by one: a range that comes, one that goes, a slot made
    /// again from the same range, an empty range, a change of ranges that
    /// are not unordered, and two heaps, each with its own answer.
    #[test]
    fn each_kind_of_change_is_seen() {
        let a = Arc::new(Copy(0));
        let pa = Arc::as_ptr(&a) as usize;
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        let set = |slot: u32, seen: Option<(u64, u64, bool)>| {
            let mut c = census.lock().unwrap();
            let ticket = fast.ticket();
            c.set(&fast, slot, ticket, seen.map(|(s, e, u)| ((pa, s, e, u), a.clone())));
        };
        let answer = || plain(&query(&census, &fast).0);
        assert_eq!(answer(), []);
        set(0, Some((0x100, 0x200, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100)], "a range came");
        let first = query(&census, &fast);
        assert!(first.1, "and the answer is kept");
        set(1, Some((0x400, 0x480, false)));
        let second = query(&census, &fast);
        assert!(second.1 && Arc::ptr_eq(&first.0, &second.0), "a range that is not unordered keeps the answer");
        set(2, Some((0x800, 0x900, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "a second range came");
        set(0, Some((0x100, 0x200, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "slot 0 made again from its range");
        set(3, Some((0x100, 0x200, true)));
        set(0, None);
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "slot 3 still views it");
        set(3, Some((0x100, 0x100, true)));
        assert_eq!(answer(), [(pa, 0x800, 0x100)], "the range went; an empty one is none");
        assert_eq!(census.lock().unwrap().unordered.len(), 2, "the empty range is kept, only not answered");
        set(2, Some((0x800, 0x900, false)));
        assert_eq!(answer(), [], "a slot from unordered to shader view");
        // Two heaps: each answers for itself.
        let other = Mutex::new(Census::new());
        let other_fast = Fast::default();
        {
            let mut c = other.lock().unwrap();
            let ticket = other_fast.ticket();
            c.set(&other_fast, 0, ticket, Some(((pa, 0x10, 0x20, true), a.clone())));
        }
        assert_eq!(plain(&query(&other, &other_fast).0), [(pa, 0x10, 0x10)]);
        assert_eq!(answer(), [], "the first heap's answer is its own");
        set(5, Some((0x30, 0x40, true)));
        assert_eq!(plain(&query(&other, &other_fast).0), [(pa, 0x10, 0x10)], "and the other's its own");
        assert_eq!(answer(), [(pa, 0x30, 0x10)]);
    }

    /// A kept answer holds the copies of the ranges it names; once the
    /// census lets go of a range, the answer lets go of its copy too.
    #[test]
    fn a_change_lets_go_of_the_kept_answer() {
        let (a, b) = (Arc::new(Copy(0)), Arc::new(Copy(1)));
        let (pa, pb) = (Arc::as_ptr(&a) as usize, Arc::as_ptr(&b) as usize);
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        let set = |slot: u32, seen: Option<(Range, Arc<Copy>)>| {
            let mut c = census.lock().unwrap();
            let ticket = fast.ticket();
            c.set(&fast, slot, ticket, seen);
        };
        set(0, Some(((pa, 0x100, 0x200, true), a.clone())));
        set(1, Some(((pb, 0x100, 0x200, true), b.clone())));
        assert_eq!(query(&census, &fast).0.len(), 2);
        assert_eq!(Arc::strong_count(&b), 5, "the test's, the slot's, the range count's, the unordered ranges' and the answer's");
        set(1, None);
        assert_eq!(Arc::strong_count(&b), 1, "only the test's: the census and its kept answer let go");
        assert_eq!(plain(&query(&census, &fast).0), [(pa, 0x100, 0x100)]);
        set(0, Some(((pa, 0x100, 0x200, false), a.clone())));
        assert_eq!(Arc::strong_count(&a), 3, "the test's, the slot's, the range count's");
    }

    /// While one thread enters ranges, others query: every answer is what
    /// the census held at some moment (here: the first n ranges entered, for
    /// an n that never falls between one thread's queries), and once the
    /// thread is done the answer is all of them.
    #[test]
    fn queries_beside_changes_see_a_census_that_was() {
        let copy = Arc::new(Copy(0));
        let pc = Arc::as_ptr(&copy) as usize;
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        const N: u64 = 3_000;
        std::thread::scope(|s| {
            s.spawn(|| {
                for k in 0..N {
                    let mut c = census.lock().unwrap();
                    let ticket = fast.ticket();
                    c.set(&fast, k as u32, ticket, Some(((pc, 16 * k, 16 * k + 8, true), copy.clone())));
                }
            });
            for _ in 0..3 {
                s.spawn(|| {
                    let mut seen = 0;
                    for _ in 0..20_000 {
                        let answer = plain(&query(&census, &fast).0);
                        let n = answer.len();
                        let expected: Vec<_> = (0..n as u64).map(|k| (pc, 16 * k as usize, 8)).collect();
                        assert_eq!(answer, expected, "the first {n} ranges");
                        assert!(n >= seen, "an answer older than the one before: {n} after {seen}");
                        seen = n;
                    }
                });
            }
        });
        assert_eq!(query(&census, &fast).0.len(), N as usize);
    }

    /// Census updates entered late (a pass's batch) and at once (the title's
    /// threads), in an order of makings the slot locks give: each slot ends
    /// with what its last making entered, and without the tickets it would
    /// not (the order of this test has late updates entered after a later
    /// making's).
    #[test]
    fn late_updates_older_than_a_slots_last_making_are_dropped() {
        let copy = Arc::new(Copy(0));
        let pc = Arc::as_ptr(&copy) as usize;
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for with_tickets in [true, false] {
            let fast = Fast::default();
            let mut census = Census::new();
            // The last making of each slot, in the order of the makings.
            let mut last: HashMap<u32, Option<Range>> = HashMap::new();
            let mut batch: Vec<(u32, u64, Option<(Range, Arc<Copy>)>)> = Vec::new();
            let mut late_after_newer = 0;
            // Without tickets every update wins as it is entered: each gets
            // a ticket of its own at entry.
            let mut entered = 0u64;
            for step in 0..40_000u64 {
                let slot = (rng.next() % 64) as u32;
                let seen = (rng.next() % 5 != 0).then(|| {
                    let start = (rng.next() % 32) * 16;
                    ((pc, start, start + 16, rng.next() % 4 == 0), copy.clone())
                });
                last.insert(slot, seen.as_ref().map(|s| s.0));
                // The making draws its ticket (under the slot's lock).
                let ticket = fast.ticket();
                if rng.next() % 2 == 0 {
                    batch.push((slot, ticket, seen));
                } else {
                    if batch.iter().any(|(s, ..)| *s == slot) {
                        late_after_newer += 1;
                    }
                    entered += 1;
                    census.set(&fast, slot, if with_tickets { ticket } else { entered }, seen);
                }
                if rng.next() % 16 == 0 || step == 39_999 {
                    for (slot, ticket, seen) in batch.drain(..) {
                        entered += 1;
                        census.set(&fast, slot, if with_tickets { ticket } else { entered }, seen);
                    }
                }
            }
            let ends: HashMap<u32, Option<Range>> =
                last.keys().map(|&s| (s, census.slots.get(&s).map(|(k, _)| *k))).collect();
            assert!(late_after_newer > 100, "late updates entered after a later making's: {late_after_newer}");
            assert_eq!(ends == last, with_tickets, "with tickets each slot ends with its last making");
            if with_tickets {
                // And the counts and unordered ranges are those of the slots.
                let mut model = Census::new();
                for (&slot, key) in &last {
                    let ticket = fast.ticket();
                    model.set(&fast, slot, ticket, key.map(|k| (k, copy.clone())));
                }
                assert_eq!(census.walked(), model.walked());
                let mut counts: Vec<_> = census.counts.iter().map(|(k, (n, _))| (*k, *n)).collect();
                let mut want: Vec<_> = model.counts.iter().map(|(k, (n, _))| (*k, *n)).collect();
                counts.sort_unstable();
                want.sort_unstable();
                assert_eq!(counts, want);
            }
        }
    }

    /// The same with threads: a "sync" making slots and entering them in
    /// batches, a "title thread" making the same slots and entering each at
    /// once, each making under the slot's own lock. Afterwards every slot's
    /// census entry is what the slot was made from last.
    #[test]
    fn threads_making_the_same_slots_leave_the_census_with_the_slots() {
        const SLOTS: usize = 128;
        let fast = Fast::default();
        let census = Mutex::new(Census::new());
        let made: Vec<Mutex<u64>> = (0..SLOTS).map(|_| Mutex::new(0)).collect();
        let next = AtomicU64::new(1);
        let copy = Arc::new(Copy(0));
        let pc = Arc::as_ptr(&copy) as usize;
        let range = |v: u64| ((pc, 16 * (v % 997), 16 * (v % 997) + 16, v % 3 == 0), copy.clone());
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut batch = Vec::new();
                for round in 0..4_000usize {
                    let slot = (round * 7) % SLOTS;
                    {
                        let mut m = made[slot].lock().unwrap();
                        let v = next.fetch_add(1, Relaxed);
                        *m = v;
                        batch.push((slot as u32, fast.ticket(), Some(range(v))));
                    }
                    if batch.len() >= 16 {
                        let mut c = census.lock().unwrap();
                        for (slot, ticket, seen) in batch.drain(..) {
                            c.set(&fast, slot, ticket, seen);
                        }
                    }
                }
                let mut c = census.lock().unwrap();
                for (slot, ticket, seen) in batch.drain(..) {
                    c.set(&fast, slot, ticket, seen);
                }
            });
            s.spawn(|| {
                for round in 0..4_000usize {
                    let slot = (round * 5) % SLOTS;
                    let mut m = made[slot].lock().unwrap();
                    let v = next.fetch_add(1, Relaxed);
                    *m = v;
                    let ticket = fast.ticket();
                    census.lock().unwrap().set(&fast, slot as u32, ticket, Some(range(v)));
                }
            });
        });
        let c = census.lock().unwrap();
        for (slot, m) in made.iter().enumerate() {
            let v = *m.lock().unwrap();
            if v == 0 {
                continue;
            }
            assert_eq!(c.slots.get(&(slot as u32)).map(|(k, _)| *k), Some(range(v).0), "slot {slot}");
        }
    }

    #[test]
    fn each_heap_is_marked_at_its_last_event() {
        assert_eq!(last_of_each([]), Vec::<bool>::new());
        assert_eq!(last_of_each([None, None]), [false, false]);
        assert_eq!(last_of_each([Some(1), None, Some(1), None]), [false, false, true, false]);
        assert_eq!(
            last_of_each([Some(1), Some(2), None, Some(1), Some(3), Some(2)]),
            [false, false, false, true, true, true]
        );
    }

    #[test]
    fn tickets_grow_with_the_slots_and_keep_the_latest() {
        let mut t = Tickets::default();
        assert!(t.newer(10, 5));
        assert!(!t.newer(10, 5), "the same update twice");
        assert!(!t.newer(10, 4), "an older one");
        assert!(t.newer(10, 6));
        assert!(t.newer(3, 1), "slots below grown ones start empty");
        assert!(t.newer(100_000, 1));
    }
}
