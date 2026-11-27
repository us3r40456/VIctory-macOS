//! mac: the locks a host heap's slots are made under, in two forms chosen
//! when a heap is made (`VICTORY_HEAP_LOCKS`, off by default;
//! `crate::mac_switches::heap_locks`).
//!
//! As it was, and is without the switch: one lock for a heap's records
//! ([`Made`], the record each title slot was last made from), held while a
//! slot's host view is made, entered in the census and kept; one lock for
//! what the slots hold ([`Pieces`]); and the census's own
//! (`crate::bindless::Viewed`). Every thread that makes a slot stands in
//! that one line: the heap sync on the submitting thread (25,000 to 40,000
//! slots a world frame), the title's threads writing views by calls and
//! their draws reading tables.
//!
//! With `VICTORY_HEAP_LOCKS=1` a slot is made under a lock of its own
//! group: [`SLOT_LOCKS`] locks, a slot's picked by a multiplicative hash of
//! its index, so that neighbours and slots a fixed distance apart have
//! different ones (avatarmanz 6c17ce9 and 11511b81: 512 locks by the low
//! bits had threads walking a heap in step in each other's way). What the
//! slots hold is in [`KEPT_PIECES`] pieces, each behind its own lock, and
//! what a slot stops holding is retired after that lock is let go
//! (hyphennn 5d659e0c). The record of a slot is an atomic in both forms:
//! written only under the slot's lock, read under it or, where a stale
//! answer only costs a second look under the lock, without one
//! ([`Made::peek`]).
//!
//! In the split form a thread that finds one of these locks held tries
//! again up to [`SPINS`] times before it sleeps for it. The locks that
//! stay shared (the census's, the retired list's) are held for a fraction
//! of a microsecond and taken for every slot made; under Wine a thread
//! that sleeps for such a lock and is woken costs far more than the lock
//! is held, and four threads making slots took longer than one (measured
//! under Wine on D3DMetal, `host/split_tests.rs`: 16,640 sleeps for the
//! census lock in 49,152 slots, 0.85 µs a slot against 0.65 alone; with
//! the tries 331 sleeps and 0.48 µs).
//!
//! `VICTORY_HEAP_LOCKS=count` keeps the one lock of each kind, taken as
//! before, and only counts ([`log`]): how often and how long a thread
//! waited for a slot lock, a kept lock, the census lock and the retired
//! list's lock, and how many slots were made, by heap syncs and by other
//! threads apart. With `=1` the same line shows the split form's numbers
//! (a wait there is one that outlasted the tries). A wait is clocked only
//! when the lock was not free (`try_lock` first), so the meter costs no
//! clock call on a free lock.
//!
//! Plain std, outside the Windows-only heap, so that the tests run on any
//! host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{
    AtomicBool, AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::{Mutex, MutexGuard, OnceLock, TryLockError};

use super::paged::{PAGE, Paged};

/// Slot locks of a heap in the split form (a power of two).
pub(crate) const SLOT_LOCKS: usize = 1 << 14;

/// Pieces of what a heap's slots hold in the split form.
pub(crate) const KEPT_PIECES: usize = 64;

/// The switch: whether heaps are made in the split form, and whether lock
/// waits and made slots are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mode {
    pub(crate) split: bool,
    pub(crate) meter: bool,
}

/// `VICTORY_HEAP_LOCKS`, read once; says so in the log when it is set to
/// something.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("VICTORY_HEAP_LOCKS").ok();
        let [split, meter] = crate::mac_switches::heap_locks(value.as_deref());
        if split {
            SPIN.store(true, Relaxed);
            tracing::info!(
                slot_locks = SLOT_LOCKS,
                kept_pieces = KEPT_PIECES,
                tries = SPINS,
                "VICTORY_HEAP_LOCKS=1: a host heap's slots are made under locks of their own, what they hold is \
                 kept in pieces, a held lock is tried again before a thread sleeps for it, and lock waits are \
                 counted (the frame line's \"heap locks\")"
            );
        } else if meter {
            tracing::info!(
                "VICTORY_HEAP_LOCKS=count: a host heap's one slot lock, kept lock and census lock as before, \
                 their waits counted (the frame line's \"heap locks\")"
            );
        }
        Mode { split, meter }
    })
}

/// The record (view index and generation; 0: none) each title slot of a
/// heap was last made from, and the locks slots are made under: one for
/// all of them, or [`SLOT_LOCKS`].
pub(crate) struct Made {
    keys: Box<[AtomicU64]>,
    locks: Box<[Mutex<()>]>,
    /// 32 less the bits of the lock count (split form).
    shift: u32,
}

impl Made {
    /// `count` slots made from no record, behind one lock or (`split`)
    /// [`SLOT_LOCKS`].
    pub(crate) fn new(count: usize, split: bool) -> Self {
        let locks = if split { SLOT_LOCKS } else { 1 };
        Made {
            keys: (0..count).map(|_| AtomicU64::new(0)).collect(),
            locks: (0..locks).map(|_| Mutex::new(())).collect(),
            shift: 32 - locks.trailing_zeros(),
        }
    }

    /// Whether slots have locks of their own.
    pub(crate) fn split(&self) -> bool {
        self.locks.len() > 1
    }

    /// Which lock slot `index` is made under.
    fn lock_of(&self, index: u32) -> usize {
        if self.locks.len() == 1 {
            0
        } else {
            (index.wrapping_mul(0x9E37_79B1) >> self.shift) as usize
        }
    }

    /// The lock slot `index` is made under, held until the guard goes. In
    /// the one-lock form it is every slot's, so a thread takes it once.
    pub(crate) fn lock(&self, index: u32) -> Slot<'_> {
        let lock = self.lock_of(index);
        Slot { made: self, lock, _held: locked(&self.locks[lock], SLOT) }
    }

    /// Every slot's lock, taken in order (a thread holding one slot's lock
    /// takes no other, so the order cannot cross another thread's).
    pub(crate) fn lock_all(&self) -> All<'_> {
        All { made: self, _held: self.locks.iter().map(|lock| locked(lock, SLOT)).collect() }
    }

    /// Slot `index`'s record as it is now, without its lock; none past the
    /// end. A slot whose record another thread is changing reads as the
    /// old or the new one, never a mix.
    pub(crate) fn peek(&self, index: u32) -> Option<u64> {
        Some(self.keys.get(index as usize)?.load(Acquire))
    }

    /// The slots there are records for.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Whether a heap sync has a slot to make again: its record now (`current`:
/// it names a view that is current; the view's index and generation)
/// against the record the host slot was last made from (none: no host
/// slot). A record that is no current view is stale where a view is shown.
pub(crate) fn stale(current: bool, view: u32, generation: u32, made: Option<u64>) -> bool {
    match made {
        Some(m) if current => m != u64::from(generation) << 32 | u64::from(view),
        Some(m) => m != 0,
        None => true,
    }
}

/// A slot's lock, held ([`Made::lock`]): the records it covers can be read
/// and set.
pub(crate) struct Slot<'a> {
    made: &'a Made,
    lock: usize,
    _held: MutexGuard<'a, ()>,
}

impl Slot<'_> {
    /// Slot `index`'s record; `index` is a slot this lock covers.
    pub(crate) fn get(&self, index: u32) -> u64 {
        debug_assert_eq!(self.made.lock_of(index), self.lock, "another slot's lock");
        self.made.keys[index as usize].load(Acquire)
    }

    /// [`Slot::get`]; none past the end.
    pub(crate) fn find(&self, index: u32) -> Option<u64> {
        debug_assert_eq!(self.made.lock_of(index), self.lock, "another slot's lock");
        self.made.peek(index)
    }

    /// Slot `index` is made from record `key` (0: none).
    pub(crate) fn set(&mut self, index: u32, key: u64) {
        debug_assert_eq!(self.made.lock_of(index), self.lock, "another slot's lock");
        self.made.keys[index as usize].store(key, Release);
    }
}

/// Every slot's lock, held ([`Made::lock_all`]).
pub(crate) struct All<'a> {
    made: &'a Made,
    _held: Vec<MutexGuard<'a, ()>>,
}

impl All<'_> {
    /// Slot `index`'s record.
    pub(crate) fn get(&self, index: u32) -> u64 {
        self.made.keys[index as usize].load(Acquire)
    }
}

/// What a heap's slots hold ([`Paged`]), behind one lock or in
/// [`KEPT_PIECES`] pieces with a lock each. Piece `p` holds the table's
/// pages `p`, `p + pieces`, `p + 2 × pieces`...: slots of one page (a
/// table's run) share a lock and a page of memory, as in the one table, and
/// neighbouring pages have different locks.
pub(crate) struct Pieces<T> {
    pieces: Box<[Mutex<Paged<T>>]>,
    len: usize,
}

impl<T> Pieces<T> {
    /// `len` empty slots in one piece or (`split`) [`KEPT_PIECES`].
    pub(crate) fn new(len: usize, split: bool) -> Self {
        let n = if split { KEPT_PIECES } else { 1 };
        let pages = len.div_ceil(PAGE);
        let pieces = (0..n)
            .map(|p| {
                // The table's pages this piece holds, the table's last
                // (which may be short) as this piece's last.
                let own = (pages + n - 1 - p) / n;
                let last = pages.checked_sub(1).filter(|last| last % n == p);
                let slots = match last {
                    Some(last) => (own - 1) * PAGE + (len - last * PAGE),
                    None => own * PAGE,
                };
                Mutex::new(Paged::new(slots))
            })
            .collect();
        Pieces { pieces, len }
    }

    /// Whether the slots are in pieces with a lock each.
    pub(crate) fn split(&self) -> bool {
        self.pieces.len() > 1
    }

    /// The piece slot `index` is in and its place there; a place past every
    /// piece's end for a slot past the table's.
    fn place(&self, index: usize) -> (usize, usize) {
        if index >= self.len {
            return (0, usize::MAX);
        }
        let n = self.pieces.len();
        let page = index / PAGE;
        (page % n, page / n * PAGE + index % PAGE)
    }

    /// Runs `f` on the piece slot `index` is in, locked, with the slot's
    /// place in it.
    pub(crate) fn with<R>(&self, index: usize, f: impl FnOnce(&mut Paged<T>, usize) -> R) -> R {
        let (piece, at) = self.place(index);
        f(&mut locked(&self.pieces[piece], KEPT), at)
    }

    /// Runs `f` on what slot `index` holds (none: nothing, or past the end).
    pub(crate) fn get<R>(&self, index: usize, f: impl FnOnce(Option<&T>) -> R) -> R {
        self.with(index, |piece, at| f(piece.get(at)))
    }

    /// Runs `f` on every slot that holds something, with its index: piece
    /// by piece, each locked while its slots are walked. In one piece the
    /// slots come in order.
    pub(crate) fn each(&self, mut f: impl FnMut(usize, &T)) {
        let n = self.pieces.len();
        for (p, piece) in self.pieces.iter().enumerate() {
            let piece = locked(piece, KEPT);
            for (at, held) in piece.iter() {
                f((at / PAGE * n + p) * PAGE + at % PAGE, held);
            }
        }
    }
}

/// The locks whose waits are counted: a heap's slot locks, its kept locks,
/// its census's lock, and the lock of the list of retired resources
/// (`crate::host::retire`).
pub(crate) const SLOT: usize = 0;
pub(crate) const KEPT: usize = 1;
pub(crate) const CENSUS: usize = 2;
pub(crate) const RETIRED: usize = 3;
const KINDS: usize = 4;

/// How often a thread tries a held lock again before it sleeps for it
/// (split form). A try is a few nanoseconds; these locks are held for a
/// microsecond at most.
const SPINS: u32 = 128;

/// Whether held locks are tried again first: the split form (set when the
/// switch is read), and tests of it.
static SPIN: AtomicBool = AtomicBool::new(false);

/// Tests of the split form run as it runs: held locks tried again first.
#[cfg(test)]
pub(crate) fn spin_in_tests() {
    SPIN.store(true, Relaxed);
}

thread_local! {
    /// The thread is in a heap sync ([`Syncing`]).
    static SYNCING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The thread is in a heap sync while this lives: its waits and the slots
/// it makes are counted as the syncs', not as the title's threads'.
pub(crate) struct Syncing(bool);

impl Syncing {
    pub(crate) fn enter() -> Self {
        Syncing(SYNCING.with(|s| s.replace(true)))
    }
}

impl Drop for Syncing {
    fn drop(&mut self) {
        SYNCING.with(|s| s.set(self.0));
    }
}

/// Whether the thread is in a heap sync.
fn syncing() -> bool {
    SYNCING.with(std::cell::Cell::get)
}

/// Since the last frame line, per lock kind and by whom (heap syncs, other
/// threads): waits for a lock that was not free, and their nanoseconds.
static WAITS: [[AtomicU64; 2]; KINDS * 2] = [const { [const { AtomicU64::new(0) }; 2] }; KINDS * 2];

/// Since the last frame line: slots made by heap syncs, and by other
/// threads.
static SLOTS_MADE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `lock`, taken. In the split form a held lock is tried again a few times
/// first; with the meter on, a wait for it is counted under `kind`.
pub(crate) fn locked<T>(lock: &Mutex<T>, kind: usize) -> MutexGuard<'_, T> {
    locked_as(lock, kind, mode().meter, SPIN.load(Relaxed))
}

fn locked_as<T>(lock: &Mutex<T>, kind: usize, meter: bool, spin: bool) -> MutexGuard<'_, T> {
    if spin {
        for _ in 0..SPINS {
            match lock.try_lock() {
                Ok(held) => return held,
                Err(TryLockError::Poisoned(e)) => return e.into_inner(),
                Err(TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
    }
    if !meter {
        return lock.lock().unwrap_or_else(|e| e.into_inner());
    }
    match lock.try_lock() {
        Ok(held) => held,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let began = std::time::Instant::now();
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            let waits = &WAITS[kind * 2 + usize::from(!syncing())];
            waits[0].fetch_add(1, Relaxed);
            waits[1].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
            held
        }
    }
}

/// A slot's host view was made (or made null) from a new record.
pub(crate) fn made() {
    if mode().meter {
        SLOTS_MADE[usize::from(!syncing())].fetch_add(1, Relaxed);
    }
}

/// The counts since the last call, each divided by `frames`: per lock kind
/// (slot, kept, census, retired) and by whom (syncs, others) waits and
/// their milliseconds, then the slots made by syncs and by others.
fn take(frames: f64) -> ([[f64; 2]; KINDS * 2], [f64; 2]) {
    let waits = std::array::from_fn(|i| {
        [WAITS[i][0].swap(0, Relaxed) as f64 / frames, WAITS[i][1].swap(0, Relaxed) as f64 / 1e6 / frames]
    });
    (waits, std::array::from_fn(|i| SLOTS_MADE[i].swap(0, Relaxed) as f64 / frames))
}

/// With the frame line (`frames` since the last one): the heap locks'
/// waits and the slots made. Nothing without the switch.
pub(crate) fn log(frames: f64) {
    let mode = mode();
    if !mode.meter {
        return;
    }
    let (w, made) = take(frames);
    tracing::info!(
        "heap locks ({}): {:.0} slots/frame made by heap syncs, {:.0} by other threads; waits for a lock that was \
         not free, per frame, by heap syncs and by other threads: slot lock {:.0} ({:.2} ms) and {:.0} ({:.2} ms), \
         kept lock {:.0} ({:.2} ms) and {:.0} ({:.2} ms), census lock {:.0} ({:.2} ms) and {:.0} ({:.2} ms), \
         retired list {:.0} ({:.2} ms) and {:.0} ({:.2} ms)",
        if mode.split {
            "VICTORY_HEAP_LOCKS=1: slots under locks of their own, what they hold in pieces; a wait is one that \
             outlasted the tries"
        } else {
            "VICTORY_HEAP_LOCKS=count: one lock of each kind, as before"
        },
        made[0],
        made[1],
        w[0][0],
        w[0][1],
        w[1][0],
        w[1][1],
        w[2][0],
        w[2][1],
        w[3][0],
        w[3][1],
        w[4][0],
        w[4][1],
        w[5][0],
        w[5][1],
        w[6][0],
        w[6][1],
        w[7][0],
        w[7][1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn one_lock_covers_every_slot_and_split_locks_spread_them() {
        let one = Made::new(100, false);
        assert!(!one.split());
        assert!((0..100).all(|i| one.lock_of(i) == 0));
        let split = Made::new(1 << 20, true);
        assert!(split.split());
        assert_eq!(split.locks.len(), SLOT_LOCKS);
        // Every lock is some slot's, and none is out of range.
        let mut used = vec![0u32; SLOT_LOCKS];
        for i in 0..1u32 << 20 {
            used[split.lock_of(i)] += 1;
        }
        assert!(used.iter().all(|&n| n > 0), "every lock is used");
        // Evenly: 64 slots a lock on average.
        assert!(used.iter().all(|&n| n < 4 * 64), "no lock has far more than its share: {:?}", used.iter().max());
        // Neighbours, and slots that threads walking a heap in step are at
        // (a table apart, a 64 KiB page of records apart), have different ones.
        for step in [1u32, 2, 8, 64, 128, 512, 2048, 4096] {
            let same = (0..100_000u32).filter(|&i| split.lock_of(i) == split.lock_of(i + step)).count();
            assert!(same < 100, "{same} of 100,000 slots share a lock with the slot {step} on");
        }
    }

    #[test]
    fn a_slot_is_stale_when_its_record_is_not_what_it_was_made_from() {
        let key = 3u64 << 32 | 7;
        assert!(!stale(true, 7, 3, Some(key)), "made from it");
        assert!(stale(true, 7, 4, Some(key)), "another generation");
        assert!(stale(true, 8, 3, Some(key)), "another view");
        assert!(stale(true, 7, 3, Some(0)), "never made");
        assert!(stale(false, 7, 3, Some(key)), "no view any more where one is shown");
        assert!(!stale(false, 7, 3, Some(0)), "no view, none shown");
        assert!(stale(true, 7, 3, None) && stale(false, 0, 0, None), "past the host heap: `mirror` says so");
    }

    #[test]
    fn records_are_read_and_set_under_their_lock() {
        for split in [false, true] {
            let made = Made::new(1000, split);
            assert_eq!(made.len(), 1000);
            assert_eq!(made.peek(5), Some(0));
            assert_eq!(made.peek(1000), None);
            {
                let mut slot = made.lock(5);
                assert_eq!((slot.get(5), slot.find(5)), (0, Some(0)));
                slot.set(5, 0x1_0000_0007);
                assert_eq!(slot.get(5), 0x1_0000_0007);
            }
            assert_eq!(made.peek(5), Some(0x1_0000_0007));
            // Past the end: a lock all the same, no record.
            assert_eq!(made.lock(5000).find(5000), None);
            let all = made.lock_all();
            assert_eq!((all.get(5), all.get(6)), (0x1_0000_0007, 0));
        }
    }

    /// The pieces hold what one table holds, at the same indices, whatever
    /// the table's length (a short last page in any piece).
    #[test]
    fn pieces_hold_what_one_table_holds() {
        for len in [0, 1, PAGE - 1, PAGE, PAGE + 1, 63 * PAGE + 7, 64 * PAGE, 64 * PAGE + 1, 200 * PAGE + 100] {
            let (one, split) = (Pieces::<usize>::new(len, false), Pieces::<usize>::new(len, true));
            assert!(!one.split() && (split.split() || KEPT_PIECES == 1));
            // The pieces' slots are the table's, each once.
            let total: usize = split.pieces.iter().map(|p| p.lock().unwrap().len()).sum();
            assert_eq!(total, len, "{len} slots");
            let mut picks: Vec<usize> = (0..len).step_by(37).chain(len.checked_sub(1)).collect();
            picks.dedup();
            for table in [&one, &split] {
                for &i in &picks {
                    assert!(table.get(i, |held| held.is_none()));
                    let old = table.with(i, |piece, at| piece.slot(at, true).expect("a slot of the table").replace(i));
                    assert_eq!(old, None, "slot {i} of {len} is its own");
                }
                for &i in &picks {
                    assert_eq!(table.get(i, |held| held.copied()), Some(i));
                }
                // Past the end: nothing held, nothing to write.
                for i in [len, len + 1, len + PAGE, len.next_multiple_of(PAGE) + 64 * PAGE, usize::MAX] {
                    assert!(table.get(i, |held| held.is_none()), "slot {i} past {len}");
                    assert!(table.with(i, |piece, at| piece.slot(at, true).is_none()), "slot {i} past {len}");
                }
                let mut seen = Vec::new();
                table.each(|i, &held| {
                    assert_eq!(i, held);
                    seen.push(i);
                });
                if !table.split() {
                    assert!(seen.is_sorted(), "one piece walks in order");
                }
                seen.sort_unstable();
                assert_eq!(seen, picks);
            }
        }
    }

    /// Slots of one page are in one piece, neighbouring pages in different
    /// ones.
    #[test]
    fn a_page_of_slots_shares_a_piece() {
        let split = Pieces::<u8>::new(1000 * PAGE, true);
        for page in [0, 1, 63, 64, 65, 999] {
            let piece = split.place(page * PAGE).0;
            assert!((0..PAGE).all(|i| split.place(page * PAGE + i).0 == piece));
            assert_ne!(split.place((page + 1) * PAGE % (1000 * PAGE)).0, piece);
        }
    }

    #[test]
    fn a_wait_is_counted_only_with_the_meter_and_only_when_the_lock_was_held() {
        let lock = Arc::new(Mutex::new(0u32));
        // Free: no wait counted either way. (The counters are the process's:
        // this test reads only the census's, which nothing else here takes.)
        drop(locked_as(&lock, CENSUS, true, false));
        drop(locked_as(&lock, CENSUS, false, true));
        assert_eq!(WAITS[CENSUS * 2 + 1][0].load(Relaxed), 0);
        let wait = |meter: bool, syncing: bool| {
            let held = lock.lock().unwrap();
            let other = {
                let lock = lock.clone();
                std::thread::spawn(move || {
                    let _syncing = syncing.then(Syncing::enter);
                    *locked_as(&lock, CENSUS, meter, meter) += 1;
                })
            };
            std::thread::sleep(std::time::Duration::from_millis(30));
            drop(held);
            other.join().unwrap();
        };
        wait(false, false);
        assert_eq!(WAITS[CENSUS * 2 + 1][0].load(Relaxed), 0, "not counted without the meter");
        wait(true, false);
        wait(true, true);
        let (waits, _) = take(1.0);
        assert_eq!((waits[CENSUS * 2][0], waits[CENSUS * 2 + 1][0]), (1.0, 1.0), "one wait by a sync, one by another");
        assert!(waits[CENSUS * 2][1] >= 10.0 && waits[CENSUS * 2 + 1][1] >= 10.0, "their time in ms: {waits:?}");
        assert_eq!(take(1.0).0[CENSUS * 2], [0.0, 0.0], "taken once");
        assert_eq!(*lock.lock().unwrap(), 3);
    }

    /// A model of a host heap under several threads, in both forms: title
    /// threads rewrite records, makers bring slots up to date as
    /// `HostHeap::mirror` does (the record read without the lock, read again
    /// under the slot's lock, the slot's two words written, its holding
    /// replaced and the old one retired), readers look at slots under their
    /// lock and at records without it. No slot is lost (after a last pass
    /// every slot shows its record), no slot is torn (a reader under the
    /// lock never sees the words of two records, nor a holding of another
    /// record than the words'), and everything a slot ever held is retired
    /// exactly once.
    #[test]
    fn slots_made_by_several_threads_are_neither_lost_nor_torn() {
        const SLOTS: u32 = 3000;
        const MAKERS: usize = 6;
        const ROUNDS: u32 = 40_000;
        struct Heap {
            /// The title's records.
            records: Vec<AtomicU64>,
            made: Made,
            /// The host slot's descriptor, as two words written one after
            /// the other: the record and its complement.
            words: Vec<[AtomicU64; 2]>,
            kept: Pieces<Arc<u64>>,
            retired: Mutex<Vec<Arc<u64>>>,
            holdings: AtomicUsize,
        }
        impl Heap {
            fn keep(&self, index: u32, holding: Option<Arc<u64>>) {
                let old = self.kept.with(index as usize, |piece, at| {
                    let slot = piece.slot(at, holding.is_some())?;
                    let old = std::mem::replace(slot, holding);
                    if self.kept.split() {
                        return old;
                    }
                    self.retired.lock().unwrap().extend(old);
                    None
                });
                self.retired.lock().unwrap().extend(old);
            }

            /// `HostHeap::mirror`'s shape.
            fn mirror(&self, index: u32) {
                let record = self.records[index as usize].load(Acquire);
                if self.made.peek(index) == Some(record) {
                    return;
                }
                let mut slot = self.made.lock(index);
                // The title may have written another record meanwhile.
                let record = self.records[index as usize].load(Acquire);
                if slot.get(index) == record {
                    return;
                }
                slot.set(index, record);
                self.words[index as usize][0].store(record, Relaxed);
                std::hint::spin_loop();
                self.words[index as usize][1].store(!record, Relaxed);
                self.holdings.fetch_add(usize::from(record != 0), Relaxed);
                self.keep(index, (record != 0).then(|| Arc::new(record)));
            }

            /// A look at a slot under its lock: its record, words and
            /// holding agree.
            fn check(&self, index: u32) {
                let slot = self.made.lock(index);
                let key = slot.get(index);
                let [a, b] = [0, 1].map(|w| self.words[index as usize][w].load(Relaxed));
                assert_eq!((a, b), (key, if key == 0 && b == 0 { 0 } else { !key }), "slot {index} torn");
                let held = self.kept.get(index as usize, |held| held.map(|h| **h));
                assert_eq!(held, (key != 0).then_some(key), "slot {index} holds another record's");
            }
        }
        for split in [false, true] {
            let heap = Arc::new(Heap {
                records: (0..SLOTS).map(|_| AtomicU64::new(0)).collect(),
                made: Made::new(SLOTS as usize, split),
                words: (0..SLOTS).map(|_| [AtomicU64::new(0), AtomicU64::new(0)]).collect(),
                kept: Pieces::new(SLOTS as usize, split),
                retired: Mutex::new(Vec::new()),
                holdings: AtomicUsize::new(0),
            });
            let threads: Vec<_> = (0..MAKERS)
                .map(|t| {
                    let heap = heap.clone();
                    std::thread::spawn(move || {
                        // A small generator of its own: the same slots are
                        // met by several threads, in runs as tables are.
                        let mut x = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1);
                        let mut next = || {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x
                        };
                        for round in 0..ROUNDS {
                            let index = (next() % u64::from(SLOTS)) as u32;
                            match t % 3 {
                                // A title thread: a new record (now and then none), made at once.
                                0 => {
                                    let record = if round % 17 == 0 { 0 } else { next() | 1 << 32 };
                                    heap.records[index as usize].store(record, Release);
                                    heap.mirror(index);
                                }
                                // A heap sync: a run of slots brought up to date.
                                1 => (index..(index + 8).min(SLOTS)).for_each(|i| heap.mirror(i)),
                                // A reader.
                                _ => {
                                    heap.check(index);
                                    let _ = heap.made.peek(index);
                                }
                            }
                        }
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            // The last sync, with the title's threads at rest.
            (0..SLOTS).for_each(|i| heap.mirror(i));
            let mut held = 0;
            for i in 0..SLOTS {
                heap.check(i);
                assert_eq!(heap.made.peek(i), Some(heap.records[i as usize].load(Acquire)), "slot {i} lost (split {split})");
                held += usize::from(heap.kept.get(i as usize, |h| h.is_some()));
            }
            let retired = heap.retired.lock().unwrap();
            assert_eq!(retired.len() + held, heap.holdings.load(Relaxed), "retired once each (split {split})");
            assert!(retired.iter().all(|r| Arc::strong_count(r) == 1), "nothing retired is still held");
        }
    }
}
