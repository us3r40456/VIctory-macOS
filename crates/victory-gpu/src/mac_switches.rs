//! The Mac line's translation switches (`victory_shader::emit::Options`,
//! `lower::mac` there): read once from the environment, and part of the
//! shader cache key ([`key`], used by `shader_cache`), so translations made
//! with and without one never mix. The rules are here, outside the
//! Windows-only shader cache, so that their tests run on any host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::OnceLock;

/// Whether pixel programs that ask for the depth test before the shader
/// get DXIL's early depth-stencil flag
/// (`victory_shader::emit::Options::early_depth`): by default under Wine
/// (`wine`), where the title's see-through pre-pass is the cost of a frame
/// and hidden fragments ran it. `VICTORY_EARLY_DEPTH=0` or `1` overrides.
pub(crate) fn early_depth(wine: bool) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_EARLY_DEPTH").ok().as_deref(), wine);
        tracing::info!(
            on,
            "VICTORY_EARLY_DEPTH: pixel programs that ask for the depth test first get the early depth-stencil flag"
        );
        on
    })
}

/// Which lanes stay out of the lock loop of a 64-bit image atomic
/// (`victory_shader::emit::Options::atomic_lock_helpers` and
/// `atomic_lock_keys`; they act only where atomic locks are on): by default
/// under Wine (`wine`) both the helper lanes of pixel programs and the lanes
/// whose `umax` key is below the texel's. `VICTORY_ATOMIC_LOCK_WRITERS=0`:
/// every lane takes the lock, as before; `helpers`: only helper lanes stay
/// out; `1`: both, anywhere.
pub(crate) fn lock_writers(wine: bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_ATOMIC_LOCK_WRITERS").ok();
        let [helpers, keys] = writers(value.as_deref(), wine);
        tracing::info!(
            helpers,
            keys,
            "VICTORY_ATOMIC_LOCK_WRITERS: lanes that cannot change the texel stay out of the lock loop of 64-bit image atomics"
        );
        [helpers, keys]
    })
}

/// Which parts of the quicker heap sync are on (`crate::descriptor::sync_heaps`
/// makes 25,000 to 40,000 host descriptor slots again in a world frame, on
/// the submitting thread): the heap census in hash maps
/// (`crate::census_maps`), and buffer shader views made from one lookup
/// (`crate::descriptor::host`, "quick"). By default both under Wine (`wine`
/// says whether; asked once), where a frame's CPU part is the cost.
/// `VICTORY_QUICK_SYNC=0`: neither, as before; `maps` or `views`: that part
/// alone; `1`: both, anywhere. Neither changes what a slot shows.
pub(crate) fn quick_sync(wine: impl FnOnce() -> bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_QUICK_SYNC").ok();
        let [maps, views] = quick(value.as_deref(), wine());
        tracing::info!(
            maps,
            views,
            "VICTORY_QUICK_SYNC: the heap census in hash maps, buffer shader views of a heap sync made from one lookup"
        );
        [maps, views]
    })
}

/// [`quick_sync`]'s rule: census maps, views.
fn quick(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("maps") => [true, false],
        Some("views") => [false, true],
        value => [setting(value, default); 2],
    }
}

/// Whether a compute group of one console wave that shares LDS in a way
/// barriers at its common blocks cannot order by themselves gets the finer
/// order of `victory_shader::emit::lower::lds_order`
/// (`victory_shader::emit::Options::lds_order`; `notes/cross-half-programs.md`,
/// "Shared LDS"): only with `VICTORY_HALVES_LDS_ORDER=1`. Exact where it
/// applies; a program it cannot order stays refused, or runs unsynced with
/// `VICTORY_HALVES_SHARED_LDS=1`.
pub(crate) fn lds_order() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_LDS_ORDER").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_LDS_ORDER=1: one-wave64 groups that share LDS get group barriers where every lane runs, when that orders the LDS exactly"
            );
        }
        on
    })
}

/// Whether a lane read across the two 32-lane halves of a console wave64,
/// in a block every lane runs once, is handed over through group-shared
/// memory at a group barrier (`victory_shader::emit::lower::exchange`,
/// `victory_shader::emit::Options::halves_exchange`;
/// `notes/ray-tracing-plan.md`): only with `VICTORY_HALVES_EXCHANGE=1`.
/// Exact where it applies; a read it does not cover stays refused, or is
/// answered by the reader's half with `VICTORY_HALVES_LOCAL=1`.
pub(crate) fn exchange() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_EXCHANGE").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_EXCHANGE=1: lane reads across 32-lane halves are handed over through group-shared memory where every lane runs"
            );
        }
        on
    })
}

/// Whether, in a program in halves with half-local reads
/// (`VICTORY_HALVES_LOCAL=1`), a read across the halves is handed over the
/// way VIctory-macOS does it (its #309, #360 and #376;
/// `victory_shader::emit::Options::halves_across`) before it is answered
/// from the reader's own half: only with `VICTORY_HALVES_ACROSS=1`. This
/// line's own exchange (`VICTORY_HALVES_EXCHANGE=1`) still comes first.
/// Unset, a translation with half-local reads is what it was before that
/// hand-over was merged; without half-local reads the hand-over is always
/// made, as in VIctory-macOS, where the read would otherwise be refused.
pub(crate) fn across() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_ACROSS").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_ACROSS=1: reads across 32-lane halves are handed over as in VIctory-macOS where every lane of the group runs together, before half-local reads answer them"
            );
        }
        on
    })
}

/// Whether a size query on a writable image asks for level 0 in every
/// program (`victory_shader::emit::Options::uav_size_level0`): only with
/// `VICTORY_UAV_SIZE_LEVEL0=1`. D3DMetal answers the undefined level DXC
/// writes with a size of 0 (measured: `uav_size_tests` in `victory-shader`),
/// so the title's 35 programs that compute with such a size are wrong
/// without it (`notes/uav-size.md`). Off by default until the picture with
/// it is checked. A list of shader hashes instead of `1` turns it on for
/// those programs only ([`uav_size_level0_for`]).
pub(crate) fn uav_size_level0() -> bool {
    uav_size_rule().0
}

/// Whether the program in `container` gets the level-0 size query: with
/// `VICTORY_UAV_SIZE_LEVEL0=1`, or when the variable lists its hash (16
/// digits, as `shader translated` lines print it; several with commas), to
/// try the change on one program at a time.
pub(crate) fn uav_size_level0_for(container: &[u8]) -> bool {
    let (all, only) = uav_size_rule();
    *all || (!only.is_empty() && only.contains(&blake3::hash(container).to_hex()[..16].to_owned()))
}

fn uav_size_rule() -> &'static (bool, Vec<String>) {
    static RULE: OnceLock<(bool, Vec<String>)> = OnceLock::new();
    RULE.get_or_init(|| {
        let rule = uav_size(std::env::var("VICTORY_UAV_SIZE_LEVEL0").ok().as_deref());
        if rule.0 {
            tracing::info!("VICTORY_UAV_SIZE_LEVEL0=1: size queries on writable images ask for level 0");
        } else if !rule.1.is_empty() {
            tracing::info!(
                programs = ?rule.1,
                "VICTORY_UAV_SIZE_LEVEL0: size queries on writable images ask for level 0 in these programs only"
            );
        }
        rule
    })
}

/// `VICTORY_UAV_SIZE_LEVEL0`'s rule: `1` is every program; a list of
/// 16-digit shader hashes is those programs; anything else is off.
fn uav_size(value: Option<&str>) -> (bool, Vec<String>) {
    let value = value.unwrap_or_default().trim();
    if value == "1" {
        return (true, Vec::new());
    }
    let hashes: Vec<String> = value.split(',').map(|h| h.trim().to_ascii_lowercase()).collect();
    if hashes.iter().all(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit())) {
        (false, hashes)
    } else {
        (false, Vec::new())
    }
}

/// Whether a console wave64 program whose hand-overs across the halves
/// sit in code only some lanes run is translated with those skips left
/// out, every lane walking the skipped code with its EXEC bit clear
/// (`victory_shader::emit::walk`, `victory_shader::emit::Options::halves_walk`;
/// `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs"): only with
/// `VICTORY_HALVES_WALK=1`, and only together with
/// `VICTORY_HALVES_EXCHANGE=1` (alone it changes nothing, and is then left
/// out of the cache key too). Exact where it applies; a program it does
/// not make exact is translated as without it.
pub(crate) fn walk() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_WALK").ok().as_deref(), false);
        if on && exchange() {
            tracing::info!(
                "VICTORY_HALVES_WALK=1: every lane walks the skips around a hand-over across 32-lane halves, where that makes the program exact"
            );
        } else if on {
            tracing::warn!("VICTORY_HALVES_WALK=1 does nothing without VICTORY_HALVES_EXCHANGE=1");
        }
        on && exchange()
    })
}

/// Whether a title copy recorded in a list keeps its order with the
/// commands before it that read what it changes, and the longest read
/// range looked at: only with `VICTORY_COPY_ORDER=split` (every range) or
/// `split:<bytes>` (ranges of at most so many bytes).
///
/// On the console a list's copies (`CopyBufferRegion`, buffer
/// `CopyResource`, `CopyMemoryToMemoryX`) and its draws and dispatches run
/// in their order. Here every copy of a host submission is done on the CPU
/// before its first GPU command, into the one host copy of each buffer, so
/// a command recorded before such a copy ran with the bytes meant for the
/// commands after it (the title's TAA passes worked on a quarter of the
/// screen, `notes/uav-size.md`, "Copy order"). With the switch the title
/// list is cut at such a copy (`crate::list`) and a queue's piece is split
/// there into two host submissions (`crate::host::copy_order_parts`): the
/// first has run on the GPU before the copy is made. `0`, `keep` and
/// anything else leave it off (the diagnostics of `crate::host` read the
/// same variable).
pub(crate) fn copy_order_split() -> Option<usize> {
    #[cfg(test)]
    if tests::FORCED_COPY_ORDER.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        return Some(usize::MAX);
    }
    static RULE: OnceLock<Option<usize>> = OnceLock::new();
    *RULE.get_or_init(|| {
        let rule = copy_order(std::env::var("VICTORY_COPY_ORDER").ok().as_deref());
        match rule {
            Some(usize::MAX) => tracing::info!(
                "VICTORY_COPY_ORDER=split: a title copy that changes what earlier commands of its submission read starts a new host submission"
            ),
            Some(bytes) => tracing::info!(
                bytes,
                "VICTORY_COPY_ORDER=split:<bytes>: a title copy that changes a range of at most so many bytes that earlier commands of its submission read starts a new host submission"
            ),
            None => {}
        }
        rule
    })
}

/// `VICTORY_COPY_ORDER`'s rule for the split: `split` is every read range,
/// `split:<bytes>` those of at most `bytes` (1 or more); anything else is
/// off.
fn copy_order(value: Option<&str>) -> Option<usize> {
    match value?.trim().strip_prefix("split")? {
        "" => Some(usize::MAX),
        bound => bound.strip_prefix(':')?.parse().ok().filter(|&bytes| bytes > 0),
    }
}

/// How the clamps of indirect draws are recorded (`crate::indirect`):
/// whether the clamps of a run of indirect draws go to the host list before
/// the one that draws, and whether a run's clamps are then one dispatch.
/// Both only with `VICTORY_BATCH_CLAMPS=1`; `hoist` moves the clamps and
/// leaves each a dispatch of its own (to tell the two steps apart in a
/// run). Off by default: each clamp is a dispatch of ours just before its
/// draw, about 4,200 a world frame, each between two draws.
pub(crate) fn batch_clamps() -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let [hoist, batch] = clamps(std::env::var("VICTORY_BATCH_CLAMPS").ok().as_deref());
        if hoist {
            tracing::info!(
                batch,
                "VICTORY_BATCH_CLAMPS: the clamps of a run of indirect draws go to the host list before it (batch: as one dispatch a run)"
            );
        }
        [hoist, batch]
    })
}

/// `VICTORY_BATCH_CLAMPS`'s rule: hoisted, batched. `1` is both, `hoist`
/// the first alone; anything else is off.
fn clamps(value: Option<&str>) -> [bool; 2] {
    match value.map(str::trim) {
        Some("1") => [true, true],
        Some("hoist") => [true, false],
        _ => [false, false],
    }
}

/// Whether the clamps of indirect dispatches go to the host list before
/// too, as those of draws do with `VICTORY_BATCH_CLAMPS` (`crate::indirect`,
/// `crate::list`): `VICTORY_DISPATCH_CLAMPS=1`, and only where
/// `VICTORY_BATCH_CLAMPS` hoists (`1` batches them, `hoist` leaves each a
/// dispatch of its own). A run of them ends at the title's next barrier,
/// pipeline sync, copy, clear or immediate write, not at its dispatches.
/// Off by default: each indirect dispatch has a clamp of ours just before
/// it, about 960 a frame at the prison.
pub(crate) fn dispatch_clamps() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = dispatch_rule(std::env::var("VICTORY_DISPATCH_CLAMPS").ok().as_deref());
        let on = asked && batch_clamps()[0];
        if on {
            tracing::info!(
                batch = batch_clamps()[1],
                "VICTORY_DISPATCH_CLAMPS: the clamps of a run of indirect dispatches go to the host list before it too (a run ends at the title's next barrier)"
            );
        } else if asked {
            tracing::warn!("VICTORY_DISPATCH_CLAMPS=1 does nothing without VICTORY_BATCH_CLAMPS=1 or hoist");
        }
        on
    })
}

/// `VICTORY_DISPATCH_CLAMPS`'s rule: on only for `1`.
fn dispatch_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// Whether view heaps' censuses (`crate::bindless::Viewed`) take their lock
/// less (`crate::census_fast`): the ranges a heap's unordered views write
/// are kept apart and their answer kept until they change, so a list's
/// heap-writing draws no longer walk every range of the census under its
/// lock at each barrier, and a heap sync enters the slots of a pass a batch
/// at a time, not one lock a slot. Only with `VICTORY_CENSUS_FAST=1`, read
/// once (at the first view heap). What a slot shows, what a submission
/// refreshes and what widened views get stay as they are.
pub(crate) fn census_fast() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = census_rule(std::env::var("VICTORY_CENSUS_FAST").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_CENSUS_FAST=1: view heap censuses keep their unordered ranges apart with a kept answer, and heap syncs enter their slots in batches"
            );
        }
        on
    })
}

/// `VICTORY_CENSUS_FAST`'s rule: on only for `1`.
fn census_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

#[cfg(test)]
mod census_tests {
    use super::census_rule;

    #[test]
    fn the_census_switch_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("true"), Some("2")] {
            assert!(!census_rule(off), "{off:?}");
        }
        assert!(census_rule(Some("1")));
        assert!(census_rule(Some(" 1 ")));
    }
}

#[cfg(test)]
mod clamp_tests {
    use super::{clamps, dispatch_rule};

    #[test]
    fn the_clamp_switch_is_off_unless_it_is_1_or_hoist() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("batch"), Some("2")] {
            assert_eq!(clamps(off), [false, false], "{off:?}");
        }
        assert_eq!(clamps(Some("1")), [true, true]);
        assert_eq!(clamps(Some(" 1 ")), [true, true]);
        assert_eq!(clamps(Some("hoist")), [true, false]);
    }

    #[test]
    fn the_dispatch_clamp_switch_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("hoist"), Some("2")] {
            assert!(!dispatch_rule(off), "{off:?}");
        }
        assert!(dispatch_rule(Some("1")));
        assert!(dispatch_rule(Some(" 1 ")));
    }
}

/// [`lock_writers`]' rule: helper lanes, low keys.
fn writers(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("helpers") => [true, false],
        value => [setting(value, default); 2],
    }
}

/// `VICTORY_ASYNC_HEAP_SYNC`'s rule (`crate::descriptor::ahead`): off when
/// unset, empty or `0`; else on, and the milliseconds between a round of
/// the heap sync and the next while the GPU still runs: 8 for `1` (or
/// anything that is no number), a larger number for itself.
pub(crate) fn heap_sync_pause(value: Option<&str>) -> Option<u64> {
    match value {
        None | Some("" | "0") => None,
        Some(v) => Some(v.parse::<u64>().ok().filter(|&ms| ms > 1).unwrap_or(8)),
    }
}

/// `VICTORY_HEAP_LOCKS`'s rule (`crate::descriptor::slots`): whether a host
/// heap's slots get locks of their own, and whether lock waits and made
/// slots are counted. Unset, empty or `0`: neither, as before; `count`:
/// the one lock as before, counted; anything else: both.
pub(crate) fn heap_locks(value: Option<&str>) -> [bool; 2] {
    match value {
        None | Some("" | "0") => [false, false],
        Some("count") => [false, true],
        Some(_) => [true, true],
    }
}

/// `VICTORY_HEAP_SYNC_THREADS`'s rule (`crate::descriptor::pool`): the
/// threads a heap sync makes a pass's changed slots on, its own among them;
/// 1 is the syncing thread alone, as before. Unset, no number, 0 or 1: 1;
/// a larger number: itself, `most` at most. Always 1 unless the heap's
/// slots have locks of their own (`split`, `VICTORY_HEAP_LOCKS=1`).
pub(crate) fn heap_sync_threads(value: Option<&str>, split: bool, most: usize) -> usize {
    let asked = value.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1);
    if split { asked.clamp(1, most.max(1)) } else { 1 }
}

/// A switch's state: `0` off, `1` on, anything else (and unset) `default`.
fn setting(value: Option<&str>, default: bool) -> bool {
    match value {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

/// The cache key `build` for translations made with these settings. A
/// setting that is off adds nothing, so the entries made without it stay
/// the ones read.
pub(crate) fn key(
    build: &[u8; 32],
    early_depth: bool,
    lock_writers: [bool; 2],
    lds_order: bool,
    exchange: bool,
    uav_size_level0: bool,
    walk: bool,
) -> [u8; 32] {
    if !early_depth
        && lock_writers == [false; 2]
        && !lds_order
        && !exchange
        && !uav_size_level0
        && !walk
    {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    if early_depth {
        h.update(b"early depth");
    }
    if lock_writers[0] {
        h.update(b"atomic lock helpers");
    }
    if lock_writers[1] {
        h.update(b"atomic lock keys");
    }
    if lds_order {
        h.update(b"lds order");
    }
    if exchange {
        h.update(b"halves exchange");
    }
    if uav_size_level0 {
        h.update(b"uav size level 0");
    }
    if walk {
        h.update(b"halves walk");
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{copy_order, heap_locks, heap_sync_pause, heap_sync_threads, key, quick, setting, uav_size, writers};

    /// How many tests have the copy-order split forced on right now.
    pub(crate) static FORCED_COPY_ORDER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// While it lives, `copy_order_split` is on for the process, as with
    /// `VICTORY_COPY_ORDER=split` (other tests then only note what their
    /// lists read).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) struct ForcedCopyOrder;

    #[cfg_attr(not(windows), allow(dead_code))]
    impl ForcedCopyOrder {
        pub(crate) fn on() -> ForcedCopyOrder {
            FORCED_COPY_ORDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ForcedCopyOrder
        }
    }

    impl Drop for ForcedCopyOrder {
        fn drop(&mut self) {
            FORCED_COPY_ORDER.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn the_copy_order_switch_is_split_with_an_optional_bound() {
        for off in [None, Some(""), Some("0"), Some("1"), Some("keep"), Some("split:"), Some("split:0"), Some("split:x"), Some("splits")] {
            assert_eq!(copy_order(off), None, "{off:?}");
        }
        assert_eq!(copy_order(Some("split")), Some(usize::MAX));
        assert_eq!(copy_order(Some(" split ")), Some(usize::MAX));
        assert_eq!(copy_order(Some("split:256")), Some(256));
        assert_eq!(copy_order(Some("split:65536")), Some(65536));
    }

    #[test]
    fn the_size_query_switch_is_all_programs_or_a_list_of_hashes() {
        assert_eq!(uav_size(None), (false, vec![]));
        assert_eq!(uav_size(Some("")), (false, vec![]));
        assert_eq!(uav_size(Some("0")), (false, vec![]));
        assert_eq!(uav_size(Some("1")), (true, vec![]));
        assert_eq!(uav_size(Some("yes")), (false, vec![]));
        let two = vec!["07f451a14b13fecb".to_owned(), "11eeb27c2d15980d".to_owned()];
        assert_eq!(uav_size(Some("07F451A14B13FECB, 11eeb27c2d15980d")), (false, two));
        // A list with a wrong entry is no list: nothing is turned on.
        assert_eq!(uav_size(Some("07f451a14b13fecb,1")), (false, vec![]));
    }

    #[test]
    fn the_heap_sync_ahead_is_off_unless_asked_and_takes_its_pause() {
        assert_eq!(heap_sync_pause(None), None);
        assert_eq!(heap_sync_pause(Some("")), None);
        assert_eq!(heap_sync_pause(Some("0")), None);
        assert_eq!(heap_sync_pause(Some("1")), Some(8));
        assert_eq!(heap_sync_pause(Some("yes")), Some(8));
        assert_eq!(heap_sync_pause(Some("4")), Some(4));
        assert_eq!(heap_sync_pause(Some("1000")), Some(1000));
    }

    #[test]
    fn the_heap_locks_are_one_and_uncounted_unless_asked() {
        assert_eq!(heap_locks(None), [false, false]);
        assert_eq!(heap_locks(Some("")), [false, false]);
        assert_eq!(heap_locks(Some("0")), [false, false]);
        assert_eq!(heap_locks(Some("count")), [false, true]);
        assert_eq!(heap_locks(Some("1")), [true, true]);
        assert_eq!(heap_locks(Some("yes")), [true, true]);
    }

    #[test]
    fn a_heap_sync_has_one_thread_unless_asked_and_the_locks_are_split() {
        for value in [None, Some(""), Some("0"), Some("1"), Some("yes"), Some("-3")] {
            assert_eq!(heap_sync_threads(value, true, 8), 1, "{value:?}");
        }
        assert_eq!(heap_sync_threads(Some("2"), true, 8), 2);
        assert_eq!(heap_sync_threads(Some(" 4 "), true, 8), 4);
        assert_eq!(heap_sync_threads(Some("64"), true, 8), 8);
        for value in [None, Some("1"), Some("4"), Some("64")] {
            assert_eq!(heap_sync_threads(value, false, 8), 1, "{value:?} without the split locks");
        }
    }

    #[test]
    fn the_quick_sync_switch_names_its_parts() {
        for default in [false, true] {
            assert_eq!(quick(Some("0"), default), [false, false]);
            assert_eq!(quick(Some("maps"), default), [true, false]);
            assert_eq!(quick(Some("views"), default), [false, true]);
            assert_eq!(quick(Some("1"), default), [true, true]);
            assert_eq!(quick(None, default), [default, default]);
            assert_eq!(quick(Some("both"), default), [default, default]);
        }
    }

    #[test]
    fn a_switch_is_off_at_0_on_at_1_and_else_the_default() {
        for default in [false, true] {
            assert!(!setting(Some("0"), default));
            assert!(setting(Some("1"), default));
            assert_eq!(setting(None, default), default);
            assert_eq!(setting(Some(""), default), default);
            assert_eq!(setting(Some("yes"), default), default);
        }
    }

    #[test]
    fn the_lock_writers_switch_has_three_states() {
        for default in [false, true] {
            assert_eq!(writers(Some("0"), default), [false, false]);
            assert_eq!(writers(Some("helpers"), default), [true, false]);
            assert_eq!(writers(Some("1"), default), [true, true]);
            assert_eq!(writers(None, default), [default, default]);
            assert_eq!(writers(Some("keys"), default), [default, default]);
        }
    }

    /// Every combination of the settings is a key of its own; all off is
    /// the key without them; early depth alone is what it was before the
    /// lock settings came, the three of them without the LDS order what
    /// they were before it came, and all four without the exchange what
    /// they were before that came; and every key without the walk is what
    /// it was before the walk came.
    #[test]
    fn each_combination_of_settings_has_cache_entries_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(
            key(&build, false, [false, false], false, false, false, false),
            build
        );
        let of = |build: &[u8; 32], n: u8| {
            key(
                build,
                n & 1 != 0,
                [n & 2 != 0, n & 4 != 0],
                n & 8 != 0,
                n & 16 != 0,
                n & 32 != 0,
                n & 64 != 0,
            )
        };
        let mut keys = std::collections::HashSet::new();
        for n in 0..128u8 {
            let k = of(&build, n);
            assert_eq!(k, of(&build, n));
            keys.insert(k);
            // Over another build (other settings, another translator).
            assert!(n == 0 || k != of(&[8u8; 32], n));
        }
        assert_eq!(keys.len(), 128);
        // The exchange and the level-0 size query, as before the walk.
        let mut two = blake3::Hasher::new();
        two.update(&build);
        two.update(b"halves exchange");
        two.update(b"uav size level 0");
        assert_eq!(of(&build, 48), *two.finalize().as_bytes());
        two = blake3::Hasher::new();
        two.update(&build);
        two.update(b"halves exchange");
        two.update(b"halves walk");
        assert_eq!(of(&build, 80), *two.finalize().as_bytes());
        let mut early = blake3::Hasher::new();
        early.update(&build);
        early.update(b"early depth");
        assert_eq!(
            key(&build, true, [false, false], false, false, false, false),
            *early.finalize().as_bytes()
        );
        let mut four = blake3::Hasher::new();
        four.update(&build);
        four.update(b"early depth");
        four.update(b"atomic lock helpers");
        four.update(b"atomic lock keys");
        four.update(b"lds order");
        assert_eq!(of(&build, 15), *four.finalize().as_bytes());
        let mut three = blake3::Hasher::new();
        three.update(&build);
        three.update(b"early depth");
        three.update(b"atomic lock helpers");
        three.update(b"atomic lock keys");
        assert_eq!(of(&build, 7), *three.finalize().as_bytes());
    }
}
