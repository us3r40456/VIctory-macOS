//! What a commit of title memory does to the aliased regions
//! (`crate::alias`, `VICTORY_ALIAS_BUFFERS`), decided apart from the Mach
//! calls so the rules run as native tests.
//!
//! The title frees and decommits memory through the host's own `VirtualFree`,
//! which the runtime does not see. Under Wine both replace the pages at those
//! addresses with fresh anonymous memory, so an alias there is broken while
//! its registry entry stays: title buffers found in it would use the old host
//! buffer, and the title's new bytes would never reach the GPU. A commit
//! through `XMemVirtualAlloc` is where such memory comes back, so that is
//! where it is put right.

/// The 64 KiB page aliases are made of (`crate::watch::PAGE`).
const PAGE: usize = 64 << 10;

/// What a commit of title memory does to the aliases.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Aliased regions (their starts) the commit shows to be stale: a new
    /// reservation over them means the title released their memory.
    pub(crate) stale: Vec<usize>,
    /// Runs of aliased regions that were not committed before the commit
    /// (decommitted by the title): pointed at their buffers' pages again,
    /// zeroed, as recommitted memory reads.
    pub(crate) again: Vec<(usize, usize)>,
    /// Whether the commit gets an alias of its own.
    pub(crate) new: bool,
}

/// The plan for a commit of `[at, at + len)`.
///
/// - `fresh`: a new reservation (the title gave no address).
/// - `eligible`: committed, read-write, 64 KiB aligned and sized, not a
///   placeholder (VIctory-macOS #207's rule for an alias of its own).
/// - `uncommitted`: the runs of the range that were not committed before
///   the call, in order (for a fresh reservation, all of it).
/// - `aliases`: the aliased regions `(start, end)` overlapping the range.
///
/// A commit at a given address over memory that was committed already gets
/// no alias: the pages hold the title's bytes, which a remap would drop (or
/// race, copied first, with the title's other threads).
pub(crate) fn plan(
    at: usize,
    len: usize,
    fresh: bool,
    eligible: bool,
    uncommitted: &[(usize, usize)],
    aliases: &[(usize, usize)],
) -> Plan {
    let end = at.saturating_add(len);
    if fresh {
        return Plan {
            stale: aliases
                .iter()
                .filter(|&&(s, e)| s < end && at < e)
                .map(|&(s, _)| s)
                .collect(),
            again: Vec::new(),
            new: eligible,
        };
    }
    let mut again = Vec::new();
    for &(a, b) in uncommitted {
        for &(s, e) in aliases {
            // Whole pages only: the remap and the title both work in them.
            let (from, to) = (
                a.max(s).max(at).next_multiple_of(PAGE),
                b.min(e).min(end) / PAGE * PAGE,
            );
            if from < to {
                again.push((from, to));
            }
        }
    }
    again.sort_unstable();
    let whole_was_free = matches!(uncommitted, [(a, b)] if *a <= at && end <= *b);
    Plan {
        stale: Vec::new(),
        again,
        new: eligible && aliases.is_empty() && whole_was_free,
    }
}

/// Whether a fresh commit of `len` bytes should be aliased under optional
/// per-region minimum (`min_bytes`), per-region maximum (`max_bytes`), and
/// total wired-memory budget (`budget_bytes`) caps.
///
/// Used on unified-memory Macs (`VICTORY_ALIAS_MIN_MB`,
/// `VICTORY_ALIAS_MAX_MB`, `VICTORY_ALIAS_BUDGET_MB`) to alias CPU-written
/// heaps while optionally keeping giant GPU-written UAV heaps (> max) in
/// private compressed host GPU memory.
pub(crate) fn region_within_limits(
    len: usize,
    current_aliased: usize,
    min_bytes: Option<usize>,
    max_bytes: Option<usize>,
    budget_bytes: Option<usize>,
) -> bool {
    if min_bytes.is_some_and(|min| len < min) {
        return false;
    }
    if max_bytes.is_some_and(|max| len > max) {
        return false;
    }
    if budget_bytes.is_some_and(|budget| current_aliased.saturating_add(len) > budget) {
        return false;
    }
    true
}

/// Parses an integer MiB threshold from an environment variable string.
/// Values of `0` or invalid strings return `None` (no limit).
pub(crate) fn parse_mib(raw: &str) -> Option<usize> {
    let mb: usize = raw.trim().parse().ok()?;
    (mb > 0).then(|| mb.saturating_mul(1 << 20))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: usize = 0x1_0000_0000;

    #[test]
    fn a_fresh_commit_gets_an_alias() {
        let p = plan(A, 4 * PAGE, true, true, &[], &[]);
        assert_eq!(
            p,
            Plan {
                new: true,
                ..Plan::default()
            }
        );
        let p = plan(A, 4 * PAGE, true, false, &[], &[]);
        assert_eq!(p, Plan::default(), "not eligible: nothing");
    }

    #[test]
    fn a_fresh_reservation_over_an_alias_shows_it_stale() {
        // The title released the region and the host handed the addresses
        // out again: the old alias goes, with or without a commit now.
        let old = [(A - PAGE, A + PAGE), (A + 8 * PAGE, A + 9 * PAGE)];
        let p = plan(A, 2 * PAGE, true, true, &[], &old);
        assert_eq!(p.stale, vec![A - PAGE]);
        assert!(p.new);
        let p = plan(A, 16 * PAGE, true, false, &[], &old);
        assert_eq!(p.stale, vec![A - PAGE, A + 8 * PAGE]);
        assert!(!p.new, "a reservation only: no alias of its own");
        let p = plan(A + PAGE, PAGE, true, true, &[], &old);
        assert!(p.stale.is_empty(), "touching is not overlapping");
    }

    #[test]
    fn a_commit_into_a_reservation_gets_an_alias_only_over_fresh_pages() {
        let all = [(A, A + 2 * PAGE)];
        assert!(
            plan(A, 2 * PAGE, false, true, &all, &[]).new,
            "all of it was reserved"
        );
        assert!(
            !plan(A, 2 * PAGE, false, true, &[(A + PAGE, A + 2 * PAGE)], &[]).new,
            "its first page held the title's bytes"
        );
        assert!(
            !plan(A, 2 * PAGE, false, true, &[], &[]).new,
            "committed already"
        );
    }

    #[test]
    fn a_recommit_inside_an_alias_points_the_pages_back_at_its_buffer() {
        // Aliased [A, A + 8 pages); the title decommitted pages 2 and 3 and
        // now commits pages 1 to 3 again.
        let alias = [(A, A + 8 * PAGE)];
        let p = plan(
            A + PAGE,
            3 * PAGE,
            false,
            true,
            &[(A + 2 * PAGE, A + 4 * PAGE)],
            &alias,
        );
        assert_eq!(p.again, vec![(A + 2 * PAGE, A + 4 * PAGE)]);
        assert!(!p.new, "inside an alias: no second one");
        assert!(p.stale.is_empty());
        // Still committed (a commit of committed pages): nothing to do.
        assert_eq!(plan(A, 8 * PAGE, false, true, &[], &alias), Plan::default());
    }

    #[test]
    fn a_recommit_over_two_aliases_and_plain_memory_is_cut_to_the_aliases() {
        let aliases = [(A, A + 2 * PAGE), (A + 4 * PAGE, A + 6 * PAGE)];
        let p = plan(A, 8 * PAGE, false, true, &[(A, A + 8 * PAGE)], &aliases);
        assert_eq!(
            p.again,
            vec![(A, A + 2 * PAGE), (A + 4 * PAGE, A + 6 * PAGE)]
        );
        assert!(!p.new, "it overlaps aliases: the rest stays plain memory");
    }

    #[test]
    fn only_whole_pages_are_pointed_back() {
        let alias = [(A, A + 4 * PAGE)];
        let p = plan(
            A,
            4 * PAGE,
            false,
            true,
            &[(A + 100, A + 2 * PAGE + 100)],
            &alias,
        );
        assert_eq!(p.again, vec![(A + PAGE, A + 2 * PAGE)]);
    }

    #[test]
    fn selective_alias_limits_respect_min_and_budget() {
        const MB: usize = 1 << 20;
        assert_eq!(parse_mib("64"), Some(64 * MB));
        assert_eq!(parse_mib(" 512 "), Some(512 * MB));
        assert_eq!(parse_mib("0"), None);
        assert_eq!(parse_mib("abc"), None);

        // No limits: everything passes.
        assert!(region_within_limits(PAGE, 0, None, None, None));

        // Min size filters out small heaps while allowing the 174 MiB constants heap.
        assert!(!region_within_limits(4 * MB, 0, Some(32 * MB), None, None));
        assert!(region_within_limits(32 * MB, 0, Some(32 * MB), None, None));
        assert!(region_within_limits(174 * MB, 0, Some(32 * MB), None, None));

        // Max size excludes giant GPU-written UAV heaps (> 512 MiB) while allowing CPU heaps.
        assert!(region_within_limits(258 * MB, 0, None, Some(512 * MB), None));
        assert!(!region_within_limits(1162 * MB, 0, None, Some(512 * MB), None));

        // Total budget caps cumulative aliased bytes.
        assert!(region_within_limits(
            174 * MB,
            300 * MB,
            Some(32 * MB),
            Some(512 * MB),
            Some(512 * MB)
        ));
        assert!(!region_within_limits(
            174 * MB,
            400 * MB,
            Some(32 * MB),
            Some(512 * MB),
            Some(512 * MB)
        ));
    }
}
