//! Passes over the screen get pipeline workers and a longer wait of their
//! own (from Belobug 18e57db, written against this line's queues in
//! [`super::make`]).
//!
//! The title fills its lighting tables with full-screen draws it makes
//! once, at the world's first frame. Queued behind a world's worth of other
//! pipelines, theirs were not made in time and the draws were dropped: on
//! the fork such a run had no diffuse light at all (grey, sepia image; white
//! bloom indoors).
//!
//! A pipeline counts as a screen pass when it has a vertex shader, no
//! vertex input, no depth or stencil test, and draws triangles. Such
//! pipelines go to [`WORKERS`] workers that make nothing else, and the title
//! may wait [`budget`] for them between two presents, counted from the first
//! such wait, in place of the half second all other pipelines share.
//!
//! macOS port, three differences from the fork:
//! - The wait is a time on the clock between two presents, not a sum of
//!   waits per frame number. Two recording threads that wait at the same
//!   time, or a frame number that moves on during a wait, cannot add up to
//!   more than the budget, which is what keeps it under the title's 30 s
//!   watchdog.
//! - The default is 15 s, not 20: frames of this line take up to 7 s on
//!   their own (10.7 s seen once), where the fork's take milliseconds.
//! - The long wait is given once per backlog. When it runs out with a
//!   pipeline still not made (runs of this line have had compiles that
//!   never end), screen passes wait as the others do until every queued one
//!   is made: a stuck compile costs one long frame, not every frame.
//!
//! `VICTORY_SCREEN_PASS_WAIT=<seconds>` sets the wait (at most 25); `0`
//! turns all of this off: no pipeline is a screen pass then, and no extra
//! worker starts.
//!
//! shortcut: the budget and the worker count are guesses; a pass the title
//! runs once whose pipeline takes longer than the budget is still lost.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Workers that make screen passes and nothing else.
pub(super) const WORKERS: usize = 2;

const DEFAULT: Duration = Duration::from_secs(15);
/// The title's watchdog ends a run whose frame takes 30 s.
const LONGEST: Duration = Duration::from_secs(25);

/// How long the title may wait for screen passes between two presents;
/// `None` when `VICTORY_SCREEN_PASS_WAIT=0` turns them off.
pub(super) fn budget() -> Option<Duration> {
    static BUDGET: OnceLock<Option<Duration>> = OnceLock::new();
    *BUDGET.get_or_init(|| parse(std::env::var("VICTORY_SCREEN_PASS_WAIT").ok().as_deref()))
}

/// The budget a value of the variable stands for (whole seconds; anything
/// else is the default).
fn parse(value: Option<&str>) -> Option<Duration> {
    let wait = value
        .and_then(|v| v.trim().parse().ok())
        .map_or(DEFAULT, |s| Duration::from_secs(s).min(LONGEST));
    (!wait.is_zero()).then_some(wait)
}

/// Where the long wait stands.
struct Waits {
    /// The present count the running wait belongs to, and when it ends.
    window: Option<(u64, Instant)>,
    /// Screen passes queued and not made yet, and all queued so far.
    open: u64,
    total: u64,
    /// A wait ran out: no long wait until `open` is back to 0.
    late: bool,
}

impl Waits {
    /// How much longer a screen pass may be waited for after `presents`
    /// presents; `None` when it waits as the other pipelines do.
    fn left(&mut self, presents: u64, now: Instant, budget: Duration) -> Option<Duration> {
        if self.late {
            return None;
        }
        let end = match self.window {
            Some((p, end)) if p == presents => end,
            _ => {
                self.window = Some((presents, now + budget));
                now + budget
            }
        };
        let left = end.saturating_duration_since(now);
        (!left.is_zero()).then_some(left)
    }

    /// A wait ended without its pipeline; whether that is news.
    fn ran_out(&mut self) -> bool {
        // Made in the meantime: nothing is late.
        let news = !self.late && self.open > 0;
        self.late |= news;
        news
    }

    /// A screen pass is made; whether that ends a late spell.
    fn made(&mut self) -> bool {
        self.open = self.open.saturating_sub(1);
        self.open == 0 && std::mem::take(&mut self.late)
    }
}

static WAITS: Mutex<Waits> = Mutex::new(Waits {
    window: None,
    open: 0,
    total: 0,
    late: false,
});

fn waits() -> std::sync::MutexGuard<'static, Waits> {
    WAITS.lock().unwrap_or_else(|e| e.into_inner())
}

/// How much longer a screen pass may be waited for before the next present;
/// `None` when it waits as the other pipelines do.
pub(super) fn left(presents: u64) -> Option<Duration> {
    waits().left(presents, Instant::now(), budget()?)
}

/// A screen pass was queued.
pub(super) fn queued() {
    let mut waits = waits();
    waits.open += 1;
    waits.total += 1;
}

/// Screen passes queued so far (for the log).
pub(super) fn total() -> u64 {
    waits().total
}

/// A screen pass was made.
pub(super) fn made() {
    if waits().made() {
        tracing::info!("screen-pass pipelines all made: they have their long wait again");
    }
}

/// A wait for a screen pass ended without its pipeline.
pub(super) fn ran_out() {
    let mut waits = waits();
    if waits.ran_out() {
        tracing::warn!(
            being_made = waits.open,
            queued_so_far = waits.total,
            "the wait for screen-pass pipelines ran out: their draws are dropped, and they wait as \
             the others do until all are made (VICTORY_SCREEN_PASS_WAIT)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new() -> Waits {
        Waits {
            window: None,
            open: 0,
            total: 0,
            late: false,
        }
    }

    /// Unset or unreadable is the default, `0` is off, and no value goes
    /// past what the title's watchdog allows.
    #[test]
    fn the_variable_sets_the_wait() {
        assert_eq!(parse(None), Some(DEFAULT));
        assert_eq!(parse(Some("")), Some(DEFAULT));
        assert_eq!(parse(Some("soon")), Some(DEFAULT));
        assert_eq!(parse(Some("-1")), Some(DEFAULT));
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some(" 8 ")), Some(Duration::from_secs(8)));
        assert_eq!(parse(Some("20")), Some(Duration::from_secs(20)));
        assert_eq!(parse(Some("600")), Some(LONGEST));
        assert!(DEFAULT <= LONGEST && LONGEST < Duration::from_secs(30));
    }

    /// Every wait before the same present ends at the same time on the
    /// clock, whenever it starts; the next present starts a new one.
    #[test]
    fn waits_between_two_presents_share_one_end() {
        let (mut w, t, budget) = (new(), Instant::now(), Duration::from_secs(15));
        let s = Duration::from_secs;
        assert_eq!(w.left(7, t, budget), Some(s(15)));
        // A second thread, 14 s later: one second, not fifteen more.
        assert_eq!(w.left(7, t + s(14), budget), Some(s(1)));
        assert_eq!(w.left(7, t + s(15), budget), None);
        assert_eq!(w.left(7, t + s(40), budget), None);
        assert_eq!(w.left(8, t + s(40), budget), Some(s(15)));
    }

    /// A wait that runs out ends the long waits until every queued screen
    /// pass is made, however many presents go by.
    #[test]
    fn a_wait_that_runs_out_is_not_repeated() {
        let (mut w, t, budget) = (new(), Instant::now(), Duration::from_secs(15));
        (w.open, w.late) = (2, false);
        assert!(w.left(1, t, budget).is_some());
        assert!(w.ran_out());
        // A second thread whose wait ends at the same time.
        assert!(!w.ran_out());
        assert_eq!(w.left(1, t, budget), None);
        assert_eq!(w.left(2, t, budget), None);
        assert!(!w.made());
        assert_eq!(w.left(3, t, budget), None);
        assert!(w.made());
        assert_eq!(w.left(4, t, budget), Some(budget));
    }

    /// A pipeline made just as its wait ends leaves nothing late.
    #[test]
    fn nothing_is_late_once_all_are_made() {
        let mut w = new();
        w.open = 1;
        assert!(!w.made());
        assert!(!w.ran_out());
        assert!(!w.late);
    }
}
