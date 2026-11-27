//! macOS port: input switches for play at a few frames a second, all off by
//! default. A child of `gameinput`, so that file keeps its upstream shape.
//!
//! `VICTORY_INPUT_LATCH=1`: the title takes a pad's current reading once a
//! frame, 250 to 500 ms apart here, and a press and release between two of
//! them left no reading it saw (sprint is tapped, so is jump). With the latch
//! each button change the title has not seen yet waits for it: a button that
//! was up in the last reading handed out shows down in the next one if it went
//! down at any time since, and one that was down shows up if it was let go at
//! any time since (pressed again as well: down in the reading after). Triggers
//! show their highest value since the last reading; sticks the latest value.
//! Readings handed out within `LATCH_HOLD` of each other agree.
//!
//! `VICTORY_STICK_DEADZONE=<share>` (for example 0.15): host pad sticks within
//! that share of full travel of the centre read as centred, and the rest of
//! the travel is spread over 0 to 1 again (a round dead zone per stick).
//! XInput values are raw: a stick at rest reads up to several per cent off
//! centre, more on a worn pad, and every wobble of it was a new reading. The
//! title has a dead zone of its own; this one comes before it.
//!
//! `VICTORY_INPUT_REFOCUS=1`: keys already down when the title's window gets
//! the focus back are left out until they are seen up. Wine's macOS driver
//! posts a key's release only while our app is active (`cocoa_app.m`,
//! `NSEventTypeKeyUp` goes through only for keys it saw go down, and only to
//! our windows), so a key let go in another app, Cmd among them after
//! Cmd-Tab, stays down in `GetAsyncKeyState` until it is pressed again.
//!
//! `VICTORY_INPUT_WATCH=1`: once a second, while a host pad is off rest or a
//! key is down, one log line (`input watch`) with what the host reports: each
//! such pad's sticks, triggers and buttons, how long its value has been
//! exactly the same and how long XInput's packet number has (a pad whose
//! reports stopped keeps both still), and the keys down. Changes nothing.

use super::{FALLBACK, PADS, PadState};

/// Devices with a latch: the host pads and the keyboard's stand-in pad.
const LATCHED: usize = FALLBACK + 1;
const _: () = assert!(FALLBACK == PADS);

/// Reads of one device this close together (µs) see the same buttons.
const LATCH_HOLD: u64 = 15_000;
/// With no reading handed out for this long (µs) the latch lets go, so that a
/// title that stops asking for a pad never finds a button stuck down.
const LATCH_MAX: u64 = 2_000_000;

#[derive(Clone, Copy, Debug, Default)]
struct Latch {
    /// Buttons the device's readings show.
    shown: u32,
    /// `shown` when the last reading was handed out: what the title saw.
    seen: u32,
    /// Since that hand-out: buttons down in some host sample, up in some,
    /// and down in a sample after one where they were up.
    down: u32,
    up: u32,
    again: u32,
    /// Pressed again after a release the title is about to see: shown down
    /// in the reading after that one.
    queued: u32,
    /// Trigger peaks since that hand-out, and what readings show.
    peak: [f32; 2],
    triggers: [f32; 2],
    /// While readings are held as handed out: since when (µs).
    holding: Option<u64>,
    /// The last hand-out (µs).
    handed: u64,
}

impl Latch {
    fn hand_out(&mut self, now: u64) {
        if self.holding.is_some() {
            return;
        }
        self.queued &= !self.shown;
        self.seen = self.shown;
        (self.down, self.up, self.again) = (0, 0, 0);
        self.peak = [0.0; 2];
        self.holding = Some(now);
        self.handed = now;
    }

    fn apply(&mut self, pad: PadState, now: u64, hold: u64) -> PadState {
        if self.holding.is_none() && now.saturating_sub(self.handed) >= LATCH_MAX {
            self.hand_out(now);
        }
        let host = pad.buttons;
        self.again |= self.up & host;
        self.down |= host;
        self.up |= !host;
        self.peak[0] = self.peak[0].max(pad.left_trigger);
        self.peak[1] = self.peak[1].max(pad.right_trigger);
        if self
            .holding
            .is_some_and(|at| now.saturating_sub(at) >= hold)
        {
            self.holding = None;
        }
        if self.holding.is_none() {
            let seen = self.seen;
            self.queued |= seen & self.up & self.again;
            self.shown = (seen & !self.up) | (!seen & (self.down | self.queued));
            self.triggers = self.peak;
        }
        PadState {
            buttons: self.shown,
            left_trigger: self.triggers[0],
            right_trigger: self.triggers[1],
            ..pad
        }
    }
}

/// Keys held across a focus change; see the module notes.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
pub(super) struct Refocus {
    /// `None` until the environment was read.
    on: Option<bool>,
    focused: bool,
    /// Down when the focus came back, and not seen up since.
    ignored: Vec<u8>,
}

#[cfg(any(windows, test))]
impl Refocus {
    /// Takes out of `held` (virtual keys the host reports down while
    /// `focused`, none without it) the keys held across a focus change.
    pub(super) fn filter(&mut self, focused: bool, held: &mut Vec<u8>) {
        let on = *self.on.get_or_insert_with(|| {
            let on = std::env::var("VICTORY_INPUT_REFOCUS")
                .is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
            if on {
                tracing::info!(
                    "input refocus: keys down when the focus comes back wait for their release (VICTORY_INPUT_REFOCUS)"
                );
            }
            on
        });
        if !on {
            return;
        }
        if focused && !self.focused && !held.is_empty() {
            tracing::info!(keys = ?held, "input refocus: keys left out until released");
            self.ignored = held.clone();
        }
        self.focused = focused;
        self.ignored.retain(|k| held.contains(k));
        held.retain(|k| !self.ignored.contains(k));
    }
}

/// What `VICTORY_INPUT_WATCH` keeps of one host pad.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug)]
struct Watched {
    pad: PadState,
    packet: u32,
    /// When the value and the packet number last changed (µs).
    changed: u64,
    packet_changed: u64,
}

/// The input watch; see the module notes.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
pub(super) struct Watch {
    /// `None` until the environment was read.
    on: Option<bool>,
    pads: [Option<Watched>; PADS],
    /// The last line (µs).
    logged: u64,
}

#[cfg(any(windows, test))]
impl Watch {
    fn on(&mut self) -> bool {
        *self.on.get_or_insert_with(|| {
            let on = std::env::var("VICTORY_INPUT_WATCH")
                .is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
            if on {
                tracing::info!(
                    "input watch: a line a second while input is off rest (VICTORY_INPUT_WATCH)"
                );
            }
            on
        })
    }

    /// Host slot `slot` reported `state` (with XInput's packet number).
    pub(super) fn pad(&mut self, slot: usize, state: Option<(PadState, u32)>, now: u64) {
        if !self.on() {
            return;
        }
        let Some(watched) = self.pads.get_mut(slot) else {
            return;
        };
        *watched = state.map(|(pad, packet)| match *watched {
            Some(w) => Watched {
                pad,
                packet,
                changed: if w.pad == pad { w.changed } else { now },
                packet_changed: if w.packet == packet {
                    w.packet_changed
                } else {
                    now
                },
            },
            None => Watched {
                pad,
                packet,
                changed: now,
                packet_changed: now,
            },
        });
    }

    /// The keys down (virtual keys) after the pads of one poll; the line
    /// logged, if one is due.
    pub(super) fn keys(&mut self, keys: &[u8], now: u64) -> Option<String> {
        use std::fmt::Write;
        if !self.on() || now.saturating_sub(self.logged) < 1_000_000 {
            return None;
        }
        let off_rest = |p: &PadState| {
            p.buttons != 0
                || [
                    p.left_trigger,
                    p.right_trigger,
                    p.left_x,
                    p.left_y,
                    p.right_x,
                    p.right_y,
                ]
                .iter()
                .any(|v| v.abs() > 0.1)
        };
        let secs = |since: u64| now.saturating_sub(since) as f64 / 1e6;
        let mut line = String::new();
        for (slot, w) in self.pads.iter().enumerate() {
            let Some(w) = w.filter(|w| off_rest(&w.pad)) else {
                continue;
            };
            let p = w.pad;
            let _ = write!(
                line,
                "pad {slot}: left {:+.2} {:+.2} right {:+.2} {:+.2} triggers {:.2} {:.2} buttons {:#x}, same for {:.1} s, packet {} same for {:.1} s; ",
                p.left_x,
                p.left_y,
                p.right_x,
                p.right_y,
                p.left_trigger,
                p.right_trigger,
                p.buttons,
                secs(w.changed),
                w.packet,
                secs(w.packet_changed),
            );
        }
        if !keys.is_empty() {
            let _ = write!(line, "keys {keys:02x?}");
        }
        if line.is_empty() {
            return None;
        }
        self.logged = now;
        tracing::info!("input watch: {line}");
        Some(line)
    }
}

/// The switches and what they keep; `Default` is all off.
#[derive(Debug, Default)]
pub(super) struct Mac {
    /// `Some(hold)` with the latch on.
    latch: Option<u64>,
    latches: [Latch; LATCHED],
    /// Share of stick travel read as centred; 0 is off.
    dead_zone: f32,
}

/// A round dead zone of `zone` on one stick.
fn dead_zone(x: f32, y: f32, zone: f32) -> (f32, f32) {
    let length = x.hypot(y);
    if length <= zone {
        return (0.0, 0.0);
    }
    let scale = ((length - zone) / (1.0 - zone)).min(1.0) / length;
    (x * scale, y * scale)
}

impl Mac {
    /// The switches from the environment, one log line for each one on.
    pub(super) fn from_env() -> Mac {
        let on =
            |name: &str| std::env::var(name).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
        let mut mac = Mac::default();
        if on("VICTORY_INPUT_LATCH") {
            tracing::info!(
                hold_ms = LATCH_HOLD / 1000,
                "input latch: presses and releases between two readings show in the next (VICTORY_INPUT_LATCH)"
            );
            mac.latch = Some(LATCH_HOLD);
        }
        let zone = std::env::var("VICTORY_STICK_DEADZONE").ok();
        match zone.as_deref().map(str::trim).filter(|z| !z.is_empty()) {
            None | Some("0") | Some("off") => {}
            Some(z) => match z.parse::<f32>() {
                Ok(share) if (0.0..0.9).contains(&share) => {
                    tracing::info!(share, "stick dead zone (VICTORY_STICK_DEADZONE)");
                    mac.dead_zone = share;
                }
                _ => tracing::warn!(
                    value = z,
                    "VICTORY_STICK_DEADZONE wants a share from 0 to 0.9"
                ),
            },
        }
        mac
    }

    /// Host pads' sticks through the dead zone.
    pub(super) fn dead_zones(&self, pads: &mut [Option<PadState>; PADS]) {
        if self.dead_zone <= 0.0 {
            return;
        }
        for pad in pads.iter_mut().flatten() {
            (pad.left_x, pad.left_y) = dead_zone(pad.left_x, pad.left_y, self.dead_zone);
            (pad.right_x, pad.right_y) = dead_zone(pad.right_x, pad.right_y, self.dead_zone);
        }
    }

    /// Device `index`'s gamepad state as its next reading shows it.
    pub(super) fn latch(
        &mut self,
        index: usize,
        pad: Option<PadState>,
        now: u64,
    ) -> Option<PadState> {
        let (Some(hold), Some(latch)) = (self.latch, self.latches.get_mut(index)) else {
            return pad;
        };
        let Some(pad) = pad else {
            *latch = Latch {
                handed: now,
                ..Latch::default()
            };
            return None;
        };
        Some(latch.apply(pad, now, hold))
    }

    /// A reading of device `index` went to the title at `now`.
    pub(super) fn handed(&mut self, index: usize, now: u64) {
        if self.latch.is_some()
            && let Some(latch) = self.latches.get_mut(index)
        {
            latch.hand_out(now);
        }
    }
}

#[cfg(test)]
pub(super) fn latch_for_tests(hold: u64) -> Mac {
    Mac {
        latch: Some(hold),
        ..Mac::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u32 = 4;

    fn pad(buttons: u32, left_trigger: f32) -> Option<PadState> {
        Some(PadState {
            buttons,
            left_trigger,
            left_x: 0.25,
            ..Default::default()
        })
    }

    fn buttons(mac: &mut Mac, host: u32, now: u64) -> u32 {
        mac.latch(0, pad(host, 0.0), now).unwrap().buttons
    }

    #[test]
    fn a_second_press_before_the_title_saw_the_release_comes_after_it() {
        let mut mac = latch_for_tests(0);
        assert_eq!(buttons(&mut mac, A, 10), A);
        mac.handed(0, 11);
        // Let go, pressed again, let go, all before the next reading.
        for (host, t) in [(0, 20), (A, 30), (0, 40)] {
            assert_eq!(buttons(&mut mac, host, t), 0);
        }
        mac.handed(0, 41);
        assert_eq!(buttons(&mut mac, 0, 50), A, "the second press");
        mac.handed(0, 51);
        assert_eq!(buttons(&mut mac, 0, 60), 0);
    }

    #[test]
    fn reads_close_together_agree() {
        let mut mac = latch_for_tests(LATCH_HOLD);
        assert_eq!(buttons(&mut mac, A, 10), A);
        mac.handed(0, 1_000);
        assert_eq!(buttons(&mut mac, 0, 9_000), A, "held as handed out");
        mac.handed(0, 9_500);
        assert_eq!(buttons(&mut mac, 0, 17_000), 0, "then the release");
    }

    #[test]
    fn a_latch_nobody_reads_lets_go() {
        let mut mac = latch_for_tests(0);
        assert_eq!(buttons(&mut mac, A, 10), A);
        assert_eq!(buttons(&mut mac, 0, 20), A);
        // Never handed out: shown as the host has it after `LATCH_MAX`.
        assert_eq!(buttons(&mut mac, 0, LATCH_MAX + 1), 0);
    }

    #[test]
    fn keys_held_across_a_focus_change_wait_for_their_release() {
        let mut refocus = Refocus {
            on: Some(true),
            ..Refocus::default()
        };
        let mut keys = |focused, mut held: Vec<u8>| {
            refocus.filter(focused, &mut held);
            held
        };
        assert_eq!(keys(true, vec![]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W']), [b'W']);
        // Away: nothing; back with Wine still holding W (its release went
        // to another app) and Cmd (0x5B).
        assert_eq!(keys(false, vec![]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B, b'A']), [b'A']);
        // W seen up once: pressed again, it counts.
        assert_eq!(keys(true, vec![0x5B]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B]), [b'W']);
        // Off: unchanged.
        let mut off = Refocus {
            on: Some(false),
            ..Refocus::default()
        };
        let mut held = vec![b'W'];
        off.filter(true, &mut held);
        assert_eq!(held, [b'W']);
    }

    #[test]
    fn the_watch_says_how_long_a_pad_has_been_still() {
        let mut watch = Watch {
            on: Some(true),
            ..Watch::default()
        };
        let held = pad(0, 0.0).unwrap();
        watch.pad(0, Some((PadState::default(), 7)), 1_000_000);
        assert_eq!(watch.keys(&[], 1_000_000), None, "at rest: nothing");
        for t in [2_000_000, 3_000_000] {
            watch.pad(0, Some((held, 8)), t);
        }
        watch.pad(0, Some((held, 8)), 4_500_000);
        let line = watch.keys(&[0x57], 4_500_000).unwrap();
        assert!(line.contains("pad 0: left +0.25 +0.00"), "{line}");
        assert!(
            line.contains("same for 2.5 s, packet 8 same for 2.5 s"),
            "{line}"
        );
        assert!(line.ends_with("keys [57]"), "{line}");
        // At most a line a second.
        assert_eq!(watch.keys(&[0x57], 5_000_000), None);
    }

    #[test]
    fn sticks_near_the_centre_read_centred() {
        let mut pads = [pad(0, 0.0), None, None, None];
        pads[0].as_mut().unwrap().right_y = -1.0;
        let mac = Mac {
            dead_zone: 0.25,
            ..Mac::default()
        };
        mac.dead_zones(&mut pads);
        let p = pads[0].unwrap();
        // 0.25 from centre is inside; full travel stays full.
        assert_eq!(
            (p.left_x, p.left_y, p.right_x, p.right_y),
            (0.0, 0.0, 0.0, -1.0)
        );
        let (x, y) = dead_zone(0.0, 0.625, 0.25);
        assert!(x == 0.0 && (y - 0.5).abs() < 1e-6);
        let (x, y) = dead_zone(0.8, 0.8, 0.25);
        assert!(
            (x - y).abs() < 1e-6 && (x.hypot(y) - 1.0).abs() < 1e-6,
            "direction kept"
        );
        // Off by default.
        let mut pads = [pad(0, 0.0), None, None, None];
        Mac::default().dead_zones(&mut pads);
        assert_eq!(pads[0].unwrap().left_x, 0.25);
    }

    #[test]
    fn triggers_keep_their_peak_and_sticks_their_latest() {
        let mut mac = latch_for_tests(0);
        mac.latch(1, pad(0, 0.75), 10);
        let shown = mac.latch(1, pad(0, 0.25), 20).unwrap();
        assert_eq!((shown.left_trigger, shown.left_x), (0.75, 0.25));
        mac.handed(1, 25);
        assert_eq!(mac.latch(1, pad(0, 0.25), 30).unwrap().left_trigger, 0.25);
        // A disconnect forgets the latch.
        mac.latch(1, pad(A, 0.0), 40);
        assert_eq!(mac.latch(1, None, 50), None);
        assert_eq!(mac.latch(1, pad(0, 0.0), 60).unwrap().buttons, 0);
    }
}
