//! Ray passes left out while nothing builds the ray-tracing structures
//! (`VICTORY_RAY_SKIP`, off by default). The idea is cool_fox's
//! (codec-d/cool_fox 8cc31ab3, `pipeline/make.rs` `left_out`); the code is
//! this line's own. Plain std, so its tests run everywhere.
//!
//! The title traces its reflections and the rays of its screen probes
//! against structures that its builder (`XboxBvhBuilder.dll`) makes every
//! frame. This line holds the builder back in every frame where it cannot
//! run whole (`crate::pipeline::held_back`, upstream #2118), and always
//! with `VICTORY_HALVES_LOCAL=1` or `VICTORY_HALVES_SHARED_LDS=1`. Then the
//! traces run against structures that were never built, and two kinds of
//! programs work on what they find:
//!
//! - `CS_RaytracingResolve_*` (technique `raytracingresolve_dispatch`, one
//!   program per material effect; 1,000 in the title's shader file): each
//!   shades the hits of its material, through an indirect dispatch whose
//!   records the tracing passes fill. With nothing built there is nothing
//!   to hit: in frame 1961 of an earlier run all six dispatches of one of
//!   them asked for no groups (`notes/frame-1961-analysis.md`). Each one
//!   still costs a bind, an argument refresh, the argument clamp's own
//!   dispatch and its barriers.
//! - `CS_ScreenProbes_ActivateRays_*` (one program, `39f0e9aeb156a61c`,
//!   a 71,816-byte container): it chooses the rays the screen probes trace.
//!
//! With the switch on, in a frame where the builder is held back, their
//! dispatches are left out: not bound, not clamped, not run, and not
//! counted as dropped; [`log`] counts them with the frame line. In a frame
//! where the builder runs they run as before. Their pipelines are made as
//! before (the early list starts them at creation), so that the A/B runs
//! make the same pipelines and a frame the builder runs in has them.
//!
//! Programs are known by name: the bytes of the name in the shader
//! container, as `crate::ambient` knows its programs. Only compute
//! containers are searched, and only while the switch is on.
//!
//! What the picture should lose (read in the notes and the code, not seen
//! in a run):
//! - The resolves: nothing, as long as their records ask for no groups.
//!   The ambient stand-in does not replace their output: it replaces what
//!   `CS_ScreenProbes_TemporalResolve` writes and holds down what
//!   `CS_UpdateHistory` writes (`crate::ambient_set`), and neither name is
//!   one of those two. Their output (a 1280x720 R11G11B10 image, by the
//!   target dumps' names) is still read by the reflection passes after
//!   them (a guess from the names). If a trace does find something in an
//!   unbuilt structure, those texels keep what the image held before the
//!   resolves instead of a material's shading of a false hit.
//! - The activation: its ray choice feeds the rest of the probe chain,
//!   which ends in `CS_ScreenProbes_TemporalResolve`. With
//!   `VICTORY_AMBIENT=1` the light image that program writes is replaced
//!   whole, so that light should not change; the program's other images
//!   (it writes several) may. Without the stand-in, the title's own
//!   indirect light changes to whatever the probes hold without new rays.
//!   Belobug's fork, which leaves this program out when no rays run
//!   (6986a57, AMD), found that the picture lost light, shadows and
//!   reflections when it ran. So `activate` and `resolve` can be left out
//!   on their own.
//!
//! `VICTORY_RAY_SKIP=1` (or `on`, `all`): both; `resolve`: the resolves
//! only; `activate`: the activation only (`resolve,activate` is both).
//! Unset, empty, `0` or `off`: nothing is left out and nothing here runs
//! but one check of a stored value per container and per pipeline.

// Used by `crate::pipeline` and `crate::draw`, which are only built for
// Windows.
#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// The resolves of the reflection hits, one per material (a bit of what a
/// program is to this module).
pub(crate) const RESOLVE: u8 = 1;
/// The screen probes' ray activation.
pub(crate) const ACTIVATE: u8 = 2;

/// The programs' names, as their shader containers hold them (followed by
/// a part of their own and `_Wrapped`).
const NAMES: [(u8, &[u8]); 2] = [
    (RESOLVE, b"CS_RaytracingResolve"),
    (ACTIVATE, b"CS_ScreenProbes_ActivateRays"),
];

/// The parts `value` of `VICTORY_RAY_SKIP` turns on, and whether it was
/// understood (a value that is not understood turns nothing on).
fn parts(value: Option<&str>) -> (u8, bool) {
    let value = value.map(str::trim).unwrap_or_default();
    if value.is_empty() || value == "0" || value.eq_ignore_ascii_case("off") {
        return (0, true);
    }
    let mut parts = 0;
    for part in value.split(',').map(str::trim) {
        parts |= match part.to_ascii_lowercase().as_str() {
            "1" | "on" | "all" => RESOLVE | ACTIVATE,
            "resolve" => RESOLVE,
            "activate" => ACTIVATE,
            _ => return (0, false),
        };
    }
    (parts, true)
}

/// The parts that are on, read once.
pub(crate) fn on() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_RAY_SKIP").ok();
        let (on, understood) = parts(value.as_deref());
        if !understood {
            tracing::warn!(
                value = value.as_deref().unwrap_or_default(),
                "VICTORY_RAY_SKIP takes 1, resolve or activate: nothing is left out"
            );
        }
        on
    })
}

/// The line said at start when something is left out.
fn start_line(parts: u8) -> Option<String> {
    let what = match parts & (RESOLVE | ACTIVATE) {
        0 => return None,
        RESOLVE => {
            "CS_RaytracingResolve (the shading of reflection hits, one program per material)"
        }
        ACTIVATE => "CS_ScreenProbes_ActivateRays (the screen probes' choice of rays)",
        _ => {
            "CS_RaytracingResolve (the shading of reflection hits, one program per material) \
             and CS_ScreenProbes_ActivateRays (the screen probes' choice of rays)"
        }
    };
    Some(format!(
        "VICTORY_RAY_SKIP: in frames where the BVH builder is held back, dispatches of {what} \
         are left out (their pipelines are still made)"
    ))
}

/// Says what is left out, once, when the host device is ready.
pub(crate) fn announce() {
    if let Some(line) = start_line(on()) {
        tracing::info!("{line}");
    }
}

/// What the program of shader container `container` is to this module:
/// bits of [`RESOLVE`] and [`ACTIVATE`], 0 for nothing.
fn kind_of(container: &[u8]) -> u8 {
    NAMES
        .iter()
        .filter(|(_, name)| container.windows(name.len()).any(|w| w == *name))
        .fold(0, |kind, (k, _)| kind | k)
}

/// The programs seen so far that a part that is on leaves out, by their
/// hash as logged.
static PROGRAMS: Mutex<Option<HashMap<String, u8>>> = Mutex::new(None);

/// Programs known so far, by part (resolves, activation).
static KNOWN: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// The title passed the shader container `container` of `stage` for the
/// first time, `hash` as logged: remembered if a part that is on leaves
/// its program out. Nothing while the switch is off.
pub(crate) fn note_program(stage: &str, hash: &str, container: &[u8]) {
    let on = on();
    if on == 0 || stage != "compute" {
        return;
    }
    let kind = kind_of(container) & on;
    if kind == 0 {
        return;
    }
    for (i, part) in [RESOLVE, ACTIVATE].into_iter().enumerate() {
        if kind & part != 0 && KNOWN[i].fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::info!(
                hash,
                "the first program VICTORY_RAY_SKIP leaves out while the BVH builder is held back: {}",
                name(part)
            );
        }
    }
    PROGRAMS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(hash.to_owned(), kind);
}

fn name(part: u8) -> &'static str {
    if part == ACTIVATE {
        "CS_ScreenProbes_ActivateRays"
    } else {
        "CS_RaytracingResolve"
    }
}

/// What a title pipeline of the shaders with these hashes is to this
/// module (0: nothing, and always while the switch is off).
pub(crate) fn kind<'a>(hashes: impl Iterator<Item = &'a str>) -> u8 {
    if on() == 0 {
        return 0;
    }
    let programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(programs) = programs.as_ref() else {
        return 0;
    };
    hashes
        .filter_map(|hash| programs.get(hash))
        .fold(0, |kind, k| kind | k)
}

/// Dispatches left out since the last [`log`], by part.
static LEFT_OUT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// A dispatch of a pipeline of `kind` was left out.
pub(crate) fn left_out(kind: u8) {
    let i = usize::from(kind & RESOLVE == 0);
    LEFT_OUT[i].fetch_add(1, Ordering::Relaxed);
}

/// The line logged with the frame line, `left` dispatches left out over
/// `frames` frames and `known` programs seen, by part.
fn frame_line(left: [u64; 2], frames: f64, known: [u64; 2]) -> String {
    let [resolve, activate] = left.map(|n| n as f64 / frames.max(1.0));
    format!(
        "ray passes left out while the BVH builder is held back (VICTORY_RAY_SKIP): \
         {resolve:.1} CS_RaytracingResolve and {activate:.1} CS_ScreenProbes_ActivateRays \
         dispatches/frame; programs known {} and {}",
        known[0], known[1]
    )
}

/// With the frame line, over `frames` frames: the dispatches left out.
/// Nothing while the switch is off.
pub(crate) fn log(frames: f64) {
    if on() == 0 {
        return;
    }
    let left = LEFT_OUT.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    let known = KNOWN.each_ref().map(|n| n.load(Ordering::Relaxed));
    tracing::info!("{}", frame_line(left, frames, known));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_names_its_parts() {
        assert_eq!(parts(None), (0, true));
        assert_eq!(parts(Some("")), (0, true));
        assert_eq!(parts(Some(" 0 ")), (0, true));
        assert_eq!(parts(Some("OFF")), (0, true));
        assert_eq!(parts(Some("1")), (RESOLVE | ACTIVATE, true));
        assert_eq!(parts(Some("all")), (RESOLVE | ACTIVATE, true));
        assert_eq!(parts(Some("resolve")), (RESOLVE, true));
        assert_eq!(parts(Some(" Activate ")), (ACTIVATE, true));
        assert_eq!(parts(Some("resolve, activate")), (RESOLVE | ACTIVATE, true));
        assert_eq!(parts(Some("yes")), (0, false));
        assert_eq!(parts(Some("resolve,x")), (0, false));
    }

    /// Programs are known by their name anywhere in the container, as the
    /// title's shader file names them; others are nothing here.
    #[test]
    fn programs_are_known_by_name() {
        let container = |name: &str| {
            let mut c = vec![0u8; 300];
            c.extend_from_slice(name.as_bytes());
            c.extend_from_slice(b"_Wrapped\0\x01\x02");
            c
        };
        assert_eq!(
            kind_of(&container("CS_RaytracingResolve_a56bfb9c")),
            RESOLVE
        );
        assert_eq!(
            kind_of(&container("CS_ScreenProbes_ActivateRays_d60186e6")),
            ACTIVATE
        );
        assert_eq!(
            kind_of(&container("CS_ScreenProbes_ActivateProbes_9083c094")),
            0
        );
        assert_eq!(
            kind_of(&container("CS_ScreenProbes_TraceRays_RayBinning_BVH")),
            0
        );
        assert_eq!(kind_of(&container("CS_RaytraceVBuffer")), 0);
        assert_eq!(kind_of(&container("CS_RaytracingResolv")), 0);
        assert_eq!(kind_of(b""), 0);
    }

    /// Nothing left out is a program the ambient stand-in follows (it works
    /// right after their dispatches), and none of those is left out.
    #[test]
    fn the_ambient_stand_ins_programs_are_never_left_out() {
        use crate::ambient_set::{PROGRAM, REFLECTED, SKY, role_of};
        for (_, name) in NAMES {
            assert_eq!(role_of(name), 0, "{}", String::from_utf8_lossy(name));
        }
        for name in [PROGRAM, REFLECTED, SKY] {
            assert_eq!(kind_of(name), 0, "{}", String::from_utf8_lossy(name));
        }
        // Nor one the estimate follows (`VICTORY_AMBIENT=estimate`).
        use crate::ambient_estimate_set as estimate;
        for (_, name) in NAMES {
            assert_eq!(estimate::role_of(name), 0, "{}", String::from_utf8_lossy(name));
        }
        for name in [estimate::RADIANCE_PROGRAM, estimate::AIR_PROGRAM] {
            assert_eq!(kind_of(name), 0, "{}", String::from_utf8_lossy(name));
        }
    }

    #[test]
    fn the_lines_say_what_is_left_out() {
        assert_eq!(start_line(0), None);
        let both = start_line(RESOLVE | ACTIVATE).unwrap();
        assert!(
            both.starts_with("VICTORY_RAY_SKIP: ")
                && both.contains("CS_RaytracingResolve")
                && both.contains("ActivateRays"),
            "{both}"
        );
        assert!(!start_line(RESOLVE).unwrap().contains("ActivateRays"));
        assert!(!start_line(ACTIVATE).unwrap().contains("RaytracingResolve"));
        let line = frame_line([3_160, 10], 10.0, [278, 1]);
        assert!(line.contains(" 316.0 CS_RaytracingResolve and 1.0 CS_ScreenProbes_ActivateRays dispatches/frame; programs known 278 and 1"), "{line}");
    }
}
