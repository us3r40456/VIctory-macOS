//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing for a long time or for good: refused (64-lane waves on a host of
//! 32-lane waves), not translated, or still being made. The stages that do
//! run then work on what the missing ones never wrote, and write the result
//! into title memory: on fxceless/victory-fork, garbage bounds that crashed
//! the title. So the builder's dispatches run only in frames where every
//! pipeline it has created has its host pipeline; otherwise none of them
//! run (the structures keep what they held).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back).
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(p));
    make::begin(p);
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let states: Vec<Option<bool>> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        pipelines
            .iter()
            .filter_map(Weak::upgrade)
            .map(|p| make::state(&p))
            .collect()
    };
    let halves = crate::shader_cache::halves_local_here();
    let open = opens(&states, halves);
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            "BVH builder {}",
            if open {
                "runs"
            } else if halves {
                "held back: its 64-lane programs would run as 32-lane halves on their own (VICTORY_HALVES_LOCAL, VICTORY_HALVES_SHARED_LDS)"
            } else {
                "held back: not all its pipelines are made"
            }
        );
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Whether the builder runs: every pipeline made, and its 64-lane programs
/// not run as two 32-lane halves that each act as a wave (`halves`:
/// `VICTORY_HALVES_LOCAL=1` or `VICTORY_HALVES_SHARED_LDS=1` on a host of
/// 32-lane waves). The builder's
/// programs exchange across the halves; run that way they hung the GPU
/// within two minutes on an RTX 4070 SUPER (upstream PR #2711, from which
/// this rule is taken, keyed there on the host's wave size alone).
fn opens(states: &[Option<bool>], halves: bool) -> bool {
    !halves && all_made(states)
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
    guard
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
        assert!(opens(&[Some(true)], false));
        assert!(!opens(&[Some(true), Some(false)], false));
        assert!(!opens(&[Some(true)], true), "64-lane programs run as halves");
        assert!(!opens(&[], true));
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
