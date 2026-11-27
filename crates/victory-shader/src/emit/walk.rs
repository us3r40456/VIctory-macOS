//! Hand-overs across the 32-lane halves in code only some lanes ran
//! (`Options::halves_walk`, with `Options::halves_exchange`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs").
//!
//! A console wave64 that runs as two host waves can hand a value across its
//! halves only in a block every lane runs exactly once
//! (`lower::exchange`). The title's tiled sun-shadow programs count lane
//! masks ("does any lane of the wave still need the wide filter?", "did any
//! lane see a caster?") behind their early-out and inside `if` bodies,
//! where the per-lane form sends the lanes outside EXEC round the code. So
//! each half answered for itself: refused, or with half-local reads two
//! answers where the console has one.
//!
//! Here such a program is translated a second way: with the skips around
//! those places left out ([`crate::ir::walk`]), so every lane walks the
//! skipped code, EXEC bit clear, and the places are blocks every lane runs.
//! The second way is taken only if it is the same program for every lane
//! and exact across the halves:
//!
//! 1. Only skips the console takes when no lane of the wave is left
//!    (`s_cbranch_execz`, and the early-out's long form), around a region
//!    with one way in and one way out whose scalars nothing after it reads
//!    (`ir::walk::skip`).
//! 2. A lane outside the EXEC the skip tested does nothing in the region
//!    ([`idle`], checked on the new program's lane masks): no vector
//!    instruction of the region can run in it, and it leaves the region
//!    outside EXEC. Its registers, the memory and the masks it is counted
//!    in are then what they were when it went round.
//! 3. The new program translates with no read across the halves answered
//!    by a half alone (half-local reads off): every such read is handed
//!    over, or was exact before.
//!
//! If any of the three fails the program is translated as before, bit for
//! bit. So is a program that is exact already, and one that reads nothing
//! across its halves where not every lane runs.
//!
//! What it costs the GPU: the walked code's scalar instructions and its
//! unguarded arithmetic run in every lane of the group, where the lanes
//! outside EXEC used to leave. Regions that hold no hand-over are still
//! gone round.
//!
//! The first round opens the places the program as it stands asks for; a
//! hand-over made there can uncover the next (a count that decided a branch
//! around another count), so the rounds repeat a few times.

use super::*;
use crate::gfx10::Fields;
use crate::ir::walk::Skip;
use crate::ir::{EXEC, Reg};
use crate::structure::{Atom, Bool, Structured};

/// How often a program is opened further before it is left as it was.
const ROUNDS: usize = 4;

/// A program in another form, with everything the emitter needs of it.
pub(super) struct Form {
    pub p: ir::Program,
    pub st: Structured,
    pub io: io::Io,
    pub res: resource::Resources,
}

/// `input`'s program translated with the skips around its hand-overs left
/// out, if that makes every read across the halves exact (the module
/// notes); `None`: translate it as it is.
pub(super) fn exact(input: &EmitInput<'_>, options: Options) -> Option<HostShader> {
    let native = input.native;
    if !(options.halves_walk
        && options.halves_exchange
        && options.host_wave32
        && native.wave64
        && native.stage == HwStage::Cs)
    {
        return None;
    }
    let start = (input.p, input.st, input.io, input.res);
    open(native, start, |(p, st, io, res), local| {
        let form = EmitInput {
            c: input.c,
            native,
            p,
            st,
            io,
            res,
        };
        let options = Options {
            halves_local: local,
            ..options
        };
        emit_form(&form, options, None)
    })
}

/// A program as the stages before the emitter leave it.
pub(super) type Stages<'a> = (
    &'a ir::Program,
    &'a Structured,
    &'a io::Io,
    &'a resource::Resources,
);

/// [`exact`] with the lowering given: `lower` translates a form of the
/// program, with half-local reads (its second argument) or without.
pub(super) fn open<T>(
    native: &NativeProgram,
    start: Stages<'_>,
    mut lower: impl FnMut(Stages<'_>, bool) -> Result<T, ShaderError>,
) -> Option<T> {
    let mut form: Option<Form> = None;
    for _ in 0..ROUNDS {
        let next = {
            let cur = match &form {
                Some(f) => (&f.p, &f.st, &f.io, &f.res),
                None => start,
            };
            // Where this form asks for a hand-over and not every lane runs.
            let (made, wanted) = lower::exchanges_wanted(|| lower(cur, true));
            made.ok()?;
            // Nothing to open, or the program is exact as it stands (what
            // was asked for there is a value nothing ends up needing).
            if wanted.is_empty() || form.is_none() && lower(cur, false).is_ok() {
                return None;
            }
            let mut around = vec![false; cur.0.blocks.len()];
            for b in wanted {
                around[b] = true;
            }
            let (p, skips) = ir::walk::walked(cur.0, &around)?;
            let st = structure::structure(&p).ok()?;
            if !idle(&p, &st, &skips) {
                return None;
            }
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).ok()?;
            let io = io::raise(&p, native).ok()?;
            Form { p, st, io, res }
        };
        if let Ok(made) = lower((&next.p, &next.st, &next.io, &next.res), false) {
            return Some(made);
        }
        form = Some(next);
    }
    None
}

/// Whether, in the program `p` that [`ir::walk::walked`] made, a lane
/// outside the EXEC a left-out skip tested does nothing in the skip's
/// region (the module notes, 2).
///
/// A loop in the region joins EXEC at its head, and the join is a mask of
/// its own to the structure pass. Such joins are taken by induction over
/// the lane's steps ([`inside`]): a join holds no lane outside the tested
/// EXEC if no mask it takes in does, the other joins being granted.
fn idle(p: &ir::Program, st: &Structured, skips: &[Skip]) -> bool {
    let mask = |v| {
        let known = st.masks.get(&v).or_else(|| st.other.get(&v));
        known.or_else(|| st.joined.get(&v)).copied()
    };
    skips.iter().all(|s| {
        let Some(tested) = mask(s.exec) else {
            return false;
        };
        let joins = inside(p, st, s, tested, &mask);
        // Whether a lane at block `at` can meet `runs` outside that EXEC.
        let outside = |at: usize, runs: Bool| {
            let mut all = vec![(st.reach[at], true), (runs, true), (tested, false)];
            all.extend(joins.iter().map(|j| (*j, false)));
            st.bools.meet(&all)
        };
        for (x, block) in p.blocks.iter().enumerate().filter(|(x, _)| s.inside[*x]) {
            for (i, inst) in block.insts.iter().enumerate() {
                if matches!(
                    inst.fields,
                    Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
                ) {
                    continue;
                }
                if inst.read(Reg::S(EXEC)).is_none() {
                    // (`v_writelane_b32` writes its lane whatever EXEC is)
                    if inst.writes.iter().any(|w| matches!(w.0, Reg::V(_))) {
                        return false;
                    }
                    continue;
                }
                if outside(x, st.guards[x][i].unwrap_or(Bool::TRUE)) {
                    return false;
                }
            }
        }
        s.leaves
            .iter()
            .all(|&(x, e)| mask(e).is_some_and(|m| !outside(x, m)))
    })
}

/// The lane-mask joins in the region of `s` (each as its own mask, an
/// [`Atom::Phi`]) that hold no lane outside `tested`: all of them taken to
/// hold first, then those taken out one of whose incoming masks is unknown
/// or can, on its edge and with the others still granted, hold a lane
/// outside `tested`, until none is. What is left holds at every step of a
/// lane: a join's value is one of its incoming masks, made of masks that
/// held it before.
fn inside(
    p: &ir::Program,
    st: &Structured,
    s: &Skip,
    tested: Bool,
    mask: &dyn Fn(ir::Value) -> Option<Bool>,
) -> Vec<Bool> {
    let own = |f: &ir::Phi| {
        mask(f.value).filter(|m| {
            matches!(st.bools.node(*m), Some((Atom::Phi(v), Bool::FALSE, Bool::TRUE)) if v == f.value)
        })
    };
    let mut joins: Vec<(usize, &ir::Phi, Bool)> = (0..p.blocks.len())
        .filter(|b| s.inside[*b])
        .flat_map(|b| p.blocks[b].phis.iter().map(move |f| (b, f)))
        .filter_map(|(b, f)| own(f).map(|m| (b, f, m)))
        .collect();
    loop {
        let granted: Vec<(Bool, bool)> = joins.iter().map(|j| (j.2, false)).collect();
        let holds = |(b, f, _): &(usize, &ir::Phi, Bool)| {
            p.blocks[*b].preds.iter().zip(&f.args).all(|(&q, &a)| {
                mask(a).is_some_and(|m| {
                    let mut all = vec![(st.reach[q], true), (m, true), (tested, false)];
                    all.extend(&granted);
                    !st.bools.meet(&all)
                })
            })
        };
        let kept: Vec<_> = joins.iter().filter(|j| holds(j)).copied().collect();
        if kept.len() == joins.len() {
            return joins.iter().map(|j| j.2).collect();
        }
        joins = kept;
    }
}
