//! Blocks that a thread group runs together: every thread exactly once, or
//! no thread at all ([`Mac::walk`], `emit::Options::halves_walk`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs").
//!
//! A hand-over across the halves ([`super::exchange`]) needs a group
//! barrier, and a barrier needs every thread of the group. Blocks every
//! lane runs exactly once ([`lds_order::once`]) have that. So does a block
//! under a branch that every thread of the group takes the same way: the
//! title's programs test a flag of a constant buffer ("is this feature on
//! for the pass?") with `s_cmp` and `s_cbranch_scc`, and count a lane mask
//! inside. Either the whole group comes to the barrier there or none of it.
//!
//! Which branches ([`decided`]): `s_cbranch_scc0` or `scc1` on an SCC that
//! is a group value. A group value ([`values`]) is a scalar that is the same
//! whenever and by whichever thread of the group it is computed:
//!
//! * an SGPR the hardware sets on entry (user data, the group's id), but
//!   not the thread-group info SGPR, whose wave index differs between the
//!   group's waves, and not EXEC;
//! * what a scalar instruction makes of group values and constants alone
//!   (never of EXEC, a lane mask or SCC of one: a first-lane read, a count
//!   of a mask and anything a wave computes are not scalar instructions of
//!   that kind);
//! * what a scalar load reads at an address made of group values, in a
//!   program that stores to no memory by address, and for a buffer whose
//!   descriptor no store or atomic of the program uses.
//!
//! A join of group values is one only where the group arrives as one: in a
//! block run together, every way into which comes from a block run together
//! (a loop's counter, joined at the loop's head from the count before the
//! loop and the count the last round left). All threads then come the same
//! way and take the same value. Any other join is each thread's own: which
//! value it holds depends on its path.
//!
//! Then a block is run together ([`blocks`]) when every branch that can go
//! round it is decided by a group value, and every branch of every cycle
//! through it is: the group goes round such a loop as one, each thread as
//! often as the others (the title's list programs count the tiles of 25
//! classes in a loop of 25 rounds). Why: take two threads of a group
//! and the first block run together that one of them reaches and the other
//! does not reach next. Their paths part at a branch before it. A decided
//! branch sends both the same way (its SCC is a group value, by the rule
//! above the same in both at the same round), so it is another one, from
//! which the block is reachable; by the rule the block is then on every way
//! from that branch to the end of the program, also on the second
//! thread's, and no cycle of undecided branches leads past it first. The
//! two rules lean on each other (a loop's counter is a group value because
//! the loop's head is run together, which it is because the loop's exit is
//! decided by the counter), so [`blocks`] starts from every block and takes
//! away until nothing changes: what is left holds by induction over the
//! group's steps.
//!
//! What this rests on, beyond the exchange's own premises: that the memory
//! a scalar load reads does not change while the group runs. The program's
//! own stores are ruled out as far as its descriptors tell (two different
//! descriptors of one buffer would not be seen); other writers do not run
//! during a dispatch. If it failed, part of a group would wait at a barrier
//! for the rest.
//!
//! Only a half's answer is handed over in such a block
//! ([`Lower::exchange_halves_in`]): it has slots of its own. Lane values
//! share two sets of slots in turn, which is right only when every
//! exchange runs ([`super::exchange`], 3), so they stay with the blocks
//! every lane runs. In a block on a cycle the same slots are stored again
//! each round, so the hand-over there ends with a second barrier
//! ([`Together::looped`], [`Lower::exchanged_halves`]).

use super::*;

/// Scalar values that are the same in every thread of a group (the module
/// notes), given the blocks taken to be run together.
fn values(x: &Parts, together: &[bool]) -> Set<Value> {
    let p = x.p;
    // Joins where the group arrives as one, all taken to hold first.
    let whole = |b: usize| together[b] && p.blocks[b].preds.iter().all(|&q| together[q]);
    let scalar = |r: Reg| matches!(r, Reg::Scc) || matches!(r, Reg::S(c) if c < crate::ir::VCC);
    let mut joins: Vec<&Phi> = (0..p.blocks.len())
        .filter(|&b| whole(b))
        .flat_map(|b| p.blocks[b].phis.iter().filter(|f| scalar(f.reg)))
        .collect();
    loop {
        let same = values_with(x, &joins);
        let held = joins.len();
        joins.retain(|f| f.args.iter().all(|a| same.contains(a)));
        if joins.len() == held {
            return same;
        }
    }
}

/// [`values`] with `joins` taken as group values.
fn values_with(x: &Parts, joins: &[&Phi]) -> Set<Value> {
    let p = x.p;
    let mut same: Set<Value> = joins.iter().map(|f| f.value).collect();
    for &(reg, v) in &p.inputs {
        let info = x
            .io
            .inputs
            .iter()
            .any(|i| i.0 == v && !matches!(i.1, Input::GroupId(_)));
        let set = x.io.initialized_sgprs.unwrap_or(0).min(crate::ir::VCC);
        if matches!(reg, Reg::S(c) if c < set) && !info {
            same.insert(v);
        }
    }
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    // Memory the program writes by address, and the buffers it writes.
    let stores = |i: &Inst| i.name().contains("store") || i.name().contains("atomic");
    let by_address = insts().any(|i| matches!(i.fields, Fields::Flat { .. }) && stores(i));
    let written: Vec<&Source> = x
        .res
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource && stores(&p.blocks[d.site.0].insts[d.site.1]))
        .map(|d| &d.source)
        .collect();
    let unwritten = |site: Site| {
        x.res
            .descriptors
            .iter()
            .filter(|d| d.site == site && d.role == Role::Resource)
            .all(|d| !written.contains(&&d.source))
    };
    // Reverse postorder: a value's instruction comes before its uses, but
    // for joins.
    for b in crate::ir::reverse_postorder(&p.blocks) {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            let pure = match inst.fields {
                Fields::Sop { .. } => ![
                    "s_getreg", "s_setreg", "s_movrel", "s_setpc", "s_swappc", "s_call", "s_rfe",
                    "s_subvector", "s_wqm",
                ]
                .iter()
                .any(|n| name.starts_with(n)),
                Fields::Smem { .. } => {
                    !by_address
                        && (name.starts_with("s_load_dword")
                            || name.starts_with("s_buffer_load_dword") && unwritten((b, i)))
                }
                _ => false,
            };
            let group = |r: &(Reg, Value)| {
                !matches!(r.0, Reg::S(c) if c == EXEC || c == EXEC + 1) && same.contains(&r.1)
            };
            if pure && inst.reads.iter().all(group) {
                same.extend(inst.writes.iter().map(|w| w.1));
            }
        }
    }
    same
}

/// Whether the branch block `b` ends in is taken the same way by every
/// thread of a group: on SCC, a group value, and a uniform branch in the
/// per-lane form too.
fn decided(x: &Parts, same: &Set<Value>, b: usize) -> bool {
    let uniform = match x.st.exits[b] {
        LaneExit::Branch { when, .. } => {
            let atoms = x.st.bools.support(when);
            atoms.iter().all(|a| matches!(a, Atom::Uniform(_)))
        }
        _ => false,
    };
    uniform
        && x.p.blocks[b].insts.last().is_some_and(|i| {
            matches!(i.name(), "s_cbranch_scc0" | "s_cbranch_scc1")
                && i.read(Reg::Scc).is_some_and(|v| same.contains(&v))
        })
}

/// What a thread group runs together.
pub(super) struct Together {
    /// Per block: every thread of the group runs it as often as every
    /// other, in step, or none does.
    pub blocks: Vec<bool>,
    /// Per block: it is on a cycle (run more than once).
    pub looped: Vec<bool>,
}

/// The blocks a thread group runs together (the module notes).
pub(super) fn blocks(x: &Parts) -> Together {
    let p = x.p;
    let n = p.blocks.len();
    if p.blocks.iter().any(|b| b.exit == Exit::Indirect) {
        return Together {
            blocks: vec![false; n],
            looped: vec![false; n],
        };
    }
    let succ = |b: usize| p.blocks[b].exit.successors();
    let reach = |from: usize, stop: &dyn Fn(usize) -> bool| {
        let mut seen = vec![false; n];
        let mut stack = succ(from);
        while let Some(b) = stack.pop() {
            if !stop(b) && !std::mem::replace(&mut seen[b], true) {
                stack.extend(succ(b));
            }
        }
        seen
    };
    // Post-dominators, as bitsets (a virtual exit after every end).
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let only = |b: usize| {
        let mut v = vec![0; words];
        v[b / 64] |= 1 << (b % 64);
        v
    };
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| if succ(b).is_empty() { only(b) } else { full.clone() })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev().filter(|&b| !succ(b).is_empty()) {
            let mut v = full.clone();
            for t in succ(b) {
                for (w, q) in v.iter_mut().zip(&pdom[t]) {
                    *w &= q;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let branches: Vec<usize> = (0..n)
        .filter(|&d| matches!(p.blocks[d].exit, Exit::Branch { .. }))
        .collect();
    // Per branch: the blocks it can go round, and those on a cycle with it.
    let after: Vec<Vec<bool>> = branches.iter().map(|&d| reach(d, &|_| false)).collect();
    let round: Vec<Vec<bool>> = branches
        .iter()
        .map(|&d| reach(d, &|b| pdom[d][b / 64] >> (b % 64) & 1 == 1))
        .collect();
    let looped: Vec<bool> = (0..n).map(|b| reach(b, &|_| false)[b]).collect();
    let before: Vec<Vec<bool>> = (0..n)
        .map(|b| if looped[b] { reach(b, &|_| false) } else { Vec::new() })
        .collect();
    // From every block, down to what holds (the module notes).
    let mut together = vec![true; n];
    loop {
        let same = values(x, &together);
        let mut next = vec![true; n];
        for (k, &d) in branches.iter().enumerate() {
            if decided(x, &same, d) {
                continue;
            }
            for b in 0..n {
                // (`d` on a cycle through `b`: each reaches the other)
                if round[k][b] || looped[b] && after[k][b] && before[b][d] {
                    next[b] = false;
                }
            }
        }
        if next == together {
            return Together { blocks: together, looped };
        }
        together = next;
    }
}

impl Lower<'_> {
    /// Whether the first-lane read `inst` (of the block being lowered) can
    /// only run in one thread of the group ([`Mac::walk`]): the group is one
    /// row of threads, and every lane that reaches the read passed a compare
    /// of the thread id with a constant (`v_cmp_eq_u32 vcc, 0, v0`, with EXEC
    /// narrowed to it), and runs it without a further guard. EXEC there is
    /// that one lane on the console too, so the first lane is the reader
    /// and the read is its own value: the title's list programs let thread
    /// 0 sum the group's counts this way.
    pub(super) fn first_alone(&self, inst: &Inst) -> bool {
        let (x, b) = (self.x, self.cur_ir);
        let (p, st) = (x.p, x.st);
        let row = self.s.threads.is_some_and(|t| t[1] * t[2] == 1);
        let i = p.blocks[b].insts.iter().position(|q| std::ptr::eq(q, inst));
        if !(x.mac.walk && row && i.is_some_and(|i| st.guards[b][i].is_none())) {
            return false;
        }
        // The thread id's x on entry.
        let id = p.inputs.iter().find(|r| r.0 == Reg::V(0)).map(|r| r.1);
        let is_id = |v: Value| {
            Some(v) == id
                && x.io
                    .inputs
                    .iter()
                    .any(|n| n.0 == v && matches!(n.1, Input::ThreadIdInGroup(0)))
        };
        let reach = [(st.reach[b], true)];
        p.blocks.iter().flat_map(|q| &q.insts).any(|c| {
            let Fields::Valu(v) = &c.fields else {
                return false;
            };
            let with_id = match v.src[..] {
                [Operand::Const(_), Operand::Vector { reg, .. }]
                | [Operand::Vector { reg, .. }, Operand::Const(_)] => {
                    c.read(Reg::V(reg)).is_some_and(is_id)
                }
                _ => false,
            };
            matches!(c.name(), "v_cmp_eq_u32" | "v_cmp_eq_i32")
                && with_id
                && c.writes.iter().any(|w| {
                    matches!(w.0, Reg::S(_)) && st.bools.decides(&reach, Atom::Lane(w.1)) == Some(true)
                })
        })
    }
}
