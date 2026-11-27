//! A program with some of its skips left out, so that every lane walks the
//! code they skipped (`emit::walk`, `emit::Options::halves_walk`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs").
//!
//! The console's compiler puts `s_cbranch_execz` around code no lane of the
//! wave is left for: an `if` body, or the rest of the program (the
//! early-out `if (!inside) return;`). The per-lane form (`structure`) turns
//! it into a branch on the lane's own bit, so the lanes outside EXEC go
//! round the code and the blocks in it are run by some lanes only. A console
//! wave64 that runs as two host waves can hand a value across its halves
//! only in a block every lane runs (`emit::lower::exchange`: the hand-over
//! needs a group barrier), so a lane mask counted inside such code got each
//! half's own answer.
//!
//! Here the branch is left out and the code goes on into the skipped
//! region: a lane outside EXEC walks it with its EXEC bit clear, as it does
//! on the console whenever some other lane of its wave is inside. Vector
//! instructions are masked by EXEC there as everywhere (the per-lane form
//! gives each a guard where reaching it no longer proves the bit), so the
//! lane changes no register and no memory; what it gains is that it arrives
//! at the region's barriers.
//!
//! Which branches ([`skip`]):
//!
//! * `s_cbranch_execz` to a block that only ends the program, and its long
//!   form for a program whose end the 16-bit branch offset does not reach,
//!   `s_cbranch_execnz` over a far jump to such a block: the early-out;
//! * `s_cbranch_execz` over a region with one way in (the branch) and one
//!   way out (the branch's target), no end of the program inside and no way
//!   back to the branch, where the target merges no scalar register or SCC
//!   that the region changed with the value from before. A lane that went
//!   round arrived there with the old scalars; now it arrives with the
//!   region's, and nothing after can tell.
//!
//! Only the skips around the blocks asked for are left out
//! ([`walked`]): every other region is still gone round, so a lane walks no
//! more than the way to the hand-overs.
//!
//! What the caller still has to check, with the lane masks of the new
//! program ([`Skip`]): that no vector instruction of a region runs in a lane
//! outside the EXEC the branch tested, and that such a lane leaves the
//! region outside EXEC. Then each lane does in the new program what it did
//! in the old one.
//!
//! The idea is hyphennn's (fxceless/victory-fork 578d459, `without_early_outs`
//! and `without_skips`) with avatarmanz 4fd34fdd's far early-out; the code
//! here is new: chosen skips only, the stricter rule for a region (no
//! scalar followed through copies, no join left unread), and the EXEC
//! values a caller needs for the check.

use super::*;

/// A branch [`walked`] left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Skip {
    /// The block that ended in it.
    pub at: usize,
    /// The blocks it went round, which every lane arriving at `at` now
    /// walks: for an early-out, all the code after it.
    pub inside: Vec<bool>,
    /// EXEC (its low dword's value in the new program) as the branch
    /// tested it.
    pub exec: Value,
    /// For a region with a join where EXEC is still read: each block of the
    /// region that leaves for the join, with EXEC at its end.
    pub leaves: Vec<(usize, Value)>,
}

/// Whether block `b` only ends the program.
fn ends_program(p: &Program, b: usize) -> bool {
    p.blocks[b].exit == Exit::End
        && p.blocks[b]
            .insts
            .iter()
            .all(|i| matches!(i.fields, Fields::Sopp { .. }))
}

/// Whether block `b` is the far jump of a branch whose target the branch's
/// 16-bit offset does not reach (`s_getpc_b64`, an offset added to it,
/// `s_setpc_b64`), to a block that only ends the program.
fn far_jump_to_end(p: &Program, b: usize) -> bool {
    let block = &p.blocks[b];
    matches!(block.exit, Exit::Jump(t) if ends_program(p, t))
        && block.insts.iter().all(|i| {
            matches!(
                i.name(),
                "s_getpc_b64" | "s_add_u32" | "s_addc_u32" | "s_setpc_b64"
            )
        })
}

/// The skip block `b` ends in, if it is one that can be left out (the
/// module notes): where the code goes on, the region, and the region's join
/// (none for an early-out).
fn skip(p: &Program, b: usize) -> Option<(usize, Vec<bool>, Option<usize>)> {
    let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
        return None;
    };
    let (next, join) = match p.blocks[b].insts.last()?.name() {
        "s_cbranch_execz" if ends_program(p, taken) => (not_taken, None),
        "s_cbranch_execnz" if far_jump_to_end(p, not_taken) => (taken, None),
        "s_cbranch_execz" => (not_taken, Some(taken)),
        _ => return None,
    };
    let mut inside = vec![false; p.blocks.len()];
    let mut stack = vec![next];
    while let Some(x) = stack.pop() {
        if Some(x) == join || inside[x] {
            continue;
        }
        if ends_program(p, x) {
            if join.is_some() {
                return None; // the region ends the program
            }
            continue;
        }
        if x == b {
            return None; // a way back to the branch
        }
        inside[x] = true;
        match p.blocks[x].exit {
            Exit::Indirect => return None,
            Exit::End if join.is_some() => return None,
            exit => stack.extend(exit.successors()),
        }
    }
    let Some(j) = join else {
        return Some((next, inside, None));
    };
    // One way in.
    let side = |x: usize| p.blocks[x].preds.iter().any(|&q| q != b && !inside[q]);
    if (0..inside.len()).any(|x| inside[x] && side(x)) {
        return None;
    }
    // The join merges no scalar the region changed.
    let join = &p.blocks[j];
    for phi in &join.phis {
        let scalar = match phi.reg {
            Reg::S(c) => c != EXEC && c != EXEC + 1,
            Reg::Scc => true,
            Reg::V(_) => false,
        };
        if !scalar {
            continue;
        }
        let args = || join.preds.iter().zip(&phi.args);
        let before: Vec<Value> = args().filter(|a| *a.0 == b).map(|a| *a.1).collect();
        if args().any(|(q, a)| inside[*q] && !before.contains(a)) {
            return None;
        }
    }
    Some((next, inside, Some(j)))
}

/// `p` without the skips ([`skip`]) that go round a block of `wanted`, and
/// those skips; `None` if there is none, or the control flow that is left
/// would need blocks copied.
pub(crate) fn walked(p: &Program, wanted: &[bool]) -> Option<(Program, Vec<Skip>)> {
    let n = p.blocks.len();
    let mut blocks = p.blocks.clone();
    let mut left = Vec::new();
    for b in 0..n {
        if let Some((next, inside, join)) = skip(p, b)
            && inside.iter().zip(wanted).any(|(i, w)| *i && *w)
        {
            // (the branch instruction stays until its EXEC has its new name)
            blocks[b].exit = Exit::Jump(next);
            left.push((b, inside, join));
        }
    }
    if left.is_empty() {
        return None;
    }
    // What nothing reaches any more (the far jump of an early-out) is empty.
    let mut reached = vec![false; n];
    let mut stack = vec![0];
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut reached[b], true) {
            stack.extend(blocks[b].exit.successors());
        }
    }
    for (block, _) in blocks.iter_mut().zip(&reached).filter(|(_, r)| !**r) {
        block.insts.clear();
        block.exit = Exit::End;
    }
    for block in &mut blocks {
        block.phis.clear();
    }
    link(&mut blocks);
    split_irreducible(&mut blocks).ok()?;
    if blocks.len() != n {
        return None;
    }
    let mut q = rename(blocks, p.wave64);
    let mut tested = Vec::new();
    for &(b, ..) in &left {
        tested.push(q.blocks[b].insts.pop()?.read(Reg::S(EXEC))?);
    }
    let order = reverse_postorder(&q.blocks);
    let (live_in, _) = liveness(&q.blocks, &order);
    let mut skips = Vec::new();
    for ((at, inside, join), exec) in left.into_iter().zip(tested) {
        let mut leaves = Vec::new();
        if let Some(j) = join
            && contains(&live_in[j], Reg::S(EXEC))
        {
            for &x in q.blocks[j].preds.iter().filter(|x| inside[**x]) {
                leaves.push((x, exec_at_end(&q, x)?));
            }
        }
        skips.push(Skip {
            at,
            inside,
            exec,
            leaves,
        });
    }
    Some((q, skips))
}

/// EXEC's low dword at the end of block `b`, where it is live: the block's
/// last write of it, else its join there, else what the block before it
/// left (without a join every way in brings the same value).
fn exec_at_end(p: &Program, mut b: usize) -> Option<Value> {
    for _ in 0..p.blocks.len() {
        let block = &p.blocks[b];
        if let Some(v) = block
            .insts
            .iter()
            .rev()
            .find_map(|i| i.write(Reg::S(EXEC)))
        {
            return Some(v);
        }
        if let Some(phi) = block.phis.iter().find(|f| f.reg == Reg::S(EXEC)) {
            return Some(phi.value);
        }
        match block.preds.first() {
            Some(&q) => b = q,
            None => return p.inputs.iter().find(|i| i.0 == Reg::S(EXEC)).map(|i| i.1),
        }
    }
    None
}

/// Names the values of `blocks` (linked, without phis) anew: the second
/// half of [`build_bounded`], which this copies so that the build itself
/// stays as it is (the test below holds the two together).
fn rename(mut blocks: Vec<Block>, wave64: bool) -> Program {
    let order = reverse_postorder(&blocks);
    let (live_in, live_out) = liveness(&blocks, &order);
    let mut values = 0u32;
    let mut new_value = || {
        values += 1;
        Value(values - 1)
    };
    let mut inputs = Vec::new();
    let mut outs: Vec<Vec<(Reg, Value)>> = vec![Vec::new(); blocks.len()];
    let mut current = [NONE; REGS];
    for &b in &order {
        current.fill(NONE);
        let block = &mut blocks[b];
        for reg in members(&live_in[b]) {
            current[index(reg)] = match block.preds[..] {
                [] => {
                    let v = new_value();
                    inputs.push((reg, v));
                    v
                }
                [p] => lookup(&outs[p], reg),
                _ => {
                    let value = new_value();
                    block.phis.push(Phi {
                        reg,
                        value,
                        args: Vec::new(),
                    });
                    value
                }
            };
        }
        for inst in &mut block.insts {
            for (reg, v) in &mut inst.reads {
                *v = current[index(*reg)];
            }
            for (reg, v) in &mut inst.writes {
                *v = new_value();
                current[index(*reg)] = *v;
            }
        }
        outs[b] = members(&live_out[b])
            .map(|reg| (reg, current[index(reg)]))
            .collect();
    }
    for block in &mut blocks {
        for phi in &mut block.phis {
            phi.args = block
                .preds
                .iter()
                .map(|&p| lookup(&outs[p], phi.reg))
                .collect();
        }
    }
    remove_trivial_phis_steps(&mut blocks, values);
    Program {
        wave64,
        blocks,
        inputs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// `s_mov_b64 s[2:3], exec`, `s_mov_b64 exec, s[2:3]`.
    const SAVE: u32 = 0xBE82_047E;
    const RESTORE: u32 = 0xBEFE_0402;
    /// `v_cmpx_ne_u32 exec, 5, v0`.
    const CMPX: u32 = 0x7DAA_0085;
    /// `s_cbranch_execz` over `n` words.
    const fn execz(n: u32) -> u32 {
        0xBF88_0000 | n
    }

    fn build_words(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        build(&code, true, 16).unwrap()
    }

    fn all(p: &Program) -> Vec<bool> {
        vec![true; p.blocks.len()]
    }

    /// An `if` whose body writes `v1`, then the end.
    fn an_if(body: &[u32]) -> Vec<u32> {
        let mut w = vec![SAVE, CMPX, execz(body.len() as u32)];
        w.extend(body);
        w.extend([RESTORE, 0x7E04_0301, S_ENDPGM]); // v_mov_b32 v2, v1
        w
    }

    /// The copy of the renaming is the build's: a program whose values are
    /// named anew is the program.
    #[test]
    fn naming_the_values_anew_gives_the_build() {
        for words in [
            an_if(&[0x7E02_0280]),
            an_if(&[0xBE84_0380, 0x7E02_0204]),
            vec![SAVE, CMPX, execz(2), 0x7E02_0280, 0x7E04_0301, S_ENDPGM],
        ] {
            let p = build_words(&words);
            let mut blocks = p.blocks.clone();
            for b in &mut blocks {
                b.phis.clear();
            }
            assert_eq!(rename(blocks, p.wave64), p);
        }
    }

    /// An `if` is walked: its branch is gone, the body follows, and the
    /// skip names the body and the EXEC the branch tested (what `v_cmpx`
    /// wrote).
    #[test]
    fn an_if_around_a_wanted_block_is_walked() {
        let p = build_words(&an_if(&[0x7E02_0280])); // v_mov_b32 v1, 0
        assert!(matches!(p.blocks[0].exit, Exit::Branch { taken: 2, not_taken: 1 }));
        let (q, skips) = walked(&p, &all(&p)).expect("walked");
        assert_eq!(q.blocks[0].exit, Exit::Jump(1));
        assert_eq!(q.blocks[0].insts.last().unwrap().name(), "v_cmpx_ne_u32");
        assert_eq!(q.blocks[2].preds, [1]);
        assert!(q.blocks.iter().all(|b| b.phis.is_empty()));
        let [skip] = &skips[..] else {
            panic!("{skips:?}")
        };
        assert_eq!((skip.at, &skip.inside[..]), (0, &[false, true, false][..]));
        assert_eq!(
            Some(skip.exec),
            q.blocks[0].insts[1].write(Reg::S(EXEC)),
            "{q:?}"
        );
        // EXEC is restored at the join before anything reads it.
        assert!(skip.leaves.is_empty());
        // Nothing asked for inside: nothing changes.
        assert!(walked(&p, &[true, false, true]).is_none());
    }

    /// An `else` reads EXEC at the join (`s_andn2_b64 exec, s[2:3], exec`):
    /// the skip says what EXEC the body leaves with.
    #[test]
    fn a_join_that_reads_exec_is_named() {
        let p = build_words(&[
            SAVE,
            CMPX,
            execz(1),
            0x7E02_0280, // v_mov_b32 v1, 0
            0x8AFE_7E02, // s_andn2_b64 exec, s[2:3], exec
            0x7E04_0301, // v_mov_b32 v2, v1
            S_ENDPGM,
        ]);
        let (q, skips) = walked(&p, &all(&p)).expect("walked");
        assert_eq!(skips.len(), 1);
        assert_eq!(skips[0].leaves, [(1, skips[0].exec)], "{q:?}");
    }

    /// A body that sets a scalar the code after the join reads is not
    /// walked: a lane that went round it had the old one.
    #[test]
    fn a_region_that_changes_a_scalar_read_later_stays_skipped() {
        // s4 is 1 before; the body sets it to 0; v2 takes it after the join.
        let p = build_words(&[
            0xBE84_0381, // s_mov_b32 s4, 1
            SAVE,
            CMPX,
            execz(1),
            0xBE84_0380, // s_mov_b32 s4, 0
            RESTORE,
            0x7E04_0204, // v_mov_b32 v2, s4
            S_ENDPGM,
        ]);
        assert!(p.blocks[2].phis.iter().any(|f| f.reg == Reg::S(4)));
        assert!(walked(&p, &all(&p)).is_none());
        // Set and used inside only: walked.
        let p = build_words(&an_if(&[0xBE84_0380, 0x7E02_0204]));
        assert!(walked(&p, &all(&p)).is_some());
    }

    /// The early-out: the branch to the end is left out and the lanes that
    /// would have left walk the rest.
    #[test]
    fn an_early_out_is_walked() {
        let p = build_words(&[
            CMPX,
            execz(2),
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E04_0301, // v_mov_b32 v2, v1
            S_ENDPGM,
        ]);
        assert!(matches!(p.blocks[0].exit, Exit::Branch { taken: 2, not_taken: 1 }));
        let (q, skips) = walked(&p, &all(&p)).expect("walked");
        assert_eq!(q.blocks[0].exit, Exit::Jump(1));
        assert_eq!(skips.len(), 1);
        assert_eq!(skips[0].inside, [false, true, false]);
        assert!(skips[0].leaves.is_empty());
    }

    /// Other branches stay: on VCC, on SCC, and a skip with a second way
    /// out.
    #[test]
    fn other_branches_are_kept() {
        // v_cmp_ne_u32 vcc, 5, v0; s_cbranch_vccz +1
        let p = build_words(&[0x7D8A_0085, 0xBF86_0001, 0x7E02_0280, 0x7E04_0301, S_ENDPGM]);
        assert!(matches!(p.blocks[0].exit, Exit::Branch { .. }));
        assert!(walked(&p, &all(&p)).is_none());
        // An `if` with an early-out inside: its region ends the program.
        let p = build_words(&[
            SAVE,
            CMPX,
            execz(3),
            CMPX,
            execz(3), // to the end
            0x7E02_0280,
            RESTORE,
            0x7E04_0301,
            S_ENDPGM,
        ]);
        let (q, skips) = walked(&p, &all(&p)).expect("the inner early-out");
        assert_eq!(skips.len(), 1, "{q:?}");
        assert_eq!(skips[0].at, 1);
        assert!(matches!(q.blocks[0].exit, Exit::Branch { .. }));
    }
}
