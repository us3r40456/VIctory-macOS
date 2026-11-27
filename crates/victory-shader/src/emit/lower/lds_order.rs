//! An exact order for LDS that a compute group of one console wave shares,
//! on a host that runs the group as several narrower waves ([`Mac::lds_order`],
//! `emit::Options::lds_order`; `notes/cross-half-programs.md`, "Shared LDS").
//!
//! The console runs such a group in lockstep, so its compiler leaves out
//! every barrier: an LDS access sees exactly the accesses of the
//! instructions before it, in every lane. [`split_wave_barriers`] gives the
//! host that order with group barriers where it can, and gives up where a
//! read in divergent code may see a write made in divergent code (no
//! barrier can go where only some lanes run). This is a second, finer
//! analysis for the programs it gives up on. It knows which dwords an
//! access touches when its address is a constant, and what kind of access
//! it is, and so can leave alone what no order changes:
//!
//! * two reads; two atomics of the same commutative kind that return
//!   nothing (`ds_add_u32` and `ds_sub_u32`; `ds_min`, `ds_max`, `ds_and`,
//!   `ds_or`, `ds_xor`); two writes of the same constant to the same dword,
//!   or of one constant to all they touch; accesses of different dwords.
//!
//! Every other pair of accesses that the console orders (a path leads from
//! one to the other) and that no barrier separates is a clash. Barriers go
//! only in blocks every lane runs exactly once, as before: a clash at an
//! access in such a block puts a barrier before it, and such a block whose
//! open accesses clash with something ahead syncs at its end when divergent
//! code follows. Every lane runs a block sequence that is a subsequence of
//! the wave's, so "before the barrier" means the same for all of them, and
//! the group barrier makes every access before it visible to every access
//! after it.
//!
//! A clash left in divergent code is allowed in one form only, a *flag*: a
//! read of a dword and writes of one constant to it (lanes that find
//! something set the flag; the others test it to stop early). The halves
//! may then read the flag before or after the other half set it, and that
//! must change nothing else:
//!
//! 1. Whether any lane sets the flag does not depend on the order. Take
//!    the first write in any order of the lanes' accesses: every read
//!    before it returned the old value, so its lane ran as in the order
//!    where no lane had written yet; in any other order that lane either
//!    runs the same way and writes, or read the constant, which some lane
//!    wrote. So at the next barrier the dword is the constant in every
//!    order or in none.
//! 2. What a lane does after such a read, up to the next block every lane
//!    runs, is nothing but more of the same ([`Order::flag_read`]): no
//!    effect other than writes of the constant to dwords the read covers,
//!    and no value made there that anything outside uses, as data, as a
//!    condition, or through a phi that the lane's way out selects. "Uses"
//!    is told per lane: a register such a value is left in may be written
//!    again under a condition and then read under the same condition (the
//!    lanes that read it are the lanes that wrote it), and a compare of it
//!    under that condition may decide only what the condition already
//!    holds for (`structure::Bools::ignores`). Anything else is a use.
//!
//! Not ordered here, and refused as before: atomics that return the old
//! value (the console hands them out in lane order within one instruction;
//! two host waves do not), exchanges, and clashes in divergent code that
//! are no flag. Two lanes that store different values to one dword in one
//! instruction have no order on the host either; the console keeps one of
//! them, and nothing here says which.
//!
//! Hosts answer nothing in this: the form uses no wave operation, only
//! `barrier` with group sync, where [`split_wave_barriers`] already puts
//! it. Where the group is two halves the program's other cross-lane
//! operations stay under the halves' rules ([`super::halves`]).

use super::*;

/// (block, instruction); instruction = the block's length for its end.
type At = (usize, usize);

pub(super) const RETURNS: Why = "LDS one console wave64 shares without barriers, on a host of 32-lane waves (no exact order: an LDS atomic that returns the old value, or an exchange)";
pub(super) const DIVERGENT: Why = "LDS one console wave64 shares without barriers, on a host of 32-lane waves (no exact order: LDS accesses in divergent code whose order matters)";
pub(super) const FLAG_USED: Why = "LDS one console wave64 shares without barriers, on a host of 32-lane waves (no exact order: a flag read in divergent code decides more than the flag)";
pub(super) const SHAPE: Why = "LDS one console wave64 shares without barriers, on a host of 32-lane waves (no exact order: control flow or LDS use the analysis does not follow)";

/// Commutative atomics that return nothing, by the operation they repeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Read,
    Write,
    Update(u8),
}

/// One LDS access of the program.
#[derive(Debug)]
struct Access {
    at: At,
    kind: Kind,
    /// The dwords it touches, each with the constant a write stores there
    /// if it is one; `None`: an address not known here, any dword.
    dwords: Option<Vec<(u32, Option<u32>)>>,
    /// The one constant a write stores in every dword it touches.
    fill: Option<u32>,
}

impl Access {
    fn dword(&self, d: u32) -> Option<Option<u32>> {
        let ds = self.dwords.as_ref()?;
        ds.iter().find(|x| x.0 == d).map(|x| x.1)
    }

    /// Whether the two may touch a common dword in a way whose order
    /// changes what memory holds or what is read.
    fn clashes(&self, other: &Access) -> bool {
        let (Some(a), Some(b)) = (&self.dwords, &other.dwords) else {
            return !self.commutes(other, None);
        };
        a.iter()
            .any(|&(d, _)| b.iter().any(|x| x.0 == d) && !self.commutes(other, Some(d)))
    }

    fn commutes(&self, other: &Access, dword: Option<u32>) -> bool {
        match (self.kind, other.kind) {
            (Kind::Read, Kind::Read) => true,
            (Kind::Update(a), Kind::Update(b)) => a == b,
            (Kind::Write, Kind::Write) => {
                self.fill.is_some() && self.fill == other.fill
                    || dword.is_some_and(|d| {
                        matches!((self.dword(d), other.dword(d)), (Some(Some(x)), Some(Some(y))) if x == y)
                    })
            }
            _ => false,
        }
    }
}

/// Blocks every lane runs exactly once: no cycle through them, on every
/// path from the entry to an end (as [`split_wave_barriers`] has them).
pub(crate) fn once(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}

struct Order<'a> {
    p: &'a Program,
    st: &'a Structured,
    once: Vec<bool>,
    accesses: Vec<Access>,
    by_place: Map<At, usize>,
    /// Where each value is written by an instruction.
    defs: Map<Value, At>,
}

/// Where a compute group that is one console wave needs group barriers on
/// a host of narrower waves so that its LDS holds what the console's does
/// (the module notes), or why no such placement is known.
pub(crate) fn barriers(p: &Program, st: &Structured) -> Result<Set<At>, Why> {
    if p.blocks.iter().any(|b| b.exit == Exit::Indirect) {
        return Err(SHAPE);
    }
    let mut defs = Map::default();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            defs.extend(inst.writes.iter().map(|w| (w.1, (b, i))));
        }
    }
    let mut o = Order {
        p,
        st,
        once: once(p),
        accesses: Vec::new(),
        by_place: Map::default(),
        defs,
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for i in 0..block.insts.len() {
            if block.insts[i].name() == "s_barrier" && !o.once[b] {
                return Err(SHAPE);
            }
            if let Some(a) = o.access((b, i))? {
                o.by_place.insert((b, i), o.accesses.len());
                o.accesses.push(a);
            }
        }
    }
    if o.accesses.len() > 128 {
        return Err(SHAPE);
    }
    o.place()
}

impl Order<'_> {
    fn succ(&self, b: usize) -> Vec<usize> {
        self.p.blocks[b].exit.successors()
    }

    /// The constant every lane that runs the instruction at `at` holds in
    /// `v`: a `v_mov_b32` of a constant that every lane reaching it ran, or
    /// that ran under the very condition `at` runs under.
    fn constant(&self, v: Value, at: At) -> Option<u32> {
        let &(b, i) = self.defs.get(&v)?;
        let inst = &self.p.blocks[b].insts[i];
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_mov_b32" || f.dpp.is_some() || f.sdwa.is_some() {
            return None;
        }
        let k = match *f.src.first()? {
            Operand::Const(c) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        let (made, used) = (self.st.guards[b][i], self.st.guards[at.0][at.1]);
        (made.is_none() || made == used).then_some(k)
    }

    /// The LDS access the instruction at `at` makes, if it makes one.
    fn access(&self, at: At) -> Result<Option<Access>, Why> {
        let inst = &self.p.blocks[at.0].insts[at.1];
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            return Ok(None);
        };
        let name = inst.name();
        if name.starts_with("ds_swizzle") || name.contains("permute") {
            return Ok(None);
        }
        if *gds {
            return Err(SHAPE);
        }
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let width = match name.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element, as the lowering has them.
        let elements = |rest: &str| -> Vec<(u32, u32)> {
            if rest.starts_with('2') {
                let scale = 4 * width * if name.contains("st64") { 64 } else { 1 };
                vec![
                    (u32::from(*offset0) * scale, width),
                    (u32::from(*offset1) * scale, width),
                ]
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width)]
            } else {
                // A byte or half of one dword.
                vec![(offset, 0)]
            }
        };
        let (kind, elements) = if let Some(rest) = name.strip_prefix("ds_read") {
            (Kind::Read, elements(rest))
        } else if let Some(rest) = name.strip_prefix("ds_write") {
            (Kind::Write, elements(rest))
        } else {
            let op = match name.strip_prefix("ds_") {
                Some("add_u32" | "sub_u32") => 0,
                Some("and_b32") => 1,
                Some("or_b32") => 2,
                Some("xor_b32") => 3,
                Some("min_i32") => 4,
                Some("max_i32") => 5,
                Some("min_u32") => 6,
                Some("max_u32") => 7,
                _ => return Err(RETURNS),
            };
            if dst.is_some() {
                return Err(RETURNS);
            }
            (Kind::Update(op), vec![(offset, 1)])
        };
        let base = match addr {
            Some(Operand::Vector { reg, .. }) => {
                inst.read(Reg::V(*reg)).and_then(|v| self.constant(v, at))
            }
            _ => None,
        };
        // (byte offset, the constant stored) per dword touched.
        let mut touched = Vec::new();
        for (e, &(off, w)) in elements.iter().enumerate() {
            for k in 0..w.max(1) {
                let stored = match (kind, w, data.get(e)) {
                    (Kind::Write, 1.., Some(Operand::Vector { reg, .. })) => inst
                        .read(Reg::V(*reg + k as u8))
                        .and_then(|v| self.constant(v, at)),
                    _ => None,
                };
                touched.push((off + 4 * k, stored));
            }
        }
        let fill = touched[0]
            .1
            .filter(|_| touched.iter().all(|t| t.1 == touched[0].1));
        let dwords = base.map(|base| {
            (touched.iter())
                .map(|&(off, stored)| (base.wrapping_add(off) >> 2, stored))
                .collect()
        });
        Ok(Some(Access {
            at,
            kind,
            dwords,
            fill,
        }))
    }

    /// The open accesses in `open` (a bit each) that clash with access `x`.
    fn clashing(&self, open: u128, x: usize) -> u128 {
        let mut out = 0;
        for y in 0..self.accesses.len() {
            if open >> y & 1 == 1 && self.accesses[x].clashes(&self.accesses[y]) {
                out |= 1 << y;
            }
        }
        out
    }

    /// Walks block `b` from the open accesses `open` at its entry, with the
    /// barriers `at`; `met` gets each access with what was open before it,
    /// and says whether a barrier goes before it. What is open at the end.
    fn walk(
        &self,
        b: usize,
        mut open: u128,
        at: &Set<At>,
        mut met: impl FnMut(usize, u128) -> bool,
    ) -> u128 {
        let insts = &self.p.blocks[b].insts;
        for (i, inst) in insts.iter().enumerate() {
            // The program's own barriers are in blocks every lane runs
            // ([`barriers`] has checked).
            if at.contains(&(b, i)) || inst.name() == "s_barrier" {
                open = 0;
            }
            if let Some(&x) = self.by_place.get(&(b, i)) {
                if met(x, open) {
                    open = 0;
                }
                open |= 1 << x;
            }
        }
        if at.contains(&(b, insts.len())) {
            open = 0;
        }
        open
    }

    /// The accesses open at each block's end, with the barriers `at`.
    fn flow(&self, at: &Set<At>) -> Vec<u128> {
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        let mut out = vec![0u128; self.p.blocks.len()];
        loop {
            let mut changed = false;
            for &b in &order {
                let open = self.p.blocks[b].preds.iter().fold(0, |a, &q| a | out[q]);
                let open = self.walk(b, open, at, |_, _| false);
                changed |= std::mem::replace(&mut out[b], open) != open;
            }
            if !changed {
                return out;
            }
        }
    }

    /// The accesses in blocks reachable from `b`'s successors.
    fn ahead(&self, b: usize) -> u128 {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.succ(b);
        let mut out = 0;
        while let Some(q) = stack.pop() {
            if std::mem::replace(&mut seen[q], true) {
                continue;
            }
            for (x, a) in self.accesses.iter().enumerate() {
                if a.at.0 == q {
                    out |= 1 << x;
                }
            }
            stack.extend(self.succ(q));
        }
        out
    }

    fn place(&self) -> Result<Set<At>, Why> {
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        let mut at: Set<At> = Set::default();
        loop {
            let out = self.flow(&at);
            let mut more: Vec<At> = Vec::new();
            // Clashes left in divergent code: (earlier, later).
            let mut left: Vec<(usize, usize)> = Vec::new();
            for &b in &order {
                // One block's barriers at a time: they change what is open
                // in every block after it.
                if !more.is_empty() {
                    break;
                }
                let open = self.p.blocks[b].preds.iter().fold(0, |a, &q| a | out[q]);
                let open = self.walk(b, open, &at, |x, open| {
                    let clash = self.clashing(open, x);
                    if clash == 0 {
                        return false;
                    }
                    if self.once[b] {
                        more.push(self.accesses[x].at);
                        return true;
                    }
                    left.extend(
                        (0..self.accesses.len())
                            .filter(|y| clash >> y & 1 == 1)
                            .map(|y| (y, x)),
                    );
                    false
                });
                // Divergent code after a block every lane runs starts from
                // synced LDS where anything ahead could clash.
                let end = (b, self.p.blocks[b].insts.len());
                if self.once[b] && open != 0 && self.succ(b).iter().any(|&s| !self.once[s]) {
                    let ahead = self.ahead(b);
                    if (0..self.accesses.len())
                        .any(|x| ahead >> x & 1 == 1 && self.clashing(open, x) != 0)
                    {
                        more.push(end);
                    }
                }
            }
            if more.is_empty() {
                self.flags(&left)?;
                return Ok(at);
            }
            at.extend(more);
        }
    }

    /// Checks that every clash left in divergent code is a flag's (the
    /// module notes).
    fn flags(&self, left: &[(usize, usize)]) -> Result<(), Why> {
        let mut reads: Vec<(usize, usize)> = Vec::new();
        for &(y, x) in left {
            let (a, b) = (&self.accesses[y], &self.accesses[x]);
            if self.once[a.at.0] || self.once[b.at.0] {
                return Err(DIVERGENT);
            }
            let (read, write) = match (a.kind, b.kind) {
                (Kind::Read, Kind::Write) => (y, x),
                (Kind::Write, Kind::Read) => (x, y),
                _ => return Err(DIVERGENT),
            };
            let (r, w) = (&self.accesses[read], &self.accesses[write]);
            let (Some(rd), Some(wd)) = (&r.dwords, &w.dwords) else {
                return Err(DIVERGENT);
            };
            // Every dword the write stores is a constant the read covers.
            if !wd
                .iter()
                .all(|&(d, k)| k.is_some() && rd.iter().any(|x| x.0 == d))
            {
                return Err(DIVERGENT);
            }
            reads.push((read, write));
        }
        reads.sort_unstable();
        reads.dedup();
        let mut checked: Vec<usize> = Vec::new();
        for &(read, _) in &reads {
            if !checked.contains(&read) {
                checked.push(read);
                let writes: Vec<usize> = (reads.iter())
                    .filter(|p| p.0 == read)
                    .map(|p| p.1)
                    .collect();
                self.flag_read(read, &writes)?;
            }
        }
        Ok(())
    }

    /// A flag read decides nothing but the flag: from it to the next block
    /// every lane runs, the only effects are the flag's own writes
    /// (`writes`), and no value made there is read outside.
    fn flag_read(&self, read: usize, writes: &[usize]) -> Result<(), Why> {
        let p = self.p;
        let (rb, ri) = self.accesses[read].at;
        let n = p.blocks.len();
        // The blocks a lane may run after the read before it is back in a
        // block every lane runs.
        let mut after = vec![false; n];
        let mut stack = self.succ(rb);
        while let Some(b) = stack.pop() {
            if self.once[b] || std::mem::replace(&mut after[b], true) {
                continue;
            }
            stack.extend(self.succ(b));
        }
        // Instruction `i` of block `b` is in that stretch.
        let inside = |b: usize, i: usize| after[b] || b == rb && i >= ri;
        let mut made: HashSet<Value> = HashSet::new();
        for (b, block) in p.blocks.iter().enumerate() {
            if after[b] {
                if block.exit.successors().is_empty() {
                    return Err(SHAPE);
                }
                made.extend(block.phis.iter().map(|phi| phi.value));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if !inside(b, i) {
                    continue;
                }
                made.extend(inst.writes.iter().map(|w| w.1));
                let own = (self.by_place.get(&(b, i)))
                    .is_some_and(|x| self.accesses[*x].kind == Kind::Read || writes.contains(x));
                if effect(inst) && !own {
                    return Err(FLAG_USED);
                }
            }
        }
        // Outside the stretch three kinds of values are not what the console
        // has: those made in it (`made`), VGPR values written under a
        // condition over such a value (`partly`: right in the lanes where
        // the condition holds, which are the lanes that wrote them), and
        // compare results of such values under that condition (`masks`).
        let mut partly: Map<Value, Bool> = Map::default();
        let mut masks: Map<Value, Bool> = Map::default();
        loop {
            let mut grew = false;
            for (b, block) in p.blocks.iter().enumerate() {
                if after[b] {
                    continue;
                }
                // A phi the stretch reaches: the lane's way out picks the
                // value, unless every way out carries the same one.
                for phi in &block.phis {
                    if made.contains(&phi.value) {
                        continue;
                    }
                    let mut out = (block.preds.iter().zip(&phi.args))
                        .filter(|&(&q, _)| after[q] || q == rb)
                        .map(|(_, a)| *a);
                    let first = out.next();
                    let picked = first.is_some_and(|f| out.any(|a| a != f));
                    let wrong = |a: &Value| {
                        made.contains(a) || partly.contains_key(a) || masks.contains_key(a)
                    };
                    if picked || phi.args.iter().any(wrong) {
                        made.insert(phi.value);
                        grew = true;
                    }
                }
                for (i, inst) in block.insts.iter().enumerate() {
                    if inside(b, i) {
                        continue;
                    }
                    let Some(g) = self.st.guards[b][i] else {
                        continue;
                    };
                    let (_, kept) = implicit(inst);
                    let mut under = false;
                    for &(reg, v) in &inst.reads {
                        if kept.contains(&reg) {
                            // Lanes outside the condition keep the old value.
                            if (made.contains(&v) || partly.contains_key(&v))
                                && let Some(new) = inst.write(reg)
                                && !made.contains(&new)
                                && !partly.contains_key(&new)
                            {
                                partly.insert(new, g);
                                grew = true;
                            }
                        } else if partly.get(&v) == Some(&g) {
                            under = true;
                        }
                    }
                    if under && inst.name().starts_with("v_cmp") {
                        for w in inst.writes.iter().filter(|w| !matches!(w.0, Reg::V(_))) {
                            grew |= masks.insert(w.1, g).is_none();
                        }
                    }
                }
            }
            if !grew {
                break;
            }
        }
        // A condition that turns on a value the stretch made, or on such a
        // compare result where its own condition does not hold.
        let turns = |c: Bool| {
            (self.st.bools.support(c).into_iter()).any(|a| match a {
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => {
                    made.contains(&v)
                        || partly.contains_key(&v)
                        || masks
                            .get(&v)
                            .is_some_and(|&g| !self.st.bools.ignores(c, a, g))
                }
                Atom::Constant(_) => false,
            })
        };
        let mask_of = |v: Value| {
            let low = self.st.highs.get(&v).copied().unwrap_or(v);
            self.st.masks.get(&low).or(self.st.other.get(&low)).copied()
        };
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                if inside(b, i) {
                    continue;
                }
                let guard = self.st.guards[b][i];
                let (exec, kept) = implicit(inst);
                for &(reg, v) in &inst.reads {
                    if exec.contains(&reg) || kept.contains(&reg) {
                        continue;
                    }
                    let partly_used = match partly.get(&v) {
                        // Its lanes hold the right value: for them alone,
                        // and what they make of it beyond VGPRs is followed
                        // for compares only.
                        Some(g) => {
                            guard != Some(*g)
                                || across_lanes(inst)
                                || !inst.name().starts_with("v_cmp")
                                    && inst.writes.iter().any(|w| !matches!(w.0, Reg::V(_)))
                        }
                        None => false,
                    };
                    let mask_used = match mask_of(v) {
                        Some(c) => turns(c),
                        None => masks.contains_key(&v),
                    };
                    if made.contains(&v) || partly_used || mask_used {
                        return Err(FLAG_USED);
                    }
                }
                if guard.is_some_and(turns) {
                    return Err(FLAG_USED);
                }
            }
            if !after[b]
                && b != rb
                && let LaneExit::Branch { when, .. } = self.st.exits[b]
                && turns(when)
            {
                return Err(FLAG_USED);
            }
        }
        Ok(())
    }
}

/// Whether an instruction reads what other lanes hold.
fn across_lanes(inst: &Inst) -> bool {
    let name = inst.name();
    name.contains("lane")
        || name.contains("permute")
        || name.contains("swizzle")
        || name.starts_with("v_mbcnt")
        || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some())
}

/// The registers a vector instruction reads without using what they hold
/// as an operand: EXEC (first), which the per-lane form has turned into the
/// instruction's guard and the lane's branches; and the VGPRs it writes
/// whole (second), whose old value the IR reads for the lanes the
/// instruction's condition leaves out. Without those an instruction names
/// as an operand; nothing for kinds not looked into here.
fn implicit(inst: &Inst) -> (Vec<Reg>, Vec<Reg>) {
    let name = inst.name();
    let mut named: Vec<&Operand> = Vec::new();
    let whole = match &inst.fields {
        Fields::Valu(v) => {
            named.extend(&v.src);
            v.sdwa.is_none()
                && v.dpp.is_none()
                && !name.contains("lane")
                && !name.contains("rel")
                && name != "v_swap_b32"
        }
        Fields::Ds { addr, data, .. } => {
            named.extend(addr.iter().chain(data));
            !name.contains("d16")
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            ..
        } => {
            named.extend(data.iter().filter(|_| !name.contains("_load_")));
            named.extend([addr, rsrc, soffset]);
            !name.contains("d16")
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            d16,
            ..
        } => {
            let loads = !(name.contains("_store") || name.contains("_atomic_"));
            named.extend(Some(data).filter(|_| !loads));
            named.extend(addr.iter().chain([rsrc]).chain(sampler));
            loads && !d16
        }
        _ => return (Vec::new(), Vec::new()),
    };
    let within = |first: u8, dwords: u8, x: u8| {
        x >= first && u16::from(x) < u16::from(first) + u16::from(dwords)
    };
    let names = |reg: Reg| {
        named.iter().any(|op| match (**op, reg) {
            (Operand::Vector { reg: r, dwords }, Reg::V(x)) => within(r, dwords, x),
            (Operand::Scalar { code, dwords }, Reg::S(x)) => within(code, dwords, x),
            (Operand::Const(252), Reg::S(x)) => x == EXEC || x == EXEC + 1,
            _ => false,
        })
    };
    let mut exec = vec![Reg::S(EXEC), Reg::S(EXEC + 1)];
    exec.retain(|&r| !names(r));
    let mut kept = Vec::new();
    if whole {
        kept.extend(
            (inst.writes.iter().map(|w| w.0)).filter(|&r| matches!(r, Reg::V(_)) && !names(r)),
        );
    }
    (exec, kept)
}

/// Whether an instruction changes anything but registers (as
/// [`Lower::inst`] tells effects).
fn effect(inst: &Inst) -> bool {
    let name = inst.name();
    matches!(inst.fields, Fields::Exp { .. })
        || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
        || name.contains("store")
        || name.contains("atomic")
        || name == "s_barrier"
        || name == "s_sendmsg"
        || matches!(inst.fields, Fields::Ds { .. })
            && !(name.starts_with("ds_read")
                || name.starts_with("ds_swizzle")
                || name.contains("permute"))
}

/// The barriers for `x`'s group of `group` threads with LDS, where
/// [`Mac::lds_order`] asks for them and the group is one console wave that
/// a host of narrower waves splits; `None` where the setting is off or does
/// not apply.
pub(super) fn wanted(x: &Parts, group: u32, lds_dwords: u32) -> Option<Result<Set<At>, Why>> {
    let wave = if x.native.wave64 { 64 } else { 32 };
    (x.mac.lds_order
        && (x.narrow_waves || x.host_wave32)
        && group > 32
        && group <= wave
        && lds_dwords > 0)
        .then(|| barriers(x.p, x.st))
}
