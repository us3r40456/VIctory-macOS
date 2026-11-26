//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact; reads into the other half have no
//! host form and are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), and `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code),
//!   `v_writelane_b32` (the lane named takes the scalar) and the row a DPP
//!   row mask names (rows 2 and 3 are the upper half's) are decided by the
//!   console lane ([`Lower::console_lane`]);
//! * across halves, refused: ballots (a mask read as data holds the other
//!   half's lanes), `v_readlane_b32` of a lane that may be in the other half
//!   (SGPR spills read back are not lane reads), and whole-wave reductions
//!   and prefix sums;
//! * `v_readfirstlane_b32` reads the host wave's first lane, which in the
//!   upper half is console lane 32 or later: kept for a value the same in
//!   every lane of the console wave ([`lane_uniform`], what the compiler
//!   reads first lanes for) and in a group of 32 threads or fewer (one host
//!   wave holds the whole console wave), refused otherwise ([`FIRST_ACROSS`]).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! LDS: the console compiler leaves out barriers between the lanes of one
//! wave, and so between all of a group's threads only when the group is one
//! wave. A group of 33 to 64 threads with LDS (two host waves sharing what
//! the console's one wave ordered by itself) is refused; groups of several
//! console waves order their LDS with barriers, and those of 32 threads or
//! fewer are one host wave.
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name. Which values are the same in every lane of the console
//! wave ([`lane_uniform`]): justrealyme's analysis (chippy01/VIctory#1),
//! ported from fxceless/victory-fork (#497 there).

use super::*;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Why a program in halves cannot run: LDS a group of one console wave
    /// shares without barriers, split across two host waves.
    pub(super) fn halves_lds(group: u32, lds_dwords: u32) -> Option<Why> {
        (lds_dwords > 0 && group > HALF && group <= 64)
            .then_some("LDS one console wave64 shares without barriers, on a host of 32-lane waves")
    }

    /// The flat thread id, relabelled so that each host wave holds a run of
    /// [`Shader::wave_size`]-aligned ids from its first lane (see
    /// [`Lower::thread_id`]); made once, in the entry block.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Whether the group fits one host wave: a console wave64 of 32 threads
    /// or fewer has no lane in the upper half.
    pub(super) fn one_host_wave(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() <= HALF)
    }

    /// Refuses a read into the other half of the console wave in halves.
    pub(super) fn same_half(&self, what: Why) -> Result<(), Why> {
        if self.halves { Err(what) } else { Ok(()) }
    }
}

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, and the value is not known to be the same there.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)";

/// Every per-lane branch (by its block) with its region: the blocks
/// reachable from it that do not post-dominate it, which only some lanes
/// run. Not branches on uniform conditions. None with an indirect exit.
fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // Post-dominators, as bitsets (a virtual exit after every end).
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for (w, p) in v.iter_mut().zip(&pdom[t]) {
                    *w &= p;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else {
            continue;
        };
        if st
            .bools
            .support(when)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(_)))
        {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The thread id's x (`v0` on entry) of a wave64 compute program whose
/// console waves are runs of 64 ids along x (a group 64-aligned along x,
/// or one row): `id >> 6` and up is one value per console wave.
fn wave_tid(x: &Parts) -> Option<Value> {
    let StageIo::Cs(cs) = &x.native.io else {
        return None;
    };
    let [tx, ty, tz] = cs.threads;
    if !x.native.wave64 || !(tx % 64 == 0 || ty * tz == 1) {
        return None;
    }
    x.p.inputs
        .iter()
        .find(|(r, _)| *r == Reg::V(0))
        .map(|(_, v)| *v)
}

/// VGPR values that are the same in every lane of a console wave holding
/// them, so that a first-lane read in either half gives the console's
/// value: written lane by lane from scalars, constants and such values, by
/// an instruction every arriving lane runs or that runs under a condition
/// itself the same in every lane (then only over an old value that is such
/// a value too); loaded from LDS, or from a buffer through a scalar
/// descriptor, at such an address; or the thread id's x shifted right past
/// the console wave ([`wave_tid`]). The console compiler keeps wave-uniform
/// data in VGPRs and fetches it with `v_readfirstlane` (a typed buffer's
/// element, say, which only a vector load reads). A scalar is one value,
/// except the dwords a waterfall loop reads, which are each lane's own.
///
/// A phi counts where every predecessor is a block every lane of the wave
/// reaches together: no branch on a lane-dependent condition can have split
/// them (a branch on alike values does not). The analysis runs to a fixed
/// point from nothing, so a loop-carried value that needs itself to be
/// alike never is.
pub(super) fn lane_uniform(x: &Parts) -> HashSet<Value> {
    let (p, st) = (x.p, x.st);
    let wave_tid = wave_tid(x);
    let regions = lane_regions(st);
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let per_lane = |v: Value| x.res.waterfall.reads.contains(&v);
    let mut same: HashSet<Value> = HashSet::new();
    loop {
        let alike_atom = |same: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => same.contains(&v),
            _ => false,
        };
        let alike_bool = |same: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| alike_atom(same, a))
        };
        // Blocks every lane of a console wave reaches together.
        let wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for (d, region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[*d] else {
                        continue;
                    };
                    if alike_bool(&same, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => vec![false; st.exits.len()],
        };
        let mut next: HashSet<Value> = HashSet::new();
        for b in crate::ir::reverse_postorder(&p.blocks) {
            let block = &p.blocks[b];
            let whole = block
                .preds
                .iter()
                .all(|&q| wave_all.get(q).copied().unwrap_or(false));
            for phi in &block.phis {
                if matches!(phi.reg, Reg::V(_))
                    && whole
                    && phi
                        .args
                        .iter()
                        .all(|a| next.contains(a) || same.contains(a))
                {
                    next.insert(phi.value);
                }
            }
            let known = |next: &HashSet<Value>, v: &Value| next.contains(v) || same.contains(v);
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Under a condition: only one every lane holds alike, and
                // over an old value that is alike too.
                if let Some(g) = st.guards[b][i] {
                    if !alike_bool(&same, g) {
                        continue;
                    }
                    let old_alike = inst
                        .writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .all(|w| inst.read(w.0).is_none_or(|old| known(&next, &old)));
                    if !old_alike {
                        continue;
                    }
                }
                // A buffer load at an alike index and offset, through a scalar
                // descriptor, reads one value in every lane.
                if let Fields::Buffer {
                    addr,
                    rsrc,
                    soffset,
                    offen,
                    idxen,
                    ..
                } = &inst.fields
                    && (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                {
                    let vector = |op: &Operand| match *op {
                        Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                            inst.read(Reg::V(reg + k))
                                .is_some_and(|v| known(&next, &v))
                        }),
                        _ => true,
                    };
                    let scalar = |op: &Operand| match *op {
                        Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                            .all(|k| inst.read(Reg::S(code + k)).is_none_or(|v| !per_lane(v))),
                        _ => true,
                    };
                    if (!(*offen || *idxen) || vector(addr)) && scalar(rsrc) && scalar(soffset) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // An LDS read at an alike address reads one value in every lane.
                if let Fields::Ds {
                    addr: Some(Operand::Vector { reg, .. }),
                    dst: Some(_),
                    data,
                    ..
                } = &inst.fields
                    && name.starts_with("ds_read")
                    && data.is_empty()
                {
                    if inst.read(Reg::V(*reg)).is_some_and(|a| known(&next, &a)) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                let ops: Vec<Operand> = match &inst.fields {
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane") =>
                    {
                        v.src.clone()
                    }
                    _ => continue,
                };
                // The thread id's x shifted past the console wave: one value
                // per console wave (the b32 shift uses five bits of its count).
                if let (
                    Some(tid),
                    "v_lshrrev_b32",
                    [
                        Operand::Const(k @ 128..=192),
                        Operand::Vector { reg, .. },
                        ..,
                    ],
                ) = (wave_tid, name, ops.as_slice())
                    && ((k - 128) & 31) >= 6
                    && inst.read(Reg::V(*reg)) == Some(tid)
                {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                    continue;
                }
                let alike = ops.iter().all(|op| match *op {
                    Operand::Vector { reg, .. } => {
                        // every dword the instruction reads from that register on
                        let read: Vec<Value> = inst
                            .reads
                            .iter()
                            .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                            .map(|r| r.1)
                            .collect();
                        !read.is_empty() && read.iter().all(|x| known(&next, x))
                    }
                    // A scalar is one value; a lane mask is not, unless a
                    // compare of alike values made it.
                    Operand::Scalar { code, .. } if code != NULL => inst
                        .read(Reg::S(code))
                        .is_none_or(|x| !is_mask(x) && !per_lane(x) || known(&next, &x)),
                    _ => true,
                });
                if alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) {
            return same;
        }
        same.extend(next);
    }
}
