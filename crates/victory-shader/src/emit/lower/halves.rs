//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact. Reads into the other half pass
//! between the two host waves through group-shared memory, between group
//! barriers ([`Lower::exchange_halves`]); barriers need the whole group, so
//! only in a block every lane of the group runs together, the same number
//! of times ([`in_step`]: once, or in a loop on a count every host wave
//! computes alike). Elsewhere they are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), and `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code),
//!   `v_writelane_b32` (the lane named takes the scalar) and the row a DPP
//!   row mask names (rows 2 and 3 are the upper half's) are decided by the
//!   console lane ([`Lower::console_lane`]);
//! * lane masks read as data (ballots) hold the other half's lanes: unless
//!   the whole group fits one host wave, the halves pass each other their
//!   ballots ([`Lower::ballot_across_halves`]), for a mask written in such a
//!   block; elsewhere refused ([`MASK_ACROSS`]);
//! * `v_readlane_b32` of a lane that may be in the other half (SGPR spills
//!   read back are not lane reads): each half passes the value of the lane
//!   with that index in it, and the console lane's half picks
//!   ([`Lower::read_lane_across`]);
//! * whole-wave reductions and prefix sums: each half reduces its own lanes,
//!   and they pass each other the result; a prefix in the upper half adds
//!   the lower half's total ([`Lower::reduce_across`]);
//! * `v_readfirstlane_b32` reads the host wave's first lane, which in the
//!   upper half is console lane 32 or later: kept as it is for a value the
//!   same in every lane of the console wave ([`lane_uniform`], what the
//!   compiler reads first lanes for) and in a group of 32 threads or fewer
//!   (one host wave holds the whole console wave); otherwise the lower
//!   half's first lane ([`Lower::first_lane_across`]); refused where no
//!   exchange can go ([`FIRST_ACROSS`]).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! Half-local reads ([`Local::reads`], `emit::Options::halves_local`, off
//! unless asked for): instead of those refusals each half answers from its
//! own 32 lanes, as if it were the whole wave. A mask read as data holds
//! the half's lanes at their console bits and none of the other half's
//! ([`Lower::half_ballot`]); a lane read names that lane of the reader's
//! half ([`Lower::own_half_lane`]); a reduction, prefix or first-lane read
//! covers the half. Exact where a wave's halves never needed each other
//! (a lane's own bit, counts and prefixes below a lane within the half, a
//! read of a lane the reader's half holds), an approximation elsewhere: a
//! count over the wave is the half's, a wave's one result becomes two. The
//! idea is upstream PR #2711's (moe-miau), written there on #2176's halves,
//! whose lanes are numbered 0 to 31 in both halves. Here the upper half's
//! lanes stay console lanes 32 to 63, so its ballot goes into the mask's
//! high dword instead of leaving that dword 0.
//!
//! LDS: the console compiler leaves out barriers between the lanes of one
//! wave, and so between all of a group's threads only when the group is one
//! wave. A group of 33 to 64 threads with LDS (two host waves sharing what
//! the console's one wave ordered by itself) is refused; groups of several
//! console waves order their LDS with barriers, and those of 32 threads or
//! fewer are one host wave. [`super::lds_order`] finds an exact order for some of
//! the refused ones (`emit::Options::lds_order`).
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

/// What a program in halves does where the console's one wave reached
/// across its two halves, in place of a refusal. All off (the default):
/// every such program is refused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Local {
    /// Reads across the halves are answered from the reader's own half
    /// (`emit::Options::halves_local`; the module notes, "Half-local
    /// reads").
    pub reads: bool,
    /// LDS a group of one console wave64 shares is left as it is: the two
    /// host waves read and write it with nothing ordering them, where the
    /// console's one wave ran its lanes in step
    /// (`emit::Options::halves_shared_lds`). A race, so a test, not a fix.
    pub lds: bool,
}

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Why a program in halves cannot run: LDS a group of one console wave
    /// shares without barriers, split across two host waves (unless asked
    /// to run it so, [`Local::lds`]).
    pub(super) fn halves_lds(x: &Parts, group: u32, lds_dwords: u32) -> Option<Why> {
        (lds_dwords > 0 && group > HALF && group <= 64 && !x.halves_local.lds)
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

    /// Where the halves pass each other a lane mask written in IR block
    /// `def` (none: a program input, in the entry): there, if every lane of
    /// the group runs it together ([`in_step`]); in the block being
    /// lowered, where the reading instruction starts ([`Lower::level`],
    /// before any guard of its own).
    pub(super) fn across_at(&self, def: Option<usize>) -> Option<usize> {
        if !self.step.get(def.unwrap_or(0)).copied().unwrap_or(false) {
            return None;
        }
        match def {
            None => Some(self.s.entry),
            Some(b) if b == self.cur_ir => Some(self.level),
            Some(b) => self.tail[b],
        }
    }

    /// Whether the halves can pass each other values where the instruction
    /// being lowered is: in halves, in a group of more than one host wave, in
    /// a block every lane of the group runs together ([`in_step`]), before
    /// anything of the instruction's own split it ([`Lower::level`]).
    pub(super) fn across_here(&self) -> bool {
        self.halves
            && !self.one_host_wave()
            && self.cur == self.level
            && self.step.get(self.cur_ir).copied().unwrap_or(false)
    }

    /// The half of the console wave this lane is in: 0 or 1.
    fn half(&mut self, at: usize) -> V {
        let f = self.relabelled_flat();
        let (five, one) = (self.s.m.i32(5), self.s.m.i32(1));
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        self.s.m.bin(at, Bin::And, slot, one)
    }

    /// Both dwords of the console wave's ballot of `bit`, in halves: each
    /// host wave's own (its half's lanes), passed between them
    /// ([`Lower::exchange_halves`]). A console wave with no upper half has a
    /// high dword of 0. Every lane of the group must run `at` together
    /// ([`Lower::across_at`]).
    pub(super) fn ballot_across_halves(&mut self, bit: V, at: usize) -> Result<(V, V), Why> {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.s.m.extract(at, r, 0);
        let zero = self.s.m.i32(0);
        self.exchange_halves(own, zero, at)
    }

    /// What each half of this lane's console wave gave as `own` (a dword
    /// the same in every lane of a host wave): (lower half's, upper half's).
    /// Each host wave stores its own in a group-shared dword of its own, and
    /// reads its console wave's two after a group barrier; a barrier before
    /// the store keeps it from overwriting what the other half has not read
    /// of an earlier exchange. A console wave with no upper half (a group not
    /// a multiple of 64) reads `missing` for it. Every lane of the group must
    /// run `at` together ([`in_step`]), in halves.
    pub(super) fn exchange_halves(&mut self, own: V, missing: V, at: usize) -> Result<(V, V), Why> {
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        // Two dwords per console wave, so the high dword of the last one is
        // inside even where the group has no upper half.
        let dwords = group.div_ceil(2 * HALF) * 2;
        if self.s.lds_dwords + dwords > GROUP_SHARED_DWORDS {
            return Err("no group-shared memory left to pass values between 32-lane halves");
        }
        let i32t = self.s.t.i32;
        let f = self.relabelled_flat();
        let words = match self.half_words {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(dwords, i32t);
                let g = self.s.m.global("half_words", arr, 3);
                self.half_words = Some(g);
                g
            }
        };
        // Host wave `2w + h` (its flat ids `32 (2w + h)` on) holds console
        // wave `w`'s half `h`.
        let five = self.s.m.i32(5);
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        let not_one = self.s.m.i32(!1);
        let low = self.s.m.bin(at, Bin::And, slot, not_one);
        let one = self.s.m.i32(1);
        let high = self.s.m.bin(at, Bin::Or, slot, one);
        let sync = |l: &mut Self| {
            // Group sync with a group-shared memory fence.
            let mode = l.s.m.i32(1 | 8);
            let void = l.s.t.void;
            l.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        };
        sync(self);
        let mine = self.s.m.element(at, words, slot);
        self.s.m.store(at, mine, own);
        sync(self);
        let lo = self.s.m.element(at, words, low);
        let lo = self.s.m.load(at, lo);
        let hi = self.s.m.element(at, words, high);
        let mut hi = self.s.m.load(at, hi);
        if !group.is_multiple_of(2 * HALF) {
            let first = self.s.m.bin(at, Bin::Shl, high, five);
            let n = self.s.m.i32(group);
            let there = self.s.m.cmp(at, pred::IULT, first, n);
            hi = self.s.m.select(at, there, hi, missing);
        }
        Ok((lo, hi))
    }

    /// `v_readlane_b32` of `x` at console lane `lane` (the low 6 bits), in
    /// halves ([`Lower::across_here`]): each half reads the lane with that
    /// index in it, and the lane's own half gives the value.
    pub(super) fn read_lane_across(&mut self, x: V, lane: V) -> Result<V, Why> {
        let c = self.cur;
        let (k31, k32, zero) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        let index = self.s.m.bin(c, Bin::And, lane, k31);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.read_lane(x, index);
        let (lo, hi) = self.exchange_halves(own, zero, c)?;
        let upper = self.s.m.bin(c, Bin::And, lane, k32);
        let upper = self.s.m.cmp(c, pred::NE, upper, zero);
        Ok(self.s.m.select(c, upper, hi, lo))
    }

    /// `v_readfirstlane_b32` of `x`, in halves ([`Lower::across_here`]): the
    /// lower half's first lane. Every lane of the group runs the block, so
    /// the console's first lane is in the lower half.
    pub(super) fn first_lane_across(&mut self, x: V) -> Result<V, Why> {
        let c = self.cur;
        let (i32t, zero) = (self.s.t.i32, self.s.m.i32(0));
        self.s.flags |= flags::WAVE_OPS;
        let own = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        Ok(self.exchange_halves(own, zero, c)?.0)
    }

    /// A whole-wave reduction or prefix of `op`, in halves
    /// ([`Lower::across_here`]): `own` is this host wave's (its half's
    /// lanes), `total` its half's total (a prefix's; the reduction itself
    /// otherwise), both as the operation's type. A reduction takes `op` of
    /// both halves' totals; a prefix in the upper half adds the lower half's
    /// total. A console wave with no upper half reads `ident` for it.
    pub(super) fn reduce_across(
        &mut self,
        op: super::super::wave::Op,
        own: V,
        total: V,
        prefix: bool,
        ident: u32,
    ) -> Result<V, Why> {
        use super::super::wave::Op;
        let c = self.cur;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let bits = if float { self.as_int(total) } else { total };
        let missing = self.s.m.i32(ident);
        let (lo, hi) = self.exchange_halves(bits, missing, c)?;
        let (lo, hi) = if float {
            (self.as_float(lo), self.as_float(hi))
        } else {
            (lo, hi)
        };
        if prefix {
            let h = self.half(c);
            let zero = self.s.m.i32(0);
            let upper = self.s.m.cmp(c, pred::NE, h, zero);
            let added = self.s.m.bin(c, Bin::Add, lo, own);
            return Ok(self.s.m.select(c, upper, added, own));
        }
        let unsigned_or_signed = |l: &mut Self, p: u8| {
            let less = l.s.m.cmp(c, p, lo, hi);
            l.s.m.select(c, less, lo, hi)
        };
        Ok(match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinU => unsigned_or_signed(self, pred::IULT),
            Op::MaxU => unsigned_or_signed(self, pred::IUGT),
            Op::MinI => unsigned_or_signed(self, pred::SLT),
            Op::MaxI => unsigned_or_signed(self, pred::SGT),
            // FMin, FMax: as the host's own wave minimum and maximum.
            Op::MinF => self.binary(36, lo, hi),
            Op::MaxF => self.binary(35, lo, hi),
            Op::AddH => return Err("a 16-bit wave reduction over both 32-lane halves"),
        })
    }

    /// Whether the Mac port's hand-over between the halves
    /// ([`Lower::across_here`], [`Lower::across_at`]; VIctory-macOS #309,
    /// #360, #376) answers a read across them before half-local reads do:
    /// where it is asked for ([`Mac::across`]), and where there are no
    /// half-local reads, so that nothing else would answer and the read
    /// would be refused. With half-local reads and without the switch a
    /// translation is what it was before the hand-over came to this line.
    /// This line's own exchange ([`super::exchange`]) comes before either.
    ///
    /// Not while the walk is on ([`Mac::walk`]) unless asked for: the walk
    /// (`emit::walk`) translates a program without half-local reads to learn
    /// whether every read across the halves is exact yet, and reads a
    /// refusal as "not yet, open this block to every lane". This hand-over
    /// in place of the refusal would end the walk early, and in a program
    /// the walk opened its first-lane rule (every lane of the group has its
    /// EXEC bit set) does not hold.
    pub(super) fn across_first(&self) -> bool {
        let x = self.x;
        self.halves && (x.mac.across || !x.halves_local.reads && !x.mac.walk)
    }

    /// Refuses a read into the other half of the console wave in halves,
    /// unless such reads are answered from the reader's own half
    /// ([`Local::reads`]).
    pub(super) fn same_half(&self, what: Why) -> Result<(), Why> {
        if self.halves && !self.x.halves_local.reads {
            Err(what)
        } else {
            Ok(())
        }
    }

    /// A host wave's ballot (its dwords `lo`, `hi`, taken in block `at`) as
    /// the console wave's mask. In halves the host wave is one half and its
    /// 32 bits are in `lo`: the lower half's are the mask's low dword, the
    /// upper half's (console lanes 32 to 63, [`Lower::console_lane`]) its
    /// high dword, and the other half's lanes are missing. Reached in
    /// halves only with half-local reads ([`Lower::same_half`]).
    pub(super) fn half_ballot(&mut self, at: usize, (lo, hi): (V, V)) -> (V, V) {
        if !self.halves {
            return (lo, hi);
        }
        let lane = self.console_lane();
        let (half, zero) = (self.s.m.i32(HALF), self.s.m.i32(0));
        let upper = self.s.m.cmp(at, pred::IUGE, lane, half);
        (
            self.s.m.select(at, upper, zero, lo),
            self.s.m.select(at, upper, lo, zero),
        )
    }

    /// The host lane a read of console lane `lane` reads. In halves (with
    /// half-local reads, [`Lower::same_half`]) that lane of the reader's own
    /// half: the lane itself where the reader's half holds it.
    pub(super) fn own_half_lane(&mut self, lane: V) -> V {
        if !self.halves {
            return lane;
        }
        let m = self.s.m.i32(HALF - 1);
        self.s.m.bin(self.cur, Bin::And, lane, m)
    }

    /// The word and bit of console lane `lane` in a host wave's ballot, and
    /// the host lane to read it at ([`Lower::own_half_lane`] for a lane
    /// known here).
    pub(super) fn ballot_bit(&self, lane: u32) -> (u32, u32, u32) {
        ballot_bit(lane, self.halves)
    }
}

/// [`Lower::ballot_bit`]: in halves the host wave's own lane `lane & 31`,
/// whose bit is in the ballot's first word (a host wave of 32 lanes has no
/// second).
fn ballot_bit(lane: u32, halves: bool) -> (u32, u32, u32) {
    let at = if halves { lane & (HALF - 1) } else { lane };
    (at / 32, at % 32, at)
}

/// Why a lane mask read as data stops in halves: it is written where the
/// lanes of the group may not all be, as often, so no group barrier can pass
/// it between the halves.
pub(super) const MASK_ACROSS: Why = "a lane mask read as data (wave64 on a 32-lane host)";

/// The group-shared dwords a D3D12 compute program may declare, at most
/// (32 KiB).
const GROUP_SHARED_DWORDS: u32 = 8192;

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, and the value is not known to be the same there.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)";

/// Every branch (by its block) with its region: the blocks reachable from
/// it that do not post-dominate it, which only the lanes taking one side may
/// run (the branch's own block among them where a cycle leads back to it).
/// None with an indirect exit.
pub(super) fn branch_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
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
        if !matches!(e, LaneExit::Branch { .. }) {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            // The branch's own block is in its region if a cycle leads back
            // to it (a loop's latch): only the lanes going round run it.
            if seen[b] || b != d && pdom[d][b / 64] >> (b % 64) & 1 == 1 {
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
    let regions = branch_regions(st);
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

#[cfg(test)]
mod tests {
    use super::ballot_bit;

    /// Whole, console lane 33 is bit 1 of the ballot's second word and is
    /// read at lane 33. In halves the host wave has 32 lanes and one ballot
    /// word: the lane asked about and the lane read are the same lane of
    /// the host wave's own half (after upstream PR #2711's
    /// `a_half_scans_ballot_bit_is_the_lane_it_reads`).
    #[test]
    fn a_ballot_is_asked_about_the_lane_that_is_read() {
        assert_eq!(ballot_bit(33, false), (1, 1, 33));
        assert_eq!(ballot_bit(31, false), (0, 31, 31));
        assert_eq!(ballot_bit(33, true), (0, 1, 1));
        assert_eq!(ballot_bit(1, true), (0, 1, 1));
        assert_eq!(ballot_bit(63, true), (0, 31, 31));
    }
}
