//! Settings of the Mac line (`emit::Options`), all off by default, in a
//! struct and a file of their own so that a new one needs no new field in
//! the test initialisers and little of `lower.rs`.
//!
//! Early depth-stencil ([`Mac::early_depth`], `emit::Options::early_depth`):
//! a pixel program whose `DB_SHADER_CONTROL` has `DEPTH_BEFORE_SHADER` (bit
//! 12) set gets DXIL's `ForceEarlyDepthStencil` shader flag (bit 3, HLSL's
//! `[earlydepthstencil]`). Both say the same thing: the depth and stencil
//! tests *and their writes* are done before the program runs, a fragment
//! that fails them never runs it (so it writes no image and no colour), and
//! a discard only takes back the colour outputs and the image writes after
//! it, not the depth write. The console's compiler sets the bit for exactly
//! that attribute; Mesa's AMD drivers set it for the same meaning (GL's
//! `early_fragment_tests`), together with `EXEC_ON_NOOP` (bit 10) when the
//! program writes memory and `KILL_ENABLE` (bit 6) when it discards. Grand
//! Theft Auto VI's see-through passes carry it (0x1450 = bits 4, 6, 10, 12
//! in the alpha pre-pass; `notes/hair.md`). Without the flag D3D12 runs a
//! program that writes images or discards before the depth test, so
//! fragments hidden behind opaque geometry record into the pre-pass images.
//!
//! The rule is narrower than "bit 12 set" where the two definitions could
//! differ or the host could refuse the program ([`early_depth`]):
//!
//! * not when the register lets the program export depth, a stencil value
//!   or a coverage mask (bits 0, 1, 2 and 8), nor when the translated
//!   program has such an output (`SV_Depth*`, `SV_StencilRef`,
//!   `SV_Coverage`): with the flag D3D ignores a depth output, and Metal
//!   refuses a function with early tests that returns depth or stencil;
//! * not when the program reads the coverage mask (`SPI_PS_INPUT_ADDR` bit
//!   14): under the flag D3D hands it the coverage left after the depth
//!   test, the console the rasterizer's unless another bit asks for that.
//!
//! No program of the title with bit 12 is in either group (3,023 pixel
//! programs counted, `notes/hair.md`), so there the rule is "bit 12". A
//! discard needs no exclusion: both definitions keep the early depth write
//! of a discarded fragment.
//!
//! Lock writers ([`Mac::lock_helpers`], [`Mac::lock_keys`];
//! `emit::Options::atomic_lock_helpers` and `atomic_lock_keys`): with
//! [`Parts::atomic_locks`] a 64-bit image atomic is a loop over the wave's
//! lanes in which each lane in its turn takes a lock word by a 32-bit
//! compare-exchange, spinning up to 65,536 tries, then loads the texel,
//! stores the result and frees the lock ([`Lower::locked_atomic64`]). With
//! these settings only the lanes that can change the texel enter that loop
//! ([`Lower::atomic64_of_writers`]); what they do in it is unchanged.
//!
//! Why not without a lock: the texel is one R32G32_UINT texel, and D3D12
//! has no atomic on one channel of a two-channel texel (its 32-bit image
//! atomics take a whole texel of a one-channel 32-bit format; the lock
//! loop's own 32-bit atomics are on the raw lock-word buffer). Whole-texel
//! loads and stores alone cannot keep a maximum: two lanes load, the larger
//! stores first, the smaller stores last. So the lock stays, and with it
//! the guarantee the title needs: when all writers of a pass have finished,
//! the texel holds the maximum of its start and of every value written,
//! both halves from the same value.
//!
//! * Helper lanes (`lock_helpers`, pixel programs). A pixel lane that the
//!   primitive does not cover, or that was discarded, runs on for its
//!   quad's derivatives and the host drops what it writes
//!   (`structure`, module notes: "helper lanes of a pixel shader count as
//!   live"). In the lock loop that is a lane whose lock exchange is
//!   dropped and returns whatever the host leaves there: where that is not
//!   "the lock was free", the lane spins its 65,536 tries while its whole
//!   wave waits, as helper lanes did in waterfall loops
//!   ([`Lower::helper_reads_own`]). Its load, store and unlock are dropped
//!   too, so leaving the loop out for it changes no memory. It is told by
//!   DXIL's `IsHelperLane`, which pixel programs already use; a lane the
//!   host does not report as one goes through the loop as before.
//! * Keys below the texel's (`lock_keys`, `umax` whose old value the
//!   program does not read: all but one of the title's 954). A lane first
//!   loads the texel without the lock and stays out when the texel's high
//!   dword is above its value's. Why that is exact:
//!   1. Every store to the texel is the lock loop's, made with the lock
//!      held, of `max(texel, value)` as a whole. So the texel never falls
//!      while unsigned maxima are its only writers, nor does its high
//!      dword, and it is always its start or one writer's whole value: no
//!      two writers' halves are ever stitched together.
//!   2. A load without the lock may fall between the two halves of a
//!      store. Its high dword alone is still a high dword the texel had or
//!      is getting: one 32-bit channel is read in one piece.
//!   3. A lane that stays out saw a high dword above its own. By 1 the
//!      texel's high dword ends at least there, so the texel ends above
//!      the lane's value: the value is not the maximum and is not missed.
//!   4. Every other lane runs the lock loop as before.
//!   The low dword is not compared: by 2 it may belong to an older value
//!   than the high dword read with it. Equal high dwords take the lock.
//!   (In the title the high dword looks like the fragment's depth and the
//!   low one like two packed 16-bit values, `notes/hair.md`.) Other writers than
//!   maxima on the same image in the same pass would void 1; the title's
//!   64-bit atomics are all `umax`, and no program with one stores to an
//!   image.
//!
//! The register's bit names are AMD's (as in Mesa's `sid.h`), written from
//! memory: the tree has no table of them.

use super::*;

/// What the Mac line's switches ask of a translation; all off (the
/// default): the translation is what it was without them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Mac {
    /// Pixel programs that ask for the depth test before the shader get
    /// the early depth-stencil flag (`emit::Options::early_depth`; the
    /// module notes).
    pub early_depth: bool,
    /// Under atomic locks, helper lanes of a pixel program stay out of the
    /// lock loop (`emit::Options::atomic_lock_helpers`; the module notes,
    /// "Lock writers").
    pub lock_helpers: bool,
    /// Under atomic locks, a lane whose `umax` value has a high dword below
    /// the texel's stays out of the lock loop
    /// (`emit::Options::atomic_lock_keys`).
    pub lock_keys: bool,
    /// LDS that a group of one console wave shares gets the finer order of
    /// [`super::lds_order`] where [`split_wave_barriers`] finds none
    /// (`emit::Options::lds_order`).
    pub lds_order: bool,
    /// In halves, a read across them in a block every lane runs once is
    /// handed over through group-shared memory, exact across the halves
    /// ([`super::exchange`], `emit::Options::halves_exchange`). Also: a
    /// lane mask held as a number whose two dwords different instructions
    /// write is tested by both ([`Lower::pair_high`]), not by the low one
    /// alone.
    pub exchange: bool,
    /// A size query on a writable image (`getDimensions`) asks for level 0
    /// instead of an undefined level (`emit::Options::uav_size_level0`).
    /// D3DMetal answers the undefined one with 0 x 0 and level 0 with the
    /// size of the view's level (measured, `uav_size_tests`). The console
    /// compiler's code adds 1 to the descriptor's "size - 1" fields, so a
    /// size of 0 reads as 65,536 x 65,536: the title's 35 programs that
    /// compute with such a size divide by it or test coordinates against
    /// it (`notes/uav-size.md`; first seen in its shadow min/max program,
    /// `07f451a14b13fecb`).
    pub uav_level0: bool,
    /// With [`Mac::exchange`]: a half's answer is also handed over in a
    /// block that a thread group runs together, under branches every
    /// thread of the group takes the same way ([`super::together`]). Set
    /// for the programs `emit::walk` opens to every lane
    /// (`emit::Options::halves_walk`).
    pub walk: bool,
    /// In halves with half-local reads ([`HalvesLocal`]), a read across
    /// the halves that [`Mac::exchange`] does not hand over is handed over
    /// the Mac port's way where every lane of the group runs the block
    /// together ([`Lower::across_first`], `emit::Options::halves_across`),
    /// and only elsewhere answered from the reader's own half. Without
    /// half-local reads that hand-over is made whatever this says.
    pub across: bool,
}

thread_local! {
    /// Size queries on writable images in the translations this thread
    /// made since [`take_uav_size_queries`]: from a descriptor's dwords,
    /// by `image_get_resinfo`.
    static UAV_SIZE_QUERIES: std::cell::Cell<[u32; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// Counts a size query on a writable image whose answer the program
/// uses, whatever [`Mac::uav_level0`] says (for `translate --list`, which
/// names the title's programs that have one; `notes/uav-size.md`).
pub(super) fn note_uav_size_query(resinfo: bool) {
    UAV_SIZE_QUERIES.with(|q| {
        let mut n = q.get();
        n[usize::from(resinfo)] += 1;
        q.set(n);
    });
}

/// Whether `p` reads one of `values` as a number: in an instruction that
/// does not take it as part of a descriptor (an image's or a buffer's
/// resource, a sampler, the base of a scalar buffer load), directly or
/// through phis. A descriptor's dwords are loaded together; only a program
/// that computes with the size dwords asks for the size.
pub(super) fn read_as_numbers(p: &Program, values: &[Value]) -> bool {
    let mut set: HashSet<Value> = values.iter().copied().collect();
    loop {
        let before = set.len();
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            if phi.args.iter().any(|a| set.contains(a)) {
                set.insert(phi.value);
            }
        }
        if set.len() == before {
            break;
        }
    }
    let within = |o: &Operand, reg: Reg| match (*o, reg) {
        (Operand::Scalar { code, dwords }, Reg::S(r)) => (code..code + dwords).contains(&r),
        _ => false,
    };
    p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        i.reads.iter().any(|&(reg, v)| {
            set.contains(&v)
                && !match &i.fields {
                    Fields::Image { rsrc, sampler, .. } => {
                        within(rsrc, reg) || sampler.as_ref().is_some_and(|s| within(s, reg))
                    }
                    Fields::Buffer { rsrc, .. } => within(rsrc, reg),
                    Fields::Smem { base, .. } => within(base, reg),
                    _ => false,
                }
        })
    })
}

/// The counts since the last call, and zeroes them: queries from a
/// descriptor's dwords, queries by `image_get_resinfo`.
pub(crate) fn take_uav_size_queries() -> [u32; 2] {
    UAV_SIZE_QUERIES.with(|q| q.replace([0; 2]))
}

/// `DB_SHADER_CONTROL`: the program exports depth (`Z_EXPORT_ENABLE`).
const Z_EXPORT: u32 = 1 << 0;
/// The program exports a stencil test value or a stencil operation value.
const STENCIL_EXPORTS: u32 = 1 << 1 | 1 << 2;
/// The program exports a coverage mask (`MASK_EXPORT_ENABLE`).
const MASK_EXPORT: u32 = 1 << 8;
/// `DEPTH_BEFORE_SHADER`: depth and stencil are tested and written before
/// the program runs.
const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
/// `SPI_PS_INPUT_ADDR`: the program reads the coverage mask
/// (`io::Input::SampleCoverage`).
const COVERAGE_INPUT: u32 = 1 << 14;

/// Whether a pixel program with these registers and translated outputs is
/// one whose early depth test D3D's flag states exactly (the module notes).
pub(crate) fn early_depth(db_shader_control: u32, input_addr: u32, outputs: &[SigElement]) -> bool {
    db_shader_control & DEPTH_BEFORE_SHADER != 0
        && db_shader_control & (Z_EXPORT | STENCIL_EXPORTS | MASK_EXPORT) == 0
        && input_addr & COVERAGE_INPUT == 0
        // SV_Coverage, SV_Depth and its two bounded forms, SV_StencilRef.
        && !outputs.iter().any(|o| matches!(o.kind, 14 | 17..=20))
}

impl Lower<'_> {
    /// Sets the early depth-stencil flag where [`Mac::early_depth`] is on
    /// and [`early_depth`] holds for the program.
    pub(super) fn early_depth_flag(&mut self) {
        if let StageIo::Ps {
            db_shader_control,
            input_addr,
            ..
        } = self.x.native.io
            && self.x.mac.early_depth
            && self.s.pixel
            && early_depth(db_shader_control, input_addr, &self.s.outputs)
            // A discard that stands in for the clipper must come before
            // the depth write (`Lower::soft_clip`).
            && !self.s.inputs.iter().any(|e| e.kind == 0 && e.name == super::mesh::CLIP_NAME)
        {
            self.s.flags |= flags::EARLY_DEPTH_STENCIL;
        }
    }
}

impl Lower<'_> {
    /// [`Lower::locked_atomic64`] for the lanes that can change the texel
    /// ([`Mac::lock_helpers`], [`Mac::lock_keys`]; the module notes, "Lock
    /// writers"); the other lanes go round it. `old_read`: the program
    /// reads the old value. Returns the old value's low and high dwords,
    /// in a lane that stays out the texel's as a plain load gives them
    /// (where `old_read` or the key test loaded them, else zeros no one
    /// reads).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn atomic64_of_writers(
        &mut self,
        h: V,
        cs: &[V],
        coords: usize,
        op: &str,
        value: V,
        compare: Option<V>,
        old_read: bool,
        not_rep: V,
    ) -> (V, V) {
        let helpers = self.x.mac.lock_helpers && self.s.pixel;
        let keys = self.x.mac.lock_keys && op == "umax" && !old_read;
        if !helpers && !keys {
            return self.locked_atomic64(h, cs, coords, op, value, compare, not_rep);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let zero = self.s.m.i32(0);
        // The texel as it is now, without the lock.
        let mut seen = (zero, zero);
        if keys || old_read {
            self.s.flags |= flags::TYPED_UAV_LOADS;
            let iu = self.s.m.undef(i32t);
            let mut args = vec![h, iu];
            args.extend(cs);
            args.extend([iu, iu, iu]);
            let r = self.s.op(
                c,
                "textureLoad.i32",
                Attr::ReadOnly,
                self.s.t.res_ret_i32,
                66,
                &args,
            );
            seen = (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1));
        }
        // The lanes that stay out.
        let mut out = None;
        if helpers {
            let i1 = self.s.t.i1;
            out = Some(
                self.s
                    .op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]),
            );
        }
        if keys {
            let k32 = {
                let i64t = self.s.m.int(64);
                self.s.m.int_const(i64t, 32)
            };
            let high = self.s.m.bin(c, Bin::LShr, value, k32);
            let high = self.s.m.cast(c, Cast::Trunc, high, i32t);
            let above = self.s.m.cmp(c, pred::IUGT, seen.1, high);
            out = Some(match out {
                Some(o) => self.s.m.bin(c, Bin::Or, o, above),
                None => above,
            });
        }
        let t = self.s.m.bool(true);
        let goes = self.s.m.bin(c, Bin::Xor, out.expect("a lane test"), t);
        let at = self.begin_guarded(Some(goes));
        let (lo, hi) = self.locked_atomic64(h, cs, coords, op, value, compare, not_rep);
        let r = self.end_guarded(at, &[(lo, seen.0), (hi, seen.1)]);
        (r[0], r[1])
    }
}
