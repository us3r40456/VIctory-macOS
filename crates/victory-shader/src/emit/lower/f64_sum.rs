//! Whole-wave sums of 64-bit floats ([`Shape::ActiveF64`], the model taken
//! from avatarmanz b17b69fd) without DXIL doubles: the Mac line's host,
//! D3DMetal, has none, so the fork's own lowering (`makeDouble`,
//! `waveActiveOp.f64`, `splitDouble`) cannot be used here.
//!
//! Only the sum the title has is lowered: of the constant 1.0 in the lanes
//! of a mask and 0.0 in the others, which is how the console compiler
//! counts lanes as a double (the water transmission passes divide a wave
//! sum by it). That sum is the number of lanes in the mask, exactly: every
//! partial sum is a whole number of at most 64, so no addition rounds and
//! their order does not matter. The count is taken as the 32-bit sums of
//! [`Lower::shape`] are (`waveActiveOp.i32` over the host's active lanes,
//! with the same refusals), and written as a 64-bit float with integer
//! operations ([`count_f64_high`]). Any other summand is refused: adding
//! doubles in general needs the arithmetic the host lacks.
//!
//! No setting turns this off: without it the programs are refused, and
//! with it the sum is exact wherever a 32-bit wave sum is (in halves with
//! half-local reads it counts the host wave's own half, as those do).
//!
//! [`Shape::ActiveF64`]: super::super::wave::Shape::ActiveF64

use super::*;

/// The high dword of the 64-bit float 1.0; its low dword is 0.
const ONE_HIGH: u32 = 0x3FF0_0000;

/// What the bits of a 32-bit float's exponent and the top of its fraction,
/// moved down 3 places into a 64-bit float's high dword, lack of that
/// float's: the difference of the two exponent biases (1023 - 127), at the
/// 64-bit exponent's place.
const BIAS_STEP: u32 = 896 << 20;

/// The high dword of `n` as a 64-bit float, for `n` below 2^21 (its low
/// dword is then 0): `n` goes to a 32-bit float (exact below 2^24), whose
/// exponent and 20 highest fraction bits are the high dword's but for the
/// bias; the 3 fraction bits that would be the low dword's top are 0 for
/// such `n`. Zero has no exponent to move and stays 0.
pub(crate) fn count_f64_high(m: &mut Module, c: usize, n: V) -> V {
    let f32t = m.float();
    let i32t = m.int(32);
    let f = m.cast(c, Cast::UiToFp, n, f32t);
    let bits = m.cast(c, Cast::Bitcast, f, i32t);
    let (k3, step, zero) = (m.i32(3), m.i32(BIAS_STEP), m.i32(0));
    let high = m.bin(c, Bin::LShr, bits, k3);
    let high = m.bin(c, Bin::Add, high, step);
    let none = m.cmp(c, pred::EQ, n, zero);
    m.select(c, none, zero, high)
}

impl Lower<'_> {
    /// The low or `high` dword of the sum over the wave of the 64-bit float
    /// (`lo`, `hi`) in the lanes of `mask` and 0.0 in the others (the
    /// module notes).
    pub(super) fn active_f64_sum(
        &mut self,
        lo: Value,
        hi: Value,
        mask: Value,
        high: bool,
    ) -> Result<V, Why> {
        // As [`Lower::shape`] has them for a 32-bit sum.
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        self.same_half("a wave reduction over both 32-lane halves (wave64 on a 32-lane host)")?;
        let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
        if self.s.m.int_value(lo) != Some(0) || self.s.m.int_value(hi) != Some(ONE_HIGH.into()) {
            return Err("a whole-wave sum of 64-bit floats other than of the constant 1.0");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        if !high {
            // A count of at most 64 has no bit in the low dword.
            return Ok(self.s.m.i32(0));
        }
        let on = self.boolean(m, c)?;
        let (one, zero) = (self.s.m.i32(1), self.s.m.i32(0));
        let x = self.s.m.select(c, on, one, zero);
        self.s.flags |= flags::WAVE_OPS;
        let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(1));
        let i32t = self.s.t.i32;
        let n = self.s.op(
            c,
            "waveActiveOp.i32",
            Attr::NoUnwind,
            i32t,
            119,
            &[x, sum, unsigned],
        );
        Ok(count_f64_high(&mut self.s.m, c, n))
    }
}
