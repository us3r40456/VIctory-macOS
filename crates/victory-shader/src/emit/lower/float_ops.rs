//! Floating-point lowering helpers for `Lower`, including optional console
//! float rule relaxations (`emit::Options::relax`).

use super::Lower;
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};

const TWO_PI: f32 = std::f32::consts::TAU;

impl<'a> Lower<'a> {
    /// A 32-bit denormal as the console's float mode reads it: a zero of its
    /// sign, decided on the bits (the host's own denormal mode varies).
    pub(super) fn ftz(&mut self, f: V) -> V {
        if self.x.mac.relax.denormals {
            return f;
        }
        let c = self.cur;
        let x = self.as_int(f);
        let (exp, sign, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x8000_0000),
            self.s.m.i32(0),
        );
        let e = self.s.m.bin(c, Bin::And, x, exp);
        let den = self.s.m.cmp(c, pred::EQ, e, zero);
        let s = self.s.m.bin(c, Bin::And, x, sign);
        let r = self.s.m.select(c, den, s, x);
        self.as_float(r)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but of the operands with denormals read as zeros and with
    /// -0 ordered below +0 (RDNA2 ISA, `v_max_f32`), where the compare and
    /// select DXIL specifies gives either zero.
    pub(super) fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let (a, b) = (self.ftz(a), self.ftz(b));
        self.max_min_flushed(max, a, b)
    }

    /// [`Self::max_min`] of operands without denormals.
    pub(super) fn max_min_flushed(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        if self.x.mac.relax.signed_zeros {
            return r;
        }
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    /// `f` as an f16 in the low bits, rounded as the program's MODE says at
    /// this point: toward zero where it set that for 16-bit results (the
    /// title does around its 16-bit interpolation), else to nearest even;
    /// f32 denormals read as zero; on the bits, where the host's own
    /// conversion rounds as it likes (toward zero on the RX 6700 XT and WARP).
    // shortcut: rounding toward +-infinity, or a mode not known at this
    // point (paths that set it differently join), rounds to nearest even.
    pub(super) fn narrow16(&mut self, f: V) -> V {
        if self.x.mac.relax.half_rounding {
            let i32t = self.s.t.i32;
            return self
                .s
                .op(self.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        }
        let rtz = self.mode16 == Some(3);
        let c = self.cur;
        let x = self.as_int(f);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k13, k16, k31) = (k(self, 13), k(self, 16), k(self, 31));
        let t = self.s.m.bin(c, Bin::LShr, x, k16);
        let m = k(self, 0x8000);
        let sign = self.s.m.bin(c, Bin::And, t, m);
        let m = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, m);
        // Normal halves: the exponent rebased, the mantissa rounded.
        let bias = k(self, 0x3800_0000);
        let rebased = self.s.m.bin(c, Bin::Sub, a, bias);
        let normal = if rtz {
            self.s.m.bin(c, Bin::LShr, rebased, k13)
        } else {
            let one = k(self, 1);
            let lsb = self.s.m.bin(c, Bin::LShr, rebased, k13);
            let lsb = self.s.m.bin(c, Bin::And, lsb, one);
            let half = k(self, 0xFFF);
            let up = self.s.m.bin(c, Bin::Add, rebased, half);
            let up = self.s.m.bin(c, Bin::Add, up, lsb);
            self.s.m.bin(c, Bin::LShr, up, k13)
        };
        let (inf, big) = (k(self, 0x7C00), k(self, if rtz { 0x7BFF } else { 0x7C00 }));
        let over = self.s.m.cmp(c, pred::IUGE, normal, inf);
        let normal = self.s.m.select(c, over, big, normal);
        let infinity = k(self, 0x7F80_0000);
        let is_inf = self.s.m.cmp(c, pred::EQ, a, infinity);
        let normal = self.s.m.select(c, is_inf, inf, normal);
        // Denormal halves: the mantissa shifted into units of 2^-24.
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, a, k23);
        let k126 = k(self, 126);
        let s = self.s.m.bin(c, Bin::Sub, k126, e);
        let far = self.s.m.cmp(c, pred::IUGT, s, k31);
        let s = self.s.m.select(c, far, k31, s);
        let (frac, hidden, zero) = (k(self, 0x7F_FFFF), k(self, 0x80_0000), k(self, 0));
        let mant = self.s.m.bin(c, Bin::And, a, frac);
        let mant = self.s.m.bin(c, Bin::Or, mant, hidden);
        let f32_denormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let mant = self.s.m.select(c, f32_denormal, zero, mant);
        let q = self.s.m.bin(c, Bin::LShr, mant, s);
        let den = if rtz {
            q
        } else {
            let one = k(self, 1);
            let s1 = self.s.m.bin(c, Bin::Sub, s, one);
            let half = self.s.m.bin(c, Bin::LShr, mant, s1);
            let half = self.s.m.bin(c, Bin::And, half, one);
            let below = self.s.m.bin(c, Bin::Shl, one, s1);
            let below = self.s.m.bin(c, Bin::Sub, below, one);
            let rest = self.s.m.bin(c, Bin::And, mant, below);
            let rest = self.s.m.cmp(c, pred::NE, rest, zero);
            let rest = self.s.m.cast(c, Cast::ZExt, rest, self.s.t.i32);
            let lsb = self.s.m.bin(c, Bin::And, q, one);
            let tie = self.s.m.bin(c, Bin::Or, rest, lsb);
            let up = self.s.m.bin(c, Bin::And, half, tie);
            self.s.m.bin(c, Bin::Add, q, up)
        };
        let min_normal = k(self, 0x3880_0000);
        let small = self.s.m.cmp(c, pred::IULT, a, min_normal);
        let r = self.s.m.select(c, small, den, normal);
        let is_nan = self.s.m.cmp(c, pred::IUGT, a, infinity);
        // A NaN keeps the top of its payload, quieted (as the RX 6700 XT,
        // RDNA2 like the console, converts).
        let payload = self.s.m.bin(c, Bin::LShr, a, k13);
        let (low10, quiet) = (k(self, 0x3FF), k(self, 0x7E00));
        let payload = self.s.m.bin(c, Bin::And, payload, low10);
        let nan = self.s.m.bin(c, Bin::Or, payload, quiet);
        let r = self.s.m.select(c, is_nan, nan, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// Fused f16 multiply-add with one nearest-even rounding. The product
    /// of two finite f16s is exact in f32. TwoSum recovers the error of the
    /// f32 addition so a rounded sum at an f16 midpoint can be resolved.
    pub(super) fn fma_f16(&mut self, a: V, b: V, addend: V, clamp: bool) -> V {
        let c = self.cur;
        let m = &mut self.s.m;
        let product = m.bin(c, Bin::Mul, a, b);
        let sum = m.bin(c, Bin::Add, product, addend);
        let addend_hi = m.bin(c, Bin::Sub, sum, product);
        let product_hi = m.bin(c, Bin::Sub, sum, addend_hi);
        let product_lo = m.bin(c, Bin::Sub, product, product_hi);
        let addend_lo = m.bin(c, Bin::Sub, addend, addend_hi);
        let error = m.bin(c, Bin::Add, product_lo, addend_lo);
        // Saturation only changes the sum to an endpoint, neither a midpoint.
        let sum = if clamp { self.unary(7, sum) } else { sum };
        let bits = self.as_int(sum);
        let error_bits = self.as_int(error);
        let m = &mut self.s.m;
        let sign_mask = m.i32(0x8000_0000);
        let sign = m.bin(c, Bin::And, bits, sign_mask);
        let magnitude = m.bin(c, Bin::Xor, bits, sign);
        let error_bits = m.bin(c, Bin::Xor, error_bits, sign);
        let error = self.as_float(error_bits);

        // Pack the f16 ourselves: the legacy conversion may round toward zero.
        let m = &mut self.s.m;
        let one = m.i32(1);
        let zero = m.i32(0);
        let k23 = m.i32(23);
        let exponent = m.bin(c, Bin::LShr, magnitude, k23);
        let min_normal = m.i32(113);
        let normal = m.cmp(c, pred::IUGE, exponent, min_normal);
        let min_midpoint = m.i32(102); // f32 exponent for 2^-25
        let tiny = m.cmp(c, pred::IULT, exponent, min_midpoint);
        let bounded = m.select(c, tiny, min_midpoint, exponent);
        let subnormal_base = m.i32(126);
        let subnormal_shift = m.bin(c, Bin::Sub, subnormal_base, bounded);
        let k13 = m.i32(13);
        let shift = m.select(c, normal, k13, subnormal_shift);
        let bias = m.i32(112 << 23);
        let normal_bits = m.bin(c, Bin::Sub, magnitude, bias);
        let fraction_mask = m.i32(0x007F_FFFF);
        let fraction = m.bin(c, Bin::And, magnitude, fraction_mask);
        let hidden_bit = m.i32(0x0080_0000);
        let subnormal_bits = m.bin(c, Bin::Or, fraction, hidden_bit);
        let significand = m.select(c, normal, normal_bits, subnormal_bits);
        let lower = m.bin(c, Bin::LShr, significand, shift);
        let unit = m.bin(c, Bin::Shl, one, shift);
        let mask = m.bin(c, Bin::Sub, unit, one);
        let remainder = m.bin(c, Bin::And, significand, mask);
        let midpoint = m.bin(c, Bin::LShr, unit, one);
        let above = m.cmp(c, pred::IUGT, remainder, midpoint);
        let at_midpoint = m.cmp(c, pred::EQ, remainder, midpoint);
        let odd = m.bin(c, Bin::And, lower, one);
        let odd = m.cmp(c, pred::NE, odd, zero);
        let fzero = m.f32(0);
        let away = m.cmp(c, pred::OGT, error, fzero);
        let exact = m.cmp(c, pred::OEQ, error, fzero);
        let even_tie = m.bin(c, Bin::And, exact, odd);
        let round_up = m.bin(c, Bin::Or, away, even_tie);
        let round_up = m.bin(c, Bin::And, at_midpoint, round_up);
        let round_up = m.bin(c, Bin::Or, above, round_up);
        let round_up = m.cast(c, Cast::ZExt, round_up, self.s.t.i32);
        let rounded = m.bin(c, Bin::Add, lower, round_up);
        let rounded = m.select(c, tiny, zero, rounded);

        let limit = m.i32(0x4780_0000); // 65536; smaller overflows round up above
        let overflow = m.cmp(c, pred::IUGE, magnitude, limit);
        let infinity = m.i32(0x7C00);
        let rounded = m.select(c, overflow, infinity, rounded);
        let k16 = m.i32(16);
        let sign = m.bin(c, Bin::LShr, sign, k16);
        let rounded = m.bin(c, Bin::Or, sign, rounded);
        let f32_infinity = m.i32(0x7F80_0000);
        let nan = m.cmp(c, pred::IUGT, magnitude, f32_infinity);
        let fallback = self.narrow16(sum);
        self.s.m.select(c, nan, fallback, rounded)
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    pub(super) fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let h = self.f32_to_f16(c, f);
        let back = self.f16_to_f32(c, h);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    pub(super) fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let (a, b, third) = (self.ftz(a), self.ftz(b), self.ftz(third));
        let lo = self.max_min_flushed(false, a, b);
        let hi = self.max_min_flushed(true, a, b);
        let t = self.max_min_flushed(false, hi, third);
        let median = self.max_min_flushed(true, lo, t);
        if self.x.mac.relax.nan {
            return median;
        }
        let least = self.max_min_flushed(false, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    pub(super) fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    pub(super) fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        let r = self.s.m.select(c, exact, zero, r);
        // The hardware returns NaN for NaN and for infinity, whose reduced
        // part is NaN. Metal's sin and cos give 0 there, so that NaN is
        // passed through here instead of relying on the host.
        let nan = self.s.m.cmp(c, pred::UNO, part, part);
        let r = self.s.m.select(c, nan, part, r);
        if cosine {
            return r;
        }
        // sin(±0) is ±0. Some Metal GPUs (M2) drop the sign through the
        // host's sin or a float select, so a zero input is passed through
        // as integer bits (from nickorlabs's #30).
        let bits = self.as_int(x);
        let mask = self.s.m.i32(0x7fff_ffff);
        let magnitude = self.s.m.bin(c, Bin::And, bits, mask);
        let zero = self.s.m.i32(0);
        let is_zero = self.s.m.cmp(c, pred::EQ, magnitude, zero);
        let result = self.as_int(r);
        let result = self.s.m.select(c, is_zero, bits, result);
        self.as_float(result)
    }

    /// `0 × anything = 0`, as in D3D9. As on RDNA2, a denormal operand counts
    /// as zero, so the test is on the exponent bits rather than a float
    /// compare (whose result would depend on the host's denormal mode).
    pub(super) fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        if self.x.mac.relax.legacy_mul {
            return m;
        }
        if self.x.mac.relax.legacy_mul_nan {
            // Ordered and not zero: a NaN or zero product is +0.
            let zero = self.s.m.f32(0);
            let number = self.s.m.cmp(c, pred::ONE, m, zero);
            return self.s.m.select(c, number, m, zero);
        }
        let exp = self.s.m.i32(0x7F80_0000);
        let izero = self.s.m.i32(0);
        let is_zero = |l: &mut Self, v: V| {
            let bits = l.as_int(v);
            let e = l.s.m.bin(c, Bin::And, bits, exp);
            l.s.m.cmp(c, pred::EQ, e, izero)
        };
        let az = is_zero(self, a);
        let bz = is_zero(self, b);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        let zero = self.s.m.f32(0);
        self.s.m.select(c, z, zero, m)
    }
}
