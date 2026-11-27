//! macOS port: the longest step the title's clamped timers take in a frame,
//! made longer.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) updates its timers once a frame
//! (the caller around +0x2226600 measures the real frame time and calls the
//! timer update +0x2226040 nine times: timer in rcx, real frame time in
//! xmm1, time scale in xmm2, "clamp" in r9b). A clamped timer's step is
//! `min(real, 1/15 s)`, raised to a least step, times the scale, and when
//! the scale is not 1 clamped once more to the same two ends. Two of the
//! nine timers are not clamped (r9b = 0); for one timer (+0xa958358) the
//! caller holds an inlined copy of the same clamp (+0x2226833) that it uses
//! in place of the call in one state.
//!
//! The 1/15 is the float at +0x6159ec0, loaded once at each of the two
//! places. The least step is a variable, not a constant: with the byte
//! +0x7453cc0 set it is the float at +0x6849844 (1/30 in the file, written
//! by the title as 1 over a rate) and after that, or without the byte, the
//! float at +0x6847660 (set to 1/3000 when the timers are set up). Neither
//! matters at our frame times.
//!
//! At 0.4 to 0.9 s a frame the clamped timers therefore run at 1/6 to 1/13
//! of the clock, and what the title drives by them is that much slower: the
//! exposure after a cut stays white for tens of frames
//! (notes/white-flash.md).
//!
//! `VICTORY_TIME_STEP_MAX=<seconds>` (over 1/15, up to 2) makes the two
//! loads read a larger constant. The 1/15 itself is read by more than sixty
//! other instructions and is left alone: each load's displacement is pointed
//! at a float the title's read-only data already holds, the one nearest to
//! the value asked for. What it changes in the title: every clamped timer
//! follows the clock up to that step, so physics, AI, animation and scripts
//! take steps the title was never made for. Nothing else is touched. Not
//! tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch.
pub const SWITCH: &str = "VICTORY_TIME_STEP_MAX";

/// The length of a load and where its displacement is; the displacement
/// counts from the load's end.
const LEN: usize = 8;
const DISP_AT: usize = 4;
/// The two loads of the cap: `vmovss xmm3, [rip + 0x3f33e73]` in the timer
/// update and `vmovss xmm0, [rip + 0x3f33685]` in the caller's inlined copy.
/// Both read the float at [`STOCK`].
const SITES: [(u32, [u8; LEN]); 2] = [
    (0x0222_6045, [0xc5, 0xfa, 0x10, 0x1d, 0x73, 0x3e, 0xf3, 0x03]),
    (0x0222_6833, [0xc5, 0xfa, 0x10, 0x05, 0x85, 0x36, 0xf3, 0x03]),
];
/// The constant the title loads: 1/15 s as the file holds it.
const STOCK: (u32, [u8; 4]) = (0x0615_9ec0, [0x89, 0x88, 0x88, 0x3d]);
/// Longer steps the title's read-only data already holds as floats that its
/// code loads, and the values they must hold.
const LONGER: [(u32, f32); 9] = [
    (0x05fc_7020, 0.1),
    (0x0615_a6b8, 0.125),
    (0x0615_9f8c, 0.2),
    (0x0615_9ca4, 0.25),
    (0x0615_a240, 1.0 / 3.0),
    (0x05fc_6edc, 0.5),
    (0x0615_9cc4, 0.75),
    (0x05fc_6ed8, 1.0),
    (0x0615_9bd4, 2.0),
];

/// The step in seconds the switch's value asks for; `None` leaves the title
/// as it is (unset, not a number, not over the title's 1/15 s, or over 2).
pub fn wanted(value: Option<&str>) -> Option<f32> {
    let stock = f32::from_le_bytes(STOCK.1);
    value?.trim().parse::<f32>().ok().filter(|s| s.is_finite() && *s > stock && *s <= 2.0)
}

/// The constant nearest to `seconds`; of two equally near, the shorter.
fn nearest(seconds: f32) -> (u32, f32) {
    LONGER.into_iter().min_by(|a, b| (a.1 - seconds).abs().total_cmp(&(b.1 - seconds).abs())).unwrap()
}

/// The step in seconds that asking for `seconds` gives.
pub fn step_for(seconds: f32) -> f32 {
    nearest(seconds).1
}

/// Both loads' bytes pointed at `target`, given the bytes now at the two
/// loads, at the stock constant and at the target. Says why if any of them
/// is not what this was written for; then nothing may be written.
fn retargeted(
    sites: [&[u8; LEN]; 2],
    stock: [u8; 4],
    target: (u32, f32),
    at_target: [u8; 4],
) -> Result<[[u8; LEN]; 2], &'static str> {
    if sites[0] != &SITES[0].1 {
        return Err("not the build it was written for (the timer update's load differs)");
    }
    if sites[1] != &SITES[1].1 {
        return Err("not the build it was written for (the inlined copy's load differs)");
    }
    if stock != STOCK.1 {
        return Err("not the build it was written for (the loads' constant is not 1/15)");
    }
    if at_target != target.1.to_le_bytes() {
        return Err("not the build it was written for (the longer constant is not where it was read)");
    }
    let mut out = [SITES[0].1, SITES[1].1];
    for (bytes, (site, _)) in out.iter_mut().zip(SITES) {
        let disp = target.0.wrapping_sub(site + LEN as u32);
        bytes[DISP_AT..].copy_from_slice(&disp.to_le_bytes());
    }
    Ok(out)
}

/// Points both loads of the image mapped at `base` at the constant for
/// `seconds`. Before any title thread runs. Gives the new longest step in
/// seconds, or says why nothing was changed.
#[cfg(windows)]
pub fn patch(base: u64, seconds: f32) -> Result<f32, &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let target = nearest(seconds);
    let site = |i: usize| (base + u64::from(SITES[i].0)) as *mut u8;
    let float_at = |rva: u32| (base + u64::from(rva)) as *const [u8; 4];
    // SAFETY: the image is mapped; the sites are inside its code and the two
    // constants are inside its read-only data.
    let bytes = unsafe {
        retargeted(
            [&*(site(0) as *const [u8; LEN]), &*(site(1) as *const [u8; LEN])],
            *float_at(STOCK.0),
            target,
            *float_at(target.0),
        )?
    };
    // Both sites are made writable before either is written, so a refusal
    // leaves the title as it was.
    let mut old = [0u32; 2];
    for i in 0..2 {
        // SAFETY: the site is the title's code; only its protection changes.
        if unsafe { VirtualProtect(site(i) as *const _, LEN, PAGE_EXECUTE_READWRITE, &mut old[i]) } == 0 {
            if i == 1 {
                // SAFETY: as above; the first site gets its protection back.
                unsafe { VirtualProtect(site(0) as *const _, LEN, old[0], &mut old[0]) };
            }
            return Err("the title's code could not be made writable");
        }
    }
    // SAFETY: both sites are writable code of the title, before any title
    // thread runs; only the displacements change. The second site's first
    // protection call may have seen the first's page already writable, so
    // the protections are put back in reverse order.
    unsafe {
        for i in 0..2 {
            std::ptr::copy_nonoverlapping(bytes[i].as_ptr(), site(i), LEN);
        }
        for i in [1, 0] {
            VirtualProtect(site(i) as *const _, LEN, old[i], &mut old[i]);
        }
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    Ok(target.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: [&[u8; LEN]; 2] = [&SITES[0].1, &SITES[1].1];

    #[test]
    fn the_switch_takes_seconds_over_the_stock_step_up_to_two() {
        assert_eq!(wanted(Some("1")), Some(1.0));
        assert_eq!(wanted(Some(" 0.5 ")), Some(0.5));
        assert_eq!(wanted(Some("2")), Some(2.0));
        assert_eq!(wanted(Some("0.07")), Some(0.07));
        for off in [None, Some(""), Some("0"), Some("0.0666"), Some("0.06666667"), Some("2.01"), Some("-1"), Some("on")]
        {
            assert_eq!(wanted(off), None, "{off:?}");
        }
        for off in ["nan", "inf", "-inf"] {
            assert_eq!(wanted(Some(off)), None, "{off}");
        }
    }

    /// Each site is a `vmovss` from `[rip + disp32]` whose displacement
    /// leads to the stock constant, which is 1/15.
    #[test]
    fn the_loads_read_the_stock_constant() {
        assert_eq!(SITES[0].1[..DISP_AT], [0xc5, 0xfa, 0x10, 0x1d], "vmovss xmm3, [rip + disp32]");
        assert_eq!(SITES[1].1[..DISP_AT], [0xc5, 0xfa, 0x10, 0x05], "vmovss xmm0, [rip + disp32]");
        for (site, bytes) in SITES {
            let disp = u32::from_le_bytes(bytes[DISP_AT..].try_into().unwrap());
            assert_eq!(site + LEN as u32 + disp, STOCK.0);
        }
        assert_eq!(f32::from_le_bytes(STOCK.1), 1.0 / 15.0);
    }

    #[test]
    fn the_nearest_longer_constant_is_taken() {
        for (asked, got) in [
            (0.07, 0.1),
            (0.1, 0.1),
            (0.12, 0.125),
            (0.2, 0.2),
            (0.3, 1.0 / 3.0),
            (0.4, 1.0 / 3.0),
            (0.45, 0.5),
            (0.5, 0.5),
            (0.7, 0.75),
            (0.9, 1.0),
            (1.0, 1.0),
            (1.4, 1.0),
            (1.6, 2.0),
            (2.0, 2.0),
        ] {
            assert_eq!(step_for(asked), got, "{asked}");
        }
        // Every constant is longer than the title's own step and in rising
        // order, so no value of the switch can shorten the step.
        assert!(LONGER[0].1 > f32::from_le_bytes(STOCK.1));
        assert!(LONGER.windows(2).all(|w| w[0].1 < w[1].1));
    }

    #[test]
    fn only_the_displacements_change() {
        for target in LONGER {
            let out = retargeted(GOOD, STOCK.1, target, target.1.to_le_bytes()).unwrap();
            for (bytes, (site, stock)) in out.iter().zip(SITES) {
                assert_eq!(bytes[..DISP_AT], stock[..DISP_AT]);
                let disp = u32::from_le_bytes(bytes[DISP_AT..].try_into().unwrap());
                assert_eq!(site + LEN as u32 + disp, target.0);
            }
        }
        // One second: the constant the timer update itself compares the
        // time scale with (+0x2226062).
        let one = retargeted(GOOD, STOCK.1, LONGER[7], 1f32.to_le_bytes()).unwrap();
        assert_eq!(one[0], [0xc5, 0xfa, 0x10, 0x1d, 0x8b, 0x0e, 0xda, 0x03]);
        assert_eq!(one[1], [0xc5, 0xfa, 0x10, 0x05, 0x9d, 0x06, 0xda, 0x03]);
    }

    #[test]
    fn anything_unexpected_is_refused() {
        let target = LONGER[7];
        let good = target.1.to_le_bytes();
        let mut other = SITES[0].1;
        other[3] = 0x35; // another register
        assert!(retargeted([&other, GOOD[1]], STOCK.1, target, good).unwrap_err().contains("timer update's load"));
        let mut moved = SITES[1].1;
        moved[4] ^= 4; // another constant
        assert!(retargeted([GOOD[0], &moved], STOCK.1, target, good).unwrap_err().contains("inlined copy's load"));
        // Already patched: refused, not patched twice.
        let done = retargeted(GOOD, STOCK.1, target, good).unwrap();
        assert!(retargeted([&done[0], &done[1]], STOCK.1, target, good).is_err());
        assert!(retargeted(GOOD, (1f32 / 30.0).to_le_bytes(), target, good).unwrap_err().contains("not 1/15"));
        assert!(retargeted(GOOD, STOCK.1, target, 1.5f32.to_le_bytes()).unwrap_err().contains("longer constant"));
        assert!(retargeted(GOOD, STOCK.1, target, [0; 4]).is_err());
    }
}
