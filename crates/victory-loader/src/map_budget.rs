//! macOS port: the title's time budget for placing map pieces, made larger.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) lets its streaming engine work
//! on map pieces for a set time each frame. The governor at +0x2294750 sets
//! that time (the dword at the engine's +0x166c8, microseconds) every frame
//! while bit 1 of its mode word (+0xab44324) is set: its estimate of the
//! work (96 a piece and 64 an entity placed) divided by a time from the
//! title's clock (the float at +0x78 of the object at +0xab44328, most
//! likely the last frame's time) in units of 33,333 microseconds, kept
//! between 250 and 1250. The engine's stage runner (+0x9d94a0) puts off the
//! map pieces whose requests may wait once the time is spent.
//!
//! On a console that is 7.5 to 37 ms of such work a second. At two frames a
//! second the division brings the budget to the lower limit, 250
//! microseconds a frame: half a millisecond a second, under Rosetta. This
//! may be why map pieces finish loading long after their place was left and
//! why 700 to 2,800 requests wait at every stop of the opening sequence
//! (notes/pool-crash.md); that is a guess the watch's `streaming` line is
//! there to check.
//!
//! `VICTORY_MAPDATA_BUDGET=<k>` (2 to 200) multiplies the governor's two
//! limits by `k`: four immediates in its code. With 40 the budget is 10 to
//! 50 ms a frame. What it changes in the title: more of a frame's time may
//! go into placing map pieces before the rest is put off, so frames in
//! which much is placed get longer by up to the larger budget. Nothing else
//! is touched. Not tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch.
pub const SWITCH: &str = "VICTORY_MAPDATA_BUDGET";

/// The governor's limits: `mov ecx, 1250; cmp eax, 1250; mov edi, 250;
/// cmovb ecx, eax; cmp ecx, 251; cmovae edi, ecx; mov [rsi + 0x50], edi`.
const SITE: u32 = 0x0229_4814;
const SITE_BYTES: [u8; 30] = [
    0xb9, 0xe2, 0x04, 0x00, 0x00, 0x3d, 0xe2, 0x04, 0x00, 0x00, 0xbf, 0xfa, 0x00, 0x00, 0x00, 0x0f, 0x42, 0xc8,
    0x81, 0xf9, 0xfb, 0x00, 0x00, 0x00, 0x0f, 0x43, 0xf9, 0x89, 0x7e, 0x50,
];
/// Where the four immediates are in the site: the upper limit twice, the
/// lower limit, and the lower limit plus one.
const UPPER_AT: [usize; 2] = [1, 6];
const LOWER_AT: usize = 11;
const ABOVE_LOWER_AT: usize = 20;
const UPPER: u32 = 1250;
const LOWER: u32 = 250;

/// The factor the switch's value asks for; `None` leaves the title as it is
/// (unset, 0, 1 or not a number from 2 to 200).
pub fn wanted(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok().filter(|k| (2..=200).contains(k))
}

/// The site's bytes with both limits multiplied by `k`.
fn scaled(k: u32) -> [u8; 30] {
    let mut bytes = SITE_BYTES;
    let mut put = |at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    for at in UPPER_AT {
        put(at, UPPER * k);
    }
    put(LOWER_AT, LOWER * k);
    put(ABOVE_LOWER_AT, LOWER * k + 1);
    bytes
}

/// Multiplies the governor's limits of the image mapped at `base` by `k`.
/// Before any title thread runs. Says why if it does not.
#[cfg(windows)]
pub fn patch(base: u64, k: u32) -> Result<(), &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let site = (base + u64::from(SITE)) as *mut u8;
    // SAFETY: the image is mapped and the site is inside its code.
    if unsafe { std::slice::from_raw_parts(site, SITE_BYTES.len()) } != SITE_BYTES {
        return Err("not the build it was written for (bytes differ)");
    }
    let bytes = scaled(k);
    let mut old = 0;
    // SAFETY: the site is the title's code, made writable for the write,
    // before any title thread runs; only four immediates change.
    unsafe {
        if VirtualProtect(site as *const _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("the title's code could not be made writable");
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), site, bytes.len());
        VirtualProtect(site as *const _, bytes.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_two_to_two_hundred() {
        assert_eq!(wanted(Some("40")), Some(40));
        assert_eq!(wanted(Some(" 2 ")), Some(2));
        assert_eq!(wanted(Some("200")), Some(200));
        for off in [None, Some(""), Some("0"), Some("1"), Some("201"), Some("-4"), Some("on")] {
            assert_eq!(wanted(off), None, "{off:?}");
        }
    }

    /// The site holds the limits where the offsets say.
    #[test]
    fn the_site_has_the_limits() {
        let dword = |at: usize| u32::from_le_bytes(SITE_BYTES[at..at + 4].try_into().unwrap());
        assert_eq!((SITE_BYTES[0], dword(UPPER_AT[0])), (0xb9, UPPER), "mov ecx, 1250");
        assert_eq!((SITE_BYTES[5], dword(UPPER_AT[1])), (0x3d, UPPER), "cmp eax, 1250");
        assert_eq!((SITE_BYTES[10], dword(LOWER_AT)), (0xbf, LOWER), "mov edi, 250");
        assert_eq!((&SITE_BYTES[18..20], dword(ABOVE_LOWER_AT)), (&[0x81, 0xf9][..], LOWER + 1), "cmp ecx, 251");
    }

    #[test]
    fn only_the_four_immediates_change() {
        let bytes = scaled(40);
        let dword = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(dword(UPPER_AT[0]), 50_000);
        assert_eq!(dword(UPPER_AT[1]), 50_000);
        assert_eq!(dword(LOWER_AT), 10_000);
        assert_eq!(dword(ABOVE_LOWER_AT), 10_001);
        let changed: Vec<usize> = (0..bytes.len()).filter(|&i| bytes[i] != SITE_BYTES[i]).collect();
        assert!(changed.iter().all(|&i| {
            [UPPER_AT[0], UPPER_AT[1], LOWER_AT, ABOVE_LOWER_AT].iter().any(|&at| (at..at + 4).contains(&i))
        }));
        assert_eq!(scaled(1), SITE_BYTES);
        // The largest factor still fits the immediates.
        assert_eq!(u32::from_le_bytes(scaled(200)[1..5].try_into().unwrap()), 250_000);
    }
}
