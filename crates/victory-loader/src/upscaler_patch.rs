//! macOS port: on a Series S the title's upscaler choice returns its own
//! upscaler (type 6), not FSR 4.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) chooses its upscaler by the
//! console it runs on: FSR 4 (its type 7) on a Series S that reports the
//! FidelityFX capability, and otherwise type 6, its own upscaler, which is
//! what a Series X runs. Under VIctory FSR 4's passes put nothing on the
//! screen unless the host's FidelityFX serves them
//! (`VICTORY_FIDELITYFX_DIR`: a PC build of it that has an upscaler for the
//! host GPU), so with `--console series-s` the world is black
//! (VIctory-macOS #172).
//!
//! fxceless/victory-fork#798 found the two functions that make the choice,
//! the chooser and the copy of it the per-frame size query uses, and makes
//! both `mov eax, 6; ret`. This does the same to the mapped image, before
//! any title thread runs, and only if all of this holds:
//!
//! * the console profile is Series S (on a Series X the title returns 6
//!   itself: nothing here runs or prints, whatever the switch says);
//! * `VICTORY_UPSCALER_PATCH` is not `0`;
//! * the host's FidelityFX does not serve the title's, unless
//!   `VICTORY_UPSCALER_PATCH=1` asks for the patch all the same;
//! * the executable is the build the offsets are for, by its BLAKE3;
//! * both functions start with the bytes expected. If either differs,
//!   neither is written.
//!
//! One start-up line says what was done. Only offsets and short byte
//! patterns of the build are stored here (docs/asset-policy.md).

/// The switch. `0`: the title's choice stays. `1`: patched even when the
/// host's FidelityFX serves. Unset or anything else: the default, patched
/// on a Series S unless the host's FidelityFX serves.
pub const SWITCH: &str = "VICTORY_UPSCALER_PATCH";

/// How the BLAKE3 (hex) of the executable the offsets are for starts.
const BUILD: &str = "b99f876b74d4a77f";

/// What is checked and what is written at each place: six bytes.
#[cfg(any(windows, test))]
type Bytes = [u8; 6];

/// The two functions that choose the upscaler, by RVA, and how each starts:
/// `sub rsp, 0x28` and the first two bytes of its next instruction.
#[cfg(any(windows, test))]
const SITES: [(u32, Bytes); 2] = [
    // The chooser.
    (0x007e_93f0, [0x48, 0x83, 0xec, 0x28, 0x80, 0x3d]),
    // The copy of the choice the per-frame size query uses.
    (0x0540_0190, [0x48, 0x83, 0xec, 0x28, 0x48, 0x8b]),
];

/// `mov eax, 6; ret`: type 6, the title's own upscaler.
#[cfg(any(windows, test))]
const RETURN_6: Bytes = [0xb8, 0x06, 0x00, 0x00, 0x00, 0xc3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Switch {
    Off,
    On,
    Default,
}

fn switch(value: Option<&str>) -> Switch {
    match value.map(str::trim) {
        Some("0") => Switch::Off,
        Some("1") => Switch::On,
        _ => Switch::Default,
    }
}

/// Whether to patch, from everything but the image's bytes. `None`: not a
/// Series S, so nothing is done or said. `Some(Err(why))`: the title's
/// choice stays. `switch_value` is the value of [`SWITCH`], `build` the
/// BLAKE3 (hex) of the original executable, and `host_fidelityfx` whether
/// the host's FidelityFX serves the title's.
pub fn wanted(
    series_s: bool,
    switch_value: Option<&str>,
    build: &str,
    host_fidelityfx: bool,
) -> Option<Result<(), &'static str>> {
    if !series_s {
        return None;
    }
    let switch = switch(switch_value);
    let known = build
        .get(..BUILD.len())
        .is_some_and(|start| start.eq_ignore_ascii_case(BUILD));
    Some(if switch == Switch::Off {
        Err("VICTORY_UPSCALER_PATCH=0")
    } else if host_fidelityfx && switch != Switch::On {
        Err(
            "the host's FidelityFX serves the title's (VICTORY_UPSCALER_PATCH=1 patches all the same)",
        )
    } else if !known {
        Err("not the build it was written for (another executable)")
    } else {
        Ok(())
    })
}

/// Whether an image of `size` bytes, whose six bytes at an RVA `read`
/// returns, has the expected bytes at both places. `read` is asked only for
/// places inside the image, and only if both are inside.
#[cfg(any(windows, test))]
fn check(size: u32, read: impl Fn(u32) -> Bytes) -> Result<(), String> {
    let len = RETURN_6.len() as u64;
    if SITES
        .iter()
        .any(|&(rva, _)| u64::from(rva) + len > u64::from(size))
    {
        return Err("not the build it was written for (the image is too small)".into());
    }
    let differ: Vec<String> = SITES
        .iter()
        .filter(|&&(rva, expect)| read(rva) != expect)
        .map(|&(rva, _)| format!("+{rva:#x}"))
        .collect();
    if differ.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "not the build it was written for (bytes differ at {})",
            differ.join(" and ")
        ))
    }
}

/// The start-up line's text, after the `upscaler` label.
#[cfg(any(windows, test))]
fn line(done: &Result<(), String>) -> String {
    match done {
        Ok(()) => {
            let places: Vec<String> = SITES.iter().map(|&(rva, _)| format!("+{rva:#x}")).collect();
            format!(
                "patched at {}: on Series S the title's choice returns type 6, its own upscaler, not FSR 4 ({SWITCH}=0 leaves it alone)",
                places.join(" and ")
            )
        }
        Err(why) => format!("not patched, the title's choice stays (FSR 4 on Series S): {why}"),
    }
}

/// Writes [`RETURN_6`] at both places of the image of `size` bytes mapped at
/// `base` if both hold the expected bytes: both or neither.
#[cfg(windows)]
fn patch(base: u64, size: u32) -> Result<(), String> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let at = |rva: u32| (base + u64::from(rva)) as *mut Bytes;
    // SAFETY: `check` asks only for places inside the mapped image.
    check(size, |rva| unsafe { at(rva).read_unaligned() })?;
    let len = RETURN_6.len();
    let mut old = [0u32; SITES.len()];
    // Both writable before either is written.
    for (i, &(rva, _)) in SITES.iter().enumerate() {
        // SAFETY: six bytes of the title's code, inside the image.
        if unsafe { VirtualProtect(at(rva).cast(), len, PAGE_EXECUTE_READWRITE, &mut old[i]) } == 0
        {
            for (j, &(rva, _)) in SITES[..i].iter().enumerate().rev() {
                // SAFETY: as above; the protection it had comes back.
                unsafe { VirtualProtect(at(rva).cast(), len, old[j], &mut old[j]) };
            }
            return Err("the title's code could not be made writable".into());
        }
    }
    for &(rva, _) in &SITES {
        // SAFETY: writable since the loop above, before any title thread runs.
        unsafe { at(rva).write_unaligned(RETURN_6) };
    }
    for (i, &(rva, _)) in SITES.iter().enumerate().rev() {
        // SAFETY: as above; the protection it had comes back.
        unsafe { VirtualProtect(at(rva).cast(), len, old[i], &mut old[i]) };
    }
    // SAFETY: the whole process's cache, as the other patches of the image do.
    unsafe { FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0) };
    Ok(())
}

/// Patches the upscaler choice of the image of `size` bytes mapped at `base`
/// if [`wanted`] says so and its bytes are the expected ones. Before any
/// title thread runs. Returns what to print at start-up; `None` when the
/// console is not a Series S.
#[cfg(windows)]
pub fn at_start(
    series_s: bool,
    switch_value: Option<&str>,
    build: &str,
    host_fidelityfx: bool,
    base: u64,
    size: u32,
) -> Option<String> {
    let done = wanted(series_s, switch_value, build, host_fidelityfx)?
        .map_err(str::to_owned)
        .and_then(|()| patch(base, size));
    Some(line(&done))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The BLAKE3 the code knows the build by (victory-game), in full.
    const KNOWN: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
    const BIG: u32 = 0x0700_0000;

    /// An image that has `first` and `second` at the two places.
    fn image(first: Bytes, second: Bytes) -> impl Fn(u32) -> Bytes {
        move |rva| match rva {
            0x7e_93f0 => first,
            0x540_0190 => second,
            other => panic!("read at +{other:#x}, which is neither place"),
        }
    }

    #[test]
    fn what_is_written_is_mov_eax_6_ret() {
        assert_eq!(RETURN_6[0], 0xb8, "mov eax, imm32");
        assert_eq!(u32::from_le_bytes(RETURN_6[1..5].try_into().unwrap()), 6);
        assert_eq!(RETURN_6[5], 0xc3, "ret");
    }

    /// The fork's numbers (fxceless/victory-fork#798), and what makes the
    /// six bytes safe to replace: each place is a function's first
    /// instruction, so nothing runs past the `ret`.
    #[test]
    fn both_places_start_a_function() {
        assert_eq!(SITES[0], (0x7e_93f0, [0x48, 0x83, 0xec, 0x28, 0x80, 0x3d]));
        assert_eq!(SITES[1], (0x540_0190, [0x48, 0x83, 0xec, 0x28, 0x48, 0x8b]));
        for (rva, expect) in SITES {
            assert_eq!(
                expect[..4],
                [0x48, 0x83, 0xec, 0x28],
                "sub rsp, 0x28 at +{rva:#x}"
            );
            assert_ne!(
                expect, RETURN_6,
                "a patched place is not taken for an unpatched one"
            );
        }
    }

    #[test]
    fn the_expected_bytes_at_both_places_pass() {
        assert_eq!(check(BIG, image(SITES[0].1, SITES[1].1)), Ok(()));
        // The second place's last byte is the image's last.
        assert_eq!(check(0x540_0196, image(SITES[0].1, SITES[1].1)), Ok(()));
    }

    #[test]
    fn other_bytes_at_either_place_fail_and_name_it() {
        let mut other = SITES[0].1;
        other[5] ^= 1;
        let first = check(BIG, image(other, SITES[1].1)).unwrap_err();
        assert!(
            first.contains("+0x7e93f0") && !first.contains("+0x5400190"),
            "{first}"
        );
        let second = check(BIG, image(SITES[0].1, SITES[0].1)).unwrap_err();
        assert!(
            second.contains("+0x5400190") && !second.contains("+0x7e93f0"),
            "{second}"
        );
        let both = check(BIG, image(RETURN_6, RETURN_6)).unwrap_err();
        assert!(both.contains("+0x7e93f0 and +0x5400190"), "{both}");
        // The places swapped are not the build either.
        assert!(check(BIG, image(SITES[1].1, SITES[0].1)).is_err());
    }

    #[test]
    fn a_smaller_image_is_not_read() {
        let never = |rva: u32| -> Bytes { panic!("read at +{rva:#x}") };
        for size in [0, 0x7e_93f0, 0x7e_93f6, 0x540_0190, 0x540_0195] {
            let why = check(size, never).unwrap_err();
            assert!(why.contains("too small"), "{size:#x}: {why}");
        }
    }

    /// The write itself, on memory of our own in place of the image.
    #[cfg(windows)]
    #[test]
    fn both_places_are_written_or_neither() {
        let mut image = vec![0u8; 0x540_1000];
        let (base, size) = (image.as_mut_ptr() as u64, image.len() as u32);
        let put = |image: &mut [u8], bytes: [Bytes; 2]| {
            for ((rva, _), bytes) in SITES.iter().zip(bytes) {
                image[*rva as usize..][..6].copy_from_slice(&bytes);
            }
        };
        put(&mut image, [SITES[0].1, SITES[0].1]);
        assert!(patch(base, size).unwrap_err().contains("+0x5400190"));
        assert_eq!(
            image[0x7e_93f0..][..6],
            SITES[0].1,
            "the first place is not written either"
        );
        put(&mut image, [SITES[0].1, SITES[1].1]);
        assert_eq!(patch(base, size), Ok(()));
        for (rva, _) in SITES {
            let at = rva as usize;
            assert_eq!(image[at..][..6], RETURN_6);
            assert_eq!(
                (image[at - 1], image[at + 6]),
                (0, 0),
                "six bytes and no more"
            );
        }
        assert!(
            patch(base, size).is_err(),
            "a second time the bytes are not the expected ones"
        );
        assert!(patch(base, 0x540_0195).is_err());
    }

    #[test]
    fn the_switch_is_off_only_at_0() {
        assert_eq!(switch(None), Switch::Default);
        assert_eq!(switch(Some("0")), Switch::Off);
        assert_eq!(switch(Some(" 0 ")), Switch::Off);
        assert_eq!(switch(Some("1")), Switch::On);
        for other in ["", "2", "00", "off", "false", "no", "on"] {
            assert_eq!(switch(Some(other)), Switch::Default, "{other:?}");
        }
    }

    #[test]
    fn a_series_x_gets_nothing_whatever_the_switch_says() {
        for value in [None, Some("0"), Some("1"), Some("x")] {
            for host in [false, true] {
                assert_eq!(wanted(false, value, KNOWN, host), None);
                assert_eq!(wanted(false, value, "00", host), None);
            }
        }
    }

    #[test]
    fn a_series_s_of_the_known_build_is_patched_unless_switched_off() {
        assert_eq!(wanted(true, None, KNOWN, false), Some(Ok(())));
        assert_eq!(wanted(true, Some("1"), KNOWN, false), Some(Ok(())));
        assert_eq!(wanted(true, Some("yes"), KNOWN, false), Some(Ok(())));
        assert_eq!(
            wanted(true, None, &KNOWN.to_uppercase(), false),
            Some(Ok(()))
        );
        let off = wanted(true, Some("0"), KNOWN, false).unwrap().unwrap_err();
        assert_eq!(off, "VICTORY_UPSCALER_PATCH=0");
        // Off also when the host's FidelityFX serves, and for another build.
        assert_eq!(wanted(true, Some("0"), KNOWN, true), Some(Err(off)));
        assert_eq!(wanted(true, Some("0"), "00", false), Some(Err(off)));
    }

    #[test]
    fn the_hosts_fidelityfx_keeps_fsr_4_unless_asked() {
        let why = wanted(true, None, KNOWN, true).unwrap().unwrap_err();
        assert!(
            why.contains("FidelityFX") && why.contains("VICTORY_UPSCALER_PATCH=1"),
            "{why}"
        );
        assert_eq!(wanted(true, Some("1"), KNOWN, true), Some(Ok(())));
    }

    #[test]
    fn another_build_is_left_alone() {
        let other = "c99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
        for build in [other, "", "b99f876b", "unknown"] {
            for value in [None, Some("1")] {
                let why = wanted(true, value, build, false).unwrap().unwrap_err();
                assert!(why.contains("not the build"), "{build:?}: {why}");
            }
        }
        assert!(KNOWN.starts_with(BUILD));
    }

    #[test]
    fn the_line_says_where_or_why_not() {
        let on = line(&Ok(()));
        assert!(
            on.starts_with("patched at +0x7e93f0 and +0x5400190: "),
            "{on}"
        );
        assert!(
            on.contains("type 6") && on.contains("VICTORY_UPSCALER_PATCH=0"),
            "{on}"
        );
        let off = line(&Err("VICTORY_UPSCALER_PATCH=0".into()));
        assert!(
            off.starts_with("not patched, ") && off.ends_with(": VICTORY_UPSCALER_PATCH=0"),
            "{off}"
        );
    }
}
