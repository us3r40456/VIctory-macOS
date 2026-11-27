//! The title's output size and render scale (`VICTORY_OUTPUT`,
//! `VICTORY_RENDER_SCALE`, `VICTORY_UPSCALER`, `VICTORY_UI_SIZE`; each off
//! without it): places in the title's code and data changed at load, each
//! compared byte for byte first and left alone, said so, if it differs
//! (another build of the title).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets):
//!
//! - **Output size.** Its settings (`settings.xml` in its storage,
//!   `video/resolutionFullscreen`, 3840x2160 as a Series X wrote it; read
//!   into the live settings at +0x204/+0x208, the defaults' 1920x1080 at
//!   +0x5b37a20 overwritten) are applied by +0x22434a0: from +0x224352b it
//!   hands them to +0x2478130, the one writer of the screen size
//!   (+0xa9438cc/+0xa9438d0), and when the height changes calls +0xd8c3a0.
//!   The back buffers (+0x247ad10, by +0x501f4a0 and +0xcc2780) take that
//!   size through the copy +0x5eb24f0 makes (+0x75e6fe0). The display mode
//!   list it matches against is its own (+0x5a38910: 1280x720, 1664x936,
//!   1920x1080, 2560x1440, 3840x2160 at 60 Hz, table +0x615c0c0), not the
//!   DXGI output's, which is why `VICTORY_DISPLAY_MODE` changes nothing.
//!   Here that block becomes: write the size asked for into the settings
//!   (+0x204/+0x208), then the title's own else-path (screen size set,
//!   +0xd8c3a0 if the height changed). Everything that reads the settings
//!   after it (the device parameters +0x1e0fb30, the render size +0x7b4e10)
//!   sees the same size. The back buffers' pool (64 MB on Series X) is
//!   checked to be used up but for 2 MB (+0x247af25, an `int3`); for another
//!   size that check is jumped over.
//! - **Render scale.** The upscaler the title picks (+0x7e93f0: 6, its own,
//!   except on Series S with a flag, 7, AMD FidelityFX) gives the render
//!   size from the output size (+0x7e92f0 → the upscaler's +0x50 method):
//!   its own (+0x7e9440) multiplies by a scale at +0x90 of it, set to 2/3 at
//!   +0x28f1e78 (so 2560x1440 for 4K, 1280x720 for 1080p);
//!   `VICTORY_RENDER_SCALE` replaces that constant (Series X; run res-4 at
//!   1.0). FidelityFX's (+0x28ef680, Series S) multiplies by 1.0 on Series X,
//!   2/3 on Series S. The fork also changes that one (`VICTORY_FFX_SCALE`),
//!   but it is left out here: at 0.5 on Series S FidelityFX's jitter phase
//!   count came back 0 and the title divided by it (+0x28f03e8).
//! - **Upscaler.** `VICTORY_UPSCALER=<kind>` makes the picker return
//!   `kind` (pull request #53): 6 the title's own, 7 FidelityFX.
//! - **Interface size.** With `VICTORY_OUTPUT` alone the interface (button
//!   prompts, subtitles, HUD) is laid out for 3840x2160 and drawn into the
//!   smaller output from its top left corner: at 1920x1080 every box is
//!   twice as large and twice as far from the corner (runs 2026-11-27
//!   03:23:57 against 03:28:13). The screen size the output patch sets is
//!   not where the 3840x2160 comes from: by DexNom (`native_1440p.rs`, which
//!   writes 2560x1440 there) the interface surface follows the largest
//!   entry of the title's own display mode list (+0x615c100, 3840x2160 at
//!   60 Hz, read only by the list's builder +0x5a38910), the console's
//!   display. A second size pair next to the screen size (+0xa9438ec, the
//!   interface surfaces of +0x5eb2590 and the settings menu's preview at
//!   +0x58048d4 read it) has no direct writer; DexNom finds it stays at
//!   3840x2160. `VICTORY_UI_SIZE=<w>x<h>` writes that mode entry. Not yet
//!   traced: which reader of the list sizes the interface (a game run
//!   decides).
// Ported from avatarmanz/VIctory-7900XTX (crates/victory-loader/src/resolution.rs,
// commit e151e71 and its follow-up for the back buffer pool check), GPL-3.0-or-later.

/// What became of a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Patched {
    Done,
    /// Its bytes are not the ones expected: left alone.
    Differs,
    Failed(&'static str),
}

/// Sizes the title has a display mode for (its own list, +0x615c0c0).
pub const MODES: [(u32, u32); 5] = [
    (1280, 720),
    (1664, 936),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];

/// `VICTORY_OUTPUT=<w>x<h>`: one of [`MODES`]. `None` if unset or not one
/// (said why).
pub fn asked_output() -> Option<(u32, u32)> {
    let spec = std::env::var("VICTORY_OUTPUT").ok()?;
    let size = parse_output(&spec);
    if size.is_none() {
        eprintln!(
            "VIctory: VICTORY_OUTPUT={spec}: expected one of the title's modes: 1920x1080, 2560x1440, 1280x720, 1664x936, 3840x2160"
        );
    }
    size
}

/// `VICTORY_UI_SIZE=<w>x<h>`: one of [`MODES`], the size the title's
/// largest display mode is given ([`DISPLAY_MODE`]). `None` if unset or not
/// one (said why).
pub fn asked_ui_size() -> Option<(u32, u32)> {
    let spec = std::env::var("VICTORY_UI_SIZE").ok()?;
    let size = parse_output(&spec);
    if size.is_none() {
        eprintln!(
            "VIctory: VICTORY_UI_SIZE={spec}: expected one of the title's modes: 1920x1080, 2560x1440, 1280x720, 1664x936, 3840x2160"
        );
    }
    size
}

pub fn parse_output(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.trim().split_once(['x', 'X'])?;
    let size = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

/// `VICTORY_RENDER_SCALE=<s>`, 0.5 to 1.0 (0.5: half the output's width and
/// height): the title's own upscaler's scale (Series X; 2/3 without it).
/// Tested at 1.0 (run res-4); below 2/3 the title's largest render size
/// (+0x94, max(2/3, scale)) stays at 2/3. `None` if unset or out of range
/// (said why).
pub fn asked_render_scale() -> Option<f32> {
    let spec = std::env::var("VICTORY_RENDER_SCALE").ok()?;
    let scale = parse_render_scale(&spec);
    if scale.is_none() {
        eprintln!("VIctory: VICTORY_RENDER_SCALE={spec}: expected 0.5 to 1.0");
    }
    scale
}

pub fn parse_render_scale(spec: &str) -> Option<f32> {
    spec.trim()
        .parse::<f32>()
        .ok()
        .filter(|s| (0.5..=1.0).contains(s))
}

/// `VICTORY_UPSCALER=<kind>`: 6 (the title's own) or 7 (FidelityFX).
pub fn asked_upscaler() -> Option<u32> {
    let spec = std::env::var("VICTORY_UPSCALER").ok()?;
    let kind = spec
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|k| matches!(k, 6 | 7));
    if kind.is_none() {
        eprintln!(
            "VIctory: VICTORY_UPSCALER={spec}: expected 6 (the title's own) or 7 (FidelityFX)"
        );
    }
    kind
}

/// A place to change: its offset in the image and the bytes there
/// (compared first; the replacement is at most as long).
pub struct Site {
    pub at: usize,
    pub what: &'static str,
    pub bytes: &'static [u8],
}

/// The screen size setter, its screen height, and what +0x22434a0 calls when
/// the height changed.
pub const SET_SCREEN_SIZE: usize = 0x2478130;
pub const SCREEN_HEIGHT: usize = 0xa9438d0;
pub const HEIGHT_CHANGED: usize = 0xd8c3a0;

/// +0x22434a0 from +0x224352b to +0x2243568: the settings' size handed to
/// the screen size.
pub const OUTPUT: Site = Site {
    at: 0x224352b,
    what: "settings' output size applied",
    bytes: &[
        0x8b, 0x0d, 0x9b, 0x03, 0x70, 0x08, // mov ecx, [+0xa9438cc]
        0x8b, 0x86, 0x04, 0x02, 0x00, 0x00, // mov eax, [rsi+204h]
        0x39, 0xc1, 0x75, 0x0d, // cmp ecx, eax ; jne
        0x8b, 0x96, 0x08, 0x02, 0x00, 0x00, // mov edx, [rsi+208h]
        0xe8, 0xea, 0x4b, 0x23, 0x00, 0xeb, 0x20, // call +0x2478130 ; jmp +0x2243568
        0x8b, 0xbe, 0x08, 0x02, 0x00, 0x00, // mov edi, [rsi+208h]
        0x8b, 0x2d, 0x7c, 0x03, 0x70, 0x08, // mov ebp, [+0xa9438d0]
        0x89, 0xc1, 0x89, 0xfa, // mov ecx, eax ; mov edx, edi
        0xe8, 0xd3, 0x4b, 0x23, 0x00, // call +0x2478130
        0x39, 0xfd, 0x74, 0x07, 0xb1, 0x01, // cmp ebp, edi ; je ; mov cl, 1
        0xe8, 0x38, 0x8e, 0xb4, 0xfe, // call +0xd8c3a0
    ],
};

/// +0x247ad10, after the back buffers are made in their pool (64 MB on
/// Series X, 82 MB on Series S, from +0x247ad45): `int3` unless less than
/// 2 MB of the pool is left over. Sized for 3840x2160; at 1920x1080 49 MB
/// are left (run res-1 stopped here). With another output size the check
/// becomes a jump over the `int3`; the pool stays as large.
pub const POOL_CHECK: Site = Site {
    at: 0x247af25,
    what: "back buffer pool's left-over check",
    bytes: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc], // cmp rax, 1FFFFFh ; jbe +1 ; int3
};

/// What goes at [`POOL_CHECK`]: `jmp +1` in place of `jbe +1`.
pub fn pool_check_code() -> Vec<u8> {
    let mut c = POOL_CHECK.bytes[..8].to_vec();
    c[6] = 0xeb;
    c
}

/// The title's own upscaler, its method +0x28f1e70: `mov dword [rcx+90h],
/// 2/3` (the scale +0x7e9440 renders at).
pub const OWN_SCALE: Site = Site {
    at: 0x28f1e78,
    what: "the title's own upscaler's render scale",
    bytes: &[0xc7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x2a, 0x3f],
};

/// The upscaler picker, +0x7e93f0, to its first `ret`.
pub const PICKER: Site = Site {
    at: 0x7e93f0,
    what: "upscaler picker",
    bytes: &[
        0x48, 0x83, 0xec, 0x28, 0x80, 0x3d, 0x6d, 0xf0, 0x3a, 0x07, 0x00, 0x75, 0x09, 0x83, 0x3d,
        0x68, 0xf0, 0x3a, 0x07, 0x00, 0x74, 0x28, 0xe8, 0x35, 0xbd, 0x88, 0xff, 0x8a, 0x0d, 0x5b,
        0xf0, 0x3a, 0x07, 0x80, 0xe1, 0x40, 0x80, 0xf9, 0x01, 0xb9, 0x07, 0x00, 0x00, 0x00, 0x83,
        0xd9, 0x00, 0x84, 0xc0, 0xb8, 0x06, 0x00, 0x00, 0x00, 0x0f, 0x45, 0xc1, 0x48, 0x83, 0xc4,
        0x28, 0xc3,
    ],
};

/// The last entry of the title's display mode list (+0x615c0c0, five of
/// 16 bytes: width, height, refresh rate as a fraction), 3840x2160 at 60/1.
/// Data, not code; only the list's builder +0x5a38910 reads it.
pub const DISPLAY_MODE: Site = Site {
    at: 0x615c100,
    what: "the title's largest display mode (its interface size)",
    bytes: &[
        0x00, 0x0f, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00, // 3840, 2160
        0x3c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // 60 / 1
    ],
};

/// What goes at [`DISPLAY_MODE`] for `width` x `height` (the refresh rate
/// kept).
pub fn display_mode_code(width: u32, height: u32) -> Vec<u8> {
    let mut c = Vec::with_capacity(DISPLAY_MODE.bytes.len());
    c.extend(width.to_le_bytes());
    c.extend(height.to_le_bytes());
    c.extend(&DISPLAY_MODE.bytes[8..]);
    c
}

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at [`OUTPUT`] for `width` x `height` (the same length).
pub fn output_code(width: u32, height: u32) -> Vec<u8> {
    let at = OUTPUT.at;
    let mut c = Vec::with_capacity(OUTPUT.bytes.len());
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [screen height]
    c.extend(rel(SCREEN_HEIGHT, at + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the setter
    c.extend(rel(SET_SCREEN_SIZE, at + c.len() + 4));
    let end = OUTPUT.bytes.len();
    c.extend([0x39, 0xfd, 0x74, (end - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call
    c.extend(rel(HEIGHT_CHANGED, at + c.len() + 4));
    c.extend([0xeb, (end - (c.len() + 2)) as u8]); // jmp end
    c.resize(end, 0xcc);
    c
}

/// What goes at [`OWN_SCALE`] for `scale`.
pub fn own_scale_code(scale: f32) -> Vec<u8> {
    let mut c = OWN_SCALE.bytes[..6].to_vec();
    c.extend(scale.to_le_bytes());
    c
}

/// `mov eax, kind ; ret`.
pub fn picker_code(kind: u32) -> Vec<u8> {
    let mut c = vec![0xb8];
    c.extend(kind.to_le_bytes());
    c.push(0xc3);
    c
}

/// Compares `site`'s bytes in the image of `size` bytes at `base`.
///
/// # Safety
/// The image is mapped.
pub unsafe fn matches(base: usize, size: usize, site: &Site) -> Patched {
    if site
        .at
        .checked_add(site.bytes.len())
        .is_none_or(|end| end > size)
    {
        return Patched::Failed("outside the image");
    }
    // SAFETY: inside the mapped image (checked above).
    let there =
        unsafe { std::slice::from_raw_parts((base + site.at) as *const u8, site.bytes.len()) };
    if there == site.bytes {
        Patched::Done
    } else {
        Patched::Differs
    }
}

/// Writes `code` over `site` if its bytes are the expected ones.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn write(base: usize, size: usize, site: &Site, code: &[u8]) -> Patched {
    // SAFETY: the caller's promise.
    unsafe { write_together(base, size, &[(site, code)]) }
}

/// Writes each code over its site, all of them or none: every site's bytes
/// are compared and every range made writable before any is copied, so a
/// mismatch or a refusal at one site leaves the others as they were too.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn write_together(base: usize, size: usize, parts: &[(&Site, &[u8])]) -> Patched {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    for (site, code) in parts {
        if code.len() > site.bytes.len() {
            return Patched::Failed("replacement longer than the bytes compared");
        }
        // SAFETY: the caller's promise.
        let found = unsafe { matches(base, size, site) };
        if found != Patched::Done {
            return found;
        }
    }
    // (address, length, protection before)
    let mut opened: Vec<(usize, usize, u32)> = Vec::with_capacity(parts.len());
    // SAFETY: inside the image (checked); made writable for this and put
    // back, the last opened first (two sites may share a page); nothing
    // runs it yet.
    unsafe {
        for (site, code) in parts {
            let at = base + site.at;
            let mut old = 0;
            if VirtualProtect(at as *const _, code.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                for &(at, len, old) in opened.iter().rev() {
                    let mut was = 0;
                    VirtualProtect(at as *const _, len, old, &mut was);
                }
                return Patched::Failed("Windows would not make it writable");
            }
            opened.push((at, code.len(), old));
        }
        for (site, code) in parts {
            std::ptr::copy_nonoverlapping(code.as_ptr(), (base + site.at) as *mut u8, code.len());
        }
        for &(at, len, old) in opened.iter().rev() {
            let mut was = 0;
            VirtualProtect(at as *const _, len, old, &mut was);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, len);
        }
    }
    Patched::Done
}

/// Sites written together, each with its replacement.
#[cfg(windows)]
type Group = Vec<(&'static Site, Vec<u8>)>;

/// Applies what the switches ask for; a line each in the log and on
/// standard output.
///
/// # Safety
/// The image of `size` bytes at `base` stays mapped and no thread runs its
/// code yet.
#[cfg(windows)]
pub unsafe fn apply_asked(base: usize, size: usize) {
    // Sites that only make sense together go in one group, written all or
    // none: a smaller output with the pool check left in stops the title,
    // and the check gone without a smaller output is pointless.
    let mut asked: Vec<(Group, String)> = Vec::new();
    if let Some((w, h)) = asked_output() {
        let mut group = vec![(&OUTPUT, output_code(w, h))];
        if (w, h) != (3840, 2160) {
            group.push((&POOL_CHECK, pool_check_code()));
        }
        asked.push((group, format!("{w}x{h}")));
    }
    if let Some((w, h)) = asked_ui_size() {
        asked.push((
            vec![(&DISPLAY_MODE, display_mode_code(w, h))],
            format!("{w}x{h}"),
        ));
    }
    if let Some(s) = asked_render_scale() {
        asked.push((vec![(&OWN_SCALE, own_scale_code(s))], format!("{s}")));
    }
    if let Some(kind) = asked_upscaler() {
        asked.push((
            vec![(&PICKER, picker_code(kind))],
            format!("returns {kind}"),
        ));
    }
    for (group, value) in asked {
        let parts: Vec<(&Site, &[u8])> = group.iter().map(|(s, c)| (*s, c.as_slice())).collect();
        // SAFETY: the caller's promise.
        let done = unsafe { write_together(base, size, &parts) };
        for (site, _) in &parts {
            let offset = format!("{:#x}", site.at);
            match done {
                Patched::Done => {
                    tracing::info!(site = %offset, what = site.what, value, "resolution: patched")
                }
                Patched::Differs => {
                    tracing::warn!(site = %offset, what = site.what, "resolution: not patched, the title's bytes at one of the sites written together are not the ones expected (another build?)")
                }
                Patched::Failed(why) => {
                    tracing::warn!(site = %offset, what = site.what, why, "resolution: not patched")
                }
            }
            println!(
                "resolution: {} ({value}) at title+{offset}: {done:?}",
                site.what
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse() {
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_output("1600x900"), None);
        assert_eq!(parse_output("1080p"), None);
        assert_eq!(parse_render_scale("0.67"), Some(0.67));
        assert_eq!(parse_render_scale("0.5"), Some(0.5));
        assert_eq!(parse_render_scale("1"), Some(1.0));
        assert_eq!(parse_render_scale("0.4"), None);
        assert_eq!(parse_render_scale("1.5"), None);
    }

    /// The calls and loads in the bytes compared are the ones the
    /// replacement keeps.
    #[test]
    fn the_output_block_is_the_titles() {
        let b = OUTPUT.bytes;
        assert_eq!(b.len(), 61);
        let target = |op_end: usize| {
            (OUTPUT.at + op_end) as i64
                + i64::from(i32::from_le_bytes(
                    b[op_end - 4..op_end].try_into().unwrap(),
                ))
        };
        assert_eq!(target(27), SET_SCREEN_SIZE as i64);
        assert_eq!(target(50), SET_SCREEN_SIZE as i64);
        assert_eq!(target(61), HEIGHT_CHANGED as i64);
        assert_eq!(target(41), SCREEN_HEIGHT as i64);
        assert_eq!(target(6), (SCREEN_HEIGHT - 4) as i64, "the width before it");
        // jne to the else path at 29; jmp +0x20 and je +7 both land at the end.
        assert_eq!((b[14], 16 + b[15] as usize), (0x75, 29));
        assert_eq!((b[27], 29 + b[28] as usize), (0xeb, 61));
        assert_eq!((b[52], 54 + b[53] as usize), (0x74, 61));
    }

    #[test]
    fn the_output_code_keeps_the_titles_else_path() {
        let c = output_code(1920, 1080);
        assert_eq!(c.len(), OUTPUT.bytes.len());
        assert_eq!(c[0], 0xb8);
        assert_eq!(c[1..5], 1920u32.to_le_bytes());
        assert_eq!(c[5..11], [0x89, 0x86, 0x04, 0x02, 0x00, 0x00]);
        assert_eq!(c[12..16], 1080u32.to_le_bytes());
        assert_eq!(c[16..22], [0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]);
        let target = |op_end: usize| {
            (OUTPUT.at + op_end) as i64
                + i64::from(i32::from_le_bytes(
                    c[op_end - 4..op_end].try_into().unwrap(),
                ))
        };
        assert_eq!(c[22..24], [0x8b, 0x2d]);
        assert_eq!(target(28), SCREEN_HEIGHT as i64);
        assert_eq!(c[32], 0xe8);
        assert_eq!(target(37), SET_SCREEN_SIZE as i64);
        assert_eq!(c[37..40], [0x39, 0xfd, 0x74]);
        assert_eq!(41 + c[40] as usize, 61, "je lands at the end");
        assert_eq!(c[43], 0xe8);
        assert_eq!(target(48), HEIGHT_CHANGED as i64);
        assert_eq!(c[48], 0xeb);
        assert_eq!(50 + c[49] as usize, 61, "jmp lands at the end");
        assert!(c[50..].iter().all(|&x| x == 0xcc));
    }

    /// The display mode entry is the last of the title's five modes, the
    /// 3840x2160 one; the replacement keeps its refresh rate.
    #[test]
    fn the_display_mode_code() {
        let b = DISPLAY_MODE.bytes;
        assert_eq!(b.len(), 16);
        let word = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        assert_eq!((word(0), word(4)), MODES[4]);
        assert_eq!((word(8), word(12)), (60, 1));
        let c = display_mode_code(1920, 1080);
        assert_eq!(c.len(), b.len());
        assert_eq!(c[..4], 1920u32.to_le_bytes());
        assert_eq!(c[4..8], 1080u32.to_le_bytes());
        assert_eq!(c[8..], b[8..]);
        assert_eq!(display_mode_code(3840, 2160), b);
    }

    #[test]
    fn the_scale_codes() {
        let own = own_scale_code(0.5);
        assert_eq!(own[..6], OWN_SCALE.bytes[..6]);
        assert_eq!(own[6..], 0.5f32.to_le_bytes());
        assert_eq!(OWN_SCALE.bytes[6..], (2.0f32 / 3.0).to_le_bytes());
        assert_eq!(picker_code(7), [0xb8, 7, 0, 0, 0, 0xc3]);
        assert_eq!(
            pool_check_code(),
            [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0xeb, 0x01]
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_site_is_written_only_over_its_bytes() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, VirtualAlloc, VirtualFree,
        };
        // SAFETY: fresh memory of our own, read-only to code as the image's is.
        unsafe {
            let place =
                VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, 0x40) as *mut u8;
            assert!(!place.is_null());
            let site = Site {
                at: 0x100,
                what: "test",
                bytes: OWN_SCALE.bytes,
            };
            std::ptr::copy_nonoverlapping(site.bytes.as_ptr(), place.add(0x100), site.bytes.len());
            let mut old = 0;
            windows_sys::Win32::System::Memory::VirtualProtect(
                place.cast(),
                0x1000,
                PAGE_EXECUTE_READ,
                &mut old,
            );
            assert_eq!(
                write(place as usize, 0x1000, &site, &own_scale_code(0.5)),
                Patched::Done
            );
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x106), 4),
                0.5f32.to_le_bytes()
            );
            // Already changed: its bytes differ now, nothing is written again.
            assert_eq!(
                write(place as usize, 0x1000, &site, &own_scale_code(0.75)),
                Patched::Differs
            );
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x106), 4),
                0.5f32.to_le_bytes()
            );
            let far = Site {
                at: 0xffa,
                what: "test",
                bytes: OWN_SCALE.bytes,
            };
            assert!(matches!(
                write(place as usize, 0x1000, &far, &own_scale_code(0.5)),
                Patched::Failed(_)
            ));
            // Written together: the second site's bytes differ, so the
            // first one is left as it was too.
            let first = Site {
                at: 0x200,
                what: "test",
                bytes: OWN_SCALE.bytes,
            };
            let second = Site {
                at: 0x300,
                what: "test",
                bytes: POOL_CHECK.bytes,
            };
            let mut old = 0;
            windows_sys::Win32::System::Memory::VirtualProtect(
                place.cast(),
                0x1000,
                0x40,
                &mut old,
            );
            std::ptr::copy_nonoverlapping(
                first.bytes.as_ptr(),
                place.add(0x200),
                first.bytes.len(),
            );
            std::ptr::write_bytes(place.add(0x300), 0x90, second.bytes.len());
            windows_sys::Win32::System::Memory::VirtualProtect(
                place.cast(),
                0x1000,
                PAGE_EXECUTE_READ,
                &mut old,
            );
            let scale = own_scale_code(0.5);
            let check = pool_check_code();
            assert_eq!(
                write_together(
                    place as usize,
                    0x1000,
                    &[(&first, &scale), (&second, &check)]
                ),
                Patched::Differs
            );
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x200), first.bytes.len()),
                first.bytes
            );
            // With both as expected, both are written.
            windows_sys::Win32::System::Memory::VirtualProtect(
                place.cast(),
                0x1000,
                0x40,
                &mut old,
            );
            std::ptr::copy_nonoverlapping(
                second.bytes.as_ptr(),
                place.add(0x300),
                second.bytes.len(),
            );
            windows_sys::Win32::System::Memory::VirtualProtect(
                place.cast(),
                0x1000,
                PAGE_EXECUTE_READ,
                &mut old,
            );
            assert_eq!(
                write_together(
                    place as usize,
                    0x1000,
                    &[(&first, &scale), (&second, &check)]
                ),
                Patched::Done
            );
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x200), scale.len()),
                scale.as_slice()
            );
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x300), check.len()),
                check.as_slice()
            );
            VirtualFree(place.cast(), 0, MEM_RELEASE);
        }
    }

    /// Windows refuses to make the second site writable (a read-only view
    /// of a section): `write_together` reports it, and both sites keep
    /// their bytes and protection. evan-kolberg's way of making the refusal
    /// happen under Wine (#278).
    #[cfg(windows)]
    #[test]
    fn a_refused_site_leaves_the_group_as_it_was() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE,
            MEM_RESERVE, MEMORY_BASIC_INFORMATION, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
            UnmapViewOfFile, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
        };
        let protection = |at: usize| {
            // SAFETY: plain data.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the given size.
            let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            assert_ne!(n, 0);
            info.Protect
        };
        // SAFETY: memory of our own, released at the end.
        unsafe {
            // The first site: a private page, read-only as the image's code.
            let first = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            ) as *mut u8;
            assert!(!first.is_null());
            std::ptr::copy_nonoverlapping(OWN_SCALE.bytes.as_ptr(), first, OWN_SCALE.bytes.len());
            let mut old = 0;
            assert_ne!(
                VirtualProtect(first.cast(), 0x1000, PAGE_READONLY, &mut old),
                0
            );
            // The second: a section seeded through a writable view, then seen
            // through a read-only one, which cannot be made writable.
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                0x1000,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let writer = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 0x1000);
            assert!(!writer.Value.is_null());
            std::ptr::copy_nonoverlapping(
                POOL_CHECK.bytes.as_ptr(),
                writer.Value.cast::<u8>(),
                POOL_CHECK.bytes.len(),
            );
            UnmapViewOfFile(writer);
            let reader = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0x1000);
            let second = reader.Value as *mut u8;
            assert!(!second.is_null());

            // One "image" from the lower of the two to past the higher.
            let base = (first as usize).min(second as usize);
            let size = (first as usize).max(second as usize) + 0x1000 - base;
            let a = Site {
                at: first as usize - base,
                what: "test",
                bytes: OWN_SCALE.bytes,
            };
            let b = Site {
                at: second as usize - base,
                what: "test",
                bytes: POOL_CHECK.bytes,
            };
            let (scale, check) = (own_scale_code(0.5), pool_check_code());
            assert!(matches!(
                write_together(base, size, &[(&a, &scale), (&b, &check)]),
                Patched::Failed(_)
            ));
            assert_eq!(
                std::slice::from_raw_parts(first, OWN_SCALE.bytes.len()),
                OWN_SCALE.bytes
            );
            assert_eq!(
                std::slice::from_raw_parts(second, POOL_CHECK.bytes.len()),
                POOL_CHECK.bytes
            );
            assert_eq!(protection(first as usize), PAGE_READONLY);
            assert_eq!(protection(second as usize), PAGE_READONLY);

            UnmapViewOfFile(reader);
            CloseHandle(section);
            VirtualFree(first.cast(), 0, MEM_RELEASE);
        }
    }
}
