//! macOS port: a way out of one title lookup that can loop forever.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) has a hash-table lookup at
//! RVA 0x50eb750 whose probe loop only ends when it finds its key. On the
//! Mac the key is sometimes absent a few minutes into the opening sequence:
//! the calling thread then probes forever at full speed, the title's other
//! threads wait for it, and its watchdog ends the run about 30 s later
//! (notes/PLAN.md, "Cause of the stall found").
//!
//! `VICTORY_GUARD_LOOKUP=1` replaces the loop's last two instructions
//! (`inc eax; jmp <loop head>`) with a jump to a small trampoline. It does
//! the same, counts the passes in the upper half of r10 (the loop reads only
//! r10d, which the function loaded with a 32-bit move), and after [`LIMIT`]
//! passes counts a give-up and leaves through the function's own epilogue,
//! as its "nothing to do" path does. This is a stopgap that keeps the run
//! alive, not a fix for the missing key.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// `inc eax; jmp <loop head>`, the end of the probe loop.
const SITE: u32 = 0x050e_b9d1;
const SITE_BYTES: [u8; 7] = [0xff, 0xc0, 0xe9, 0x37, 0xfe, 0xff, 0xff];
/// `and eax, r10d`, the loop head.
const LOOP_HEAD: u32 = 0x050e_b80f;
const LOOP_HEAD_BYTES: [u8; 3] = [0x44, 0x21, 0xd0];
/// `add rsp, 0x28; pop rbx; ...; ret`.
const EPILOGUE: u32 = 0x050e_b910;
const EPILOGUE_BYTES: [u8; 5] = [0x48, 0x83, 0xc4, 0x28, 0x5b];
/// Passes before giving up: more groups than a table of the title's memory
/// could hold (4 million groups of 144 bytes are 576 MB).
const LIMIT: u32 = 1 << 22;

/// Where the give-up counter lives once [`patch`] has run (0 before).
static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn rel32(code: &mut Vec<u8>, at: u64, target: u64) -> Option<()> {
    let next = at.checked_add(code.len() as u64 + 4)?;
    let delta = i32::try_from((target as i64).checked_sub(next as i64)?).ok()?;
    code.extend(delta.to_le_bytes());
    Some(())
}

/// The trampoline placed at `at` for the image at `base`, counting give-ups
/// in the dword at `counter`; `None` if a target is out of rel32 reach.
fn trampoline(base: u64, at: u64, counter: u64) -> Option<Vec<u8>> {
    let mut c = vec![
        0xff, 0xc0, // inc eax (the instruction we replaced)
        0x49, 0xc1, 0xc2, 0x20, // rol r10, 32: the pass count comes down
        0x49, 0xff, 0xc2, // inc r10
        0x41, 0x81, 0xfa, // cmp r10d, LIMIT
    ];
    c.extend(LIMIT.to_le_bytes());
    c.extend([0x49, 0xc1, 0xc2, 0x20]); // rol r10, 32: the mask is back; ZF is kept
    c.extend([0x0f, 0x85]); // jne <loop head>
    rel32(&mut c, at, base + u64::from(LOOP_HEAD))?;
    c.extend([0xf0, 0xff, 0x05]); // lock inc dword [rip + counter]
    rel32(&mut c, at, counter)?;
    c.push(0xe9); // jmp <epilogue>
    rel32(&mut c, at, base + u64::from(EPILOGUE))?;
    Some(c)
}

/// `jmp <trampoline>` and two nops over the seven bytes at the site.
fn jump(site: u64, at: u64) -> Option<[u8; 7]> {
    let delta = i32::try_from((at as i64).checked_sub(site as i64 + 5)?).ok()?;
    let d = delta.to_le_bytes();
    Some([0xe9, d[0], d[1], d[2], d[3], 0x90, 0x90])
}

/// Patches the lookup of the image mapped at `base` if it is the build this
/// was written for. Before any title thread runs. Says what it did.
#[cfg(windows)]
pub fn patch(base: u64) -> Result<(), &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let at = |rva: u32| (base + u64::from(rva)) as *const u8;
    // SAFETY: the image is mapped and these RVAs are inside its code.
    let known = unsafe {
        std::slice::from_raw_parts(at(SITE), SITE_BYTES.len()) == SITE_BYTES
            && std::slice::from_raw_parts(at(LOOP_HEAD), LOOP_HEAD_BYTES.len()) == LOOP_HEAD_BYTES
            && std::slice::from_raw_parts(at(EPILOGUE), EPILOGUE_BYTES.len()) == EPILOGUE_BYTES
    };
    if !known {
        return Err("not the build it was written for (bytes differ)");
    }
    // One page of code and one of data within reach of a rel32 jump.
    const PAGE: usize = 0x1000;
    let mut pool = std::ptr::null_mut();
    for step in 1..2048u64 {
        let Some(want) = base.checked_sub(step << 20) else { break };
        // SAFETY: a plain reservation at a hint; failure returns null.
        pool = unsafe { VirtualAlloc(want as *const _, 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if !pool.is_null() {
            break;
        }
    }
    if pool.is_null() {
        return Err("no memory near the image for the trampoline");
    }
    let code_at = pool as u64;
    let counter = code_at + PAGE as u64;
    let site = base + u64::from(SITE);
    let (Some(code), Some(jump)) = (trampoline(base, code_at, counter), jump(site, code_at)) else {
        return Err("the trampoline is out of reach");
    };
    let mut old = 0;
    // SAFETY: our pages hold the code; the site is the title's code, made
    // writable for the write, before any title thread runs.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), code_at as *mut u8, code.len());
        VirtualProtect(pool, PAGE, PAGE_EXECUTE_READ, &mut old);
        if VirtualProtect(site as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("the title's code could not be made writable");
        }
        std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
        VirtualProtect(site as *const _, jump.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    COUNTER.store(counter as usize, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Starts a thread that logs when the patched lookup has given up (once a
/// second at most, with the total so far).
#[cfg(windows)]
pub fn watch() {
    let at = COUNTER.load(std::sync::atomic::Ordering::Relaxed) as *const std::sync::atomic::AtomicU32;
    if at.is_null() {
        return;
    }
    let at = at as usize;
    let _ = std::thread::Builder::new().name("victory lookup guard".into()).spawn(move || {
        let mut seen = 0;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            // SAFETY: our own page, alive for the process's lifetime.
            let total = unsafe { &*(at as *const std::sync::atomic::AtomicU32) }
                .load(std::sync::atomic::Ordering::Relaxed);
            if total != seen {
                tracing::warn!(
                    total,
                    new = total.wrapping_sub(seen),
                    "the title's lookup at +0x50eb750 gave up: its key is not in the table (VICTORY_GUARD_LOOKUP)"
                );
                seen = total;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trampoline_jumps_back_into_the_image() {
        let (base, at, counter) = (0x6fff_ed55_0000u64, 0x6fff_ed45_0000u64, 0x6fff_ed45_1000u64);
        let code = trampoline(base, at, counter).unwrap();
        assert_eq!(code.len(), 38);
        assert_eq!(&code[..2], &SITE_BYTES[..2], "the replaced inc eax comes first");
        let target = |end: usize| {
            let delta = i32::from_le_bytes(code[end - 4..end].try_into().unwrap());
            (at + end as u64).wrapping_add(delta as i64 as u64)
        };
        assert_eq!(target(26), base + u64::from(LOOP_HEAD));
        assert_eq!(target(33), counter);
        assert_eq!(target(38), base + u64::from(EPILOGUE));
        assert_eq!(u32::from_le_bytes(code[12..16].try_into().unwrap()), LIMIT);
    }

    #[test]
    fn the_site_jump_lands_on_the_trampoline_and_fills_seven_bytes() {
        let (site, at) = (0x6fff_f263_b9d1u64, 0x6fff_ed45_0000u64);
        let j = jump(site, at).unwrap();
        let delta = i32::from_le_bytes(j[1..5].try_into().unwrap());
        assert_eq!((site + 5).wrapping_add(delta as i64 as u64), at);
        assert_eq!(&j[5..], &[0x90, 0x90]);
        assert!(jump(site, site + (1 << 40)).is_none(), "out of reach is refused");
    }

    /// The replaced jump's own target is the loop head we return to.
    #[test]
    fn the_replaced_jump_went_to_the_loop_head() {
        let delta = i32::from_le_bytes(SITE_BYTES[3..7].try_into().unwrap());
        assert_eq!((i64::from(SITE) + 7 + i64::from(delta)) as u32, LOOP_HEAD);
    }
}
