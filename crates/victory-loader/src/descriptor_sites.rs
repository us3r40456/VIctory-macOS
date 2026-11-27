//! macOS port: the title's own code that writes descriptor records into its
//! shader-visible heap tells the layer which slots it wrote
//! (`VICTORY_HEAP_REPORTS`, read in `victory-gpu`'s `descriptor::reports`;
//! off unless asked for). From avatarmanz 9997d255 ("renderer port 2"): the
//! sites, their bytes and the code they jump to are that fork's.
//!
//! What the title does (Grand Theft Auto VI 1.0.2.1, b99f876b...): each
//! draw's and each dispatch's descriptor tables are taken from its frame
//! allocator, whose memory is the placed shader-visible view heap, and
//! filled by two functions, +0x4d9efd0 (graphics) and +0x4eec8e0 (compute),
//! three loops each, one record at a time: `and r, 7FE0h` (the slot's byte
//! offset in the table), `vmovups ymm0, [record]`, `vmovups [table + r],
//! ymm0`. The fork found them from the write faults they take and says a
//! search of all of the code for that instruction shape finds these six
//! stores and no other.
//!
//! Each store becomes a jump to code of ours that does the store, then, if
//! its address lies in the heap `victory-gpu` names in its cells, sets the
//! slot's byte in a map of slots and its 128 slots' byte in a map of
//! chunks, and jumps back. The flags are clobbered: at every site the next
//! instructions are a `movzx` and a `cmp`, which sets them before any use
//! (part of the bytes checked). Every register is put back.
//!
//! What it costs a store here: the store itself, unchanged (the same
//! instruction at another address: it faults on a guarded heap page as it
//! did, and is resumed as it was), two jumps and twenty plain
//! instructions; no call, no lock. The code touches the stack (three
//! pushes), the cells and the two maps, which are committed memory of
//! ours: nothing in it can raise an exception that the store did not.
//!
//! All six windows are compared with the bytes expected before anything
//! is written; if one differs (another build of the title, or a store
//! already patched by `VICTORY_PATCH_MASKED=<rva>`) none is patched and
//! the reason is returned.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch (its forms are `victory-gpu`'s).
pub const SWITCH: &str = "VICTORY_HEAP_REPORTS";

/// The cells the code reads, `#[repr(C)]` in `victory-gpu`
/// (`descriptor::reports::Marks`): the heap's first byte, its length in
/// bytes (0: none, nothing is marked), the map of slots (a byte each), the
/// map of chunks (a byte for 128 slots).
pub const MARKS_LEN: usize = 32;

/// A store of the title's to hook: the window's offset in the image, the
/// store's offset in the window and its length, and the window's bytes
/// (from the `and` to the `cmp`).
pub struct Site {
    pub window: usize,
    pub store: usize,
    pub len: usize,
    pub bytes: &'static [u8],
}

/// The six record stores of the table fillers (image offsets).
pub const SITES: [Site; 6] = [
    // +0x4d9efd0, loop 1: [rcx+rsi]
    Site {
        window: 0x4d9f04f,
        store: 10,
        len: 5,
        bytes: &[
            0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6,
            0x47, 0x0b, 0x39, 0xc5,
        ],
    },
    // loop 2: [rcx+rsi]
    Site {
        window: 0x4d9f10d,
        store: 10,
        len: 5,
        bytes: &[
            0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6,
            0x47, 0x0f, 0x41, 0x39, 0xc5,
        ],
    },
    // loop 3: [r9+rsi]
    Site {
        window: 0x4d9f19c,
        store: 12,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc4, 0xc1, 0x7c, 0x10, 0x02, 0xc4, 0xc1, 0x7c, 0x11, 0x04,
            0x31, 0x44, 0x0f, 0xb6, 0x4f, 0x0d, 0x44, 0x39, 0xc8,
        ],
    },
    // +0x4eec8e0, loop 1: [r9+r8]
    Site {
        window: 0x4eec95d,
        store: 11,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x01,
            0x0f, 0xb6, 0x42, 0x0b, 0x39, 0xc5,
        ],
    },
    // loop 2: [rdi+r8]
    Site {
        window: 0x4eeca08,
        store: 10,
        len: 6,
        bytes: &[
            0x81, 0xe7, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x06, 0xc4, 0xa1, 0x7c, 0x11, 0x04, 0x07, 0x0f,
            0xb6, 0x72, 0x0d, 0x39, 0xf0,
        ],
    },
    // loop 3: [r10+r8]
    Site {
        window: 0x4eeca82,
        store: 11,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe2, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x02,
            0x0f, 0xb6, 0x42, 0x0f, 0x41, 0x39, 0xc6,
        ],
    },
];

/// `lea rax, [base+index]` for a `vmovups [base+index], ymm0` (VEX, ModRM
/// 04h with a SIB byte, no displacement, the stack pointer not its base):
/// `None` for another form.
fn lea_of(store: &[u8]) -> Option<[u8; 4]> {
    let (x, b, op_at) = match *store.first()? {
        0xc5 => (false, false, 2),
        0xc4 => {
            let p = *store.get(1)?;
            if p & 0x1f != 1 {
                return None;
            }
            (p & 0x40 == 0, p & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let (op, modrm, sib) = (*store.get(op_at)?, *store.get(op_at + 1)?, *store.get(op_at + 2)?);
    if op != 0x11 || modrm != 0x04 || store.len() != op_at + 3 {
        return None;
    }
    // No base (mode 0, base 101), or the stack pointer as base, or rsp as index (none).
    if sib & 7 == 5 || (sib & 7 == 4 && !b) || (sib >> 3 & 7 == 4 && !x) {
        return None;
    }
    Some([0x48 | u8::from(x) << 1 | u8::from(b), 0x8d, 0x04, sib])
}

/// The code at `to` for the store `store` at `at`: the store, the marks,
/// back to `at + store.len()`. `None` for a store of another form, or with
/// the way back out of a jump's reach.
fn stub(store: &[u8], at: u64, to: u64, marks: u64) -> Option<Vec<u8>> {
    let lea = lea_of(store)?;
    let mut out = store.to_vec();
    out.push(0x50); // push rax
    out.extend(lea); // lea rax, [base+index]
    out.extend([0x52, 0x51]); // push rdx ; push rcx
    out.extend([0x48, 0xba]); // mov rdx, marks
    out.extend(marks.to_le_bytes());
    out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
    out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
    out.extend([0x73, 0x18]); // jae past the marks
    let marking = out.len();
    out.extend([0x48, 0xc1, 0xe8, 0x05]); // shr rax, 5: the slot
    out.extend([0x48, 0x8b, 0x4a, 0x10]); // mov rcx, [rdx+16]
    out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
    out.extend([0x48, 0xc1, 0xe8, 0x07]); // shr rax, 7: the chunk
    out.extend([0x48, 0x8b, 0x4a, 0x18]); // mov rcx, [rdx+24]
    out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
    debug_assert_eq!(out.len() - marking, 0x18, "what the jae jumps over");
    out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
    let back = at as i64 + store.len() as i64;
    let next = to as i64 + out.len() as i64 + 5;
    out.push(0xe9); // jmp back
    out.extend(i32::try_from(back - next).ok()?.to_le_bytes());
    Some(out)
}

/// Why the stores were not patched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// A window lies outside the image.
    Outside,
    /// The bytes at the window with this image offset are not the ones
    /// expected: another build, or already patched.
    Differs(usize),
    /// No memory for the code within a jump's reach of the title's.
    NoRoom,
    /// A store is not of the form the code is written for.
    Form(usize),
    /// The title's code could not be made writable.
    Protected,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::Outside => write!(f, "not the build it was written for (the image is too small)"),
            Refused::Differs(at) => write!(f, "not the build it was written for (bytes differ at +{at:#x})"),
            Refused::NoRoom => write!(f, "no memory near the image for the code"),
            Refused::Form(at) => write!(f, "the store at +{at:#x} is not of the expected form, or out of reach"),
            Refused::Protected => write!(f, "the title's code could not be made writable"),
        }
    }
}

/// Hooks every one of `sites` in the image of `size` bytes at `base`, or
/// none: the code reads the cells at `marks` ([`MARKS_LEN`] bytes). How
/// many were hooked.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet; `marks` stays
/// valid for as long as the image runs.
#[cfg(windows)]
pub unsafe fn apply(base: usize, size: usize, marks: usize, sites: &[Site]) -> Result<usize, Refused> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // Every window first: nothing is written unless all are as expected.
    for site in sites {
        if site.window.checked_add(site.bytes.len()).is_none_or(|end| end > size)
            || site.store + site.len > site.bytes.len()
        {
            return Err(Refused::Outside);
        }
        // SAFETY: inside the mapped image (checked above).
        let there = unsafe { std::slice::from_raw_parts((base + site.window) as *const u8, site.bytes.len()) };
        if there != site.bytes {
            return Err(Refused::Differs(site.window));
        }
    }
    // One page of code within reach of a rel32 jump: below the image, as
    // the other patches' code; above it if nothing is free below (an image
    // mapped low, as a test's is).
    const PAGE: usize = 0x1000;
    let above = (base + size).next_multiple_of(1 << 20);
    let wants = (1..2048usize)
        .filter_map(|step| base.checked_sub(step << 20))
        .chain((0..2048usize).filter_map(|step| above.checked_add(step << 20)).take_while(|want| want - base < 1 << 31));
    let mut page = std::ptr::null_mut();
    for want in wants {
        // SAFETY: a plain reservation at a hint; failure returns null.
        page = unsafe { VirtualAlloc(want as *const _, PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if !page.is_null() {
            break;
        }
    }
    if page.is_null() {
        return Err(Refused::NoRoom);
    }
    // The code of every store, then the jumps to it.
    let mut code: Vec<u8> = Vec::new();
    let mut jumps: Vec<(usize, Vec<u8>)> = Vec::new();
    for site in sites {
        let at = base + site.window + site.store;
        let to = page as usize + code.len();
        let store = &site.bytes[site.store..site.store + site.len];
        let (Some(body), Some(jump)) =
            (stub(store, at as u64, to as u64, marks as u64), crate::masked::jump(at as u64, site.len, to as u64))
        else {
            return Err(Refused::Form(site.window + site.store));
        };
        code.extend(body);
        code.resize(code.len().next_multiple_of(16), 0xcc);
        jumps.push((at, jump));
    }
    if code.len() > PAGE {
        return Err(Refused::NoRoom);
    }
    let mut old = 0;
    // SAFETY: our page holds the code and is made executable before a jump
    // leads to it; each site is the title's store, made writable for the
    // write and put back, before any title thread runs.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), page.cast::<u8>(), code.len());
        VirtualProtect(page, PAGE, PAGE_EXECUTE_READ, &mut old);
        // Writable first, all of them: none is patched if one cannot be.
        for (at, jump) in &jumps {
            if VirtualProtect(*at as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return Err(Refused::Protected);
            }
            VirtualProtect(*at as *const _, jump.len(), old, &mut old);
        }
        for (at, jump) in &jumps {
            if VirtualProtect(*at as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) != 0 {
                std::ptr::copy_nonoverlapping(jump.as_ptr(), *at as *mut u8, jump.len());
                VirtualProtect(*at as *const _, jump.len(), old, &mut old);
            }
        }
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    Ok(jumps.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sites_lea_their_stores_address() {
        let leas: Vec<[u8; 4]> = SITES.iter().map(|s| lea_of(&s.bytes[s.store..s.store + s.len]).unwrap()).collect();
        assert_eq!(leas[0], [0x48, 0x8d, 0x04, 0x31]); // lea rax, [rcx+rsi]
        assert_eq!(leas[1], [0x48, 0x8d, 0x04, 0x31]);
        assert_eq!(leas[2], [0x49, 0x8d, 0x04, 0x31]); // [r9+rsi]
        assert_eq!(leas[3], [0x4b, 0x8d, 0x04, 0x01]); // [r9+r8]
        assert_eq!(leas[4], [0x4a, 0x8d, 0x04, 0x07]); // [rdi+r8]
        assert_eq!(leas[5], [0x4b, 0x8d, 0x04, 0x02]); // [r10+r8]
        for s in &SITES {
            // The window starts with the `and r, 7FE0h` that makes the
            // offset a slot's, the store follows the record's load, and a
            // movzx and the cmp that sets the flags follow the store.
            let rex = usize::from(s.bytes[0] == 0x41);
            assert_eq!((s.bytes[rex], &s.bytes[rex + 2..rex + 6]), (0x81, &[0xe0, 0x7f, 0x00, 0x00][..]));
            let after = &s.bytes[s.store + s.len..];
            let movzx = if after[0] == 0x0f { 0 } else { 1 };
            assert_eq!(&after[movzx..movzx + 2], &[0x0f, 0xb6]);
            assert_eq!(after[after.len() - 2], 0x39, "cmp");
            // Room for the jump.
            assert!(s.len >= 5);
        }
        // A store with a displacement, of the stack, of another kind: not taken.
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x44, 0x31, 0x08]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x04, 0x24]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x10, 0x04, 0x31]).is_none());
    }

    /// The code for each store: the store's own bytes first, the cells'
    /// address, pushes and pops that pair up, and a jump that lands right
    /// behind the store; out of a jump's reach there is none.
    #[test]
    fn the_code_does_the_store_first_and_comes_back_behind_it() {
        for s in &SITES {
            let store = &s.bytes[s.store..s.store + s.len];
            let (at, to, marks) = (0x1_4000_0000u64 + s.window as u64, 0x1_3ff0_0040u64, 0x7ff6_1234_5678u64);
            let code = stub(store, at, to, marks).expect("the store's form");
            assert_eq!(&code[..s.len], store, "the store itself, unchanged");
            assert_eq!(code[s.len], 0x50, "then rax is saved");
            let cells = code.windows(10).position(|w| w[..2] == [0x48, 0xba]).expect("mov rdx, marks");
            assert_eq!(code[cells + 2..cells + 10], marks.to_le_bytes());
            assert_eq!(code[code.len() - 8..code.len() - 5], [0x59, 0x5a, 0x58], "rcx, rdx, rax are put back");
            let rel = i32::from_le_bytes(code[code.len() - 4..].try_into().unwrap());
            assert_eq!(code[code.len() - 5], 0xe9);
            assert_eq!((to + code.len() as u64).wrapping_add_signed(i64::from(rel)), at + s.len as u64);
            // The jump past the marks lands on the pops.
            let jae = code.windows(2).position(|w| w == [0x73, 0x18]).expect("jae");
            assert_eq!(jae + 2 + 0x18, code.len() - 8);
            assert!(code.len() <= 80, "{} bytes", code.len());
            assert!(stub(store, at, at + (3 << 30), marks).is_none(), "out of reach");
        }
    }

    /// The cells `victory-gpu` keeps, for a test.
    #[cfg(windows)]
    #[repr(C)]
    struct Marks {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    /// Each site's window in code of a test's, twice, one copy hooked, both
    /// run (under Wine: by Rosetta): the record is stored as before, the
    /// registers the window leaves are the same, and the slot and its chunk
    /// are marked when the table is in the heap, not otherwise. (The
    /// fork's test, with this line's `apply`.)
    #[cfg(windows)]
    #[test]
    fn a_hooked_site_stores_and_marks_its_slot() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // mov to, from (64 bits)
        let mov = |to: u8, from: u8| [0x48 | (from >> 3) << 2 | (to >> 3), 0x89, 0xc0 | (from & 7) << 3 | (to & 7)];
        for site in &SITES {
            let store = &site.bytes[site.store..site.store + site.len];
            let lea = lea_of(store).unwrap();
            let base = lea[3] & 7 | (lea[0] & 1) << 3;
            let index = lea[3] >> 3 & 7 | (lea[0] & 2) << 2;
            let load_at = site.store - if site.bytes[site.store - 5] == 0xc4 { 5 } else { 4 };
            let load = &site.bytes[load_at..site.store];
            let load_reg = if load[0] == 0xc4 { load[4] & 7 | 8 } else { load[3] & 7 };
            // fn(offset: rcx, table: rdx, record: r8, scratch: r9) -> rax:
            // the store's address as the window leaves its registers.
            let mut code: Vec<u8> = vec![0x56, 0x57, 0x53]; // push rsi ; push rdi ; push rbx
            code.extend(mov(11, 1)); // r11 = offset
            code.extend(mov(3, 2)); // rbx = table
            code.extend(mov(6, 8)); // rsi = record
            for scratch in [7, 10, 2] {
                code.extend(mov(scratch, 9)); // rdi, r10, rdx: what the movzx reads
            }
            // The window's `and` masks the offset's register; the table is in
            // the other. (The record first: rsi may be one of them.)
            let rex = usize::from(site.bytes[0] == 0x41);
            let offset = site.bytes[rex + 1] & 7 | if rex == 1 { 8 } else { 0 };
            assert!(offset == base || offset == index);
            let table = if offset == base { index } else { base };
            code.extend(mov(load_reg, 6));
            code.extend(mov(table, 3));
            code.extend(mov(offset, 11));
            let at = code.len();
            code.extend(site.bytes);
            code.extend(lea); // lea rax, [base+index]
            code.extend([0x5b, 0x5f, 0x5e]); // pop rbx ; pop rdi ; pop rsi
            code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
            type F = unsafe extern "system" fn(u64, *mut u8, *const u8, *mut u8) -> u64;
            // SAFETY: fresh memory for code and data, one copy changed before it runs.
            unsafe {
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(place, 0);
                let (plain, hooked) = (place + 0x100 - at, place + 0x1100 - at);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), hooked as *mut u8, code.len());
                let heap = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
                let scratch = [0u8; 64];
                let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let marks = Box::new(Marks { start: heap as u64 + 0x2000, len: 0x8000, slots: slots.as_mut_ptr(), chunks: chunks.as_mut_ptr() });
                let mine = [Site { window: 0x1100, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), Ok(1));
                assert_eq!(*((place + 0x1100 + site.store) as *const u8), 0xe9);
                let (plain, hooked): (F, F) = (std::mem::transmute(plain), std::mem::transmute(hooked));
                let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x5a);
                // In the heap: slot 0x40 + 3 of its start.
                let table = heap.add(0x2000 + 0x40 * 32);
                let first = plain(0x60 | 0x8000_0000, table.add(0x4000), record.as_ptr(), scratch.as_ptr().cast_mut());
                let second = hooked(0x60 | 0x8000_0000, table, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(first.wrapping_sub(0x4000), second, "the registers the window leaves");
                assert_eq!(std::slice::from_raw_parts(table.add(0x60), 32), &record);
                assert_eq!(std::slice::from_raw_parts(table.add(0x4060), 32), &record);
                let slot = 0x40 + 3;
                assert_eq!(slots[slot], 1);
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 1);
                assert_eq!(chunks[slot / 128], 1);
                // Before the heap, and past it: stored, not marked.
                slots.fill(0);
                chunks.fill(0);
                for outside in [heap.add(0x1000), heap.add(0x2000 + 0x8000)] {
                    hooked(0x20, outside, record.as_ptr(), scratch.as_ptr().cast_mut());
                    assert_eq!(std::slice::from_raw_parts(outside.add(0x20), 32), &record);
                    assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                }
                // No heap named (length 0): stored, not marked.
                let none = Box::new(Marks { start: 0, len: 0, slots: std::ptr::null_mut(), chunks: std::ptr::null_mut() });
                let copy = place + 0x2100 - at;
                std::ptr::copy_nonoverlapping(code.as_ptr(), copy as *mut u8, code.len());
                assert_eq!(apply(place, 0x10000, &raw const *none as usize, &[Site { window: 0x2100, ..*site }]), Ok(1));
                let unnamed: F = std::mem::transmute(copy);
                unnamed(0x40, table, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(std::slice::from_raw_parts(table.add(0x40), 32), &record);
                drop((marks, none));
            }
        }
    }

    /// One window that differs: nothing is written anywhere.
    #[cfg(windows)]
    #[test]
    fn other_bytes_leave_every_store_alone() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // SAFETY: fresh memory of our own.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            let good = SITES[0].bytes;
            let mut bad = SITES[1].bytes.to_vec();
            bad[3] = 0x7e; // and ecx, 7EE0h
            std::ptr::copy_nonoverlapping(good.as_ptr(), place.add(0x100), good.len());
            std::ptr::copy_nonoverlapping(bad.as_ptr(), place.add(0x200), bad.len());
            let mine = [Site { window: 0x100, ..SITES[0] }, Site { window: 0x200, ..SITES[1] }];
            assert_eq!(apply(place as usize, 0x10000, 0, &mine), Err(Refused::Differs(0x200)));
            assert_eq!(std::slice::from_raw_parts(place.add(0x100), good.len()), good, "the good one is not patched");
            assert_eq!(std::slice::from_raw_parts(place.add(0x200), bad.len()), &bad[..]);
            let outside = [Site { window: 0xfff0, ..SITES[0] }];
            assert_eq!(apply(place as usize, 0x10000, 0, &outside), Err(Refused::Outside));
        }
    }
}
