//! Finds the AVX masked stores in a title's code (`vpmaskmovd`,
//! `vpmaskmovq`, `vmaskmovps` and `vmaskmovpd` to memory).
//!
//! Rosetta cannot resume one of these that faults partway (#14), so
//! fault-based write tracking is off under Rosetta and every submission
//! copies title memory whole: about 129 GB per frame once the world
//! streams in (#50). Knowing where the title uses these stores is the
//! first step to patching them so tracking can come back on. The scan
//! reads bytes, not decoded instructions, so a match inside another
//! instruction's bytes is possible, though rare.

use victory_pe::ImageInfo;

/// `IMAGE_SCN_MEM_EXECUTE`.
const EXECUTE: u32 = 0x2000_0000;

/// One masked store found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskedStore {
    pub rva: u32,
    pub kind: MaskedKind,
    /// 256-bit (`ymm`) rather than 128-bit.
    pub wide: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskedKind {
    Vpmaskmovd,
    Vpmaskmovq,
    Vmaskmovps,
    Vmaskmovpd,
    /// Not masked, but a vector store Rosetta also cannot resume (#50).
    Vcvtps2ph,
    Vextractf128,
    Vextracti128,
}

/// The masked store whose encoding starts at `at`, if one does: a
/// three-byte VEX prefix (`c4`) in the 0F38 map with the 66 prefix, then
/// opcode 8e (`vpmaskmov`, W picks d or q), 2e (`vmaskmovps`) or 2f
/// (`vmaskmovpd`), and a memory operand.
fn decode(code: &[u8], at: usize) -> Option<(MaskedKind, bool)> {
    let &[c4, b1, b2, opcode, modrm] = code.get(at..at + 5)? else {
        return None;
    };
    if c4 != 0xc4 || b2 & 3 != 1 || modrm >> 6 == 3 {
        return None;
    }
    let w = b2 & 0x80 != 0;
    let kind = match (b1 & 0x1f, opcode) {
        (2, 0x8e) if w => MaskedKind::Vpmaskmovq,
        (2, 0x8e) => MaskedKind::Vpmaskmovd,
        (2, 0x2e) => MaskedKind::Vmaskmovps,
        (2, 0x2f) => MaskedKind::Vmaskmovpd,
        (3, 0x1d) if !w => MaskedKind::Vcvtps2ph,
        (3, 0x19) if !w && b2 & 4 != 0 => MaskedKind::Vextractf128,
        (3, 0x39) if !w && b2 & 4 != 0 => MaskedKind::Vextracti128,
        _ => return None,
    };
    Some((kind, b2 & 4 != 0))
}

/// Every masked store in the executable sections of `bytes` (the image
/// file `info` was parsed from), by RVA.
pub fn masked_stores(info: &ImageInfo, bytes: &[u8]) -> Vec<MaskedStore> {
    let mut found = Vec::new();
    for s in info
        .sections
        .iter()
        .filter(|s| s.characteristics & EXECUTE != 0)
    {
        let start = (s.file_offset as usize).min(bytes.len());
        let end = start.saturating_add(s.file_size as usize).min(bytes.len());
        let code = &bytes[start..end];
        for at in 0..code.len() {
            if let Some((kind, wide)) = decode(code, at) {
                found.push(MaskedStore {
                    rva: s.rva + at as u32,
                    kind,
                    wide,
                });
            }
        }
    }
    found
}

/// One line for the start-up report: how many of each kind, and the first
/// few RVAs.
pub fn summary(found: &[MaskedStore]) -> String {
    if found.is_empty() {
        return "none in code".to_owned();
    }
    let count = |k| found.iter().filter(|f| f.kind == k).count();
    let first: Vec<String> = found
        .iter()
        .take(16)
        .map(|f| format!("{:#x}", f.rva))
        .collect();
    format!(
        "{} in code (vpmaskmovd {}, vpmaskmovq {}, vmaskmovps {}, vmaskmovpd {}, vcvtps2ph {}, vextractf128 {}, vextracti128 {}), first at {}{}",
        found.len(),
        count(MaskedKind::Vpmaskmovd),
        count(MaskedKind::Vpmaskmovq),
        count(MaskedKind::Vmaskmovps),
        count(MaskedKind::Vmaskmovpd),
        count(MaskedKind::Vcvtps2ph),
        count(MaskedKind::Vextractf128),
        count(MaskedKind::Vextracti128),
        first.join(" "),
        if found.len() > 16 { " ..." } else { "" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_store_seen_in_the_rosetta_hang() {
        // `vpmaskmovd [rax], ymm6, ymm4` from #1 and #50: c4 e2 45 8e 30.
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x8e, 0x30], 0),
            Some((MaskedKind::Vpmaskmovd, true))
        );
        // W set: vpmaskmovq; 128-bit.
        assert_eq!(
            decode(&[0xc4, 0xe2, 0xc1, 0x8e, 0x30], 0),
            Some((MaskedKind::Vpmaskmovq, false))
        );
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x2e, 0x08], 0),
            Some((MaskedKind::Vmaskmovps, true))
        );
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x2f, 0x08], 0),
            Some((MaskedKind::Vmaskmovpd, true))
        );
    }

    #[test]
    fn leaves_loads_registers_and_other_maps_alone() {
        // The load forms (8c, 2c) are not stores.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8c, 0x30], 0), None);
        // A register operand (mod 3) is not a store to memory.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8e, 0xf0], 0), None);
        // The 0F map, or no 66 prefix.
        assert_eq!(decode(&[0xc4, 0xe1, 0x45, 0x8e, 0x30], 0), None);
        assert_eq!(decode(&[0xc4, 0xe2, 0x44, 0x8e, 0x30], 0), None);
        // Too short.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8e], 0), None);
    }

    #[test]
    fn the_summary_counts_and_lists() {
        let f = |rva, kind| MaskedStore {
            rva,
            kind,
            wide: true,
        };
        let found = [
            f(0x1000, MaskedKind::Vpmaskmovd),
            f(0x2000, MaskedKind::Vmaskmovps),
        ];
        assert_eq!(
            summary(&found),
            "2 in code (vpmaskmovd 1, vpmaskmovq 0, vmaskmovps 1, vmaskmovpd 0, vcvtps2ph 0, vextractf128 0, vextracti128 0), first at 0x1000 0x2000"
        );
        assert_eq!(summary(&[]), "none in code");
    }
}

/// A decoded masked store, enough to redo it with plain stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Store {
    /// Instruction length in bytes.
    pub len: usize,
    /// Lanes of 8 bytes (`q`, `pd`) rather than 4.
    pub wide_lanes: bool,
    /// 256-bit registers.
    pub ymm: bool,
    /// The mask register (VEX.vvvv) and the stored register (ModRM.reg).
    pub mask: u8,
    pub src: u8,
    /// The memory operand: REX X and B, then ModRM, an optional SIB and
    /// the displacement.
    pub x: bool,
    pub b: bool,
    pub modrm: u8,
    pub sib: Option<u8>,
    pub disp: i32,
    /// For an unmasked store (`vcvtps2ph`, `vextract*128`, a plain 256-bit
    /// move): the three-byte VEX prefix's two bytes, the opcode, the
    /// immediate if there is one, and the bytes stored.
    pub plain: Option<(u8, u8, u8, Option<u8>, u8)>,
}

impl Store {
    fn lanes(&self) -> usize {
        match (self.ymm, self.wide_lanes) {
            (true, false) => 8,
            (true, true) | (false, false) => 4,
            (false, true) => 2,
        }
    }
    fn lane_bytes(&self) -> u8 {
        if self.wide_lanes { 8 } else { 4 }
    }
    /// RIP-relative: ModRM mod 0, r/m 5.
    fn rip_relative(&self) -> bool {
        self.modrm >> 6 == 0 && self.modrm & 7 == 5
    }
    /// A SIB with no base (mod 0, base 5): an absolute disp32.
    fn no_base(&self) -> bool {
        self.modrm >> 6 == 0 && self.sib.is_some_and(|s| s & 7 == 5)
    }
    /// The base register is rsp.
    fn rsp_base(&self) -> bool {
        !self.b && self.sib.is_some_and(|s| s & 7 == 4)
    }
}

/// The memory operand whose ModRM byte is at `at`: ModRM, the SIB byte if
/// any, the displacement, and where the operand ends.
fn operand(code: &[u8], at: usize) -> Option<(u8, Option<u8>, i32, usize)> {
    let modrm = *code.get(at)?;
    let mut end = at + 1;
    let (md, rm) = (modrm >> 6, modrm & 7);
    let sib = if rm == 4 {
        end += 1;
        Some(*code.get(at + 1)?)
    } else {
        None
    };
    let disp_len = match md {
        1 => 1,
        2 => 4,
        _ if rm == 5 || sib.is_some_and(|s| s & 7 == 5) => 4,
        _ => 0,
    };
    let bytes = code.get(end..end + disp_len)?;
    let disp = match disp_len {
        1 => i32::from(bytes[0] as i8),
        4 => i32::from_le_bytes(bytes.try_into().ok()?),
        _ => 0,
    };
    Some((modrm, sib, disp, end + disp_len))
}

/// Decodes the whole masked store at `at`: its operands and length.
pub(crate) fn decode_store(code: &[u8], at: usize) -> Option<Store> {
    let (kind, ymm) = decode(code, at)?;
    let (b1, b2) = (code[at + 1], code[at + 2]);
    let (modrm, sib, disp, mut end) = operand(code, at + 4)?;
    let plain = match kind {
        MaskedKind::Vcvtps2ph | MaskedKind::Vextractf128 | MaskedKind::Vextracti128 => {
            let imm = *code.get(end)?;
            end += 1;
            let bytes = if kind == MaskedKind::Vcvtps2ph && !ymm {
                8
            } else {
                16
            };
            Some((b1, b2, code[at + 3], Some(imm), bytes))
        }
        _ => None,
    };
    Some(Store {
        plain,
        len: end - at,
        wide_lanes: matches!(kind, MaskedKind::Vpmaskmovq | MaskedKind::Vmaskmovpd),
        ymm,
        mask: !(b2 >> 3) & 0xf,
        src: ((modrm >> 3) & 7) | if b1 & 0x80 == 0 { 8 } else { 0 },
        x: b1 & 0x40 == 0,
        b: b1 & 0x20 == 0,
        modrm,
        sib,
        disp,
    })
}

/// Decodes the plain 256-bit store at `at` that need not be aligned:
/// `vmovups`, `vmovupd` or `vmovdqu` to memory, with a two- or three-byte
/// VEX prefix. Rosetta can fail these the same way when they fault
/// partway (#144). They are far more common than
/// masked stores and the scan does not look for them, so only RVAs named
/// in `VICTORY_PATCH_MASKED` are decoded this way. Under 5 bytes there is
/// no room for the jump, so those are left alone.
pub(crate) fn decode_wide_move(code: &[u8], at: usize) -> Option<Store> {
    let (b1, b2, vex) = match *code.get(at)? {
        0xc4 => (*code.get(at + 1)?, *code.get(at + 2)?, 3),
        // The two-byte form as the three-byte one: its R, no X or B
        // (set bits), the 0F map, W 0.
        0xc5 => {
            let p = *code.get(at + 1)?;
            (p & 0x80 | 0x61, p & 0x7f, 2)
        }
        _ => return None,
    };
    let opcode = *code.get(at + vex)?;
    // The 0F map, no second source (vvvv 1111), 256-bit; then 11 with no
    // prefix or 66 (`vmovups`, `vmovupd`) or 7f with F3 (`vmovdqu`).
    let moves = matches!((opcode, b2 & 3), (0x11, 0 | 1) | (0x7f, 2));
    if b1 & 0x1f != 1 || b2 & 0x7c != 0x7c || !moves {
        return None;
    }
    let (modrm, sib, disp, end) = operand(code, at + vex + 1)?;
    if modrm >> 6 == 3 || end - at < 5 {
        return None;
    }
    Some(Store {
        plain: Some((b1, b2, opcode, None, 32)),
        len: end - at,
        wide_lanes: false,
        ymm: true,
        mask: 0,
        src: ((modrm >> 3) & 7) | if b1 & 0x80 == 0 { 8 } else { 0 },
        x: b1 & 0x40 == 0,
        b: b1 & 0x20 == 0,
        modrm,
        sib,
        disp,
    })
}

/// Bytes the stack moves before the trampoline's `lea`: rax, rcx, flags.
const PUSHED: i32 = 24;

/// The trampoline for `s`, placed at `at`, for the store at `site`: the
/// same stores done one lane at a time with plain `mov`s (which Rosetta
/// can resume after a fault), then a jump back past the store. Flags,
/// rax, rcx and the vector registers are kept.
pub(crate) fn trampoline(s: &Store, site: u64, at: u64) -> Option<Vec<u8>> {
    let mut c = vec![0x50, 0x51, 0x9c]; // push rax; push rcx; pushfq
    // lea rax, [operand]
    c.push(0x48 | u8::from(s.x) << 1 | u8::from(s.b));
    c.push(0x8d);
    let rm = s.modrm & 7;
    if s.rip_relative() {
        c.push(rm); // mod 0, reg 0 (rax), r/m 5
        let target = (site as i64) + s.len as i64 + i64::from(s.disp);
        let next = at as i64 + c.len() as i64 + 4;
        c.extend_from_slice(&i32::try_from(target - next).ok()?.to_le_bytes());
    } else if s.no_base() {
        c.push(rm);
        c.push(s.sib?);
        c.extend_from_slice(&s.disp.to_le_bytes());
    } else {
        // Always a disp32 form (mod 2), so rsp's shift always fits.
        c.push(0x80 | rm);
        if let Some(sib) = s.sib {
            c.push(sib);
        }
        let disp = if s.rsp_base() {
            s.disp.checked_add(PUSHED)?
        } else {
            s.disp
        };
        c.extend_from_slice(&disp.to_le_bytes());
    }
    if let Some((b1, b2, opcode, imm, bytes)) = s.plain {
        // The same instruction into [rsp] (X and B cleared, ModRM.reg
        // kept), then the bytes copied out 8 at a time with plain moves.
        c.extend_from_slice(&[0x48, 0x83, 0xec, 0x20]); // sub rsp, 32
        c.extend_from_slice(&[
            0xc4,
            b1 | 0x60,
            b2,
            opcode,
            0x40 | (s.modrm & 0x38) | 4,
            0x24,
            0,
        ]);
        c.extend(imm);
        for k in 0..bytes / 8 {
            c.extend_from_slice(&[0x48, 0x8b, 0x4c, 0x24, k * 8]); // mov rcx, [rsp + 8k]
            c.extend_from_slice(&[0x48, 0x89, 0x48, k * 8]); // mov [rax + 8k], rcx
        }
        c.extend_from_slice(&[0x48, 0x83, 0xc4, 0x20]); // add rsp, 32
        return tail(c, site, s.len, at);
    }
    c.extend_from_slice(&[0x48, 0x83, 0xec, 0x40]); // sub rsp, 64
    // vmovdqu [rsp], mask; vmovdqu [rsp + 32], src (F3 0F 7F /r).
    for (reg, disp) in [(s.mask, None), (s.src, Some(0x20u8))] {
        c.push(0xc4);
        c.push(if reg & 8 == 0 { 0x80 } else { 0 } | 0x60 | 0x01);
        c.push(0x78 | if s.ymm { 4 } else { 0 } | 0x02);
        c.push(0x7f);
        c.push(if disp.is_some() { 0x40 } else { 0 } | (reg & 7) << 3 | 4);
        c.push(0x24);
        if let Some(d) = disp {
            c.push(d);
        }
    }
    let (n, size) = (s.lanes(), s.lane_bytes());
    let rex = if size == 8 { Some(0x48) } else { None };
    for i in 0..n as u8 {
        let off = i * size;
        // mov ecx/rcx, [rsp + off]; test; jns skip
        c.extend(rex);
        c.extend_from_slice(&[0x8b, 0x4c, 0x24, off]);
        c.extend(rex);
        c.extend_from_slice(&[0x85, 0xc9]);
        // The two moves after the jump.
        let skip = 2 * u8::from(rex.is_some()) + 7;
        c.extend_from_slice(&[0x79, skip]);
        // mov ecx/rcx, [rsp + 32 + off]; mov [rax + off], ecx/rcx
        c.extend(rex);
        c.extend_from_slice(&[0x8b, 0x4c, 0x24, 0x20 + off]);
        c.extend(rex);
        c.extend_from_slice(&[0x89, 0x48, off]);
    }
    c.extend_from_slice(&[0x48, 0x83, 0xc4, 0x40]); // add rsp, 64
    tail(c, site, s.len, at)
}

/// Restores flags, rcx and rax, and jumps back past the store.
fn tail(mut c: Vec<u8>, site: u64, len: usize, at: u64) -> Option<Vec<u8>> {
    c.extend_from_slice(&[0x9d, 0x59, 0x58]); // popfq; pop rcx; pop rax
    let back = (site as i64) + len as i64;
    let next = at as i64 + c.len() as i64 + 5;
    c.push(0xe9);
    c.extend_from_slice(&i32::try_from(back - next).ok()?.to_le_bytes());
    Some(c)
}

/// The bytes that replace the store at `site`: a jump to `to`, then
/// `nop`s to the store's length.
pub(crate) fn jump(site: u64, len: usize, to: u64) -> Option<Vec<u8>> {
    let rel = i32::try_from(to as i64 - (site as i64 + 5)).ok()?;
    let mut c = vec![0xe9];
    c.extend_from_slice(&rel.to_le_bytes());
    c.resize(len.max(5), 0x90);
    Some(c)
}

/// The bytes of the executable section that holds `rva`, in the image
/// file, and where `rva` is in them.
#[cfg(windows)]
fn code_at<'a>(info: &ImageInfo, file: &'a [u8], rva: u32) -> Option<(&'a [u8], usize)> {
    let s = info
        .sections
        .iter()
        .find(|s| s.characteristics & EXECUTE != 0 && s.rva <= rva && rva - s.rva < s.file_size)?;
    let start = s.file_offset as usize;
    let code = file.get(start..start.checked_add(s.file_size as usize)?)?;
    Some((code, (rva - s.rva) as usize))
}

/// Patches the stores of the image mapped at `base` that `wanted` picks
/// to jump to trampolines of plain stores, and returns how many it
/// patched. Before any title thread runs.
#[cfg(windows)]
pub fn patch(base: u64, info: &ImageInfo, file: &[u8], wanted: &Wanted) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
        VirtualAlloc, VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut stores: Vec<(u32, Store)> = masked_stores(info, file)
        .into_iter()
        .filter(|m| wanted.picks(m.rva))
        .filter_map(|m| {
            let (code, at) = code_at(info, file, m.rva)?;
            Some((m.rva, decode_store(code, at)?))
        })
        .collect();
    // Named RVAs the scan did not find: plain 256-bit stores (#144).
    for &rva in &wanted.rvas {
        if stores.iter().any(|(r, _)| *r == rva) {
            continue;
        }
        match code_at(info, file, rva).and_then(|(code, at)| decode_wide_move(code, at)) {
            Some(store) => stores.push((rva, store)),
            None => tracing::warn!(
                rva = format_args!("{rva:#x}"),
                "not a masked store or a plain 256-bit store; left as it is"
            ),
        }
    }
    if stores.is_empty() {
        return 0;
    }
    // Trampolines within reach of a rel32 jump: below the image first.
    const SIZE: usize = 1 << 20;
    let mut pool = std::ptr::null_mut();
    for step in 1..2048u64 {
        let Some(want) = base.checked_sub(step << 20) else {
            break;
        };
        // SAFETY: a plain reservation at a hint; failure returns null.
        pool = unsafe {
            VirtualAlloc(
                want as *const _,
                SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if !pool.is_null() {
            break;
        }
    }
    if pool.is_null() {
        tracing::warn!("no memory near the image for masked store trampolines");
        return 0;
    }
    let mut used = 0usize;
    let mut patched = 0;
    for (rva, store) in &stores {
        let site = base + u64::from(*rva);
        let at = pool as u64 + used as u64;
        let (Some(code), Some(jump)) = (trampoline(store, site, at), jump(site, store.len, at))
        else {
            tracing::warn!(rva = format_args!("{rva:#x}"), "masked store out of reach");
            continue;
        };
        if used + code.len() > SIZE {
            break;
        }
        let mut old = 0;
        // SAFETY: our pool has room for the code; the site is the title's
        // code, writable for the moment of the write, before any title
        // thread runs.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
            if VirtualProtect(
                site as *const _,
                jump.len(),
                PAGE_EXECUTE_READWRITE,
                &mut old,
            ) == 0
            {
                continue;
            }
            std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
            VirtualProtect(site as *const _, jump.len(), old, &mut old);
        }
        used = (used + code.len() + 15) & !15;
        patched += 1;
    }
    let mut old = 0;
    // SAFETY: our pool; the whole image range for the cache flush.
    unsafe {
        VirtualProtect(pool, SIZE, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    patched
}

/// What to patch, from `VICTORY_PATCH_MASKED`: its value (`all` or hex
/// RVAs, for [`wanted_from`]), nothing for `none`, and `all` when it is
/// unset and `needed` (fault tracking under Rosetta).
pub fn choice(value: Option<String>, needed: bool) -> Option<String> {
    match value.as_deref() {
        Some("none") => None,
        Some(_) => value,
        None => needed.then(|| "all".to_owned()),
    }
}

/// The stores `VICTORY_PATCH_MASKED` asks to patch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Wanted {
    /// Every masked store the scan finds.
    pub all: bool,
    /// Stores named by RVA: masked ones, or plain 256-bit ones the scan
    /// does not look for (#144).
    pub rvas: Vec<u32>,
}

impl Wanted {
    /// Whether the masked store at `rva` is wanted.
    pub fn picks(&self, rva: u32) -> bool {
        self.all || self.rvas.contains(&rva)
    }
}

/// What `VICTORY_PATCH_MASKED` asks for: a comma list of `all` and hex
/// RVAs, such as `all,1e09b31`. Nothing when unset.
pub fn wanted_from(value: Option<&str>) -> Wanted {
    let mut wanted = Wanted::default();
    for item in value.unwrap_or_default().split(',').map(str::trim) {
        if item == "all" {
            wanted.all = true;
        } else if let Ok(rva) = u32::from_str_radix(item.trim_start_matches("0x"), 16) {
            wanted.rvas.push(rva);
        }
    }
    wanted
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    #[test]
    fn stores_decode_with_their_operands_and_length() {
        // The store from the Rosetta hang (#1): VEX.vvvv 1000 is mask ymm7,
        // ModRM.reg 6 the stored ymm6.
        let s = decode_store(&[0xc4, 0xe2, 0x45, 0x8e, 0x30], 0).unwrap();
        assert_eq!(
            (s.len, s.mask, s.src, s.ymm, s.wide_lanes),
            (5, 7, 6, true, false)
        );
        // vpmaskmovd [rsp+0x10], ymm1, ymm2: SIB and disp8.
        let s = decode_store(&[0xc4, 0xe2, 0x75, 0x8e, 0x54, 0x24, 0x10], 0).unwrap();
        assert_eq!((s.len, s.disp, s.rsp_base()), (7, 0x10, true));
        // vmaskmovpd [rip+0x100], xmm1, xmm2: RIP-relative disp32.
        let s = decode_store(&[0xc4, 0xe2, 0x71, 0x2f, 0x15, 0, 1, 0, 0], 0).unwrap();
        assert_eq!(
            (s.len, s.disp, s.rip_relative(), s.lanes()),
            (9, 0x100, true, 2)
        );
    }

    #[test]
    fn the_store_from_the_second_hang_decodes() {
        // vcvtps2ph [r8+rdx*2], ymm0, 3 from run 11 in #50.
        let s = decode_store(&[0xc4, 0xc3, 0x7d, 0x1d, 0x04, 0x50, 0x03], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.x, s.sib, s.plain.map(|p| (p.3, p.4))),
            (7, true, false, Some(0x50), Some((Some(3), 16)))
        );
    }

    #[test]
    fn the_plain_store_from_144_decodes() {
        // vmovups [r8+0x34], ymm0 at 0x1E09B31 in #144.
        let s = decode_wide_move(&[0xc4, 0xc1, 0x7c, 0x11, 0x40, 0x34], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.x, s.src, s.disp, s.plain),
            (6, true, false, 0, 0x34, Some((0xc1, 0x7c, 0x11, None, 32)))
        );
        // vmovdqu [rcx+8], ymm9 with the two-byte prefix: R comes along.
        let s = decode_wide_move(&[0xc5, 0x7e, 0x7f, 0x49, 0x08], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.src, s.plain),
            (5, false, 9, Some((0x61, 0x7e, 0x7f, None, 32)))
        );
        // vmovupd [rsp+0x20], ymm1: the SIB byte.
        let s = decode_wide_move(&[0xc5, 0xfd, 0x11, 0x4c, 0x24, 0x20], 0).unwrap();
        assert_eq!((s.len, s.disp, s.rsp_base()), (6, 0x20, true));
    }

    #[test]
    fn only_unaligned_256_bit_stores_are_plain_wide_moves() {
        let no = |bytes: &[u8]| assert_eq!(decode_wide_move(bytes, 0), None, "{bytes:02x?}");
        // 128-bit: one store Rosetta can resume.
        no(&[0xc4, 0xc1, 0x78, 0x11, 0x40, 0x34]);
        // Aligned: vmovaps (29), vmovdqa (66 7f); 32 aligned bytes never
        // cross a page.
        no(&[0xc4, 0xc1, 0x7c, 0x29, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7d, 0x7f, 0x40, 0x34]);
        // Loads (10, F3 6f) and a register operand.
        no(&[0xc4, 0xc1, 0x7c, 0x10, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7e, 0x6f, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7c, 0x11, 0xc0, 0x90]);
        // vvvv not 1111, or not the 0F map.
        no(&[0xc4, 0xc1, 0x74, 0x11, 0x40, 0x34]);
        no(&[0xc4, 0xc2, 0x7c, 0x11, 0x40, 0x34]);
        // vmovups [rax], ymm0: 4 bytes, no room for the jump.
        no(&[0xc5, 0xfc, 0x11, 0x00]);
        // Cut short.
        no(&[0xc4, 0xc1, 0x7c, 0x11, 0x40]);
    }

    #[test]
    fn the_jump_pads_to_the_store_length() {
        assert_eq!(
            jump(0x1000, 7, 0x2000).unwrap(),
            vec![0xe9, 0xfb, 0x0f, 0, 0, 0x90, 0x90]
        );
        assert!(jump(0, 5, 1 << 40).is_none());
    }

    #[test]
    fn the_list_picks_rvas() {
        let w = wanted_from(Some("1dac734,0x10"));
        assert!(w.picks(0x1dac734) && w.picks(0x10) && !w.picks(0x11));
        assert!(wanted_from(Some("all")).picks(5));
        assert!(!wanted_from(None).picks(5));
        assert_eq!(wanted_from(Some("")), Wanted::default());
        // `all` and RVAs together.
        assert_eq!(
            wanted_from(Some("all, 1e09b31,1E09B37")),
            Wanted {
                all: true,
                rvas: vec![0x1e09b31, 0x1e09b37]
            }
        );
    }

    #[test]
    fn stores_are_patched_by_default_only_when_needed() {
        let s = |v: &str| Some(v.to_owned());
        assert_eq!(choice(None, true), s("all"));
        assert_eq!(choice(None, false), None);
        assert_eq!(choice(s("none"), true), None);
        assert_eq!(choice(s("1e09b31"), true), s("1e09b31"));
        assert_eq!(choice(s("all"), false), s("all"));
    }

    /// Real code with a masked store, run as written and then patched:
    /// both write the same lanes. Needs a CPU (or Rosetta) with AVX2.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_store_writes_what_the_store_did() {
        use windows_sys::Win32::System::Memory::*;
        // f(dst rcx, mask rdx, src r8): vmovdqu ymm1, [rdx]; vmovdqu ymm2,
        // [r8]; <store>; vzeroupper; ret.
        let cases: [&[u8]; 10] = [
            &[0xc4, 0xe2, 0x75, 0x8e, 0x11],       // vpmaskmovd [rcx], ymm1, ymm2
            &[0xc4, 0xe2, 0xf5, 0x8e, 0x51, 0x08], // vpmaskmovq [rcx+8], ymm1, ymm2
            &[0xc4, 0xe2, 0x71, 0x2e, 0x54, 0x01, 0x04], // vmaskmovps [rcx+rax+4], xmm1, xmm2 (rax 0)
            &[0xc4, 0xe3, 0x7d, 0x1d, 0x09, 0x00],       // vcvtps2ph [rcx], ymm1, 0
            &[0xc4, 0xe3, 0x79, 0x1d, 0x49, 0x04, 0x03], // vcvtps2ph [rcx+4], xmm1, 3
            &[0xc4, 0xe3, 0x7d, 0x19, 0x49, 0x10, 0x01], // vextractf128 [rcx+16], ymm1, 1
            &[0xc4, 0xe3, 0x7d, 0x39, 0x09, 0x01],       // vextracti128 [rcx], ymm1, 1
            &[0xc4, 0xe1, 0x7c, 0x11, 0x51, 0x04],       // vmovups [rcx+4], ymm2
            &[0xc4, 0xe1, 0x7d, 0x11, 0x11],             // vmovupd [rcx], ymm2
            &[0xc5, 0xfe, 0x7f, 0x49, 0x0c],             // vmovdqu [rcx+12], ymm1
        ];
        for (k, store) in cases.into_iter().enumerate() {
            let mut code = vec![
                0x31, 0xc0, 0xc5, 0xfe, 0x6f, 0x0a, 0xc4, 0xc1, 0x7e, 0x6f, 0x10,
            ];
            let site_at = code.len();
            code.extend_from_slice(store);
            code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
            let run = |patch: bool| -> [u32; 12] {
                // SAFETY: our own executable page, written then run.
                unsafe {
                    let page = VirtualAlloc(
                        std::ptr::null(),
                        1 << 16,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_EXECUTE_READWRITE,
                    ) as *mut u8;
                    std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
                    if patch {
                        let s = decode_store(&code, site_at)
                            .or_else(|| decode_wide_move(&code, site_at))
                            .unwrap();
                        let site = page as u64 + site_at as u64;
                        let at = page as u64 + 0x1000;
                        let t = trampoline(&s, site, at).unwrap();
                        std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
                        let j = jump(site, s.len, at).unwrap();
                        std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
                    }
                    let f: extern "win64" fn(*mut u32, *const u32, *const u32) =
                        std::mem::transmute(page);
                    let mask: [u32; 8] = [
                        0x8000_0000,
                        0,
                        0xffff_ffff,
                        0,
                        0,
                        0x8000_0000,
                        0x8000_0000,
                        1,
                    ];
                    let src: [u32; 8] = [11, 22, 33, 44, 55, 66, 77, 88];
                    let mut dst = [7u32; 12];
                    f(dst.as_mut_ptr(), mask.as_ptr(), src.as_ptr());
                    VirtualFree(page.cast(), 0, MEM_RELEASE);
                    dst
                }
            };
            let (plain, patched) = (run(false), run(true));
            assert_eq!(plain, patched, "case {k}");
            assert_ne!(plain, [7u32; 12], "case {k} wrote something");
        }
    }

    /// The point of the patch: a patched store into a read-only page
    /// faults on a plain store, a handler makes the page writable, and the
    /// store resumes. The original masked store would hang Rosetta here
    /// (#14), so it is never run this way.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_store_resumes_after_a_write_fault() {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::*;
        use windows_sys::Win32::System::Memory::*;
        static PAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        unsafe extern "system" fn unguard(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes valid pointers.
            unsafe {
                let r = &*(*info).ExceptionRecord;
                let page = PAGE.load(std::sync::atomic::Ordering::Relaxed);
                if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                    && (r.ExceptionInformation[1] & !0xfff) == page
                {
                    let mut old = 0;
                    VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old);
                    return -1; // EXCEPTION_CONTINUE_EXECUTION
                }
            }
            0
        }
        let store = [0xc4, 0xe2, 0x75, 0x8e, 0x11]; // vpmaskmovd [rcx], ymm1, ymm2
        let mut code = vec![0xc5, 0xfe, 0x6f, 0x0a, 0xc4, 0xc1, 0x7e, 0x6f, 0x10];
        let site_at = code.len();
        code.extend_from_slice(&store);
        code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
        // SAFETY: our own pages and handler.
        unsafe {
            let exec = VirtualAlloc(
                std::ptr::null(),
                1 << 16,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            std::ptr::copy_nonoverlapping(code.as_ptr(), exec, code.len());
            let s = decode_store(&code, site_at).unwrap();
            let site = exec as u64 + site_at as u64;
            let at = exec as u64 + 0x1000;
            let t = trampoline(&s, site, at).unwrap();
            std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
            let j = jump(site, s.len, at).unwrap();
            std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
            let data = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READONLY,
            ) as *mut u32;
            PAGE.store(data as usize, std::sync::atomic::Ordering::Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(unguard));
            let f: extern "win64" fn(*mut u32, *const u32, *const u32) = std::mem::transmute(exec);
            let mask: [u32; 8] = [0, 0x8000_0000, 0, 0, 0, 0, 0, 0x8000_0000];
            let src: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            f(data, mask.as_ptr(), src.as_ptr());
            RemoveVectoredExceptionHandler(handler);
            let got = std::slice::from_raw_parts(data, 8).to_vec();
            assert_eq!(got, vec![0, 2, 0, 0, 0, 0, 0, 8]);
            VirtualFree(data.cast(), 0, MEM_RELEASE);
            VirtualFree(exec.cast(), 0, MEM_RELEASE);
        }
    }

    /// A patched `vmovups` like the one in #144, 32 bytes that start in a
    /// writable page and end in a read-only one. A plain `mov` faults
    /// instead, a handler makes the page writable, and the store resumes
    /// with every byte written. One of the `mov`s itself crosses the page
    /// boundary.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_wide_store_resumes_across_a_page() {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::*;
        use windows_sys::Win32::System::Memory::*;
        static PAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        unsafe extern "system" fn unguard(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes valid pointers.
            unsafe {
                let r = &*(*info).ExceptionRecord;
                let page = PAGE.load(std::sync::atomic::Ordering::Relaxed);
                // A write that crosses into the page may report its start,
                // in the page before.
                let at = r.ExceptionInformation[1];
                if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                    && at.wrapping_sub(page - 0x1000) < 0x2000
                {
                    let mut old = 0;
                    VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old);
                    return -1; // EXCEPTION_CONTINUE_EXECUTION
                }
            }
            0
        }
        // f(dst rcx, _, src r8): vmovdqu ymm2, [r8]; vmovups [rcx+4], ymm2;
        // vzeroupper; ret.
        let mut code = vec![0xc4, 0xc1, 0x7e, 0x6f, 0x10];
        let site_at = code.len();
        code.extend_from_slice(&[0xc4, 0xe1, 0x7c, 0x11, 0x51, 0x04]);
        code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
        // SAFETY: our own pages and handler.
        unsafe {
            let exec = VirtualAlloc(
                std::ptr::null(),
                1 << 16,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            std::ptr::copy_nonoverlapping(code.as_ptr(), exec, code.len());
            let s = decode_wide_move(&code, site_at).unwrap();
            let site = exec as u64 + site_at as u64;
            let at = exec as u64 + 0x1000;
            let t = trampoline(&s, site, at).unwrap();
            std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
            let j = jump(site, s.len, at).unwrap();
            std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
            let data = VirtualAlloc(
                std::ptr::null(),
                0x2000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as *mut u8;
            let guarded = data.add(0x1000);
            let mut old = 0;
            VirtualProtect(guarded.cast(), 0x1000, PAGE_READONLY, &mut old);
            PAGE.store(guarded as usize, std::sync::atomic::Ordering::Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(unguard));
            let f: extern "win64" fn(*mut u8, *const u32, *const u32) = std::mem::transmute(exec);
            // The store covers 12 bytes before the boundary and 20 after.
            let dst = guarded.sub(16);
            let src: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            f(dst, std::ptr::null(), src.as_ptr());
            RemoveVectoredExceptionHandler(handler);
            let got = std::slice::from_raw_parts(dst.add(4).cast::<u32>(), 8).to_vec();
            assert_eq!(got, src.to_vec());
            // The handler ran: the page is writable now.
            VirtualProtect(guarded.cast(), 0x1000, PAGE_READWRITE, &mut old);
            assert_eq!(old, PAGE_READWRITE);
            VirtualFree(data.cast(), 0, MEM_RELEASE);
            VirtualFree(exec.cast(), 0, MEM_RELEASE);
        }
    }
}
