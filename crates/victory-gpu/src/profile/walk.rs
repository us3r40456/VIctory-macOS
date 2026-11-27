//! The walk of a stopped thread's stack, in a copy taken while it stood
//! still (`super::sampler`), from hyphennn f85101b
//! (`crates/victory-gpu/src/profile.rs` there: `Sampler::walk`,
//! `frame_fits`, `epilogue_fits`), cut to what the sampler here needs.
//!
//! The thread runs again while its copy is walked, so every read goes to
//! the copy: the stack pointer, and every saved register that points into
//! the copied part of the stack (frame pointers are such registers), are
//! moved into the copy before each step, and a step is taken only if all
//! it can read lies in the copy or in the zeroes after it.

use std::ptr::null_mut;

use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, IMAGE_RUNTIME_FUNCTION_ENTRY, RtlLookupFunctionEntry, RtlVirtualUnwind,
    UNW_FLAG_NHANDLER,
};

/// Zeroed bytes kept after the room for a stack's copy: what one frame's
/// unwinding may read beyond the copy at most.
pub(super) const SLACK: usize = 2 << 20;

/// A stack copied by the sampler: `got` bytes from the stack pointer, at
/// `base + offset` (`offset` keeps the stack's 16-byte alignment, which the
/// unwinder's vector loads need), in `room` bytes of ours that are zeroes
/// after the copy.
pub(super) struct Copied {
    pub(super) base: usize,
    pub(super) room: usize,
    pub(super) offset: usize,
    pub(super) got: usize,
}

/// Walks the copy of the stack of a thread whose registers are `context`
/// (its stack being `low..high`, as its thread block said) into `frames`,
/// from the instruction pointer up: how many were found. Return addresses
/// are taken minus one, inside their call instruction.
///
/// # Safety
/// `copied` describes memory of ours as its doc says.
pub(super) unsafe fn walk(
    context: &mut CONTEXT,
    copied: &Copied,
    low: usize,
    high: usize,
    frames: &mut [u64],
) -> usize {
    if frames.is_empty() {
        return 0;
    }
    let lo = context.Rsp as usize;
    let got = copied.got;
    let start = copied.base + copied.offset;
    let end = start + got;
    let limit = copied.base + copied.room;
    let top = if lo >= low && lo < high {
        high
    } else {
        lo.saturating_add(got)
    };
    frames[0] = context.Rip;
    let mut n = 1;
    let (mut blind, mut last) = (0, 0);
    while n < frames.len() {
        // Into the copy: what points into the stack as it was.
        for register in [
            &mut context.Rsp,
            &mut context.Rbx,
            &mut context.Rbp,
            &mut context.Rsi,
            &mut context.Rdi,
            &mut context.R12,
            &mut context.R13,
            &mut context.R14,
            &mut context.R15,
        ] {
            let value = *register as usize;
            if value >= lo && value < top {
                *register = if value - lo < got {
                    (start + (value - lo)) as u64
                } else {
                    // Above what was copied: reads there find zeroes.
                    end as u64
                };
            }
        }
        let rsp = context.Rsp as usize;
        // Out of the copy, or not upwards: the walk ends.
        if rsp < start || rsp + 8 > end || rsp <= last {
            break;
        }
        last = rsp;
        let mut image = 0;
        // SAFETY: looks up the unwind data covering `Rip`, if any; the
        // thread that was stopped runs again, so its locks are free.
        let entry = unsafe { RtlLookupFunctionEntry(context.Rip, &mut image, null_mut()) };
        if entry.is_null() {
            // Code without unwind data is a leaf function (the system's
            // call stubs are): its return address is at `rsp`. Three in a
            // row is no stack of calls.
            blind += 1;
            if blind > 2 {
                break;
            }
            // SAFETY: `rsp..rsp + 8` is inside the copy.
            context.Rip = unsafe { (rsp as *const u64).read_unaligned() };
            context.Rsp += 8;
        } else {
            blind = 0;
            // SAFETY: `entry` is `image`'s, as just looked up.
            if !unsafe { frame_fits(image, entry, context, start, end, limit) } {
                break;
            }
            let (mut data, mut frame) = (null_mut(), 0);
            // SAFETY: `entry` belongs to `image`; the context is the
            // caller's aligned copy, and what the step reads lies in our
            // copy of the stack or the zeroes after it (checked above).
            unsafe {
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    image,
                    context.Rip,
                    entry,
                    context,
                    &mut data,
                    &mut frame,
                    null_mut(),
                )
            };
        }
        if context.Rip == 0 {
            break;
        }
        frames[n] = context.Rip - 1;
        n += 1;
    }
    n
}

/// Whether unwinding one frame of the function `entry` of the image at
/// `image` from `context` reads only our copy of a stack (`start..end`)
/// and the zeroes after it (to `limit`): the stack pointer, and the frame
/// pointer if the function has set one up, must lie in the copy, and what
/// the function's unwind codes take off the stack must fit in the zeroes.
///
/// `RtlVirtualUnwind` trusts the registers it is given. Ours come from a
/// thread stopped anywhere, and a return address may be none (code
/// without unwind data in the middle of a stack): a frame pointer that
/// points nowhere would be read through.
///
/// # Safety
/// `entry` is the function table entry of `image` for `context.Rip`.
unsafe fn frame_fits(
    image: u64,
    entry: *const IMAGE_RUNTIME_FUNCTION_ENTRY,
    context: &CONTEXT,
    start: usize,
    end: usize,
    limit: usize,
) -> bool {
    // SAFETY: the caller's entry, and the unwind data it names in its
    // loaded image: a header of four bytes, `count` two-byte codes, and
    // after them (made even) the entry of the function this one is a
    // part of, if the header says so.
    unsafe {
        let [begin, finish, mut data] = entry.cast::<[u32; 3]>().read_unaligned();
        let at = context
            .Rip
            .wrapping_sub(image)
            .wrapping_sub(u64::from(begin));
        // The frame register the function itself names (its first part).
        let mut pointer = 0;
        // Bytes above the frame's base that unwinding may read: the
        // return address, a machine frame's tail.
        let mut reach = 64usize;
        // Bases other than the stack pointer: frame pointers set up.
        let mut bases = [0usize; 4];
        let mut based = 0;
        for part in 0..8 {
            if data & 1 != 0 {
                return false;
            }
            let info = (image + u64::from(data)) as *const u8;
            let [version, _prolog, count, frame] = info.cast::<[u8; 4]>().read();
            if !matches!(version & 7, 1 | 2) {
                return false;
            }
            let chained = version >> 3 & 4 != 0;
            if part == 0 {
                pointer = frame & 15;
            }
            let codes = info.add(4);
            let count = usize::from(count);
            let slot = |i: usize| codes.add(2 * i).cast::<u16>().read_unaligned();
            let mut i = 0;
            let (mut found, mut set) = (false, false);
            while i < count {
                let [offset, op] = codes.add(2 * i).cast::<[u8; 2]>().read();
                let (op, op_info) = (op & 15, usize::from(op >> 4));
                i += match op {
                    // A register pushed.
                    0 => {
                        reach += 8;
                        1
                    }
                    // A large allocation: its size in the next one or two slots.
                    1 if op_info == 0 => {
                        reach += usize::from(slot(i + 1)) * 8;
                        2
                    }
                    1 => {
                        reach += usize::from(slot(i + 1)) | usize::from(slot(i + 2)) << 16;
                        3
                    }
                    // A small allocation.
                    2 => {
                        reach += op_info * 8 + 8;
                        1
                    }
                    // The frame pointer set: done already if the thread is
                    // past that instruction, or in a later part of the function.
                    3 => {
                        found = true;
                        set = part > 0 || chained || at >= u64::from(offset);
                        1
                    }
                    // Registers saved in the frame (near, far), an epilogue
                    // note and a spare code, vector registers saved.
                    4 | 8 => 2,
                    5 | 9 => 3,
                    6 => 1,
                    7 => 2,
                    // A machine frame.
                    10 => {
                        reach += 48;
                        1
                    }
                    _ => return false,
                };
            }
            if frame & 15 != 0 && (set || !found || part > 0 || chained) {
                let register = gpr(context, frame & 15) as usize;
                let back = usize::from(frame >> 4) * 16;
                if register < start + back || register > end || based == bases.len() {
                    return false;
                }
                bases[based] = register - back;
                based += 1;
            }
            if !chained {
                let rsp = context.Rsp as usize;
                let code = (image + u64::from(finish)).saturating_sub(context.Rip);
                return rsp >= start
                    && rsp < end
                    && reach <= SLACK
                    && rsp + reach <= limit
                    && bases[..based].iter().all(|base| base + reach <= limit)
                    && epilogue_fits(context, code as usize, pointer, start, end);
            }
            let parent = codes
                .add(2 * ((count + 1) & !1))
                .cast::<[u32; 3]>()
                .read_unaligned();
            data = parent[2];
        }
        false
    }
}

/// Whether the instructions at `context.Rip` (`code` bytes of them are the
/// function's), if the unwinder takes them for the end of a function,
/// leave the stack pointer inside our copy of a stack (`start..end`).
/// `pointer` is the function's frame register (0: none).
///
/// Where a thread stands at a function's last instructions, the unwinder
/// does not use the unwind codes: it runs those instructions itself, an
/// `add rsp, n` or a `lea rsp, [frame register + n]`, then the `pop`s,
/// then the return. An address that is no instruction's start can read as
/// a `pop rsp`, or as an `add` of any number: this reads the bytes as the
/// unwinder does and refuses what would take the stack pointer out of the
/// copy.
///
/// # Safety
/// `code` bytes at `context.Rip` are readable.
unsafe fn epilogue_fits(
    context: &CONTEXT,
    code: usize,
    pointer: u8,
    start: usize,
    end: usize,
) -> bool {
    let rip = context.Rip as *const u8;
    // Past the function's end reads as a breakpoint: nothing of an epilogue.
    let byte = |i: usize| {
        if i < code {
            // SAFETY: inside the `code` bytes the caller says are readable.
            unsafe { rip.add(i).read() }
        } else {
            0xcc
        }
    };
    let number = |i: usize| i32::from_le_bytes([byte(i), byte(i + 1), byte(i + 2), byte(i + 3)]);
    let moved = |from: u64, by: i32| (from as usize).wrapping_add(by as isize as usize);
    let mut rsp = context.Rsp as usize;
    let mut at = 0;
    if byte(0) == 0x48 && byte(1) == 0x83 && byte(2) == 0xc4 {
        // add rsp, n (a byte).
        rsp = moved(context.Rsp, i32::from(byte(3) as i8));
        at = 4;
    } else if byte(0) == 0x48 && byte(1) == 0x81 && byte(2) == 0xc4 {
        // add rsp, n.
        rsp = moved(context.Rsp, number(3));
        at = 7;
    } else if byte(0) & 0xfe == 0x48 && byte(1) == 0x8d {
        // lea rsp, [register + n]: taken only from the frame register.
        let register = (byte(0) & 1) << 3 | byte(2) & 7;
        let (near, far) = (byte(2) & 0xf8 == 0x60, byte(2) & 0xf8 == 0xa0);
        if register != 0 && register == pointer && (near || far) {
            let by = if near {
                i32::from(byte(3) as i8)
            } else {
                number(3)
            };
            rsp = moved(gpr(context, register), by);
            at = if near { 4 } else { 7 };
        }
    }
    // The `pop`s.
    let (mut pops, mut pops_rsp) = (0, false);
    loop {
        let (op, rex) = (byte(at), byte(at) & 0xf0 == 0x40);
        let register = if rex { byte(at + 1) } else { op };
        if register & 0xf8 != 0x58 || pops == 64 {
            break;
        }
        pops_rsp |= register & 7 == 4 && !(rex && op & 1 != 0);
        at += if rex { 2 } else { 1 };
        pops += 1;
    }
    // A return or a jump: only then are the instructions before it run.
    let leaves = matches!(byte(at), 0xc3 | 0xc2 | 0xeb | 0xe9)
        || byte(at) == 0xf3 && byte(at + 1) == 0xc3
        || byte(at) == 0xff && byte(at + 1) == 0x25
        || byte(at) & 0xf0 == 0x40 && byte(at + 1) == 0xff && byte(at + 2) & 0x38 == 0x20;
    !leaves || (!pops_rsp && pops < 64 && rsp >= start && rsp + 8 * (pops + 1) <= end)
}

/// The general register numbered `n` as unwind data numbers them.
fn gpr(context: &CONTEXT, n: u8) -> u64 {
    match n & 15 {
        0 => context.Rax,
        1 => context.Rcx,
        2 => context.Rdx,
        3 => context.Rbx,
        4 => context.Rsp,
        5 => context.Rbp,
        6 => context.Rsi,
        7 => context.Rdi,
        8 => context.R8,
        9 => context.R9,
        10 => context.R10,
        11 => context.R11,
        12 => context.R12,
        13 => context.R13,
        14 => context.R14,
        _ => context.R15,
    }
}
