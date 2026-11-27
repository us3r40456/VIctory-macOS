//! macOS port: the time a frame the title's streaming request list may
//! spend issuing its recorded requests, made larger.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) plays a cutscene's recorded
//! streaming requests from a request list (state `Prestream` in the title's
//! crash log). The list's update (+0x227c000, once a frame for each list in
//! the table at +0xab442c0) issues an entry's steps in a loop that goes on
//! while steps are left and either under 500 microseconds were spent in
//! this update or the entry is not time-sliced. An entry is time-sliced
//! while its byte +0x7f is set and the title's check +0x4f862f0 says no;
//! the update writes that back to +0x7f. The 500.0 is loaded once into
//! xmm6 (+0x227c081) and compared at +0x227c28e, +0x227c4da and, for
//! entries whose byte +0x82 is 1, +0x227c5d8.
//!
//! The 500 microseconds are a fixed value, not a share of the frame: at two
//! frames a second a time-sliced list gets 1 ms a second, against 15 ms a
//! second on a console (notes/streaming-budgets.md). Whether any entry is
//! time-sliced in the opening sequence is not known; the watch's
//! `request lists` line is there to show it.
//!
//! `VICTORY_SRL_SLICE=<k>` (2 to 200) makes that one load read a larger
//! constant. The 500.0 itself is shared by hundreds of instructions and is
//! left alone: the load's displacement is pointed at one of four floats the
//! title's read-only data already holds, 5,000, 10,000, 20,000 or 50,000,
//! whichever is nearest to 500 times `k`. What it changes in the title: a
//! time-sliced list may issue requests for up to that long in a frame, so
//! such frames get longer by up to the larger slice and more requests are
//! in flight at once. Nothing else is touched. Not tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch.
pub const SWITCH: &str = "VICTORY_SRL_SLICE";

/// The load: `vmovss xmm6, [rip + 0x3ede24b]`, which is the float at
/// [`STOCK`].
const SITE: u32 = 0x0227_c081;
const SITE_BYTES: [u8; 8] = [0xc5, 0xfa, 0x10, 0x35, 0x4b, 0xe2, 0xed, 0x03];
/// Where the displacement is in the site; it counts from the site's end.
const DISP_AT: usize = 4;
/// The constant the title loads, and its value in microseconds.
const STOCK: (u32, f32) = (0x0615_a2d4, 500.0);
/// Larger floats in the title's read-only data, and the values they must
/// hold.
const LARGER: [(u32, f32); 4] =
    [(0x0614_f628, 5_000.0), (0x0615_9c54, 10_000.0), (0x0615_b8e4, 20_000.0), (0x0615_a414, 50_000.0)];

/// The factor the switch's value asks for; `None` leaves the title as it is
/// (unset, 0, 1 or not a number from 2 to 200).
pub fn wanted(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok().filter(|k| (2..=200).contains(k))
}

/// The larger constant nearest to 500 times `k`; of two equally near, the
/// larger.
fn nearest(k: u32) -> (u32, f32) {
    let want = STOCK.1 * k as f32;
    LARGER.into_iter().rev().min_by(|a, b| (a.1 - want).abs().total_cmp(&(b.1 - want).abs())).unwrap()
}

/// The slice in microseconds that `k` gives.
pub fn slice_for(k: u32) -> u32 {
    nearest(k).1 as u32
}

/// The site's bytes with the load pointed at `target`, given the bytes now
/// at the site, at the stock constant and at the target. Says why if any
/// of them is not what this was written for.
fn retargeted(site: &[u8; 8], stock: [u8; 4], target: (u32, f32), at_target: [u8; 4]) -> Result<[u8; 8], &'static str> {
    if *site != SITE_BYTES {
        return Err("not the build it was written for (the load's bytes differ)");
    }
    if stock != STOCK.1.to_le_bytes() {
        return Err("not the build it was written for (the load's constant is not 500.0)");
    }
    if at_target != target.1.to_le_bytes() {
        return Err("not the build it was written for (the larger constant is not where it was read)");
    }
    let mut bytes = *site;
    let disp = target.0.wrapping_sub(SITE + SITE_BYTES.len() as u32);
    bytes[DISP_AT..].copy_from_slice(&disp.to_le_bytes());
    Ok(bytes)
}

/// Points the load of the image mapped at `base` at the constant for `k`.
/// Before any title thread runs. Gives the new slice in microseconds, or
/// says why nothing was changed.
#[cfg(windows)]
pub fn patch(base: u64, k: u32) -> Result<u32, &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let target = nearest(k);
    let site = (base + u64::from(SITE)) as *mut u8;
    let float_at = |rva: u32| (base + u64::from(rva)) as *const [u8; 4];
    // SAFETY: the image is mapped; the site is inside its code and the two
    // constants are inside its read-only data.
    let bytes = unsafe { retargeted(&*(site as *const [u8; 8]), *float_at(STOCK.0), target, *float_at(target.0))? };
    let mut old = 0;
    // SAFETY: the site is the title's code, made writable for the write,
    // before any title thread runs; only the displacement changes.
    unsafe {
        if VirtualProtect(site as *const _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("the title's code could not be made writable");
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), site, bytes.len());
        VirtualProtect(site as *const _, bytes.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    Ok(target.1 as u32)
}

// The watch's `request lists` line (`VICTORY_MAPDATA_WATCH=1`,
// crate::pool_watch).

/// The table of request lists: their number, then up to six pointers.
#[cfg(windows)]
const LIST_COUNT: u32 = 0x0ab4_42cc;
#[cfg(windows)]
const LISTS: u32 = 0x0ab4_42d0;
#[cfg(windows)]
const MAX_LISTS: usize = 6;
/// The entry the map budget's governor reads (pointer; crate::map_budget).
#[cfg(windows)]
const GOVERNED: u32 = 0x0ab4_4328;
/// A list: its flags (bit 7: an entry was time-sliced in the last update),
/// its recording (pointer), its state, its entries (pointer, 0x84 bytes
/// each), and the number and the indices (pointer to dwords) of those the
/// update goes through.
const LIST: usize = 0x50;
const LIST_RECORDING: usize = 0x20;
const LIST_STATE: usize = 0x28;
const LIST_ENTRIES: usize = 0x38;
const LIST_ACTIVE: usize = 0x44;
const LIST_INDICES: usize = 0x48;
const SLICED_LAST_UPDATE: u32 = 0x80;
/// A recording: its tracks (pointer, 0x20 bytes each; a track's number of
/// steps is the word at +0x8).
#[cfg(windows)]
const RECORDING_TRACKS: usize = 0x10;
#[cfg(windows)]
const TRACK: usize = 0x20;
#[cfg(windows)]
const TRACK_STEPS: usize = 0x8;
/// An entry: its state, the next step to issue, the step asked for, the
/// look-ahead in ms, whether it is time-sliced, and the byte that puts its
/// catching up under the slice too.
const ENTRY: usize = 0x84;
const ENTRY_STATE: usize = 0xc;
const ENTRY_NEXT: usize = 0x5c;
const ENTRY_ASKED: usize = 0x60;
const ENTRY_AHEAD: usize = 0x6c;
const ENTRY_SLICED: usize = 0x7f;
const ENTRY_CATCHING_UP: usize = 0x82;
/// Entries named in a list.
#[cfg(windows)]
const NAMED: usize = 6;

/// One entry of a request list as the line shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    index: u32,
    state: u8,
    next: u32,
    asked: u32,
    /// The steps of the entry's track, if the recording could be read.
    steps: Option<u16>,
    ahead_ms: u32,
    sliced: u8,
    catching_up: u8,
}

#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn entry_from(index: u32, bytes: &[u8; ENTRY], steps: Option<u16>) -> Entry {
    let dword = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    Entry {
        index,
        state: bytes[ENTRY_STATE],
        next: dword(ENTRY_NEXT),
        asked: dword(ENTRY_ASKED),
        steps,
        ahead_ms: dword(ENTRY_AHEAD),
        sliced: bytes[ENTRY_SLICED],
        catching_up: bytes[ENTRY_CATCHING_UP],
    }
}

/// One request list as the line shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct List {
    flags: u32,
    state: u8,
    /// The entries its update goes through; `entries` are those read.
    active: u32,
    entries: Vec<Entry>,
}

/// A list's fixed part: flags, state, recording, entries, active, indices.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn list_from(bytes: &[u8; LIST]) -> (u32, u8, usize, usize, u32, usize) {
    let pointer = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    let dword = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    (dword(0), bytes[LIST_STATE], pointer(LIST_RECORDING), pointer(LIST_ENTRIES), dword(LIST_ACTIVE), pointer(LIST_INDICES))
}

/// What the watch could read: the lists, and the time-sliced byte of the
/// entry the governor reads (if there is one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seen {
    lists: Option<Vec<List>>,
    governed: Option<u8>,
}

/// The `request lists` line.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) fn line(seen: &Seen) -> String {
    let governed = match seen.governed {
        Some(sliced) => format!("the governor's entry has time-sliced {sliced}"),
        None => "no governor's entry".to_owned(),
    };
    let Some(lists) = &seen.lists else {
        return format!("request lists: unknown; {governed}");
    };
    let sliced = lists.iter().flat_map(|l| &l.entries).filter(|e| e.sliced != 0).count();
    let mut text = format!(
        "request lists: {} with {} entries active, {sliced} time-sliced; {governed}",
        lists.len(),
        lists.iter().map(|l| l.active).sum::<u32>()
    );
    for (n, list) in lists.iter().enumerate() {
        let sliced_before = if list.flags & SLICED_LAST_UPDATE != 0 { ", sliced in its last update" } else { "" };
        text += &format!("; list {n} state {} flags {:#x}{sliced_before}:", list.state, list.flags);
        if list.active == 0 {
            text += " idle";
        }
        for e in &list.entries {
            let steps = e.steps.map_or("?".to_owned(), |s| s.to_string());
            text += &format!(
                " e{} s{} step {}/{steps} asked {} ahead {} ms sliced {} catching-up {},",
                e.index, e.state, e.next, e.asked, e.ahead_ms, e.sliced, e.catching_up
            );
        }
        if text.ends_with(',') {
            text.pop();
        }
        if list.active as usize > list.entries.len() {
            text += &format!(" (+{} not shown)", list.active as usize - list.entries.len());
        }
    }
    text
}

/// Reads the request lists of the image mapped at `base`. Without the
/// title's lock: a list that changes under us gives a line that is off for
/// one sample.
#[cfg(windows)]
pub(crate) fn seen(reader: &mut crate::pool_watch::Reader, base: usize) -> Seen {
    let governed = reader
        .pointer(base + GOVERNED as usize)
        .filter(|&entry| entry != 0)
        .and_then(|entry| reader.bytes::<1>(entry + ENTRY_SLICED))
        .map(|b| b[0]);
    let Some(count) = reader.dword(base + LIST_COUNT as usize).filter(|&n| n as usize <= MAX_LISTS) else {
        return Seen { lists: None, governed };
    };
    let mut lists = Vec::new();
    for n in 0..count as usize {
        let Some(bytes) = reader.pointer(base + LISTS as usize + 8 * n).and_then(|list| reader.bytes::<LIST>(list))
        else {
            return Seen { lists: None, governed };
        };
        let (flags, state, recording, entries, active, indices) = list_from(&bytes);
        let tracks = if recording == 0 { None } else { reader.pointer(recording + RECORDING_TRACKS) };
        let mut read = Vec::new();
        for i in 0..(active as usize).min(NAMED) {
            let Some(index) = reader.dword(indices.wrapping_add(4 * i)).filter(|&index| index < 1 << 16) else { break };
            let Some(entry) = reader.bytes::<ENTRY>(entries.wrapping_add(ENTRY * index as usize)) else { break };
            let steps = tracks
                .and_then(|t| reader.bytes::<2>(t.wrapping_add(TRACK * index as usize + TRACK_STEPS)))
                .map(u16::from_le_bytes);
            read.push(entry_from(index, &entry, steps));
        }
        lists.push(List { flags, state, active, entries: read });
    }
    Seen { lists: Some(lists), governed }
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

    /// The site's displacement leads to the constant these offsets name.
    #[test]
    fn the_load_reads_the_stock_constant() {
        assert_eq!(SITE_BYTES[..DISP_AT], [0xc5, 0xfa, 0x10, 0x35], "vmovss xmm6, [rip + disp32]");
        let disp = u32::from_le_bytes(SITE_BYTES[DISP_AT..].try_into().unwrap());
        assert_eq!(SITE + 8 + disp, STOCK.0);
    }

    #[test]
    fn the_nearest_larger_constant_is_taken() {
        for (k, us) in [(2, 5_000), (10, 5_000), (14, 5_000), (15, 10_000), (20, 10_000), (29, 10_000)] {
            assert_eq!(slice_for(k), us, "{k}");
        }
        for (k, us) in [(30, 20_000), (40, 20_000), (69, 20_000), (70, 50_000), (100, 50_000), (200, 50_000)] {
            assert_eq!(slice_for(k), us, "{k}");
        }
    }

    #[test]
    fn only_the_displacement_changes() {
        for target in LARGER {
            let bytes = retargeted(&SITE_BYTES, 500f32.to_le_bytes(), target, target.1.to_le_bytes()).unwrap();
            assert_eq!(bytes[..DISP_AT], SITE_BYTES[..DISP_AT]);
            let disp = u32::from_le_bytes(bytes[DISP_AT..].try_into().unwrap());
            assert_eq!(SITE + 8 + disp, target.0);
        }
        let x40 = retargeted(&SITE_BYTES, 500f32.to_le_bytes(), LARGER[2], 20_000f32.to_le_bytes()).unwrap();
        assert_eq!(x40, [0xc5, 0xfa, 0x10, 0x35, 0x5b, 0xf8, 0xed, 0x03]);
    }

    #[test]
    fn anything_unexpected_is_refused() {
        let (stock, target) = (500f32.to_le_bytes(), LARGER[2]);
        let good = target.1.to_le_bytes();
        let mut other = SITE_BYTES;
        other[3] = 0x3d; // another register
        assert!(retargeted(&other, stock, target, good).unwrap_err().contains("load's bytes"));
        let mut moved = SITE_BYTES;
        moved[4] ^= 4; // another constant
        assert!(retargeted(&moved, stock, target, good).unwrap_err().contains("load's bytes"));
        assert!(retargeted(&SITE_BYTES, 250f32.to_le_bytes(), target, good).unwrap_err().contains("not 500.0"));
        assert!(retargeted(&SITE_BYTES, stock, target, 20_001f32.to_le_bytes()).unwrap_err().contains("larger constant"));
        assert!(retargeted(&SITE_BYTES, stock, target, [0; 4]).is_err());
    }

    fn entry(next: u32, asked: u32, sliced: u8) -> [u8; ENTRY] {
        let mut bytes = [0u8; ENTRY];
        bytes[ENTRY_STATE] = 2;
        bytes[ENTRY_NEXT..ENTRY_NEXT + 4].copy_from_slice(&next.to_le_bytes());
        bytes[ENTRY_ASKED..ENTRY_ASKED + 4].copy_from_slice(&asked.to_le_bytes());
        bytes[ENTRY_AHEAD..ENTRY_AHEAD + 4].copy_from_slice(&2000u32.to_le_bytes());
        bytes[ENTRY_SLICED] = sliced;
        bytes
    }

    #[test]
    fn an_entry_gives_its_numbers() {
        let mut bytes = entry(120, 130, 1);
        bytes[ENTRY_CATCHING_UP] = 1;
        let e = entry_from(3, &bytes, Some(400));
        assert_eq!((e.index, e.state, e.next, e.asked, e.steps), (3, 2, 120, 130, Some(400)));
        assert_eq!((e.ahead_ms, e.sliced, e.catching_up), (2000, 1, 1));
    }

    #[test]
    fn a_list_gives_its_fixed_part() {
        let mut bytes = [0u8; LIST];
        bytes[..4].copy_from_slice(&0xa2u32.to_le_bytes());
        bytes[LIST_STATE] = 2;
        bytes[LIST_RECORDING..LIST_RECORDING + 8].copy_from_slice(&0x1000u64.to_le_bytes());
        bytes[LIST_ENTRIES..LIST_ENTRIES + 8].copy_from_slice(&0x2000u64.to_le_bytes());
        bytes[LIST_ACTIVE..LIST_ACTIVE + 4].copy_from_slice(&7u32.to_le_bytes());
        bytes[LIST_INDICES..LIST_INDICES + 8].copy_from_slice(&0x3000u64.to_le_bytes());
        assert_eq!(list_from(&bytes), (0xa2, 2, 0x1000, 0x2000, 7, 0x3000));
    }

    #[test]
    fn the_line_says_whether_the_slice_applies() {
        let list = List {
            flags: 0xa2,
            state: 2,
            active: 3,
            entries: vec![entry_from(3, &entry(120, 130, 1), Some(400)), entry_from(0, &entry(7, 7, 0), None)],
        };
        assert_eq!(
            line(&Seen { lists: Some(vec![list]), governed: Some(0) }),
            "request lists: 1 with 3 entries active, 1 time-sliced; the governor's entry has time-sliced 0; \
             list 0 state 2 flags 0xa2, sliced in its last update: \
             e3 s2 step 120/400 asked 130 ahead 2000 ms sliced 1 catching-up 0, \
             e0 s2 step 7/? asked 7 ahead 2000 ms sliced 0 catching-up 0 (+1 not shown)"
        );
        let idle = List { flags: 0x20, state: 0, active: 0, entries: Vec::new() };
        assert_eq!(
            line(&Seen { lists: Some(vec![idle]), governed: None }),
            "request lists: 1 with 0 entries active, 0 time-sliced; no governor's entry; list 0 state 0 flags 0x20: idle"
        );
        assert_eq!(line(&Seen::default()), "request lists: unknown; no governor's entry");
        assert_eq!(
            line(&Seen { lists: Some(Vec::new()), governed: None }),
            "request lists: 0 with 0 entries active, 0 time-sliced; no governor's entry"
        );
    }
}
