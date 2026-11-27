//! macOS port: map pieces nobody asks for any more are put out when the
//! pool of buildings fills up.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) loads most pieces of map near
//! the player by command: a command forces a piece on (its word in the box
//! streamer's table gets 0x180, +0x23d54e0) and a later one lets go of it
//! (0x80 is cleared, 0x100 stays, +0x3030f0). A piece let go of stays loaded.
//! In the runs here nothing removes it afterwards: when the opening sequence
//! stops, the pieces of every place visited are still loaded, about 4,000 of
//! them with 59,000 entities in that state more than 3 km from the player,
//! and the pool of buildings (22,500 entries) has run out: the `int3` at
//! +0x5149aec (notes/pool-crash.md). What removes such pieces on a console is not
//! found; the guess is the streaming engine making room when its memory is
//! full, which it never is at the speed we stream.
//!
//! `VICTORY_MAPDATA_EVICT=1` does that removal for the title. The call in
//! the box streamer's per-frame thunk (+0x22a2a43) goes to a trampoline that
//! makes the title's call and then calls [`tick`] on the title's thread. At
//! most once a second, and only while the pool of buildings is at least half
//! full, a pass looks for loaded pieces that
//! - have exactly 0x100 in the low 9 bits of their box word (forced on once,
//!   let go of since, no search asking, not switched by script),
//! - lie 1.5 km or more from the player (by their first entity), and
//! - have been like that for ten seconds without a break,
//!
//! and asks the title's own "remove if nobody needs it" (+0x23cac30) to
//! remove up to 64 of them, after clearing the "do not delete" of their
//! streaming record (+0x4f66640) as the title's callers do. That routine
//! still refuses a piece with references or dependents, and it removes
//! through the streaming engine as the title's own removals do.
//!
//! Two more conditions since the first run with the switch (13:15), which
//! stopped on a read of a removed piece's contents (+0x2294954): the
//! streaming engine still had a request for that piece in one of its stage
//! lists, and the map store's "may this request wait" callback reads the
//! contents without a check. A piece is now left alone
//! - if the title's own check before a removal (+0x51c6150, the first thing
//!   its removal of a piece and its parents does) says "not now", or
//! - if a request with its object number waits in one of the engine's 17
//!   stage lists at that moment.
//!
//! What this changes in the title's behaviour: pieces the title would have
//! kept loaded (for a return to the place, or until its own clean-up) are
//! gone ten seconds after they are let go of and 1.5 km away, whenever the
//! pool is half full; a scene that shows such a place from further away than
//! that, with the player elsewhere, can lose detail until the title asks for
//! the pieces again. Pieces without the 0x100 are never touched: with box
//! word 0 are the far-detail map that is always loaded and also what the
//! search loaded at the player's first place and never removed (1,100
//! pieces far away at the stop); the two cannot be told apart here, so that
//! leftover stays. A stopgap, not tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The switch.
pub const SWITCH: &str = "VICTORY_MAPDATA_EVICT";

/// The box streamer's per-frame thunk: `sub rsp, 0x28; lea rcx, <store>;
/// call <box streamer of>; mov rcx, rax; call <update>; mov al, 1;
/// add rsp, 0x28; ret`.
const THUNK: u32 = 0x022a_2a30;
const THUNK_BYTES: [u8; 31] = [
    0x48, 0x83, 0xec, 0x28, 0x48, 0x8d, 0x0d, 0xf5, 0x17, 0xd8, 0x07, 0xe8, 0xa0, 0xa7, 0xb4, 0x02, 0x48, 0x89,
    0xc1, 0xe8, 0x28, 0x29, 0x13, 0x00, 0xb0, 0x01, 0x48, 0x83, 0xc4, 0x28, 0xc3,
];
/// The thunk's `call <update>`, and the update.
const CALL_SITE: u32 = THUNK + 19;
const UPDATE: u32 = 0x023d_5370;
/// The store's "remove if nobody needs it" (store, slot):
/// `push rsi; push rdi; push rbx; sub rsp, 0x30; mov rsi, rcx; mov ecx, [rcx+8]`.
const REMOVE: u32 = 0x023c_ac30;
const REMOVE_BYTES: [u8; 13] = [0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x30, 0x48, 0x89, 0xce, 0x8b, 0x49, 0x08];
/// "Clear these flags of this object's streaming record" (object, flags):
/// `sub rsp, 0x28; and edx, 0xf1`.
const UNREQUIRE: u32 = 0x04f6_6640;
const UNREQUIRE_BYTES: [u8; 10] = [0x48, 0x83, 0xec, 0x28, 0x81, 0xe2, 0xf1, 0x00, 0x00, 0x00];
/// The title's check before it removes a piece and its parents (store,
/// slot): not zero means "not now" (a load of one of them is under way).
/// `push r15; push r14; push r13; push r12; push rsi; push rdi; push rbp;
/// push rbx; sub rsp, 0x28`.
const BUSY: u32 = 0x051c_6150;
const BUSY_BYTES: [u8; 16] =
    [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xec, 0x28];

/// The low 9 bits of a box word that mean "forced on once, let go of since,
/// no search asking, not switched by script".
const LET_GO: u16 = 0x100;
/// A pass runs while the buildings in use are at least this share of the pool.
const FULL_PERCENT: u64 = 50;
/// How far from the player a piece must lie, in metres, and for how long.
const AWAY: f32 = 1500.0;
const AGE: Duration = Duration::from_secs(10);
/// Passes are at least this far apart, and ask for at most this many pieces.
const PASS_EVERY: Duration = Duration::from_secs(1);
const MOST: usize = 64;

/// What the switch's value asks for.
pub fn wanted(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether a loaded piece with this box word at this place is one to put
/// out, seen from the player's place.
fn let_go_and_away(box_word: u16, place: Option<(f32, f32)>, player: (f32, f32)) -> bool {
    box_word & 0x1ff == LET_GO && place.is_some_and(|(x, y)| (x - player.0).hypot(y - player.1) >= AWAY)
}

/// Whether a pool with this many of its entries in use is full enough.
fn full_enough(used: u32, capacity: u32) -> bool {
    capacity != 0 && u64::from(used) * 100 >= u64::from(capacity) * FULL_PERCENT
}

/// Since when each piece has been one to put out, without a break.
#[derive(Default)]
struct Waits {
    since: HashMap<u32, Instant>,
}

impl Waits {
    /// Takes this pass's pieces to put out: those not among them any more
    /// are forgotten, new ones start to wait, and up to `most` that have
    /// waited `age` come back, the longest wait first.
    fn pass(&mut self, pieces: &[u32], now: Instant, age: Duration, most: usize) -> Vec<u32> {
        let seen: std::collections::HashSet<u32> = pieces.iter().copied().collect();
        self.since.retain(|slot, _| seen.contains(slot));
        for &slot in pieces {
            self.since.entry(slot).or_insert(now);
        }
        let mut due: Vec<(Instant, u32)> = self
            .since
            .iter()
            .filter(|&(_, &since)| now.saturating_duration_since(since) >= age)
            .map(|(&slot, &since)| (since, slot))
            .collect();
        due.sort();
        due.into_iter().take(most).map(|(_, slot)| slot).collect()
    }

    /// The piece is gone.
    fn gone(&mut self, slot: u32) {
        self.since.remove(&slot);
    }

    /// The title did not remove the piece: it waits again from `now`, behind
    /// the others.
    fn refused(&mut self, slot: u32, now: Instant) {
        self.since.insert(slot, now);
    }
}

/// The rel32 that reaches `target` from a call or jump whose rel32 ends at
/// `end`; `None` if it is out of reach.
fn rel32(end: u64, target: u64) -> Option<[u8; 4]> {
    Some(i32::try_from((target as i64).checked_sub(end as i64)?).ok()?.to_le_bytes())
}

/// The trampoline placed at `at` for the image at `base`: the thunk's call,
/// then a call of `tick`. `None` if the update is out of rel32 reach.
fn trampoline(base: u64, at: u64, tick: u64) -> Option<Vec<u8>> {
    let mut c = vec![
        0x53, // push rbx (the stack is 16-aligned again)
        0x48, 0x83, 0xec, 0x20, // sub rsp, 0x20: the callees' shadow space
        0xe8, // call <update>, rcx as the thunk set it
    ];
    c.extend(rel32(at + c.len() as u64 + 4, base + u64::from(UPDATE))?);
    c.extend([0x48, 0xb8]); // mov rax, tick
    c.extend(tick.to_le_bytes());
    c.extend([
        0xff, 0xd0, // call rax
        0x48, 0x83, 0xc4, 0x20, // add rsp, 0x20
        0x5b, // pop rbx
        0xc3, // ret
    ]);
    Some(c)
}

#[cfg(windows)]
mod live {
    use super::*;
    use crate::pool_watch::{
        BUILDINGS, ENGINE, FOCUS, POSITION, REQUEST_NEXT, Reader, STAGE_COUNT, STAGES, STORE, place_of,
    };
    use std::sync::Mutex;

    struct Evict {
        base: usize,
        reader: Reader,
        waits: Waits,
        last_pass: Option<Instant>,
        last_log: Instant,
        /// Pieces asked for and gone since the last log line, and in all;
        /// pieces left alone because the title's check said "not now" or a
        /// request for them waited.
        asked: u32,
        gone: u32,
        all_gone: u64,
        busy: u32,
        requested: u32,
    }

    static STATE: Mutex<Option<Evict>> = Mutex::new(None);

    /// Called by the trampoline on the title's thread, once a frame.
    extern "win64" fn tick() {
        // Nothing here may unwind into the title's code.
        let _ = std::panic::catch_unwind(|| {
            if let Ok(mut state) = STATE.try_lock()
                && let Some(evict) = state.as_mut()
            {
                evict.pass();
            }
        });
    }

    impl Evict {
        fn pass(&mut self) {
            let now = Instant::now();
            if self.last_pass.is_some_and(|last| now.saturating_duration_since(last) < PASS_EVERY) {
                return;
            }
            if self.last_pass.is_none() {
                tracing::info!("map pieces put out: the title's thread calls in; nothing is done below half a pool");
            }
            self.last_pass = Some(now);
            self.reader.forget();
            let (base, store) = (self.base, self.base + STORE as usize);
            let reader = &mut self.reader;
            let full = reader.pool(base + BUILDINGS as usize).is_some_and(|p| full_enough(p.used, p.capacity));
            let player = reader
                .pointer(base + FOCUS as usize)
                .and_then(|entity| reader.bytes::<16>(entity + POSITION))
                .as_ref()
                .and_then(place_of);
            let (Some(player), true) = (player, full) else {
                self.waits = Waits::default();
                return;
            };
            let (Some(slots), Some(size), Some(capacity), Some(bits), Some(box_words), Some(first_object)) = (
                reader.pointer(store + 0x38),
                reader.dword(store + 0x40),
                reader.dword(store + 0x44),
                reader.pointer(store + 0x60),
                reader.pointer(store + 0xe0),
                reader.dword(store + 0x8),
            ) else {
                return;
            };
            let (size, capacity) = (size as usize, capacity as usize);
            if slots == 0 || bits == 0 || box_words == 0 || !(8..=0x1000).contains(&size) || capacity > 1 << 22 {
                return;
            }
            let mut pieces = Vec::new();
            for byte in 0..capacity.div_ceil(8) {
                let Some([in_use]) = reader.bytes::<1>(bits + byte).filter(|b| b[0] != 0) else { continue };
                for i in (byte * 8..capacity.min(byte * 8 + 8)).filter(|i| in_use >> (i % 8) & 1 == 1) {
                    let Some(contents) = reader.pointer(slots + i * size).filter(|&c| c != 0) else { continue };
                    let Some(box_word) = reader.bytes::<2>(box_words + 2 * i).map(u16::from_le_bytes) else {
                        continue;
                    };
                    if box_word & 0x1ff != LET_GO {
                        continue;
                    }
                    let (Some(entities), Some(n)) = (reader.pointer(contents + 0x8), reader.dword(contents + 0x10))
                    else {
                        continue;
                    };
                    // Entities the title left out are null: the first that is there.
                    let first = (0..n.min(16) as usize)
                        .find_map(|k| reader.pointer(entities + 8 * k).filter(|&e| e != 0));
                    let place = first.and_then(|e| reader.bytes::<16>(e + POSITION)).as_ref().and_then(place_of);
                    if let_go_and_away(box_word, place, player) {
                        pieces.push(i as u32);
                    }
                }
            }
            // SAFETY: the image is mapped, these are the title's own
            // routines (their first bytes were checked in `patch`), with the
            // arguments the title's callers give them, on the title's thread
            // at the point of the frame where its box streamer is updated.
            let (unrequire, remove, busy) = unsafe {
                (
                    std::mem::transmute::<usize, unsafe extern "win64" fn(u32, u32)>(base + UNREQUIRE as usize),
                    std::mem::transmute::<usize, unsafe extern "win64" fn(usize, u32) -> u32>(base + REMOVE as usize),
                    std::mem::transmute::<usize, unsafe extern "win64" fn(usize, u32) -> u32>(base + BUSY as usize),
                )
            };
            let due = self.waits.pass(&pieces, now, AGE, MOST);
            // The objects the engine has a request for in a stage list now.
            let mut waiting = std::collections::HashSet::new();
            if !due.is_empty()
                && let Some(engine) = self.reader.pointer(base + ENGINE as usize).filter(|&e| e != 0)
            {
                for stage in 0..STAGE_COUNT {
                    let mut request = self.reader.pointer(engine + STAGES + 8 * stage).unwrap_or(0);
                    for _ in 0..1 << 14 {
                        let Some(object) = self.reader.dword(request) else { break };
                        waiting.insert(object & 0x0fff_ffff);
                        request = self.reader.pointer(request + REQUEST_NEXT).unwrap_or(0);
                    }
                }
            }
            for slot in due {
                let object = first_object.wrapping_add(slot);
                if waiting.contains(&(object & 0x0fff_ffff)) {
                    self.requested += 1;
                    self.waits.refused(slot, now);
                    continue;
                }
                // SAFETY: as above; it only reads the streaming records of
                // the piece and its parents.
                if unsafe { busy(store, slot) } & 0xff != 0 {
                    self.busy += 1;
                    self.waits.refused(slot, now);
                    continue;
                }
                // SAFETY: as above; flag 1 is the "do not delete" the
                // title's callers clear before this removal.
                unsafe {
                    unrequire(object, 1);
                    remove(store, slot);
                }
                self.asked += 1;
                if self.reader.pointer(slots + slot as usize * size) == Some(0) {
                    self.gone += 1;
                    self.waits.gone(slot);
                } else {
                    self.waits.refused(slot, now);
                }
            }
            if self.asked + self.busy + self.requested != 0
                && now.saturating_duration_since(self.last_log) >= Duration::from_secs(5)
            {
                self.all_gone += u64::from(self.gone);
                tracing::info!(
                    "map pieces put out: {} of {} asked for in the last seconds, {} in all; left alone {} not now by the title's check, {} with a request waiting; {} let go of and away now",
                    self.gone,
                    self.asked,
                    self.all_gone,
                    self.busy,
                    self.requested,
                    pieces.len()
                );
                (self.asked, self.gone, self.busy, self.requested, self.last_log) = (0, 0, 0, 0, now);
            }
        }
    }

    /// Points the box streamer's per-frame thunk of the image mapped at
    /// `base` at a trampoline that also calls [`tick`]. Before any title
    /// thread runs. Says why if it does not.
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
            std::slice::from_raw_parts(at(THUNK), THUNK_BYTES.len()) == THUNK_BYTES
                && std::slice::from_raw_parts(at(REMOVE), REMOVE_BYTES.len()) == REMOVE_BYTES
                && std::slice::from_raw_parts(at(UNREQUIRE), UNREQUIRE_BYTES.len()) == UNREQUIRE_BYTES
                && std::slice::from_raw_parts(at(BUSY), BUSY_BYTES.len()) == BUSY_BYTES
        };
        if !known {
            return Err("not the build it was written for (bytes differ)");
        }
        // One page of code within reach of a rel32 call.
        const PAGE: usize = 0x1000;
        let mut page = std::ptr::null_mut();
        for step in 1..2048u64 {
            let Some(want) = base.checked_sub(step << 20) else { break };
            // SAFETY: a plain reservation at a hint; failure returns null.
            page = unsafe { VirtualAlloc(want as *const _, PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            if !page.is_null() {
                break;
            }
        }
        if page.is_null() {
            return Err("no memory near the image for the trampoline");
        }
        let site = base + u64::from(CALL_SITE);
        let (Some(code), Some(call)) =
            (trampoline(base, page as u64, tick as *const () as u64), rel32(site + 5, page as u64))
        else {
            return Err("the trampoline is out of reach");
        };
        *STATE.lock().map_err(|_| "its state is not usable")? = Some(Evict {
            base: base as usize,
            reader: Reader::default(),
            waits: Waits::default(),
            last_pass: None,
            last_log: Instant::now(),
            asked: 0,
            gone: 0,
            all_gone: 0,
            busy: 0,
            requested: 0,
        });
        let mut old = 0;
        // SAFETY: our page holds the code; the site is the rel32 of the
        // title's call, made writable for the write, before any title
        // thread runs.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
            VirtualProtect(page, PAGE, PAGE_EXECUTE_READ, &mut old);
            if VirtualProtect((site + 1) as *const _, call.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return Err("the title's code could not be made writable");
            }
            std::ptr::copy_nonoverlapping(call.as_ptr(), (site + 1) as *mut u8, call.len());
            VirtualProtect((site + 1) as *const _, call.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
        }
        Ok(())
    }
}

#[cfg(windows)]
pub use live::patch;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_turns_it_on() {
        assert!(wanted(Some("1")));
        assert!(!wanted(Some("0")));
        assert!(!wanted(None));
    }

    /// The thunk's second call is the one replaced, and it goes to the update.
    #[test]
    fn the_site_is_the_thunks_call_of_the_update() {
        let at = (CALL_SITE - THUNK) as usize;
        assert_eq!(THUNK_BYTES[at], 0xe8);
        let delta = i32::from_le_bytes(THUNK_BYTES[at + 1..at + 5].try_into().unwrap());
        assert_eq!((i64::from(CALL_SITE) + 5 + i64::from(delta)) as u32, UPDATE);
        // What follows the call is the thunk's own return of "true".
        assert_eq!(&THUNK_BYTES[at + 5..], &[0xb0, 0x01, 0x48, 0x83, 0xc4, 0x28, 0xc3]);
    }

    #[test]
    fn the_trampoline_makes_the_thunks_call_then_ours() {
        let (base, at, tick) = (0x6fff_ed55_0000u64, 0x6fff_ed35_0000u64, 0x1_4012_3456u64);
        let code = trampoline(base, at, tick).unwrap();
        assert_eq!(code.len(), 28);
        assert_eq!(&code[..6], &[0x53, 0x48, 0x83, 0xec, 0x20, 0xe8]);
        let delta = i32::from_le_bytes(code[6..10].try_into().unwrap());
        assert_eq!((at + 10).wrapping_add(delta as i64 as u64), base + u64::from(UPDATE));
        assert_eq!(&code[10..12], &[0x48, 0xb8]);
        assert_eq!(u64::from_le_bytes(code[12..20].try_into().unwrap()), tick);
        assert_eq!(&code[20..], &[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20, 0x5b, 0xc3]);
        // One push and 0x20 of shadow space: 16-aligned at both calls.
        assert_eq!((8 + 8 + 0x20) % 16, 0);
        assert!(trampoline(base, base - (3 << 30), tick).is_none(), "out of reach is refused");
        assert_eq!(rel32(0x1005, 0x2000), Some(0xffbi32.to_le_bytes()));
    }

    #[test]
    fn only_pieces_let_go_of_and_away_are_put_out() {
        let player = (0.0, 0.0);
        let away = Some((0.0, 1500.0));
        assert!(let_go_and_away(0x100, away, player));
        assert!(let_go_and_away(0x4100, away, player), "the bits above the low nine are the kind of piece");
        assert!(!let_go_and_away(0x100, Some((1000.0, 1000.0)), player), "1.4 km is near");
        assert!(!let_go_and_away(0x100, None, player), "a piece without a place stays");
        for kept in [0x000, 0x101, 0x180, 0x181, 0x1c1, 0x140, 0x080, 0x001] {
            assert!(!let_go_and_away(kept, away, player), "{kept:#x}");
        }
    }

    #[test]
    fn a_pass_runs_from_half_a_pool() {
        assert!(!full_enough(11_249, 22_500));
        assert!(full_enough(11_250, 22_500));
        assert!(full_enough(22_500, 22_500));
        assert!(!full_enough(0, 0));
    }

    #[test]
    fn a_piece_goes_after_ten_seconds_without_a_break() {
        let start = Instant::now();
        let at = |s: u64| start + Duration::from_secs(s);
        let mut waits = Waits::default();
        assert!(waits.pass(&[1, 2], at(0), AGE, MOST).is_empty());
        assert!(waits.pass(&[1, 2, 3], at(9), AGE, MOST).is_empty());
        // 2 was asked for again in between: its wait starts over.
        assert!(waits.pass(&[1, 3], at(9), AGE, MOST).is_empty());
        assert_eq!(waits.pass(&[1, 2, 3], at(10), AGE, MOST), [1]);
        waits.gone(1);
        assert_eq!(waits.pass(&[2, 3], at(19), AGE, MOST), [3]);
        // The title refused 3: it waits again; 2 has waited since second 10.
        waits.refused(3, at(19));
        assert_eq!(waits.pass(&[2, 3], at(20), AGE, MOST), [2]);
        assert_eq!(waits.pass(&[2, 3], at(29), AGE, 1), [2], "the longest wait first");
        assert_eq!(waits.pass(&[2, 3], at(29), AGE, MOST), [2, 3]);
        assert!(waits.pass(&[], at(40), AGE, MOST).is_empty());
        assert!(waits.since.is_empty());
    }
}
