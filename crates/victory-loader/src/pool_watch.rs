//! macOS port: the title's pools of buildings and objects, watched.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) stops on its own `int3` at
//! +0x5149aec about five minutes into the opening sequence: its pool of
//! buildings (22,500 entries) has no free entry when the entities of a piece
//! of map are made (notes/pool-crash.md).
//!
//! `VICTORY_MAPDATA_WATCH=1` logs every five seconds how full the building
//! and object pools are, how many entries of each were ever freed, and how
//! many interiors wait in the list of the interiors' per-frame step. It
//! reads the title's memory and changes nothing.
//!
//! More lines say where and for whom:
//! - `buildings by place`: the buildings in use counted by 500 m square of
//!   the map (an entity's position is at +0x70: x, y and z as floats, and in
//!   the fourth dword two signed words that count 32 m in x and in y, as the
//!   title's own conversion at +0x4dc3790 reads them). Squares are named by
//!   their centre in km. This tells buildings of places the camera has left
//!   from too many at the place it is at.
//! - `interiors`: the interiors by state (0 none to 4 switched on) and, for
//!   those wanted with their entities, by the module that wants them most
//!   (14 bytes at +0xb2, one a module: 1 off, 4 with entities, 0x10 all).
//! - `focus`, once a second: the place of the entity the interiors measure
//!   their distance from (pointer at +0xab56e38, read in 1,788 places of
//!   the code: most likely the player, not checked) with the buildings in
//!   use, to follow the cuts of a sequence between the five-second lines.
//! - `map pieces`: the pieces of map the title's store (+0xa024230) has
//!   loaded, with their entities, by the same squares (capacity at +0x44;
//!   the run of 12:34 showed 187,000 slots of 48 bytes, which fits).
//! - `map pieces near`, `mid`, `far` (within 1 km of the focus, within 3 km,
//!   further) and `nowhere`: the loaded pieces by what the title's own
//!   "remove if nobody needs it" (+0x23cac30) looks at. A group is written
//!   `b<box> s<state> d<dependents> pieces/entities`:
//!   - box: the piece's word in the box streamer's table (pointer at store
//!     +0xe0), low 9 bits. Any of 0x79 set means a search still asks for the
//!     piece; with 0x80 set the piece is switched by script and 0x100 says on.
//!   - state, dependents: from the piece's streaming record (the engine
//!     at +0x764b240 keeps 28-byte records in 503 buckets): bits 5 to 8 of
//!     the word at +0x1a, and the word at +0x14. The title's own check
//!     before it removes a piece (+0x51c6150) takes state 5 as "can go
//!     now" and 0 to 4 as "not yet". `s-` if there is no record. (The flags
//!     byte at +0x19 and the reference words read 0 for every piece in the
//!     run of 12:56 and are no longer shown.)
//!
//!   Pieces far from the focus that still have a search bit are held by
//!   whoever sets that bit; with no bit they are free to go and something
//!   does not remove them.
//! - `streaming`: the engine's time budget for placing map pieces in one
//!   frame (dword at engine +0x166c8, microseconds; 0 is no limit; the
//!   title's governor at +0x2294750 sets it to its estimate of the work
//!   divided by the frame time in 33 ms units, kept between 250 and 1250),
//!   the governor's mode word (+0xab44324), and how many requests wait in
//!   each of the engine's 17 stage lists (heads at engine +0x167b0, a
//!   request's next at +0x80), as `stage:count`.
//! - `request lists`: the cutscene's streaming request lists
//!   (crate::srl_slice): how many, their entries in the update, and how many
//!   of those are time-sliced (entry byte +0x7f), which is what makes the
//!   500 microseconds a frame apply; then per list its state and flags and
//!   per entry its state, next step of its track's steps, the step asked
//!   for, the look-ahead, and the two bytes that put it under the slice.
//!
//! That step (RVA 0x16cd4d0) was first read as the step of the whole map
//! and paced by `VICTORY_MAPDATA_STEPS`; it is the state machine of the
//! interiors only (its pools are made under the names "InteriorProxy" and
//! "InteriorInst"), a run with the pacing stopped at the same place, and the
//! pacing is gone again.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch for the log line.
pub const WATCH_SWITCH: &str = "VICTORY_MAPDATA_WATCH";

/// The interiors' per-frame step, `push r15; push r14; push r13; push r12`,
/// here to tell the build.
const STEP: u32 = 0x016c_d4d0;
const STEP_BYTES: [u8; 8] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54];
/// Its one per-frame caller: `sub rsp, 0x28; xor ecx, ecx; mov dl, 1;
/// call <step>; mov al, 1; add rsp, 0x28; ret`.
const THUNK: u32 = 0x022a_23a0;
const THUNK_BYTES: [u8; 20] = [
    0x48, 0x83, 0xec, 0x28, 0x31, 0xc9, 0xb2, 0x01, 0xe8, 0x23, 0xb1, 0x42, 0xff, 0xb0, 0x01, 0x48, 0x83, 0xc4,
    0x28, 0xc3,
];
/// Head of the list of interiors with a change of state waiting: nodes of
/// `{ interior, next }`, new ones put at the head.
const WAITING: u32 = 0x076d_5098;
/// Pointers to the pools of buildings and of objects. A pool's header has
/// its storage at +0x0, its entry size at +0x8, its capacity at +0xc, the
/// entries in use at +0x10, in the upper half of the free list's head at
/// +0x20 a count of the entries ever freed, and at +0x28 a dword per entry
/// whose low 17 bits are all set while the entry is in use.
pub(crate) const BUILDINGS: u32 = 0x0a9e_c938;
const OBJECTS: u32 = 0x0abb_e2e8;
/// The pool of interiors (0x150 bytes each): the 14 wanted levels at +0xb2,
/// the state at +0xc1.
const INTERIORS: u32 = 0x09e5_76c8;
const INTERIOR_WANTED: usize = 0xb2;
const MODULES: usize = 14;
/// The store of map pieces: its slots at +0x38, a slot's size at +0x40, the
/// number of slots at +0x44 (a guess), a bit per slot in use at +0x60. A
/// slot starts with the pointer to its loaded contents, which has its
/// entities' pointers at +0x8 and their number at +0x10.
pub(crate) const STORE: u32 = 0x0a02_4230;
/// Pointer to the entity the interiors' distance check measures from.
pub(crate) const FOCUS: u32 = 0x0ab5_6e38;
/// The streaming engine (pointer): 503 buckets of 0xa0 bytes; a bucket's
/// records (28 bytes, the object's number in the low 28 bits of the first
/// dword) are counted at +0x24 and start at the pointer at +0x28, both
/// shifted by 16 bytes times the dword at the engine's +0x14.
pub(crate) const ENGINE: u32 = 0x0764_b240;
const BUCKETS: usize = 503;
const RECORD: usize = 0x1c;
/// In the engine: the time budget of a frame for placing map pieces, the
/// heads of its stage lists, their number, and a request's next pointer.
const BUDGET: usize = 0x166c8;
pub(crate) const STAGES: usize = 0x167b0;
pub(crate) const STAGE_COUNT: usize = 17;
pub(crate) const REQUEST_NEXT: usize = 0x80;
/// The mode word of the governor that sets the budget.
const GOVERNOR_MODE: u32 = 0x0ab4_4324;
/// A piece is near the focus within this, far beyond the second.
const NEAR: f32 = 1000.0;
const FAR: f32 = 3000.0;
/// Groups named in a class of pieces.
const GROUPS: usize = 6;
/// An entity's position, and the side of the squares the title counts in
/// the position's fourth dword.
pub(crate) const POSITION: usize = 0x70;
const SECTOR: f32 = 32.0;
/// The side of a square of the log's lines, and the map's reach: a
/// position further out than this is not one.
const SQUARE: f32 = 500.0;
const REACH: f32 = 16_000.0;
/// Squares named in a line.
const NAMED: usize = 8;
/// A walk of the waiting list stops here: the list is read without the
/// title's lock, so a node freed under us can lead anywhere.
const WALK_LIMIT: u32 = 1 << 16;

/// What the switch's value asks for.
pub fn watch_wanted(value: Option<&str>) -> bool {
    value == Some("1")
}

/// One pool's numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    pub used: u32,
    pub capacity: u32,
    /// Entries ever returned to the pool.
    pub freed: u32,
}

/// A pool's numbers from the 0x28 bytes at the start of its header.
fn pool_from(header: &[u8; 0x28]) -> Pool {
    let dword = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    Pool { capacity: dword(0xc), used: dword(0x10), freed: dword(0x24) }
}

/// The x and y of the map, in metres, of an entity's 16 position bytes;
/// `None` if they are not a place on the map.
pub(crate) fn place_of(position: &[u8; 16]) -> Option<(f32, f32)> {
    let float = |at: usize| f32::from_le_bytes(position[at..at + 4].try_into().unwrap());
    let word = |at: usize| f32::from(i16::from_le_bytes(position[at..at + 2].try_into().unwrap()));
    let (x, y) = (float(0) + SECTOR * word(12), float(4) + SECTOR * word(14));
    (x.abs() <= REACH && y.abs() <= REACH && float(8).abs() <= REACH).then_some((x, y))
}

/// The once-a-second line.
fn focus_line(focus: Option<(f32, f32)>, buildings: Option<Pool>) -> String {
    let focus = match focus {
        Some((x, y)) => format!("{:.2},{:.2} km", x / 1000.0, y / 1000.0),
        None => "none".to_owned(),
    };
    match buildings {
        Some(p) => format!("focus {focus}; buildings {} ({} freed so far)", p.used, p.freed),
        None => format!("focus {focus}"),
    }
}

/// Counts by 500 m square.
#[derive(Default)]
struct Places {
    /// Per square: entries, and whatever they carry (a piece's entities).
    squares: std::collections::HashMap<(i32, i32), (u32, u32)>,
    /// Entries with no place on the map.
    nowhere: u32,
}

impl Places {
    fn add(&mut self, place: Option<(f32, f32)>, carried: u32) {
        match place {
            Some((x, y)) => {
                let square = ((x / SQUARE).floor() as i32, (y / SQUARE).floor() as i32);
                let at = self.squares.entry(square).or_default();
                *at = (at.0 + 1, at.1.saturating_add(carried));
            }
            None => self.nowhere += 1,
        }
    }

    /// The fullest squares by their centre in km, "x,y n" (or "x,y n/carried"),
    /// then what is left.
    fn text(&self, carried: bool) -> String {
        let mut squares: Vec<_> = self.squares.iter().collect();
        squares.sort_by_key(|&(&square, &(n, c))| (std::cmp::Reverse(if carried { c } else { n }), square));
        let km = |i: i32| (i as f32 + 0.5) * SQUARE / 1000.0;
        let mut out: Vec<String> = squares
            .iter()
            .take(NAMED)
            .map(|&(&(x, y), &(n, c))| {
                if carried { format!("{:.2},{:.2} {n}/{c}", km(x), km(y)) } else { format!("{:.2},{:.2} {n}", km(x), km(y)) }
            })
            .collect();
        if squares.len() > NAMED {
            let rest = squares[NAMED..].iter().fold((0u32, 0u32), |a, &(_, &(n, c))| (a.0 + n, a.1.saturating_add(c)));
            let rest = if carried { format!("{}/{}", rest.0, rest.1) } else { rest.0.to_string() };
            out.push(format!("{} more squares {rest}", squares.len() - NAMED));
        }
        if self.nowhere != 0 {
            out.push(format!("nowhere {}", self.nowhere));
        }
        if out.is_empty() { "none".to_owned() } else { out.join("; ") }
    }
}

/// What the title's removal looks at for one piece: the box streamer's
/// word (low 9 bits), the streaming record's state (`None` without a
/// record), its dependents.
type Need = (u16, Option<u8>, u16);

/// The state in a streaming record: bits 5 to 8 of the word at +0x1a.
fn state_of(record: &[u8; RECORD]) -> u8 {
    (u16::from_le_bytes([record[0x1a], record[0x1b]]) >> 5 & 0xf) as u8
}

/// Per stage list of the engine: the stage, the requests waiting in it, and
/// whether the walk was cut short.
type Waiting = Vec<(usize, u32, bool)>;

/// The `streaming` line.
fn streaming_line(budget: Option<u32>, mode: Option<u32>, waiting: &[(usize, u32, bool)]) -> String {
    let budget = match budget {
        Some(0) => "no limit".to_owned(),
        Some(us) => format!("{us} us a frame"),
        None => "unknown".to_owned(),
    };
    let mode = mode.map_or("unknown".to_owned(), |m| format!("{m:#x}"));
    let waiting: Vec<String> = waiting
        .iter()
        .filter(|&&(_, n, _)| n != 0)
        .map(|&(stage, n, cut)| format!("{stage}:{n}{}", if cut { "+" } else { "" }))
        .collect();
    format!(
        "streaming: map placing budget {budget}, governor mode {mode}, requests waiting by stage {}",
        if waiting.is_empty() { "none".to_owned() } else { waiting.join(" ") }
    )
}

/// The class of a piece by its distance from the focus: near, mid, far,
/// nowhere (no place, or no focus to measure from).
fn class_of(place: Option<(f32, f32)>, focus: Option<(f32, f32)>) -> usize {
    match (place, focus) {
        (Some((x, y)), Some((fx, fy))) => match (x - fx).hypot(y - fy) {
            d if d < NEAR => 0,
            d if d < FAR => 1,
            _ => 2,
        },
        _ => 3,
    }
}
const CLASSES: [&str; 4] = ["near", "mid", "far", "nowhere"];

/// The pieces of one class by need, each with its entities.
#[derive(Default)]
struct Needs {
    groups: std::collections::HashMap<Need, (u32, u32)>,
}

impl Needs {
    fn add(&mut self, need: Need, entities: u32) {
        let at = self.groups.entry(need).or_default();
        *at = (at.0 + 1, at.1.saturating_add(entities));
    }

    /// The total, then the groups with the most entities.
    fn text(&self) -> String {
        let total = self.groups.values().fold((0u32, 0u32), |a, &(n, e)| (a.0 + n, a.1.saturating_add(e)));
        let mut groups: Vec<_> = self.groups.iter().collect();
        groups.sort_by_key(|&(&need, &(n, e))| (std::cmp::Reverse(e), std::cmp::Reverse(n), need));
        let mut out: Vec<String> = groups
            .iter()
            .take(GROUPS)
            .map(|&(&(b, state, d), &(n, e))| {
                let state = state.map_or("-".to_owned(), |s| s.to_string());
                format!("b{b:03x} s{state} d{d} {n}/{e}")
            })
            .collect();
        if groups.len() > GROUPS {
            let rest = groups[GROUPS..].iter().fold((0u32, 0u32), |a, &(_, &(n, e))| (a.0 + n, a.1.saturating_add(e)));
            out.push(format!("{} more groups {}/{}", groups.len() - GROUPS, rest.0, rest.1));
        }
        format!("{}/{}: {}", total.0, total.1, if out.is_empty() { "none".to_owned() } else { out.join("; ") })
    }
}

/// The interiors by state, and those wanted with their entities by module.
#[derive(Default, Debug, PartialEq, Eq)]
struct Interiors {
    states: [u32; 5],
    odd_state: u32,
    wanted_by: [u32; MODULES],
}

impl Interiors {
    /// One interior's 16 bytes from +0xb2: the wanted levels, then at +0xc1
    /// its state.
    fn add(&mut self, bytes: &[u8; 16]) {
        match self.states.get_mut(usize::from(bytes[15])) {
            Some(n) => *n += 1,
            None => self.odd_state += 1,
        }
        // As the title's step does: the highest level wins, the first module
        // that has it. 4 and up means "with its entities".
        let wanted = &bytes[..MODULES];
        let most = wanted.iter().copied().max().unwrap_or(0);
        if (4..=0x10).contains(&most)
            && let Some(module) = wanted.iter().position(|&level| level == most)
        {
            self.wanted_by[module] += 1;
        }
    }

    fn text(&self) -> String {
        let [a, b, c, d, e] = self.states;
        let by: Vec<String> =
            self.wanted_by.iter().enumerate().filter(|&(_, &n)| n != 0).map(|(m, n)| format!("{m}:{n}")).collect();
        let odd = if self.odd_state != 0 { format!(" (other {})", self.odd_state) } else { String::new() };
        format!(
            "interiors: in state 0/1/2/3/4 {a}/{b}/{c}/{d}/{e}{odd}, wanted with entities by module {}",
            if by.is_empty() { "none".to_owned() } else { by.join(" ") }
        )
    }
}

/// The log line.
fn line(buildings: Option<Pool>, objects: Option<Pool>, waiting: Option<(u32, bool)>) -> String {
    let pool = |p: Option<Pool>| match p {
        Some(p) => format!("{} of {} ({} freed so far)", p.used, p.capacity, p.freed),
        None => "not made yet".to_owned(),
    };
    let waiting = match waiting {
        Some((n, true)) => format!("{n} or more"),
        Some((n, false)) => n.to_string(),
        None => "unknown".to_owned(),
    };
    format!(
        "map data: buildings {}, objects {}, interiors waiting for their step {waiting}",
        pool(buildings),
        pool(objects)
    )
}

/// Whether the image mapped at `base` is the build these offsets are for.
#[cfg(windows)]
fn known(base: u64) -> bool {
    let at = |rva: u32| (base + u64::from(rva)) as *const u8;
    // SAFETY: the image is mapped and these RVAs are inside its code.
    unsafe {
        std::slice::from_raw_parts(at(STEP), STEP_BYTES.len()) == STEP_BYTES
            && std::slice::from_raw_parts(at(THUNK), THUNK_BYTES.len()) == THUNK_BYTES
    }
}

/// Reads of the title's memory that do not fault on a wild pointer: an
/// address is asked about before it is read, and the last few answers (whole
/// runs of readable pages) are kept until [`Reader::forget`].
#[cfg(windows)]
#[derive(Default)]
pub(crate) struct Reader {
    /// Runs of committed, readable pages, `start..end`.
    runs: [(usize, usize); 16],
    next: usize,
}

#[cfg(windows)]
impl Reader {
    /// Forgets the answers: the title maps and unmaps memory as it runs.
    pub(crate) fn forget(&mut self) {
        *self = Self::default();
    }

    fn readable(&mut self, address: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
        };
        if self.runs.iter().any(|&(start, end)| (start..end).contains(&address)) {
            return true;
        }
        // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: reads the address space's bookkeeping, not `address` itself.
        let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
        let ok = n != 0
            && info.State == MEM_COMMIT
            && info.Protect & PAGE_GUARD == 0
            && info.Protect & 0xff != PAGE_NOACCESS;
        if ok {
            let start = info.BaseAddress as usize;
            self.runs[self.next] = (start, start.saturating_add(info.RegionSize));
            self.next = (self.next + 1) % self.runs.len();
        }
        ok
    }

    pub(crate) fn bytes<const N: usize>(&mut self, address: usize) -> Option<[u8; N]> {
        if address == 0 || !self.readable(address) || !self.readable(address + N - 1) {
            return None;
        }
        // SAFETY: both ends are in committed, readable pages; the bytes are
        // plain data that another thread may change under us.
        Some(unsafe { std::ptr::read_volatile(address as *const [u8; N]) })
    }

    pub(crate) fn pointer(&mut self, address: usize) -> Option<usize> {
        self.bytes::<8>(address).map(usize::from_le_bytes)
    }

    pub(crate) fn pool(&mut self, at: usize) -> Option<Pool> {
        let header = self.pointer(at)?;
        self.bytes::<0x28>(header).map(|h| pool_from(&h))
    }

    pub(crate) fn dword(&mut self, address: usize) -> Option<u32> {
        self.bytes::<4>(address).map(u32::from_le_bytes)
    }

    /// The entries in use of the pool whose pointer is at `at`: each one's
    /// address goes to `each`. `None` if the pool is not there.
    fn in_use(&mut self, at: usize, mut each: impl FnMut(&mut Self, usize)) -> Option<()> {
        let header = self.pointer(at)?;
        let h = self.bytes::<0x38>(header)?;
        let pointer = |at: usize| usize::from_le_bytes(h[at..at + 8].try_into().unwrap());
        let dword = |at: usize| u32::from_le_bytes(h[at..at + 4].try_into().unwrap());
        let (storage, size, capacity, flags, mask) = (pointer(0), dword(8), dword(0xc), pointer(0x28), dword(0x30));
        if storage == 0 || flags == 0 || capacity > 1 << 22 {
            return None;
        }
        for i in 0..capacity as usize {
            // As the title tests it: in use while every bit of the mask is set.
            if self.dword(flags + 4 * i).is_some_and(|f| !f & mask == 0) {
                each(self, storage + i * size as usize);
            }
        }
        Some(())
    }

    /// The buildings (or objects) in use by place.
    fn places(&mut self, at: usize) -> Option<Places> {
        let mut places = Places::default();
        self.in_use(at, |reader, entity| {
            places.add(reader.bytes::<16>(entity + POSITION).as_ref().and_then(place_of), 0);
        })?;
        Some(places)
    }

    fn interiors(&mut self, at: usize) -> Option<Interiors> {
        let mut interiors = Interiors::default();
        self.in_use(at, |reader, interior| {
            if let Some(bytes) = reader.bytes::<16>(interior + INTERIOR_WANTED) {
                interiors.add(&bytes);
            }
        })?;
        Some(interiors)
    }

    /// The streaming record of the object numbered `object`.
    fn record(&mut self, engine: usize, object: u32) -> Option<[u8; RECORD]> {
        let shift = self.dword(engine + 0x14).filter(|&s| s < 8)? as usize;
        let bucket = engine + (object as usize % BUCKETS) * 0xa0 + shift * 0x10;
        let (n, records) = (self.dword(bucket + 0x24)? as usize, self.pointer(bucket + 0x28)?);
        (0..n.min(1 << 14)).find_map(|k| {
            let at = records + k * RECORD;
            (self.dword(at)? & 0x0fff_ffff == object).then(|| self.bytes::<RECORD>(at))?
        })
    }

    /// The engine's budget, and per stage list the requests waiting (and
    /// whether the walk was cut short: the lists change under us).
    fn streaming(&mut self, base: usize) -> (Option<u32>, Option<u32>, Waiting) {
        let mode = self.dword(base + GOVERNOR_MODE as usize);
        let Some(engine) = self.pointer(base + ENGINE as usize).filter(|&e| e != 0) else {
            return (None, mode, Vec::new());
        };
        let waiting = (0..STAGE_COUNT)
            .map(|stage| {
                let (mut request, mut n) = (self.pointer(engine + STAGES + 8 * stage).unwrap_or(0), 0u32);
                while request != 0 && n < 1 << 14 {
                    n += 1;
                    match self.pointer(request + REQUEST_NEXT) {
                        Some(next) => request = next,
                        None => return (stage, n, true),
                    }
                }
                (stage, n, request != 0)
            })
            .collect();
        (self.dword(engine + BUDGET), mode, waiting)
    }

    /// The store's loaded pieces: the slots in use, the pieces by the place
    /// of their first entity (each carrying its number of entities), and
    /// the pieces by need in the four classes of distance from `focus`.
    fn pieces(&mut self, base: usize, focus: Option<(f32, f32)>) -> Option<(u32, Places, [Needs; 4])> {
        let store = base + STORE as usize;
        let (slots, size, capacity, bits) = (
            self.pointer(store + 0x38)?,
            self.dword(store + 0x40)? as usize,
            self.dword(store + 0x44)? as usize,
            self.pointer(store + 0x60)?,
        );
        if slots == 0 || bits == 0 || !(8..=0x1000).contains(&size) || capacity > 1 << 22 {
            return None;
        }
        // For the needs: the store's first object number, the box
        // streamer's word per slot, the engine. Without them the needs say so.
        let (first_object, box_words, engine) = (
            self.dword(store + 0x8).unwrap_or(0),
            self.pointer(store + 0xe0).unwrap_or(0),
            self.pointer(base + ENGINE as usize).unwrap_or(0),
        );
        let (mut used, mut places, mut needs) = (0, Places::default(), <[Needs; 4]>::default());
        for i in 0..capacity {
            if !self.bytes::<1>(bits + i / 8).is_some_and(|b| b[0] >> (i % 8) & 1 == 1) {
                continue;
            }
            used += 1;
            let Some(contents) = self.pointer(slots + i * size).filter(|&c| c != 0) else { continue };
            let (Some(entities), Some(n)) = (self.pointer(contents + 0x8), self.dword(contents + 0x10)) else {
                continue;
            };
            let n = n.min(1 << 20);
            // Entities the title left out are null: the first that is there.
            let first = (0..n.min(16) as usize)
                .find_map(|k| self.pointer(entities + 8 * k).filter(|&e| e != 0));
            let place = first.and_then(|e| self.bytes::<16>(e + POSITION)).as_ref().and_then(place_of);
            places.add(place, n);
            let box_word = self.bytes::<2>(box_words + 2 * i).map_or(0xffff, u16::from_le_bytes);
            let record = self.record(engine, first_object.wrapping_add(i as u32));
            let need: Need = match record.as_ref() {
                Some(r) => (box_word & 0x1ff, Some(state_of(r)), u16::from_le_bytes([r[0x14], r[0x15]])),
                None => (box_word & 0x1ff, None, 0),
            };
            needs[class_of(place, focus)].add(need, n);
        }
        Some((used, places, needs))
    }

    /// Nodes in the waiting list, and whether the walk was cut short.
    fn waiting(&mut self, head: usize) -> Option<(u32, bool)> {
        let mut node = self.pointer(head)?;
        let mut n = 0;
        while node != 0 {
            if n == WALK_LIMIT {
                return Some((n, true));
            }
            let Some(pair) = self.bytes::<16>(node) else { return Some((n, true)) };
            node = usize::from_le_bytes(pair[8..].try_into().unwrap());
            n += 1;
        }
        Some((n, false))
    }
}

/// Starts a thread that logs the pools and the interiors' waiting list of
/// the image mapped at `base` every five seconds. Says why if it does not.
#[cfg(windows)]
pub fn watch(base: u64) -> Result<(), &'static str> {
    if !known(base) {
        return Err("not the build it was written for (bytes differ)");
    }
    let base = base as usize;
    std::thread::Builder::new()
        .name("victory pool watch".into())
        .spawn(move || {
            let mut reader = Reader::default();
            let mut second = 0u64;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                second += 1;
                reader.forget();
                let buildings = reader.pool(base + BUILDINGS as usize);
                let focus = reader
                    .pointer(base + FOCUS as usize)
                    .and_then(|entity| reader.bytes::<16>(entity + POSITION))
                    .as_ref()
                    .and_then(place_of);
                tracing::info!("{}", focus_line(focus, buildings));
                if !second.is_multiple_of(5) {
                    continue;
                }
                let objects = reader.pool(base + OBJECTS as usize);
                let waiting = reader.waiting(base + WAITING as usize);
                tracing::info!("{}", line(buildings, objects, waiting));
                let text = |places: Option<Places>| places.map_or("unknown".to_owned(), |p| p.text(false));
                tracing::info!(
                    "buildings by place (500 m squares, centre in km, count): {}",
                    text(reader.places(base + BUILDINGS as usize))
                );
                tracing::info!("objects by place: {}", text(reader.places(base + OBJECTS as usize)));
                match reader.interiors(base + INTERIORS as usize) {
                    Some(interiors) => tracing::info!("{}", interiors.text()),
                    None => tracing::info!("interiors: unknown"),
                }
                // The store's dwords from +0x40 go along: the slot size, then
                // the guessed capacity and its neighbours, to check the guess.
                let fields = reader
                    .bytes::<0x14>(base + STORE as usize + 0x40)
                    .map(|b| b.chunks(4).map(|d| u32::from_le_bytes(d.try_into().unwrap()).to_string()).collect::<Vec<_>>().join(" "))
                    .unwrap_or_default();
                match reader.pieces(base, focus) {
                    Some((used, places, needs)) => {
                        tracing::info!(
                            "map pieces: {used} slots in use (store +0x40.. {fields}); loaded by place (pieces/entities): {}",
                            places.text(true)
                        );
                        for (class, needs) in CLASSES.iter().zip(&needs) {
                            tracing::info!("map pieces {class} (box, state, dependents, pieces/entities) {}", needs.text());
                        }
                    }
                    None => tracing::info!("map pieces: the store is not as read (store +0x40.. {fields})"),
                }
                let (budget, mode, stages) = reader.streaming(base);
                tracing::info!("{}", streaming_line(budget, mode, &stages));
                tracing::info!("{}", crate::srl_slice::line(&crate::srl_slice::seen(&mut reader, base)));
            }
        })
        .map(|_| ())
        .map_err(|_| "its thread did not start")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_turns_the_watch_on() {
        assert!(watch_wanted(Some("1")));
        assert!(!watch_wanted(Some("0")));
        assert!(!watch_wanted(Some("")));
        assert!(!watch_wanted(None));
    }

    /// The thunk's call goes to the step these offsets name.
    #[test]
    fn the_thunk_calls_the_step() {
        assert_eq!(THUNK_BYTES[8], 0xe8);
        let delta = i32::from_le_bytes(THUNK_BYTES[9..13].try_into().unwrap());
        assert_eq!((i64::from(THUNK) + 13 + i64::from(delta)) as u32, STEP);
    }

    #[test]
    fn a_pool_header_gives_its_numbers() {
        let mut header = [0u8; 0x28];
        header[0x8..0xc].copy_from_slice(&0xe0u32.to_le_bytes());
        header[0xc..0x10].copy_from_slice(&22_500u32.to_le_bytes());
        header[0x10..0x14].copy_from_slice(&22_196u32.to_le_bytes());
        // The free list's head: an empty list (0x1fffe) after 5821 frees.
        header[0x20..0x28].copy_from_slice(&0x16bd_0001_fffeu64.to_le_bytes());
        assert_eq!(pool_from(&header), Pool { used: 22_196, capacity: 22_500, freed: 5821 });
    }

    fn position(x: f32, y: f32, z: f32, sx: i16, sy: i16) -> [u8; 16] {
        let mut p = [0u8; 16];
        p[0..4].copy_from_slice(&x.to_le_bytes());
        p[4..8].copy_from_slice(&y.to_le_bytes());
        p[8..12].copy_from_slice(&z.to_le_bytes());
        p[12..14].copy_from_slice(&sx.to_le_bytes());
        p[14..16].copy_from_slice(&sy.to_le_bytes());
        p
    }

    #[test]
    fn a_position_is_its_floats_plus_32_m_squares() {
        // The waterfront of the crash logs, about -488, -5413.
        assert_eq!(place_of(&position(24.0, 27.0, 3.0, -16, -170)), Some((-488.0, -5413.0)));
        assert_eq!(place_of(&position(5.0, 6.0, 0.0, 0, 0)), Some((5.0, 6.0)));
        assert_eq!(place_of(&position(f32::NAN, 0.0, 0.0, 0, 0)), None);
        assert_eq!(place_of(&position(0.0, 0.0, 0.0, 1000, 0)), None, "32 km out is not on the map");
        assert_eq!(place_of(&position(0.0, 0.0, 1.0e9, 0, 0)), None);
    }

    #[test]
    fn pieces_are_classed_by_their_distance_from_the_focus() {
        let focus = Some((0.0, 0.0));
        assert_eq!(class_of(Some((600.0, 700.0)), focus), 0);
        assert_eq!(class_of(Some((-1000.0, 100.0)), focus), 1);
        assert_eq!(class_of(Some((2999.0, 0.0)), focus), 1);
        assert_eq!(class_of(Some((3000.0, 1.0)), focus), 2);
        assert_eq!(class_of(None, focus), 3);
        assert_eq!(class_of(Some((1.0, 1.0)), None), 3);
        assert_eq!(CLASSES[2], "far");
    }

    #[test]
    fn needs_name_the_groups_with_the_most_entities_first() {
        let mut needs = Needs::default();
        needs.add((0x008, Some(5), 0), 20);
        needs.add((0x008, Some(5), 0), 30);
        needs.add((0x000, Some(3), 1), 400);
        needs.add((0x180, None, 0), 1);
        assert_eq!(needs.text(), "4/451: b000 s3 d1 1/400; b008 s5 d0 2/50; b180 s- d0 1/1");
        assert_eq!(Needs::default().text(), "0/0: none");
        for k in 0..(GROUPS as u16 + 3) {
            needs.add((k, Some(0), 0), 1000 + u32::from(k));
        }
        assert!(needs.text().starts_with("13/9487: b008 s0 d0 1/1008; b007 s0 d0 1/1007; "));
        assert!(needs.text().ends_with("; 6 more groups 7/3454"), "{}", needs.text());
    }

    #[test]
    fn a_records_state_is_four_bits_of_its_last_word() {
        let mut record = [0u8; RECORD];
        record[0x1a..].copy_from_slice(&(5u16 << 5 | 0x1f | 0xfe00).to_le_bytes());
        assert_eq!(state_of(&record), 5);
        record[0x1a..].copy_from_slice(&(0xfu16 << 5).to_le_bytes());
        assert_eq!(state_of(&record), 15);
    }

    #[test]
    fn the_streaming_line_names_the_stages_with_requests() {
        assert_eq!(
            streaming_line(Some(250), Some(0x6), &[(0, 0, false), (7, 12, false), (8, 16384, true)]),
            "streaming: map placing budget 250 us a frame, governor mode 0x6, requests waiting by stage 7:12 8:16384+"
        );
        assert_eq!(
            streaming_line(Some(0), None, &[]),
            "streaming: map placing budget no limit, governor mode unknown, requests waiting by stage none"
        );
        assert!(streaming_line(None, Some(0), &[]).starts_with("streaming: map placing budget unknown, governor mode 0x0"));
    }

    #[test]
    fn the_focus_line_is_in_km() {
        let pool = Pool { used: 10_853, capacity: 22_500, freed: 4361 };
        assert_eq!(
            focus_line(Some((-488.4, -5413.8)), Some(pool)),
            "focus -0.49,-5.41 km; buildings 10853 (4361 freed so far)"
        );
        assert_eq!(focus_line(None, None), "focus none");
    }

    #[test]
    fn places_name_the_fullest_squares_first() {
        let mut places = Places::default();
        for _ in 0..3 {
            places.add(Some((-488.0, -5413.0)), 0);
        }
        places.add(Some((737.0, 6523.0)), 0);
        places.add(None, 0);
        assert_eq!(places.text(false), "-0.25,-5.25 3; 0.75,6.75 1; nowhere 1");
        assert_eq!(Places::default().text(false), "none");

        let mut pieces = Places::default();
        pieces.add(Some((10.0, 10.0)), 5);
        pieces.add(Some((20.0, 20.0)), 7);
        pieces.add(Some((-10.0, 10.0)), 100);
        assert_eq!(pieces.text(true), "-0.25,0.25 1/100; 0.25,0.25 2/12");

        let mut many = Places::default();
        for i in 0..(NAMED as i32 + 2) {
            for _ in 0..=i {
                many.add(Some((i as f32 * SQUARE + 1.0, 1.0)), 0);
            }
        }
        assert!(many.text(false).starts_with("4.75,0.25 10; 4.25,0.25 9; "));
        assert!(many.text(false).ends_with("; 2 more squares 3"));
    }

    #[test]
    fn interiors_count_by_state_and_by_the_module_that_wants_most() {
        let mut interiors = Interiors::default();
        let mut off = [1u8; 16];
        off[14] = 0;
        off[15] = 0;
        interiors.add(&off);
        let mut on = off;
        on[5] = 0x10;
        on[9] = 0x10;
        on[2] = 4;
        on[15] = 4;
        interiors.add(&on);
        let mut partly = off;
        partly[6] = 4;
        partly[15] = 2;
        interiors.add(&partly);
        let mut shell = off;
        shell[3] = 2;
        shell[15] = 1;
        interiors.add(&shell);
        let mut odd = off;
        odd[15] = 9;
        interiors.add(&odd);
        assert_eq!(interiors.states, [1, 1, 1, 0, 1]);
        assert_eq!(interiors.odd_state, 1);
        assert_eq!(interiors.wanted_by[5], 1, "the first module with the highest level");
        assert_eq!(interiors.wanted_by[6], 1);
        assert_eq!(interiors.wanted_by.iter().sum::<u32>(), 2);
        assert_eq!(
            interiors.text(),
            "interiors: in state 0/1/2/3/4 1/1/1/0/1 (other 1), wanted with entities by module 5:1 6:1"
        );
    }

    #[test]
    fn the_line_says_what_is_known() {
        let full = Pool { used: 22_500, capacity: 22_500, freed: 5821 };
        assert_eq!(
            line(Some(full), None, Some((12, false))),
            "map data: buildings 22500 of 22500 (5821 freed so far), objects not made yet, \
             interiors waiting for their step 12"
        );
        assert!(line(None, None, Some((WALK_LIMIT, true))).ends_with("65536 or more"));
        assert!(line(None, None, None).ends_with("unknown"));
    }
}
