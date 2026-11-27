//! Mesh shader output signatures, counts and lane indices.

use super::{Attr, Bin, HashMap, Lower, SigElement, V, Why, pred};
use crate::emit::mesh_size;
use crate::emit::shader::Mesh;

/// Semantic key `(UPPERCASE_NAME, index)` -> 4-bit absolute channel mask of
/// arbitrary (`kind == 0`) inputs read by the paired pixel shader.
pub(crate) type KeptOutputs = HashMap<(String, u32), u8>;

/// Keeps only arbitrary (`kind == 0`) signature elements whose semantic has at
/// least one channel read in `keep`, trimming unread leading and trailing
/// channels. System values (`kind != 0`) stay untouched.
pub(super) fn prune_signature(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    elements.retain_mut(|e| {
        if e.kind != 0 {
            return true;
        }
        if e.row.is_none() {
            return false;
        }
        let key = (e.name.to_ascii_uppercase(), e.index);
        let used = keep.get(&key).copied().unwrap_or(0);
        let rel = (used >> e.col) & ((1u8 << e.cols) - 1);
        if rel == 0 {
            return false;
        }
        let first = rel.trailing_zeros() as u8;
        let last = 7 - rel.leading_zeros() as u8;
        e.col += first;
        e.cols = last - first + 1;
        true
    });
}

/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the rows to merge (no semantic has this name); the value is the mode.
const MERGE_KEY: &str = "\u{0}merge-row";
/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the host register of a console row ([`plan_merge`]); the value is the
/// register.
const ORDER_KEY: &str = "\u{0}host-row";
/// The semantic of a merged row; its index is the row.
const MERGED_NAME: &str = "VROW";
/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the columns of a row the pixel shader reads, where an element of the row
/// spans a column it does not read ([`split_rows`]); the value is the mask.
const SPLIT_KEY: &str = "\u{0}split-row";
/// The semantic of a part of a split element; its index is four times the
/// row plus the part's first column.
const PART_NAME: &str = "VPART";
/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the rows whose clip distances the pixel shader clips by ([`soften_clip`]);
/// the value is the element's column mask.
const CLIP_KEY: &str = "\u{0}soft-clip";
/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the columns of a row's clip distance element that can clip
/// ([`idle_clip`], [`trim_clip`]); the value is their mask, 0 for none.
const IDLE_KEY: &str = "\u{0}idle-clip";
/// The semantic of clip distances passed as an ordinary element; its index
/// is four times the row plus the element's first column.
pub(super) const CLIP_NAME: &str = "VCLIP";

/// macOS port (Metal's 32 KiB mesh output limit): marks in `keep` the rows
/// of `outputs` that hold two or more arbitrary elements once pruned, all of
/// one component type, so both stages of a linked pair make each such row
/// one element ([`merge_rows`]). D3DMetal sizes a row of several elements
/// (`xyz` and `w`, say) well above the 16 bytes one `float4` takes.
///
/// With `reorder` the arbitrary rows also trade registers among themselves so
/// that whole `float4` rows come first, then three, two and one components:
/// Metal lays the rows out in register order, a `float4` on a 16-byte
/// boundary and three components packed in 12 bytes, so a `float4` after an
/// odd number of `xyz` rows costs 4 bytes of padding (notes/mesh-size.md).
///
/// With `distances_last` as well, rows that hold only clip or cull distances
/// take the last registers, after the arbitrary rows. This alone saves
/// nothing: D3DMetal declares the distances twice, as Metal's clip-distance
/// array right after the position (one 4-byte slot, wherever their register
/// is) and as an ordinary output of their width at their register.
///
/// With `xyz_first` as well, and only where such rows exist, the arbitrary
/// rows go three components first, then two and one, the whole `float4`
/// rows last: after that 4-byte slot `float4` rows cost Metal padding that
/// rows of three do not (measured with Metal itself, notes/mesh-size.md,
/// "Correction, 26 November afternoon").
///
/// The three paragraphs above are the reasons those levels were made
/// for; measured, they are wrong (`emit::mesh_size` has the rule). With
/// `fitted` the plan follows the measured rule instead: an element the pixel
/// shader reads with a gap is declared as its read parts ([`split_rows`]),
/// which drops 4 bytes for each component nothing reads, and the rows,
/// those of distances alone among them, take the order in which the rule's
/// size of a vertex is smallest ([`fitted_order`]).
///
/// With `soft_clip` (the bytes a vertex may take) as well, a pair that is
/// still over them gives up the clipper: D3DMetal adds 16 bytes to a
/// vertex with clip distances, so the distances go out as an ordinary
/// element and the pixel shader discards where one is negative
/// ([`soften_clip`], [`Lower::soft_clip`]). Not exact as the rest is: see
/// there.
///
/// `live_clip` (from [`idle_clip`]; before the clipper is given up) names
/// rows whose clip distance element has columns that never clip, with the
/// mask of those that can: both stages then declare only these
/// ([`trim_clip`]), which is exact and saves 4 bytes a distance, or the
/// distances' 16 and the clipper's 16 where none can clip.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_merge(
    keep: &mut KeptOutputs,
    outputs: &[SigElement],
    widen: bool,
    reorder: bool,
    distances_last: bool,
    xyz_first: bool,
    fitted: bool,
    soft_clip: Option<u32>,
    live_clip: &[(u32, u8)],
) {
    for &(r, mask) in live_clip {
        keep.insert((IDLE_KEY.to_string(), r), mask);
    }
    let mut pruned = outputs.to_vec();
    prune_signature(&mut pruned, keep);
    let arbitrary = |r: u32| pruned.iter().filter(move |e| e.kind == 0 && e.row == Some(r));
    let rows: std::collections::BTreeSet<u32> =
        pruned.iter().filter(|e| e.kind == 0).filter_map(|e| e.row).collect();
    for r in rows {
        let first = arbitrary(r).next().expect("a row from the list");
        let same = arbitrary(r).all(|e| e.comp == first.comp && e.half == first.half);
        if same && (widen || arbitrary(r).count() > 1) {
            keep.insert((MERGE_KEY.to_string(), r), 1 + u8::from(widen));
        }
    }
    if reorder {
        merge_rows(&mut pruned, keep);
        // Rows still shared by elements of several types keep their place.
        let mut rows: Vec<(u32, u8)> = Vec::new();
        for e in pruned.iter().filter(|e| e.kind == 0) {
            let Some(r) = e.row else { continue };
            match rows.iter_mut().find(|(row, _)| *row == r) {
                Some((_, cols)) => *cols = 0,
                None => rows.push((r, e.cols)),
            }
        }
        rows.retain(|&(_, cols)| cols != 0);
        let mut registers: Vec<u32> = rows.iter().map(|&(r, _)| r).collect();
        // Rows of distances alone, in their order (pos1 and pos2 find them
        // by kind and index, not by row).
        let distance = |e: &SigElement| matches!(e.kind, 6 | 7);
        let mut dists: Vec<u32> = Vec::new();
        if distances_last {
            dists = pruned.iter().filter(|e| distance(e)).filter_map(|e| e.row).collect();
            dists.sort_unstable();
            dists.dedup();
            dists.retain(|&r| pruned.iter().all(|e| e.row != Some(r) || distance(e)));
        }
        if xyz_first && !dists.is_empty() {
            // 3, 2, 1, then 4 components.
            rows.sort_by_key(|&(r, cols)| (if cols == 4 { 3 } else { 3 - cols }, r));
        } else {
            rows.sort_by_key(|&(r, cols)| (std::cmp::Reverse(cols), r));
        }
        registers.extend(&dists);
        rows.extend(dists.iter().map(|&r| (r, 0)));
        registers.sort_unstable();
        if fitted {
            // The columns each row's elements are read at; a row with an
            // element that spans more is declared in parts.
            let mut read: HashMap<u32, u8> = HashMap::new();
            for e in outputs.iter().filter(|e| e.kind == 0) {
                let Some(r) = e.row else { continue };
                let used = keep.get(&(e.name.to_ascii_uppercase(), e.index)).copied().unwrap_or(0);
                *read.entry(r).or_insert(0) |= used & (((1u8 << e.cols) - 1) << e.col);
            }
            for e in pruned.iter().filter(|e| e.kind == 0) {
                let Some(r) = e.row else { continue };
                let span = ((1u8 << e.cols) - 1) << e.col;
                let mask = read.get(&r).copied().unwrap_or(0);
                if mask & span != span && mask & span != 0 {
                    keep.insert((SPLIT_KEY.to_string(), r), mask);
                }
            }
            split_rows(&mut pruned, keep);
            // The members of a vertex after the position, as D3DMetal lays
            // them out: the signature's order, which is the registers'.
            // The rows that move take the registers of the rows that move;
            // every other row stays where it is.
            let members = |r: u32| -> Vec<u8> {
                pruned.iter().filter(|e| e.row == Some(r) && e.kind != 3).map(|e| e.cols).collect()
            };
            let mut all: Vec<u32> = pruned.iter().filter(|e| e.kind != 3).filter_map(|e| e.row).collect();
            all.sort_unstable();
            all.dedup();
            let free: Vec<Vec<u8>> = rows.iter().map(|&(r, _)| members(r)).collect();
            let slots: Vec<Option<Vec<u8>>> =
                all.iter().map(|r| (!registers.contains(r)).then(|| members(*r))).collect();
            let fixed = mesh_size::fixed_bytes(&pruned);
            let mut order = fitted_order(fixed, &slots, &free);
            // Still over: without the clipper's 16 bytes, where every clip
            // distance element has a row to itself.
            let clips: Vec<(u32, u8)> = pruned
                .iter()
                .filter(|e| e.kind == 6)
                .filter_map(|e| Some((e.row?, ((1u8 << e.cols) - 1) << e.col)))
                .collect();
            let alone = clips.iter().all(|&(r, _)| pruned.iter().filter(|e| e.row == Some(r)).count() == 1);
            if let Some(most) = soft_clip
                && !clips.is_empty()
                && alone
                && order.len() == rows.len()
                && laid_out(fixed, &slots, &free, &order) > most
            {
                for (r, mask) in clips {
                    keep.insert((CLIP_KEY.to_string(), r), mask);
                }
                order = fitted_order(fixed - 16, &slots, &free);
            }
            if order.len() == rows.len() {
                rows = order.into_iter().map(|i| rows[i]).collect();
            }
        }
        for (&(r, _), register) in rows.iter().zip(registers) {
            keep.insert((ORDER_KEY.to_string(), r), register as u8);
        }
    }
}

/// The host register [`plan_merge`] gave the arbitrary (or distance) element
/// at console row `row`, if it moved it.
fn host_row(keep: &KeptOutputs, row: u32) -> Option<u32> {
    keep.get(&(ORDER_KEY.to_string(), row)).map(|&r| u32::from(r))
}

/// The elements [`plan_merge`] may give other registers: arbitrary ones, and
/// clip and cull distances.
fn movable(e: &SigElement) -> bool {
    matches!(e.kind, 0 | 6 | 7)
}

/// Gives the elements the registers [`plan_merge`] chose. Last: lowering
/// finds an element by its console row.
pub(super) fn renumber_rows(elements: &mut [SigElement], keep: &KeptOutputs) {
    for e in elements.iter_mut().filter(|e| movable(e)) {
        if let Some(register) = e.row.and_then(|r| host_row(keep, r)) {
            e.row = Some(register);
        }
    }
}

/// Makes each row [`plan_merge`] marked one element from column 0 to the
/// row's last used column (all four when the plan widens), named by its row,
/// so the mesh and the pixel shader of a pair declare the same thing whatever
/// either leaves out. Exports and reads find their channel by row and
/// column as before.
pub(super) fn merge_rows(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    let mut done = std::collections::BTreeSet::new();
    let mut at = 0;
    while at < elements.len() {
        let e = &elements[at];
        let mode = match (e.kind, e.row) {
            (0, Some(r)) => keep.get(&(MERGE_KEY.to_string(), r)).copied().map(|m| (r, m)),
            _ => None,
        };
        let Some((r, mode)) = mode else {
            at += 1;
            continue;
        };
        if !done.insert(r) {
            // A later element of a merged row: the first one holds it.
            elements.remove(at);
            continue;
        }
        let (mut end, mut usage) = (0, 0);
        for o in elements.iter().filter(|o| o.kind == 0 && o.row == Some(r)) {
            end = end.max(o.col + o.cols);
            usage |= o.usage << o.col;
        }
        let e = &mut elements[at];
        e.name = MERGED_NAME.to_string();
        e.index = r;
        e.col = 0;
        e.cols = if mode == 2 { 4 } else { end };
        e.usage = usage;
        at += 1;
    }
    split_rows(elements, keep);
    trim_clip(elements, keep);
    // The signature lists the elements that move in the order of their host
    // registers, in the places those elements had.
    let slots: Vec<usize> = (0..elements.len())
        .filter(|&i| movable(&elements[i]) && elements[i].row.is_some_and(|r| host_row(keep, r).is_some()))
        .collect();
    let mut moved: Vec<SigElement> = slots.iter().map(|&i| elements[i].clone()).collect();
    moved.sort_by_key(|e| e.row.and_then(|r| host_row(keep, r)));
    for (i, e) in slots.into_iter().zip(moved) {
        elements[i] = e;
    }
    soften_clip(elements, keep);
}

/// The clip distance elements of the mesh program's signature with a
/// distance that never clips, as (row, mask of the columns that can clip):
/// a distance every export writes as a constant that is not negative (a
/// `v_mov_b32` of +0.0 or more, for every lane; the title's ground
/// materials write three of their four so). The clipper removes what is under zero, so such a
/// distance removes nothing. Only elements whose other columns are one run
/// (or none) are named, and none where a row holds anything else.
pub(crate) fn idle_clip(p: &crate::ir::Program, io: &crate::io::Io, outputs: &[SigElement]) -> Vec<(u32, u8)> {
    use crate::io::Target;
    use crate::gfx10::Fields;
    // A value that is, for every lane, a constant that is not negative: one
    // a `v_mov_b32` gives, a VGPR nothing set on entry (lowered as zero:
    // `Lower::inputs`), or a phi of such values (itself among them).
    fn idle(p: &crate::ir::Program, io: &crate::io::Io, v: crate::ir::Value, seen: &mut Vec<crate::ir::Value>) -> bool {
        if seen.contains(&v) {
            return true;
        }
        seen.push(v);
        if let Some(phi) = p.blocks.iter().flat_map(|b| &b.phis).find(|f| f.value == v) {
            return seen.len() < 64 && phi.args.iter().all(|&a| idle(p, io, a, seen));
        }
        if p.inputs.iter().any(|i| i.1 == v && matches!(i.0, crate::ir::Reg::V(_))) {
            return !io.inputs.iter().any(|i| i.0 == v);
        }
        let inst = p.blocks.iter().flat_map(|b| &b.insts).find(|i| i.writes.iter().any(|w| w.1 == v));
        inst.is_some_and(|i| match &i.fields {
            Fields::Valu(f) if i.name() == "v_mov_b32" && f.neg == 0 && f.abs == 0 => {
                f.src.first().and_then(|&o| super::scalar_constant(o)).is_some_and(|bits| bits <= 0x7F80_0000)
            }
            _ => false,
        })
    }
    let idle = |v: crate::ir::Value| idle(p, io, v, &mut Vec::new());
    // The distance channels in the order the position exports carry them.
    let mut dists: Vec<&SigElement> = outputs.iter().filter(|e| matches!(e.kind, 6 | 7) && e.row.is_some()).collect();
    dists.sort_by_key(|e| (e.kind, e.index));
    let flat: Vec<(u32, u8)> =
        dists.iter().flat_map(|e| (0..e.cols).map(move |c| (e.row.unwrap_or(0), e.col + c))).collect();
    let mut found = Vec::new();
    for e in dists.iter().filter(|e| e.kind == 6) {
        let Some(r) = e.row else { continue };
        if outputs.iter().filter(|o| o.row == Some(r)).count() != 1 {
            continue;
        }
        let mut live = 0u8;
        for col in e.col..e.col + e.cols {
            let Some(k) = flat.iter().position(|&at| at == (r, col)) else { continue };
            // What writes the channel: its position export, and a
            // parameter export of the same row.
            let mut writes = 0;
            let mut all_idle = true;
            for x in &io.exports {
                let ch = match x.target {
                    Target::Position(n) if usize::from(n) == 1 + k / 4 => k % 4,
                    Target::Param { row, .. } if u32::from(row) == r => usize::from(col),
                    _ => continue,
                };
                if let Some(v) = x.values[ch] {
                    writes += 1;
                    all_idle &= idle(v);
                }
            }
            if writes == 0 || !all_idle {
                live |= 1 << col;
            }
        }
        let span = ((1u8 << e.cols) - 1) << e.col;
        let run = live == 0 || (live >> live.trailing_zeros()) & ((live >> live.trailing_zeros()) + 1) == 0;
        if live != span && run {
            found.push((r, live));
        }
    }
    found
}

/// Declares a clip distance element [`plan_merge`] has the columns for
/// ([`idle_clip`]) as those columns only, or not at all, in both stages.
/// Exports to the columns left out are dropped as those to a pruned
/// element are.
fn trim_clip(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    elements.retain_mut(|e| {
        let live = match (e.kind, e.row) {
            (6, Some(r)) => keep.get(&(IDLE_KEY.to_string(), r)).copied(),
            _ => None,
        };
        let Some(live) = live.map(|m| m & (((1u8 << e.cols) - 1) << e.col)) else { return true };
        if live == 0 {
            return false;
        }
        let col = live.trailing_zeros() as u8;
        let cols = (live >> col).trailing_ones() as u8;
        e.usage = (e.usage >> (col - e.col)) & ((1 << cols) - 1);
        (e.col, e.cols) = (col, cols);
        true
    });
}

/// Declares the clip distance elements of the rows [`plan_merge`] marked for
/// it as ordinary elements, in their places and registers, in both stages:
/// the mesh shader writes the distances to them as before, and the pixel
/// shader discards by them ([`Lower::soft_clip`]).
fn soften_clip(elements: &mut [SigElement], keep: &KeptOutputs) {
    for e in elements.iter_mut().filter(|e| e.kind == 6) {
        let Some(r) = e.row else { continue };
        if keep.contains_key(&(CLIP_KEY.to_string(), r)) {
            e.name = CLIP_NAME.to_string();
            e.index = 4 * r + u32::from(e.col);
            e.kind = 0;
            // Clip distances are interpolated with perspective.
            e.interp = 2;
        }
    }
}

/// The pixel shader's inputs get the softened clip distances it did not
/// declare (a pixel shader need not declare `SV_ClipDistance` to be
/// clipped by it).
pub(super) fn soft_clip_inputs(inputs: &mut Vec<SigElement>, keep: &KeptOutputs) {
    let mut rows: Vec<(u32, u8)> =
        keep.iter().filter(|(k, _)| k.0 == CLIP_KEY).map(|(k, &mask)| (k.1, mask)).collect();
    rows.sort_unstable();
    for (r, mask) in rows {
        let col = mask.trailing_zeros() as u8;
        let index = 4 * r + u32::from(col);
        if mask == 0 || inputs.iter().any(|e| e.name == CLIP_NAME && e.index == index) {
            continue;
        }
        inputs.push(SigElement {
            name: CLIP_NAME.to_string(),
            index,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(r),
            col,
            cols: 8 - mask.leading_zeros() as u8 - col,
            usage: 0,
            half: false,
        });
    }
}

/// Where an element stands among the clip then cull distances the position
/// exports carry, softened clip distances with the others; `None` for
/// other elements.
pub(super) fn distance(e: &SigElement) -> Option<(u8, u32)> {
    match e.kind {
        6 => Some((0, e.index)),
        7 => Some((1, e.index)),
        0 if e.name == CLIP_NAME => Some((0, e.index)),
        _ => None,
    }
}

/// The offset after the rows of `slots`, those that move in `order`.
fn laid_out(start: u32, slots: &[Option<Vec<u8>>], free: &[Vec<u8>], order: &[usize]) -> u32 {
    let mut next = order.iter();
    slots.iter().fold(start, |at, slot| {
        let row = match slot {
            Some(row) => Some(row),
            None => next.next().and_then(|&i| free.get(i)),
        };
        row.map_or(at, |row| row.iter().fold(at, |at, &cols| mesh_size::after(at, cols)))
    })
}

/// Declares each arbitrary element of a row [`plan_merge`] marked for it as
/// its parts: one element for each run of columns the pixel shader reads,
/// named by the row and the run's first column, in the element's place and
/// register. D3DMetal sizes a vertex by its elements' components
/// (`emit::mesh_size`), so a component between two read ones costs 4 bytes
/// only while its element spans it. Both stages of a pair declare the same
/// parts; exports and reads find their channel by row and column as before,
/// and an export to a column no part holds is dropped as one to a pruned
/// element is.
fn split_rows(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    let mut at = 0;
    while at < elements.len() {
        let e = &elements[at];
        let mask = match (e.kind, e.row) {
            (0, Some(r)) => keep.get(&(SPLIT_KEY.to_string(), r)).copied(),
            _ => None,
        };
        let span = ((1u8 << e.cols) - 1) << e.col;
        let Some(mask) = mask.map(|m| m & span).filter(|&m| m != span && m != 0) else {
            at += 1;
            continue;
        };
        let e = elements.remove(at);
        let mut col = e.col;
        while col < e.col + e.cols {
            if mask >> col & 1 == 0 {
                col += 1;
                continue;
            }
            let cols = (mask >> col).trailing_ones() as u8;
            let mut part = e.clone();
            part.name = PART_NAME.to_string();
            part.index = 4 * e.row.expect("a row") + u32::from(col);
            part.col = col;
            part.cols = cols;
            part.usage = (e.usage >> (col - e.col)) & ((1 << cols) - 1);
            elements.insert(at, part);
            at += 1;
            col += cols;
        }
    }
}

/// The order of the rows that may move in which a vertex is smallest by the
/// measured rule (`emit::mesh_size`): indices into `free`, each row its
/// elements' components in the signature's order. `slots` are a vertex's
/// rows after the position in register order: `None` takes the next of the
/// rows that move, the others stay and hold these elements. `start` is
/// what the rule puts before them. Among equal orders the one that keeps
/// the given order of `free` longest is taken, so the result is the same
/// wherever it is computed.
fn fitted_order(start: u32, slots: &[Option<Vec<u8>>], free: &[Vec<u8>]) -> Vec<usize> {
    // The rows that move, by shape: which is placed matters, not which row.
    let mut shapes: Vec<&Vec<u8>> = Vec::new();
    for f in free {
        if !shapes.contains(&f) {
            shapes.push(f);
        }
    }
    let mut left: Vec<u8> = shapes.iter().map(|s| free.iter().filter(|f| f == s).count() as u8).collect();
    // Bytes from an offset to the end of a row placed there. Alignments
    // are 4, 8, 12 and 16, so only the offset modulo 48 matters.
    let place = |at: u32, row: &[u8]| row.iter().fold(at, |at, &cols| mesh_size::after(at, cols)) - at;
    // The fewest bytes the slots from `slot` on take, from this offset.
    fn least(
        slot: usize,
        at: u32,
        left: &mut Vec<u8>,
        slots: &[Option<Vec<u8>>],
        shapes: &[&Vec<u8>],
        place: &dyn Fn(u32, &[u8]) -> u32,
        memo: &mut HashMap<(usize, u32, Vec<u8>), u32>,
    ) -> u32 {
        let Some(here) = slots.get(slot) else { return 0 };
        let key = (slot, at % 48, left.clone());
        if let Some(&known) = memo.get(&key) {
            return known;
        }
        let best = match here {
            Some(row) => {
                let n = place(at % 48, row);
                n + least(slot + 1, at + n, left, slots, shapes, place, memo)
            }
            None => {
                let mut best = u32::MAX;
                for i in 0..shapes.len() {
                    if left[i] == 0 {
                        continue;
                    }
                    left[i] -= 1;
                    let n = place(at % 48, shapes[i]);
                    best = best.min(n + least(slot + 1, at + n, left, slots, shapes, place, memo));
                    left[i] += 1;
                }
                best
            }
        };
        memo.insert(key, best);
        best
    }
    let mut memo = HashMap::new();
    let mut order = Vec::with_capacity(free.len());
    let mut taken = vec![false; free.len()];
    let mut at = start;
    for (slot, here) in slots.iter().enumerate() {
        let row = match here {
            Some(row) => row,
            None => {
                let goal = least(slot, at, &mut left, slots, &shapes, &place, &mut memo);
                // The first shape, in the given order, that reaches it.
                let mut found = None;
                for i in 0..shapes.len() {
                    if left[i] == 0 {
                        continue;
                    }
                    left[i] -= 1;
                    let n = place(at % 48, shapes[i]);
                    let rest = least(slot + 1, at + n, &mut left, slots, &shapes, &place, &mut memo);
                    if n + rest == goal {
                        found = Some(i);
                        break;
                    }
                    left[i] += 1;
                }
                let Some(shape) = found else { break };
                let row = (0..free.len())
                    .find(|&i| !taken[i] && &free[i] == shapes[shape])
                    .expect("a row of the shape counted");
                taken[row] = true;
                order.push(row);
                &free[row]
            }
        };
        at += place(at % 48, row);
    }
    order
}

impl Lower<'_> {
    /// Pixel: the discard that stands in for the clipper where the pair's
    /// clip distances are an ordinary element ([`soften_clip`]): first
    /// thing, a pixel with a negative distance is discarded.
    ///
    /// Not exact as the clipper is. The distances are the same and are
    /// interpolated the same way, so the pixels kept are the clipper's but
    /// for rounding where a distance passes zero; a multisampled target
    /// loses a pixel whole where the clipper would cut through it; the
    /// pixel shader runs for what the clipper would have removed; and the
    /// depth test cannot be moved before the shader
    /// ([`Lower::early_depth_flag`] leaves such a program alone).
    pub(super) fn soft_clip(&mut self) {
        if !self.s.pixel {
            return;
        }
        let ids: Vec<usize> =
            (0..self.s.inputs.len()).filter(|&i| self.s.inputs[i].kind == 0 && self.s.inputs[i].name == CLIP_NAME).collect();
        let e = self.s.entry;
        let (zero, f32t, i32t, void) = (self.s.m.f32(0), self.s.t.f32, self.s.t.i32, self.s.t.void);
        let mut kill: Option<V> = None;
        for id in ids {
            for col in 0..self.s.inputs[id].cols {
                self.s.inputs[id].usage |= 1 << col;
                let args = [self.s.m.i32(id as u32), self.s.m.i32(0), self.s.m.i8(col), self.s.m.undef(i32t)];
                let d = self.s.op(e, "loadInput.f32", Attr::ReadNone, f32t, 4, &args);
                let out = self.s.m.cmp(e, pred::OLT, d, zero);
                kill = Some(match kill {
                    Some(k) => self.s.m.bin(e, Bin::Or, k, out),
                    None => out,
                });
            }
        }
        if let Some(kill) = kill {
            self.s.op(e, "discard", Attr::NoUnwind, void, 82, &[kill]);
        }
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    pub(super) fn mesh_prologue(&mut self, mesh: Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    pub(super) fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, index: u32, kind: u8, comp: u8, row: u32, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.to_string(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A row shared by `xyz` and `w` elements becomes one `float4` named by
    /// the row in both stages, whichever of the two a stage declares first;
    /// rows of one element, of mixed types and system values stay.
    #[test]
    fn rows_of_several_elements_merge_the_same_way_in_both_stages() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("TEXCOORD", 4, 0, 9, 5, 0, 3),
            element("TEXCOORD", 7, 0, 9, 5, 3, 1),
            element("TEXCOORD", 5, 0, 9, 6, 0, 3),
            element("COLOR", 3, 0, 5, 7, 0, 1),
            element("TEXCOORD", 20, 0, 9, 7, 1, 1),
        ];
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, false, false, false, false, false, None, &[]);
        let mut mesh = outputs.clone();
        merge_rows(&mut mesh, &keep);
        let shape = |els: &[SigElement]| -> Vec<(String, u32, Option<u32>, u8, u8)> {
            els.iter().map(|e| (e.name.clone(), e.index, e.row, e.col, e.cols)).collect()
        };
        assert_eq!(
            shape(&mesh),
            vec![
                ("SV_Position".to_string(), 0, Some(0), 0, 4),
                ("VROW".to_string(), 5, Some(5), 0, 4),
                ("TEXCOORD".to_string(), 5, Some(6), 0, 3),
                ("COLOR".to_string(), 3, Some(7), 0, 1),
                ("TEXCOORD".to_string(), 20, Some(7), 1, 1),
            ]
        );
        assert_eq!(mesh[1].usage, 0xF);
        // The pixel shader declares the `w` element first and no position.
        let mut pixel = vec![outputs[2].clone(), outputs[1].clone(), outputs[3].clone()];
        merge_rows(&mut pixel, &keep);
        assert_eq!(shape(&pixel), shape(&mesh[1..3]));
        // Widening makes every arbitrary row of one type a whole float4.
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, true, false, false, false, false, None, &[]);
        let mut wide = outputs.clone();
        merge_rows(&mut wide, &keep);
        assert_eq!(shape(&wide)[2], ("VROW".to_string(), 6, Some(6), 0, 4));
    }

    /// Reordering gives the widest rows the lowest registers, the same in
    /// both stages, and leaves system values where they were.
    #[test]
    fn reordered_rows_put_whole_rows_first() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("TEXCOORD", 0, 0, 9, 1, 0, 3),
            element("TEXCOORD", 1, 0, 9, 2, 0, 4),
            element("TEXCOORD", 2, 0, 9, 3, 0, 1),
            element("TEXCOORD", 3, 0, 9, 4, 0, 3),
            element("TEXCOORD", 4, 0, 9, 4, 3, 1),
        ];
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, false, true, false, false, false, None, &[]);
        let finish = |mut els: Vec<SigElement>| -> Vec<(String, u32, Option<u32>, u8)> {
            merge_rows(&mut els, &keep);
            renumber_rows(&mut els, &keep);
            els.iter().map(|e| (e.name.clone(), e.index, e.row, e.cols)).collect()
        };
        let mesh = finish(outputs.clone());
        assert_eq!(
            mesh,
            vec![
                ("SV_Position".to_string(), 0, Some(0), 4),
                ("TEXCOORD".to_string(), 1, Some(1), 4),
                ("VROW".to_string(), 4, Some(2), 4),
                ("TEXCOORD".to_string(), 0, Some(3), 3),
                ("TEXCOORD".to_string(), 2, Some(4), 1),
            ]
        );
        let pixel = finish(vec![outputs[5].clone(), outputs[3].clone(), outputs[4].clone(), outputs[1].clone(), outputs[2].clone()]);
        assert_eq!(pixel, mesh[1..]);
    }

    /// With distances last, a row of clip distances takes the last register
    /// in both stages, the arbitrary rows the ones before it (widest first),
    /// and the signature lists them in that order; without, it stays.
    #[test]
    fn clip_distances_take_the_last_register() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 9, 1, 0, 4),
            element("TEXCOORD", 0, 0, 9, 2, 0, 4),
            element("TEXCOORD", 1, 0, 9, 3, 0, 3),
            element("TEXCOORD", 2, 0, 9, 4, 0, 4),
            element("TEXCOORD", 3, 0, 9, 5, 0, 2),
        ];
        let reads = || {
            let mut keep = KeptOutputs::new();
            for e in outputs.iter().filter(|e| e.kind == 0) {
                keep.insert((e.name.to_ascii_uppercase(), e.index), 0xF);
            }
            keep
        };
        let shape = |keep: &KeptOutputs, mut els: Vec<SigElement>| -> Vec<(String, Option<u32>)> {
            merge_rows(&mut els, keep);
            renumber_rows(&mut els, keep);
            els.iter().map(|e| (e.name.clone(), e.row)).collect()
        };
        let mut keep = reads();
        plan_merge(&mut keep, &outputs, false, true, true, false, false, None, &[]);
        let mesh = shape(&keep, outputs.clone());
        let row = |name: &str, r: u32| (name.to_string(), Some(r));
        assert_eq!(
            mesh,
            vec![
                row("SV_Position", 0),
                row("TEXCOORD", 1),
                row("TEXCOORD", 2),
                row("TEXCOORD", 3),
                row("TEXCOORD", 4),
                row("SV_ClipDistance", 5),
            ]
        );
        let registers = |els: &[SigElement]| -> Vec<(u32, Option<u32>)> {
            let mut els = els.to_vec();
            merge_rows(&mut els, &keep);
            renumber_rows(&mut els, &keep);
            els.iter().map(|e| (e.index, e.row)).collect()
        };
        // The float4 rows first, then xyz, then xy.
        assert_eq!(
            registers(&outputs[2..]),
            vec![(0, Some(1)), (2, Some(2)), (1, Some(3)), (3, Some(4))]
        );
        // The pixel shader declares the clip distance and its inputs in
        // another order and without the position: the same registers.
        let pixel = vec![outputs[4].clone(), outputs[1].clone(), outputs[2].clone(), outputs[5].clone(), outputs[3].clone()];
        assert_eq!(shape(&keep, pixel), mesh[1..]);
        // Level 3 leaves the clip distance at its register.
        let mut keep = reads();
        plan_merge(&mut keep, &outputs, false, true, false, false, false, None, &[]);
        assert_eq!(shape(&keep, outputs.clone())[1], row("SV_ClipDistance", 1));
    }

    /// Level 5: with clip distances the rows of three components come first,
    /// then two, the float4 rows last and the distances after them, the same
    /// in both stages; without distances the order is level 4's.
    #[test]
    fn with_clip_distances_rows_of_three_come_first() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 9, 1, 0, 4),
            element("TEXCOORD", 0, 0, 9, 2, 0, 4),
            element("TEXCOORD", 1, 0, 9, 3, 0, 3),
            element("TEXCOORD", 2, 0, 9, 4, 0, 4),
            element("TEXCOORD", 3, 0, 9, 5, 0, 2),
        ];
        let plan = |outputs: &[SigElement], xyz_first: bool| {
            let mut keep = KeptOutputs::new();
            for e in outputs.iter().filter(|e| e.kind == 0) {
                keep.insert((e.name.to_ascii_uppercase(), e.index), 0xF);
            }
            plan_merge(&mut keep, outputs, false, true, true, xyz_first, false, None, &[]);
            keep
        };
        let shape = |keep: &KeptOutputs, mut els: Vec<SigElement>| -> Vec<(String, u32, Option<u32>)> {
            merge_rows(&mut els, keep);
            renumber_rows(&mut els, keep);
            els.iter().map(|e| (e.name.clone(), e.index, e.row)).collect()
        };
        let keep = plan(&outputs, true);
        let mesh = shape(&keep, outputs.clone());
        let el = |name: &str, index: u32, r: u32| (name.to_string(), index, Some(r));
        assert_eq!(
            mesh,
            vec![
                el("SV_Position", 0, 0),
                el("TEXCOORD", 1, 1),
                el("TEXCOORD", 3, 2),
                el("TEXCOORD", 0, 3),
                el("TEXCOORD", 2, 4),
                el("SV_ClipDistance", 0, 5),
            ]
        );
        let pixel = vec![outputs[4].clone(), outputs[1].clone(), outputs[2].clone(), outputs[5].clone(), outputs[3].clone()];
        assert_eq!(shape(&keep, pixel), mesh[1..]);
        // No distances: as level 4, widest first.
        let plain: Vec<SigElement> = outputs.iter().filter(|e| e.kind != 6).cloned().collect();
        let keep = plan(&plain, true);
        assert_eq!(shape(&keep, plain.clone()), shape(&plan(&plain, false), plain.clone()));
        assert_eq!(shape(&keep, plain.clone())[1], el("TEXCOORD", 0, 2));
    }
    /// The rows of a ground-material shape of 18 registers that Metal
    /// refused at levels 3, 4 and 5: the position, four clip distances, and
    /// 16 rows of arbitrary elements (three and two elements in rows 5 and
    /// 7). The pixel shader reads all but the third component of row 15
    /// and, with `seven` false, the fourth of row 7.
    fn ground_shape(seven: bool) -> (Vec<SigElement>, KeptOutputs) {
        let mut outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 9, 1, 0, 4),
        ];
        // (row, column, components) of each arbitrary element.
        let mut shape: Vec<(u32, u8, u8)> = vec![(2, 0, 4), (3, 0, 4), (4, 0, 4), (5, 0, 1), (5, 1, 2), (5, 3, 1), (6, 0, 4), (7, 0, 3), (7, 3, 1)];
        shape.extend((8..14).map(|r| (r, 0, 3)));
        shape.extend([(14, 0, 2), (15, 0, 4), (16, 0, 3), (17, 0, 3)]);
        let mut keep = KeptOutputs::new();
        for (n, (row, col, cols)) in shape.into_iter().enumerate() {
            let e = element("TEXCOORD", n as u32, 0, 9, row, col, cols);
            let read = match (row, col) {
                (15, _) => 0xB,
                (7, 3) if !seven => 0,
                _ => e.mask_for_test(),
            };
            if read != 0 {
                keep.insert((e.name.clone(), e.index), read);
            }
            outputs.push(e);
        }
        (outputs, keep)
    }

    impl SigElement {
        fn mask_for_test(&self) -> u8 {
            ((1u8 << self.cols) - 1) << self.col
        }
    }

    /// Level 6 on the two ground shapes: the element read at x, y and w is
    /// declared as two parts in its register, the rows take the order in
    /// which the measured rule's vertex is smallest, and the pixel shader,
    /// which lists its inputs in another order, declares the same. The
    /// first shape then fits Metal's 256 bytes a vertex exactly (the probe
    /// measured this very signature: `mesh_size::MEASURED`); the second
    /// is 260 and cannot fit: the position, the 16 bytes clip distances
    /// add, the distances and its 53 read components are 260 bytes with
    /// nothing wasted.
    #[test]
    fn fitted_rows_follow_the_measured_rule() {
        use crate::emit::mesh_probe_tests::line;
        let finish = |keep: &KeptOutputs, mut els: Vec<SigElement>| -> Vec<SigElement> {
            prune_signature(&mut els, keep);
            merge_rows(&mut els, keep);
            renumber_rows(&mut els, keep);
            els
        };
        let plan = |seven: bool, level: u8| {
            let (outputs, mut keep) = ground_shape(seven);
            plan_merge(&mut keep, &outputs, false, true, level >= 4, level == 5, level >= 6, (level == 7).then_some(256), &[]);
            let mesh = finish(&keep, outputs.clone());
            (outputs, keep, mesh)
        };
        // As the game measured them at levels 3, 4 and 5.
        for (seven, sizes) in [(false, [264, 272, 272]), (true, [272, 272, 272])] {
            for (level, size) in (3..).zip(sizes) {
                assert_eq!(mesh_size::struct_bytes(&plan(seven, level).2), size, "level {level}");
            }
        }
        let (outputs, keep, mesh) = plan(false, 6);
        assert_eq!(line(&mesh), FITTED, "the signature the probe measured");
        assert_eq!(mesh_size::struct_bytes(&mesh), 256);
        // The parts: x and y, and w, in one register.
        let parts: Vec<(u32, u8, u8, Option<u32>)> =
            mesh.iter().filter(|e| e.name == PART_NAME).map(|e| (e.index, e.col, e.cols, e.row)).collect();
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!((parts[0].0, parts[0].1, parts[0].2), (60, 0, 2));
        assert_eq!((parts[1].0, parts[1].1, parts[1].2), (63, 3, 1));
        assert_eq!(parts[0].3, parts[1].3);
        // Every register from 0 to 17 once, but for the parts' twice.
        let mut registers: Vec<u32> = mesh.iter().filter_map(|e| e.row).collect();
        registers.sort_unstable();
        registers.dedup();
        assert_eq!(registers, (0..18).collect::<Vec<u32>>());
        // The pixel shader: its inputs backwards, without the position.
        let pixel: Vec<SigElement> = outputs[1..].iter().rev().cloned().collect();
        let shape = |els: &[SigElement]| -> Vec<(String, u32, Option<u32>, u8, u8)> {
            els.iter().map(|e| (e.name.clone(), e.index, e.row, e.col, e.cols)).collect()
        };
        assert_eq!(shape(&finish(&keep, pixel)), shape(&mesh[1..]));
        // The second shape: smaller than at any level before, not enough.
        let (_, _, mesh) = plan(true, 6);
        assert_eq!(line(&mesh), UNFITTED, "the signature the probe measured");
        assert_eq!(mesh_size::struct_bytes(&mesh), 260);
    }

    /// Level 7 on the two ground shapes: the first fits at level 6 and is
    /// made as there; the second gives up the clipper, its distances an
    /// ordinary element in both stages (also in a pixel shader that did
    /// not declare them), and is 244 bytes by the rule and by the probe.
    #[test]
    fn a_pair_still_over_gives_up_the_clipper() {
        use crate::emit::mesh_probe_tests::line;
        let made = |seven: bool, level: u8| {
            let (outputs, mut keep) = ground_shape(seven);
            plan_merge(&mut keep, &outputs, false, true, true, false, true, (level == 7).then_some(256), &[]);
            let finish = |mut els: Vec<SigElement>, pixel: bool| {
                prune_signature(&mut els, &keep);
                merge_rows(&mut els, &keep);
                if pixel {
                    soft_clip_inputs(&mut els, &keep);
                }
                renumber_rows(&mut els, &keep);
                els
            };
            let mesh = finish(outputs.clone(), false);
            let pixel = finish(outputs[1..].iter().rev().cloned().collect(), true);
            // A pixel shader that declares no clip distances.
            let bare = finish(outputs[2..].to_vec(), true);
            (mesh, pixel, bare)
        };
        let shape = |els: &[SigElement]| -> Vec<(String, u32, u8, Option<u32>, u8, u8)> {
            let mut s: Vec<_> = els.iter().map(|e| (e.name.clone(), e.index, e.kind, e.row, e.col, e.cols)).collect();
            s.sort();
            s
        };
        let (mesh, ..) = made(false, 7);
        assert_eq!(line(&mesh), FITTED);
        assert!(mesh.iter().any(|e| e.kind == 6));
        let (mesh, pixel, bare) = made(true, 7);
        assert_eq!(line(&mesh), SOFT, "the signature the probe measured");
        assert_eq!(mesh_size::struct_bytes(&mesh), 244);
        assert!(!mesh.iter().any(|e| e.kind == 6));
        let clip: Vec<&SigElement> = mesh.iter().filter(|e| e.name == CLIP_NAME).collect();
        assert_eq!(clip.len(), 1);
        assert_eq!((clip[0].index, clip[0].kind, clip[0].cols, clip[0].interp), (4, 0, 4, 2));
        assert_eq!(distance(clip[0]), Some((0, 4)));
        assert_eq!(shape(&pixel), shape(&mesh[1..]));
        assert_eq!(shape(&bare), shape(&mesh[1..]));
        // Level 6 keeps the clipper whatever the size.
        assert!(made(true, 6).0.iter().any(|e| e.kind == 6));
    }

    /// Levels 7 and 8 on the two ground shapes, whose programs write three
    /// of the four clip distances as the constant 0: both stages declare
    /// the one that can clip, the clipper stays, and both shapes fit (244
    /// and 248 bytes, by the rule and by the probe). Nothing is left for
    /// the pixel shader to clip. With no distance that can clip the
    /// element goes, and the clipper's 16 bytes with it.
    #[test]
    fn clip_distances_that_never_clip_are_left_out() {
        use crate::emit::mesh_probe_tests::line;
        let made = |seven: bool, live: u8| {
            let (outputs, mut keep) = ground_shape(seven);
            plan_merge(&mut keep, &outputs, false, true, true, false, true, Some(256), &[(1, live)]);
            let finish = |mut els: Vec<SigElement>, pixel: bool| {
                prune_signature(&mut els, &keep);
                merge_rows(&mut els, &keep);
                if pixel {
                    soft_clip_inputs(&mut els, &keep);
                }
                renumber_rows(&mut els, &keep);
                els
            };
            (finish(outputs.clone(), false), finish(outputs[1..].iter().rev().cloned().collect(), true))
        };
        let shape = |els: &[SigElement]| -> Vec<(String, u32, u8, Option<u32>, u8, u8)> {
            let mut s: Vec<_> = els.iter().map(|e| (e.name.clone(), e.index, e.kind, e.row, e.col, e.cols)).collect();
            s.sort();
            s
        };
        for (seven, text, bytes) in [(false, ONE_FITTED, 244), (true, ONE_SECOND, 248)] {
            let (mesh, pixel) = made(seven, 0b0001);
            assert_eq!(line(&mesh), text, "the signature the probe measured");
            assert_eq!(mesh_size::struct_bytes(&mesh), bytes);
            let clip: Vec<&SigElement> = mesh.iter().filter(|e| e.kind == 6).collect();
            assert_eq!(clip.len(), 1);
            assert_eq!((clip[0].name.as_str(), clip[0].index, clip[0].col, clip[0].cols), ("SV_ClipDistance", 0, 0, 1));
            assert!(!mesh.iter().any(|e| e.name == CLIP_NAME));
            assert_eq!(shape(&pixel), shape(&mesh[1..]));
        }
        // No distance can clip: 16 + 53 components and nothing else.
        let (mesh, pixel) = made(true, 0);
        assert!(!mesh.iter().chain(&pixel).any(|e| e.kind == 6 || e.name == CLIP_NAME));
        assert_eq!(mesh_size::struct_bytes(&mesh), 16 + 4 * 53);
    }

    /// The ground shapes with one clip distance, as levels 7 and 8 make them.
    const ONE_FITTED: &str =
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C1@17";
    const ONE_SECOND: &str =
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f2@7 f1@7.3 C1@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17";

    /// The second ground shape as level 7 made it before clip distances
    /// that never clip were left out (and makes a shape with four live ones).
    const SOFT: &str =
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f3@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f4@14 f4@15 f2@16 f2@17 f1@17.3";

    /// The first ground shape as level 6 makes it, in the probe's notation.
    const FITTED: &str =
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C4@17";
    /// The second.
    const UNFITTED: &str =
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 C4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 f2@17 f1@17.3";

    /// The order found is the smallest of all orders, rows that stay are
    /// left in their slots, and equal rows keep their order.
    #[test]
    fn the_fitted_order_is_the_smallest() {
        let size = laid_out;
        // Every order of up to seven rows, by Heap's algorithm.
        fn orders(n: usize) -> Vec<Vec<usize>> {
            fn go(k: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
                if k <= 1 {
                    out.push(a.clone());
                    return;
                }
                for i in 0..k {
                    go(k - 1, a, out);
                    a.swap(if k % 2 == 0 { i } else { 0 }, k - 1);
                }
            }
            let mut out = Vec::new();
            go(n, &mut (0..n).collect(), &mut out);
            out
        }
        let cases: [(u32, Vec<Option<Vec<u8>>>, Vec<Vec<u8>>); 5] = [
            (16, vec![None; 4], vec![vec![4], vec![3], vec![3], vec![2]]),
            (32, vec![None; 7], vec![vec![4], vec![4], vec![3], vec![3], vec![3], vec![2, 1], vec![2]]),
            (32, vec![None, None, Some(vec![1, 1]), None, None, None], vec![vec![4], vec![3], vec![3], vec![1], vec![2]]),
            (16, vec![Some(vec![3]), None, None, None], vec![vec![4], vec![4], vec![1]]),
            (0, vec![None; 6], vec![vec![3], vec![4], vec![1], vec![2], vec![3], vec![4]]),
        ];
        for (start, slots, free) in cases {
            let order = fitted_order(start, &slots, &free);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..free.len()).collect::<Vec<usize>>(), "{order:?}");
            let least = orders(free.len()).iter().map(|o| size(start, &slots, &free, o)).min().expect("an order");
            assert_eq!(size(start, &slots, &free, &order), least, "{start} {slots:?} {free:?}: {order:?}");
            // Equal rows in their given order.
            for (i, a) in order.iter().enumerate() {
                for b in &order[i + 1..] {
                    assert!(free[*a] != free[*b] || a < b, "{order:?}");
                }
            }
        }
        // Rows that already waste nothing stay as they are given.
        assert_eq!(fitted_order(16, &[None, None, None], &[vec![4], vec![4], vec![2]]), vec![0, 1, 2]);
    }
}
