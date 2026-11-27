//! The pure half of a title-memory page table (victory-gpu's `memory`
//! owns the GPU it is written to): translated shaders that read or write
//! memory at addresses they compute look each address up in a two-level
//! table — per 64 KiB page below 2^40, the heap index of a pair of raw
//! views of the host copy holding the page and the low dword of that
//! copy's title address. Block 0 is the null one: pages it owns belong
//! to no copy and read as missing.

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
pub const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
pub const LEVEL1: usize = 4096;
/// Level-2 blocks a table has room for besides the null one.
pub const CAPACITY: usize = 512;

/// The working buffers of a page-table build, kept between builds (grown
/// as needed, never shrunk): the copies' visiting order, the level-1
/// index, and the table's dwords. Each entry's copy is built straight
/// into [`Outputs::owners`].
#[derive(Default)]
pub struct Scratch {
    order: Vec<usize>,
    level1: Vec<u32>,
    dwords: Vec<u32>,
}

/// What a build fills, borrowed from the scratch and the outputs it
/// filled: the table's dwords, the level-1 index of each block (block 0
/// the null one), the copy each entry names, and whether the copies'
/// regions did not all fit (the rest read as missing).
pub struct Built<'scratch, 'out> {
    pub dwords: &'scratch [u32],
    pub blocks: &'out [usize],
    pub owners: &'out [Option<usize>],
    pub full: bool,
}

/// The outputs a build fills, kept between builds and handed to whoever
/// reads the table: the level-1 index of each block, and the copy each
/// entry names.
#[derive(Default)]
pub struct Outputs {
    pub blocks: Vec<usize>,
    pub owners: Vec<Option<usize>>,
}

impl Scratch {
    /// Fills `out` for host copies of title memory `copies` (start,
    /// bytes), copy k's view pair at `pairs[k]`, `null` the null pair:
    /// the table's dwords, the level-1 index of each block, and the copy
    /// each entry names. Copies a page's larger owner keeps the page
    /// from; one holding part of it only takes a free page. At most
    /// [`CAPACITY`] blocks besides the null one; past that the smallest
    /// copies read as missing (`full`).
    pub fn build<'scratch, 'out>(
        &'scratch mut self,
        copies: &[(u64, u64)],
        pairs: &[u32],
        null: u32,
        out: &'out mut Outputs,
    ) -> Built<'scratch, 'out> {
        self.build_inner(copies, pairs, None, null, out)
    }

    /// Like [`Scratch::build`], but writes `lows[k]` as each copy's base low
    /// dword instead of `copies[k].0 as u32`. When a view pair covers a whole
    /// backing buffer from byte 0 (`FirstElement = 0`) and copy `k` lies at
    /// `base` inside that buffer, passing `lows[k] = start.wrapping_sub(base) as u32`
    /// makes the shader's `addr_lo.wrapping_sub(lows[k])` evaluate directly to
    /// `base + (addr - start)`.
    pub fn build_with_lows<'scratch, 'out>(
        &'scratch mut self,
        copies: &[(u64, u64)],
        pairs: &[u32],
        lows: &[u32],
        null: u32,
        out: &'out mut Outputs,
    ) -> Built<'scratch, 'out> {
        debug_assert_eq!(copies.len(), lows.len());
        self.build_inner(copies, pairs, Some(lows), null, out)
    }

    fn build_inner<'scratch, 'out>(
        &'scratch mut self,
        copies: &[(u64, u64)],
        pairs: &[u32],
        lows: Option<&[u32]>,
        null: u32,
        out: &'out mut Outputs,
    ) -> Built<'scratch, 'out> {
        self.order.clear();
        self.order.extend(0..copies.len());
        self.order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
        self.level1.clear();
        self.level1.resize(LEVEL1, 0);
        out.blocks.clear();
        // Block 0 is the null block (no copy).
        out.blocks.push(usize::MAX);
        // Entry `e` of the table (`blocks`-indexed): the copy it names.
        let owners = &mut out.owners;
        owners.clear();
        owners.resize(BLOCK, None);
        let mut full = false;
        for &k in &self.order {
            let (start, len) = copies[k];
            let Some(end) = start.checked_add(len) else {
                continue;
            };
            if len == 0 || end > 1 << 40 {
                continue;
            }
            let (first, last) = (start >> 16, (end - 1) >> 16);
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it (its first or last page,
            // where it starts or ends inside one) only takes it if free.
            let (head, tail) = (start & 0xFFFF != 0, end & 0xFFFF != 0);
            // A block's pages at a time: the pages of a copy in one block
            // are one run of entries.
            let mut page = first;
            while page <= last {
                let l1 = (page >> 12) as usize;
                let upto = last.min(page | 0xFFF);
                if self.level1[l1] == 0 {
                    if out.blocks.len() > CAPACITY {
                        full = true;
                        page = upto + 1;
                        continue;
                    }
                    self.level1[l1] = out.blocks.len() as u32;
                    out.blocks.push(l1);
                    owners.resize(owners.len() + BLOCK, None);
                }
                let base = self.level1[l1] as usize * BLOCK;
                let run =
                    &mut owners[base + (page & 0xFFF) as usize..=base + (upto & 0xFFF) as usize];
                let (mut from, mut to) = (0, run.len());
                if page == first && head {
                    run[0].get_or_insert(k);
                    from = 1;
                }
                if upto == last && tail && to > from {
                    run[to - 1].get_or_insert(k);
                    to -= 1;
                }
                run[from..to].fill(Some(k));
                page = upto + 1;
            }
        }
        // The level-1 index, then a pair and an address dword per entry,
        // written in place: the dwords are most of what a rebuild costs.
        let len = LEVEL1 + 2 * owners.len();
        self.dwords.resize(len, 0);
        let (index, entries) = self.dwords.split_at_mut(LEVEL1);
        index.copy_from_slice(&self.level1);
        let (entries, _) = entries.as_chunks_mut::<2>();
        if let Some(lows) = lows {
            for (entry, o) in entries.iter_mut().zip(owners.iter()) {
                *entry = match *o {
                    Some(k) => [pairs[k], lows[k]],
                    None => [null, 0],
                };
            }
        } else {
            for (entry, o) in entries.iter_mut().zip(owners.iter()) {
                *entry = match *o {
                    Some(k) => [pairs[k], copies[k].0 as u32],
                    None => [null, 0],
                };
            }
        }
        Built {
            dwords: &self.dwords,
            blocks: &out.blocks,
            owners: &out.owners,
            full,
        }
    }
}

/// A fresh page table for host copies of title memory `copies` (start,
/// bytes), copy k's view pair at `pairs[k]`, `null` the null pair: the
/// allocating form of [`Scratch::build`].
pub fn table(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let mut out = Outputs::default();
    let mut scratch = Scratch::default();
    let built = scratch.build(copies, pairs, null, &mut out);
    let dwords = built.dwords.to_vec();
    (dwords, out.blocks, out.owners)
}

#[cfg(test)]
mod tests {
    use super::{BLOCK, CAPACITY, LEVEL1, Outputs, Scratch};

    /// The build as it was, page by page: dwords, blocks, owners, full.
    fn page_by_page(
        copies: &[(u64, u64)],
        pairs: &[u32],
        null: u32,
    ) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>, bool) {
        let mut order: Vec<usize> = (0..copies.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
        let mut level1 = vec![0u32; LEVEL1];
        let mut blocks = vec![usize::MAX];
        let mut owner: Vec<Option<(usize, bool)>> = vec![None; BLOCK];
        let mut full = false;
        for &k in &order {
            let (start, len) = copies[k];
            let Some(end) = start.checked_add(len) else {
                continue;
            };
            if len == 0 || end > 1 << 40 {
                continue;
            }
            for page in start >> 16..=(end - 1) >> 16 {
                let l1 = (page >> 12) as usize;
                if level1[l1] == 0 {
                    if blocks.len() > CAPACITY {
                        full = true;
                        continue;
                    }
                    level1[l1] = blocks.len() as u32;
                    blocks.push(l1);
                    owner.resize(owner.len() + BLOCK, None);
                }
                let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
                let whole = page << 16 >= start && (page + 1) << 16 <= end;
                if whole || owner[entry].is_none() {
                    owner[entry] = Some((k, whole));
                }
            }
        }
        let owners: Vec<Option<usize>> = owner.iter().map(|o| o.map(|o| o.0)).collect();
        let mut dwords = level1.clone();
        for o in &owner {
            dwords.extend(match *o {
                Some((k, _)) => [pairs[k], copies[k].0 as u32],
                None => [null, 0],
            });
        }
        (dwords, blocks, owners, full)
    }

    /// A block at a time, the build names the same copy for every entry
    /// and writes the same dwords as one that walks page by page: copies
    /// that start and end inside a page, share pages, cross blocks, are
    /// empty or reach past 2^40, and more regions than fit, rebuilt into
    /// the same buffers (larger, then smaller).
    #[test]
    fn a_block_at_a_time_builds_what_page_by_page_did() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut scratch, mut out) = (Scratch::default(), Outputs::default());
        let mut overflowed = 0;
        for round in 0..40u64 {
            let (regions, count) = if round % 5 == 4 {
                (2_000, 1_500)
            } else {
                (1 + next() % 40, (next() % 300) as usize)
            };
            let mut copies: Vec<(u64, u64)> = (0..count)
                .map(|_| {
                    let start = ((next() % regions) << 28) + next() % (1 << 28);
                    let len = match next() % 6 {
                        0 => 0,
                        1 => next() % 0x1_0000,
                        2 => (next() % 64) << 16,
                        3 => next() % (600 << 20),
                        _ => next() % (4 << 20),
                    };
                    (start, len)
                })
                .collect();
            copies.push(((1 << 40) - 0x8000, 0x1_0000));
            copies.push((u64::MAX - 4, 16));
            let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 0x40_000 | k).collect();
            let built = scratch.build(&copies, &pairs, 7, &mut out);
            let (dwords, blocks, owners, full) = page_by_page(&copies, &pairs, 7);
            assert_eq!(built.blocks, blocks.as_slice(), "round {round}");
            assert_eq!(built.owners, owners.as_slice(), "round {round}");
            assert_eq!(built.full, full, "round {round}");
            assert!(
                built.dwords == dwords.as_slice(),
                "round {round}: the dwords"
            );
            overflowed += usize::from(full);
        }
        assert_eq!(overflowed, 8, "the rounds past the capacity overflow it");
    }

    #[test]
    fn build_with_lows_maps_buffer_suboffsets_and_wraparound() {
        // Two 64 KiB views sharing one 32 MiB backing buffer at distinct bases,
        // including one where `base > start` so `start.wrapping_sub(base)` wraps.
        let copies = [(0x0004_0000u64, 0x1_0000u64), (0x10_0020_0000u64, 0x2_0000u64)];
        let bases = [0x0100_0000u64, 0x0040_0000u64];
        let pairs = [42u32, 42u32];
        let lows: Vec<u32> = copies
            .iter()
            .zip(&bases)
            .map(|(&(start, _), &base)| start.wrapping_sub(base) as u32)
            .collect();
        let (mut scratch, mut out) = (Scratch::default(), Outputs::default());
        let built = scratch.build_with_lows(&copies, &pairs, &lows, 7, &mut out);
        let look_up = |address: u64| -> (u32, u32) {
            let (lo, hi) = (address as u32, (address >> 32) as u32);
            let l1 = hi << 4 | lo >> 28;
            let block = built.dwords[l1 as usize];
            let entry = block << 12 | (lo >> 16 & 0xFFF);
            let at = LEVEL1 + 2 * entry as usize;
            (built.dwords[at], lo.wrapping_sub(built.dwords[at + 1]))
        };
        assert_eq!(look_up(0x0004_0124), (42, 0x0100_0124));
        assert_eq!(look_up(0x10_0021_8008), (42, 0x0041_8008));
        assert_eq!(look_up(0x0008_0000).0, 7);
    }
}
