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
/// index, each entry's copy and whether it holds the entry's page whole,
/// and the table's dwords.
#[derive(Default)]
pub struct Scratch {
    order: Vec<usize>,
    level1: Vec<u32>,
    /// Entry `k` of a built table (`k` is `blocks`-indexed): the copy it
    /// names, and whether that copy holds its page whole.
    owner: Vec<Option<(usize, bool)>>,
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
        self.order.clear();
        self.order.extend(0..copies.len());
        self.order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
        self.level1.clear();
        self.level1.resize(LEVEL1, 0);
        out.blocks.clear();
        // Block 0 is the null block (no copy).
        out.blocks.push(usize::MAX);
        self.owner.clear();
        self.owner.resize(BLOCK, None);
        let mut full = false;
        for &k in &self.order {
            let (start, len) = copies[k];
            let Some(end) = start.checked_add(len) else {
                continue;
            };
            if len == 0 || end > 1 << 40 {
                continue;
            }
            for page in start >> 16..=(end - 1) >> 16 {
                let l1 = (page >> 12) as usize;
                if self.level1[l1] == 0 {
                    if out.blocks.len() > CAPACITY {
                        full = true;
                        continue;
                    }
                    self.level1[l1] = out.blocks.len() as u32;
                    out.blocks.push(l1);
                    self.owner.resize(self.owner.len() + BLOCK, None);
                }
                let entry = self.level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
                // Larger copies come first: a copy holding the whole page
                // takes it from them; one holding part of it only takes
                // it if free.
                let whole = page << 16 >= start && (page + 1) << 16 <= end;
                if whole || self.owner[entry].is_none() {
                    self.owner[entry] = Some((k, whole));
                }
            }
        }
        out.owners.clear();
        out.owners.extend(self.owner.iter().map(|o| o.map(|o| o.0)));
        self.dwords.clear();
        self.dwords.extend(self.level1.iter().copied());
        for o in &self.owner {
            let entry = match *o {
                Some((k, _)) => [pairs[k], copies[k].0 as u32],
                None => [null, 0],
            };
            self.dwords.extend(entry);
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
