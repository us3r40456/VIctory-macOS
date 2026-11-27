//! A table of optional entries that gets its memory a page at a time, for
//! what a host heap keeps per slot (`HostHeap::kept`): a view heap has over
//! two million slots, and a title views host resources through a fraction of
//! them. A dense table writes every slot when the heap is made; this one
//! touches the memory of the slots that hold something.

/// Slots to a page: 4 KiB of the 16-byte entries a host heap keeps. A title
/// writes its slots in runs (tables) that can sit anywhere in the heap, and
/// a run takes the pages it touches whole, so a smaller page leaves more of
/// the heap unmade (runs of 64 slots at random places, 130,000 slots in a
/// 2.36 million slot heap: 7.7 MB at 256 slots, 13.9 at 512, 22.7 at
/// 1,024; slots in one stretch take the same at any size). A smaller page
/// also makes the table of pages, 16 bytes a page and made with the heap,
/// larger: 147 KB at 256 slots, 590 KB at 64.
pub(super) const PAGE: usize = 256;

/// `len` slots, each empty or holding a `T`. A slot's page is made when
/// the slot first holds something, and stays made: what a slot holds is
/// released as in a dense table, by replacing it or with the table, and in
/// the same order (by slot).
pub(crate) struct Paged<T> {
    pages: Vec<Option<Box<[Option<T>]>>>,
    len: usize,
}

impl<T> Paged<T> {
    /// `len` empty slots, with no page made.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            pages: std::iter::repeat_with(|| None)
                .take(len.div_ceil(PAGE))
                .collect(),
            len,
        }
    }

    /// The slots there are (mac: for `super::slots`' tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// What slot `index` holds; none for an empty slot, a page not made, or
    /// a slot past the end.
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        self.pages
            .get(index / PAGE)?
            .as_ref()?
            .get(index % PAGE)?
            .as_ref()
    }

    /// The slots that hold something, in order, with their indices: the
    /// pages made are walked, the others skipped.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.pages.iter().enumerate().flat_map(|(p, page)| {
            page.iter()
                .flat_map(|slots| slots.iter().enumerate())
                .filter_map(move |(i, slot)| slot.as_ref().map(|t| (p * PAGE + i, t)))
        })
    }

    /// Slot `index`, to write; none past the end. Its page is made only if
    /// `make` (the write is of something): with no page the slot is empty,
    /// and writing nothing there is no change, so none is returned.
    pub(crate) fn slot(&mut self, index: usize, make: bool) -> Option<&mut Option<T>> {
        if index >= self.len {
            return None;
        }
        let page = &mut self.pages[index / PAGE];
        if page.is_none() && !make {
            return None;
        }
        let first = index - index % PAGE;
        let page = page.get_or_insert_with(|| {
            std::iter::repeat_with(|| None)
                .take(PAGE.min(self.len - first))
                .collect()
        });
        Some(&mut page[index % PAGE])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn made<T>(table: &Paged<T>) -> usize {
        table.pages.iter().flatten().count()
    }

    #[test]
    fn a_table_makes_pages_only_for_what_it_holds() {
        let mut table = Paged::<u32>::new(10 * PAGE);
        assert_eq!(made(&table), 0);
        assert_eq!(table.get(5), None);
        // Writing nothing where nothing is held makes no page.
        assert!(table.slot(5, false).is_none());
        assert_eq!(made(&table), 0);
        *table.slot(5, true).unwrap() = Some(7);
        *table.slot(7 * PAGE + 3, true).unwrap() = Some(9);
        assert_eq!(made(&table), 2);
        assert_eq!(
            (table.get(5), table.get(7 * PAGE + 3)),
            (Some(&7), Some(&9))
        );
        // Slots of a made page are empty until written, as are the rest.
        assert_eq!(
            (table.get(6), table.get(PAGE), table.get(8 * PAGE)),
            (None, None, None)
        );
        // A page that is made takes writes of nothing: the old entry comes back.
        assert_eq!(table.slot(5, false).unwrap().take(), Some(7));
        assert_eq!(table.get(5), None);
        assert_eq!(made(&table), 2);
        // Replacing returns the old entry.
        let slot = table.slot(7 * PAGE + 3, true).unwrap();
        assert_eq!(slot.replace(10), Some(9));
        assert_eq!(table.get(7 * PAGE + 3), Some(&10));
    }

    #[test]
    fn slots_past_the_end_hold_nothing_and_make_no_page() {
        // The last page is short: its slots past the end do not exist.
        let end = 2 * PAGE + 10;
        let mut table = Paged::<u32>::new(end);
        assert!(table.slot(end, true).is_none());
        assert!(table.slot(end + 1, true).is_none());
        assert!(table.slot(3 * PAGE, true).is_none());
        assert!(table.slot(usize::MAX, true).is_none());
        assert_eq!(made(&table), 0);
        assert_eq!(table.pages[2], None);
        *table.slot(end - 1, true).unwrap() = Some(1);
        assert_eq!(table.pages[2].as_ref().map(|p| p.len()), Some(10));
        assert_eq!(
            (table.get(end - 1), table.get(end), table.get(usize::MAX)),
            (Some(&1), None, None)
        );
        // No slots, no pages.
        let mut none = Paged::<u32>::new(0);
        assert!(none.slot(0, true).is_none());
        assert_eq!(none.get(0), None);
    }

    /// Releases what a dense table's `Vec` does, in the same order.
    #[test]
    fn a_table_drops_its_entries_in_slot_order() {
        struct Entry(usize, Arc<Mutex<Vec<usize>>>);
        impl Drop for Entry {
            fn drop(&mut self) {
                self.1.lock().unwrap().push(self.0);
            }
        }
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let mut table = Paged::new(5 * PAGE);
        let slots = [4 * PAGE + 1, 3, 2 * PAGE, PAGE + 5, 2];
        for &i in &slots {
            *table.slot(i, true).unwrap() = Some(Entry(i, dropped.clone()));
        }
        // Replaced by an entry that drops when the table does.
        let old = table
            .slot(3, true)
            .unwrap()
            .replace(Entry(3, dropped.clone()));
        assert!(dropped.lock().unwrap().is_empty());
        drop(old);
        assert_eq!(*dropped.lock().unwrap(), [3]);
        drop(table);
        assert_eq!(
            *dropped.lock().unwrap(),
            [3, 2, 3, PAGE + 5, 2 * PAGE, 4 * PAGE + 1]
        );
    }
}
