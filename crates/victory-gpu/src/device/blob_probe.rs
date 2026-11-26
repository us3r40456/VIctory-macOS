//! Compare two immediate reads of the serialized PSO description and its first
//! two contiguous blocks. The third compute address has no known length.
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use windows_sys::Win32::Foundation::{GetLastError, SetLastError};

const SAMPLES_PER_SLOT: u32 = 16;
const MAX_BLOCK: usize = 4096;

fn read(address: usize, buffer: &mut [u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut count = 0;
    // SAFETY: the OS checks the source in our process; the slice owns the
    // destination. Discard the whole read unless every byte was copied.
    unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            address as *const c_void,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            &mut count,
        ) != 0
            && count == buffer.len()
    }
}

struct Sample {
    blocks: Vec<u8>,
    split: usize,
}

fn sample(
    address: usize,
    mut read: impl FnMut(usize, &mut [u8]) -> bool,
) -> Result<Sample, &'static str> {
    let mut header = [0u8; 64];
    if address == 0 || address.checked_add(header.len()).is_none() || !read(address, &mut header) {
        return Err("unreadable description");
    }
    let word = |offset| {
        let value = u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
        usize::try_from(value).map_err(|_| "word out of range")
    };
    let first = word(8)?;
    let first_len = word(16)?;
    let second = word(24)?;
    let second_len = word(32)?;
    if first_len == 0 || second_len == 0 || first_len > MAX_BLOCK || second_len > MAX_BLOCK {
        return Err("block size out of range");
    }
    if first == 0
        || first.checked_add(first_len) != Some(second)
        || second.checked_add(second_len).is_none()
    {
        return Err("invalid block range");
    }

    // One read covers both blocks. Re-read the same addresses, then the
    // description itself. Equality is a repeated observation, not a lock.
    let mut blocks = vec![0; first_len + second_len];
    let mut again = vec![0; blocks.len()];
    if !read(first, &mut blocks) || !read(first, &mut again) {
        return Err("unreadable blocks");
    }
    let mut end = [0; 64];
    if !read(address, &mut end) {
        return Err("unreadable description");
    }
    if end != header || blocks != again {
        return Err("changed between reads");
    }
    Ok(Sample {
        blocks,
        split: first_len,
    })
}

fn reserve(counter: &AtomicU32) -> Option<u32> {
    counter
        .fetch_update(Relaxed, Relaxed, |n| {
            (n < SAMPLES_PER_SLOT).then_some(n + 1)
        })
        .ok()
        .map(|n| n + 1)
}

pub(super) fn observe(slot: usize, desc: *const c_void) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static GRAPHICS: AtomicU32 = AtomicU32::new(0);
    static COMPUTE: AtomicU32 = AtomicU32::new(0);
    if ENABLED.get() == Some(&false) {
        return;
    }
    // SAFETY: this is the caller's thread-local Win32 error. The environment
    // lookup, failed memory reads and tracing may each change it.
    let error = unsafe { GetLastError() };
    if *ENABLED.get_or_init(|| crate::switch::on("VICTORY_PSO_BLOB_HASHES")) {
        let counter = match slot {
            0x1c8 => &GRAPHICS,
            0x218 => &COMPUTE,
            _ => unreachable!("only deserialize slots take an opaque description"),
        };
        if let Some(index) = reserve(counter) {
            let slot = format_args!("{slot:#x}");
            match sample(desc as usize, read) {
                Ok(s) => {
                    let (packet, metadata) = s.blocks.split_at(s.split);
                    let packet_zero = packet.iter().all(|&b| b == 0);
                    let metadata_zero = metadata.iter().all(|&b| b == 0);
                    tracing::info!(
                        %slot, index, outcome = if packet_zero && metadata_zero { "all_zero" } else { "stable" },
                        packet_bytes = packet.len(), packet_blake3 = %blake3::hash(packet),
                        metadata_bytes = metadata.len(), metadata_blake3 = %blake3::hash(metadata),
                        packet_zero, metadata_zero, "console PSO block sample"
                    );
                }
                Err(reason) => {
                    tracing::info!(%slot, index, outcome = "inconclusive", reason, "console PSO block sample")
                }
            }
        }
    }
    // SAFETY: restore exactly the error present on entry.
    unsafe { SetLastError(error) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(ptr: usize, first: usize, second: usize) -> [u64; 8] {
        [
            0,
            ptr as u64,
            first as u64,
            ptr.saturating_add(first) as u64,
            second as u64,
            0,
            0,
            u64::MAX,
        ]
    }

    #[test]
    fn copied_blocks_preserve_the_packet_boundary() {
        for (first, second) in [(28, 20), (380, 188), (504, 80)] {
            let blocks: Vec<_> = (0..first + second).map(|i| i as u8).collect();
            let desc = header(blocks.as_ptr() as usize, first, second);
            let s = sample(desc.as_ptr() as usize, read).unwrap();
            assert_eq!(s.blocks.split_at(s.split), blocks.split_at(first));
        }
    }

    #[test]
    fn unreadable_and_unbounded_inputs_fail_without_following_other_pointers() {
        assert!(sample(1, read).is_err());
        assert!(sample(usize::MAX, read).is_err());
        for (ptr, first, second) in [
            (1, 28, 20),
            (1, MAX_BLOCK + 1, 20),
            (1, 28, 0),
            (usize::MAX - 28, 28, 1),
        ] {
            let desc = header(ptr, first, second);
            assert!(sample(desc.as_ptr() as usize, read).is_err());
        }
        let mut desc = header(1, 28, 20);
        desc[3] += 1;
        assert!(matches!(
            sample(desc.as_ptr() as usize, read),
            Err("invalid block range")
        ));
    }

    #[test]
    fn disagreement_in_either_read_is_inconclusive() {
        let blocks = [0u8; 48];
        let desc = header(blocks.as_ptr() as usize, 28, 20);
        for changed_read in [3, 4] {
            let mut reads = 0;
            let sample = sample(desc.as_ptr() as usize, |src, dst| {
                reads += 1;
                let ok = read(src, dst);
                if reads == changed_read {
                    dst[0] ^= 1;
                }
                ok
            });
            assert!(matches!(sample, Err("changed between reads")));
        }
    }

    #[test]
    fn sample_reservations_stop_at_the_slot_limit() {
        let counter = AtomicU32::new(0);
        for n in 1..=SAMPLES_PER_SLOT {
            assert_eq!(reserve(&counter), Some(n));
        }
        assert_eq!(reserve(&counter), None);
        assert_eq!(reserve(&counter), None);
    }

    #[test]
    fn invalid_description_leaves_the_callers_last_error_intact() {
        // SAFETY: the error slot belongs to this test thread.
        unsafe { SetLastError(0x1234) };
        observe(0x218, std::ptr::dangling());
        // SAFETY: reading this thread's error slot.
        assert_eq!(unsafe { GetLastError() }, 0x1234);
    }
}
