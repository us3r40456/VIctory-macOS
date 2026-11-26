//! Measures Windows section-view churn under the current Wine server (#155).
//! Run on its own with the same Wine client and bottle for each server:
//! `tools/macos/test-wine.sh -p victory-hle --release --test wine_views -- --ignored --nocapture`.

#![cfg(windows)]

use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::ptr;
use std::time::Instant;
use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
    UnmapViewOfFile,
};

const VIEW_BYTES: usize = 64 * 1024;
const LIVE_VIEWS: usize = 20_000;
const PAIRS: usize = 1_000;

struct View(MEMORY_MAPPED_VIEW_ADDRESS);

impl View {
    fn map(section: &OwnedHandle) -> io::Result<Self> {
        // SAFETY: the section is live and contains VIEW_BYTES bytes at offset zero.
        let view =
            unsafe { MapViewOfFile(section.as_raw_handle(), FILE_MAP_WRITE, 0, 0, VIEW_BYTES) };
        if view.Value.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(view))
    }

    fn unmap(&mut self) -> io::Result<()> {
        if !self.0.Value.is_null() {
            // SAFETY: this is the base of the view we own.
            if unsafe { UnmapViewOfFile(self.0) } == 0 {
                return Err(io::Error::last_os_error());
            }
            self.0.Value = ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for View {
    fn drop(&mut self) {
        let _ = self.unmap();
    }
}

#[test]
#[ignore = "manual section-view remap timing"]
fn time_remaps_with_twenty_thousand_live_views() -> io::Result<()> {
    // SAFETY: a new anonymous paging-file-backed section, with default security.
    let raw = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            ptr::null(),
            PAGE_READWRITE,
            0,
            VIEW_BYTES as u32,
            ptr::null(),
        )
    };
    if raw.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: CreateFileMappingW returned a new handle owned by this test.
    let section = unsafe { OwnedHandle::from_raw_handle(raw) };
    let mut views = (0..LIVE_VIEWS)
        .map(|_| View::map(&section))
        .collect::<io::Result<Vec<_>>>()?;
    const MARKER: u32 = 0x1234_abcd;
    // SAFETY: each writable view starts at the same aligned section offset.
    unsafe { views[0].0.Value.cast::<u32>().write_volatile(MARKER) };

    for trial in 0..3 {
        let start = Instant::now();
        for j in 0..PAIRS {
            // A stride coprime to LIVE_VIEWS samples throughout the live set.
            let slot = &mut views[((trial * PAIRS + j) * 7919) % LIVE_VIEWS];
            slot.unmap()?;
            *slot = View::map(&section)?;
            // SAFETY: the new view maps the same aligned, initialized word.
            let observed = unsafe { slot.0.Value.cast::<u32>().read_volatile() };
            assert_eq!(observed, MARKER);
        }
        let micros = start.elapsed().as_secs_f64() * 1e6 / PAIRS as f64;
        println!("{LIVE_VIEWS} live views, {PAIRS} checked remaps: {micros:.1} us/pair");
    }
    Ok(())
}
