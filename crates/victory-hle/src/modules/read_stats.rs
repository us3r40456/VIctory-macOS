//! `VICTORY_READ_STATS=1`: requests and bytes read from each of the title's
//! files, through DirectStorage and `ReadFile`, logged every 30 s, the 30
//! most read first. Which files a slow share serves most (#135), to copy
//! those to fast local storage.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Whether the statistics are kept.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_READ_STATS").is_some_and(|v| v != "0"))
}

/// Requests and bytes by file name.
static STATS: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

/// One read of `bytes` from the file the title named `name`.
pub fn add(name: &str, bytes: u64) {
    static REPORT: std::sync::Once = std::sync::Once::new();
    REPORT.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("read stats".into())
            .spawn(report);
    });
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let e = stats
        .get_or_insert_with(HashMap::new)
        .entry(name.to_owned())
        .or_default();
    e.0 += 1;
    e.1 += bytes;
}

/// One `ReadFile` of `bytes` through host handle `handle`, its file named
/// once per handle.
#[cfg(windows)]
pub fn add_handle(handle: *mut std::ffi::c_void, bytes: u64) {
    static NAMES: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
    let name = {
        let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
        names
            .get_or_insert_with(HashMap::new)
            .entry(handle as usize)
            .or_insert_with(|| {
                let mut path = [0u16; 512];
                // SAFETY: a handle the title passed and our buffer with its length.
                let n = unsafe {
                    windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW(
                        handle as _,
                        path.as_mut_ptr(),
                        path.len() as u32,
                        0,
                    )
                } as usize;
                let full = String::from_utf16_lossy(&path[..n.min(path.len())]);
                // From the title folder on, as DirectStorage names them.
                match full.to_ascii_lowercase().find("game-files\\") {
                    Some(at) => full[at + "game-files\\".len()..].to_owned(),
                    None => full,
                }
            })
            .clone()
    };
    add(&format!("ReadFile {name}"), bytes);
}

fn report() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        let mut rows: Vec<(String, (u64, u64))> = {
            let stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
            stats
                .iter()
                .flatten()
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        };
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.1));
        let total: u64 = rows.iter().map(|r| r.1.1).sum();
        tracing::info!(
            files = rows.len(),
            mb = total >> 20,
            "read stats: most read files so far"
        );
        for (name, (requests, bytes)) in rows.iter().take(30) {
            tracing::info!(mb = bytes >> 20, requests, "read stats: {name}");
        }
    }
}
