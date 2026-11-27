//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod ambient;
#[cfg(windows)]
mod ambient_estimate;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ambient_estimate_set;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ambient_set;
#[cfg(windows)]
mod alias;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod alias_plan;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod batch;
#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(any(windows, test))]
mod caption;
#[cfg(any(windows, test))]
mod census_fast;
#[cfg(any(windows, test))]
mod census_maps;
#[cfg(windows)]
mod composite;
#[cfg(windows)]
mod compute;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
// mac: plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod flights;
pub mod frame;
mod hold;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(all(windows, test))]
mod list_bench;
#[cfg(windows)]
mod memory;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod placement;
#[cfg(windows)]
mod primitive;
// mac: a sampling profile of the process's threads (`VICTORY_PROFILE`).
pub mod profile;
pub mod query;
pub mod queue;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod physical_runs;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ranges;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ray_skip;
#[cfg(any(windows, test))]
mod recent;
// The capture format is plain std: read, and tested, everywhere.
pub mod replay;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod mac_switches;
mod switch;
#[cfg(windows)]
mod tdr;
// Its rules are plain std, so their tests run everywhere.
#[cfg(any(windows, test))]
mod texback;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod widened;
// Whether the host opens heaps over process memory (ignored by default).
#[cfg(all(windows, test))]
mod inplace_probe;

/// mac: the title's descriptor stores report the slots they write
/// (`VICTORY_HEAP_REPORTS`, `descriptor::reports`): what the launcher and
/// the import resolver need of it.
#[cfg(windows)]
pub mod heap_reports {
    pub use crate::descriptor::reports::{MARKS_LEN, Mode, hooked, import, marks_address, mode, wanted};
}

#[cfg(all(windows, feature = "diagnostics"))]
pub use copystat::CountingAlloc;
#[cfg(windows)]
pub use alias::{
    alias_region, committed as alias_committed, enabled as alias_buffers, unalias_region,
    uncommitted as alias_uncommitted,
};
#[cfg(all(windows, feature = "diagnostics"))]
pub use alias::{PhysicalHeap, physical_heap};
#[cfg(windows)]
pub use host::{MeshSupport, host_mesh_support, on_failure};
#[cfg(windows)]
pub use physical_runs::{mapped as physical_mapped, unmapped as physical_unmapped};
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, fault_tracking, rosetta, stores_need_patching, title_memory_aliased,
    title_memory_remapped, writing as kernel_will_write,
};

/// [`shader_cache::under_wine`], asked once: it looks `wine_get_version` up
/// in ntdll, which takes Wine's loader lock, and draws ask it per descriptor,
/// barrier and heap write. Not in `shader_cache`, whose bytes key the
/// translation caches.
#[cfg(windows)]
pub(crate) fn under_wine() -> bool {
    static WINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WINE.get_or_init(shader_cache::under_wine)
}

/// A whole file, read without asking for its size or other metadata.
/// `std::fs::read` and `read_to_end` call `File::metadata()`, which Wine
/// answers with `NtQueryVolumeInformationFile`: an ioctl to its mount
/// manager whose completion runs as a system APC inside Wine's virtual
/// memory lock. Under Rosetta a thread was paused there by SIGUSR1 with
/// the lock held, and every other thread then waited on it for good (#50).
/// The shader cache read files this way on the pipeline workers.
pub fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        match file.read(&mut chunk) {
            Ok(0) => return Ok(out),
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

#[cfg(windows)]
pub use victory_pipeline::make_dir;

#[cfg(test)]
mod read_file_tests {
    #[test]
    fn whole_files_come_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        for len in [0usize, 1, 65_535, 65_536, 65_537, 300_000] {
            let path = dir.path().join(format!("f{len}"));
            let data: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            std::fs::write(&path, &data).unwrap();
            assert_eq!(super::read_file(&path).unwrap(), data, "{len} bytes");
        }
        assert!(super::read_file(&dir.path().join("missing")).is_err());
    }
}
