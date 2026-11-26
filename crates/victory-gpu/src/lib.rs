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
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(any(windows, test))]
mod caption;
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
pub mod query;
pub mod queue;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ranges;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod switch;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;
// Whether the host opens heaps over process memory (ignored by default).
#[cfg(all(windows, test))]
mod inplace_probe;

#[cfg(all(windows, feature = "diagnostics"))]
pub use copystat::CountingAlloc;
#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, fault_tracking, rosetta, stores_need_patching, title_memory_aliased,
    title_memory_remapped, writing as kernel_will_write,
};

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

/// Makes folder `dir` and its parents unless it is there. Not
/// `std::fs::create_dir_all` alone, which on a folder that is there asks for
/// its metadata (`Path::is_dir`): the same mount manager call as in
/// [`read_file`]. The pipeline library makes its folder at every save, on
/// the pipeline workers.
pub fn make_dir(dir: &std::path::Path) -> std::io::Result<()> {
    match std::fs::create_dir(dir) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir_all(dir),
        made => made,
    }
}

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
