//! Portable publication of derived files into a shared cache directory.
//!
//! Several copies of the runner can prepare the same title at once. Each
//! writer stages into its own file, so two of them cannot truncate each
//! other's bytes or rename a file the other already published. The final
//! rename is atomic within the directory, and the staging file is removed
//! if publication fails.
//!
//! A rename can be refused with nothing wrong: Wine does not replace a file
//! that is open, and when two writers race, the second rename can find the
//! first writer's file still open. A writer whose rename fails therefore
//! looks at the destination again, and has nothing left to do when its
//! bytes are already there.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` unless it is already there, using a staging file
/// unique to this process and call.
pub fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if holds(path, bytes) {
        return Ok(());
    }
    publish(path, bytes)
}

/// Whether `path` can be read and holds exactly `bytes`.
fn holds(path: &Path, bytes: &[u8]) -> bool {
    std::fs::read(path).is_ok_and(|existing| existing == bytes)
}

/// Stages `bytes` next to `path` and renames the staging file over it. A
/// refused rename still counts when `path` holds `bytes` by then, because
/// another writer has published the same bytes.
fn publish(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let sequence = NEXT_STAGING.fetch_add(1, Ordering::Relaxed);
    let staging = path.with_file_name(format!(
        ".{name}.{}.{sequence}.tmp",
        std::process::id()
    ));
    std::fs::write(&staging, bytes)?;
    if let Err(error) = std::fs::rename(&staging, path) {
        let _ = std::fs::remove_file(&staging);
        if holds(path, bytes) {
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two publishers staging the same destination at once both succeed,
    /// and neither leaves a staging file behind.
    #[test]
    fn concurrent_publishers_do_not_share_a_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let bytes = b"a prepared image".to_vec();
        let (first, second) = (path.clone(), path.clone());
        let (first_bytes, second_bytes) = (bytes.clone(), bytes.clone());
        let (one, two) = (
            std::thread::spawn(move || write_if_changed(&first, &first_bytes)),
            std::thread::spawn(move || write_if_changed(&second, &second_bytes)),
        );
        one.join().unwrap().unwrap();
        two.join().unwrap().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != path)
            .collect();
        assert!(leftovers.is_empty(), "staging files were removed: {leftovers:?}");
    }

    /// The publisher that loses that race: the destination already holds
    /// its bytes and is open, which is when Wine refuses the rename. It
    /// still succeeds, and removes its staging file.
    #[test]
    fn a_rename_onto_an_open_file_with_the_same_bytes_is_a_publication() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let bytes = b"a prepared image";
        std::fs::write(&path, bytes).unwrap();
        let open = std::fs::File::open(&path).unwrap();
        publish(&path, bytes).unwrap();
        drop(open);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != path)
            .collect();
        assert!(
            leftovers.is_empty(),
            "staging files were removed: {leftovers:?}"
        );
    }

    #[test]
    fn a_failed_publication_removes_its_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing-parent").join("image.bin");
        assert!(write_if_changed(&path, b"bytes").is_err());
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert!(leftovers.is_empty(), "no staging file survives an error");
    }
}
