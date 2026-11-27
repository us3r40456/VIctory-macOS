use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};

use crate::Converter;

/// Identifies one conversion result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey(blake3::Hash);

impl CacheKey {
    fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(converter.id().as_bytes());
        h.update(&[0]);
        h.update(&converter.version().to_le_bytes());
        h.update(input);
        CacheKey(h.finalize())
    }

    fn to_hex(self) -> String {
        self.0.to_hex().to_string()
    }
}

/// Content-addressed store under the user's cache directory.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache { root })
    }

    fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Returns the cached output for `input`, converting it first if needed.
    ///
    /// Asks for no file's or folder's metadata, which under Wine is a call
    /// to its mount manager: the pipeline workers look up every shader here,
    /// and such calls have hung them (VIctory-macOS issue #50).
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<PathBuf> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if published(&path) {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }
        let output = converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))?;
        let dir = path.parent().expect("hashed path has a parent");
        make_dir(dir)?;
        // Write-then-rename so readers never see partial output.
        let (tmp, mut file) = tempfile_in(dir)?;
        let written = file.write_all(&output);
        drop(file);
        if let Err(err) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(err.into());
        }
        if let Err(err) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            if !published(&path) {
                return Err(err.into());
            }
        }
        Ok(path)
    }
}

/// Whether `path` holds an entry. On Windows the file is opened instead of
/// asked for its metadata (`Path::is_file`), whose volume serial number Wine
/// gets from its mount manager. A folder does not open as a file there.
fn published(path: &Path) -> bool {
    if cfg!(windows) {
        File::open(path).is_ok()
    } else {
        path.is_file()
    }
}

/// Makes folder `dir` and its parents unless it is there. Not
/// `create_dir_all` alone, which asks for the metadata of a folder that is
/// there (`Path::is_dir`): a call to Wine's mount manager, like the ones
/// [`Cache::get_or_convert`] avoids. The cache's 256 folders are there after
/// the first few hundred entries, and the GPU pipeline library makes its
/// folder at every save, on the pipeline workers.
pub fn make_dir(dir: &Path) -> io::Result<()> {
    match std::fs::create_dir(dir) {
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => std::fs::create_dir_all(dir),
        made => made,
    }
}

fn tempfile_in(dir: &Path) -> io::Result<(PathBuf, File)> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    create_tempfile_in(dir, &COUNTER)
}

fn create_tempfile_in(dir: &Path, counter: &AtomicU64) -> io::Result<(PathBuf, File)> {
    // An interrupted writer can leave a file behind, and process IDs are
    // reused. Keep it intact and try a new name rather than failing a miss.
    for _ in 0..128 {
        let n = counter.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".tmp-{}-{n}", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a unique cache staging file after 128 attempts",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Upper {
        version: u32,
        runs: AtomicUsize,
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            "test.upper"
        }
        fn version(&self) -> u32 {
            self.version
        }
        fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_publication_removes_the_staging_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[cfg(windows)]
    #[test]
    fn accepts_entry_published_concurrently_while_open() {
        use std::os::windows::fs::OpenOptionsExt;

        struct RacingUpper {
            cache: Cache,
            held: std::sync::Mutex<Option<std::fs::File>>,
        }

        impl Converter for RacingUpper {
            fn id(&self) -> &'static str {
                "test.upper"
            }
            fn version(&self) -> u32 {
                1
            }
            fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
                let output = input.to_ascii_uppercase();
                let key = CacheKey::new(self, input);
                let path = self.cache.path_for(&key);
                let dir = path.parent().unwrap();
                std::fs::create_dir_all(dir)?;
                std::fs::write(&path, &output)?;
                // FILE_SHARE_READ permits readers but prevents replacement while held.
                *self.held.lock().unwrap() = Some(
                    std::fs::OpenOptions::new()
                        .read(true)
                        .share_mode(1)
                        .open(&path)?,
                );
                let (probe, mut probe_file) = tempfile_in(dir)?;
                probe_file.write_all(&output)?;
                drop(probe_file);
                assert!(
                    std::fs::rename(&probe, &path).is_err(),
                    "the open reader must prevent replacing the published entry"
                );
                std::fs::remove_file(probe)?;
                Ok(output)
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = RacingUpper {
            cache: cache.clone(),
            held: std::sync::Mutex::new(None),
        };
        let path = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"ABC");
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[test]
    fn staging_retries_collisions_without_changing_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let counter = AtomicU64::new(0);
        for n in 0..2 {
            std::fs::write(
                dir.path().join(format!(".tmp-{}-{n}", std::process::id())),
                b"stale",
            )
            .unwrap();
        }

        let (path, mut file) = create_tempfile_in(dir.path(), &counter).unwrap();
        file.write_all(b"fresh").unwrap();
        drop(file);
        assert_eq!(
            path.file_name().unwrap(),
            format!(".tmp-{}-2", std::process::id()).as_str()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"fresh");
        for n in 0..2 {
            assert_eq!(
                std::fs::read(dir.path().join(format!(".tmp-{}-{n}", std::process::id()))).unwrap(),
                b"stale"
            );
        }
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn staging_collision_retries_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let counter = AtomicU64::new(0);
        for n in 0..128 {
            std::fs::write(
                dir.path().join(format!(".tmp-{}-{n}", std::process::id())),
                b"stale",
            )
            .unwrap();
        }

        let error = create_tempfile_in(dir.path(), &counter).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(counter.load(Ordering::Relaxed), 128);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 128);
    }

    #[test]
    fn staging_does_not_retry_other_io_errors() {
        let dir = tempfile::tempdir().unwrap();
        let counter = AtomicU64::new(0);
        let missing = dir.path().join("missing");

        let error = create_tempfile_in(&missing, &counter).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        assert!(!missing.exists());
    }

    #[test]
    fn folders_are_made_once_with_their_parents() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("a").join("b");
        make_dir(&deep).unwrap();
        assert!(deep.is_dir());
        make_dir(&deep).unwrap();
        std::fs::write(deep.join("kept"), b"kept").unwrap();
        make_dir(&deep).unwrap();
        assert_eq!(std::fs::read(deep.join("kept")).unwrap(), b"kept");
    }

    #[test]
    fn only_files_count_as_entries() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        assert!(published(&file));
        assert!(!published(&dir.path().join("missing")));
        assert!(!published(dir.path()));
    }

    #[test]
    fn version_bump_invalidates() {
        let v1 = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            version: 2,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }
}
