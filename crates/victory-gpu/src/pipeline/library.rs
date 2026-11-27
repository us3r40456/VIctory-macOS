//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//!
//!    An entry can also stop loading while its library is still accepted
//!    (one that refers to a cache of the host's own that is gone). Its name
//!    is then refused when the pipeline is made and stored again, so the
//!    library would never hold that pipeline again, and every later run
//!    would make it. Once more of what a run asks of the library is stale
//!    like this than loads (`worn_out`), a new library takes its place: what
//!    the run loads or makes from then on goes into it, and the next save
//!    writes it.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written. Loads take no lock at
//! all, since a load adds nothing to the library: a save holds the lock
//! while it serialises the whole library (hundreds of megabytes in the
//! world), and loads that waited on it stalled every worker for as long.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library
    /// (and `Shared::loads`, the same one, after a new library replaced it).
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// The library for loads, beside the one stores and saves lock.
struct Shared {
    device: ID3D12Device1,
    /// The library as opened, `state`'s until a new one replaces it,
    /// loaded from without its lock.
    loads: ID3D12PipelineLibrary1,
    /// Whether that library held pipelines when it was opened.
    opened_with_pipelines: bool,
    /// Pipelines loaded from it, and made again because one stored under
    /// their name would not load (stale), in this run.
    loaded: AtomicU64,
    stale: AtomicU64,
    /// The library that replaced it (`worn_out`), `state`'s from then on,
    /// loaded from without the lock too.
    fresh: OnceLock<ID3D12PipelineLibrary1>,
    /// Pipelines loaded from the opened library since, for the next save to
    /// store in the new one.
    kept: Mutex<Vec<(HSTRING, ID3D12PipelineState)>>,
    state: Mutex<Library>,
}

// SAFETY: as for `Library`.
unsafe impl Send for Shared {}
// SAFETY: the libraries are free-threaded but for loads of one name, which
// `NAMES` keeps apart; the rest is behind `state`'s and `kept`'s locks.
unsafe impl Sync for Shared {}

/// Pipelines a library must have refused as stale before it is replaced.
const STALE_MIN: u64 = 32;

/// Whether a library that refused `stale` pipelines whose stored entries did
/// not load, and loaded `loaded`, is replaced by a new one: once at least
/// `STALE_MIN` were stale, and more of them than loaded. A few stale entries
/// leave the library as it is, since it also holds the pipelines of places
/// the run has not been to; when most of what the run asked of it did not
/// load, those likely do not load either.
fn worn_out(stale: u64, loaded: u64) -> bool {
    stale >= STALE_MIN && stale > loaded
}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not take the library file's place while the
/// file was mapped (Wine refuses a rename over a mapped file): it takes the
/// place at the next start, before the file is mapped.
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Shared> {
    static LIBRARY: OnceLock<Option<Shared>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| crate::read_file(&d.join(file)).ok());
            let blob = DIR.get().map_or_else(Blob::default, |d| Blob::load(d));
            let tag = tag(blob.bytes());
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(PREBUILT_FILE))
                && !l._blob.bytes().is_empty()
            {
                l.prebuilt = listed(&list, &tag);
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(|l| Shared::new(device, l))
        })
        .as_ref()
}

/// The file's mark: "VPL2", four zero bytes, and then an FNV-1a checksum of
/// the library, 16 bytes in all, so that the library starts 16 bytes into
/// the file and at an address as aligned in a mapping of it. Files of the
/// mark before ("VPL1" and the checksum, 12 bytes) are still read, into
/// memory.
const MAGIC: &[u8; 4] = b"VPL2";
const HEAD: usize = 16;
const OLD_MAGIC: &[u8; 4] = b"VPL1";
const OLD_HEAD: usize = 12;

fn checksum(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The mark that goes in front of the library bytes `blob` in the file. It is
/// not joined to them: a save writes the two one after the other, so a library
/// of hundreds of megabytes is not copied to put the mark in front of it.
fn mark(blob: &[u8]) -> [u8; HEAD] {
    let mut head = [0; HEAD];
    head[..4].copy_from_slice(MAGIC);
    head[8..].copy_from_slice(&checksum(blob).to_le_bytes());
    head
}

/// Where the library bytes of a sealed file start (after a mark of now or
/// of before), or `None` if the mark or checksum is wrong. Not every host
/// checks the bytes itself (D3DMetal accepts any), so a damaged file is
/// refused here before the host sees it.
fn sealed(file: &[u8]) -> Option<usize> {
    let (head, sum) = match file.get(..4)? {
        m if m == MAGIC => (HEAD, file.get(8..HEAD)?),
        m if m == OLD_MAGIC => (OLD_HEAD, file.get(4..OLD_HEAD)?),
        _ => return None,
    };
    let sum = u64::from_le_bytes(sum.try_into().ok()?);
    (sum == checksum(&file[head..])).then_some(head)
}

/// The library bytes of a sealed file read into memory, or `None` if the
/// mark or checksum is wrong. The bytes stay in the file's buffer, with the
/// mark taken off the front of it.
fn unseal(mut file: Vec<u8>) -> Option<Vec<u8>> {
    let head = sealed(&file)?;
    file.drain(..head);
    Some(file)
}

/// Whether the library file is mapped, not read into memory, and saved
/// through a mapping of the file it is saved to (on by default;
/// `VICTORY_LIBRARY_MAP=0`: read and serialised in memory, as before).
///
/// The library is hundreds of megabytes (1.3-1.5 GB measured on Windows by
/// codec-d/cool_fox, whose `VICTORY_LIBRARY_MAP` this follows; about 1.7 GB
/// on NVIDIA, see [`zeroed`]), and the host reads it for as long as it
/// lives. Read into memory it was the process's own memory for the whole
/// run, and while it was read a buffer that grew to up to twice its size.
/// Mapped, its pages are the file's: under Wine on macOS a mapped 512 MiB
/// file read through added 8 MB to the process's physical footprint, the
/// same file read into a buffer 520 MB (`vmmap`), and the system takes such
/// pages back without writing them anywhere when memory runs short. A save
/// serialises into a mapping of its file, not into one more buffer of the
/// library's size.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory, read-only or writable, with the file
/// held open until the view is unmapped (as codec-d/cool_fox does: on
/// Windows a mapping whose file was closed and renamed over read zeros as
/// its pages left memory). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    _file: std::fs::File,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for writing.
    fn map(file: std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY,
            PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        let (protect, access) = match write {
            true => (PAGE_READWRITE, FILE_MAP_WRITE),
            false => (PAGE_READONLY, FILE_MAP_READ),
        };
        // SAFETY: a file this holds open; the view keeps the mapping alive
        // after its handle is closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                protect,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, access, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped {
            at,
            len,
            _file: file,
        })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped. Opened as the standard library opens a
    /// file to read, so others may still read, rename and delete it.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        Mapped::map(file, len, false)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, this one's alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: this one's view, unmapped once.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// Read into memory (none: an empty library; `VICTORY_LIBRARY_MAP=0`,
    /// a file of the mark before, or one that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped, its mark checked: the library starts
    /// [`HEAD`] bytes in.
    Mapped(Mapped),
}

impl Default for Blob {
    fn default() -> Blob {
        Blob::Held(Vec::new())
    }
}

impl Blob {
    /// The library file in `dir`, after a library a save left beside it
    /// ([`NEXT_FILE`]) took its place; empty if there is none or it is
    /// damaged.
    fn load(dir: &std::path::Path) -> Blob {
        let path = dir.join(LIBRARY_FILE);
        take_next(dir);
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            match sealed(file.bytes()) {
                Some(HEAD) => return Blob::Mapped(file),
                // The mark before: the library starts 12 bytes in, at an
                // address no host is promised to read well, so it is read
                // into memory (once; the next save writes the mark of now).
                Some(_) => {}
                None => return Blob::default(),
            }
        }
        let read = crate::read_file(&path).ok().and_then(unseal);
        Blob::Held(read.unwrap_or_default())
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => &file.bytes()[HEAD..],
        }
    }
}

/// A library a save left beside the library file takes the file's place
/// (before it is mapped, when nothing holds it).
fn take_next(dir: &std::path::Path) {
    let next = dir.join(NEXT_FILE);
    if next.exists() && std::fs::rename(&next, dir.join(LIBRARY_FILE)).is_err() {
        tracing::warn!("the host pipeline library saved last could not take the old one's place");
    }
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // The `windows` binding hands the method the pointer of the slice it
        // is given, and the pointer of an empty Rust slice is dangling rather
        // than null. D3DMetal 3.0 reads any non-null pointer as a serialized
        // blob, even at length zero, and refuses it, so the fallback that
        // starts an empty library fails too and a run is left with no host
        // pipeline library at all. The method's blob is optional, so an empty
        // library is made through the vtable with a null pointer. (Found by
        // nickorlabs in #28; measured here on an M5 Pro with CrossOver 26.3's
        // D3DMetal 3.0.)
        let data = if bytes.is_empty() {
            core::ptr::null()
        } else {
            bytes.as_ptr().cast::<core::ffi::c_void>()
        };
        let mut raw = core::ptr::null_mut();
        // SAFETY: `device` is a live device, and `bytes` outlives the call
        // because the blob is kept beside the library. A null pointer with
        // size 0 is what the method documents for "no initial data".
        unsafe {
            (device.vtable().CreatePipelineLibrary)(
                device.as_raw(),
                data,
                bytes.len(),
                &ID3D12PipelineLibrary::IID,
                &mut raw,
            )
            .ok()?;
            ID3D12PipelineLibrary::from_raw(raw).cast()
        }
    };
    if !blob.bytes().is_empty() {
        let started = std::time::Instant::now();
        match make(blob.bytes()) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                let (bytes, mapped) = (blob.bytes().len(), matches!(blob, Blob::Mapped(_)));
                tracing::info!(bytes, mapped, ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::default(),
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    use crate::shader_cache;
    let translation = shader_cache::translation_build(shader_cache::runtime_options());
    tag_with_settings(blob, translation, super::link_mesh_outputs())
}

fn tag_with_settings(blob: &[u8], translation: &[u8; 32], link_mesh_outputs: bool) -> String {
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translation,
        &[u8::from(link_mesh_outputs)],
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag)
        .chain(keys)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| {
        l.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prebuilt
            .contains(key)
    })
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.state.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    library.pipeline(name, load, create)
}

impl Shared {
    fn new(device: ID3D12Device1, l: Library) -> Shared {
        Shared {
            device,
            loads: l.library.clone(),
            opened_with_pipelines: !l._blob.bytes().is_empty(),
            loaded: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            fresh: OnceLock::new(),
            kept: Mutex::new(Vec::new()),
            state: Mutex::new(l),
        }
    }

    /// `pipeline`, from this library.
    fn pipeline(
        &self,
        name: &str,
        load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
        create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
    ) -> windows::core::Result<ID3D12PipelineState> {
        let wide = HSTRING::from(name);
        let _held = NAMES.hold(name);
        // Outside the lock: loads take up to seconds, and a save holds the
        // lock for as long as it serialises. The libraries' blobs live as
        // long as they do, which are never dropped. A new library first:
        // all it holds was made in this run.
        if let Some(fresh) = self.fresh.get()
            && let Ok(pso) = load(fresh, PCWSTR(wide.as_ptr()))
        {
            LOADED.fetch_add(1, Ordering::Relaxed);
            return Ok(pso);
        }
        if let Ok(pso) = load(&self.loads, PCWSTR(wide.as_ptr())) {
            LOADED.fetch_add(1, Ordering::Relaxed);
            self.loaded.fetch_add(1, Ordering::Relaxed);
            // A new library gets it at the next save (one started meanwhile
            // misses it, and the next run makes it).
            if self.fresh.get().is_some() {
                let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
                kept.push((wide, pso.clone()));
            }
            return Ok(pso);
        }
        // Compiled outside the lock: compiles take up to seconds.
        #[cfg(feature = "diagnostics")]
        LAST.set(Some((name.to_owned(), false)));
        let pso = create()?;
        MADE.fetch_add(1, Ordering::Relaxed);
        let mut l = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a live pipeline; a name already stored (a description that
        // differs) is refused, which leaves the library as it was.
        if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
            l.added += 1;
        } else if self.opened_with_pipelines && self.fresh.get().is_none() {
            // The name is in the library as opened (one stored in this run
            // loads), but did not load with the description it is the hash
            // of: the library never holds this pipeline again.
            let stale = self.stale.fetch_add(1, Ordering::Relaxed) + 1;
            if worn_out(stale, self.loaded.load(Ordering::Relaxed)) {
                self.replace(&mut l, &wide, &pso);
            }
        }
        Ok(pso)
    }

    /// Puts a new library, with `pso` stored in it, in the place of the one
    /// opened, whose entries more often did not load than loaded. The stale
    /// pipelines before `pso` are not in it: the next run makes and stores
    /// them.
    fn replace(&self, l: &mut Library, name: &HSTRING, pso: &ID3D12PipelineState) {
        let Some(fresh) = open(&self.device, Default::default()).map(|f| f.library) else {
            return;
        };
        tracing::warn!(
            stale = self.stale.load(Ordering::Relaxed),
            loaded = self.loaded.load(Ordering::Relaxed),
            "host pipelines stored in the library do not load; a new library replaces it"
        );
        l.library = fresh.clone();
        // The pre-build's list holds for the bytes of the library as opened.
        l.prebuilt.clear();
        l.prebuilt_changed = false;
        // SAFETY: a live pipeline, in a library that holds none yet.
        if unsafe { l.library.StorePipeline(name, pso) }.is_ok() {
            l.added += 1;
        }
        let _ = self.fresh.set(fresh);
    }

    /// Stores in the new library the pipelines loaded from the one opened
    /// since it was replaced.
    fn store_kept(&self, l: &mut Library) {
        let kept = std::mem::take(&mut *self.kept.lock().unwrap_or_else(|e| e.into_inner()));
        for (name, pso) in kept {
            // SAFETY: a live pipeline; one kept twice is refused the second
            // time, which leaves the library as it was.
            if unsafe { l.library.StorePipeline(&name, &pso) }.is_ok() {
                l.added += 1;
            }
        }
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`); the early list alone if
/// that is all that changed.
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else {
        return;
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let (blob, prebuilt) = {
        let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
        library.store_kept(&mut l);
        if l.added == 0 && !l.prebuilt_changed {
            // Nothing new is in the library, so its file stays as it is, and
            // with it the pre-build's list that holds for its bytes. A load
            // reads the early list on its own, not from the library, so a
            // change of it is written alone: serialising the library holds
            // this lock against every worker's stores, for a list of 270 KB
            // at most.
            if !USED_CHANGED.swap(false, Ordering::Relaxed) {
                return;
            }
            (None, None)
        } else {
            // SAFETY: a plain size query on a live library.
            let size = unsafe { l.library.GetSerializedSize() };
            let Some(room) = Room::new(dir, size) else {
                // Left for a later save: the pipelines stay counted as added.
                tracing::warn!(
                    bytes = size,
                    "no memory to serialise the host pipeline library; saved later"
                );
                return;
            };
            match room.serialise(&l.library) {
                Ok((staged, tag)) => {
                    l.added = 0;
                    l.prebuilt_changed = false;
                    // The early list is written below too, so a change of it
                    // that was waiting is saved here.
                    USED_CHANGED.store(false, Ordering::Relaxed);
                    // Kept beside the bytes it holds for (also when a run adds
                    // pipelines).
                    let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag, &l.prebuilt));
                    (Some(staged), prebuilt)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host pipeline library not serialised");
                    return;
                }
            }
        }
    };
    let used = used().as_ref().map(Used::list).unwrap_or_default();
    let bytes = blob.as_ref().map_or(0, Staged::len);
    let written = crate::make_dir(dir)
        .and_then(|()| match blob {
            Some(staged) => staged.place(dir),
            None => Ok(()),
        })
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), &[used.join("\n").as_bytes()]))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(PREBUILT_FILE), &[list.as_bytes()]),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            // Zero if only the early list was written.
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// `n` zero bytes, or `None` when the host has no memory for them. The
/// library is about 1.7 GB on NVIDIA (each pipeline's machine code), and an
/// infallible allocation of that size aborted the whole process mid-run
/// ("memory allocation of 2650101862 bytes failed", RTX 4050 Laptop).
fn zeroed(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Room a library is serialised into: a new file beside the library's,
/// mapped ([`mapped`]), or a buffer.
enum Room {
    /// The file, of [`HEAD`] bytes for the mark and the library's after.
    Mapped(std::path::PathBuf, Mapped),
    Bytes(Vec<u8>),
}

impl Room {
    /// Room for `size` serialised bytes (`None` if none could be had).
    fn new(dir: &std::path::Path, size: usize) -> Option<Room> {
        if mapped() && crate::make_dir(dir).is_ok() {
            let tmp = staging(&dir.join(LIBRARY_FILE));
            if let Some(file) = Mapped::create(&tmp, HEAD + size) {
                return Some(Room::Mapped(tmp, file));
            }
            let _ = std::fs::remove_file(&tmp);
        }
        zeroed(size).map(Room::Bytes)
    }

    /// `library` serialised into the room (a mapped file sealed, unmapped
    /// and closed), with the tag of its bytes ([`tag`]).
    fn serialise(
        self,
        library: &ID3D12PipelineLibrary1,
    ) -> windows::core::Result<(Staged, String)> {
        match self {
            Room::Mapped(tmp, mut file) => {
                let all = file.bytes_mut();
                // SAFETY: room of the size the library asked for, after the
                // mark's.
                if let Err(e) = unsafe { library.Serialize(&mut all[HEAD..]) } {
                    drop(file);
                    let _ = std::fs::remove_file(&tmp);
                    return Err(e);
                }
                let head = mark(&all[HEAD..]);
                all[..HEAD].copy_from_slice(&head);
                let (len, tag) = (all.len() - HEAD, tag(&all[HEAD..]));
                drop(file);
                Ok((Staged::File(tmp, len), tag))
            }
            Room::Bytes(mut bytes) => {
                // SAFETY: a buffer of the size the library asks for.
                unsafe { library.Serialize(&mut bytes) }?;
                let tag = tag(&bytes);
                Ok((Staged::Bytes(bytes), tag))
            }
        }
    }
}

/// A serialised library on its way to the library file.
enum Staged {
    /// A sealed file beside the library's, and the library's bytes in it.
    File(std::path::PathBuf, usize),
    /// The library's bytes, the mark not yet in front of them.
    Bytes(Vec<u8>),
}

impl Staged {
    /// The library's bytes.
    fn len(&self) -> usize {
        match self {
            Staged::File(_, len) => *len,
            Staged::Bytes(bytes) => bytes.len(),
        }
    }

    /// Puts the library in its file in `dir`: renamed over the file, or,
    /// where that is refused (Wine refuses a rename over a file it has
    /// mapped, and the running library's file is mapped), beside it for the
    /// next start ([`NEXT_FILE`]).
    fn place(self, dir: &std::path::Path) -> std::io::Result<()> {
        let path = dir.join(LIBRARY_FILE);
        let tmp = match self {
            Staged::File(tmp, _) => tmp,
            Staged::Bytes(bytes) => {
                use std::io::Write;
                let tmp = staging(&path);
                let written = std::fs::File::create(&tmp).and_then(|mut file| {
                    [&mark(&bytes)[..], &bytes]
                        .iter()
                        .try_for_each(|part| file.write_all(part))
                });
                if let Err(e) = written {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(e);
                }
                tmp
            }
        };
        let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
        if std::fs::rename(&tmp, &path).is_ok() {
            return Ok(());
        }
        let beside = std::fs::rename(&tmp, dir.join(NEXT_FILE));
        match &beside {
            Ok(()) => tracing::info!(
                "host pipeline library saved beside its mapped file, for the next start"
            ),
            Err(_) => {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        beside
    }
}

/// Held by a rename into the cache folder: concurrent worker and shutdown
/// saves do not race on the same staging file or target rename.
static WRITING: Mutex<()> = Mutex::new(());

/// A new name beside `path` for a file on its way to `path`.
fn staging(path: &std::path::Path) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    path.with_file_name(name)
}

/// Writes `parts` one after the other to a temporary file beside `path`, then
/// renames it over `path`: a reader never sees a part. Serialised so
/// concurrent worker and shutdown saves do not race on the same staging file
/// or target rename. Parts, so that a file of a mark and a large blob is
/// written without joining them in a buffer.
fn write_atomically(path: &std::path::Path, parts: &[&[u8]]) -> std::io::Result<()> {
    use std::io::Write;
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = staging(path);
    let res = std::fs::File::create(&tmp)
        .and_then(|mut file| parts.iter().try_for_each(|part| file.write_all(part)))
        .and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created
        .1
        .get_or_insert_with(HashSet::new)
        .insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    /// Each earlier key with its place in the list as loaded: the list is
    /// kept most recent run first, so a lower place is a more recent run.
    before: HashMap<String, usize>,
    now: HashSet<String>,
}

impl Used {
    /// The list as saved: one key a line, most recent run first (a list
    /// in no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used {
            before,
            now: HashSet::new(),
        }
    }

    /// The list to keep: this run's keys, then earlier runs' others, most
    /// recent first, at most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let mut earlier: Vec<_> = self
            .before
            .iter()
            .filter(|(k, _)| !self.now.contains(*k))
            .collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        let earlier = earlier.into_iter().map(|(k, _)| k);
        self.now
            .iter()
            .chain(earlier)
            .take(EARLY_MAX)
            .cloned()
            .collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
/// Whether a title pipeline was drawn or dispatched with since the early list
/// was last written.
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| {
                crate::read_file(&d.join(EARLY_FILE))
                    .ok()
                    .and_then(|b| String::from_utf8(b).ok())
            })
            .unwrap_or_default();
        let read = Used::read(&listed);
        tracing::info!(
            pipelines = read.before.len(),
            "title pipelines to start early"
        );
        *used = Some(read);
    }
    used
}

/// Reads the early list now if not yet (a pre-build asks for no title
/// pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains_key(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used()
        .get_or_insert_with(Used::default)
        .now
        .insert(key.to_owned())
    {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A size the host cannot give is refused, not fatal; others are zeros.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_refused() {
        assert!(zeroed(usize::MAX / 2).is_none());
        assert_eq!(zeroed(100).as_deref(), Some(&[0u8; 100][..]));
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys =
            |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let before = keys(0..EARLY_MAX)
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, i))
            .collect();
        let used = Used {
            before,
            now: keys(EARLY_MAX - 10..EARLY_MAX + 10),
        };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(
            list.iter().collect::<HashSet<_>>().len(),
            EARLY_MAX,
            "no key twice"
        );
    }

    /// A full list makes room by the oldest runs' keys, and keeps that
    /// order across saves.
    #[test]
    fn a_full_early_list_drops_the_oldest_runs_keys() {
        let key = |i: usize| format!("{i:032x}");
        // As saved: the last run's keys, then the run before's.
        let saved: Vec<String> = (0..EARLY_MAX).map(key).collect();
        let mut used = Used::read(&saved.join("\n"));
        used.now = (EARLY_MAX..EARLY_MAX + 10).map(key).collect();
        used.now.insert(key(EARLY_MAX - 1)); // the oldest key, used again
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        // The ten that go are the oldest ones not used again.
        let dropped: HashSet<String> = (EARLY_MAX - 11..EARLY_MAX - 1).map(key).collect();
        assert!(list.iter().all(|k| !dropped.contains(k)));
        // Read back, this run's keys come first, then the earlier ones in order.
        let again = Used::read(&list.join("\n"));
        assert!(used.now.iter().all(|k| again.before[k] < used.now.len()));
        assert!(again.before[&key(0)] < again.before[&key(1)]);
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn changing_translation_settings_invalidates_the_prebuilt_list() {
        use crate::shader_cache::{runtime_options, translation_build};

        let options = runtime_options();
        let current = |locks, bias, narrow, link| {
            let options = victory_shader::emit::Options {
                atomic_locks: locks,
                bias_as_gradients: bias,
                narrow_waves: narrow,
                ..options
            };
            tag_with_settings(b"library", translation_build(options), link)
        };
        let keys = HashSet::from(["pipeline".to_owned()]);
        let original = current(false, false, false, false);
        let saved = list(&original, &keys);
        assert_eq!(listed(saved.as_bytes(), &original), keys);
        for changed in [
            current(true, false, false, false),
            current(false, true, false, false),
            current(false, false, true, false),
            current(false, false, false, true),
        ] {
            assert!(listed(saved.as_bytes(), &changed).is_empty());
        }
        // The locks' rounds change their translation, so they key it too.
        let rounds = |on| {
            translation_build(victory_shader::emit::Options {
                atomic_locks: true,
                atomic_lock_rounds: on,
                ..options
            })
        };
        assert_ne!(rounds(true), rounds(false));
        // And a reduced umax changes it again.
        let umax = |on| {
            translation_build(victory_shader::emit::Options {
                atomic_locks: true,
                atomic_lock_rounds: true,
                atomic_umax: on,
                ..options
            })
        };
        assert_ne!(umax(true), umax(false));
        assert_eq!(
            tag(b"library"),
            current(
                options.atomic_locks,
                options.bias_as_gradients,
                options.narrow_waves,
                super::super::link_mesh_outputs(),
            )
        );
    }

    #[test]
    fn sealed_files_round_trip_and_damage_is_refused() {
        let blob = b"library bytes".to_vec();
        let mut file = [&mark(&blob)[..], &blob].concat();
        assert_eq!(unseal(file.clone()), Some(blob));
        file[20] ^= 1;
        assert_eq!(unseal(file), None);
        assert_eq!(unseal(b"short".to_vec()), None);
    }

    /// The file is "VPL2", four zero bytes, the library's FNV-1a checksum
    /// (little endian) and the library, which starts 16 bytes in; files of
    /// earlier builds ("VPL1" and the checksum, 12 bytes) still load.
    #[test]
    fn the_sealed_file_formats() {
        // From the FNV-1a (64 bit) test vectors.
        assert_eq!(checksum(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(checksum(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(checksum(b"foobar"), 0x8594_4171_f739_67e8);
        let sum = 0x8594_4171_f739_67e8u64.to_le_bytes();
        let mut file = [&b"VPL2\0\0\0\0"[..], &sum, b"foobar"].concat();
        assert_eq!([&mark(b"foobar")[..], b"foobar"].concat(), file);
        assert_eq!(sealed(&file), Some(16));
        assert_eq!(unseal(file.clone()), Some(b"foobar".to_vec()));
        let mut old = [&b"VPL1"[..], &sum, b"foobar"].concat();
        assert_eq!(sealed(&old), Some(12));
        assert_eq!(unseal(old.clone()), Some(b"foobar".to_vec()));
        // An empty library, and a mark or checksum that is not right.
        assert_eq!(unseal(mark(b"").to_vec()), Some(Vec::new()));
        for file in [&mut file, &mut old] {
            file[0] = b'X';
            assert_eq!(unseal(file.clone()), None);
            file[0] = b'V';
            file[9] ^= 1;
            assert_eq!(unseal(file.clone()), None);
        }
        assert_eq!(sealed(b"VPL2\0\0\0\0"), None);
    }

    /// A library file of the mark of now is mapped, one of the mark before
    /// is read into memory, a damaged or missing one is no library: each
    /// with the library's bytes as the file holds them.
    #[test]
    fn library_files_are_mapped_or_read_as_their_mark_says() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        let library: Vec<u8> = (0..200_000u32).map(|i| (i % 253) as u8).collect();
        std::fs::write(&path, [&mark(&library)[..], &library].concat()).unwrap();
        let blob = Blob::load(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)), "the mark of now: mapped");
        assert_eq!(blob.bytes(), &library[..]);
        assert_eq!(
            blob.bytes().as_ptr() as usize % 16,
            0,
            "the library aligned"
        );
        drop(blob);
        let old = [&b"VPL1"[..], &checksum(&library).to_le_bytes(), &library].concat();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::load(dir.path());
        assert!(matches!(blob, Blob::Held(_)), "the mark before: read");
        assert_eq!(blob.bytes(), &library[..]);
        drop(blob);
        let mut damaged = [&mark(&library)[..], &library].concat();
        damaged[1000] ^= 1;
        std::fs::write(&path, &damaged).unwrap();
        assert!(Blob::load(dir.path()).bytes().is_empty(), "damaged: none");
        std::fs::remove_file(&path).unwrap();
        assert!(Blob::load(dir.path()).bytes().is_empty(), "missing: none");
    }

    /// A save while the library's file is mapped goes over it where the
    /// system lets it and beside it where it does not (Wine refuses a rename
    /// over a mapped file); the running library still reads the bytes it
    /// was made from, and the next start loads what was saved, with nothing
    /// left beside.
    #[test]
    fn a_save_over_the_mapped_library_file_is_the_next_start_s_library() {
        let dir = tempfile::tempdir().unwrap();
        let first: Vec<u8> = vec![0x11; 300_000];
        let second: Vec<u8> = vec![0x22; 400_000];
        let path = dir.path().join(LIBRARY_FILE);
        std::fs::write(&path, [&mark(&first)[..], &first].concat()).unwrap();
        let running = Blob::load(dir.path());
        assert!(matches!(running, Blob::Mapped(_)));
        Staged::Bytes(second.clone()).place(dir.path()).unwrap();
        // A second save in the same run goes the same way.
        Staged::Bytes(second.clone()).place(dir.path()).unwrap();
        assert!(
            running.bytes() == &first[..],
            "the running library's bytes kept"
        );
        drop(running);
        let next = Blob::load(dir.path());
        assert!(
            next.bytes() == &second[..],
            "the saved library at the next start"
        );
        assert!(!dir.path().join(NEXT_FILE).exists());
        let names: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().collect();
        assert_eq!(names.len(), 1, "no staging file left");
    }

    /// A library serialised into a mapped file is that file at the next
    /// start: mapped, accepted by the host, with its pipelines.
    #[test]
    fn a_library_serialised_into_a_mapped_file_opens_mapped() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let dir = tempfile::tempdir().unwrap();
        let library = open(&device, Blob::default()).expect("an empty library");
        // SAFETY: a plain query on a live library.
        let size = unsafe { library.library.GetSerializedSize() };
        let room = Room::new(dir.path(), size).expect("room");
        assert!(matches!(room, Room::Mapped(..)));
        let (staged, tag_of) = room.serialise(&library.library).unwrap();
        assert!(matches!(staged, Staged::File(_, n) if n == size));
        staged.place(dir.path()).unwrap();
        let blob = Blob::load(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)));
        assert_eq!(tag(blob.bytes()), tag_of);
        let again = open(&device, blob).expect("the library again");
        assert!(matches!(again._blob, Blob::Mapped(_)), "accepted as mapped");
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Blob::default()).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        // The library again from its bytes in memory, and from a file it was
        // serialised into through a mapping, mapped ([`mapped`]).
        let dir = tempfile::tempdir().unwrap();
        let (staged, _) = Room::new(dir.path(), made.serialized().len())
            .expect("room")
            .serialise(&made.2)
            .unwrap();
        staged.place(dir.path()).unwrap();
        for blob in [Blob::Held(made.serialized()), Blob::load(dir.path())] {
            let mapped = matches!(blob, Blob::Mapped(_));
            let again = open(&device, blob).expect("the library again");
            assert_eq!(matches!(again._blob, Blob::Mapped(_)), mapped);
            loads_and_stores(
                &Shared(device.clone(), root.clone(), again.library.clone()),
                &codes,
            );
        }
        fn loads_and_stores(shared: &Shared, codes: &[Vec<u8>]) {
            let stores = Mutex::new(());
            std::thread::scope(|s| {
                for t in 0..8 {
                    let (codes, stores) = (&codes, &stores);
                    s.spawn(move || {
                        for i in (t..N).step_by(8) {
                            shared.load(i, &codes[i]).expect("a stored pipeline");
                            assert!(
                                shared.load(i, &codes[(i + 1) % N]).is_err(),
                                "another description"
                            );
                            assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                            let _one = stores.lock().unwrap();
                            shared.store(N + i, &codes[N + i]);
                            shared.serialized();
                        }
                    });
                }
            });
            for (i, code) in codes.iter().enumerate() {
                shared.load(i, code).expect("every pipeline");
            }
        }
    }

    /// A stored pipeline loads while a save holds the library's lock (one
    /// serialises the whole library, for seconds in the world).
    #[test]
    fn a_pipeline_loads_while_a_save_holds_the_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let (Some(device), Some(library)) = (
            crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()),
            library(),
        ) else {
            return; // no host GPU here
        };
        // No resources, and the root signature made apart: a compiler need
        // not keep one the source names (Wine's does not).
        let source = "[numthreads(1, 1, 1)] void main() {}";
        let mut blob = None;
        // SAFETY: the source outlives the call; the blob is read within it.
        let code = unsafe {
            if D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .is_err()
            {
                return; // no shader compiler here
            }
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        struct Made(ID3D12Device1, ID3D12RootSignature, Vec<u8>);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Made {}
        impl Made {
            fn get(&self, made: bool) -> windows::core::Result<ID3D12PipelineState> {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: self.2.as_ptr().cast(),
                        BytecodeLength: self.2.len(),
                    },
                    ..Default::default()
                };
                pipeline(
                    "test: a pipeline loads while a save holds the library",
                    // SAFETY: the description points at live data.
                    |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                    || match made {
                        // SAFETY: as above.
                        true => unsafe { self.0.CreateComputePipelineState(&desc) },
                        false => panic!("made again, not loaded"),
                    },
                )
            }
        }
        let made = Made(device, root, code);
        made.get(true).expect("made and stored");
        let saving = library.state.lock().unwrap_or_else(|e| e.into_inner());
        let (done, loaded) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let made = &made;
            s.spawn(move || done.send(made.get(false).is_ok()));
            let loaded = loaded.recv_timeout(std::time::Duration::from_secs(30));
            drop(saving);
            assert_eq!(loaded, Ok(true), "the load waited for the save");
        });
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        // A damaged file is refused by the mark check before the host sees
        // it, because not every host checks the bytes itself.
        let blob = unseal(b"not a pipeline library".repeat(16)).unwrap_or_default();
        let library = open(&device, Blob::Held(blob)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.bytes().is_empty());
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &[&payload]).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }

    #[test]
    fn write_atomically_writes_its_parts_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let blob = vec![7u8; 100_000];
        write_atomically(&target, &[&mark(&blob), &[], &blob]).unwrap();
        assert_eq!(unseal(std::fs::read(&target).unwrap()), Some(blob));
        // Over a file that is there, with no staging file left.
        write_atomically(&target, &[b"a", b"b"]).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"ab");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_library_is_replaced_once_more_was_stale_than_loaded() {
        assert!(!worn_out(STALE_MIN - 1, 0), "too few to tell");
        assert!(worn_out(STALE_MIN, 0));
        assert!(worn_out(STALE_MIN + 1, STALE_MIN));
        assert!(!worn_out(STALE_MIN, STALE_MIN), "as many loaded");
        assert!(!worn_out(100, 1800));
    }

    /// A library whose entries do not load with the descriptions their names
    /// are the hashes of (as when they refer to a cache of the host's own
    /// that is gone) is replaced: the pipelines the run makes or loads from
    /// then on go into a new library, and load from the bytes it saves.
    #[test]
    fn a_library_of_entries_that_do_not_load_is_replaced() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const STALE: usize = STALE_MIN as usize;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let compile = |source: &str| {
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        // What the library holds under each name, and what the run asks for.
        let (Some(held), Some(asked)) = (
            compile("[numthreads(1, 1, 1)] void main() {}"),
            compile("[numthreads(2, 1, 1)] void main() {}"),
        ) else {
            return; // no shader compiler here
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        let desc = |code: &[u8]| D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        let serialized = |library: &ID3D12PipelineLibrary1| {
            // SAFETY: a buffer of the size asked for; nothing is stored
            // meanwhile.
            unsafe {
                let mut blob = vec![0; library.GetSerializedSize()];
                library.Serialize(&mut blob).unwrap();
                blob
            }
        };
        let name = |i: usize| format!("test: a stale pipeline {i}");
        // A library holding `held` under every name, as opened in a run.
        let first = open(&device, Blob::default()).expect("an empty library");
        // SAFETY: the description points at live data.
        let pso: ID3D12PipelineState =
            unsafe { device.CreateComputePipelineState(&desc(&held)) }.unwrap();
        for i in 0..STALE + 4 {
            // SAFETY: a live pipeline.
            unsafe { first.library.StorePipeline(&HSTRING::from(name(i)), &pso) }.unwrap();
        }
        let opened = open(&device, Blob::Held(serialized(&first.library))).expect("the library again");
        let shared = Shared::new(device.clone(), opened);
        let get = |i: usize, code: &[u8], made: bool| {
            let desc = desc(code);
            shared
                .pipeline(
                    &name(i),
                    // SAFETY: the description points at live data.
                    |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                    || match made {
                        // SAFETY: as above.
                        true => unsafe { device.CreateComputePipelineState(&desc) },
                        false => panic!("{} made again, not loaded", name(i)),
                    },
                )
                .expect("a pipeline");
        };
        for i in 0..STALE - 1 {
            get(i, &asked, true);
        }
        assert!(shared.fresh.get().is_none(), "too few stale to tell");
        get(STALE - 1, &asked, true);
        assert!(shared.fresh.get().is_some(), "replaced");
        // In the new library from then on: what is made, and what loads from
        // the library as opened (its own description).
        get(STALE - 1, &asked, false);
        get(STALE, &held, false);
        get(STALE + 1, &held, false);
        get(STALE + 2, &asked, true);
        get(STALE + 3, &asked, true);
        get(STALE + 2, &asked, false);
        // What the next save writes.
        let bytes = {
            let mut l = shared.state.lock().unwrap();
            shared.store_kept(&mut l);
            assert_eq!(l.added, 5);
            serialized(&l.library)
        };
        let saved = open(&device, Blob::Held(bytes)).expect("the new library");
        let loads = |i: usize, code: &[u8]| {
            let (name, desc) = (HSTRING::from(name(i)), desc(code));
            // SAFETY: the description points at live data.
            let pso: windows::core::Result<ID3D12PipelineState> =
                unsafe { saved.library.LoadComputePipeline(&name, &desc) };
            pso.is_ok()
        };
        for i in [STALE - 1, STALE + 2, STALE + 3] {
            assert!(loads(i, &asked), "{} as made", name(i));
        }
        for i in [STALE, STALE + 1] {
            assert!(loads(i, &held), "{} as loaded", name(i));
        }
        // Made before the library was replaced: the next run makes it.
        assert!(!loads(0, &asked));
    }
}
