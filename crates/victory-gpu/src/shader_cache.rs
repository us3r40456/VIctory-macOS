//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder
/// of this translator, `None` if it cannot be opened.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    let dir = dir.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(&dir) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "shader cache");
            let _ = CACHE.set(cache);
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            None
        }
    }
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings (the loop limit, which a diagnostics build can
/// change at run time, and whether the host's waves are 32 lanes at most,
/// [`host_wave32`]).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(victory_shader::SOURCE_HASH, ENCODING, limit, host_wave32())
    })
}

/// Whether translations are for a host of 32-lane waves: the host GPU's
/// (made here if it is not yet: the cache belongs to the GPU it was filled
/// on, as the pipeline library does); none without one.
fn host_wave32() -> bool {
    crate::host::host().is_some_and(crate::host::Host::wave32)
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(translator: &str, encoding: &[u8], limit: u32, wave32: bool) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    h.update(&[u8::from(wave32)]);
    *h.finalize().as_bytes()
}

/// Whether translations put 64-bit image atomics under locks: under Wine,
/// whose D3DMetal claims the atomics and leaves the texels untouched.
/// `VICTORY_ATOMIC_LOCKS=0` or `1` overrides.
fn atomic_locks() -> bool {
    static LOCKS: OnceLock<bool> = OnceLock::new();
    *LOCKS.get_or_init(|| match std::env::var("VICTORY_ATOMIC_LOCKS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether the lanes of a wave first try their 64-bit atomics' locks in
/// rounds before the lanes still not done take turns (`emit`, "Atomic
/// locks"; #240): wherever [`atomic_locks`] are on. In the game the rounds
/// took a world frame from 7.9 to 2.5 s on an M2 Max, from 5.29 to 0.71 s
/// on an M4 Pro and from 2.9 to 0.88 s on an M4, and left no GPU resets.
/// `VICTORY_ATOMIC_LOCK_ROUNDS=0` goes back to the turns alone.
fn atomic_lock_rounds() -> bool {
    static ROUNDS: OnceLock<bool> = OnceLock::new();
    *ROUNDS.get_or_init(|| {
        let var = std::env::var("VICTORY_ATOMIC_LOCK_ROUNDS").ok();
        rounds_on(atomic_locks(), var.as_deref())
    })
}

/// Whether a glc-clear 64-bit `umax` first reduces each texel's lanes of
/// a wave to their maximum key, one representative locking (`emit`,
/// "Atomic locks"; #240): opt-in while it is measured,
/// `VICTORY_ATOMIC_UMAX=1`. Only with [`atomic_lock_rounds`].
fn atomic_umax() -> bool {
    static UMAX: OnceLock<bool> = OnceLock::new();
    *UMAX.get_or_init(|| {
        atomic_lock_rounds() && std::env::var("VICTORY_ATOMIC_UMAX").as_deref() == Ok("1")
    })
}

/// The rounds, given whether the locks are on and
/// `VICTORY_ATOMIC_LOCK_ROUNDS`: on with the locks unless it is `0`.
fn rounds_on(locks: bool, var: Option<&str>) -> bool {
    locks && var != Some("0")
}

/// Whether compute shaders' biased samples run as scaled gradient samples:
/// under Wine, whose D3DMetal ignores the bias there (#30).
/// `VICTORY_BIAS_AS_GRADIENTS=0` or `1` overrides.
fn bias_as_gradients() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_BIAS_AS_GRADIENTS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether translations take wave64 programs to run in 32-lane host waves:
/// under Wine, whose D3DMetal reports 32 lanes and ignores `WaveSize(64)`.
/// `VICTORY_NARROW_WAVES=0` or `1` overrides.
fn narrow_waves() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_NARROW_WAVES").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether translations for a host of 32-lane waves let a wave64 program
/// that reads across the two halves run with each half answering from its
/// own lanes, instead of refusing it
/// (`victory_shader::emit::Options::halves_local`): only with
/// `VICTORY_HALVES_LOCAL=1`. An approximation, so never on by itself.
pub fn halves_local() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_HALVES_LOCAL");
        if on {
            tracing::info!(
                "VICTORY_HALVES_LOCAL=1: wave64 programs that read across 32-lane halves run with each half as its own wave"
            );
        }
        on
    })
}

/// Whether translations for a host of 32-lane waves let a group of one
/// console wave64 share LDS between its two halves with nothing ordering
/// them, instead of refusing it
/// (`victory_shader::emit::Options::halves_shared_lds`): only with
/// `VICTORY_HALVES_SHARED_LDS=1`. The halves race, so a test, not a fix.
pub fn halves_shared_lds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_HALVES_SHARED_LDS");
        if on {
            tracing::info!(
                "VICTORY_HALVES_SHARED_LDS=1: one-wave64 groups share LDS between their two 32-lane halves without sync"
            );
        }
        on
    })
}

/// Whether one-wave groups that share LDS get the exact order of
/// `victory_shader::emit::Options::lds_order` where there is one
/// (`VICTORY_HALVES_LDS_ORDER=1`; the rule is in `mac_switches`).
pub fn halves_lds_order() -> bool {
    crate::mac_switches::lds_order()
}

/// Whether lane reads across 32-lane halves are handed over through
/// group-shared memory where every lane runs
/// (`victory_shader::emit::Options::halves_exchange`,
/// `VICTORY_HALVES_EXCHANGE=1`; the rule is in `mac_switches`).
pub fn halves_exchange() -> bool {
    crate::mac_switches::exchange()
}

/// Whether, with that, every lane walks the skips around a hand-over
/// (`victory_shader::emit::Options::halves_walk`, `VICTORY_HALVES_WALK=1`;
/// the rule is in `mac_switches`).
pub fn halves_walk() -> bool {
    crate::mac_switches::walk()
}

/// Whether, with half-local reads, the Mac port's hand-over across the
/// halves comes before them (`victory_shader::emit::Options::halves_across`,
/// `VICTORY_HALVES_ACROSS=1`; the rule is in `mac_switches`).
pub fn halves_across() -> bool {
    crate::mac_switches::across()
}

/// Whether programs run in either of those ways here: asked for, on a host
/// of 32-lane waves (elsewhere the settings change no translation).
#[cfg(windows)]
pub(crate) fn halves_local_here() -> bool {
    (halves_local() || halves_shared_lds()) && host_wave32()
}

/// Whether this runs under Wine (on a Mac: D3DMetal's D3D12).
#[cfg(windows)]
pub(crate) fn under_wine() -> bool {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: ntdll is always loaded; both names end with a NUL.
    unsafe {
        GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_version".as_ptr().cast(),
        )
    }
    .is_some()
}

#[cfg(not(windows))]
pub(crate) fn under_wine() -> bool {
    false
}

/// [`build`] for translations with host workarounds (atomic locks, their
/// rounds and umax reduction, bias as gradients, narrow waves, half-local
/// reads and shared LDS in halves, `local`): their own cache entries, one
/// per combination.
fn options_build(
    locks: bool,
    [rounds, umax]: [bool; 2],
    bias: bool,
    narrow: bool,
    local: [bool; 2],
) -> &'static [u8; 32] {
    static BUILDS: [OnceLock<[u8; 32]>; 128] = [const { OnceLock::new() }; 128];
    let n = usize::from(locks)
        | usize::from(bias) << 1
        | usize::from(narrow) << 2
        | usize::from(local[0]) << 3
        | usize::from(local[1]) << 4
        | usize::from(rounds) << 5
        | usize::from(umax) << 6;
    BUILDS[n].get_or_init(|| options_key(build(), locks, [rounds, umax], bias, narrow, local))
}

/// [`options_build`]'s key over the build `build`. A setting that is off
/// adds nothing, so the entries made without it stay the ones read.
fn options_key(
    build: &[u8; 32],
    locks: bool,
    [rounds, umax]: [bool; 2],
    bias: bool,
    narrow: bool,
    local: [bool; 2],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    if locks {
        h.update(b"atomic locks");
    }
    if rounds {
        h.update(b"atomic lock rounds");
    }
    if umax {
        h.update(b"atomic umax");
    }
    if bias {
        h.update(b"bias as gradients");
    }
    if narrow {
        h.update(b"narrow waves");
    }
    if local[0] {
        h.update(b"halves local");
    }
    if local[1] {
        h.update(b"halves shared lds");
    }
    *h.finalize().as_bytes()
}

/// What the translations made with `options` are keyed by: [`build`], or
/// with host workarounds [`options_build`], and over either the Mac line's
/// settings (`crate::mac_switches::key`). Also what the pre-build's list of
/// the host pipeline library holds for (`crate::pipeline::library`).
pub(crate) fn translation_build(options: victory_shader::emit::Options) -> &'static [u8; 32] {
    let locks = options.atomic_locks;
    let bias = options.bias_as_gradients;
    let narrow = options.narrow_waves;
    let rounds = locks && options.atomic_lock_rounds;
    let umax = rounds && options.atomic_umax;
    let local = [options.halves_local, options.halves_shared_lds];
    let build = if !locks && !bias && !narrow && local == [false; 2] {
        build()
    } else {
        options_build(locks, [rounds, umax], bias, narrow, local)
    };
    // The Mac line's settings, over whichever of those it is.
    let key = crate::mac_switches::key(
        build,
        options.early_depth,
        [options.atomic_lock_helpers, options.atomic_lock_keys],
        options.lds_order,
        options.halves_exchange,
        options.uav_size_level0,
        options.halves_walk,
    );
    // And the Mac port's hand-over before half-local reads
    // (`VICTORY_HALVES_ACROSS=1`); off, it adds nothing.
    let key = if options.halves_across {
        let mut h = blake3::Hasher::new();
        h.update(&key);
        h.update(b"halves across");
        *h.finalize().as_bytes()
    } else {
        key
    };
    if key == *build {
        return build;
    }
    // A handful of combinations in a run: each key kept once, for good.
    static KEYS: Mutex<Vec<&'static [u8; 32]>> = Mutex::new(Vec::new());
    let mut keys = KEYS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(kept) = keys.iter().find(|k| ***k == key) {
        return kept;
    }
    let kept: &'static [u8; 32] = Box::leak(Box::new(key));
    keys.push(kept);
    kept
}

pub(crate) fn runtime_options() -> victory_shader::emit::Options {
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    // Under Wine (D3DMetal), 64-bit image atomics run under locks: its own
    // leave the texels untouched (`emit`, "Atomic locks"). On a host of
    // 32-lane waves, wave64 programs run in halves (`build` keys on it).
    // Which lanes take those locks matters only where there are locks: off
    // elsewhere, so that the switch then makes no cache entries of its own.
    let lock_writers = if atomic_locks() {
        crate::mac_switches::lock_writers(under_wine())
    } else {
        [false; 2]
    };
    victory_shader::emit::Options {
        view_marks: true,
        atomic_locks: atomic_locks(),
        atomic_lock_rounds: atomic_lock_rounds(),
        atomic_umax: atomic_umax(),
        bias_as_gradients: bias_as_gradients(),
        host_wave32: host_wave32(),
        narrow_waves: narrow_waves(),
        halves_local: halves_local(),
        halves_shared_lds: halves_shared_lds(),
        early_depth: crate::mac_switches::early_depth(under_wine()),
        atomic_lock_helpers: lock_writers[0],
        atomic_lock_keys: lock_writers[1],
        lds_order: crate::mac_switches::lds_order(),
        halves_exchange: crate::mac_switches::exchange(),
        uav_size_level0: crate::mac_switches::uav_size_level0(),
        halves_walk: crate::mac_switches::walk(),
        halves_across: crate::mac_switches::across(),
    }
}

/// The translation of a console shader container, from the cache when
/// there is one, and whether it came from there.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    let mut options = runtime_options();
    // One program at a time, if the switch lists hashes (mac).
    options.uav_size_level0 = crate::mac_switches::uav_size_level0_for(container);
    let translate =
        |c: &[u8]| victory_shader::emit::translate_with(c, options).map_err(|e| e.to_string());
    let build = translation_build(options);
    match CACHE.get() {
        Some(cache) => cached(cache, build, container, translate),
        None => (translate(container), false),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit;
/// `made` keeps a miss's translation, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms here).
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let made = (self.translate)(&input[32..]);
        let bytes = encode(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return (shader, false);
        }
        if let Some(shader) = crate::read_file(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    (
        made().unwrap_or_else(|| (converter.translate)(container)),
        false,
    )
}

const MAGIC: &[u8; 4] = b"VSH2";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, root_srv_bytes.len() as u32);
    for &(slot, end) in root_srv_bytes {
        word(&mut out, u32::from(slot));
        word(&mut out, end);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_srv_bytes = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_srv_bytes.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    let memory = r.word()?;
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        view_marks: memory & 32 != 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounds are on wherever the locks are, `VICTORY_ATOMIC_LOCK_ROUNDS=0`
    /// turns them off, and without the locks there are none (#240).
    #[test]
    fn lock_rounds_are_on_with_the_locks_unless_turned_off() {
        assert!(rounds_on(true, None), "on by default with the locks");
        assert!(rounds_on(true, Some("1")), "the old opt-in still works");
        assert!(!rounds_on(true, Some("0")), "0 goes back to the turns");
        assert!(!rounds_on(false, None), "no rounds without the locks");
        assert!(!rounds_on(false, Some("1")), "nor when asked for");
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            root_srv_bytes: vec![(1, 48), (5, 16384)],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &[7; 32], b"a container", translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the host's wave width) makes
    /// another key, so no entry written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let k = key("0123456789abcdef", b"encoding", 1 << 20, false);
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20, false));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20, false));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20, false));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19, false));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 20, true));
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                host_wave32()
            )
        );
        // Half-local reads (`VICTORY_HALVES_LOCAL=1`) and shared LDS in
        // halves (`VICTORY_HALVES_SHARED_LDS=1`) are entries of their own
        // in every combination; off, the key is what it was without them.
        for n in 0..8u8 {
            let (l, b, w) = (n & 1 != 0, n & 2 != 0, n & 4 != 0);
            let keys: std::collections::HashSet<[u8; 32]> =
                [[false, false], [true, false], [false, true], [true, true]]
                    .map(|local| options_key(&k, l, [false; 2], b, w, local))
                    .into();
            assert_eq!(keys.len(), 4);
        }
        let mut narrow = blake3::Hasher::new();
        narrow.update(&k);
        narrow.update(b"narrow waves");
        assert_eq!(
            options_key(&k, false, [false; 2], false, true, [false, false]),
            *narrow.finalize().as_bytes()
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
