//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex or mesh plus pixel pipelines from streams and
//! compute pipelines get host pipelines; amplification, tessellation and
//! geometry stages, and the plain graphics description, keep their
//! containers only. The 32-byte packet after the header stays zero: titles
//! only copy it inline, which our null command-list limits keep from
//! running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
#[cfg(windows)]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    #[cfg(windows)]
    full_hash: [u8; 32],
    /// The container's length, as logged once its bytes are gone.
    size: usize,
    /// Released after translation unless an experimental mesh/pixel pair may need it.
    #[cfg(any(windows, test))]
    container: Mutex<Vec<u8>>,
    /// Its full BLAKE3 once the container is on disk instead
    /// (`VICTORY_SPILL_SHADERS`, [`spill`]).
    #[cfg(windows)]
    spilled: OnceLock<[u8; 32]>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut created = false;
    let shader = {
        let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
        shaders
            .get_or_insert_with(HashMap::new)
            .entry(*full.as_bytes())
            .or_insert_with(|| {
                created = true;
                // The programs the ambient stand-in follows are known by
                // name (`crate::ambient`; nothing while it is off).
                #[cfg(windows)]
                crate::ambient::note_program(&full.to_hex()[..16], bytes);
                // So are the ray passes `VICTORY_RAY_SKIP` leaves out
                // (`crate::ray_skip`; nothing while it is off).
                #[cfg(windows)]
                crate::ray_skip::note_program(stage, &full.to_hex()[..16], bytes);
                Arc::new(Shader::with_hash(stage, bytes, full))
            })
            .clone()
    };
    // Written outside the lock every pipeline creation takes.
    #[cfg(windows)]
    if created {
        // With the frame recorder asked for (`VICTORY_RECORD`), the
        // container is kept beside its captures, so that a replay can
        // translate it again (one look at a flag without it).
        crate::replay::record::container(&full.to_hex()[..16], bytes);
        spill(&shader, full.as_bytes());
    }
    #[cfg(not(windows))]
    let _ = created;
    shader
}

/// Folder for shader containers kept on disk until translated
/// (`VICTORY_SPILL_SHADERS=1`): the title passes tens of thousands of
/// distinct containers and draws with a few hundred, and holding them all
/// was ~700 MB of an 8 GB Mac's memory (#135).
#[cfg(windows)]
static SPILL_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The file a container with BLAKE3 `full` is kept in.
#[cfg(windows)]
fn spill_path(dir: &std::path::Path, full: &[u8; 32]) -> std::path::PathBuf {
    let hex = blake3::Hash::from(*full).to_hex();
    dir.join(&hex[..2]).join(format!("{hex}.bin"))
}

/// Moves `shader`'s container to disk if spilling is on: written under a
/// temporary name and renamed, so a file found there is whole; one already
/// there from an earlier run is kept.
#[cfg(windows)]
fn spill(shader: &Shader, full: &[u8; 32]) {
    if let Some(dir) = SPILL_DIR.get() {
        spill_to(dir, shader, full);
    }
}

/// [`spill`] into `dir`.
#[cfg(windows)]
fn spill_to(dir: &std::path::Path, shader: &Shader, full: &[u8; 32]) {
    let path = spill_path(dir, full);
    let there = std::fs::File::open(&path).is_ok();
    if !there {
        let Some(shard) = path.parent() else { return };
        let mut container = shader.container.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let ok = crate::make_dir(shard).is_ok()
            && std::fs::write(&tmp, &*container).is_ok()
            && std::fs::rename(&tmp, &path).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        *container = Vec::new();
    } else {
        *shader.container.lock().unwrap_or_else(|e| e.into_inner()) = Vec::new();
    }
    let _ = shader.spilled.set(*full);
}

impl Shader {
    fn new(stage: &'static str, bytes: &[u8]) -> Shader {
        Self::with_hash(stage, bytes, blake3::hash(bytes))
    }

    fn with_hash(stage: &'static str, bytes: &[u8], digest: blake3::Hash) -> Shader {
        Shader {
            stage,
            hash: digest.to_hex()[..16].to_owned(),
            #[cfg(windows)]
            full_hash: *digest.as_bytes(),
            size: bytes.len(),
            #[cfg(any(windows, test))]
            container: Mutex::new(bytes.to_vec()),
            #[cfg(windows)]
            spilled: OnceLock::new(),
            #[cfg(windows)]
            translated: OnceLock::new(),
        }
    }

    #[cfg(all(test, windows))]
    pub(crate) fn for_test(stage: &'static str, translated: Translated) -> Arc<Shader> {
        let shader = Shader::new(stage, b"");
        let _ = shader.translated.set(Some(translated));
        Arc::new(shader)
    }
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use (or taken from the cache); a
    /// shader that does not translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let (started, cpu) = (std::time::Instant::now(), thread_cpu());
                let (done, cached) = {
                    let mut container = self.container.lock().unwrap_or_else(|e| e.into_inner());
                    if container.is_empty()
                        && let (Some(full), Some(dir)) = (self.spilled.get(), SPILL_DIR.get())
                    {
                        *container = crate::read_file(&spill_path(dir, full)).unwrap_or_default();
                    }
                    let done = translate(&container);
                    // After the translation, not before: if it panics, the
                    // bytes are still there for the next try.
                    let keep = link_mesh_outputs() && done.0.as_ref().is_ok_and(|t| {
                        self.stage == "pixel"
                            || (self.stage == "mesh"
                                && victory_shader::emit::mesh_outputs(&t.host.dxil)
                                    .is_some_and(|o| o.bytes() > link_mesh_over()))
                    });
                    if !keep {
                        *container = Vec::new();
                    }
                    done
                };
                let ms = started.elapsed().as_millis() as u64;
                let (user_ms, kernel_ms) = thread_cpu_since(cpu);
                match &done {
                    Ok(t) => {
                        tracing::info!(
                            stage = self.stage,
                            hash = %self.hash,
                            user_data = t.host.user_data.len(),
                            samplers = t.host.samplers.len(),
                            cached,
                            ms,
                            user_ms,
                            kernel_ms,
                            "shader translated"
                        );
                        // D3DMetal makes such a pipeline and marks it a no-op:
                        // its draws are dropped without an error (#132).
                        use victory_shader::emit::{METAL_MESH_OUTPUT_BYTES, mesh_outputs};
                        if let Some(o) = mesh_outputs(&t.host.dxil)
                            .filter(|o| o.bytes() > METAL_MESH_OUTPUT_BYTES)
                            .filter(|_| crate::shader_cache::under_wine())
                        {
                            tracing::warn!(
                                hash = %self.hash,
                                bytes = o.bytes(),
                                limit = METAL_MESH_OUTPUT_BYTES,
                                vertices = o.vertices,
                                primitives = o.primitives,
                                vertex_components = o.vertex_components,
                                primitive_components = o.primitive_components,
                                "mesh shader outputs exceed Metal's limit; its draws are likely dropped"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        cached,
                        ms,
                        user_ms,
                        kernel_ms,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

/// This thread's processor time so far, user and kernel, in 100 ns units.
#[cfg(windows)]
fn thread_cpu() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let mut t = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [created, exited, kernel, user] = &mut t;
    // SAFETY: four times of our own thread into our own variables.
    unsafe { GetThreadTimes(GetCurrentThread(), created, exited, kernel, user) };
    let v = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    (v(user), v(kernel))
}

/// This thread's processor time since `then` ([`thread_cpu`]): user and
/// kernel milliseconds. Against the wall time, it tells work from waiting
/// (for a processor, a lock, the disk).
#[cfg(windows)]
fn thread_cpu_since(then: (u64, u64)) -> (u64, u64) {
    let now = thread_cpu();
    ((now.0 - then.0) / 10_000, (now.1 - then.1) / 10_000)
}

/// The translation, and whether it came from the cache.
#[cfg(windows)]
fn translate(container: &[u8]) -> (Result<Translated, String>, bool) {
    let (host, cached) = crate::shader_cache::host_shader(container);
    (host.and_then(|host| with_resources(container, host)), cached)
}

#[cfg(windows)]
fn with_resources(
    container: &[u8],
    host: victory_shader::emit::HostShader,
) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let used = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
                )
            })
            .map(|p| p.used)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    Ok(Translated { host, used })
}

#[cfg(windows)]
fn link_mesh_outputs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // On by default under Wine (D3DMetal): an over-limit mesh pipeline there
    // is made, draws nothing and its submission never completes (#172).
    // mac: and whenever the rows of a linked pair are merged
    // (`merge_mesh_rows`), which is a way of linking.
    *ON.get_or_init(|| {
        merge_mesh_rows() != 0
            || match std::env::var_os("VICTORY_LINK_MESH_OUTPUTS") {
                Some(v) => crate::switch::value_is_on(Some(&v)),
                None => crate::shader_cache::under_wine(),
            }
    })
}

/// macOS port, `VICTORY_MERGE_MESH_ROWS`: a linked mesh and pixel pair also
/// gets each output row that holds several elements as one element (1), or
/// every row as a whole `float4` (2), or as 1 with the rows reordered (3), or
/// as 3 with the clip and cull distances in the last registers (4, a test),
/// or as 4 with the three-component rows first, `float4` rows last (5, a test),
/// or in the order that is smallest by the measured rule, with components
/// the pixel shader never reads left out of the elements around them (6),
/// or as 6 with the clip distances the mesh program writes as a constant
/// that is not negative, which clip nothing, left out (8, exact: the last
/// two ground pairs then fit with the clipper), or as 8 with the clip
/// distances of a pair still over Metal's limit passed as an ordinary
/// element that the pixel shader discards by (7, not exact, and costly
/// where it applies: such a pixel shader loses its early depth test);
/// see `victory_shader::emit::translate_linked`.
#[cfg(windows)]
pub(crate) fn merge_mesh_rows() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        // On under D3DMetal, whose limit this is for, at level 8: the
        // measured smallest order, and clip distances a mesh program writes
        // as a constant that clips nothing left out of the pair. Exact; all
        // four pairs that were no-ops draw, with the early depth test kept
        // (run 26 November 17:47: no size error, 270 to 350 ms of GPU time
        // in the prison yard, as at level 6). `=7` adds a pixel-shader clip
        // for a pair still over the limit (none now), `=6` is the order
        // alone, `=3` the reorder before the measured rule, `=0` off.
        let default = if crate::shader_cache::under_wine() { 8 } else { 0 };
        std::env::var("VICTORY_MERGE_MESH_ROWS").ok().and_then(|v| v.parse().ok()).filter(|m| *m <= 8).unwrap_or(default)
    })
}

/// The output size, as VIctory counts it, above which a mesh shader is
/// translated again with its pixel shader: Metal's limit, or with
/// `VICTORY_MERGE_MESH_ROWS` a lower mark (`VICTORY_LINK_MESH_OVER`, bytes;
/// 24576 if unset), because D3DMetal counts rows of several elements larger.
#[cfg(windows)]
pub(crate) fn link_mesh_over() -> u32 {
    static OVER: OnceLock<u32> = OnceLock::new();
    *OVER.get_or_init(|| {
        if merge_mesh_rows() == 0 {
            return victory_shader::emit::METAL_MESH_OUTPUT_BYTES;
        }
        std::env::var("VICTORY_LINK_MESH_OVER").ok().and_then(|v| v.parse().ok()).unwrap_or(24576)
    })
}

/// Whether a mesh pipeline whose outputs are still over Metal's 32 KiB after
/// linking is refused rather than made. On by default under Wine: on
/// D3DMetal (3.0 and 4.0 beta 3) such a pipeline hangs the queue for good,
/// and the title's watchdog then ends the run (#172). Its draws are
/// dropped instead. `VICTORY_MESH_OVER_LIMIT=1` makes it anyway.
#[cfg(windows)]
pub(crate) fn refuse_over_limit_mesh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_MESH_OVER_LIMIT") {
        Some(v) => !crate::switch::value_is_on(Some(&v)),
        None => crate::shader_cache::under_wine(),
    })
}

#[cfg(windows)]
struct LinkedPair {
    mesh: Arc<Shader>,
    pixel: Option<Arc<Shader>>,
}

#[cfg(windows)]
type LinkedKey = ([u8; 32], Option<[u8; 32]>);

#[cfg(windows)]
impl Shader {
    fn with_translation(&self, translated: Translated) -> Arc<Shader> {
        let cell = OnceLock::new();
        let _ = cell.set(Some(translated));
        Arc::new(Shader {
            stage: self.stage,
            hash: self.hash.clone(),
            full_hash: self.full_hash,
            size: self.size,
            container: Mutex::new(Vec::new()),
            spilled: OnceLock::new(),
            translated: cell,
        })
    }
}

/// Re-translates a mesh shader together with its paired pixel shader (if any)
/// so arbitrary mesh outputs unread by the pixel shader are pruned under
/// Metal's 32 KiB mesh output limit (`victory_shader::emit::METAL_MESH_OUTPUT_BYTES`).
#[cfg(windows)]
pub(crate) fn translate_linked(
    ms: &Shader,
    ps: Option<&Shader>,
) -> Option<(Arc<Shader>, Option<Arc<Shader>>)> {
    static LINKED: Mutex<Option<HashMap<LinkedKey, Option<LinkedPair>>>> = Mutex::new(None);
    let key = (ms.full_hash, ps.map(|p| p.full_hash));
    if let Some(hit) = LINKED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return hit
            .as_ref()
            .map(|pair| (pair.mesh.clone(), pair.pixel.clone()));
    }
    let mesh_bytes = ms
        .container
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let pixel_bytes = ps.map(|p| {
        p.container
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    });
    let options = crate::shader_cache::runtime_options();
    let result = match victory_shader::emit::translate_linked(
        &mesh_bytes,
        pixel_bytes.as_deref(),
        options,
        merge_mesh_rows(),
    ) {
        Ok((ms_host, ps_host)) => {
            let outputs = victory_shader::emit::mesh_outputs(&ms_host.dxil);
            let ms_bytes = outputs.map(|o| o.bytes());
            // Metal's own count (`MeshOutputs::metal_bytes`, measured):
            // what D3DMetal refuses the pipeline for when it is over.
            let metal_bytes = outputs.map(|o| o.metal_bytes());
            let ms_used = ms.translated()?.used.clone();
            let mesh = ms.with_translation(Translated {
                host: ms_host,
                used: ms_used,
            });
            let pixel = match (ps, ps_host) {
                (Some(p), Some(host)) => Some(p.with_translation(Translated {
                    host,
                    used: p.translated()?.used.clone(),
                })),
                _ => None,
            };
            tracing::info!(
                mesh = %ms.hash,
                pixel = ps.map_or("none", |p| p.hash.as_str()),
                bytes = ms_bytes,
                metal_bytes,
                "mesh and pixel linked under 32 KiB output budget"
            );
            if metal_bytes.is_some_and(|b| b > victory_shader::emit::METAL_MESH_OUTPUT_BYTES) {
                tracing::warn!(
                    mesh = %ms.hash,
                    pixel = ps.map_or("none", |p| p.hash.as_str()),
                    metal_bytes,
                    "linked mesh outputs are over Metal's 32768 bytes: under D3DMetal the pipeline draws nothing"
                );
            }
            Some(LinkedPair { mesh, pixel })
        }
        Err(e) => {
            tracing::warn!(
                mesh = %ms.hash,
                pixel = ps.map_or("none", |p| p.hash.as_str()),
                error = %e,
                "linked mesh/pixel translation failed; using standalone translation"
            );
            None
        }
    };
    let out = result
        .as_ref()
        .map(|pair| (pair.mesh.clone(), pair.pixel.clone()));
    LINKED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    out
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// Whether it is one of the title's acceleration-structure builder's
    /// ([`builder`]).
    #[cfg(windows)]
    builder: bool,
    /// What its programs are to the ambient stand-in ([`crate::ambient`]):
    /// bits, 0 for nothing (always while the stand-in is off).
    #[cfg(windows)]
    ambient: u8,
    /// Which ray pass `VICTORY_RAY_SKIP` leaves out it is
    /// ([`ray_pass_left_out`]): bits, 0 for none (always while it is off).
    #[cfg(windows)]
    ray_pass: u8,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.size, hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if it is skipped, one of its
/// shaders does not translate, the host refuses it, or it is still being made
/// (its first use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Whether a dispatch with the title pipeline is held back: it is the
/// acceleration-structure builder's, which does not run this frame
/// ([`builder::runs`]).
#[cfg(windows)]
pub(crate) fn held_back(this: This) -> bool {
    !this.is_null() && pipeline(this).builder && !builder::runs()
}

/// Whether a dispatch with the title pipeline is left out: it is one of
/// the ray passes `VICTORY_RAY_SKIP` names (`crate::ray_skip`), and the
/// acceleration-structure builder does not run this frame
/// ([`builder::runs`]), so there is nothing to trace against. Counted
/// there when it is.
#[cfg(windows)]
pub(crate) fn ray_pass_left_out(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    let kind = pipeline(this).ray_pass;
    if kind == 0 || builder::runs() {
        return false;
    }
    crate::ray_skip::left_out(kind);
    true
}

/// [`ray_pass_left_out`], asked ahead and not counted (mac: hoisted clamps
/// of indirect dispatches, `crate::list`).
#[cfg(windows)]
pub(crate) fn ray_pass_is_left_out(this: This) -> bool {
    !this.is_null() && pipeline(this).ray_pass != 0 && !builder::runs()
}

/// What the title pipeline's programs are to the ambient stand-in
/// ([`crate::ambient`]): bits, 0 for nothing.
#[cfg(windows)]
pub(crate) fn ambient_role(this: This) -> u8 {
    if this.is_null() { 0 } else { pipeline(this).ambient }
}

/// A title DLL was mapped at `base`, `size` bytes (the acceleration-
/// structure builder's pipelines are known by its image).
#[cfg(windows)]
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    builder::note_title_dll(file, base, size);
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics and by `shader
/// translated` lines.
#[cfg(any(windows, test))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

/// `VICTORY_SKIP_PIPELINES=<hash>,...`: a pipeline with any of these shaders
/// gets no host pipeline, so its draws and dispatches are dropped as those of
/// a shader that does not translate are. Read once; in every build (it was a
/// diagnostics switch), to leave the title's costly passes out of a normal
/// run. Unset, the list is empty and [`skipped_shader`] returns at its first
/// line: once per pipeline the title creates, nothing per draw.
#[cfg(windows)]
fn skip_hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        let hashes = parse_skip_hashes(&value);
        if let Some(line) = skip_line(&value, hashes.len()) {
            tracing::info!("{line}");
        }
        hashes
    })
}

/// The line logged once when `VICTORY_SKIP_PIPELINES` is set to `value` and
/// `hashes` of its entries are shader hashes; none when it is unset or empty.
#[cfg(any(windows, test))]
fn skip_line(value: &str, hashes: usize) -> Option<String> {
    (!value.trim().is_empty()).then(|| {
        format!(
            "VICTORY_SKIP_PIPELINES: pipelines with any of {hashes} shader hashes get no host pipeline; \
             their draws and dispatches are dropped"
        )
    })
}

/// Reads `VICTORY_SKIP_PIPELINES` now (the host device is ready), so that its
/// line is logged at start and not with the title's first pipeline.
#[cfg(windows)]
pub(crate) fn note_skips() {
    let _ = skip_hashes();
}

#[cfg(windows)]
fn skipped_shader<'a>(shaders: &'a [Arc<Shader>], hashes: &[String]) -> Option<&'a str> {
    if hashes.is_empty() {
        return None;
    }
    shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && pipeline(this).host.skipped_shader.is_some()
}

/// The shaders of `this`, as host pipelines are named (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shaders_of(this: This) -> String {
    if this.is_null() {
        return "no pipeline".into();
    }
    format!("{:?}", pipeline(this).shaders)
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        p.host
            .skipped_shader
            .as_deref()
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops
        .get_or_insert_with(HashMap::new)
        .entry(why)
        .or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    // SAFETY: the title passes `size` bytes of shader code.
    code_of(code).map(|bytes| shader(stage, unsafe { &*bytes }))
}

/// The bytes a stage's bytecode names, if any.
fn code_of(code: &Bytecode) -> Option<*const [u8]> {
    (!code.code.is_null() && code.size != 0)
        .then(|| std::ptr::slice_from_raw_parts(code.code, code.size))
}

/// `code`: the title's bytecode pointers (by which a pipeline of the
/// acceleration-structure builder is known, [`builder`]).
fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    code: &[*const u8],
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let builder = !code.is_empty() && builder::created_by_builder(code);
    #[cfg(not(windows))]
    let _ = code;
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(
        shaders.clone(),
        graphics,
        skip_hashes(),
    ));
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    if builder {
        builder::note(&host);
    }
    #[cfg(windows)]
    let ambient = crate::ambient::role(shaders.iter().map(|s| s.hash.as_str()));
    // Never a program the stand-in works after (it would not run).
    #[cfg(windows)]
    let ray_pass = if ambient == 0 {
        crate::ray_skip::kind(shaders.iter().map(|s| s.hash.as_str()))
    } else {
        0
    };
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        #[cfg(windows)]
        ambient,
        #[cfg(windows)]
        ray_pass,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        &[],
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        &[desc.stage.code],
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes a whole stream.
    let stream = unsafe { std::slice::from_raw_parts(desc.stream, desc.size) };
    #[allow(unused_variables)]
    let (shaders, graphics) = read_stream(stream, copy);
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        &[],
        iid,
        out,
    )
}

#[cfg(windows)]
type Graphics = host::GraphicsState;
#[cfg(not(windows))]
type Graphics = ();

/// A stream's shaders (each made by `shader` from its stage and bytecode)
/// and graphics state. Its pointers (shader code, input elements) must be
/// valid: the title's own, or the pre-build's.
fn read_stream(
    stream: &[u8],
    shader: impl Fn(&'static str, &Bytecode) -> Option<Arc<Shader>>,
) -> (Vec<Arc<Shader>>, Graphics) {
    let mut shaders = Vec::new();
    #[allow(unused_mut, clippy::let_unit_value)]
    let mut graphics = Graphics::default();
    let mut at = 0;
    while at + 8 <= stream.len() {
        let kind = u32::from_le_bytes(stream[at..at + 4].try_into().expect("4 bytes"));
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if at + size > stream.len() {
            break;
        }
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8, which the
            // slice does not promise to be aligned.
            let code = unsafe { stream.as_ptr().add(at + 8).cast::<Bytecode>().read_unaligned() };
            shaders.extend(shader(stage, &code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, stream.as_ptr().add(at))
        };
        at += size;
    }
    (shaders, graphics)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
mod builder;
#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;
#[cfg(any(windows, test))]
#[cfg_attr(not(windows), allow(dead_code))]
mod screen;
#[cfg(all(test, windows, feature = "diagnostics"))]
mod placed_uav_tests;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    if std::env::var_os("VICTORY_SPILL_SHADERS").is_some_and(|v| v != "0") {
        let spill = dir.with_file_name("shader-containers");
        if crate::make_dir(&spill).is_ok() {
            tracing::info!(dir = %spill.display(), "shader containers kept on disk until translated");
            let _ = SPILL_DIR.set(spill);
        }
    }
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

/// The pre-build: a title pipeline made ahead of a run into the pipeline
/// library (`use_pipeline_cache`), by the code a run makes it with, so the
/// run finds it there under the same name.
#[cfg(windows)]
pub struct Prebuild {
    shaders: Vec<Arc<Shader>>,
    graphics: Option<Box<host::GraphicsState>>,
}

// SAFETY: as `make::Pending`: the graphics state's input elements point
// into its own names, and it is only read.
#[cfg(windows)]
unsafe impl Send for Prebuild {}

#[cfg(windows)]
impl Prebuild {
    /// The pipeline a creation stream describes, as the title passes it to
    /// the device.
    ///
    /// # Safety
    /// The stream's pointers point at live shader code and input elements.
    pub unsafe fn graphics(stream: &[u8]) -> Prebuild {
        // Shaders of their own: a pre-build must not keep every container.
        let own = |stage, code: &Bytecode| {
            // SAFETY: the caller's pointers.
            code_of(code).map(|b| Arc::new(Shader::new(stage, unsafe { &*b })))
        };
        let (shaders, graphics) = read_stream(stream, own);
        Prebuild {
            shaders,
            graphics: Some(Box::new(graphics)),
        }
    }

    /// The compute pipeline of a container.
    pub fn compute(container: &[u8]) -> Prebuild {
        Prebuild {
            shaders: vec![Arc::new(Shader::new("compute", container))],
            graphics: None,
        }
    }

    /// What a run lists the pipeline by (its early list).
    pub fn key(&self) -> String {
        make::key(&self.shaders, self.graphics.as_deref())
    }

    /// Its shaders and state as a diagnostics build logs them.
    pub fn describe(&self) -> String {
        let state = self.graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
        format!("{:?}\t{state}", self.shaders)
    }

    /// Makes the host pipeline, or finds it in the library; why not if it
    /// gets none.
    pub fn make(&self) -> Result<(), String> {
        let key = self.key();
        library::keep_early_list();
        // Without the run's log lines per shader and pipeline: tens of
        // thousands of them would bury the pre-build's own report.
        let quiet = tracing::Dispatch::none();
        let made = tracing::dispatcher::with_default(&quiet, || {
            host::HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        #[cfg(feature = "diagnostics")]
        make::log_built(&key);
        if made.is_some() {
            // shortcut: a pipeline the library refused to store is listed
            // too; a run then makes it at first use.
            library::note_prebuilt(key);
        }
        library::save(false);
        match made {
            Some(_) => Ok(()),
            None => Err(match self.shaders.iter().find(|s| s.translated().is_none()) {
                Some(s) => format!("{} shader does not translate", s.stage),
                None => "the host refuses it".into(),
            }),
        }
    }
}

/// Host pipelines made, and loaded from the library, since the start.
#[cfg(windows)]
pub fn pipeline_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (library::MADE.load(Relaxed), library::LOADED.load(Relaxed))
}

/// Whether a pre-build has put the title pipeline `key` in the library
/// (a second pre-build skips it without translating or loading anything).
#[cfg(windows)]
pub fn prebuilt(key: &str) -> bool {
    library::prebuilt(key)
}

/// Whether an earlier run drew or dispatched with the title pipeline `key`.
#[cfg(windows)]
pub fn wanted_early(key: &str) -> bool {
    library::wanted_early(key)
}

#[cfg(test)]
mod tests {
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    /// Unset or empty, `VICTORY_SKIP_PIPELINES` gives no hashes and no line;
    /// set, one line with the count of hashes taken (0 if none was one).
    #[test]
    fn skip_line_only_when_set() {
        assert!(super::parse_skip_hashes("").is_empty());
        assert_eq!(super::skip_line("", 0), None);
        assert_eq!(super::skip_line("  ", 0), None);
        let value = "0123456789abcdef,0011223344556677,xyz";
        let hashes = super::parse_skip_hashes(value);
        let line = super::skip_line(value, hashes.len()).expect("a line when set");
        assert!(line.starts_with("VICTORY_SKIP_PIPELINES: ") && line.contains(" 2 shader hashes "), "{line}");
        let none = super::skip_line("xyz", 0).expect("a line when set");
        assert!(none.contains(" 0 shader hashes "), "{none}");
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// The slice a title's stream (or a pre-build's) is in need not be
    /// 8-aligned: its bytecode is read as it lies.
    #[test]
    fn a_stream_of_any_alignment_is_read() {
        let mut backing = [0u64; 5];
        // SAFETY: the 40 bytes of `backing`, one off its (aligned) start.
        let stream = unsafe { std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>().add(1), 32) };
        stream[..4].copy_from_slice(&1u32.to_le_bytes()); // vertex shader
        stream[8..16].copy_from_slice(&0x1000usize.to_le_bytes());
        stream[16..24].copy_from_slice(&7usize.to_le_bytes());
        assert_ne!(stream.as_ptr() as usize % 8, 0);
        let seen = std::cell::RefCell::new(Vec::new());
        read_stream(stream, |stage, code| {
            seen.borrow_mut().push((stage, code.code as usize, code.size));
            None
        });
        assert_eq!(*seen.borrow(), [("vertex", 0x1000, 7)]);
    }

    /// A spilled container leaves memory, is on disk whole, and one already
    /// there is kept rather than written again.
    #[cfg(windows)]
    #[test]
    fn spilled_containers_round_trip_through_their_file() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i * 31) as u8).collect();
        let full = *blake3::hash(&bytes).as_bytes();
        let s = Shader::new("pixel", &bytes);
        spill_to(dir.path(), &s, &full);
        assert!(s.container.lock().unwrap().is_empty(), "left memory");
        assert_eq!(s.spilled.get(), Some(&full));
        assert_eq!(crate::read_file(&spill_path(dir.path(), &full)).unwrap(), bytes);
        let again = Shader::new("pixel", &bytes);
        spill_to(dir.path(), &again, &full);
        assert!(again.container.lock().unwrap().is_empty() && again.spilled.get() == Some(&full));
    }

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.lock().unwrap().clone()))
            .collect()
    }

    /// A compute pipeline the acceleration-structure builder creates (its
    /// bytecode in the builder's image) is known as the builder's, and its
    /// dispatches are held back while its host pipeline is not made: here
    /// never, as its bytes are no shader container. Pipelines of other
    /// code are never held back.
    #[cfg(windows)]
    #[test]
    fn builder_dispatches_wait_for_every_builder_pipeline() {
        let _one = builder::test_lock();
        let code = [0xB0u8; 64];
        let create = |code: &[u8]| {
            let desc = ComputeHead {
                root: std::ptr::null(),
                stage: Bytecode {
                    code: code.as_ptr(),
                    size: code.len(),
                },
            };
            let mut p: This = std::ptr::null();
            assert_eq!(
                create_compute(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
                S_OK
            );
            p
        };
        builder::note_title_dll("xboxbvhbuilder.DLL", code.as_ptr() as usize, code.len());
        let p = create(&code);
        let other_code = [0xC0u8; 64];
        let other = create(&other_code);
        builder::note_title_dll("XboxBvhBuilder.dll", 0, 0);
        assert!(
            pipeline(p).builder,
            "its bytecode is in the builder's image"
        );
        assert!(!pipeline(other).builder);
        // Its host pipeline is started at once and refused.
        let refused = (0..500).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            make::state(&pipeline(p).host).is_some()
        });
        assert!(refused, "the builder's pipeline was started");
        assert_eq!(make::state(&pipeline(p).host), Some(false));
        assert!(
            held_back(p),
            "one builder pipeline missing: the builder waits"
        );
        assert!(!held_back(other));
        for q in [p, other] {
            slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
        }
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    #[cfg(windows)]
    #[test]
    fn translation_lets_the_container_go() {
        // Not a container: it does not translate, and goes all the same.
        let s = Shader::new("compute", &[1, 2, 3]);
        assert!(s.translated().is_none());
        assert!(s.container.lock().unwrap().is_empty());
        assert_eq!(s.size, 3);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }

    /// The pre-build reads a stream as the title's creation does: the same
    /// shaders and state, so the same key and the same host pipeline name.
    #[cfg(windows)]
    #[test]
    fn a_prebuild_is_the_pipeline_the_title_creates() {
        let (vs, ps) = ([7u8; 9], [5u8; 6]);
        let name = c"TEXCOORD";
        // One input element: name, index 2, format 16, slot 1, offset 8.
        let mut element = [0u8; 32];
        element[..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
        for (i, w) in [2u32, 16, 1, 8, 0, 0].iter().enumerate() {
            element[8 + 4 * i..12 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        let mut stream = vec![0u8; 0x2b0];
        let put = |s: &mut Vec<u8>, at: usize, v: u64| {
            s[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        let kinds = [(0x10, 1), (0x28, 2), (0x40, 3), (0x58, 4), (0x70, 5), (0x88, 8)];
        for (at, kind) in kinds.into_iter().chain([(0x210, 21), (0x250, 17)]) {
            put(&mut stream, at, kind);
        }
        put(&mut stream, 0x18, vs.as_ptr() as u64);
        put(&mut stream, 0x20, vs.len() as u64);
        put(&mut stream, 0x30, ps.as_ptr() as u64);
        put(&mut stream, 0x38, ps.len() as u64);
        put(&mut stream, 0x8c, 1 << 32); // independent blend
        put(&mut stream, 0x1d8, 9 | 0xffff_ffff << 32);
        put(&mut stream, 0x1e0, 10 | 3 << 32); // solid fill
        put(&mut stream, 0x254, 1);
        put(&mut stream, 0x260, 12);
        put(&mut stream, 0x268, element.as_ptr() as u64);
        put(&mut stream, 0x270, 1);
        put(&mut stream, 0x278, 14 | 4 << 32);
        put(&mut stream, 0x280, 15 | 87 << 32);
        put(&mut stream, 0x2a0, 1 << 32);
        put(&mut stream, 0x2a8, 16 | 40 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        assert_eq!(create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p), S_OK);
        // SAFETY: the stream points at live code and elements.
        let pre = unsafe { Prebuild::graphics(&stream) };
        let title = pipeline(p);
        assert_eq!(pre.key(), title.host.key);
        let state = title.host.graphics.as_deref().unwrap().describe();
        assert!(state.contains(r#"\"TEXCOORD\" 2 16 1 8 0 0"#), "{state}");
        assert_eq!(pre.describe(), format!("{:?}\t{state}", title.shaders));
        // Another state is another pipeline.
        put(&mut stream, 0x1e0, 10 | 2 << 32);
        // SAFETY: as above.
        assert_ne!(unsafe { Prebuild::graphics(&stream) }.key(), title.host.key);
    }
}

#[cfg(all(test, windows))]
mod depth_bounds_test;
