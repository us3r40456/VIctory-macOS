//! Direct Metal batched command-stream encoder (`CommandStream`) and persistent
//! `MTLBinaryArchive` management (Step 2B-3).
//!
//! Serializes an entire frame's compute, blit, and graphics commands into a
//! single flat `Vec<u32>` buffer submitted through one Wine unixlib call
//! (`OP_SUBMIT_COMMANDS`), eliminating per-draw Wine/D3DMetal stalls.

pub const CMD_COMPUTE_BEGIN: u32 = 1;
pub const CMD_COMPUTE_END: u32 = 2;
pub const CMD_SET_COMPUTE_PSO: u32 = 3;
pub const CMD_SET_COMPUTE_BUFFER: u32 = 4;
pub const CMD_SET_COMPUTE_BYTES: u32 = 5;
pub const CMD_SET_COMPUTE_TEXTURE: u32 = 6;
pub const CMD_DISPATCH_TG: u32 = 7;
pub const CMD_DISPATCH_INDIRECT: u32 = 8;
pub const CMD_BLIT_COPY_BUFFER: u32 = 9;
pub const CMD_BLIT_FILL_BUFFER: u32 = 10;
pub const CMD_RENDER_BEGIN: u32 = 11;
pub const CMD_RENDER_END: u32 = 12;
pub const CMD_SET_RENDER_PSO: u32 = 13;
pub const CMD_SET_VIEWPORT: u32 = 14;
pub const CMD_SET_VERTEX_BUFFER: u32 = 15;
pub const CMD_SET_VERTEX_BYTES: u32 = 16;
pub const CMD_SET_FRAGMENT_BUFFER: u32 = 17;
pub const CMD_SET_FRAGMENT_BYTES: u32 = 18;
pub const CMD_SET_FRAGMENT_TEXTURE: u32 = 19;
pub const CMD_DRAW_PRIMITIVES: u32 = 20;
pub const CMD_DRAW_INDEXED: u32 = 21;
pub const CMD_MEMORY_BARRIER: u32 = 22;

/// Metal primitive topology (`MTLPrimitiveType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PrimitiveType {
    Point = 0,
    Line = 1,
    LineStrip = 2,
    Triangle = 3,
    TriangleStrip = 4,
}

/// Metal render-pass load action (`MTLLoadAction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LoadAction {
    DontCare = 0,
    Load = 1,
    Clear = 2,
}

/// Metal index buffer element type (`MTLIndexType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    UInt16 = 0,
    UInt32 = 1,
}

/// Flat packed binary command buffer builder for `OP_SUBMIT_COMMANDS`.
#[derive(Default, Clone)]
pub struct CommandStream {
    words: Vec<u32>,
}

impl CommandStream {
    pub fn with_capacity(words: usize) -> Self {
        Self {
            words: Vec::with_capacity(words),
        }
    }

    pub fn clear(&mut self) {
        self.words.clear();
    }

    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    #[inline]
    fn push_u64(&mut self, v: u64) {
        self.words.push(v as u32);
        self.words.push((v >> 32) as u32);
    }

    #[inline]
    fn push_f32(&mut self, v: f32) {
        self.words.push(v.to_bits());
    }

    #[inline]
    fn push_bytes(&mut self, data: &[u8]) {
        self.words.push(data.len() as u32);
        let full = data.chunks_exact(4);
        let rem = full.remainder();
        for c in full {
            self.words
                .push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        if !rem.is_empty() {
            let mut tail = [0u8; 4];
            tail[..rem.len()].copy_from_slice(rem);
            self.words.push(u32::from_le_bytes(tail));
        }
    }

    pub fn compute_begin(&mut self) -> &mut Self {
        self.words.push(CMD_COMPUTE_BEGIN);
        self
    }

    pub fn compute_end(&mut self) -> &mut Self {
        self.words.push(CMD_COMPUTE_END);
        self
    }

    pub fn set_compute_pso(&mut self, pso: u64) -> &mut Self {
        self.words.push(CMD_SET_COMPUTE_PSO);
        self.push_u64(pso);
        self
    }

    pub fn set_compute_buffer(&mut self, slot: u32, buf: u64, offset: u64) -> &mut Self {
        self.words.push(CMD_SET_COMPUTE_BUFFER);
        self.words.push(slot);
        self.push_u64(buf);
        self.push_u64(offset);
        self
    }

    pub fn set_compute_bytes(&mut self, slot: u32, data: &[u8]) -> &mut Self {
        self.words.push(CMD_SET_COMPUTE_BYTES);
        self.words.push(slot);
        self.push_bytes(data);
        self
    }

    pub fn set_compute_texture(&mut self, slot: u32, tex: u64) -> &mut Self {
        self.words.push(CMD_SET_COMPUTE_TEXTURE);
        self.words.push(slot);
        self.push_u64(tex);
        self
    }

    pub fn dispatch_tg(&mut self, groups: [u32; 3], threads_per_tg: [u32; 3]) -> &mut Self {
        self.words.push(CMD_DISPATCH_TG);
        self.words.extend_from_slice(&groups);
        self.words.extend_from_slice(&threads_per_tg);
        self
    }

    pub fn dispatch_indirect(
        &mut self,
        buf: u64,
        offset: u64,
        threads_per_tg: [u32; 3],
    ) -> &mut Self {
        self.words.push(CMD_DISPATCH_INDIRECT);
        self.push_u64(buf);
        self.push_u64(offset);
        self.words.extend_from_slice(&threads_per_tg);
        self
    }

    pub fn blit_copy_buffer(
        &mut self,
        src: u64,
        src_offset: u64,
        dst: u64,
        dst_offset: u64,
        len: u64,
    ) -> &mut Self {
        self.words.push(CMD_BLIT_COPY_BUFFER);
        self.push_u64(src);
        self.push_u64(src_offset);
        self.push_u64(dst);
        self.push_u64(dst_offset);
        self.push_u64(len);
        self
    }

    pub fn blit_fill_buffer(&mut self, dst: u64, offset: u64, len: u64, value: u8) -> &mut Self {
        self.words.push(CMD_BLIT_FILL_BUFFER);
        self.push_u64(dst);
        self.push_u64(offset);
        self.push_u64(len);
        self.words.push(u32::from(value));
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_begin(
        &mut self,
        color_tex: u64,
        load_action: LoadAction,
        clear_rgba: [f32; 4],
        depth_tex: u64,
        depth_load: LoadAction,
        clear_depth: f32,
    ) -> &mut Self {
        self.words.push(CMD_RENDER_BEGIN);
        self.push_u64(color_tex);
        self.words.push(load_action as u32);
        for c in clear_rgba {
            self.push_f32(c);
        }
        self.push_u64(depth_tex);
        self.words.push(depth_load as u32);
        self.push_f32(clear_depth);
        self
    }

    pub fn render_end(&mut self) -> &mut Self {
        self.words.push(CMD_RENDER_END);
        self
    }

    pub fn set_render_pso(&mut self, pso: u64) -> &mut Self {
        self.words.push(CMD_SET_RENDER_PSO);
        self.push_u64(pso);
        self
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, zmin: f32, zmax: f32) -> &mut Self {
        self.words.push(CMD_SET_VIEWPORT);
        self.push_f32(x);
        self.push_f32(y);
        self.push_f32(w);
        self.push_f32(h);
        self.push_f32(zmin);
        self.push_f32(zmax);
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buf: u64, offset: u64) -> &mut Self {
        self.words.push(CMD_SET_VERTEX_BUFFER);
        self.words.push(slot);
        self.push_u64(buf);
        self.push_u64(offset);
        self
    }

    pub fn set_vertex_bytes(&mut self, slot: u32, data: &[u8]) -> &mut Self {
        self.words.push(CMD_SET_VERTEX_BYTES);
        self.words.push(slot);
        self.push_bytes(data);
        self
    }

    pub fn set_fragment_buffer(&mut self, slot: u32, buf: u64, offset: u64) -> &mut Self {
        self.words.push(CMD_SET_FRAGMENT_BUFFER);
        self.words.push(slot);
        self.push_u64(buf);
        self.push_u64(offset);
        self
    }

    pub fn set_fragment_bytes(&mut self, slot: u32, data: &[u8]) -> &mut Self {
        self.words.push(CMD_SET_FRAGMENT_BYTES);
        self.words.push(slot);
        self.push_bytes(data);
        self
    }

    pub fn set_fragment_texture(&mut self, slot: u32, tex: u64) -> &mut Self {
        self.words.push(CMD_SET_FRAGMENT_TEXTURE);
        self.words.push(slot);
        self.push_u64(tex);
        self
    }

    pub fn draw_primitives(
        &mut self,
        prim: PrimitiveType,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
    ) -> &mut Self {
        self.words.push(CMD_DRAW_PRIMITIVES);
        self.words.push(prim as u32);
        self.words.push(vertex_start);
        self.words.push(vertex_count);
        self.words.push(instance_count);
        self
    }

    pub fn draw_indexed(
        &mut self,
        prim: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: u64,
        index_offset: u64,
        instance_count: u32,
    ) -> &mut Self {
        self.words.push(CMD_DRAW_INDEXED);
        self.words.push(prim as u32);
        self.words.push(index_count);
        self.words.push(index_type as u32);
        self.push_u64(index_buffer);
        self.push_u64(index_offset);
        self.words.push(instance_count);
        self
    }

    pub fn memory_barrier(&mut self, scope: u32) -> &mut Self {
        self.words.push(CMD_MEMORY_BARRIER);
        self.words.push(scope);
        self
    }
}

/// Converts a Wine `Z:\...` path into a macOS POSIX path so `victory_metal.so`
/// reads and writes `MTLBinaryArchive` files directly on the host filesystem.
pub fn wine_path_to_posix(path: &std::path::Path) -> Option<String> {
    let s = path.to_string_lossy();
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('/') {
        return Some(trimmed.replace('\\', "/"));
    }
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 3 && (bytes[0] == b'Z' || bytes[0] == b'z') && bytes[1] == b':' {
        let rest = &trimmed[2..];
        let posix = rest.replace('\\', "/");
        return Some(if posix.starts_with('/') {
            posix
        } else {
            format!("/{posix}")
        });
    }
    None
}

#[cfg(windows)]
mod archive_global {
    use std::path::Path;
    use std::sync::Mutex;

    use super::wine_path_to_posix;
    use crate::metal::bridge::{self, ShaderSourceKind};

    struct GlobalArchive {
        handle: u64,
        posix_path: String,
        dirty: bool,
        compiled: u64,
    }

    static ARCHIVE: Mutex<Option<GlobalArchive>> = Mutex::new(None);

    /// Opens `<dir>/victory-metal.mtlbarchive` (when `VICTORY_METAL_ARCHIVE=1`
    /// or `VICTORY_BACKEND=metal`) so precompiled Metal shaders persist on disk.
    pub fn open_global_archive(cache_dir: &Path) {
        if !crate::metal::metal_archive_enabled() {
            return;
        }
        let Some(b) = bridge::global() else {
            return;
        };
        let file = cache_dir.join("victory-metal.mtlbarchive");
        let Some(posix) = wine_path_to_posix(&file) else {
            return;
        };
        match b.open_archive(Some(&posix)) {
            Ok((handle, loaded)) => {
                tracing::info!(
                    path = %posix,
                    loaded_existing = loaded,
                    "victory_metal: MTLBinaryArchive ready"
                );
                let mut g = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(old) = g.replace(GlobalArchive {
                    handle,
                    posix_path: posix,
                    dirty: false,
                    compiled: 0,
                }) {
                    b.close_archive(old.handle);
                }
            }
            Err(e) => {
                tracing::warn!(err = e, path = %posix, "victory_metal: could not open MTLBinaryArchive");
            }
        }
    }

    /// Compiles a DXIL compute container into the global `MTLBinaryArchive`
    /// (called alongside D3D12 pipeline creation when `VICTORY_METAL_ARCHIVE=1`).
    pub fn warm_dxil_compute(dxil: &[u8], entry: &str) -> bool {
        if !crate::metal::metal_archive_enabled() {
            return false;
        }
        let Some(b) = bridge::global() else {
            return false;
        };
        let arch = {
            let g = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
            g.as_ref().map(|a| a.handle).unwrap_or(0)
        };
        if arch == 0 {
            return false;
        }
        match b.create_compute_pipeline(dxil, entry, ShaderSourceKind::Dxil, arch) {
            Ok(info) => {
                b.destroy_pipeline(info.handle);
                let mut g = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(a) = g.as_mut() {
                    a.dirty = true;
                    a.compiled += 1;
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Serializes the global `MTLBinaryArchive` to disk if new pipelines were added.
    pub fn save_global_archive() {
        let Some(b) = bridge::global() else {
            return;
        };
        let mut g = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(a) = g.as_mut() else {
            return;
        };
        if !a.dirty {
            return;
        }
        match b.save_archive(a.handle, Some(&a.posix_path)) {
            Ok(()) => {
                a.dirty = false;
                tracing::info!(
                    path = %a.posix_path,
                    compiled = a.compiled,
                    "victory_metal: saved MTLBinaryArchive"
                );
            }
            Err(e) => {
                tracing::warn!(
                    err = e,
                    path = %a.posix_path,
                    "victory_metal: failed to serialize MTLBinaryArchive"
                );
            }
        }
    }
}

#[cfg(windows)]
pub use archive_global::{open_global_archive, save_global_archive, warm_dxil_compute};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_stream_packs_compute_and_render_words() {
        let mut s = CommandStream::with_capacity(64);
        s.compute_begin()
            .set_compute_pso(0x1122_3344_5566_7788)
            .set_compute_buffer(0, 42, 256)
            .set_compute_bytes(1, &[1, 2, 3, 4, 5])
            .dispatch_tg([4, 2, 1], [64, 1, 1])
            .compute_end()
            .render_begin(7, LoadAction::Clear, [0.1, 0.2, 0.3, 1.0], 0, LoadAction::DontCare, 1.0)
            .set_render_pso(99)
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1)
            .render_end();
        let w = s.as_words();
        assert_eq!(w[0], CMD_COMPUTE_BEGIN);
        assert_eq!(w[1], CMD_SET_COMPUTE_PSO);
        assert_eq!(w[2], 0x5566_7788);
        assert_eq!(w[3], 0x1122_3344);
        assert_eq!(*w.last().unwrap(), CMD_RENDER_END);
    }

    #[test]
    fn wine_z_drive_path_maps_to_macos_posix_path() {
        assert_eq!(
            wine_path_to_posix(std::path::Path::new(r"Z:\Users\dylan\cache\pipelines")),
            Some("/Users/dylan/cache/pipelines".into())
        );
        assert_eq!(
            wine_path_to_posix(std::path::Path::new("/tmp/test.mtlbarchive")),
            Some("/tmp/test.mtlbarchive".into())
        );
    }

    #[cfg(windows)]
    #[test]
    fn metal_backend_dxil_compute_and_binary_archive_roundtrip() {
        let Some(bridge) = crate::metal::bridge::global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        if !bridge.info().irconverter {
            eprintln!("skipping: libmetalirconverter.dylib unavailable");
            return;
        }
        #[cfg(victory_dxc_detile)]
        {
            let dxil: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpudetile.dxil"));
            let archive_path = format!(
                "/tmp/victory-metal-test-archive-{}.mtlbarchive",
                std::process::id()
            );
            let (arch1, existed1) = bridge
                .open_archive(Some(&archive_path))
                .expect("create MTLBinaryArchive");
            assert!(!existed1);

            let pso1 = bridge
                .create_compute_pipeline(
                    dxil,
                    "main",
                    crate::metal::bridge::ShaderSourceKind::Dxil,
                    arch1,
                )
                .expect("compile gpudetile.dxil -> MTLComputePipelineState");
            assert!(pso1.max_threads_per_tg >= 64);
            assert!(pso1.simd_width >= 32);

            bridge
                .save_archive(arch1, Some(&archive_path))
                .expect("serialize MTLBinaryArchive to disk");
            bridge.destroy_pipeline(pso1.handle);
            bridge.close_archive(arch1);

            // Reopen serialized binary archive from disk and re-create pipeline.
            let (arch2, existed2) = bridge
                .open_archive(Some(&archive_path))
                .expect("reopen MTLBinaryArchive");
            assert!(existed2, "reopened archive must load existing file from disk");
            let t0 = std::time::Instant::now();
            let pso2 = bridge
                .create_compute_pipeline(
                    dxil,
                    "main",
                    crate::metal::bridge::ShaderSourceKind::Dxil,
                    arch2,
                )
                .expect("create pipeline backed by reloaded MTLBinaryArchive");
            let reload_ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "victory_metal MTLBinaryArchive DXIL roundtrip: max_tg={} simd={} reload={:.2} ms",
                pso2.max_threads_per_tg, pso2.simd_width, reload_ms
            );
            bridge.destroy_pipeline(pso2.handle);
            bridge.close_archive(arch2);
            let _ = std::fs::remove_file(format!("Z:{}", archive_path.replace('/', "\\")));
        }
    }

    /// Heavy-scene benchmark (#221): encodes a zero-copy 16 MiB VirtualAlloc
    /// compute fill + 2,500 real graphics draws onto a 1920x1080 render target
    /// inside ONE direct Metal CommandStream batch and verifies exact pixels.
    #[cfg(windows)]
    #[test]
    fn metal_backend_2500_draws_and_zero_copy_compute_in_single_batch() {
        use crate::metal::bridge::ShaderSourceKind;
        use crate::metal::pixel_format as pf;
        use crate::metal::registry::BackingKey;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let Some(bridge) = crate::metal::bridge::global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };

        const COMPUTE_MSL: &[u8] = br#"
#include <metal_stdlib>
using namespace metal;
kernel void fill_const(device float4 *dst [[buffer(0)]], constant uint &count [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        dst[gid] = float4(0.25f, 0.75f, 0.50f, 1.0f);
    }
}
"#;
        const RENDER_MSL: &[u8] = br#"
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 col; };
vertex VOut vs_tri(uint vid [[vertex_id]], device const float4 *colors [[buffer(0)]], constant uint &draw_idx [[buffer(1)]]) {
    float2 p[3] = { float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f) };
    VOut o;
    o.pos = float4(p[vid], 0.0f, 1.0f);
    o.col = colors[draw_idx & 1023u];
    return o;
}
fragment float4 fs_tri(VOut in [[stage_in]]) {
    return in.col;
}
"#;

        let c_pso = bridge
            .create_compute_pipeline(COMPUTE_MSL, "fill_const", ShaderSourceKind::Msl, 0)
            .expect("compile compute MSL");
        let r_pso = bridge
            .create_render_pipeline(
                RENDER_MSL,
                "vs_tri",
                Some(RENDER_MSL),
                "fs_tri",
                ShaderSourceKind::Msl,
                pf::RGBA8_UNORM,
                0,
                false,
                0,
            )
            .expect("compile render MSL");

        const BUF_LEN: usize = 16 << 20; // 16 MiB zero-copy VirtualAlloc
        // SAFETY: standalone VirtualAlloc released at end of test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                BUF_LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!ptr.is_null());
        let base = ptr as u64;
        let (buf, _) = bridge
            .register_buffer(base, BUF_LEN as u64, BackingKey::VirtualAlloc(base), false)
            .expect("register 16 MiB zero-copy VirtualAlloc");

        let (w, h) = (1920u32, 1080u32);
        let rt = bridge
            .create_texture(w, h, pf::RGBA8_UNORM, false)
            .expect("create 1080p render target");

        const DRAWS: u32 = 2500;
        let mut stream = CommandStream::with_capacity(DRAWS as usize * 12 + 64);
        let elem_count: u32 = 4096;
        stream
            .compute_begin()
            .set_compute_pso(c_pso.handle)
            .set_compute_buffer(0, buf.handle, 0)
            .set_compute_bytes(1, &elem_count.to_le_bytes())
            .dispatch_tg([elem_count / 64, 1, 1], [64, 1, 1])
            .compute_end()
            .render_begin(rt, LoadAction::Clear, [0.0, 0.0, 0.0, 1.0], 0, LoadAction::DontCare, 1.0)
            .set_render_pso(r_pso)
            .set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0)
            .set_vertex_buffer(0, buf.handle, 0);
        for d in 0..DRAWS {
            stream
                .set_vertex_bytes(1, &d.to_le_bytes())
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1);
        }
        stream.render_end();

        // Warm-up + measured submission of all 2,500 draws in ONE unixlib call!
        let _ = bridge.submit_commands(stream.as_words(), true).expect("warmup submit");
        let res = bridge
            .submit_commands(stream.as_words(), true)
            .expect("2500-draw direct Metal submit");
        assert_eq!(res.draws_encoded, DRAWS);
        assert_eq!(res.dispatches_encoded, 1);

        let mut pixels = vec![[0u8; 4]; (w * h) as usize];
        assert!(bridge.readback_texture_to_ptr(rt, pixels.as_mut_ptr().cast(), w * 4, w, h));
        let center = pixels[((h / 2) * w + (w / 2)) as usize];
        assert!(
            center[0].abs_diff(64) <= 2
                && center[1].abs_diff(191) <= 2
                && center[2].abs_diff(128) <= 2
                && center[3] == 255,
            "2500-draw render target center pixel mismatch: {center:?}"
        );

        let enc_ms = res.encode_ns as f64 / 1e6;
        let gpu_ms = res.gpu_ns as f64 / 1e6;
        let wall_ms = res.wall_ns as f64 / 1e6;
        let equiv_fps = 1000.0 / wall_ms.max(0.001);
        eprintln!(
            "victory_metal 2500 draws + 16 MiB zero-copy compute @ 1080p: encode {:.2} ms, gpu {:.2} ms, wall {:.2} ms (~{:.0} FPS!), center={center:?}",
            enc_ms, gpu_ms, wall_ms, equiv_fps
        );

        bridge.destroy_texture(rt);
        bridge.destroy_pipeline(r_pso);
        bridge.destroy_pipeline(c_pso.handle);
        assert!(bridge.unregister_view(base, false));
        // SAFETY: `ptr` came from `VirtualAlloc`.
        unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
    }
}
