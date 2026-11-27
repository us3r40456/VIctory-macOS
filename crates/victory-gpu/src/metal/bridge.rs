//! Live Windows PE -> CrossOver Wine unixlib bridge (`victory_metal.dll` ->
//! `victory_metal.so`).
//!
//! Loads `victory_metal.dll` dynamically from `WINEDLLPATH` so stock CrossOver
//! Wine requires zero rebuilds. Every call goes directly through Wine's
//! `__wine_unix_call_dispatcher` (~33 ns round-trip).

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

use super::registry::{BackingKey, CanonicalBuffer, CanonicalRegistry, RegisterAction, ResolvedBuffer};
use super::{
    BenchArgs, ConvertDxilArgs, FlushResidencyArgs, OP_BENCH_ZERO_COPY, OP_CONVERT_DXIL,
    OP_FLUSH_RESIDENCY, OP_PING, OP_REGISTER_BUFFER, OP_STATS, OP_UNREGISTER_BUFFER, PingArgs,
    PingInfo, RegisterBufferArgs, Stats, UnregisterBufferArgs,
};

type FnAvailable = unsafe extern "system" fn() -> i32;
type FnCall = unsafe extern "system" fn(u32, *mut c_void) -> i32;

/// Loaded `victory_metal.dll` entry points + global canonical buffer registry.
pub struct MetalBridge {
    call: FnCall,
    ping: PingInfo,
    registry: Mutex<CanonicalRegistry>,
}

// SAFETY: `FnCall` is a process-lifetime function pointer into the loaded DLL
// and `victory_metal.so` serializes all Metal state internally.
unsafe impl Send for MetalBridge {}
// SAFETY: as above.
unsafe impl Sync for MetalBridge {}

impl MetalBridge {
    /// Attempts to load `victory_metal.dll` and ping the Apple Silicon Metal
    /// unixlib. Returns `None` outside macOS/Wine or if `victory_metal.dll`
    /// is not on `WINEDLLPATH`.
    fn load() -> Option<Self> {
        let dll_path = std::env::var("VICTORY_METAL_DLL")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| std::ffi::CString::new(s).ok());
        // SAFETY: NUL-terminated path passed to Win32 loader APIs.
        let (available, call) = unsafe {
            let mut module = std::ptr::null_mut();
            if let Some(ref path) = dll_path {
                module = LoadLibraryA(path.as_ptr().cast());
            }
            if module.is_null() {
                module = LoadLibraryA(c"victory_metal.dll".as_ptr().cast());
            }
            if module.is_null() {
                tracing::debug!(
                    "victory_metal.dll not found (checked VICTORY_METAL_DLL and WINEDLLPATH); Metal unixlib bridge inactive"
                );
                return None;
            }
            let p_avail = GetProcAddress(module, c"victory_metal_available".as_ptr().cast());
            let p_call = GetProcAddress(module, c"victory_metal_call".as_ptr().cast());
            let (Some(a), Some(c)) = (p_avail, p_call) else {
                tracing::warn!("victory_metal.dll missing expected exports");
                return None;
            };
            (
                std::mem::transmute::<unsafe extern "system" fn() -> isize, FnAvailable>(a),
                std::mem::transmute::<unsafe extern "system" fn() -> isize, FnCall>(c),
            )
        };
        // SAFETY: validated export from `victory_metal.dll`.
        if unsafe { available() } == 0 {
            tracing::warn!("victory_metal.dll loaded, but Wine unixlib failed to initialize");
            return None;
        }
        let mut raw = PingArgs::default();
        // SAFETY: `raw` is a properly aligned C-layout argument struct.
        let status = unsafe { call(OP_PING, (&raw mut raw).cast()) };
        if status != 0 {
            tracing::warn!(status = format_args!("{status:#x}"), "victory_metal OP_PING failed");
            return None;
        }
        let ping = PingInfo::from_raw(&raw);
        tracing::info!(
            device = %ping.device_name,
            abi = ping.abi_version,
            unified = ping.unified_memory,
            residency_set = ping.residency_set,
            metalfx = ping.metalfx,
            irconverter = ping.irconverter,
            recommended_mb = ping.recommended_vram >> 20,
            "victory_metal unixlib bridge active (Route A.5)"
        );
        Some(Self {
            call,
            ping,
            registry: Mutex::new(CanonicalRegistry::new()),
        })
    }

    pub fn info(&self) -> &PingInfo {
        &self.ping
    }

    /// Raw ping call (useful for measuring PE -> unixlib dispatch latency).
    pub fn ping_raw(&self) -> Option<PingInfo> {
        let mut raw = PingArgs::default();
        // SAFETY: valid C-layout struct for `OP_PING`.
        let status = unsafe { (self.call)(OP_PING, (&raw mut raw).cast()) };
        (status == 0).then(|| PingInfo::from_raw(&raw))
    }

    /// Registers a CPU virtual address range `[cpu_addr, cpu_addr + length)`
    /// backed by `backing` into the global `MTLResidencySet` zero-copy.
    ///
    /// Enforces the `#221` #4 single-canonical-`MTLBuffer` invariant: if
    /// `backing` is already registered (for instance, a second `MapViewOfFile3`
    /// view of the same physical section), attaches the CPU view to the
    /// existing canonical `MTLBuffer` without allocating a duplicate wrapper.
    pub fn register_buffer(
        &self,
        cpu_addr: u64,
        length: u64,
        backing: BackingKey,
        defer_commit: bool,
    ) -> Option<(CanonicalBuffer, bool)> {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        match reg.plan_register(backing) {
            RegisterAction::ReuseCanonical(_) => {
                let buf = reg.attach_view(cpu_addr, length, backing)?;
                Some((buf, true))
            }
            RegisterAction::CreateNew { abi_backing_key } => {
                let mut args = RegisterBufferArgs {
                    cpu_addr,
                    length,
                    backing_key: abi_backing_key,
                    flags: u32::from(defer_commit),
                    ..Default::default()
                };
                // SAFETY: valid C-layout struct for `OP_REGISTER_BUFFER`.
                let status = unsafe { (self.call)(OP_REGISTER_BUFFER, (&raw mut args).cast()) };
                if status != 0 || args.status != 0 || args.handle == 0 {
                    tracing::warn!(
                        cpu_addr = format_args!("{cpu_addr:#x}"),
                        length,
                        ntstatus = format_args!("{status:#x}"),
                        err = args.status,
                        "victory_metal newBufferWithBytesNoCopy failed"
                    );
                    return None;
                }
                let buf = reg.insert_canonical(
                    cpu_addr,
                    length,
                    backing,
                    args.handle,
                    args.gpu_address,
                );
                Some((buf, args.reused != 0))
            }
        }
    }

    /// Resolves any title CPU address range `[addr, addr + len)` to its
    /// canonical zero-copy `MTLBuffer` handle and GPU address.
    pub fn resolve(&self, addr: u64, len: u64) -> Option<ResolvedBuffer> {
        let reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.resolve(addr, len)
    }

    /// Unregisters a previously registered CPU view; releases the canonical
    /// `MTLBuffer` from `MTLResidencySet` once its last view is removed.
    pub fn unregister_view(&self, cpu_addr: u64, defer_commit: bool) -> bool {
        let drop_handle = {
            let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
            reg.remove_view(cpu_addr)
        };
        if let Some(handle) = drop_handle {
            let mut args = UnregisterBufferArgs {
                handle,
                flags: u32::from(defer_commit),
                remaining_refs: 0,
            };
            // SAFETY: valid C-layout struct for `OP_UNREGISTER_BUFFER`.
            let status = unsafe { (self.call)(OP_UNREGISTER_BUFFER, (&raw mut args).cast()) };
            return status == 0;
        }
        true
    }

    /// Flushes any deferred `MTLResidencySet` additions/removals in one commit.
    pub fn flush_residency(&self) -> Option<(u32, u32)> {
        let mut args = FlushResidencyArgs::default();
        // SAFETY: valid C-layout struct for `OP_FLUSH_RESIDENCY`.
        let status = unsafe { (self.call)(OP_FLUSH_RESIDENCY, (&raw mut args).cast()) };
        (status == 0).then_some((args.pending_before, args.total_allocations))
    }

    /// Runs a real Metal compute checksum read over `[offset, offset + length)`
    /// of zero-copy buffer `handle`. Returns `(checksum, gpu_ns, wall_ns)`.
    pub fn bench_read(&self, handle: u64, offset: u64, length: u64) -> Option<(u64, u64, u64)> {
        let mut args = BenchArgs {
            handle,
            offset,
            length,
            mode: 0,
            ..Default::default()
        };
        // SAFETY: valid C-layout struct for `OP_BENCH_ZERO_COPY`.
        let status = unsafe { (self.call)(OP_BENCH_ZERO_COPY, (&raw mut args).cast()) };
        (status == 0).then_some((args.checksum_out, args.gpu_ns, args.wall_ns))
    }

    /// Runs a real Metal compute write (`word[i] = seed ^ i`) directly into
    /// zero-copy buffer `handle`. Returns `(gpu_ns, wall_ns)`.
    pub fn bench_write(&self, handle: u64, offset: u64, length: u64, seed: u32) -> Option<(u64, u64)> {
        let mut args = BenchArgs {
            handle,
            offset,
            length,
            mode: 1,
            seed,
            ..Default::default()
        };
        // SAFETY: valid C-layout struct for `OP_BENCH_ZERO_COPY`.
        let status = unsafe { (self.call)(OP_BENCH_ZERO_COPY, (&raw mut args).cast()) };
        (status == 0).then_some((args.gpu_ns, args.wall_ns))
    }

    /// Converts an SM 6.x DXIL container to Apple `.metallib` bytecode via
    /// `libmetalirconverter.dylib` inside the unixlib.
    pub fn convert_dxil(&self, dxil: &[u8], entry: &str) -> Result<(u32, Vec<u8>), i32> {
        let entry_c = std::ffi::CString::new(entry).map_err(|_| -2)?;
        // Query output size first.
        let mut query = ConvertDxilArgs {
            dxil_ptr: dxil.as_ptr() as u64,
            dxil_len: dxil.len() as u64,
            entry_ptr: entry_c.as_ptr() as u64,
            out_ptr: 0,
            out_cap: 0,
            out_len: 0,
            stage: 0,
            status: 0,
        };
        // SAFETY: pointers remain valid through the synchronous unixlib call.
        let nt = unsafe { (self.call)(OP_CONVERT_DXIL, (&raw mut query).cast()) };
        if nt != 0 || query.status != 0 || query.out_len == 0 {
            return Err(if query.status != 0 { query.status } else { -2 });
        }
        let mut out = vec![0u8; query.out_len as usize];
        let mut fill = ConvertDxilArgs {
            dxil_ptr: dxil.as_ptr() as u64,
            dxil_len: dxil.len() as u64,
            entry_ptr: entry_c.as_ptr() as u64,
            out_ptr: out.as_mut_ptr() as u64,
            out_cap: out.len() as u64,
            out_len: 0,
            stage: 0,
            status: 0,
        };
        // SAFETY: `out` buffer has exact capacity `query.out_len`.
        let nt = unsafe { (self.call)(OP_CONVERT_DXIL, (&raw mut fill).cast()) };
        if nt != 0 || fill.status != 0 {
            return Err(if fill.status != 0 { fill.status } else { -2 });
        }
        out.truncate(fill.out_len as usize);
        Ok((fill.stage, out))
    }

    pub fn stats(&self) -> Option<Stats> {
        let mut raw = Stats::default();
        // SAFETY: valid C-layout struct for `OP_STATS`.
        let status = unsafe { (self.call)(OP_STATS, (&raw mut raw).cast()) };
        (status == 0).then_some(raw)
    }
}

/// Process-wide singleton Metal unixlib bridge (`None` if unavailable).
pub fn global() -> Option<&'static MetalBridge> {
    static BRIDGE: OnceLock<Option<MetalBridge>> = OnceLock::new();
    BRIDGE.get_or_init(MetalBridge::load).as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
    };

    #[test]
    fn metal_unixlib_ping_and_call_latency() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        let info = bridge.info();
        assert_eq!(info.abi_version, 1);
        assert!(!info.device_name.is_empty());
        assert!(info.unified_memory, "Apple Silicon must report unified memory");
        assert!(info.residency_set, "macOS 15+ must support MTLResidencySet");
        assert!(info.metalfx, "Apple Silicon must support MetalFX");

        // Measure PE -> Wine unixlib call latency over 10,000 pings.
        const N: u32 = 10_000;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            assert!(bridge.ping_raw().is_some());
        }
        let ns_per_call = t0.elapsed().as_nanos() as f64 / f64::from(N);
        eprintln!(
            "victory_metal ping: device='{}' unified={} residency={} metalfx={} irconverter={} latency={:.1} ns/call",
            info.device_name,
            info.unified_memory,
            info.residency_set,
            info.metalfx,
            info.irconverter,
            ns_per_call,
        );
    }

    #[test]
    fn metal_zero_copy_64mb_virtual_alloc_cpu_gpu_cpu() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        const LEN: usize = 64 << 20; // 64 MiB
        // SAFETY: standalone 64 MiB VirtualAlloc freed before test exit.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u32>();
        assert!(!ptr.is_null(), "VirtualAlloc 64 MiB failed");
        let words = LEN / 4;
        // Fill on CPU and compute expected checksum.
        let mut expected: u64 = 0;
        // SAFETY: `ptr` owns `words` u32 elements.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, words);
            for (i, slot) in slice.iter_mut().enumerate() {
                let v = (i as u32).wrapping_mul(2654435761) ^ 0x5a5a_1234;
                *slot = v;
                expected = expected.wrapping_add(u64::from(v));
            }
        }

        let base = ptr as u64;
        let (buf, reused) = bridge
            .register_buffer(base, LEN as u64, BackingKey::VirtualAlloc(base), false)
            .expect("zero-copy register 64 MiB VirtualAlloc");
        assert!(!reused);
        assert_ne!(buf.gpu_address, 0);

        // Warm-up + measured GPU zero-copy read.
        let (sum0, _, _) = bridge.bench_read(buf.handle, 0, LEN as u64).expect("warmup read");
        assert_eq!(sum0, expected, "GPU zero-copy read checksum mismatch");
        let (sum, gpu_ns, wall_ns) = bridge.bench_read(buf.handle, 0, LEN as u64).expect("bench read");
        assert_eq!(sum, expected);
        let read_gbps = (LEN as f64) / (gpu_ns.max(1) as f64);

        // GPU writes pattern directly into VirtualAlloc memory -> CPU reads zero-copy!
        let seed = 0xdead_beefu32;
        let (w_gpu_ns, w_wall_ns) = bridge
            .bench_write(buf.handle, 0, LEN as u64, seed)
            .expect("bench write");
        let write_gbps = (LEN as f64) / (w_gpu_ns.max(1) as f64);

        // SAFETY: GPU command buffer already completed; read back directly on CPU.
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, words);
            for (idx, &val) in [0usize, 1, 4095, 65536, words / 2, words - 1]
                .into_iter()
                .map(|i| (i, &slice[i]))
            {
                assert_eq!(val, seed ^ (idx as u32), "GPU->CPU zero-copy mismatch at word {idx}");
            }
        }

        eprintln!(
            "victory_metal 64 MiB VirtualAlloc zero-copy: read {:.1} GB/s (gpu {:.2} ms, wall {:.2} ms) | write {:.1} GB/s (gpu {:.2} ms, wall {:.2} ms)",
            read_gbps,
            gpu_ns as f64 / 1e6,
            wall_ns as f64 / 1e6,
            write_gbps,
            w_gpu_ns as f64 / 1e6,
            w_wall_ns as f64 / 1e6,
        );

        assert!(bridge.unregister_view(base, false));
        // SAFETY: `ptr` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn metal_zero_copy_section_canonical_invariant() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        const LEN: usize = 4 << 20; // 4 MiB shared section
        // SAFETY: valid anonymous pagefile-backed section with two views.
        let (mapping, v1, v2) = unsafe {
            let m = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!m.is_null(), "CreateFileMappingW failed");
            let a = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, LEN);
            let b = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, LEN);
            assert!(!a.Value.is_null() && !b.Value.is_null() && a.Value != b.Value);
            (m, a, b)
        };

        let backing = BackingKey::PhysicalSection {
            section_id: 0x221_0004,
            offset: 0,
        };
        let (b1, reused1) = bridge
            .register_buffer(v1.Value as u64, LEN as u64, backing, false)
            .expect("register view 1");
        assert!(!reused1);

        // Registering second view of the SAME physical section MUST reuse canonical buffer 1!
        let (b2, reused2) = bridge
            .register_buffer(v2.Value as u64, LEN as u64, backing, false)
            .expect("attach view 2");
        assert!(reused2, "second view must share canonical MTLBuffer (#221 #4)");
        assert_eq!(b1.handle, b2.handle);
        assert_eq!(b1.gpu_address, b2.gpu_address);

        // Write through CPU view 2; Metal GPU reads through canonical buffer 1 -> 0 stale words!
        let words = LEN / 4;
        let mut expected: u64 = 0;
        // SAFETY: `v2` maps `words` u32 elements of the shared section.
        unsafe {
            let s2 = std::slice::from_raw_parts_mut(v2.Value.cast::<u32>(), words);
            for (i, slot) in s2.iter_mut().enumerate() {
                let v = (i as u32).wrapping_mul(17) ^ 0x1357_2468;
                *slot = v;
                expected = expected.wrapping_add(u64::from(v));
            }
        }
        let resolved = bridge.resolve(v2.Value as u64 + 1024, 4096).expect("resolve view 2");
        assert_eq!(resolved.handle, b1.handle);
        assert_eq!(resolved.offset, 1024);

        let (sum, _, _) = bridge
            .bench_read(resolved.handle, 0, LEN as u64)
            .expect("GPU read canonical buffer after view 2 write");
        assert_eq!(sum, expected, "0 stale words across dual section views");

        assert!(bridge.unregister_view(v1.Value as u64, false));
        assert!(bridge.unregister_view(v2.Value as u64, false));
        // SAFETY: unmapping views and closing section handle.
        unsafe {
            UnmapViewOfFile(v2);
            UnmapViewOfFile(v1);
            CloseHandle(mapping);
        }
    }

    #[test]
    fn metal_irconverter_converts_dxil_to_metallib() {
        let Some(bridge) = global() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        if !bridge.info().irconverter {
            eprintln!("skipping: libmetalirconverter.dylib not loaded (set VICTORY_METAL_IR_LIB or DXC_LIB)");
            return;
        }
        #[cfg(victory_dxc_detile)]
        let dxil: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpudetile.dxil"));
        #[cfg(not(victory_dxc_detile))]
        {
            eprintln!("skipping: gpudetile.dxil not embedded (build with VICTORY_HELPER_DXC)");
            return;
        }
        #[cfg(victory_dxc_detile)]
        {
            let (stage, metallib) = bridge
                .convert_dxil(dxil, "main")
                .expect("libmetalirconverter failed to convert gpudetile.dxil");
            assert!(metallib.len() >= 16, "metallib too small: {}", metallib.len());
            assert_eq!(&metallib[0..4], b"MTLB", "expected Apple MetalLib magic 'MTLB'");
            eprintln!(
                "victory_metal IRConverter: gpudetile.dxil ({} B) -> metallib ({} B, stage={})",
                dxil.len(),
                metallib.len(),
                stage,
            );
        }
    }
}
