//! GPU execution backend abstraction (`GpuBackend`, Step 2B-3).
//!
//! Isolates the reference D3D12/D3DMetal host path (`D3D12Backend`) from the
//! native Apple Silicon Metal command encoder (`MetalBackend`, selected with
//! `VICTORY_BACKEND=metal`).

use std::sync::OnceLock;

/// Active host GPU backend selected via `VICTORY_BACKEND=d3d12|metal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    D3D12,
    Metal,
}

impl BackendKind {
    pub fn from_env() -> Self {
        static KIND: OnceLock<BackendKind> = OnceLock::new();
        *KIND.get_or_init(|| {
            if std::env::var("VICTORY_BACKEND")
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("metal"))
            {
                BackendKind::Metal
            } else {
                BackendKind::D3D12
            }
        })
    }
}

/// Summary metrics for one submitted command batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStats {
    pub draws: u32,
    pub dispatches: u32,
    pub encode_ns: u64,
    pub gpu_ns: u64,
    pub wall_ns: u64,
}

/// Common surface between the reference `D3D12Backend` and the native Apple
/// Silicon `MetalBackend`.
pub trait GpuBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn name(&self) -> &str;
    fn supports_zero_copy(&self) -> bool;
    fn supports_frame_interpolation(&self) -> bool;
}

/// Reference D3D12 / D3DMetal backend backed by `crate::host::Host`.
pub struct D3D12Backend;

impl GpuBackend for D3D12Backend {
    fn kind(&self) -> BackendKind {
        BackendKind::D3D12
    }

    fn name(&self) -> &str {
        "D3D12"
    }

    fn supports_zero_copy(&self) -> bool {
        false
    }

    fn supports_frame_interpolation(&self) -> bool {
        false
    }
}

/// Native Apple Silicon Metal backend routing directly through `victory_metal.so`
/// (`MTLCommandBuffer` + global zero-copy `MTLResidencySet` + `MTLBinaryArchive`
/// + `MTLFXTemporalScaler` + `MTLFXFrameInterpolator` + `CAMetalLayer`).
#[cfg(windows)]
pub struct MetalBackend {
    device_name: String,
    frame_interp: bool,
}

#[cfg(windows)]
impl MetalBackend {
    /// Initializes the native Metal backend if `victory_metal.dll` is loaded.
    pub fn new() -> Option<Self> {
        let b = crate::metal::bridge::global()?;
        let info = b.info();
        Some(Self {
            device_name: info.device_name.clone(),
            frame_interp: info.frame_interpolator,
        })
    }

    /// Submits a packed `CommandStream` directly into one `MTLCommandBuffer`.
    pub fn submit(
        &self,
        stream: &crate::metal::encoder::CommandStream,
        wait_completed: bool,
    ) -> Result<BatchStats, i32> {
        let b = crate::metal::bridge::global().ok_or(-1)?;
        let res = b.submit_commands(stream.as_words(), wait_completed)?;
        Ok(BatchStats {
            draws: res.draws_encoded,
            dispatches: res.dispatches_encoded,
            encode_ns: res.encode_ns,
            gpu_ns: res.gpu_ns,
            wall_ns: res.wall_ns,
        })
    }
}

#[cfg(windows)]
impl GpuBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn name(&self) -> &str {
        &self.device_name
    }

    fn supports_zero_copy(&self) -> bool {
        true
    }

    fn supports_frame_interpolation(&self) -> bool {
        self.frame_interp
    }
}

/// Process-wide native `MetalBackend` instance (when `victory_metal.dll` is available).
#[cfg(windows)]
pub fn metal() -> Option<&'static MetalBackend> {
    static METAL: OnceLock<Option<MetalBackend>> = OnceLock::new();
    METAL.get_or_init(MetalBackend::new).as_ref()
}

/// Active `GpuBackend` selected by `VICTORY_BACKEND` (falling back cleanly to
/// `D3D12Backend` if Metal is unavailable).
pub fn active() -> &'static dyn GpuBackend {
    static D3D12: D3D12Backend = D3D12Backend;
    #[cfg(windows)]
    if BackendKind::from_env() == BackendKind::Metal
        && let Some(m) = metal()
    {
        return m;
    }
    &D3D12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_is_d3d12_reference() {
        assert_eq!(D3D12Backend.kind(), BackendKind::D3D12);
        assert_eq!(D3D12Backend.name(), "D3D12");
    }

    #[cfg(windows)]
    #[test]
    fn metal_backend_initializes_on_apple_silicon() {
        let Some(m) = metal() else {
            eprintln!("skipping: victory_metal.dll not loaded on WINEDLLPATH");
            return;
        };
        assert_eq!(m.kind(), BackendKind::Metal);
        assert!(m.supports_zero_copy());
        assert!(!m.name().is_empty());
    }
}
