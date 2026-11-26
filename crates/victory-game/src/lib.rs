//! Everything specific to Grand Theft Auto VI: its container formats, build
//! identities, and the hooks we place in its code (located by per-build
//! offsets or byte-pattern signatures). What may and may not be stored here
//! is defined in docs/asset-policy.md.

pub mod cityhash;
pub mod pipelines;
pub mod sgd5;
pub mod shaderpack;

/// A specific build of the Grand Theft Auto VI executable we know how to run.
#[derive(Debug, Clone, Copy)]
pub struct KnownBuild {
    /// Our own label, e.g. "1.0.0 retail".
    pub label: &'static str,
    /// BLAKE3 of the original executable, hex.
    pub blake3: &'static str,
    /// RPF9 archive the title needs during startup, relative to its executable.
    pub startup_archive: &'static str,
    /// Turns off the title's hang watchdog (`VICTORY_NO_HANG_WATCHDOG=1`):
    /// once a second it crashes the title on purpose if a registered
    /// thread's heartbeat is older than its limit, or the main thread has
    /// not finished a frame for 30 checks, which a world load streamed from
    /// slow storage and the first world frames outlast (#135).
    pub hang_watchdog: &'static [CodePatch],
}

/// Bytes to replace at an RVA of the mapped image, applied only where the
/// original bytes are found there.
#[derive(Debug, Clone, Copy)]
pub struct CodePatch {
    pub rva: u32,
    pub original: &'static [u8],
    pub replacement: &'static [u8],
}

/// Executables identified from a local package.
pub const KNOWN_BUILDS: &[KnownBuild] = &[KnownBuild {
    label: "Series 1.0.2.1",
    blake3: "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93",
    startup_archive: "appdata0.rpf",
    // Each jump to the crash path made a two-byte NOP: after the heartbeat
    // comparison, and where each of the loop's two forms counts down its
    // 30 checks without a finished frame.
    hang_watchdog: &[
        CodePatch {
            rva: 0x2215bdc,
            original: &[0x73, 0x2e],
            replacement: &[0x66, 0x90],
        },
        CodePatch {
            rva: 0x2215c70,
            original: &[0x74, 0x04],
            replacement: &[0x66, 0x90],
        },
        CodePatch {
            rva: 0x2215d84,
            original: &[0x74, 0x04],
            replacement: &[0x66, 0x90],
        },
    ],
}];

/// Looks up a build by the BLAKE3 hash of its executable.
pub fn identify(blake3_hex: &str) -> Option<&'static KnownBuild> {
    KNOWN_BUILDS
        .iter()
        .find(|b| b.blake3.eq_ignore_ascii_case(blake3_hex))
}
