//! One frame's host D3D12 work recorded and run again without the title, to
//! find out what it costs the GPU and to try changes on it in seconds
//! (docs/replay.md). After the Windows fork's tool of the same name
//! (avatarmanz/VIctory-7900XTX, `crates/victory-gpu/src/replay`).
//!
//! [`record`] hooks the host's D3D12 objects (`hook`) and keeps what a frame
//! needs; `snap` writes one frame down; [`play`] runs it. `wire` is what a
//! capture holds, `pso` pipeline descriptions, `states` the states textures
//! are in, `edit` the experiments and lists cut into pieces, `gpu` D3D12
//! helpers, `plain` what needs no D3D12.
//!
//! The recorder is at the D3D12 boundary, below the host layer: whatever the
//! host layer does (its own passes, uploads, read-backs, the present's copy)
//! is in a capture as the calls it made, and nothing of the host layer is
//! needed to run them again. The host layer only says when the device was
//! made, what its objects are called and when a frame ends.

// Elsewhere than Windows only what its tests need is used.
#![cfg_attr(not(windows), allow(dead_code))]

mod edit;
#[cfg(windows)]
mod gpu;
#[cfg(windows)]
mod hook;
pub mod look;
mod plain;
#[cfg(windows)]
pub mod play;
#[cfg(windows)]
mod pso;
#[cfg(windows)]
pub mod record;
#[cfg(windows)]
mod snap;
mod states;
#[cfg(all(windows, feature = "replay-testing"))]
#[doc(hidden)]
pub mod testing;
mod wire;
