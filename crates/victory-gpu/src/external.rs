//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, moved to the host resource `state` in the
/// title list `list`, and the state it is in: `state` without the bits the
/// host copy cannot take (see `host::legal_state`). What the list records
/// next finds it there and must leave it there. The list holds the texture
/// until it is reset. `None` for a buffer, or without a host list.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    let texture = crate::resource::host_resource(resource)?.clone();
    crate::list::ensure_state(host, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}

/// Executes all commands recorded so far in `list`'s current host segment and
/// reopens it so an out-of-band MetalFX pass (`MTLFXTemporalScaler`) observes
/// prior rendering and writes its upscaled output before subsequent commands.
pub fn flush_and_reopen(list: *const Object) -> bool {
    let Some(host) = crate::host::host() else {
        return false;
    };
    let mut ran = false;
    crate::list::with_state(list, |host_list, state| {
        host_list.close();
        host.execute(&[host_list]);
        host_list.reset();
        state.host_state_lost();
        ran = true;
    });
    ran
}

/// Returns a title texture's host `ID3D12Resource` and its current global host
/// resource state (after a segment flush).
pub fn current_texture(resource: *const Object) -> Option<(ID3D12Resource, u32)> {
    let texture = crate::resource::host_resource(resource)?.clone();
    let state = crate::resource::host_transition(resource, 0)
        .map(|(_, before)| {
            crate::resource::host_transition(resource, before);
            before
        })
        .unwrap_or(0);
    let legal = crate::host::legal_state(&texture, state);
    Some((texture, legal))
}
