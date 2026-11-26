//! Trace the interface and result of each enabled console PSO probe on Windows.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::System::Threading::GetCurrentThreadId;

use crate::object::{self, HRESULT, Iid, This};

const TARGET: &str = "victory_gpu::pso_calls";

pub(super) struct Call {
    id: u64,
    out: *mut This,
}

impl Call {
    // Keep this stack frame for the caller offset below.
    #[inline(never)]
    pub(super) fn begin(
        slot: usize,
        input: *const c_void,
        patches: Option<(u32, *const c_void)>,
        iid: *const Iid,
        out: *mut This,
    ) -> Option<Self> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_TRACE_PSO_CALLS"))
            || !tracing::enabled!(target: TARGET, tracing::Level::INFO)
        {
            return None;
        }

        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut guid = [0u8; 16];
        let iid_text = if iid.is_null() {
            "null".into()
        } else if crate::host::read_into(iid.cast(), &mut guid) {
            object::guid_text(&guid)
        } else {
            "unreadable".into()
        };
        // Skip object::caller, this function and the device vtable entry.
        let caller = object::caller(3);
        // SAFETY: gets the ID of the calling Windows thread; no arguments.
        let tid = unsafe { GetCurrentThreadId() };
        tracing::info!(
            target: TARGET, id, tid, frame = crate::frame::current(),
            slot = format_args!("{slot:#x}"), abi_args = if patches.is_some() { 6 } else { 4 },
            %caller, iid = %iid_text, ?input, ?patches, ?out, "console PSO call"
        );
        Some(Self { id, out })
    }

    pub(super) fn finish(self, hr: HRESULT) {
        let mut bytes = [0u8; size_of::<usize>()];
        // Only traced paths reach here. Each writes the output slot when present.
        let result = if self.out.is_null() {
            "no output slot".into()
        } else if crate::host::read_into(self.out.cast(), &mut bytes) {
            format!("{:#x}", usize::from_ne_bytes(bytes))
        } else {
            "unreadable".into()
        };
        tracing::info!(
            target: TARGET, id = self.id, hr = format_args!("{:#010x}", hr as u32),
            %result, "console PSO return"
        );
    }
}
