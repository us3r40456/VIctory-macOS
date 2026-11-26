//! Small fixed patches to a title's code at known RVAs.

/// Replaces the bytes at `rva` of the image mapped at `base` with
/// `replacement` if `original` is there, before any title thread runs.
/// False if the bytes differ (another build) or the page could not be made
/// writable.
pub fn patch_code(base: u64, rva: u32, original: &[u8], replacement: &[u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if original.len() != replacement.len() {
        return false;
    }
    let at = (base + u64::from(rva)) as *mut u8;
    // SAFETY: an RVA inside the mapped image; read before any change.
    let found = unsafe { std::slice::from_raw_parts(at, original.len()) };
    if found != original {
        return false;
    }
    let mut old = 0;
    // SAFETY: the title's code, writable for the moment of the write.
    unsafe {
        if VirtualProtect(at.cast(), original.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        std::ptr::copy_nonoverlapping(replacement.as_ptr(), at, replacement.len());
        VirtualProtect(at.cast(), original.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at.cast(), original.len());
    }
    true
}
