//! Read the active initialization node on the Series 1.0.2.1 registry assert.
//! Unwind a copy of the faulting context. Only RBX and R12 in the init caller
//! identify the node and manager. The memory fields are sampled at the assert.

use super::{DUMP_REGISTRY, locate, page_readable, walk_frames};
use crate::LoadedImage;
use std::fmt::LowerHex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, ReadProcessMemory};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThreadId, GetCurrentThreadStackLimits,
};

const BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const VERIFY: usize = 0x1dac734;
const INIT_RETURN: usize = 0x63e78;
static IMAGE: OnceLock<usize> = OnceLock::new();
static REPORTED: AtomicBool = AtomicBool::new(false);

/// Enables the active-node snapshot alongside VICTORY_DUMP_REGISTRY for its known build.
pub fn configure_init_frame(hash: &str, image: &LoadedImage) {
    if DUMP_REGISTRY.get().is_some()
        && hash == BUILD
        && image.info().size_of_image as usize > VERIFY
    {
        let _ = IMAGE.set(image.base());
        eprintln!("VIctory: init-frame armed for the registry assert");
    }
}

pub(super) fn observe(address: usize, context: &CONTEXT) {
    let (Some(&base), Some(dir)) = (IMAGE.get(), DUMP_REGISTRY.get()) else {
        return;
    };
    report_once(base, dir, &REPORTED, address, context);
}

fn report_once(
    base: usize,
    dir: &std::path::Path,
    reported: &AtomicBool,
    address: usize,
    context: &CONTEXT,
) {
    if address.checked_sub(base) != Some(VERIFY) || reported.swap(true, Ordering::Relaxed) {
        return;
    }

    let (mut low, mut high) = (0, 0);
    // SAFETY: the vectored handler runs on the faulting thread.
    let tid = unsafe {
        GetCurrentThreadStackLimits(&mut low, &mut high);
        GetCurrentThreadId()
    };
    let caller = find_init_frame(*context, low, high, (base + INIT_RETURN) as u64);
    let text = match caller {
        Some((depth, node, manager)) => format!(
            "sample=registry_assert tid={tid} assert=image+{VERIFY:#x} frame={depth} caller=image+{INIT_RETURN:#x}\n{}\n",
            fields(node, manager)
        ),
        None => format!(
            "sample=registry_assert tid={tid} assert=image+{VERIFY:#x} frame=not_found caller=image+{INIT_RETURN:#x}\n"
        ),
    };
    eprintln!("VIctory: init-frame {}", text.trim_end());
    let path = dir.join("init-frame.txt");
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, &text)) {
        eprintln!(
            "VIctory: init-frame could not write {}: {e}",
            path.display()
        );
    }
}

fn find_init_frame(
    context: CONTEXT,
    low: usize,
    high: usize,
    target: u64,
) -> Option<(usize, usize, usize)> {
    let mut found = None;
    walk_frames(context, low, high, |depth, c| {
        if depth > 0 && c.Rip == target {
            found = Some((depth, c.Rbx as usize, c.R12 as usize));
            return false;
        }
        true
    });
    found
}

pub(crate) fn field<const N: usize>(base: usize, offset: usize) -> Option<[u8; N]> {
    let at = base.checked_add(offset)?;
    if base == 0 || !page_readable(at) || !page_readable(at.checked_add(N - 1)?) {
        return None;
    }
    let mut data = [0u8; N];
    let mut count = 0;
    // SAFETY: the OS copies into a fixed local buffer and handles concurrent
    // changes to the mapping. No reference into live title memory is formed.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as _,
            data.as_mut_ptr().cast(),
            N,
            &mut count,
        )
    };
    (ok != 0 && count == N).then_some(data)
}

fn hex<T: LowerHex>(value: Option<T>) -> String {
    value.map_or_else(|| "?".into(), |v| format!("{v:#x}"))
}

pub(crate) fn fields(node: usize, manager: usize) -> String {
    let link = field(node, 0x80).map(usize::from_le_bytes);
    let same = match link {
        Some(p) if p != 0 && p == manager => "yes",
        Some(_) => "no",
        None => "?",
    };
    let callback = field(node, 0x50)
        .map(usize::from_le_bytes)
        .map(|p| if p == 0 { "0x0".to_owned() } else { locate(p) });
    let required = field(node, 0xe4).map(u16::from_le_bytes);
    let flags = field(manager, 0x5e).map(u16::from_le_bytes);
    let mask = field(node, 0xe1).map(u8::from_le_bytes);
    let mode = field(manager, 0x70).map(u8::from_le_bytes);
    let expected = field(node, 0xe2).map(u8::from_le_bytes);
    format!(
        "node={node:#x} manager={manager:#x} node_80={} same_manager={same} callback={}\nnode_e4={} manager_5e={} node_e1={} manager_70={} node_e2={}",
        hex(link),
        callback.as_deref().unwrap_or("?"),
        hex(required),
        hex(flags),
        hex(mask),
        hex(mode),
        hex(expected)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        IMAGE_RUNTIME_FUNCTION_ENTRY, IMAGE_RUNTIME_FUNCTION_ENTRY_0, RtlAddFunctionTable,
        RtlDeleteFunctionTable,
    };

    #[test]
    fn restores_the_init_callers_nonvolatile_registers() {
        // This buffer supplies unwind metadata for a synthetic frame. No code in it runs.
        #[repr(align(16))]
        struct UnwindImage([u8; 256]);
        let mut image = UnwindImage([0; 256]);
        // UNWIND_INFO for a frame which pushed RBX, then R12.
        image.0[128..136].copy_from_slice(&[1, 3, 2, 0, 3, 0xc0, 1, 0x30]);
        let base = image.0.as_ptr() as u64;
        let entry = IMAGE_RUNTIME_FUNCTION_ENTRY {
            BeginAddress: 16,
            EndAddress: 80,
            Anonymous: IMAGE_RUNTIME_FUNCTION_ENTRY_0 {
                UnwindInfoAddress: 128,
            },
        };
        let target = base + 96;
        let stack = [0x1200u64, 0xb00, target, 0];
        let low = stack.as_ptr() as usize;
        let high = low + size_of_val(&stack);
        let context = CONTEXT {
            Rip: base + 32,
            Rsp: low as u64,
            Rbx: 33,
            R12: 44,
            ..Default::default()
        };
        // SAFETY: the table and its image live until removed below.
        assert!(unsafe { RtlAddFunctionTable(&entry, 1, base) });
        let found = find_init_frame(context, low, high, target);
        let absent = find_init_frame(context, low, high, target + 1);
        // SAFETY: the registration above succeeded and remains live.
        assert!(unsafe { RtlDeleteFunctionTable(&entry) });
        assert_eq!(found, Some((1, 0xb00, 0x1200)));
        assert_eq!(absent, None);
        assert_eq!((context.Rbx, context.R12, context.Rip), (33, 44, base + 32));
    }

    #[test]
    fn only_the_matching_assert_writes_a_snapshot_and_only_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("init-frame.txt");
        let done = AtomicBool::new(false);
        // A leaf stack whose immediate caller is the expected init return.
        let base = 0x10000usize;
        let stack = [(base + INIT_RETURN) as u64, 0];
        let mut context = CONTEXT {
            Rip: (base + VERIFY) as u64,
            Rsp: stack.as_ptr() as u64,
            ..Default::default()
        };
        report_once(base, dir.path(), &done, base + VERIFY - 1, &context);
        assert!(!path.exists());
        report_once(base, dir.path(), &done, base + VERIFY, &context);
        let first = std::fs::read_to_string(&path).unwrap();
        assert!(first.contains("frame=1 caller=image+0x63e78"), "{first}");
        assert!(first.contains("node_e4=?"));
        context.Rsp = 0;
        report_once(base, dir.path(), &done, base + VERIFY, &context);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), first);
        report_once(
            base,
            dir.path(),
            &AtomicBool::new(false),
            base + VERIFY,
            &context,
        );
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("frame=not_found")
        );
    }

    #[test]
    fn reports_the_link_and_field_widths_and_preserves_missing_values() {
        let mut node = [0u8; 0xf0];
        let mut manager = [0u8; 0x78];
        node[0xe4..0xe6].copy_from_slice(&0x1802u16.to_le_bytes());
        manager[0x5e..0x60].copy_from_slice(&0x8104u16.to_le_bytes());
        node[0xe1..0xe3].copy_from_slice(&[0x80, 0x08]);
        manager[0x70] = 0x18;
        let mp = std::hint::black_box(&manager).as_ptr() as usize;
        node[0x80..0x88].copy_from_slice(&mp.to_le_bytes());
        let np = std::hint::black_box(&node).as_ptr() as usize;
        let line = fields(np, mp);
        assert!(line.contains("callback=0x0"));
        assert!(line.contains("same_manager=yes"));
        assert!(
            line.contains(
                "node_e4=0x1802 manager_5e=0x8104 node_e1=0x80 manager_70=0x18 node_e2=0x8"
            )
        );
        assert!(fields(np, 0).contains("same_manager=no"));
        assert!(fields(0, 0).contains("node_e4=? manager_5e=? node_e1=? manager_70=? node_e2=?"));
        assert_eq!(field::<2>(usize::MAX, 1), None);
    }
}
