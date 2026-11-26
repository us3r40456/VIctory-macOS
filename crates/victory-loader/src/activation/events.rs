use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;

use super::{log_line, snapshot};

#[derive(Clone, Copy)]
struct Caller {
    address: usize,
    rva: Option<usize>,
}

impl fmt::Display for Caller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.address)?;
        if let Some(rva) = self.rva {
            write!(f, "(image+{rva:#x})")?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct Events {
    sequence: u64,
    registrations: u64,
    entries: u64,
    by_code: BTreeMap<u32, u64>,
    normal_returns: u64,
    // A single thread may enter the handler recursively.
    pending: BTreeMap<u64, (u32, Caller)>,
    invalid: u64,
}

impl Events {
    #[allow(
        clippy::too_many_arguments,
        reason = "matches the fixed trace callback"
    )]
    pub(super) fn record(
        &mut self,
        image: usize,
        size: usize,
        kind: u32,
        tid: u32,
        code: u32,
        caller: usize,
        entry: u64,
    ) -> u64 {
        self.sequence += 1;
        let seq = self.sequence;
        let caller = Caller {
            address: caller,
            rva: caller.checked_sub(image).filter(|&rva| rva < size),
        };
        match kind {
            0 => {
                self.registrations += 1;
                log_line(format_args!(
                    "seq={seq} registration event=0x9a0892b1 handler=image+0x1d65f10 tid={tid} caller={caller}"
                ));
                return 0;
            }
            1 => {
                self.entries += 1;
                *self.by_code.entry(code).or_default() += 1;
                if code == 0x20 {
                    self.pending.insert(seq, (tid, caller));
                    log_line(format_args!(
                        "seq={seq} enter ecx=0x20 tid={tid} caller={caller} before {}",
                        snapshot(image)
                    ));
                } else {
                    log_line(format_args!(
                        "seq={seq} enter ecx={code:#x} tid={tid} caller={caller}"
                    ));
                }
                return seq;
            }
            2 if code == 0x20 => {
                if let Entry::Occupied(pending) = self.pending.entry(entry)
                    && pending.get().0 == tid
                {
                    let (_, entry_caller) = pending.remove();
                    self.normal_returns += 1;
                    log_line(format_args!(
                        "seq={seq} leave ecx=0x20 entry={entry} tid={tid} caller={caller} entry_caller={entry_caller} after {}",
                        snapshot(image)
                    ));
                    return 0;
                }
            }
            _ => {}
        }
        self.invalid += 1;
        log_line(format_args!(
            "seq={seq} invalid kind={kind} ecx={code:#x} entry={entry} tid={tid} caller={caller}"
        ));
        0
    }

    pub(super) fn summary(&self, reason: &str, incomplete: bool) {
        log_line(format_args!(
            "summary reason={reason} events={} registrations={} handler_entries={} ecx20_entries={} ecx20_returns={} pending={} invalid={} complete={}",
            self.sequence,
            self.registrations,
            self.entries,
            self.by_code.get(&0x20).copied().unwrap_or(0),
            self.normal_returns,
            self.pending.len(),
            self.invalid,
            !incomplete && self.invalid == 0
        ));
        for code in [0, 0x20] {
            if !self.by_code.contains_key(&code) {
                log_line(format_args!("count ecx={code:#x} entries=0"));
            }
        }
        for (code, count) in &self.by_code {
            log_line(format_args!("count ecx={code:#x} entries={count}"));
        }
        for (entry, (tid, caller)) in &self.pending {
            log_line(format_args!(
                "pending ecx=0x20 entry={entry} tid={tid} caller={caller}"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Events;

    #[test]
    fn nested_returns_match_the_entry_and_thread_even_with_a_shared_caller() {
        let mut events = Events::default();
        let caller = 0x1234;
        let outer = events.record(0, 0, 1, 10, 0x20, caller, 0);
        let nested = events.record(0, 0, 1, 10, 0x20, caller, 0);
        let concurrent = events.record(0, 0, 1, 20, 0x20, caller, 0);

        events.record(0, 0, 2, 20, 0x20, caller, nested);
        assert_eq!(events.invalid, 1);
        assert_eq!(events.normal_returns, 0);
        assert_eq!(events.pending.len(), 3);
        assert_eq!(events.pending[&nested].0, 10);

        events.record(0, 0, 2, 10, 0x20, caller, nested);
        events.record(0, 0, 2, 20, 0x20, caller, concurrent);
        assert_eq!(events.pending.len(), 1);
        assert!(events.pending.contains_key(&outer));
        events.record(0, 0, 2, 10, 0x20, caller, outer);
        assert!(events.pending.is_empty());
        assert_eq!(events.entries, 3);
        assert_eq!(events.by_code[&0x20], 3);
        assert_eq!(events.normal_returns, 3);
    }
}
