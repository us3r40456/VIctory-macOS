//! Correlate the observed path to the state-3 store within one init frame.

use super::{TRACE, log_line};
use crate::crash::{init_frame, locate};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;

const SITES: [&str; 7] = [
    "word",
    "byte",
    "optional_check",
    "predicate_call",
    "predicate_return",
    "accepted",
    "skip_store",
];

const WORD: usize = 0;
const BYTE: usize = 1;
const PREDICATE_CALL: usize = 3;
const PREDICATE_RETURN: usize = 4;
const ACCEPTED: usize = 5;
const SKIP: usize = 6;

#[derive(Default)]
struct Passage {
    number: u64,
    last_site: usize,
    predicate_target: Option<usize>,
    returned: Option<u8>,
}

#[derive(Default)]
pub(super) struct GateCounts {
    visits: [u64; 7],
    rejects: [u64; 4],
    nodes: BTreeSet<usize>,
    managers: BTreeSet<usize>,
    // Nested and simultaneous initializers can use the same node pointer.
    pending: BTreeMap<(u32, usize, usize, usize), Passage>,
    replaced: u64,
}

impl GateCounts {
    fn visit(
        &mut self,
        site: usize,
        tid: u32,
        sp: usize,
        node: usize,
        manager: usize,
        rax: usize,
    ) -> String {
        self.visits[site] += 1;
        self.nodes.insert(node);
        self.managers.insert(manager);
        let seq: u64 = self.visits.iter().sum();
        let key = (tid, sp, node, manager);
        if site == WORD
            && self
                .pending
                .insert(
                    key,
                    Passage {
                        number: seq,
                        ..Default::default()
                    },
                )
                .is_some()
        {
            self.replaced += 1;
        }
        let mut outcome = "?";
        let mut passage = self.pending.get_mut(&key);
        if let Some(p) = passage.as_deref_mut() {
            match site {
                PREDICATE_CALL => {
                    p.predicate_target = init_frame::field(rax, 0x50).map(usize::from_le_bytes);
                }
                PREDICATE_RETURN => {
                    p.returned = (p.last_site == PREDICATE_CALL).then_some(rax as u8)
                }
                SKIP => {
                    let (index, label) = match p.last_site {
                        WORD => (0, "word"),
                        BYTE => (1, "byte"),
                        PREDICATE_RETURN if p.returned == Some(0) => (2, "predicate"),
                        _ => (3, "unknown"),
                    };
                    self.rejects[index] += 1;
                    outcome = label;
                }
                _ => {}
            }
            p.last_site = site;
        } else if site == SKIP {
            self.rejects[3] += 1;
            outcome = "unknown";
        }
        let pass = passage
            .as_ref()
            .map(|p| p.number.to_string())
            .unwrap_or_else(|| "?".into());
        let target = passage
            .as_ref()
            .and_then(|p| p.predicate_target)
            .map(|p| if p == 0 { "0x0".into() } else { locate(p) })
            .unwrap_or_else(|| "?".into());
        let returned = passage
            .as_ref()
            .and_then(|p| p.returned)
            .map(|v| format!("{v:#x}"))
            .unwrap_or_else(|| "?".into());
        if matches!(site, ACCEPTED | SKIP) {
            self.pending.remove(&key);
        }
        let label = SITES[site];
        let fields = init_frame::fields(node, manager).replace('\n', " ");
        format!(
            "init_gate seq={seq} pass={pass} site={label} tid={tid} sp={sp:#x} {fields} reject={outcome} predicate_target={target} predicate_al={returned}"
        )
    }

    pub(super) fn summary(&self, reason: &str, incomplete: bool) {
        let [
            word,
            byte,
            _,
            optional_calls,
            optional_returns,
            accepted,
            skipped,
        ] = self.visits;
        let [r1, r2, r3, unknown] = self.rejects;
        log_line(format_args!(
            "init_gate summary reason={reason} word={word} byte={byte} predicate_calls={optional_calls} predicate_returns={optional_returns} accepted={accepted} skip_store={skipped} reject_word={r1} reject_byte={r2} reject_predicate={r3} reject_unknown={unknown} distinct_nodes={} distinct_managers={} open={} replaced={} io_complete={}",
            self.nodes.len(),
            self.managers.len(),
            self.pending.len(),
            self.replaced,
            !incomplete,
        ));
    }
}

pub(super) extern "system" fn record(
    site: u32,
    tid: u32,
    sp: usize,
    node: usize,
    manager: usize,
    rax: usize,
) {
    let Some(trace) = TRACE.get() else { return };
    let Some(counts) = &trace.gate else { return };
    trace.active.fetch_add(1, Ordering::SeqCst);
    if std::panic::catch_unwind(|| {
        let mut counts = counts.lock().unwrap_or_else(|e| e.into_inner());
        if trace.finished.load(Ordering::SeqCst) {
            return;
        }
        if (site as usize) >= SITES.len() {
            trace.incomplete.store(true, Ordering::Relaxed);
            return;
        }
        let event = counts.visit(site as usize, tid, sp, node, manager, rax);
        log_line(format_args!("{event}"));
    })
    .is_err()
    {
        trace.incomplete.store(true, Ordering::Relaxed);
    }
    trace.active.fetch_sub(1, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_init_frame_is_counted_at_its_observed_destination() {
        let mut c = GateCounts::default();
        let mut event = |site, tid, sp, ax| c.visit(site, tid, sp, 0, 0, ax);
        event(0, 3, 8, 0);
        event(0, 5, 8, 0);
        event(1, 5, 8, 0);
        assert!(event(6, 3, 8, 0).contains("reject=word"));
        assert!(event(6, 5, 8, 0).contains("reject=byte"));

        let mut object = [0usize; 11];
        object[10] = locate as *const () as usize;
        let addr = std::hint::black_box(&object).as_ptr() as usize;
        for site in [0, 1, 2, 3] {
            event(site, 3, 8, addr);
        }
        // A nested accepted call on the same thread with the same node.
        for site in [0, 1, 2, 5] {
            event(site, 3, 4, 0);
        }
        event(4, 3, 8, 0x100);
        let rejection = event(6, 3, 8, 0);
        assert!(rejection.contains("reject=predicate"));
        assert!(rejection.contains(&format!("predicate_target={}", locate(object[10]))));
        assert!(rejection.contains("predicate_al=0x0"));
        assert!(event(6, 3, 8, 0).contains("reject=unknown"));
        for site in [0, 1, 2, 4] {
            event(site, 3, 8, 0);
        }
        assert!(event(6, 3, 8, 0).contains("reject=unknown"));
        assert_eq!(c.rejects, [1, 1, 1, 2]);
        assert_eq!(c.visits[5], 1);
        assert!(c.pending.is_empty());
    }
}
