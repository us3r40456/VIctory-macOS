//! A capture looked at without a GPU (on any host): where it is, what its
//! lists hold, and a summary of the frame: its resources, what each
//! submission does, which pipelines run how often, into what, and how much
//! of the frame is the host layer's own work.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use super::edit;
use super::wire::{self, Barrier, Cmd, Frame, Loc, MISSING, NONE, Place, Va};

/// Whether file `path` is there (by opening it: `crate::read_file`).
fn is_there(path: &Path) -> bool {
    File::open(path).is_ok()
}

/// The folder of a capture: `folder`, or the last `capture-<n>` in it that
/// has a `frame.bin`.
pub fn find(folder: &Path) -> Result<PathBuf> {
    if is_there(&folder.join("frame.bin")) {
        return Ok(folder.to_owned());
    }
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in std::fs::read_dir(folder)
        .with_context(|| format!("{}", folder.display()))?
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(n) = name.strip_prefix("capture-").and_then(|n| n.parse().ok()) else {
            continue;
        };
        if is_there(&entry.path().join("frame.bin")) && best.as_ref().is_none_or(|b| b.0 < n) {
            best = Some((n, entry.path()));
        }
    }
    best.map(|b| b.1)
        .ok_or_else(|| anyhow!("no capture (frame.bin) in {}", folder.display()))
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The names of the title's shaders by the hash the runtime logs them by:
/// a file of lines `<hash> <name>`.
#[derive(Default)]
pub struct Names(HashMap<String, String>);

impl Names {
    pub fn read(path: Option<&Path>) -> Result<Names> {
        let Some(path) = path else {
            return Ok(Names::default());
        };
        let text = crate::read_file(path).with_context(|| format!("{}", path.display()))?;
        let text = String::from_utf8_lossy(&text);
        Ok(Names(
            text.lines()
                .filter_map(|l| {
                    let (hash, name) = l.trim().split_once(char::is_whitespace)?;
                    Some((hash.to_owned(), name.trim().to_owned()))
                })
                .collect(),
        ))
    }

    /// A pipeline's name as the runtime gave it, with the name of every
    /// shader hash in it that has one after it: `[mesh 01..ef a title shader,
    /// pixel ...]`.
    pub fn of(&self, name: &str) -> String {
        if self.0.is_empty() {
            return name.to_owned();
        }
        let mut out = String::with_capacity(name.len());
        let mut rest = name;
        while !rest.is_empty() {
            let word = rest
                .find(|c: char| !c.is_ascii_hexdigit())
                .unwrap_or(rest.len());
            if word == 16
                && let Some(known) = self.0.get(&rest[..16])
            {
                out.push_str(&rest[..16]);
                out.push(' ');
                out.push_str(known);
                rest = &rest[16..];
                continue;
            }
            // To the end of this word, and the character after it.
            let next = rest[word..].chars().next().map_or(0, char::len_utf8);
            out.push_str(&rest[..word + next]);
            rest = &rest[word + next..];
        }
        out
    }
}

/// How the CPU sees a resource's memory: "GPU" (not at all), "mapped"
/// (system memory both sides see), "upload", "read-back".
fn memory(frame: &Frame, r: &wire::Resource) -> &'static str {
    let props = match &r.place {
        Place::Committed { props, .. } => Some(props.clone()),
        Place::Placed { heap, .. } => frame
            .heaps
            .iter()
            .find(|h| h.id == *heap)
            .map(|h| h.props.clone()),
        Place::BackBuffer {} => None,
    };
    match props.map(|p| (p.kind, p.cpu)) {
        Some((2, _)) | Some((5, _)) => "upload",
        Some((3, _)) => "read-back",
        Some((4, 3)) | Some((4, 2)) => "mapped",
        _ => "GPU",
    }
}

fn is_buffer(r: &wire::Resource) -> bool {
    r.desc.dimension == 1
}

/// A resource in a few words.
fn describe(r: &wire::Resource) -> String {
    let d = &r.desc;
    if is_buffer(r) {
        format!("buffer #{} {} KB \"{}\"", r.id, d.width >> 10, r.name)
    } else {
        format!(
            "#{} {}x{}{} f{}{}",
            r.id,
            d.width,
            d.height,
            if d.depth > 1 {
                format!("x{}", d.depth)
            } else {
                String::new()
            },
            d.format,
            if d.mips > 1 {
                format!(" {} levels", d.mips)
            } else {
                String::new()
            },
        )
    }
}

/// Writes, one line each, the frame's resources and the commands of lists
/// (by their place in the frame).
pub fn listing(folder: &Path, lists: (u32, u32), line: &mut dyn FnMut(&str)) -> Result<()> {
    let frame = Frame::read(&find(folder)?.join("frame.bin"))?;
    let pipelines: HashMap<u32, &str> = frame
        .pipelines
        .iter()
        .map(|p| (p.id, p.name.as_str()))
        .collect();
    for r in &frame.resources {
        if is_buffer(r) {
            line(&format!(
                "buffer {} {} bytes, {}: {}",
                r.id,
                r.desc.width,
                memory(&frame, r),
                r.name
            ));
        } else {
            let d = &r.desc;
            line(&format!(
                "texture {} {}x{}x{} mips {} f{}: {}",
                r.id, d.width, d.height, d.depth, d.mips, d.format, r.name
            ));
        }
    }
    let mut place = 0;
    for (i, s) in frame.submits.iter().enumerate() {
        line(&format!(
            "submission {i}: {} lists, {} runs of bytes the CPU wrote ({} bytes), {} views",
            s.lists.len(),
            s.deltas.len(),
            s.deltas.iter().map(|d| d.len).sum::<u64>(),
            s.views.len()
        ));
        for &l in s.lists.iter().filter(|&&l| l != MISSING) {
            let here = place;
            place += 1;
            if here < lists.0 || here > lists.1 {
                continue;
            }
            line(&format!("list {here} of submission {i}"));
            for (n, c) in Cmd::all(&frame.lists[l as usize].commands.0)?
                .iter()
                .enumerate()
            {
                match c {
                    Cmd::Pipeline { pso } => {
                        line(&format!(
                            "{n} Pipeline {pso} {}",
                            pipelines.get(pso).copied().unwrap_or_default()
                        ));
                    }
                    c => line(&format!("{n} {c:?}")),
                }
            }
        }
    }
    Ok(())
}

/// What a pipeline does in the frame.
#[derive(Default)]
struct Use {
    commands: u32,
    lists: HashSet<u32>,
    submissions: HashSet<usize>,
    /// Draws: the targets drawn into, by how often.
    targets: BTreeMap<String, u32>,
    /// Dispatches: thread groups in all, and the largest one.
    groups: u64,
    largest: [u32; 3],
    indirect: u32,
}

/// Writes a summary of the frame: `top` pipelines by commands, with
/// `names` for the title's shaders.
pub fn summary(
    folder: &Path,
    names: &Names,
    top: usize,
    line: &mut dyn FnMut(&str),
) -> Result<()> {
    let frame = Frame::read(&find(folder)?.join("frame.bin"))?;
    for n in &frame.notes {
        line(&format!("recorder: {n}"));
    }
    let by_id: HashMap<u32, &wire::Resource> = frame.resources.iter().map(|r| (r.id, r)).collect();
    // Resources by where they lie.
    let mut classes: BTreeMap<(&str, &str), (u32, u64)> = BTreeMap::new();
    for r in &frame.resources {
        let what = if is_buffer(r) { "buffers" } else { "textures" };
        let c = classes.entry((what, memory(&frame, r))).or_default();
        // A texture's size is not in its description: counted only.
        *c = (c.0 + 1, c.1 + if is_buffer(r) { r.desc.width } else { 0 });
    }
    for ((what, memory), (n, bytes)) in &classes {
        line(&format!(
            "{n} {what} in {memory} memory{}",
            if *bytes > 0 {
                format!(", {} MB", bytes >> 20)
            } else {
                String::new()
            }
        ));
    }
    let on_disk: u64 = frame
        .contents
        .iter()
        .filter(|c| c.file != wire::ZERO)
        .map(|c| c.len)
        .sum();
    line(&format!(
        "contents of GPU memory: {} MB in {} pieces; views: {} heaps, {} runs",
        on_disk >> 20,
        frame.contents.len(),
        frame.descriptor_heaps.len(),
        frame
            .descriptor_heaps
            .iter()
            .map(|h| h.runs.len())
            .sum::<usize>()
    ));
    // Mapped buffers: those with bytes at the frame's start, and those the
    // GPU may write (a view for unordered access, a root view of that kind,
    // a copy's destination).
    let mut may_write: HashSet<u32> = HashSet::new();
    let views = frame
        .descriptor_heaps
        .iter()
        .flat_map(|h| h.runs.iter().map(|r| &r.2))
        .chain(frame.submits.iter().flat_map(|s| s.views.iter().map(|v| &v.2)));
    for s in views.filter(|s| s.kind == wire::slot::UAV) {
        may_write.extend([s.res, s.counter]);
    }
    for c in frame.lists.iter().filter_map(|l| Cmd::all(&l.commands.0).ok()).flatten() {
        match c {
            Cmd::RootView { kind: wire::slot::UAV, va: Va::At { res, .. }, .. } => _ = may_write.insert(res),
            Cmd::CopyBuffer { dst, .. } | Cmd::CopyResource { dst, .. } | Cmd::ResolveQuery { dst, .. } => _ = may_write.insert(dst),
            Cmd::CopyTexture { dst: Loc::Placed { res, .. }, .. } => _ = may_write.insert(res),
            _ => {}
        }
    }
    let with_bytes: HashSet<u32> = frame.submits.first().iter().flat_map(|s| s.deltas.iter().map(|d| d.res)).collect();
    let mapped = |keep: &dyn Fn(&wire::Resource) -> bool| {
        let all: Vec<&wire::Resource> = frame
            .resources
            .iter()
            .filter(|r| is_buffer(r) && matches!(memory(&frame, r), "mapped" | "upload") && keep(r))
            .collect();
        format!("{} ({} MB)", all.len(), all.iter().map(|r| r.desc.width).sum::<u64>() >> 20)
    };
    line(&format!(
        "buffers the CPU maps: {} hold bytes at the frame's start, {} the GPU may write, {} either",
        mapped(&|r| with_bytes.contains(&r.id)),
        mapped(&|r| may_write.contains(&r.id)),
        mapped(&|r| with_bytes.contains(&r.id) || may_write.contains(&r.id)),
    ));
    // How far into each mapped buffer its bytes reach: the end of what is
    // not zero at the start, or written by the CPU or a copy later.
    let mut reach: HashMap<u32, u64> = HashMap::new();
    for d in frame.submits.iter().flat_map(|s| &s.deltas) {
        let r = reach.entry(d.res).or_default();
        *r = (*r).max(d.at + d.len);
    }
    let reached: u64 = frame
        .resources
        .iter()
        .filter(|r| is_buffer(r) && matches!(memory(&frame, r), "mapped" | "upload"))
        .map(|r| reach.get(&r.id).copied().unwrap_or(0).min(r.desc.width).next_multiple_of(65536))
        .sum();
    line(&format!(
        "of the buffers the CPU maps, {} MB lie before each one's last byte that is not zero or that the CPU writes in the frame",
        reached >> 20
    ));
    let mesh: HashSet<u32> = frame
        .pipelines
        .iter()
        .filter(|p| {
            p.parts
                .iter()
                .any(|part| matches!(part, wire::Part::Code { kind: 25, .. }))
        })
        .map(|p| p.id)
        .collect();
    let dispatches: HashSet<u32> = frame
        .signatures
        .iter()
        .filter(|s| s.args.iter().any(|a| a[0] == 2))
        .map(|s| s.id)
        .collect();
    let pipeline_names: HashMap<u32, &str> = frame
        .pipelines
        .iter()
        .map(|p| (p.id, p.name.as_str()))
        .collect();
    let mut uses: HashMap<u32, Use> = HashMap::new();
    let mut kinds_all: BTreeMap<&str, u32> = BTreeMap::new();
    // Copies: by kind, how many and (where it is told) how many bytes.
    let mut copies: BTreeMap<String, (u32, u64)> = BTreeMap::new();
    let mut transitions: BTreeMap<String, u32> = BTreeMap::new();
    let mut place = 0u32;
    for (i, s) in frame.submits.iter().enumerate() {
        let mut kinds: BTreeMap<&str, u32> = BTreeMap::new();
        let mut commands = 0;
        let mut passes = 0;
        for &l in s.lists.iter().filter(|&&l| l != MISSING) {
            let here = place;
            place += 1;
            let mut bound = NONE;
            let mut targets = String::new();
            // Target changes with work between them: render passes.
            let mut drew = true;
            for c in Cmd::all(&frame.lists[l as usize].commands.0)? {
                commands += 1;
                match &c {
                    Cmd::Pipeline { pso } => bound = *pso,
                    Cmd::Targets { colors, depth } => {
                        let name = |s: &wire::Slot| {
                            by_id
                                .get(&s.res)
                                .map_or(format!("#{}", s.res), |r| describe(r))
                        };
                        let mut all: Vec<String> = colors.iter().map(name).collect();
                        if let Some(d) = depth {
                            all.push(format!("depth {}", name(d)));
                        }
                        targets = all.join(" + ");
                        if drew {
                            passes += 1;
                            drew = false;
                        }
                    }
                    Cmd::Barriers { all } => {
                        for b in all {
                            if let Barrier::Transition {
                                before, after, res, ..
                            } = b
                            {
                                let what = if by_id.get(res).is_some_and(|r| is_buffer(r)) {
                                    "buffer"
                                } else {
                                    "texture"
                                };
                                *transitions
                                    .entry(format!("{what} {before:#x} -> {after:#x}"))
                                    .or_default() += 1;
                            }
                        }
                    }
                    Cmd::CopyBuffer { dst, src, len, .. } => {
                        let side = |id: &u32| by_id.get(id).map_or("?", |r| memory(&frame, r));
                        let c = copies
                            .entry(format!("buffer, {} to {}", side(src), side(dst)))
                            .or_default();
                        *c = (c.0 + 1, c.1 + len);
                    }
                    Cmd::CopyTexture { dst, src, .. } => {
                        let what = match (dst, src) {
                            (Loc::Placed { .. }, _) => "texture to buffer",
                            (_, Loc::Placed { .. }) => "buffer to texture",
                            _ => "texture to texture",
                        };
                        let bytes = match (dst, src) {
                            (Loc::Placed { size, pitch, .. }, _)
                            | (_, Loc::Placed { size, pitch, .. }) => {
                                u64::from(*pitch) * u64::from(size[1]) * u64::from(size[2].max(1))
                            }
                            _ => 0,
                        };
                        let c = copies.entry(what.to_owned()).or_default();
                        *c = (c.0 + 1, c.1 + bytes);
                    }
                    Cmd::CopyResource { .. } => copies.entry("whole resource".into()).or_default().0 += 1,
                    _ => {}
                }
                if !c.works() && !matches!(c, Cmd::Barriers { .. }) {
                    continue;
                }
                drew = drew || c.works();
                let kind = edit::kind(&c, &dispatches, &mesh, bound);
                *kinds.entry(kind).or_default() += 1;
                *kinds_all.entry(kind).or_default() += 1;
                if c.runs_shaders() {
                    let u = uses.entry(bound).or_default();
                    u.commands += 1;
                    u.lists.insert(here);
                    u.submissions.insert(i);
                    match &c {
                        Cmd::Dispatch { groups } => {
                            let n: u64 = groups.iter().map(|&g| u64::from(g)).product();
                            u.groups += n;
                            let most: u64 = u.largest.iter().map(|&g| u64::from(g)).product();
                            if n > most {
                                u.largest = *groups;
                            }
                        }
                        Cmd::Indirect { signature, .. } => {
                            u.indirect += 1;
                            if !dispatches.contains(signature) {
                                *u.targets.entry(targets.clone()).or_default() += 1;
                            }
                        }
                        _ => *u.targets.entry(targets.clone()).or_default() += 1,
                    }
                }
            }
        }
        let wrote: u64 = s.deltas.iter().map(|d| d.len).sum();
        line(&format!(
            "submission {i}: {} lists, {commands} calls, {passes} target changes; CPU wrote {} KB in {} runs, {} views; {}",
            s.lists.len(),
            wrote >> 10,
            s.deltas.len(),
            s.views.len(),
            kinds
                .iter()
                .map(|(k, n)| format!("{n} {k}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    line(&format!(
        "the frame: {}",
        kinds_all
            .iter()
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for (what, (n, bytes)) in &copies {
        line(&format!("copies, {what}: {n}, {} KB", bytes >> 10));
    }
    let mut moved: Vec<(&String, &u32)> = transitions.iter().collect();
    moved.sort_by(|a, b| b.1.cmp(a.1));
    line(&format!(
        "transitions: {}",
        moved
            .iter()
            .take(12)
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    // The host layer's own passes, and the title's.
    let ours = |id: &u32| {
        pipeline_names
            .get(id)
            .is_none_or(|n| !n.starts_with('['))
    };
    let mut all: Vec<(&u32, &Use)> = uses.iter().collect();
    all.sort_by(|a, b| b.1.commands.cmp(&a.1.commands));
    let own: u32 = all.iter().filter(|u| ours(u.0)).map(|u| u.1.commands).sum();
    let total: u32 = all.iter().map(|u| u.1.commands).sum();
    line(&format!(
        "{} pipelines run {total} commands; {own} of them run passes of the host layer's own",
        all.len()
    ));
    let shown = |id: &u32, u: &Use| {
        let name = pipeline_names.get(id).copied().unwrap_or("?");
        let mut what = Vec::new();
        if u.groups > 0 {
            what.push(format!(
                "{} thread groups in all, the largest dispatch {}x{}x{}",
                u.groups, u.largest[0], u.largest[1], u.largest[2]
            ));
        }
        if u.indirect > 0 {
            what.push(format!("{} by arguments from a buffer", u.indirect));
        }
        let mut targets: Vec<(&String, &u32)> = u.targets.iter().collect();
        targets.sort_by(|a, b| b.1.cmp(a.1));
        for (t, n) in targets.iter().take(2) {
            what.push(format!("{n} into {}", if t.is_empty() { "no target" } else { t.as_str() }));
        }
        format!(
            "  {:5} commands in {} lists of {} submissions: pipeline {id} {}: {}",
            u.commands,
            u.lists.len(),
            u.submissions.len(),
            names.of(if name.is_empty() { "unnamed" } else { name }),
            what.join("; ")
        )
    };
    line("the host layer's own passes:");
    for (id, u) in all.iter().filter(|u| ours(u.0)) {
        line(&shown(id, u));
    }
    line(&format!("the title's pipelines with the most commands (of {}):", all.iter().filter(|u| !ours(u.0)).count()));
    for (id, u) in all.iter().filter(|u| !ours(u.0)).take(top) {
        line(&shown(id, u));
    }
    // Root views with addresses in no buffer, and the like.
    let unknown = frame
        .lists
        .iter()
        .filter_map(|l| Cmd::all(&l.commands.0).ok())
        .flatten()
        .filter(|c| matches!(c, Cmd::RootView { va: Va::Unknown { .. }, .. }))
        .count();
    if unknown > 0 {
        line(&format!("{unknown} root views name memory in no known buffer"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_are_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find(dir.path()).is_err());
        for n in [1, 3, 2] {
            let c = dir.path().join(format!("capture-{n}"));
            std::fs::create_dir(&c).unwrap();
            if n != 3 {
                std::fs::write(c.join("frame.bin"), b"").unwrap();
            }
        }
        // The last one that is complete.
        assert_eq!(find(dir.path()).unwrap(), dir.path().join("capture-2"));
        let one = dir.path().join("capture-1");
        assert_eq!(find(&one).unwrap(), one);
    }

    #[test]
    fn shaders_get_their_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("names.txt");
        std::fs::write(
            &path,
            "0123456789abcdef a title shader\nfedcba9876543210 a title shader\n1111111111111111\n",
        )
        .unwrap();
        let names = Names::read(Some(&path)).unwrap();
        assert_eq!(
            names.of("[mesh 0123456789abcdef, pixel fedcba9876543210]"),
            "[mesh 0123456789abcdef a title shader, pixel fedcba9876543210 a title shader]"
        );
        assert_eq!(
            names.of("[compute 1111111111111111]"),
            "[compute 1111111111111111]"
        );
        assert_eq!(names.of("victory: clamp"), "victory: clamp");
        assert_eq!(Names::default().of("[mesh 0123456789abcdef]"), "[mesh 0123456789abcdef]");
    }
}
