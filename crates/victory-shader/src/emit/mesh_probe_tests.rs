//! macOS port: a probe that measures how D3DMetal sizes a mesh shader's
//! outputs. Metal refuses a mesh pipeline whose outputs are over 32,768
//! bytes; D3DMetal then prints `Total mesh size (N) exceeds ...` on the
//! process's (Unix) standard output and hands back a pipeline that draws
//! nothing. The probe makes pipelines of hand-assembled mesh and pixel
//! shaders with exactly the output signature a line of text describes and
//! reads N back from the file the process's output was sent to.
//!
//! Sizes under the limit are measured too. The total is `a * vertices +
//! b * primitives`, the vertices rounded up to a multiple of 4 (the sweep's
//! first rows show it, and each row's own make is checked against it), so
//! the bytes of one vertex are the difference of two refused makes that
//! declare 256 and 128 vertices, divided by 128. Where those would fit,
//! per-primitive rows are added until both are refused; they are the same
//! in both makes and cancel. A primitive's bytes come the same way from 256
//! and 128 primitives, padded with vertex rows.
//!
//! No game and no game data. Run it with `tools/macos/mesh-probe.sh`.
//! What it measured, and the rule that gives all of it, are in
//! `mesh_size`: the sweep is that table, measured again.
//!
//! A signature, one line: elements separated by spaces, in signature order.
//!
//! ```text
//! P            SV_Position (float4, noperspective)
//! f1..f4       a float element of 1 to 4 components, perspective
//! c1..c4       the same, constant (flat);  n1..n4 noperspective;
//!              e1..e4 centroid;  s1..s4 sample
//! i1..i4       signed integers (constant);  u1..u4 unsigned
//! C1..C4       SV_ClipDistance;  K1..K4 SV_CullDistance
//! q<element>   a per-primitive element (qf4, qu1)
//! +<element>   in the register of the element before it, next column
//! <element>@7  in register 7 (@7.2: from column 2); without, the next one
//! f3:COLOR2    named COLOR2 (without: TEXCOORD<n>, counted in the line's
//!              order; PRIMDATA<n> per primitive)
//! f4!b         only the components of this mask (hex) are written and read
//! C4^4         with this DXIL interpolation mode (before `!` and `:`)
//! <element>*5  five times
//! v=128 p=128  the declared maximum vertices and primitives
//! ps=read      the pixel shader declares and reads the elements (default);
//!              ps=decl declares them only; ps=none has no inputs
//! pd=0         the pixel shader leaves out the clip and cull distances
//! pp=1         the pixel shader declares SV_Position
//! ```

use super::container::SigElement;

/// Metal's limit, in bytes.
#[cfg(windows)]
const LIMIT: u32 = super::METAL_MESH_OUTPUT_BYTES;

/// A signature to measure.
#[derive(Debug, Clone)]
pub(crate) struct Probe {
    pub vertex: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    vertices: u32,
    primitives: u32,
    /// 0: the pixel shader has no inputs; 1: it declares the elements;
    /// 2: it reads them too.
    pixel: u8,
    pixel_distances: bool,
    pixel_position: bool,
}

pub(crate) fn parse(text: &str) -> Result<Probe, String> {
    let mut p = Probe {
        vertex: Vec::new(),
        prims: Vec::new(),
        vertices: 128,
        primitives: 128,
        pixel: 2,
        pixel_distances: true,
        pixel_position: false,
    };
    // (element, per-primitive, register given)
    let mut elements: Vec<(SigElement, bool, bool)> = Vec::new();
    // TEXCOORD, SV_ClipDistance, SV_CullDistance, PRIMDATA.
    let mut indices = [0u32; 4];
    for token in text.split_whitespace() {
        if let Some((key, value)) = token.split_once('=') {
            let number = || value.parse::<u32>().map_err(|_| format!("{token}: not a number"));
            match key {
                "v" => p.vertices = number()?,
                "p" => p.primitives = number()?,
                "ps" => {
                    p.pixel = match value {
                        "none" => 0,
                        "decl" => 1,
                        "read" => 2,
                        _ => return Err(format!("{token}: none, decl or read")),
                    }
                }
                "pd" => p.pixel_distances = number()? != 0,
                "pp" => p.pixel_position = number()? != 0,
                _ => return Err(format!("{token}: unknown option")),
            }
            continue;
        }
        let (token, times) = match token.split_once('*') {
            Some((t, n)) => (t, n.parse::<u32>().map_err(|_| format!("{token}: bad count"))?),
            None => (token, 1),
        };
        for _ in 0..times {
            let (same, t) = match token.strip_prefix('+') {
                Some(t) => (true, t),
                None => (false, token),
            };
            let (prim, t) = match t.strip_prefix('q') {
                Some(t) => (true, t),
                None => (false, t),
            };
            let (shape, place) = match t.split_once('@') {
                Some((s, at)) => (s, Some(at)),
                None => (t, None),
            };
            // <letter><components>[^<mode>][!<mask>][:<NAME><index>]
            let (shape, semantic) = match shape.split_once(':') {
                Some((s, name)) => (s, Some(name)),
                None => (shape, None),
            };
            let (shape, written) = match shape.split_once('!') {
                Some((s, mask)) => (s, Some(u8::from_str_radix(mask, 16).map_err(|_| format!("{token}: bad mask"))?)),
                None => (shape, None),
            };
            let (shape, mode) = match shape.split_once('^') {
                Some((s, mode)) => (s, Some(mode.parse::<u8>().map_err(|_| format!("{token}: bad mode"))?)),
                None => (shape, None),
            };
            let mut chars = shape.chars();
            let letter = chars.next().ok_or_else(|| format!("{token}: empty"))?;
            let cols = match chars.as_str() {
                "" if letter == 'P' => 4,
                n => n.parse::<u8>().ok().filter(|n| (1..=4).contains(n)).ok_or_else(|| format!("{token}: 1 to 4 components"))?,
            };
            let (kind, comp, interp, name, counter) = match letter {
                'P' => (3, 9, 4, "SV_Position", None),
                'f' => (0, 9, 2, "TEXCOORD", Some(0)),
                'c' => (0, 9, 1, "TEXCOORD", Some(0)),
                'n' => (0, 9, 4, "TEXCOORD", Some(0)),
                'e' => (0, 9, 3, "TEXCOORD", Some(0)),
                's' => (0, 9, 6, "TEXCOORD", Some(0)),
                'i' => (0, 4, 1, "TEXCOORD", Some(0)),
                'u' => (0, 5, 1, "TEXCOORD", Some(0)),
                'C' => (6, 9, 2, "SV_ClipDistance", Some(1)),
                'K' => (7, 9, 2, "SV_CullDistance", Some(2)),
                _ => return Err(format!("{token}: unknown element")),
            };
            let (name, counter) = if prim && kind == 0 { ("PRIMDATA", Some(3)) } else { (name, counter) };
            let (name, index) = match semantic {
                Some(text) => {
                    let name = text.trim_end_matches(|c: char| c.is_ascii_digit());
                    let index = text[name.len()..].parse::<u32>().map_err(|_| format!("{token}: a name and an index"))?;
                    (name, index)
                }
                None => {
                    let index = counter.map_or(0, |c: usize| {
                        indices[c] += 1;
                        indices[c] - 1
                    });
                    (name, index)
                }
            };
            let interp = mode.unwrap_or(interp);
            let last = elements.iter().rev().find(|e| e.1 == prim);
            let (row, col, placed) = match (place, same) {
                (Some(at), _) => {
                    let (r, c) = at.split_once('.').unwrap_or((at, "0"));
                    let bad = || format!("{token}: bad register");
                    (r.parse::<u32>().map_err(|_| bad())?, c.parse::<u8>().map_err(|_| bad())?, true)
                }
                (None, true) => {
                    let (e, _, placed) = last.ok_or_else(|| format!("{token}: nothing before it"))?;
                    (e.row.unwrap_or(0), e.col + e.cols, *placed)
                }
                // Numbered below.
                (None, false) => (u32::MAX, 0, false),
            };
            if col + cols > 4 {
                return Err(format!("{token}: past the register's fourth column"));
            }
            let element = SigElement {
                name: name.to_string(),
                index,
                kind,
                comp,
                // Per-primitive values are not interpolated.
                interp: if prim { 1 } else { interp },
                row: Some(row),
                col,
                cols,
                usage: written.unwrap_or((1 << cols) - 1) & ((1 << cols) - 1),
                half: false,
            };
            elements.push((element, prim, placed || same && row != u32::MAX));
        }
    }
    // Registers not given: the next one, the per-primitive elements' after
    // the vertex elements' (as DXC allocates them, so the pixel shader's
    // inputs do not overlap).
    let mut next = 0;
    for prim in [false, true] {
        let mut at = 0;
        for i in 0..elements.len() {
            if elements[i].1 != prim {
                continue;
            }
            let (e, _, placed) = &mut elements[i];
            if !*placed {
                if e.col == 0 {
                    at = next;
                }
                e.row = Some(at);
            }
            at = e.row.unwrap_or(0);
            next = next.max(at + 1);
        }
    }
    for (e, prim, _) in elements {
        if prim { p.prims.push(e) } else { p.vertex.push(e) }
    }
    if p.vertices < 3 || p.primitives < 1 || p.vertices > 256 || p.primitives > 256 {
        return Err("3 to 256 vertices and 1 to 256 primitives".to_string());
    }
    Ok(p)
}

/// A signature in the notation [`parse`] reads: each element's kind,
/// components and place (names and modes left out: they change no size).
pub(crate) fn line(elements: &[SigElement]) -> String {
    let mut out = Vec::new();
    for e in elements {
        let Some(row) = e.row else { continue };
        let letter = match (e.kind, e.comp) {
            (3, _) => "P".to_string(),
            (6, _) => format!("C{}", e.cols),
            (7, _) => format!("K{}", e.cols),
            (_, 9) => format!("f{}", e.cols),
            (_, 4) => format!("i{}", e.cols),
            _ => format!("u{}", e.cols),
        };
        let col = if e.col > 0 { format!(".{}", e.col) } else { String::new() };
        out.push(format!("{letter}@{row}{col}"));
    }
    out.join(" ")
}

/// `N` of `Total mesh size (N)`.
fn total_in(text: &str) -> Option<u32> {
    let at = text.find("Total mesh size (")? + "Total mesh size (".len();
    let end = text[at..].find(')')?;
    text[at..at + end].parse().ok()
}

#[cfg(windows)]
mod gpu {
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::super::HostShader;
    use super::super::bitcode::{Attr, Bin, Cast, Term, pred};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host, Root};
    use super::super::shader::{Mesh, Shader};
    use super::super::mesh_size;
    use super::{LIMIT, Probe, parse, total_in};

    /// Makes each shader's bytes its own, so nothing D3DMetal or Metal cached
    /// for an earlier make (of this or another run) answers for it.
    static NONCE: AtomicU32 = AtomicU32::new(0);

    fn nonce() -> u32 {
        let _ = NONCE.compare_exchange(
            0,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_millis() as u32 | 1),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        NONCE.fetch_add(1, Ordering::Relaxed)
    }

    /// The mesh shader: three threads, each its corner of one triangle that
    /// covers the target, every output element stored whole.
    fn mesh_shader(p: &Probe, nonce: u32) -> Vec<u8> {
        let mut s = Shader::new(false);
        s.threads = Some([3, 1, 1]);
        s.mesh = Some(Mesh {
            max_vertices: p.vertices,
            max_primitives: p.primitives,
            topology: 2,
        });
        s.outputs = p.vertex.clone();
        s.prims = p.prims.clone();
        let b = s.entry;
        let (void, i32t) = (s.t.void, s.t.i32);
        let (v, n) = (s.m.i32(p.vertices), s.m.i32(p.primitives));
        s.op(b, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, n]);
        let thread = s.op(b, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let (zero, one, two) = (s.m.i32(0), s.m.i32(1), s.m.i32(2));
        s.op(b, "emitIndices", Attr::NoUnwind, void, 169, &[zero, zero, one, two]);
        let float = |s: &mut Shader, f: f32| s.m.f32(f.to_bits());
        let (far, near) = (float(&mut s, 3.0), float(&mut s, -1.0));
        let (is1, is2) = (s.m.cmp(b, pred::EQ, thread, one), s.m.cmp(b, pred::EQ, thread, two));
        let position = [
            s.m.select(b, is1, far, near),
            s.m.select(b, is2, far, near),
            // In [0.5, 0.75): the make's own bits.
            s.m.f32(0x3F00_0000 | nonce & 0x3F_FFFF),
            float(&mut s, 1.0),
        ];
        for (primitive, elements) in [(false, p.vertex.clone()), (true, p.prims.clone())] {
            for (id, e) in elements.iter().enumerate() {
                for c in (0..e.cols).filter(|c| e.usage >> c & 1 != 0) {
                    let (suffix, value) = match (e.kind, e.comp) {
                        (3, _) => (".f32", position[usize::from(c)]),
                        // Distances inside every plane.
                        (6 | 7, _) => (".f32", float(&mut s, 1.0)),
                        (_, 9) => (".f32", float(&mut s, 0.125 * f32::from(c + 1))),
                        _ => (".i32", s.m.i32(u32::from(c) + 1)),
                    };
                    let (name, opcode, index) = if primitive {
                        ("storePrimitiveOutput", 172, zero)
                    } else {
                        ("storeVertexOutput", 171, thread)
                    };
                    let args = [s.m.i32(id as u32), zero, s.m.i8(c), value, index];
                    s.op(b, &format!("{name}{suffix}"), Attr::NoUnwind, void, opcode, &args);
                }
            }
        }
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// The pixel shader: target 0 is (the sum of what it reads, 0.5, the
    /// make's own bits, 1).
    fn pixel_shader(p: &Probe, nonce: u32) -> Vec<u8> {
        let mut s = Shader::new(true);
        if p.pixel > 0 {
            for e in p.vertex.iter().chain(&p.prims) {
                let declared = match e.kind {
                    0 => true,
                    3 => p.pixel_position,
                    _ => p.pixel_distances,
                };
                if declared {
                    let read = e.kind == 0 && p.pixel == 2;
                    s.inputs.push(SigElement {
                        usage: if read { e.usage } else { 0 },
                        ..e.clone()
                    });
                }
            }
        }
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        });
        let b = s.entry;
        let (void, i32t, f32t) = (s.t.void, s.t.i32, s.t.f32);
        let mut sum = s.m.f32(0);
        for (id, e) in s.inputs.clone().iter().enumerate() {
            for c in (0..e.cols).filter(|c| e.usage >> c & 1 != 0) {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(c), s.m.undef(i32t)];
                let x = if e.comp == 9 {
                    s.op(b, "loadInput.f32", Attr::ReadNone, f32t, 4, &args)
                } else {
                    let x = s.op(b, "loadInput.i32", Attr::ReadNone, i32t, 4, &args);
                    s.m.cast(b, Cast::UiToFp, x, f32t)
                };
                sum = s.m.bin(b, Bin::Add, sum, x);
            }
        }
        let color = [
            sum,
            s.m.f32(0.5f32.to_bits()),
            s.m.f32(0x3F00_0000 | nonce & 0x3F_FFFF),
            s.m.f32(1.0f32.to_bits()),
        ];
        for (c, v) in color.into_iter().enumerate() {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(c as u8), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, void, 5, &args);
        }
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// The file the process's output goes to, read from where the last read
    /// ended: D3DMetal writes its messages to the Unix standard output (the
    /// game's log has them for that reason), not to a D3D12 message queue.
    struct Messages {
        path: std::path::PathBuf,
        at: usize,
    }

    impl Messages {
        fn open() -> Option<Self> {
            let path = std::path::PathBuf::from(std::env::var_os("VICTORY_MESH_PROBE_LOG")?);
            let at = std::fs::read(&path).ok()?.len();
            Some(Messages { path, at })
        }

        fn take(&mut self) -> String {
            let all = std::fs::read(&self.path).unwrap_or_default();
            let new = String::from_utf8_lossy(all.get(self.at..).unwrap_or_default()).into_owned();
            self.at = all.len().max(self.at);
            new
        }
    }

    /// What a make gave: the total Metal refused it for, or another message of
    /// D3DMetal's about the pipeline, and whether a triangle drew.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct Made {
        total: Option<u32>,
        other: Option<String>,
        drawn: Option<bool>,
        millis: f64,
    }

    impl Made {
        fn word(&self) -> String {
            match (self.total, &self.other) {
                (Some(n), _) => format!("REFUSED {n}"),
                (None, Some(m)) => format!("ERROR {m}"),
                (None, None) => "ok".to_string(),
            }
        }
    }

    fn make(host: &Host, messages: &mut Messages, p: &Probe, draw: bool) -> Result<Made, String> {
        use windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_MESH;
        let n = nonce();
        let ms = HostShader {
            stage: crate::native::HwStage::Ms,
            dxil: mesh_shader(p, n),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_srv_bytes: Vec::new(),
            memory: false,
            memory_writes: false,
            view_addresses: false,
            heap: false,
            heap_writes: false,
            view_marks: false,
        };
        let ps = pixel_shader(p, n);
        let root = host.root_signature_for(&[(&ms, D3D12_SHADER_VISIBILITY_MESH)])?;
        messages.take();
        let started = std::time::Instant::now();
        let pso = host.mesh_pipeline_with(&root, &ms.dxil, &ps)?;
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let mut text = messages.take();
        // The message is written before the make returns (checked: a wait
        // here never found one more); the wait is for a D3DMetal that changes.
        let wait = std::env::var("VICTORY_MESH_PROBE_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if wait > 0 && total_in(&text).is_none() {
            std::thread::sleep(std::time::Duration::from_millis(wait));
            text.push_str(&messages.take());
        }
        let total = total_in(&text);
        let other = text
            .lines()
            .find(|l| l.contains("[ERROR]") && !l.contains("Total mesh size"))
            .map(|l| l.chars().take(200).collect());
        let drawn = if draw {
            let texels = host.draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(vec![0]))],
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (8, 8),
                    mesh_groups: 1,
                },
            )?;
            messages.take();
            Some(texels.iter().all(|t| t[1] == 0.5 && t[3] == 1.0))
        } else {
            None
        };
        Ok(Made { total, other, drawn, millis })
    }

    /// One row of the table.
    #[derive(Debug, Clone, Default)]
    struct Measured {
        /// The make as the line says.
        given: Made,
        /// Bytes of one vertex, the per-primitive float4 rows added to measure
        /// them, and what the 256-vertex make's total holds besides vertices.
        vertex: Option<(f64, u32, f64)>,
        /// Bytes of one primitive and the vertex float4 rows added.
        primitive: Option<(f64, u32)>,
        note: String,
    }

    /// The line with `v` vertices, `p` primitives and float4 rows added.
    fn variant(text: &str, v: u32, p: u32, vertex_rows: u32, primitive_rows: u32) -> Result<Probe, String> {
        let mut t = format!("{text} v={v} p={p}");
        if vertex_rows > 0 {
            write!(t, " f4*{vertex_rows}").expect("a string");
        }
        if primitive_rows > 0 {
            write!(t, " qf4*{primitive_rows}").expect("a string");
        }
        parse(&t)
    }

    fn measure(host: &Host, messages: &mut Messages, text: &str, draw: bool) -> Result<Measured, String> {
        let probe = parse(text)?;
        let mut m = Measured {
            given: make(host, messages, &probe, draw)?,
            ..Default::default()
        };
        // A vertex: 256 and 128 vertices, both refused.
        for pad in [0, 8, 16, 24] {
            let high = make(host, messages, &variant(text, 256, 256, 0, pad)?, false)?;
            let low = make(host, messages, &variant(text, 128, 256, 0, pad)?, false)?;
            if let Some(other) = high.other.or(low.other) {
                write!(m.note, " [{other}]").expect("a string");
                break;
            }
            if let (Some(high), Some(low)) = (high.total, low.total) {
                let a = f64::from(high - low) / 128.0;
                m.vertex = Some((a, pad, f64::from(high) - 256.0 * a));
                break;
            }
        }
        // A primitive: 256 and 128 primitives, both refused.
        if !probe.prims.is_empty() {
            for pad in [0, 7, 15, 23] {
                let high = make(host, messages, &variant(text, 256, 256, pad, 0)?, false)?;
                let low = make(host, messages, &variant(text, 256, 128, pad, 0)?, false)?;
                if let (Some(high), Some(low)) = (high.total, low.total) {
                    m.primitive = Some((f64::from(high - low) / 128.0, pad));
                    break;
                }
            }
        }
        // The rule, and the sweep's table, against the measurement.
        let rule = (mesh_size::struct_bytes(&probe.vertex), mesh_size::struct_bytes(&probe.prims));
        let measured = (m.vertex.map(|v| v.0), m.primitive.map(|p| p.0));
        if measured.0.is_some_and(|a| a != f64::from(rule.0)) || measured.1.is_some_and(|b| b != f64::from(rule.1)) {
            write!(m.note, " [the rule: {} a vertex, {} a primitive]", rule.0, rule.1).expect("a string");
        }
        if let Some(&(_, vertex, primitive)) = mesh_size::MEASURED.iter().find(|row| row.0 == text) {
            let same = measured.0 == Some(f64::from(vertex))
                && measured.1.unwrap_or(0.0) == f64::from(primitive);
            if !same {
                write!(m.note, " [measured before: {vertex} a vertex, {primitive} a primitive]").expect("a string");
            }
        }
        // The make as given against the measured sizes.
        if let Some((a, ..)) = m.vertex {
            let b = m.primitive.map_or(0.0, |(b, _)| b);
            if probe.prims.is_empty() || m.primitive.is_some() {
                let predicted = a * f64::from(probe.vertices.next_multiple_of(4)) + b * f64::from(probe.primitives);
                let agrees = match m.given.total {
                    Some(n) => f64::from(n) == predicted,
                    None => predicted <= f64::from(LIMIT),
                };
                if !agrees && m.given.other.is_none() {
                    write!(m.note, " [as given: {} but a * v + b * p = {predicted}]", m.given.word()).expect("a string");
                }
            }
        }
        if let Some(drawn) = m.given.drawn {
            if drawn != (m.given.total.is_none() && m.given.other.is_none()) {
                write!(m.note, " [drawn {drawn}]").expect("a string");
            }
        }
        Ok(m)
    }

    /// What a `make`'s first error line says, shortened to its text.
    fn short(message: &str) -> String {
        let m = message.split("Error Domain=").nth(1).unwrap_or(message);
        let m = m.split("UserInfo=").next().unwrap_or(m);
        m.trim().to_string()
    }

    /// Measures the sweep (the lines of `mesh_size::MEASURED`, or those of
    /// the file `VICTORY_MESH_PROBE_SIGS` names) on the hardware device and
    /// prints a table, with a note at each row where the measured rule
    /// (`mesh_size`) or the sweep's own table gives another size; `VICTORY_MESH_PROBE_OUT` names a
    /// file that gets it too. `VICTORY_MESH_PROBE_DRAW=1` also draws each line's
    /// own pipeline, to see that a refused one draws nothing and the others
    /// draw. Run by `tools/macos/mesh-probe.sh`, which sends the process's
    /// output to the file `VICTORY_MESH_PROBE_LOG` names.
    #[test]
    #[ignore = "a measuring tool for D3DMetal (tools/macos/mesh-probe.sh), not a check"]
    fn mesh_output_sizes_as_d3dmetal_counts_them() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
        };
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(mut messages) = Messages::open() else {
            eprintln!("VICTORY_MESH_PROBE_LOG does not name the file this process's output goes to; skipped");
            return;
        };
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; skipped");
            return;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        // SAFETY: the query writes the structure of the given size.
        let queried = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
            eprintln!("no mesh shaders on the hardware device; skipped");
            return;
        }
        let lines: Vec<String> = match std::env::var_os("VICTORY_MESH_PROBE_SIGS") {
            Some(path) => std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy()))
                .lines()
                .map(str::to_string)
                .collect(),
            None => mesh_size::MEASURED.iter().map(|m| m.0.to_string()).collect(),
        };
        let draw = std::env::var("VICTORY_MESH_PROBE_DRAW").is_ok_and(|v| v == "1");
        let started = std::time::Instant::now();
        // Each row as it is measured, so a make that ends the process (D3DMetal
        // does not survive every signature) leaves the rows before it and, as
        // the output's last line, the one it was at.
        let out = std::env::var_os("VICTORY_MESH_PROBE_OUT");
        let mut table = String::new();
        let mut row = |text: String| {
            println!("{text}");
            table.push_str(&text);
            table.push('\n');
            if let Some(path) = &out {
                std::fs::write(path, &table).unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy()));
            }
        };
        row(format!(
            "{:<44} | {:<14} | {:>8} | {:>3} | {:>8} | {:>3} | {:>6} |",
            "signature", "as given", "a vertex", "pad", "a prim.", "pad", "rest"
        ));
        let mut makes = 0.0;
        for line in &lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                row(line.to_string());
                continue;
            }
            println!("measuring: {line}");
            let m = match measure(&host, &mut messages, line, draw) {
                Ok(m) => m,
                Err(e) => {
                    row(format!("{line:<44} | {e}"));
                    continue;
                }
            };
            makes += m.given.millis;
            let number = |x: Option<f64>| x.map_or("?".to_string(), |x| format!("{x}"));
            let given = match (&m.given.other, m.given.drawn) {
                (Some(other), _) => format!("ERROR {}", short(other)),
                (None, Some(d)) => format!("{}{}", m.given.word(), if d { ", drawn" } else { ", not drawn" }),
                (None, None) => m.given.word(),
            };
            row(format!(
                "{line:<44} | {given:<14} | {:>8} | {:>3} | {:>8} | {:>3} | {:>6} |{}",
                number(m.vertex.map(|v| v.0)),
                m.vertex.map_or("-".to_string(), |v| v.1.to_string()),
                if m.primitive.is_some() { number(m.primitive.map(|v| v.0)) } else { "-".to_string() },
                m.primitive.map_or("-".to_string(), |v| v.1.to_string()),
                number(m.vertex.map(|v| v.2)),
                m.note,
            ));
        }
        row(format!(
            "# {} lines in {:.1} s (the lines' own makes {:.0} ms)",
            lines.len(),
            started.elapsed().as_secs_f64(),
            makes
        ));
    }
}

/// The signature text makes the registers it says.
#[test]
fn probe_lines_parse() {
    let shape = |els: &[SigElement]| -> Vec<(u8, Option<u32>, u8, u8)> {
        els.iter().map(|e| (e.kind, e.row, e.col, e.cols)).collect()
    };
    let p = parse("P C4 f3 +f1 f2*2 u1@9.2 qf4 qu1 v=200 ps=decl").unwrap();
    assert_eq!(
        shape(&p.vertex),
        vec![
            (3, Some(0), 0, 4),
            (6, Some(1), 0, 4),
            (0, Some(2), 0, 3),
            (0, Some(2), 3, 1),
            (0, Some(3), 0, 2),
            (0, Some(4), 0, 2),
            (0, Some(9), 2, 1),
        ]
    );
    assert_eq!(shape(&p.prims), vec![(0, Some(10), 0, 4), (0, Some(11), 0, 1)]);
    assert_eq!((p.vertices, p.primitives, p.pixel), (200, 128, 1));
    assert_eq!(p.vertex[4].index, 2);
    assert_eq!(p.prims[1].name, "PRIMDATA");
    assert_eq!(total_in("x \"Total mesh size (34816) exceeds the maximum\""), Some(34816));
    let p = parse("P f4^1!b:COLOR3@2 C2^4").unwrap();
    let e = &p.vertex[1];
    assert_eq!((e.name.as_str(), e.index, e.interp, e.usage, e.row), ("COLOR", 3, 1, 0xB, Some(2)));
    assert_eq!((p.vertex[2].interp, p.vertex[2].row), (4, Some(3)));
    for line in super::mesh_size::MEASURED.iter().map(|m| m.0).filter(|l| !l.starts_with('#')) {
        let p = parse(line).unwrap_or_else(|e| panic!("{line}: {e}"));
        assert!(p.vertex.iter().any(|e| e.kind == 3), "{line}");
        assert!(p.pixel <= 2 && (p.pixel_distances || !p.pixel_position), "{line}");
    }
    let text = "P@0 C4@1 f3@2 f1@2.3 u2@3 K1@5";
    assert_eq!(line(&parse(text).unwrap().vertex), text);
    assert!(parse("P f5").is_err());
    assert!(parse("P f3 +f2").is_err());
}
