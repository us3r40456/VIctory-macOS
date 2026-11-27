//! Non-provoking vertex values on the host (#203). The console keeps each
//! interpolated attribute of a triangle as its value at the first vertex P0
//! and the differences P10 = P1 - P0 and P20 = P2 - P0, and a pixel
//! program can read any of the three (`v_interp_mov_f32`). P0 is a flat
//! input. For P10 and P20 the host's own read (`GetAttributeAtVertex`)
//! makes a pipeline that D3DMetal runs as a no-op, so a pixel shader reads
//! them as flat inputs of their own ([`VERTEX_DELTAS`], indexed by the
//! attribute's row), in rows taken from [`VERTEX_DELTA_TOP`] down. A
//! geometry shader made for the vertex shader passes its outputs through
//! and adds those inputs, computed per triangle from its three vertices
//! ([`vertex_delta_gs`]).
//!
//! shortcut: the vertices are taken in the order the host gives them to
//! a geometry shader (strips: every other triangle's first two swapped, to
//! keep their winding), which is assumed to be the console's.

use super::bitcode::{Attr, Bin, Term};
use super::container::SigElement;
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The semantics of P10 and P20 inputs; the index is the attribute's row.
pub const VERTEX_DELTAS: [&str; 2] = ["VICTORY_P10", "VICTORY_P20"];

/// The row of the first such input; the next ones go below it.
pub const VERTEX_DELTA_TOP: u32 = 31;

/// The rows of a container's signature part with a register.
fn elements(
    container: &[u8],
    part: FourCC,
    what: &'static str,
) -> Result<Vec<SigElement>, ShaderError> {
    let c = Container::parse(container)?;
    let part = c.part(part).ok_or(ShaderError::Malformed(what))?;
    let mut elements = super::signature(part.data)?;
    elements.retain(|e| e.row.is_some());
    Ok(elements)
}

/// Whether the pixel shader container `ps` reads P10 or P20 values, so it
/// needs [`vertex_delta_gs`] before it.
pub fn needs_vertex_deltas(ps: &[u8]) -> bool {
    elements(ps, FourCC::ISG1, "").is_ok_and(|inputs| {
        inputs
            .iter()
            .any(|e| VERTEX_DELTAS.contains(&e.name.as_str()))
    })
}

/// The geometry shader that gives the pixel shader container `ps` the P10
/// and P20 values it reads of the outputs of the vertex shader container
/// `vs` (triangles only).
pub fn vertex_delta_gs(vs: &[u8], ps: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let mut outputs = elements(vs, FourCC::OSG1, "a vertex shader without outputs")?;
    for e in &mut outputs {
        e.usage = (1 << e.cols) - 1;
    }
    let refuse = |what| ShaderError::Unsupported { offset: 0, what };
    // Each delta input with the vertex output (element, first column) it
    // is taken from.
    let mut deltas = Vec::new();
    for d in elements(ps, FourCC::ISG1, "a pixel shader without inputs")? {
        let Some(vertex) = VERTEX_DELTAS.iter().position(|&n| n == d.name) else {
            continue;
        };
        let row = d.row.expect("kept with a row");
        if outputs.iter().any(|o| o.row == Some(row)) {
            return Err(refuse(
                "vertex outputs in the rows of non-provoking vertex values",
            ));
        }
        let from = outputs
            .iter()
            .position(|o| {
                o.row == Some(d.index) && o.col <= d.col && d.col + d.cols <= o.col + o.cols
            })
            .ok_or(refuse("a non-provoking vertex value of no vertex output"))?;
        if outputs[from].comp != 9 {
            return Err(refuse("a non-provoking vertex value of an integer output"));
        }
        let first = d.col - outputs[from].col;
        deltas.push((
            SigElement {
                interp: 0,
                usage: (1 << d.cols) - 1,
                ..d
            },
            vertex as u32 + 1,
            from,
            first,
        ));
    }
    if deltas.is_empty() {
        return Err(ShaderError::Malformed(
            "a pixel shader that reads no non-provoking vertex value",
        ));
    }
    let mut s = Shader::new(false);
    s.gs = Some(3);
    let b = s.entry;
    let load = |s: &mut Shader, id: usize, e: &SigElement, col: u8, k: u32| {
        let (ty, suffix) = if e.comp == 9 {
            (s.t.f32, ".f32")
        } else {
            (s.t.i32, ".i32")
        };
        let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
        s.op(
            b,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        )
    };
    // The differences, the same at every vertex.
    let mut values = Vec::new();
    for (d, vertex, from, first) in &deltas {
        let e = &outputs[*from];
        let mut cols = Vec::new();
        for col in 0..d.cols {
            let at = first + col;
            let (pk, p0) = (
                load(&mut s, *from, e, at, *vertex),
                load(&mut s, *from, e, at, 0),
            );
            cols.push(s.m.bin(b, Bin::Sub, pk, p0));
        }
        values.push(cols);
    }
    for k in 0..3u32 {
        for (id, e) in outputs.iter().enumerate() {
            let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
            for col in 0..e.cols {
                let v = load(&mut s, id, e, col, k);
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                s.op(
                    b,
                    &format!("storeOutput{suffix}"),
                    Attr::NoUnwind,
                    s.t.void,
                    5,
                    &args,
                );
            }
        }
        for (j, cols) in values.iter().enumerate() {
            for (col, &v) in cols.iter().enumerate() {
                let id = s.m.i32((outputs.len() + j) as u32);
                let args = [id, s.m.i32(0), s.m.i8(col as u8), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
        }
        let stream = [s.m.i8(0)];
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
    }
    let stream = [s.m.i8(0)];
    s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    s.inputs = outputs.clone();
    s.outputs = outputs;
    s.outputs.extend(deltas.into_iter().map(|(d, ..)| d));
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;
    use super::super::validator::Validator;
    use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};

    fn element(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A vertex shader copying POSITION and TEXCOORD (two channels, row 1)
    /// to its outputs.
    fn vertex_shader() -> Vec<u8> {
        let mut s = Shader::new(false);
        let b = s.entry;
        let outputs = vec![
            element("SV_Position", 3, 0, 0, 4),
            element("TEXCOORD", 0, 0, 1, 2),
        ];
        for (id, e) in outputs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [
                    s.m.i32(id as u32),
                    s.m.i32(0),
                    s.m.i8(col),
                    s.m.undef(s.t.i32),
                ];
                let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
        }
        s.inputs = vec![
            element("POSITION", 0, 0, 0, 4),
            element("TEXCOORD", 0, 0, 1, 2),
        ];
        s.outputs = outputs;
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// A console pixel program writing target 0 from TEXCOORD (attribute 0,
    /// row 1): P10 of x, P20 of x, P10 of y, P0 of y.
    fn pixel_shader() -> Vec<u8> {
        let code: [u32; 7] = [
            0xC802_0000, // v_interp_mov_f32 v0, p10, attr0.x
            0xC806_0001, // v_interp_mov_f32 v1, p20, attr0.x
            0xC80A_0100, // v_interp_mov_f32 v2, p10, attr0.y
            0xC80E_0102, // v_interp_mov_f32 v3, p0, attr0.y
            0xF800_180F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: true,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 1,
                    mask: 0b11,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        super::super::lower::lower(&super::super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![SigElement {
                usage: 0,
                ..element("TEXCOORD", 0, 0, 1, 2)
            }],
            outputs: vec![element("SV_Target", 16, 0, 0, 4)],
            prims: Vec::new(),
            mesh_stride: 0,
            atomic_locks: false,
            halves_local: Default::default(),
            mac: Default::default(),
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: false,
            host_wave32: false,
            view_marks: false,
        })
        .unwrap()
        .dxil
    }

    /// The pixel shader reads P10 and P20 as inputs of their own, which the
    /// geometry shader made for the vertex shader computes from each
    /// triangle's vertices: every pixel of a triangle gets its differences,
    /// and P0 stays the provoking vertex's value.
    #[test]
    fn non_provoking_vertex_values_reach_the_pixel_shader() {
        let (vs, ps) = (vertex_shader(), pixel_shader());
        assert!(super::needs_vertex_deltas(&ps));
        assert!(!super::needs_vertex_deltas(&vs));
        let gs = super::vertex_delta_gs(&vs, &ps).unwrap();
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        // A triangle over the whole target: TEXCOORD (1, 10), (4, 30), (9, 70).
        let vertices = [
            [-1.0, -1.0, 0.5, 1.0, 1.0, 10.0],
            [3.0, -1.0, 0.5, 1.0, 4.0, 30.0],
            [-1.0, 3.0, 0.5, 1.0, 9.0, 70.0],
        ]
        .concat();
        let mut drawn = false;
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 24,
                size: (8, 8),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            for (i, t) in texels.iter().enumerate() {
                assert_eq!(*t, [3.0, 8.0, 20.0, 10.0], "warp {warp} pixel {i}");
            }
            drawn = true;
        }
        assert!(drawn, "no device");
    }
}
