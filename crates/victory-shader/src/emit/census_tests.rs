//! An ISA census: every opcode of every table is embedded in a minimal
//! pixel program and driven through the whole translation pipeline. The
//! outcome must be a translation or a refusal naming what it lacks; a
//! panic is a bug on a path a title could reach with the same encoding.

use super::container::{self, SigElement};
use super::{HostShader, emit};
use crate::dxbc::{Container, FourCC};
use crate::gfx10::opcodes;
use crate::native::{HwStage, NativeProgram, StageIo};
use crate::{ShaderError, gfx10, io, ir, resource, structure};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn program(instr: &[u32]) -> NativeProgram {
    let mut words = vec![0x7E02_0285];
    words.extend_from_slice(instr);
    words.extend_from_slice(&[0xF800_1883, 0x0000_0100, 0xBF81_0000]);
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 2,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn output_signature() -> Container<'static> {
    let el = |name: &str, kind, comp| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row: None,
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    };
    let osg1 = container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
    let bytes: &'static [u8] = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]).leak();
    Container::parse(bytes).unwrap()
}

/// The instruction words each family's opcode is encoded with. Sources
/// are inline constants and registers read as v0: the census measures
/// whether the pipeline can shape and express the operation, not its
/// operands.
fn words(family: &str, op: u16) -> Vec<u32> {
    let op = u32::from(op);
    match family {
        "SOP2" => vec![0x8000_0000 | op << 23 | 128 | 128 << 8],
        "SOPK" => vec![0xB000_0000 | op << 23],
        "SOP1" => vec![0xBE80_0000 | op << 8 | 128],
        "SOPC" => vec![0xBF00_0000 | op << 16 | 128 | 128 << 8],
        "SOPP" => vec![0xBF80_0000 | (op & 0x7F) << 16],
        "SMEM" => vec![0xF400_0000 | op << 18, 0],
        "VOP2" => vec![op << 25 | 257 | 1 << 9],
        "VOP1" => vec![0x7E00_0000 | op << 9 | 257],
        "VOPC" => vec![0x7C00_0000 | op << 17 | 257 | 1 << 9],
        "VOP3" => vec![0xD400_0000 | op << 16, 128 | 128 << 9 | 128 << 18],
        "VOP3P" => vec![0xCC00_0000 | op << 16, 128 | 128 << 9 | 128 << 18],
        "VINTRP" => vec![0xC800_0000 | op << 16],
        "DS" => vec![0xD800_0000 | op << 18, 0],
        "MUBUF" => vec![0xE000_0000 | (op & 0x7F) << 18 | (op >> 7 & 1) << 25, 0],
        "MTBUF" => vec![0xE800_0000 | (op & 7) << 16, (op >> 3 & 1) << 21],
        "MIMG" => vec![0xF000_0000 | (op & 0x7F) << 18 | (op >> 7 & 1), 0],
        "GLOBAL" => vec![0xDC00_0000 | 2 << 14 | op << 18, 0],
        "SCRATCH" => vec![0xDC00_0000 | 1 << 14 | op << 18, 0],
        "FLAT" => vec![0xDC00_0000 | op << 18, 0],
        _ => unreachable!("{family}"),
    }
}

/// `v_madmk_f32` and its siblings consume the following dword as their
/// inline constant, and `s_setreg_imm32_b32` the dword after it.
fn eats_a_literal(name: &str) -> bool {
    matches!(
        name,
        "v_madmk_f32"
            | "v_madak_f32"
            | "v_fmamk_f32"
            | "v_fmaak_f32"
            | "v_madmk_f16"
            | "v_madak_f16"
            | "v_fmamk_f16"
            | "v_fmaak_f16"
            | "s_setreg_imm32_b32"
    )
}

fn families() -> Vec<(&'static str, &'static [(u16, &'static str)])> {
    vec![
        ("SOP2", opcodes::SOP2),
        ("SOPK", opcodes::SOPK),
        ("SOP1", opcodes::SOP1),
        ("SOPC", opcodes::SOPC),
        ("SOPP", opcodes::SOPP),
        ("SMEM", opcodes::SMEM),
        ("VOP2", opcodes::VOP2),
        ("VOP1", opcodes::VOP1),
        ("VOPC", opcodes::VOPC),
        ("VOP3", opcodes::VOP3),
        ("VOP3P", opcodes::VOP3P),
        ("VINTRP", opcodes::VINTRP),
        ("DS", opcodes::DS),
        ("MUBUF", opcodes::MUBUF),
        ("MTBUF", opcodes::MTBUF),
        ("MIMG", opcodes::MIMG),
        ("GLOBAL", opcodes::GLOBAL),
        ("SCRATCH", opcodes::SCRATCH),
        ("FLAT", opcodes::FLAT),
    ]
}

/// Translate one program; `Ok` if the pipeline emits a shader.
fn translate(c: &Container, native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    emit(c, native, &p, &st, &io, &res)
}

#[test]
fn every_opcode_translate_or_refuse_never_panic() {
    let c = &output_signature();
    // The program around the instruction must itself be translatable.
    match translate(c, &program(&words("VOP1", 1))) {
        Ok(_) => {}
        Err(e) => panic!("baseline: {e}"),
    }

    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut panics = vec![];
    let mut wrong_name = vec![];
    for (family, table) in families() {
        for &(op, name) in table {
            let mut instr = words(family, op);
            if eats_a_literal(name) {
                instr.push(0);
            }
            let native = program(&instr);
            // The decoder must read back the name the table promises, or
            // the encoder (this test) and the decoder disagree.
            match gfx10::decode(native.patched_code().as_slice()) {
                Ok(decoded) if decoded[1].name() != Some(name) => wrong_name.push(format!(
                    "{family} {op:#05x} {name} decodes as {:?}",
                    decoded[1].name()
                )),
                Err(e) => wrong_name.push(format!("{family} {op:#05x} {name} {e}")),
                Ok(_) => {}
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| translate(c, &native)));
            match outcome {
                Ok(Ok(_)) => ok += 1,
                Ok(Err(e)) => {
                    refused += 1;
                    eprintln!("census {family} {op:#05x} {name} refused: {e}");
                }
                Err(_) => panics.push(format!("{family} {op:#05x} {name}")),
            }
        }
    }
    eprintln!(
        "census: {ok} translated, {refused} refused, {} panicked",
        panics.len()
    );
    assert!(
        wrong_name.is_empty(),
        "the decoder names these differently:\n{}",
        wrong_name.join("\n")
    );
    assert!(
        panics.is_empty(),
        "panicking opcodes:\n{}",
        panics.join("\n")
    );
}
