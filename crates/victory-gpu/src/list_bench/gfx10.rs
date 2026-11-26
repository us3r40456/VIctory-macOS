//! Shaders for the list benchmark, hand-assembled for the console's GPU
//! (RDNA 2, GFX10.3) from AMD's public "RDNA 2 Instruction Set
//! Architecture" reference and packed in the shader container a title hands
//! D3D12.X (the layout `victory_shader::native` reads): the conformance
//! title's GPU frame shaders (branch `tests/gpu-frame-workload`). A vertex
//! shader reading two constant tables, a pixel shader reading two constant
//! tables, a texture and a sampler, and a compute shader reading a texture
//! and writing a texture and a raw buffer, so draws and dispatches bind
//! tables as a title's do.

/// Operand codes (the ISA's SSRC/SRC encoding): SGPRs 0-105, `m0` 124,
/// integer 0 128, 1-64 129-192, -1 193, 0.5 240, 1.0 242, a literal dword
/// 255, VGPRs 256-511.
const fn v(n: u32) -> u32 {
    256 + n
}
const ZERO: u32 = 128;
const ONE: u32 = 129;
const MINUS_ONE: u32 = 193;
const HALF: u32 = 240;
const F1: u32 = 242;
const LITERAL: u32 = 255;
/// SMEM's SOFFSET "none".
const NULL: u32 = 125;

#[derive(Default)]
struct Asm(Vec<u32>);

impl Asm {
    /// SOP1: `s_mov_b32` is 3.
    fn sop1(&mut self, op: u32, sdst: u32, ssrc: u32) {
        self.0.push(0xBE80_0000 | sdst << 16 | op << 8 | ssrc);
    }
    fn s_mov(&mut self, sdst: u32, ssrc: u32) {
        self.sop1(3, sdst, ssrc);
    }
    /// SOP2: `s_lshl_b32` is 0x1e.
    fn sop2(&mut self, op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) {
        self.0
            .push(0x8000_0000 | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0);
    }
    /// SOPK `s_movk_i32` (0): a sign-extended 16-bit constant.
    fn s_movk(&mut self, sdst: u32, imm: u16) {
        self.0.push(0xB000_0000 | sdst << 16 | u32::from(imm));
    }
    /// SOPP: `s_endpgm` 1, `s_waitcnt` 0xc.
    fn sopp(&mut self, op: u32, imm: u16) {
        self.0.push(0xBF80_0000 | op << 16 | u32::from(imm));
    }
    /// `s_waitcnt lgkmcnt(0)` (VM_CNT and EXP_CNT left at their maxima).
    fn wait_scalar(&mut self) {
        self.sopp(0xc, 0xC07F);
    }
    /// `s_waitcnt vmcnt(0)`.
    fn wait_vector(&mut self) {
        self.sopp(0xc, 0x3F70);
    }
    fn end(&mut self) {
        self.sopp(1, 0);
    }
    /// SMEM: `s_load_dwordx4` 2, `_x8` 3, `s_buffer_load_dwordx16` 0xc;
    /// `offset` in bytes.
    fn smem(&mut self, op: u32, sdata: u32, sbase: u32, offset: u32) {
        self.0
            .push(0xF400_0000 | op << 18 | sdata << 6 | sbase >> 1);
        self.0.push(NULL << 25 | offset);
    }
    /// VOP1: `v_mov_b32` 1, `v_exp_f32` 0x25, `v_log_f32` 0x27,
    /// `v_rsq_f32` 0x2e.
    fn vop1(&mut self, op: u32, vdst: u32, src0: u32) {
        self.0
            .push(0x7E00_0000 | (vdst & 0xFF) << 17 | op << 9 | src0);
    }
    /// Destinations and `vsrc1` are VGPRs, by number or operand code.
    /// VOP2: `v_add_f32` 3, `v_sub_f32` 4,
    /// `v_mul_f32` 8, `v_max_f32` 0x10, `v_lshlrev_b32` 0x1a,
    /// `v_add_nc_u32` 0x25, `v_fmac_f32` 0x2b (vdst += src0 * vsrc1).
    fn vop2(&mut self, op: u32, vdst: u32, src0: u32, vsrc1: u32) {
        self.0
            .push(op << 25 | (vdst & 0xFF) << 17 | (vsrc1 & 0xFF) << 9 | src0);
    }
    fn add(&mut self, d: u32, a: u32, b: u32) {
        self.vop2(3, d, a, b);
    }
    fn sub(&mut self, d: u32, a: u32, b: u32) {
        self.vop2(4, d, a, b);
    }
    fn mul(&mut self, d: u32, a: u32, b: u32) {
        self.vop2(8, d, a, b);
    }
    fn fmac(&mut self, d: u32, a: u32, b: u32) {
        self.vop2(0x2b, d, a, b);
    }
    /// VOP3 (only-VOP3 opcodes): `v_mad_u32_u24` 0x143.
    fn vop3(&mut self, op: u32, vdst: u32, src0: u32, src1: u32, src2: u32) {
        self.0.push(0xD400_0000 | op << 16 | (vdst & 0xFF));
        self.0.push(src2 << 18 | src1 << 9 | src0);
    }
    /// VINTRP `v_interp_p1_f32` (0) then `v_interp_p2_f32` (1) of one
    /// attribute channel, from the barycentrics i (v`i`) and j (v`i` + 1).
    fn interpolate(&mut self, vdst: u32, i: u32, attr: u32, chan: u32) {
        for (op, bary) in [(0, i), (1, i + 1)] {
            self.0
                .push(0xC800_0000 | vdst << 18 | op << 16 | attr << 10 | chan << 8 | bary);
        }
    }
    /// EXP: target `mrt0` 0, `pos0` 12, `param<n>` 32 + n.
    fn export(&mut self, target: u32, enable: u32, done: bool, valid: bool, src: [u32; 4]) {
        self.0.push(
            0xF800_0000 | u32::from(valid) << 12 | u32::from(done) << 11 | target << 4 | enable,
        );
        self.0
            .push(src[3] << 24 | src[2] << 16 | src[1] << 8 | src[0]);
    }
    /// MIMG, a 2D image, all four channels: `image_load` 0 (unnormalised),
    /// `image_store` 8 (unnormalised), `image_sample` 0x20.
    fn image(&mut self, op: u32, vdata: u32, vaddr: u32, srsrc: u32, ssamp: u32) {
        let unnormalised = u32::from(op != 0x20);
        self.0
            .push(0xF000_0000 | op << 18 | unnormalised << 12 | 0xF << 8 | 1 << 3);
        self.0
            .push(ssamp >> 2 << 21 | srsrc >> 2 << 16 | vdata << 8 | vaddr);
    }
    /// MUBUF `buffer_store_dwordx4` (0x1e) at the byte offset in `vaddr`.
    fn store4(&mut self, vdata: u32, vaddr: u32, srsrc: u32) {
        self.0.push(0xE000_0000 | 0x1e << 18 | 1 << 12);
        self.0
            .push(ZERO << 24 | srsrc >> 2 << 16 | vdata << 8 | vaddr);
    }
    /// A 64-bit address in SGPRs `pair`, `pair + 1` of the descriptor table
    /// whose 32-bit address is in `table` (tables live in the low 4 GiB of
    /// the heap's space: the high half is 0).
    fn table(&mut self, pair: u32, table: u32) {
        self.s_mov(pair, table);
        self.s_mov(pair + 1, ZERO);
    }
}

/// The vertex shader (wave32). User data: s0 the scene constants' table,
/// s1 the draw's. Inputs (the fetch shader's VGPRs): position v4-v6,
/// normal v7-v9, texture coordinates v10-v11. Outputs: the world position
/// (param0), the world normal (param1), the coordinates (param2), the clip
/// position (pos0).
fn vertex() -> Vec<u32> {
    let mut a = Asm::default();
    a.table(2, 0);
    a.smem(2, 8, 2, 0); // s_load_dwordx4 s[8:11], s[2:3], 0
    a.table(4, 1);
    a.smem(2, 12, 4, 0); // s_load_dwordx4 s[12:15], s[4:5], 0
    a.wait_scalar();
    a.smem(0xc, 16, 8, 0); // s_buffer_load_dwordx16 s[16:31]: view-projection rows
    a.smem(0xc, 32, 12, 0); // s_buffer_load_dwordx16 s[32:47]: world rows
    a.wait_scalar();
    // World position v12-v14: row r . (position, 1).
    for r in 0..3 {
        let (d, row) = (v(12 + r), 32 + 4 * r);
        a.mul(d, row, 4);
        a.fmac(d, row + 1, 5);
        a.fmac(d, row + 2, 6);
        a.add(d, row + 3, 12 + r);
    }
    // World normal v15-v17: the rows' first three columns . normal.
    for r in 0..3 {
        let (d, row) = (v(15 + r), 32 + 4 * r);
        a.mul(d, row, 7);
        a.fmac(d, row + 1, 8);
        a.fmac(d, row + 2, 9);
    }
    // Clip position v18-v21: view-projection row r . (world, 1).
    for r in 0..4 {
        let (d, row) = (v(18 + r), 16 + 4 * r);
        a.mul(d, row, 12);
        a.fmac(d, row + 1, 13);
        a.fmac(d, row + 2, 14);
        a.add(d, row + 3, 18 + r);
    }
    a.export(32, 0x7, false, false, [12, 13, 14, 14]);
    a.export(33, 0x7, false, false, [15, 16, 17, 17]);
    a.export(34, 0x3, false, false, [10, 11, 11, 11]);
    a.export(12, 0xF, true, false, [18, 19, 20, 21]);
    a.end();
    a.0
}

/// The pixel shader (wave32). User data: s0 the scene constants' table, s1
/// the draw's, s2 the texture's, s3 the sampler's, then (s4) the
/// primitive's attribute location. The perspective-centre
/// barycentrics arrive in v0-v1. Colour = texture x (ambient + n.l x
/// light) x albedo + specular x (n.h)^shininess x light + emitted.
fn pixel() -> Vec<u32> {
    let mut a = Asm::default();
    // The primitive's attribute location arrives after the user data;
    // interpolation reads it from M0.
    a.s_mov(124, 4);
    a.table(6, 0);
    a.smem(2, 8, 6, 0); // s_load_dwordx4 s[8:11]: scene constants
    a.table(60, 1);
    a.smem(2, 12, 60, 0); // s_load_dwordx4 s[12:15]: draw constants
    a.table(62, 2);
    a.smem(3, 48, 62, 0); // s_load_dwordx8 s[48:55]: texture
    a.table(64, 3);
    a.smem(2, 56, 64, 0); // s_load_dwordx4 s[56:59]: sampler
    // World position v2-v4, normal v5-v7, coordinates v8-v9.
    for c in 0..3 {
        a.interpolate(2 + c, 0, 0, c);
    }
    for c in 0..3 {
        a.interpolate(5 + c, 0, 1, c);
    }
    for c in 0..2 {
        a.interpolate(8 + c, 0, 2, c);
    }
    a.wait_scalar();
    a.image(0x20, 10, 8, 48, 56); // image_sample v[10:13], v[8:9]
    // Light s16-s18, eye s20-s22, light colour s24-s26, ambient s28-s30;
    // albedo s32-s34, shininess s35, specular s36-s38, emitted s40-s42.
    a.smem(0xc, 16, 8, 0x40);
    a.smem(0xc, 32, 12, 0x40);
    // Normalise (x, y, z) in VGPRs `x`..`x + 2` with `t` for the length.
    let normalise = |a: &mut Asm, x: u32, t: u32| {
        a.mul(v(t), v(x), x);
        a.fmac(v(t), v(x + 1), x + 1);
        a.fmac(v(t), v(x + 2), x + 2);
        a.vop1(0x2e, v(t), v(t)); // v_rsq_f32
        for c in 0..3 {
            a.mul(v(x + c), v(x + c), t);
        }
    };
    normalise(&mut a, 5, 14);
    a.wait_scalar();
    // View direction v15-v17 = eye - position, then the half vector.
    for c in 0..3 {
        a.sub(v(15 + c), 20 + c, 2 + c);
    }
    normalise(&mut a, 15, 18);
    for c in 0..3 {
        a.add(v(15 + c), 16 + c, 15 + c);
    }
    normalise(&mut a, 15, 18);
    // n.l in v19, n.h in v20, both at least 0.
    a.mul(v(19), 16, 5);
    a.fmac(v(19), 17, 6);
    a.fmac(v(19), 18, 7);
    a.vop2(0x10, v(19), ZERO, 19);
    a.mul(v(20), v(15), 5);
    a.fmac(v(20), v(16), 6);
    a.fmac(v(20), v(17), 7);
    a.vop2(0x10, v(20), ZERO, 20);
    // (n.h)^shininess as 2^(shininess x log2(n.h)).
    a.vop1(0x27, v(20), v(20));
    a.mul(v(20), 35, 20);
    a.vop1(0x25, v(20), v(20));
    a.wait_vector();
    for c in 0..3 {
        let d = 21 + c;
        a.vop1(1, v(d), 28 + c); // ambient
        a.fmac(v(d), 24 + c, 19); // + n.l x light
        a.mul(v(d), v(10 + c), d); // x texture
        a.mul(v(d), 32 + c, d); // x albedo
        a.mul(v(24), 36 + c, 20); // specular x (n.h)^shininess
        a.mul(v(24), 24 + c, 24); // x light
        a.add(v(d), v(d), 24);
        a.add(v(d), 40 + c, d); // + emitted
    }
    a.vop1(1, v(24), F1);
    a.export(0, 0xF, true, true, [21, 22, 23, 24]);
    a.end();
    a.0
}

/// The compute shader (wave32, 8 x 8 threads): a horizontal [1 2 1] / 4
/// blur of the frame. User data: s0 the table of the frame (SRV, at 0), the
/// texture shown (UAV, at 32 bytes) and the read-back buffer (raw UAV, at
/// 64 bytes; four floats per pixel, row after row). The group id arrives
/// in s1-s2, the thread id in v0-v1. A neighbour past the edge reads 0.
fn compute() -> Vec<u32> {
    let mut a = Asm::default();
    a.table(4, 0);
    a.smem(3, 8, 4, 0); // s_load_dwordx8 s[8:15]: the frame
    a.smem(3, 16, 4, 0x20); // s_load_dwordx8 s[16:23]: the texture shown
    a.smem(2, 24, 4, 0x40); // s_load_dwordx4 s[24:27]: the buffer
    a.sop2(0x1e, 28, 1, 3 + 128); // s_lshl_b32 s28, s1, 3
    a.sop2(0x1e, 29, 2, 3 + 128); // s_lshl_b32 s29, s2, 3
    a.vop2(0x25, v(2), 28, 0); // x = group x * 8 + thread x
    a.vop2(0x25, v(3), 29, 1);
    a.vop2(0x25, v(4), MINUS_ONE, 2); // left
    a.vop1(1, v(5), v(3));
    a.vop2(0x25, v(6), ONE, 2); // right
    a.vop1(1, v(7), v(3));
    a.wait_scalar();
    a.image(0, 8, 2, 8, 0); // image_load v[8:11], v[2:3]: centre
    a.image(0, 12, 4, 8, 0); // left
    a.image(0, 16, 6, 8, 0); // right
    a.wait_vector();
    for c in 0..4 {
        let d = 20 + c;
        a.add(v(d), v(12 + c), 16 + c);
        a.mul(v(d), LITERAL, d);
        a.0.push(0.25f32.to_bits());
        a.fmac(v(d), HALF, 8 + c);
    }
    a.image(8, 20, 2, 16, 0); // image_store v[20:23], v[2:3]
    a.s_movk(30, super::SIZE as u16);
    a.vop3(0x143, 24, v(3), 30, v(2)); // v_mad_u32_u24 v24, y, width, x
    a.vop2(0x1a, v(24), 4 + 128, 24); // v_lshlrev_b32 v24, 4, v24
    a.store4(20, 24, 24);
    a.end();
    a.0
}

/// One element of an `ISG1`/`OSG1` part: name, system value (0 none, 1
/// position, 64 target), register and component mask.
struct Element(&'static str, u32, u32, u8);

/// The signature part: elements of 32 bytes (stream, name offset, semantic
/// index, system value, component type 3 = float, register, mask, mask
/// read or written, minimum precision 0), then the names.
fn signature(elements: &[Element], input: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut names = Vec::new();
    let table = 8 + 32 * elements.len();
    out.extend((elements.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    // The semantic index follows the name's last digit (TEXCOORD1 is
    // TEXCOORD, index 1).
    for e in elements {
        let digits = e.0.trim_start_matches(|c: char| !c.is_ascii_digit());
        let base = &e.0[..e.0.len() - digits.len()];
        let index: u32 = digits.parse().unwrap_or(0);
        let name_at = (table + names.len()) as u32;
        names.extend(base.as_bytes());
        names.push(0);
        let used = if input { e.3 } else { 0 };
        for w in [0, name_at, index, e.1, 3, e.2] {
            out.extend(w.to_le_bytes());
        }
        out.extend([e.3, used, 0, 0]);
        out.extend(0u32.to_le_bytes());
    }
    out.extend(names);
    out.resize(out.len().next_multiple_of(4), 0);
    out
}

/// What a program's record says besides its code.
struct Program {
    stage: usize,
    code: Vec<u32>,
    vgprs: u32,
    sgprs: u32,
    /// Root parameters: (slot, user-data dword), each a descriptor table.
    layout: Vec<(u8, u8)>,
    /// Descriptors read: (type: 0 SRV, 1 sampler, 2 CBV, 3 UAV; slot,
    /// user-data dword, offset in the table in dwords, register, dwords).
    used: Vec<(u8, u8, u8, u32, u32, u8)>,
    stage_data: Vec<u8>,
}

/// Stage slots in the blob's table of record offsets.
const VS: usize = 0x28;
const PS: usize = 0x30;
const CS: usize = 0x60;
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;

fn put(data: &mut [u8], at: usize, bytes: &[u8]) {
    data[at..at + bytes.len()].copy_from_slice(bytes);
}

/// The `FSSM` blob of one program.
fn fssm(p: &Program) -> Vec<u8> {
    const RECORD: usize = 0x68;
    let mut payload = vec![0u8; RECORD + 0x54];
    put(&mut payload, 0, &6u64.to_le_bytes());
    for at in (8..=CS).step_by(8) {
        put(&mut payload, at, &ABSENT.to_le_bytes());
    }
    put(&mut payload, p.stage, &(RECORD as u64).to_le_bytes());
    payload.extend(&p.stage_data);
    let layout_at = payload.len();
    for &(slot, user_data) in &p.layout {
        // Root type 3 (a table), range at 0, one SGPR, register 0.
        for w in [3u32, 0, 1, 0] {
            payload.extend(w.to_le_bytes());
        }
        payload.extend([user_data, slot, 0, 0]);
    }
    let used_at = payload.len();
    for &(kind, slot, user_data, offset, register, dwords) in &p.used {
        payload.extend([kind, 3, 0, 0]);
        for w in [offset, register, 0] {
            payload.extend(w.to_le_bytes());
        }
        payload.extend([dwords, user_data, slot, 0]);
    }
    payload.resize(payload.len().next_multiple_of(256), 0);
    let code_at = payload.len();
    payload.extend(p.code.iter().flat_map(|w| w.to_le_bytes()));
    let record = &mut payload[RECORD..RECORD + 0x54];
    put(record, 0, &(code_at as u64).to_le_bytes());
    put(record, 8, &(4 * p.code.len() as u64).to_le_bytes());
    put(record, 0x14, &(1u32 << 5).to_le_bytes()); // wave32
    put(record, 0x28, &(layout_at as u64).to_le_bytes());
    put(record, 0x30, &(used_at as u64).to_le_bytes());
    put(record, 0x38, &(p.layout.len() as u32).to_le_bytes());
    put(record, 0x3c, &(p.used.len() as u32).to_le_bytes());
    put(record, 0x50, &(p.vgprs | p.sgprs << 20).to_le_bytes());
    let mut out = b"FSSM".to_vec();
    out.extend(0x101u32.to_le_bytes());
    out.extend((payload.len() as u64).to_le_bytes());
    out.extend(payload);
    out
}

/// The container: `ISG1`, `OSG1` (if any), and the `DXIL` part (a program
/// header, the `XDAT` index, the packed programs).
fn container(kind: u32, p: &Program, signatures: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let unpacked = fssm(p);
    let stream = miniz_oxide::deflate::compress_to_vec_zlib(&unpacked, 6);
    let mut blob = b"XBCH".to_vec();
    blob.extend(3u32.to_le_bytes());
    blob.extend((stream.len() as u64).to_le_bytes());
    blob.extend((unpacked.len() as u64).to_le_bytes());
    blob.extend(stream);
    // Program header: kind and shader model 6.5, size in dwords, "DXIL",
    // version 1.5, then (no bitcode) an offset and size of none.
    let index = 0x18 + 12 + 12;
    let mut dxil = Vec::new();
    let total = (index + blob.len()).next_multiple_of(4);
    for w in [kind << 16 | 0x65, (total / 4) as u32] {
        dxil.extend(w.to_le_bytes());
    }
    dxil.extend(b"DXIL");
    for w in [0x105u32, (total - 8) as u32, 0] {
        dxil.extend(w.to_le_bytes());
    }
    dxil.extend(b"XDAT");
    for w in [1u32, 0x24, 0x0d, index as u32, blob.len() as u32] {
        dxil.extend(w.to_le_bytes());
    }
    dxil.extend(blob);
    dxil.resize(total, 0);
    let mut parts: Vec<(&[u8; 4], &[u8])> = signatures.iter().map(|(f, d)| (*f, &d[..])).collect();
    parts.push((b"DXIL", &dxil));
    let table = 0x20 + 4 * parts.len();
    let size = table + parts.iter().map(|p| 8 + p.1.len()).sum::<usize>();
    let mut out = b"DXBC".to_vec();
    out.extend([0; 16]);
    out.extend(1u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((size as u32).to_le_bytes());
    out.extend((parts.len() as u32).to_le_bytes());
    let mut at = table;
    for p in &parts {
        out.extend((at as u32).to_le_bytes());
        at += 8 + p.1.len();
    }
    for (fourcc, data) in parts {
        out.extend(fourcc);
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
    }
    out
}

/// The vertex stage's signatures: what the input layout feeds it and what
/// it hands the pixel shader.
fn vertex_signatures() -> (Vec<u8>, Vec<u8>) {
    let inputs = [
        Element("POSITION", 0, 0, 0x7),
        Element("NORMAL", 0, 1, 0x7),
        Element("TEXCOORD", 0, 2, 0x3),
    ];
    (signature(&inputs, true), signature(&interpolants(), false))
}

fn interpolants() -> [Element; 4] {
    [
        Element("SV_Position", 1, 0, 0xF),
        Element("TEXCOORD0", 0, 1, 0x7),
        Element("TEXCOORD1", 0, 2, 0x7),
        Element("TEXCOORD2", 0, 3, 0x3),
    ]
}

pub(super) fn vertex_container() -> Vec<u8> {
    let mut data = vec![0u8; 0x190];
    // Inputs: usage (6 generic), row, mask, first VGPR, highest channel.
    let inputs: [[u8; 5]; 3] = [[6, 0, 7, 4, 2], [6, 1, 7, 7, 2], [6, 2, 3, 10, 1]];
    for (i, e) in inputs.iter().enumerate() {
        put(&mut data, 0x24 + 5 * i, e);
    }
    put(&mut data, 0xc4, &3u32.to_le_bytes());
    // Outputs: usage, row, `param` number, row, slot, flags.
    for n in 0..3u8 {
        put(
            &mut data,
            0xc8 + 6 * n as usize,
            &[6, n + 1, n, n + 1, n, 0],
        );
    }
    put(&mut data, 0x188, &3u32.to_le_bytes());
    let p = Program {
        stage: VS,
        code: vertex(),
        vgprs: 24,
        sgprs: 48,
        layout: vec![(0, 0), (1, 1)],
        used: vec![(2, 0, 0, 0, 0, 4), (2, 1, 1, 0, 1, 4)],
        stage_data: data,
    };
    let (isg, osg) = vertex_signatures();
    container(1, &p, &[(b"ISG1", isg), (b"OSG1", osg)])
}

pub(super) fn pixel_container() -> Vec<u8> {
    let mut data = vec![0u8; 0x14c + 8 * (4 + 400)];
    // Inputs: usage, row, mask, attribute, default, second row, slot, flags.
    for n in 0..3u8 {
        let mask = if n == 2 { 3 } else { 7 };
        put(
            &mut data,
            0x24 + 8 * n as usize,
            &[6, n + 1, mask, n, 0, n + 1, n, 0],
        );
    }
    put(&mut data, 0x124, &3u32.to_le_bytes());
    put(&mut data, 0x134, &2u32.to_le_bytes()); // SPI_PS_INPUT_ADDR: centre i/j
    // No export patches: the shader's exports carry 32-bit values.
    for mrt in 0..8 {
        put(&mut data, 0x14c + mrt * 404, &u32::MAX.to_le_bytes());
    }
    let p = Program {
        stage: PS,
        code: pixel(),
        vgprs: 32,
        sgprs: 66,
        layout: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
        used: vec![
            (2, 0, 0, 0, 0, 4),
            (2, 1, 1, 0, 1, 4),
            (0, 2, 2, 0, 0, 8),
            (1, 3, 3, 0, 0, 4),
        ],
        stage_data: data,
    };
    let isg = signature(&interpolants(), true);
    let osg = signature(&[Element("SV_Target", 64, 0, 0xF)], false);
    container(0, &p, &[(b"ISG1", isg), (b"OSG1", osg)])
}

pub(super) fn compute_container() -> Vec<u8> {
    let mut data = vec![0u8; 0x40];
    // COMPUTE_PGM_RSRC2: one user SGPR, group ids x and y, thread ids x
    // and y.
    let rsrc2 = 1 << 1 | 1 << 7 | 1 << 8 | 1 << 11;
    put(&mut data, 0x24, &(rsrc2 as u32).to_le_bytes());
    for (i, n) in [super::GROUP, super::GROUP, 1].iter().enumerate() {
        put(&mut data, 0x28 + 4 * i, &n.to_le_bytes());
    }
    let p = Program {
        stage: CS,
        code: compute(),
        vgprs: 32,
        sgprs: 32,
        layout: vec![(0, 0)],
        used: vec![(0, 0, 0, 0, 0, 8), (3, 0, 0, 8, 0, 8), (3, 0, 0, 16, 1, 4)],
        stage_data: data,
    };
    container(5, &p, &[])
}
