//! What the estimated stand-in for ray-traced indirect light
//! (`crate::ambient_estimate`, `VICTORY_AMBIENT=estimate`) is told by its
//! switches, which of the title's programs it follows, its passes' shader
//! and constants, and the same arithmetic on the CPU to test against.
//! Plain std, so its tests run everywhere.

// Used by `crate::ambient_estimate`, which is only built for Windows.
#![cfg_attr(not(windows), allow(dead_code))]

/// The share of the estimate `VICTORY_AMBIENT`'s value asks for, if it
/// asks for the estimate: `estimate` (all of it) or `estimate:<share>`. A
/// share that is not a number above 0 counts as 1. Any other value is the
/// constant stand-in's (`crate::ambient_set::is_on`), which this leaves as
/// it is.
pub(crate) fn share(value: Option<&str>) -> Option<f32> {
    let value = value?.trim().to_ascii_lowercase();
    let rest = value.strip_prefix("estimate")?;
    if rest.is_empty() {
        return Some(1.0);
    }
    let share = rest.strip_prefix(':')?.trim().parse().ok();
    Some(share.filter(|s: &f32| s.is_finite() && *s > 0.0).unwrap_or(1.0))
}

/// `VICTORY_AMBIENT_LEFT=<multiple>`: how much of what the title's resolve
/// left in the light image counts, as a multiple of the estimate's
/// luminance there; the written light is the larger of the two. 0 (the
/// default here): nothing of it, the estimate replaces the image, as the
/// constant stand-in does.
///
/// The fork has 4, fixed: there the resolve leaves the sky's light
/// outdoors. On this line it leaves next to nothing where the sun does not
/// reach (the cell without a stand-in is dark, `notes/PLAN.md`, 26 November
/// 17:25) and bright specks on foliage (the boat dump's frame 1766).
pub(crate) fn left(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|m: &f32| m.is_finite() && *m >= 0.0)
        .unwrap_or(0.0)
}

/// How a texel's light is put together from the points' and the slow mean
/// (numbers fixed in the fork): the part of it that is the slow mean of the
/// whole picture; how strongly the points' directions turn it by the
/// texel's normal; and the least and the most it may be, as multiples of
/// the points' mean there.
pub(crate) const MIX: [f32; 4] = [0.2, 1.2, 0.2, 2.5];

/// `VICTORY_AMBIENT_MIX=<mean part>,<turn>,<least>,<most>`: [`MIX`] with
/// the values given in their places (one left out or unreadable keeps its
/// default; the mean part is held to 0..1, the others to 0 and above, the
/// most to no less than the least).
///
/// For the first looks on this line. Outdoors the fork has the sky's light
/// from the title's own resolve under the estimate; here the estimate is
/// alone, and it sees the sky behind whatever faces the camera (the sky's
/// cells lie far away along the view), so such a surface gets the least
/// multiple of what is around it plus the mean part. A larger mean part or
/// a smaller turn gives it more of the sky.
pub(crate) fn mix(value: Option<&str>) -> [f32; 4] {
    let mut mix = MIX;
    for (to, v) in mix.iter_mut().zip(value.unwrap_or_default().split(',')) {
        if let Some(v) = v.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v >= 0.0) {
            *to = v;
        }
    }
    mix[0] = mix[0].min(1.0);
    mix[3] = mix[3].max(mix[2]);
    mix
}

/// The parts of the estimate the probe leaves out in turn
/// (`VICTORY_AMBIENT_PROBE`), by step: bits of what is not recorded, and
/// what the frame line calls the step.
pub(crate) const NO_GATHER: u8 = 1;
pub(crate) const NO_SHADE: u8 = 2;
pub(crate) const NO_HOLD: u8 = 4;
pub(crate) const NO_KEEP: u8 = 8;
pub(crate) const NO_RAISE: u8 = 16;
pub(crate) const PROBE: [(u8, &str); 7] = [
    (0, "nothing"),
    (NO_GATHER, "the gather (the cells' places and the points' light: the points stay as they were)"),
    (NO_SHADE, "the creases' shade"),
    (NO_HOLD, "the reflections' hold"),
    (NO_KEEP, "the keeping (the cells and the slow mean stay as they were)"),
    (NO_RAISE, "the raising with its gather (the title's own light image stays)"),
    (NO_GATHER | NO_HOLD | NO_KEEP | NO_RAISE, "every pass"),
];

/// `VICTORY_AMBIENT_PROBE=<n>`: how many frame lines each step of the probe
/// lasts (a whole number above 0; unset, 0 or unreadable: no probe).
///
/// To find which of the estimate's passes costs GPU time in a scene, in one
/// run of a normal build: the passes are left out in turn ([`PROBE`]), each
/// step for `n` frame lines, and the estimate's own line says which step
/// the frame line above it was measured with. The picture changes with the
/// steps: a diagnostic, not a way to play.
pub(crate) fn probe(value: Option<&str>) -> Option<u32> {
    value.and_then(|v| v.trim().parse().ok()).filter(|n| *n > 0)
}

/// The probe's step after `lines` frame lines, each step lasting `each`.
pub(crate) fn probe_step(lines: u64, each: u32) -> usize {
    (lines / u64::from(each.max(1)) % PROBE.len() as u64) as usize
}

/// What `VICTORY_AMBIENT_FLOOR` is when unset.
pub(crate) const FLOOR: f32 = 0.3;

/// `VICTORY_AMBIENT_FLOOR=<share>`: the least any texel gets, as a share of
/// the constant stand-in's light (`crate::ambient`: by default the light
/// that follows the sky dome, `gain * mean^power + least`). The share cuts
/// the sky's part; the formula's least level, the night's, is kept whole.
/// 0: no floor, the estimate alone, as in the fork. 1: never darker than
/// the constant stand-in.
///
/// Where little light is on screen the estimate has nothing to spread: a
/// room lit only by its own bounce (the windows off screen, no lamp, no sun
/// patch) settles at black, since each frame's light is the walls' colour
/// times the last frame's, and a night street stays black where the
/// constant shows the asphalt (the opening, run 21:38:20 of 26 November:
/// 0.0002 to 0.03 written where the constant's least is 0.05). The fork has
/// the title's own resolve under the estimate there; on this line that
/// leaves nothing. The floor is the light from what is not on screen.
pub(crate) fn floor(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|s: &f32| s.is_finite() && *s >= 0.0)
        .unwrap_or(FLOOR)
}

/// `VICTORY_AMBIENT_REFLECT_FLOOR=<part>`: the least a reflection holds, as
/// a part of the light around the point seen there: the same light the
/// hold's cap is a part of (the `HOLD` pass of [`SOURCE`]: the points' light
/// there with the slow mean's part, times the estimate's share, or the
/// floor's light where that is more). Unset, 0 or unreadable: none, the
/// hold only caps. A reflection darker than that is replaced by it, in the
/// colour of that light, as the constant stand-in's least (`crate::ambient`,
/// the file's tenth number) does; one brighter is left as it is. It rides
/// on the hold: nothing with `VICTORY_AMBIENT_REFLECT=0`.
///
/// Without rays the title's reflection image is 0 almost everywhere indoors
/// (the cell, capture 01:50 in `notes/black-objects.md`: 93% of the frame).
/// Metal gets no diffuse light and all of its light through that image, so
/// steel there is black; the lighting weighs the reflection by each
/// surface's specular colour, so this lights steel (F0 about 0.55) about
/// fourteen times as much as a painted wall (F0 0.04).
/// shortcut: stands in for what the title's traced reflections would see.
pub(crate) fn reflect_floor(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|p: &f32| p.is_finite() && *p >= 0.0)
        .unwrap_or(0.0)
}

/// What the floor is made of, as the passes are given it
/// ([`Pass::sky`], [`Pass::floor`]): `share` of the constant stand-in's
/// light, which is `level` times `tint`, or with `follow` (gain, least,
/// power) and the sky dome drawn (`dome`) follows the dome's mean.
pub(crate) fn floor_of(
    share: f32,
    level: f32,
    tint: [f32; 3],
    follow: Option<[f32; 3]>,
    dome: bool,
) -> ([f32; 3], [f32; 3]) {
    match follow.filter(|_| dome && share > 0.0) {
        // The sky's part cut to the share, the least level whole.
        Some([gain, least, power]) => ([gain * share, least, power], [0.0; 3]),
        // The constant light (also until the dome is drawn).
        None => ([0.0, 0.0, 1.0], tint.map(|t| t * level * share)),
    }
}

/// The program that keeps the light visible surfaces send out, for the
/// title's own screen traces, by the start of its name, and where it writes
/// it (R11G11B10_FLOAT, half the picture's size): root parameter and place
/// in its table, in dwords. One program of that name in the title's shader
/// database (`e6d8226a6870f577`), with that place.
pub(crate) const RADIANCE_PROGRAM: &[u8] = b"CS_CacheScreenSpaceRadiance";
pub(crate) const RADIANCE: (u8, u32) = (1, 16);

/// The program that adds the lamps' light to what the air scatters (the
/// last that writes that volume before it is summed up along the view), by
/// the start of its name, and where the volume is (R11G11B10_FLOAT, three
/// dimensions; the program reads and writes it through one view). One
/// program of that name in the title's shader database
/// (`b7de1a805fbbac3c`), with that place.
pub(crate) const AIR_PROGRAM: &[u8] = b"CS_AccumulateLocalLightInscatteringVolume";
pub(crate) const AIR: (u8, u32) = (1, 32);

/// What a program is to the estimate, beside `crate::ambient_set`'s bits
/// (`RESOLVES` and its siblings).
pub(crate) const KEEPS: u8 = 8;
pub(crate) const LIGHTS_AIR: u8 = 16;
pub(crate) const BLURS_REFLECTIONS: u8 = 32;

/// The program that writes the reflections the lighting reads, by the
/// start of its name, and where (the same place as `CS_UpdateHistory`'s,
/// `crate::ambient_set::REFLECTIONS`: the same table layout, the same
/// register u31). In the denoiser `sr_denoising` the temporal resolve, the
/// spatial filter, this blur and `CS_UpdateHistory` run in that order, then
/// the lighting. The blur reads the resolved reflections (t389) and others
/// and writes u31 and u36; `CS_UpdateHistory` reads the same t389 and writes
/// only u31 of its own table, the history the next frame's temporal resolve
/// reads (both read in their code; `notes/black-objects.md`, "Reflection
/// floor (prep42): why it did nothing"). A hold after `CS_UpdateHistory`
/// alone changes the history, not what the lighting reads.
pub(crate) const BLUR_PROGRAM: &[u8] = b"CS_BilateralBlur_e877ef6e";

/// `VICTORY_AMBIENT_AIR=<multiple>`: the most the air scatters, as a
/// multiple of the light around; unset or 0: what the title has.
///
/// The air of a room is lit by the same probes: without rays they hold the
/// sky's light indoors too, a brown veil at sunset.
pub(crate) fn air(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|m: &f32| m.is_finite() && *m > 0.0)
        .unwrap_or(0.0)
}

/// What the program of shader container `container` is to the estimate
/// alone (the resolve is `crate::ambient_set`'s).
pub(crate) fn role_of(container: &[u8]) -> u8 {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    let mut role = 0;
    if has(RADIANCE_PROGRAM) {
        role |= KEEPS;
    }
    if has(AIR_PROGRAM) {
        role |= LIGHTS_AIR;
    }
    if has(BLUR_PROGRAM) {
        role |= BLURS_REFLECTIONS;
    }
    role
}

/// What the estimate calls a role in the log.
pub(crate) fn role_name(role: u8) -> &'static str {
    match role {
        KEEPS => "keeps the light visible surfaces send out",
        LIGHTS_AIR => "adds the lamps' light to what the air scatters",
        BLURS_REFLECTIONS => "writes the reflections the lighting reads (held with VICTORY_AMBIENT_REFLECT_FLOOR)",
        _ => "several",
    }
}

/// Cells the picture's light is averaged in, and points the light around is
/// worked out at.
pub(crate) const CX: u32 = 64;
pub(crate) const CY: u32 = 36;
pub(crate) const CELLS: u32 = CX * CY;
pub(crate) const PX: u32 = 160;
pub(crate) const PY: u32 = 90;

/// Bytes of the cells' buffer (their light, the slow mean, their places,
/// the floor's light)
/// and of the points' (three `float4` a point).
pub(crate) const CELL_BYTES: u64 = (2 * CELLS as u64 + 2) * 16;
/// Where the floor's light is among the cells' `float4`s (after the cells'
/// light, the slow mean and the cells' places).
pub(crate) const FLOOR_AT: u32 = 2 * CELLS + 1;
pub(crate) const POINT_BYTES: u64 = PX as u64 * PY as u64 * 48;

/// The share of the way the slow mean goes at every keeping, while it does
/// not settle by the clock: 25 frames, 0.8 s at the console's 30 frames a
/// second, 8 s at 3.
pub(crate) const FOLLOW: f32 = 0.04;

/// How the slow mean of the light around settles by the clock: the time
/// constant in milliseconds; the factor of a change (either way) that is
/// taken at once, as a new room is (0: none); and for how many seconds
/// after such a jump what the title's resolve left counts for no more than
/// the estimate, rising back to `VICTORY_AMBIENT_LEFT` times it by the end
/// (0: no such time; nothing while that multiple is 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Settle {
    pub(crate) ms: f32,
    pub(crate) jump: f32,
    pub(crate) fresh: f32,
}

/// What `VICTORY_AMBIENT_SETTLE` is when unset: by the clock, as in the
/// fork from 9576c45f on (rooms stopped being too bright or too dark for
/// seconds after walking in). At this line's frame rates the slow mean
/// then goes most of the way in one frame.
const SETTLE_UNSET: &str = "250,3";

/// `VICTORY_AMBIENT_SETTLE=<time constant in ms>[,<jump factor>[,<fresh
/// seconds>]]` (jump 3 and fresh 0 when left out); unset: 250,3. 0: `None`,
/// the slow mean goes [`FOLLOW`] of the way a frame whatever the frame
/// rate.
pub(crate) fn settle(value: Option<&str>) -> Option<Settle> {
    let mut parts = value.unwrap_or(SETTLE_UNSET).split(',').map(|p| p.trim().parse::<f32>().ok());
    let ms = parts.next().flatten().filter(|v| v.is_finite() && *v > 0.0)?;
    let jump = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(3.0);
    let fresh = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.0);
    Some(Settle { ms, jump, fresh })
}

/// The milliseconds a keeping counts for, `since` those after the one
/// before (`None`: the first): a frame of 33 for the first and after a
/// pause of over a second, 1 for two in one millisecond.
pub(crate) fn frame_ms(since: Option<u64>) -> f32 {
    match since {
        Some(0) => 1.0,
        Some(ms) if ms <= 1000 => ms as f32,
        _ => 33.0,
    }
}

/// The share of the way the slow mean goes at a keeping that counts for
/// `frame_ms`: the same share a second whatever the frame rate, if it
/// settles by the clock.
pub(crate) fn follow(settle: Option<Settle>, frame_ms: f32) -> f32 {
    match settle {
        Some(settle) => 1.0 - (-frame_ms / settle.ms).exp(),
        None => FOLLOW,
    }
}

/// The passes. From the fork's shader (`victory-gpu/src/ambient.rs` at
/// avatarmanz 9576c45f), with what Wine's HLSL compiler (vkd3d-shader 1.18
/// in CrossOver 26.3; `docs/macos.md`) and D3DMetal need changed:
/// - no `[RootSignature]`: Wine ignores it, so `crate::ambient_estimate`
///   builds it in code (`crate::compute::pipeline`): the constants in `b0`,
///   a table of `u0` and `t0`, then the two buffers as root views `u1` to
///   `u4`;
/// - one entry point, `main`, chosen by a `#define` put before the source
///   ([`part`]): `crate::compute::pipeline` compiles `main`;
/// - no stores through `RWStructuredBuffer` (not implemented in Wine's
///   compiler): each buffer is read through a structured view and written
///   through a raw one, as `crate::twin`'s shader does;
/// - no `GetDimensions` (Wine has no such method on unordered views): the
///   sizes are constants;
/// - the light visible surfaces send out is read through a shader resource
///   view, not through an unordered one: an R11G11B10_FLOAT image is not
///   one Metal reads and writes in one view;
/// - no `isnan`, `isinf`: comparisons, false for a NaN;
/// - conversions are written as casts, `near` (a keyword to some compilers)
///   is `near_z`, the constant buffer is all scalars, and what a
///   conditional expression would evaluate on both sides is behind an `if`
///   (as in `crate::ambient_set::SOURCE`);
/// - the light probe's sums (`InterlockedAdd` on a raw buffer) are left
///   out: `crate::ambient_estimate::log` reads the buffers instead.
pub(crate) const SOURCE: &str = r#"
#define CX 64
#define CY 36
#define CELLS 2304
#define PX 160
#define PY 90
#define FLOOR_AT (2 * CELLS + 1)
cbuffer Set : register(b0) {
    float level; float strength; float radius; float near_z;
    float tan_y; float follow; float most; uint width;
    uint height; uint flat; float air; uint slices;
    float jump; float fresh; float dt; float held;
    float mean_part; float turn; float least_part; float most_part;
    float sky_gain; float sky_least; float sky_power; float floor_r;
    float floor_g; float floor_b; uint sky_width; uint sky_height;
    float reflect_least;
};
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are, and at FLOOR_AT the least light any texel
// gets. Read through the structured view, written through the raw one (the
// same buffer).
RWStructuredBuffer<float4> cells : register(u1);
RWByteAddressBuffer cells_out : register(u2);
// Three a point: red, green and blue, each a direction and a mean.
RWStructuredBuffer<float4> points : register(u3);
RWByteAddressBuffer points_out : register(u4);
float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

#ifdef SEES
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), (int2)size - 1), 0));
    float z = min(near_z / max(d, 1e-7), 1000.0);
    float2 ndc = ((float2)p + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
#endif

#ifdef PACKS
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2((float)(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp((int)floor(log2(max(top, 1e-9))) + 16, 0, 31);
    float scale = exp2((float)(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2((float)(24 - e)); }
    uint3 m = min((uint3)(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | ((uint)e << 27);
}
#endif

#ifdef KEEP
Texture2D<float4> radiance : register(t0);
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float3 sum = 0.0;
    for (uint y = 0; y < 5; y++)
        for (uint x = 0; x < 5; x++) {
            uint2 p = (uint2)(((float2)at.xy + (float2(x, y) + 0.5) / 5.0) * float2(width, height) / float2(CX, CY));
            float3 c = radiance.Load(int3(p, 0)).rgb;
            // Neither negative, infinite nor a NaN (a comparison with one
            // is false).
            bool sane = c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0 && c.r < 65100.0 && c.g < 65100.0 && c.b < 65100.0;
            if (!sane) c = float3(0.0, 0.0, 0.0);
            // Lamps and glints count, but not for everything.
            sum += c / max(1.0, luma(c) / 20.0);
        }
    cells_out.Store4((at.y * CX + at.x) * 16, asuint(float4(sum / 25.0, 1.0)));
}
#endif

#ifdef MEAN
Texture2DArray<float4> sky : register(t1);
[numthreads(1, 1, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float3 sum = 0.0;
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
    float4 was = cells[CELLS];
    float3 mean = sum / CELLS;
    // (`jump` > 0: a change of more than that factor either way is
    // followed at once, as a new room is.)
    float a = max(luma(mean), 1e-5), b = max(luma(was.rgb), 1e-5);
    bool jumped = jump > 0.0 && max(a / b, b / a) > jump;
    // Its fourth part: seconds since the light around last jumped (from
    // 0.001; 0: nothing kept yet).
    float age = 0.001;
    if (was.a > 0.0 && !jumped) {
        age = was.a + dt;
        mean = lerp(was.rgb, mean, follow);
    }
    cells_out.Store4(CELLS * 16, asuint(float4(mean, age)));
    // The floor: the least any texel gets, the light from what is not on
    // screen. A share of the constant stand-in's light: of its level and
    // colour, or (gain above 0) of the light that follows the sky dome, as
    // `crate::ambient_set::SOURCE` works it out.
    float3 least_light = float3(floor_r, floor_g, floor_b);
    if (sky_gain > 0.0) {
        float3 all = 0.0;
        const int N = 8;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                // The sun's own texels are held down: it lights by itself.
                float3 there = sky.Load(int4((x * 2 + 1) * sky_width / (N * 2), (y * 2 + 1) * sky_height / (N * 2), 0, 0)).rgb;
                all += clamp(there, 0.0, 8.0);
            }
        float3 dome = all / (N * N);
        float m = luma(dome);
        // The sky's colour, half washed out; a dark sky has none worth
        // following.
        const float3 NIGHT = float3(0.6, 0.75, 1.0);
        float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), dome / max(m, 1e-6), 0.5), saturate(m / 0.05));
        least_light = hue * (sky_gain * pow(max(m, 0.0), sky_power) + sky_least);
    }
    cells_out.Store4(FLOOR_AT * 16, asuint(float4(least_light, 1.0)));
}
#endif

#ifdef PLACE
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen((int2)(((float2)at.xy + 0.5) * size / float2(CX, CY)), size);
    cells_out.Store4((CELLS + 1 + at.y * CX + at.x) * 16, asuint(float4(p, p.z * p.z)));
}
#endif

#ifdef GATHER
[numthreads(8, 2, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen((int2)(((float2)at.xy + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0.0;
    float3 toward = 0.0, light = 0.0, red = 0.0, green = 0.0, blue = 0.0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 48;
    points_out.Store4(at0, asuint(float4(red / whole - light.r * toward, light.r)));
    points_out.Store4(at0 + 16, asuint(float4(green / whole - light.g * toward, light.g)));
    points_out.Store4(at0 + 32, asuint(float4(blue / whole - light.b * toward, light.b)));
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    if (at.x >= width || at.y >= height) return;
    float2 size = float2(width, height);
    int2 p = (int2)at.xy;
    float4 around = cells[CELLS];
    // Without the scene's depth: the slow mean alone.
    float3 found = around.rgb * level;
    // How open the texel is (1 without depth).
    float open = 1.0;
    if (flat == 0) {
        float3 c = seen(p, size);
        float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
        float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
        float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
        float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
        float3 n = normalize(cross(dy, dx) + float3(0.0, 0.0, -1e-9));
        if (dot(n, c) > 0.0) n = -n;
        // How open it is (as in beloooo's cd9f78ce).
        float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
        float noise = frac(52.9829189 * frac(dot((float2)p, float2(0.06711056, 0.00583715))));
        if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
            float shut = 0.0;
            const int N = 12;
            for (int i = 0; i < N; i++) {
                float a = i * 2.399963 + noise * 6.2831853;
                float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
                float3 v = seen(p + (int2)round(o), size) - c;
                float vv = dot(v, v);
                float over = dot(v, n) - 0.004 * c.z - 0.005;
                shut += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
            }
            open = saturate(1.0 - strength * 2.0 * shut / N);
        }
        // Between the four points around, those at its own depth first;
        // which four is dithered a little.
        float2 g = ((float2)p + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
        int2 g0 = (int2)floor(g);
        float2 f = g - (float2)g0;
        float4 sr = 0.0, sg = 0.0, sb = 0.0;
        float sum = 0.0;
        for (int k = 0; k < 4; k++) {
            int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX - 1, PY - 1));
            float z = seen((int2)(((float2)q + 0.5) * size / float2(PX, PY)), size).z;
            float bx = (k & 1) != 0 ? f.x : 1.0 - f.x;
            float by = (k >> 1) != 0 ? f.y : 1.0 - f.y;
            float w = (bx * by + 1e-3) / (0.02 + abs(z - c.z) / c.z);
            uint at0 = (uint)(q.y * PX + q.x) * 3;
            sr += w * points[at0];
            sg += w * points[at0 + 1];
            sb += w * points[at0 + 2];
            sum += w;
        }
        float3 all = float3(sr.w, sg.w, sb.w) / sum;
        float3 lit = all + turn * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
        lit = clamp(lit, least_part * all, most_part * all);
        found = lerp(lit, around.rgb, mean_part) * level * open;
    }
    // No less than the floor, shaded as the estimate is.
    found = max(found, cells[FLOOR_AT].rgb * open);
    // What the program left counts as far as it can be true (`most` times
    // the estimate; 0: not at all): the sky's light outdoors, not the
    // blocks of other memory. For `fresh` seconds after the light around
    // jumped (a new room) it counts for no more than the estimate, then up
    // to `most` times it again: its history holds the place before.
    float3 written = found;
    if (most > 0.0) {
        float3 was = unpack(light[at.xy]);
        float bound = most;
        if (fresh > 0.0) bound = lerp(1.0, most, saturate(around.a / fresh));
        float top = max(luma(found) * bound, 1e-6);
        was *= min(1.0, top / max(luma(was), 1e-6));
        written = max(was, found);
    }
    light[at.xy] = pack(written);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is).
[numthreads(8, 8, 2)]
void main(uint3 at : SV_DispatchThreadID) {
    if (at.x >= width || at.y >= height || at.z >= slices) return;
    uint2 q = min((uint2)(((float2)at.xy + 0.5) / float2(width, height) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float away = (float)at.z / (float)slices;
    float cap = air * max(lerp(luma(cells[CELLS].rgb), luma(seen_there), away), luma(cells[FLOOR_AT].rgb));
    float3 c = scattered[at];
    float has = luma(c);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef HOLD
RWTexture2D<uint> light : register(u0);
// Reflections: no brighter than a part of the light around the point seen
// there (the slow mean's part of it as for the light itself), and, with
// reflect_least, no darker than another part of it.
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    if (at.x >= width || at.y >= height) return;
    uint2 q = min((uint2)(((float2)at.xy + 0.5) / float2(width, height) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float cap = max(luma(lerp(seen_there, cells[CELLS].rgb, mean_part)) * level, luma(cells[FLOOR_AT].rgb)) * held;
    float3 there = unpack(light[at.xy]);
    float has = luma(there);
    if (has > cap) {
        light[at.xy] = pack(there * (cap / has));
    } else if (reflect_least > 0.0) {
        float3 around = lerp(seen_there, cells[CELLS].rgb, mean_part) * level;
        if (luma(around) < luma(cells[FLOOR_AT].rgb)) around = cells[FLOOR_AT].rgb;
        float3 least = around * reflect_least;
        if (has < luma(least)) light[at.xy] = pack(least);
    }
}
#endif
"#;

/// The passes of [`SOURCE`], by the `#define` that chooses each.
pub(crate) const KEEP: &str = "KEEP";
pub(crate) const MEAN: &str = "MEAN";
pub(crate) const PLACE: &str = "PLACE";
pub(crate) const GATHER: &str = "GATHER";
pub(crate) const RAISE: &str = "RAISE";
pub(crate) const AIR_PASS: &str = "AIR";
pub(crate) const HOLD: &str = "HOLD";

/// The source of pass `pass`: [`SOURCE`] after the `#define`s that choose
/// its `main` and the helpers it uses.
pub(crate) fn part(pass: &str) -> String {
    let helpers = match pass {
        PLACE | GATHER => "#define SEES\n",
        RAISE => "#define SEES\n#define PACKS\n",
        HOLD => "#define PACKS\n",
        _ => "",
    };
    format!("#define {pass}\n{helpers}{SOURCE}")
}

/// What one pass is given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pass {
    /// The share of the estimate.
    pub(crate) level: f32,
    /// Strength, radius, near plane, tangent of half the vertical field of
    /// view (`crate::ambient_set::shade`).
    pub(crate) shade: [f32; 4],
    /// The share of the way the slow mean goes at this keeping.
    pub(crate) follow: f32,
    /// [`left`].
    pub(crate) most: f32,
    /// The size of the image the pass reads or writes.
    pub(crate) size: [u32; 2],
    /// No depth image: the slow mean alone is written.
    pub(crate) flat: bool,
    /// [`air`], and the slices of the volume the air's pass bounds.
    pub(crate) air: f32,
    pub(crate) slices: u32,
    /// [`Settle`]'s jump factor and fresh seconds (0: none), and the
    /// seconds this keeping counts for.
    pub(crate) jump: f32,
    pub(crate) fresh: f32,
    pub(crate) dt: f32,
    /// The most a reflection may hold, as a part of the light around
    /// (`crate::ambient_set::keep`).
    pub(crate) held: f32,
    /// [`mix`]: the slow mean's part, the turn, the least and the most.
    pub(crate) mix: [f32; 4],
    /// The floor ([`floor_of`]): gain (0: `floor` is the floor's light),
    /// least level and power of its sky part; its light without one; and
    /// the size of the sky dome's image.
    pub(crate) sky: [f32; 3],
    pub(crate) floor: [f32; 3],
    pub(crate) sky_size: [u32; 2],
    /// [`reflect_floor`]: the least a reflection holds, as a part of the
    /// light around (0: none).
    pub(crate) reflect_least: f32,
}

/// Dwords of the passes' constant buffer.
pub(crate) const CONSTANTS: usize = 29;

/// A pass's root constants, in the order the constant buffer declares them
/// (all scalars, so four to a row whatever the compiler's packing).
pub(crate) fn constants(p: &Pass) -> [u32; CONSTANTS] {
    let [strength, radius, near, tan_y] = p.shade;
    let floats = [p.level, strength, radius, near, tan_y, p.follow, p.most];
    let mut out = [0; CONSTANTS];
    for (to, f) in out.iter_mut().zip(floats) {
        *to = f.to_bits();
    }
    out[7..12].copy_from_slice(&[p.size[0], p.size[1], u32::from(p.flat), p.air.to_bits(), p.slices]);
    out[12..16].copy_from_slice(&[p.jump, p.fresh, p.dt, p.held].map(f32::to_bits));
    out[16..20].copy_from_slice(&p.mix.map(f32::to_bits));
    out[20..23].copy_from_slice(&p.sky.map(f32::to_bits));
    out[23..26].copy_from_slice(&p.floor.map(f32::to_bits));
    out[26..28].copy_from_slice(&p.sky_size);
    out[28] = p.reflect_least.to_bits();
    out
}

/// The line logged once at start when the estimate is on.
pub(crate) fn start_line(
    share: f32,
    shade: [f32; 4],
    most: f32,
    air: f32,
    settle: Option<Settle>,
    held: f32,
    mix: [f32; 4],
    floor: f32,
) -> String {
    let floored = if floor > 0.0 {
        format!(
            "no texel gets less than the floor, {floor} of the constant stand-in's light \
             (VICTORY_AMBIENT_FLOOR)"
        )
    } else {
        "no floor under it".to_owned()
    };
    let shaded = if shade[0] > 0.0 { "shaded from depth" } else { "not shaded" };
    let left = if most > 0.0 {
        format!("what the title's resolve left counts up to {most} times it")
    } else {
        "what the title's resolve left is replaced".to_owned()
    };
    let air = if air > 0.0 {
        format!("the air scatters no more than {air} times the light around (VICTORY_AMBIENT_AIR)")
    } else {
        "what the air scatters is left alone".to_owned()
    };
    let settles = match settle {
        Some(Settle { ms, jump, fresh }) => format!(
            "the mean of the light around settles by the clock (VICTORY_AMBIENT_SETTLE: {ms} ms, a \
             change by more than {jump} times at once, {fresh} s fresh)"
        ),
        None => format!("the mean of the light around goes {FOLLOW} of the way a frame"),
    };
    let reflections = if held > 0.0 {
        format!("reflections held to {held} of the light around")
    } else {
        "reflections left alone".to_owned()
    };
    format!(
        "VICTORY_AMBIENT=estimate: the screen probes' indirect diffuse light is replaced by an \
         estimate from the light visible surfaces send out ({share} of it; {CX}x{CY} cells, \
         {PX}x{PY} points; a texel is {} the slow mean, turned {} by its normal, between {} and {} \
         times the mean around it), {shaded}; {floored}; {left}; {reflections}; {air}; {settles}",
        mix[0], mix[1], mix[2], mix[3]
    )
}

fn luminance(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

/// What the buffers show of the estimate, for the frame line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Seen {
    /// The slow mean of the light around.
    pub(crate) mean: [f32; 3],
    /// The floor's light.
    pub(crate) floor: [f32; 3],
    /// The points' luminance: mean, least and most.
    pub(crate) points: [f32; 3],
    /// The mean luminance written (each point's share of the estimate, or
    /// the floor where that is more), and the part of the points under the
    /// floor.
    pub(crate) written: f32,
    pub(crate) under: f32,
}

/// What the points' light (each point's red, green and blue means) comes
/// to with the estimate's `share` over a floor of luminance `floor`: the
/// mean, the least and the most luminance of the points, the mean
/// luminance written and the part of the points under the floor. Points
/// that are not a number are left out.
pub(crate) fn over_the_picture(points: impl Iterator<Item = [f32; 3]>, share: f32, floor: f32) -> ([f32; 3], f32, f32) {
    let (mut sum, mut least, mut most, mut n) = (0.0f64, f32::INFINITY, 0.0f32, 0u32);
    let (mut written, mut under) = (0.0f64, 0u32);
    for l in points.map(luminance).filter(|l| l.is_finite()) {
        sum += f64::from(l);
        least = least.min(l);
        most = most.max(l);
        written += f64::from((l * share).max(floor));
        under += u32::from(l * share < floor);
        n += 1;
    }
    if n == 0 {
        return ([0.0; 3], 0.0, 0.0);
    }
    let per = |v: f64| (v / f64::from(n)) as f32;
    ([per(sum), least, most], per(written), per(f64::from(under)))
}

/// The line logged with the frame line: the slow mean of the light around,
/// the floor and the points' luminance (`seen`; `None`: nothing kept yet),
/// and the keepings and raisings recorded a frame.
pub(crate) fn frame_line(
    share: f32,
    floor: f32,
    seen: Option<Seen>,
    per_frame: [f64; 4],
    probe: Option<usize>,
) -> String {
    let [keeps, raises, airs, holds] = per_frame;
    let probe = match probe {
        Some(step) => format!(
            "; probe (VICTORY_AMBIENT_PROBE) step {step} of {}: these frames were recorded without {}",
            PROBE.len(),
            PROBE[step % PROBE.len()].1
        ),
        None => String::new(),
    };
    let what = match seen {
        Some(Seen { mean: [r, g, b], floor: least_light, points: [mean, least, most], written, under }) => format!(
            "the light around (slow mean) is {r:.4} {g:.4} {b:.4}, luminance {:.4}; the picture's \
             {PX}x{PY} points hold a luminance of {mean:.4} on average (least {least:.4}, most \
             {most:.4}); the floor (VICTORY_AMBIENT_FLOOR={floor}) is {:.4}, and {:.0}% of the points \
             are under it; texels are written about {written:.4} (the share {share} of the \
             estimate, or the floor; the constant stand-in writes its level)",
            luminance([r, g, b]),
            luminance(least_light),
            under * 100.0,
        ),
        None => format!(
            "nothing kept yet: no dispatch of {} seen, the title's own light stays",
            String::from_utf8_lossy(RADIANCE_PROGRAM)
        ),
    };
    format!(
        "ambient estimate (VICTORY_AMBIENT=estimate): {what}; {keeps:.1} keepings, {raises:.1} \
         raisings, {holds:.1} holds of reflections and {airs:.1} bounds of the air/frame{probe}"
    )
}

/// The passes' arithmetic on the CPU, as [`SOURCE`] has it: what the tests
/// here and the GPU tests of `crate::ambient_estimate` compare with.
#[cfg(test)]
pub(crate) mod reference {
    use super::{CELLS, CX, CY, PX, PY};

    pub(crate) type V3 = [f32; 3];
    pub(crate) type V4 = [f32; 4];

    pub(crate) fn luma(c: V3) -> f32 {
        c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
    }

    fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dot(a: V3, b: V3) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn cross(a: V3, b: V3) -> V3 {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }

    /// A picture's depth (reversed: `near / distance`), its size, the near
    /// plane and the tangent of half the vertical field of view.
    pub(crate) struct Scene<'a> {
        pub(crate) depth: &'a dyn Fn(u32, u32) -> f32,
        pub(crate) size: [u32; 2],
        pub(crate) near: f32,
        pub(crate) tan_y: f32,
    }

    impl Scene<'_> {
        /// `seen` of the shader: the point at a texel (clamped to the
        /// picture), in metres from the eye.
        pub(crate) fn seen(&self, p: [i32; 2]) -> V3 {
            let [w, h] = self.size;
            let at = [p[0].clamp(0, w as i32 - 1), p[1].clamp(0, h as i32 - 1)];
            let d = (self.depth)(at[0] as u32, at[1] as u32);
            let z = (self.near / d.max(1e-7)).min(1000.0);
            let (w, h) = (w as f32, h as f32);
            let ndc = [(p[0] as f32 + 0.5) / w * 2.0 - 1.0, (p[1] as f32 + 0.5) / h * 2.0 - 1.0];
            [ndc[0] * self.tan_y * w / h * z, -ndc[1] * self.tan_y * z, z]
        }

        /// The texel at the middle of place (`x`, `y`) of a grid of `nx` x
        /// `ny` over the picture.
        fn middle(&self, x: u32, y: u32, nx: u32, ny: u32) -> [i32; 2] {
            let at = |i: u32, n: u32, size: u32| ((i as f32 + 0.5) * size as f32 / n as f32) as i32;
            [at(x, nx, self.size[0]), at(y, ny, self.size[1])]
        }

        /// The `PLACE` pass: where each cell is, and its weight.
        pub(crate) fn place(&self) -> Vec<V4> {
            (0..CELLS)
                .map(|i| {
                    let p = self.seen(self.middle(i % CX, i / CX, CX, CY));
                    [p[0], p[1], p[2], p[2] * p[2]]
                })
                .collect()
        }

        /// The `GATHER` pass: at each point, per colour a direction and a
        /// mean of the cells' `light`, weighted by the share of the point's
        /// surroundings each cell covers.
        pub(crate) fn gather(&self, light: &[V3], places: &[V4]) -> Vec<[V4; 3]> {
            (0..PX * PY)
                .map(|i| {
                    let p = self.seen(self.middle(i % PX, i / PX, PX, PY));
                    let r2 = 0.25 + 0.0025 * p[2] * p[2];
                    let mut whole = 0.0f32;
                    let (mut toward, mut all) = ([0.0f32; 3], [0.0f32; 3]);
                    let mut by = [[0.0f32; 3]; 3];
                    for (l, c) in light.iter().zip(places) {
                        let d = sub([c[0], c[1], c[2]], p);
                        let dd = dot(d, d);
                        let w = c[3] / (dd + r2);
                        let dir = d.map(|v| v / (dd + 0.25 * r2).sqrt());
                        whole += w;
                        for k in 0..3 {
                            toward[k] += w * dir[k];
                            all[k] += w * l[k];
                            for (colour, sum) in by.iter_mut().enumerate() {
                                sum[k] += w * l[colour] * dir[k];
                            }
                        }
                    }
                    let whole = whole.max(1e-6);
                    let all = all.map(|v| v / whole);
                    let toward = toward.map(|v| v / whole);
                    let one = |colour: usize| {
                        let d: Vec<f32> =
                            (0..3).map(|k| by[colour][k] / whole - all[colour] * toward[k]).collect();
                        [d[0], d[1], d[2], all[colour]]
                    };
                    [one(0), one(1), one(2)]
                })
                .collect()
        }

        /// The normal at texel `p`, toward the eye, from the depth alone,
        /// and how open the texel is to the light around (1 in the open).
        pub(crate) fn surface(&self, p: [i32; 2], strength: f32, radius: f32) -> (V3, f32) {
            let c = self.seen(p);
            let at = |dx: i32, dy: i32| self.seen([p[0] + dx, p[1] + dy]);
            let (l, r, u, d) = (at(-1, 0), at(1, 0), at(0, -1), at(0, 1));
            let dx = if (l[2] - c[2]).abs() < (r[2] - c[2]).abs() { sub(c, l) } else { sub(r, c) };
            let dy = if (u[2] - c[2]).abs() < (d[2] - c[2]).abs() { sub(c, u) } else { sub(d, c) };
            let mut n = cross(dy, dx);
            n[2] -= 1e-9;
            let len = dot(n, n).sqrt();
            let mut n = n.map(|v| v / len);
            if dot(n, c) > 0.0 {
                n = n.map(|v| -v);
            }
            let px = (radius * self.size[1] as f32 / (2.0 * self.tan_y * c[2])).min(128.0);
            let mut open = 1.0;
            if strength > 0.0 && px >= 1.5 && c[2] < 400.0 {
                const N: usize = 12;
                let noise = noise(p);
                let mut shut = 0.0;
                for i in 0..N {
                    let a = i as f32 * 2.399963 + noise * 6.2831853;
                    let far = ((i as f32 + 0.5) / N as f32).sqrt() * px;
                    // (`round` of the shader rounds halves to even.)
                    let o = [(a.cos() * far).round_ties_even() as i32, (a.sin() * far).round_ties_even() as i32];
                    let v = sub(self.seen([p[0] + o[0], p[1] + o[1]]), c);
                    let vv = dot(v, v);
                    let over = dot(v, n) - 0.004 * c[2] - 0.005;
                    shut += over.max(0.0) / (vv + 1e-6).sqrt() * (1.0 - vv / (radius * radius)).clamp(0.0, 1.0);
                }
                open = (1.0 - strength * 2.0 * shut / N as f32).clamp(0.0, 1.0);
            }
            (n, open)
        }

        /// The `RAISE` pass at texel `p`: the light written there, before
        /// it is packed. `was` is what the title's resolve left.
        pub(crate) fn raise(&self, p: [i32; 2], points: &[[V4; 3]], around: V4, set: &super::Pass, was: V3) -> V3 {
            self.raise_over(p, points, around, set, was, [0.0; 3])
        }

        /// [`Scene::raise`] over the floor's light `least`
        /// ([`floor_light`]): no texel gets less, shaded as the estimate
        /// is.
        pub(crate) fn raise_over(
            &self,
            p: [i32; 2],
            points: &[[V4; 3]],
            around: V4,
            set: &super::Pass,
            was: V3,
            least: V3,
        ) -> V3 {
            let age = around[3];
            let around = [around[0], around[1], around[2]];
            let mut found = around.map(|v| v * set.level);
            let mut open = 1.0;
            if !set.flat {
                let c = self.seen(p);
                let n;
                (n, open) = self.surface(p, set.shade[0], set.shade[1]);
                let noise = noise(p);
                let g = [0, 1].map(|k| {
                    (p[k] as f32 + 0.5) / self.size[k] as f32 * [PX, PY][k] as f32 - 0.5 + (noise - 0.5) * 0.5
                });
                let g0 = g.map(f32::floor);
                let f = [g[0] - g0[0], g[1] - g0[1]];
                let mut sums = [[0.0f32; 4]; 3];
                let mut sum = 0.0;
                for k in 0..4 {
                    let q = [
                        (g0[0] as i32 + (k & 1)).clamp(0, PX as i32 - 1) as u32,
                        (g0[1] as i32 + (k >> 1)).clamp(0, PY as i32 - 1) as u32,
                    ];
                    let z = self.seen(self.middle(q[0], q[1], PX, PY))[2];
                    let bx = if k & 1 != 0 { f[0] } else { 1.0 - f[0] };
                    let by = if k >> 1 != 0 { f[1] } else { 1.0 - f[1] };
                    let w = (bx * by + 1e-3) / (0.02 + (z - c[2]).abs() / c[2]);
                    let point = &points[(q[1] * PX + q[0]) as usize];
                    for (to, from) in sums.iter_mut().zip(point) {
                        for (to, from) in to.iter_mut().zip(from) {
                            *to += w * from;
                        }
                    }
                    sum += w;
                }
                let [mean_part, turn, least, most] = set.mix;
                for (k, found) in found.iter_mut().enumerate() {
                    let s = sums[k];
                    let all = s[3] / sum;
                    let lit = all + turn * dot([s[0], s[1], s[2]], n) / sum;
                    let lit = lit.max(least * all).min(most * all);
                    *found = (lit + (around[k] - lit) * mean_part) * set.level * open;
                }
            }
            let found = [0, 1, 2].map(|k| found[k].max(least[k] * open));
            if set.most <= 0.0 {
                return found;
            }
            let mut bound = set.most;
            if set.fresh > 0.0 {
                bound = 1.0 + (set.most - 1.0) * (age / set.fresh).clamp(0.0, 1.0);
            }
            let top = (luma(found) * bound).max(1e-6);
            let scale = (top / luma(was).max(1e-6)).min(1.0);
            [0, 1, 2].map(|k| (was[k] * scale).max(found[k]))
        }
    }

    /// The shader's noise at a texel, in [0, 1).
    pub(crate) fn noise(p: [i32; 2]) -> f32 {
        let inner = p[0] as f32 * 0.06711056 + p[1] as f32 * 0.00583715;
        (52.9829189 * inner.fract()).fract()
    }

    /// The `KEEP` pass: the light of each cell, the mean of 25 texels of
    /// `radiance` (`size` texels), bright ones held down.
    pub(crate) fn keep(radiance: &dyn Fn(u32, u32) -> V3, size: [u32; 2]) -> Vec<V3> {
        (0..CELLS)
            .map(|i| {
                let mut sum = [0.0f32; 3];
                for s in 0..25u32 {
                    let at = |cell: u32, sample: u32, size: u32, cells: u32| {
                        ((cell as f32 + (sample as f32 + 0.5) / 5.0) * size as f32 / cells as f32) as u32
                    };
                    let mut c = radiance(at(i % CX, s % 5, size[0], CX), at(i / CX, s / 5, size[1], CY));
                    if !c.iter().all(|&v| (0.0..65100.0).contains(&v)) {
                        c = [0.0; 3];
                    }
                    let held = (luma(c) / 20.0).max(1.0);
                    for k in 0..3 {
                        sum[k] += c[k] / held;
                    }
                }
                sum.map(|v| v / 25.0)
            })
            .collect()
    }

    /// The `MEAN` pass: the slow mean after a keeping of `cells`, from the
    /// one before (`was`; its fourth part 0: none), and in its fourth part
    /// the seconds since it last jumped (by more than `jump` times; 0:
    /// never), this keeping counting for `dt`.
    pub(crate) fn mean(cells: &[V3], was: V4, follow: f32, jump: f32, dt: f32) -> V4 {
        let mut now = [0.0f32; 3];
        for c in cells {
            for k in 0..3 {
                now[k] += c[k];
            }
        }
        let now = now.map(|v| v / cells.len() as f32);
        let (a, b) = (luma(now).max(1e-5), luma([was[0], was[1], was[2]]).max(1e-5));
        let jumped = jump > 0.0 && (a / b).max(b / a) > jump;
        if was[3] > 0.0 && !jumped {
            let m = [0, 1, 2].map(|k| was[k] + (now[k] - was[k]) * follow);
            [m[0], m[1], m[2], was[3] + dt]
        } else {
            [now[0], now[1], now[2], 0.001]
        }
    }

    /// The `AIR` pass: what the air scatters at a place a share `away` of
    /// the volume's depth, `c`, bounded by `air` times the light around
    /// (the slow mean `around` near the eye, the light of the point seen in
    /// that direction, `seen_there`, far away).
    pub(crate) fn bound_air(c: V3, air: f32, around: V3, seen_there: V3, away: f32) -> V3 {
        bound_air_over(c, air, around, seen_there, away, [0.0; 3])
    }

    /// [`bound_air`] with the floor's light `least` counted as light
    /// around.
    pub(crate) fn bound_air_over(c: V3, air: f32, around: V3, seen_there: V3, away: f32, least: V3) -> V3 {
        let (near, far) = (luma(around), luma(seen_there));
        let cap = air * (near + (far - near) * away).max(luma(least));
        let has = luma(c);
        if has > cap { c.map(|v| v * (cap / has)) } else { c }
    }

    /// The `HOLD` pass: a reflection `there`, held to the part `held` of
    /// the light around the point seen there (`seen_there`, a fifth of it
    /// the slow mean `around`), `level` being the estimate's share.
    pub(crate) fn hold(there: V3, set: &super::Pass, around: V3, seen_there: V3) -> V3 {
        hold_over(there, set, around, seen_there, [0.0; 3])
    }

    /// [`hold`] with the floor's light `least` counted as light around.
    /// With [`super::Pass::reflect_least`], a reflection no brighter than
    /// the cap and darker than that part of the light around is that part
    /// of it, in its colour.
    pub(crate) fn hold_over(there: V3, set: &super::Pass, around: V3, seen_there: V3, least: V3) -> V3 {
        let light = [0, 1, 2].map(|k| (seen_there[k] + (around[k] - seen_there[k]) * set.mix[0]) * set.level);
        let cap = luma(light).max(luma(least)) * set.held;
        let has = luma(there);
        if has > cap {
            return there.map(|v| v * (cap / has));
        }
        if set.reflect_least > 0.0 {
            let around = if luma(light) < luma(least) { least } else { light };
            let floor = around.map(|v| v * set.reflect_least);
            if has < luma(floor) {
                return floor;
            }
        }
        there
    }

    /// The light that follows the sky dome, as the `MEAN` pass (and
    /// `crate::ambient_set::SOURCE`) works it out from the dome's mean
    /// `dome`, with gain, least level and power `follow`.
    pub(crate) fn sky_light(dome: V3, follow: [f32; 3]) -> V3 {
        let [gain, least, power] = follow;
        let m = luma(dome);
        let night = [0.6, 0.75, 1.0];
        let t = (m / 0.05).clamp(0.0, 1.0);
        let level = gain * m.max(0.0).powf(power) + least;
        [0, 1, 2].map(|k| {
            let washed = 1.0 + (dome[k] / m.max(1e-6) - 1.0) * 0.5;
            (night[k] + (washed - night[k]) * t) * level
        })
    }

    /// The floor's light a pass with `set` writes: its constant light, or
    /// with a sky part the light that follows a dome of mean `dome`.
    pub(crate) fn floor_light(set: &super::Pass, dome: V3) -> V3 {
        if set.sky[0] > 0.0 { sky_light(dome, set.sky) } else { set.floor }
    }

    /// R9G9B9E5 as the shader packs it.
    pub(crate) fn pack(c: V3) -> u32 {
        let c = c.map(|v| v.clamp(0.0, 65408.0));
        let top = c[0].max(c[1]).max(c[2]);
        let mut e = (top.max(1e-9).log2().floor() as i32 + 16).clamp(0, 31);
        let mut scale = 2f32.powi(24 - e);
        if top * scale >= 511.5 {
            e = (e + 1).min(31);
            scale = 2f32.powi(24 - e);
        }
        let m = c.map(|v| ((v * scale + 0.5) as u32).min(511));
        m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
    }

    pub(crate) fn unpack(v: u32) -> V3 {
        let scale = 2f32.powi((v >> 27) as i32 - 24);
        [v & 511, (v >> 9) & 511, (v >> 18) & 511].map(|m| m as f32 * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{Scene, V3, keep, luma, mean, pack, unpack};
    use super::*;

    /// `estimate` and `estimate:<share>` ask for the estimate; everything
    /// else is left to the constant stand-in.
    #[test]
    fn the_estimate_is_asked_for_by_name() {
        for other in [None, Some(""), Some("0"), Some("off"), Some("1"), Some("0.6"), Some("on"), Some("constant:0.3")] {
            assert_eq!(share(other), None, "{other:?}");
        }
        assert_eq!(share(Some("estimate")), Some(1.0));
        assert_eq!(share(Some(" Estimate ")), Some(1.0));
        assert_eq!(share(Some("estimate:0.8")), Some(0.8));
        assert_eq!(share(Some("estimate: 2")), Some(2.0));
        // A share that is no number above 0 counts as 1.
        for odd in ["estimate:", "estimate:x", "estimate:0", "estimate:-1", "estimate:nan", "estimate:inf"] {
            assert_eq!(share(Some(odd)), Some(1.0), "{odd}");
        }
        // Not the estimate: another word that starts the same.
        assert_eq!(share(Some("estimated")), None);
        // Every value that asks for the estimate also turns the stand-in on
        // for `crate::ambient`, whose hooks the estimate runs from.
        for on in ["estimate", "estimate:0.8", "estimate:0", "Estimate"] {
            assert!(crate::ambient_set::is_on(Some(on)), "{on}");
        }
    }

    /// The air is left alone unless a multiple above 0 is set; then it
    /// scatters no more than that many times the light around: the slow
    /// mean near the eye, what is seen in its direction far away.
    #[test]
    fn the_air_is_bounded_by_the_light_around() {
        for unset in [None, Some(""), Some("0"), Some("-2"), Some("x"), Some("inf")] {
            assert_eq!(air(unset), 0.0, "{unset:?}");
        }
        assert_eq!(air(Some(" 3 ")), 3.0);
        use super::reference::bound_air;
        let (around, window) = ([0.1f32; 3], [2.0f32; 3]);
        // Near the eye: three times the room's light, its colour kept.
        let veil = bound_air([3.0, 2.0, 1.0], 3.0, around, window, 0.0);
        assert!(near(luma(veil), 0.3, 1e-4) && near(veil[0] / veil[2], 3.0, 1e-4), "{veil:?}");
        // Toward the window, far away: three times the window's.
        assert_eq!(bound_air([3.0, 2.0, 1.0], 3.0, around, window, 1.0), [3.0, 2.0, 1.0]);
        let half = bound_air([9.0; 3], 3.0, around, window, 0.5);
        assert!(near(half[0], 3.0 * 1.05, 1e-4), "{half:?}");
        // Darker air is not raised.
        assert_eq!(bound_air([0.01; 3], 3.0, around, window, 0.0), [0.01; 3]);
    }

    /// A reflection holds no more than a part of the light around the
    /// point it is seen at, keeps its colour, and is never raised.
    #[test]
    fn reflections_are_held_to_the_light_around() {
        use super::reference::hold;
        let (around, there) = ([0.2f32; 3], [1.0f32, 1.0, 1.0]);
        // The light around the point: 0.8 of its own, 0.2 of the slow mean.
        let glaze = hold([6.0, 4.0, 2.0], &SET, around, there);
        assert!(near(luma(glaze), 0.5 * 0.84, 1e-4) && near(glaze[0] / glaze[2], 3.0, 1e-4), "{glaze:?}");
        assert_eq!(hold([0.1; 3], &SET, around, there), [0.1; 3]);
        // With the estimate's share.
        assert!(near(hold([9.0; 3], &Pass { level: 0.5, ..SET }, around, there)[0], 0.21, 1e-4));
        // In a dark room, the sky's reflection goes with the room.
        assert!(near(hold([9.0; 3], &SET, [0.02; 3], [0.01; 3])[0], 0.5 * 0.012, 1e-4));
        // With all of it the slow mean.
        assert!(near(hold([9.0; 3], &Pass { mix: [1.0, 1.2, 0.2, 2.5], ..SET }, around, there)[0], 0.1, 1e-4));
    }

    /// The reflection floor: off unless set; set, a black reflection is
    /// that part of the light around (in its colour, against the floor's
    /// light where the room is darker), a brighter one is left alone, and
    /// the cap still holds.
    #[test]
    fn the_reflection_floor_is_a_part_of_the_light_around() {
        assert_eq!(reflect_floor(None), 0.0);
        assert_eq!(reflect_floor(Some("x")), 0.0);
        assert_eq!(reflect_floor(Some("-1")), 0.0);
        assert_eq!(reflect_floor(Some(" 0.15 ")), 0.15);
        use super::reference::{hold, hold_over};
        let (around, seen) = ([0.2f32, 0.2, 0.4], [1.0f32, 0.5, 0.25]);
        let least = [0.25f32, 0.25, 0.3];
        // Unset (and 0): exactly the hold without it.
        for there in [[0.0; 3], [0.001, 0.0, 0.0], [0.1; 3], [9.0, 4.0, 2.0]] {
            assert_eq!(hold_over(there, &SET, around, seen, least), hold_over(there, &Pass { reflect_least: 0.0, ..SET }, around, seen, least));
            assert_eq!(hold(there, &Pass { reflect_least: 0.0, ..SET }, around, seen), hold(there, &SET, around, seen));
        }
        let floored = Pass { reflect_least: 0.15, ..SET };
        // The light around there: 0.8 of the point's, 0.2 of the slow mean.
        let light = [0.84f32, 0.44, 0.28];
        let got = hold([0.0; 3], &floored, around, seen);
        assert!((0..3).all(|k| near(got[k], light[k] * 0.15, 1e-6)), "{got:?}");
        // With the estimate's share.
        let half = Pass { level: 0.5, ..floored };
        let got = hold([0.0; 3], &half, around, seen);
        assert!((0..3).all(|k| near(got[k], light[k] * 0.5 * 0.15, 1e-6)), "{got:?}");
        // A brighter reflection is left alone; one above the cap is capped.
        assert_eq!(hold([0.2, 0.1, 0.05], &floored, around, seen), [0.2, 0.1, 0.05]);
        assert_eq!(hold([9.0; 3], &floored, around, seen), hold([9.0; 3], &SET, around, seen));
        // In a room darker than the floor's light: a part of the floor's.
        let got = hold_over([0.0; 3], &floored, [0.002; 3], [0.002; 3], least);
        assert!((0..3).all(|k| near(got[k], least[k] * 0.15, 1e-6)), "{got:?}");
    }

    #[test]
    fn the_mix_keeps_the_forks_numbers_unless_set() {
        assert_eq!(mix(None), [0.2, 1.2, 0.2, 2.5]);
        assert_eq!(mix(Some("")), MIX);
        assert_eq!(mix(Some("0.5")), [0.5, 1.2, 0.2, 2.5]);
        assert_eq!(mix(Some("0.5, 0 ,1,4")), [0.5, 0.0, 1.0, 4.0]);
        assert_eq!(mix(Some("x,,0.4")), [0.2, 1.2, 0.4, 2.5]);
        // Held to what makes sense: a part, nothing negative, most above least.
        assert_eq!(mix(Some("3,-1,5,nan")), [1.0, 1.2, 5.0, 5.0]);
    }

    /// The probe leaves the passes out in turn, each step for as many frame
    /// lines as it is told, and starts with nothing left out.
    #[test]
    fn the_probe_leaves_parts_out_in_turn() {
        for off in [None, Some(""), Some("0"), Some("x"), Some("-2"), Some("1.5")] {
            assert_eq!(probe(off), None, "{off:?}");
        }
        assert_eq!(probe(Some(" 2 ")), Some(2));
        let steps: Vec<usize> = (0..16).map(|lines| probe_step(lines, 2)).collect();
        assert_eq!(steps, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0]);
        assert_eq!(probe_step(9, 0), 2);
        assert_eq!(PROBE[0].0, 0);
        // Each part is left out alone once, and the last step leaves out
        // every pass that is recorded.
        for part in [NO_GATHER, NO_SHADE, NO_HOLD, NO_KEEP, NO_RAISE] {
            assert_eq!(PROBE.iter().filter(|(out, _)| *out == part).count(), 1, "{part}");
        }
        assert_eq!(PROBE[6].0 & (NO_KEEP | NO_RAISE | NO_HOLD), NO_KEEP | NO_RAISE | NO_HOLD);
    }

    /// The floor: a share of the constant stand-in's light, by default 0.3,
    /// its sky part cut to the share and the least level whole.
    #[test]
    fn the_floor_is_a_share_of_the_constant_light() {
        assert_eq!(floor(None), 0.3);
        assert_eq!(floor(Some("x")), 0.3);
        assert_eq!(floor(Some("-1")), 0.3);
        assert_eq!(floor(Some(" 0 ")), 0.0);
        assert_eq!(floor(Some("1")), 1.0);
        let (tint, follow) = ([1.0, 0.98, 0.94], Some([0.36, 0.05, 0.4]));
        // Following the dome: the gain cut, the least and the power kept.
        assert_eq!(floor_of(0.5, 0.6, tint, follow, true), ([0.18, 0.05, 0.4], [0.0; 3]));
        // Until the dome is drawn, and with the sky not followed: the
        // constant level and colour.
        let constant = ([0.0, 0.0, 1.0], [0.3, 0.6 * 0.98 * 0.5, 0.6 * 0.94 * 0.5]);
        assert_eq!(floor_of(0.5, 0.6, tint, follow, false), constant);
        assert_eq!(floor_of(0.5, 0.6, tint, None, true), constant);
        // No floor: nothing, whatever the sky.
        assert_eq!(floor_of(0.0, 0.6, tint, follow, true), ([0.0, 0.0, 1.0], [0.0; 3]));
        use super::reference::{floor_light, sky_light};
        // At night the sky's part is next to nothing and the least level
        // stays: 0.05 and a little, bluish.
        let night = sky_light([0.001; 3], [0.36 * 0.3, 0.05, 0.4]);
        assert!(luma(night) > 0.04 && luma(night) < 0.06 && night[2] > night[0], "{night:?}");
        // The constant stand-in's own night light: a level of 0.07, bluish.
        assert!(near(luma(sky_light([0.001; 3], [0.36, 0.05, 0.4])), 0.054, 0.03));
        // By day: the share of the sky's part plus the least, in the sky's
        // colour half washed out.
        let day = sky_light([4.0, 5.0, 8.0], [0.36 * 0.3, 0.05, 0.4]);
        let m = luma([4.0, 5.0, 8.0]);
        assert!(near(luma(day), 0.108 * m.powf(0.4) + 0.05, 1e-4) && day[2] > day[1] && day[1] > day[0], "{day:?}");
        let set = Pass { sky: [0.36 * 0.3, 0.05, 0.4], ..SET };
        assert_eq!(floor_light(&set, [4.0, 5.0, 8.0]), day);
        assert_eq!(floor_light(&Pass { floor: [0.2, 0.1, 0.05], ..SET }, [4.0, 5.0, 8.0]), [0.2, 0.1, 0.05]);
    }

    /// With a floor no texel gets less than it: a room with nothing lit on
    /// screen is the floor (shaded in creases), a lit one is untouched, and
    /// reflections and the air are bounded against the floor too.
    #[test]
    fn no_texel_gets_less_than_the_floor() {
        let depth = wall(4.0);
        let scene = Scene { depth: &depth, size: [640, 360], near: 0.1, tan_y: 0.45 };
        let least = [0.25f32, 0.25, 0.3];
        // The apartment of the opening: nearly nothing on screen.
        let dark = vec![[0.002f32; 3]; CELLS as usize];
        let points = scene.gather(&dark, &scene.place());
        let got = scene.raise_over([320, 180], &points, [0.002, 0.002, 0.002, 1.0], &SET, [0.0; 3], least);
        assert_eq!(got, least);
        assert!(scene.raise([320, 180], &points, [0.002, 0.002, 0.002, 1.0], &SET, [0.0; 3])[0] < 0.003);
        // The same without depth.
        let flat = Pass { flat: true, ..SET };
        assert_eq!(scene.raise_over([320, 180], &points, [0.002, 0.002, 0.002, 1.0], &flat, [0.0; 3], least), least);
        // A lit room keeps its own light, each colour by itself.
        let lit = vec![[1.0f32, 0.5, 0.1]; CELLS as usize];
        let points = scene.gather(&lit, &scene.place());
        let got = scene.raise_over([320, 180], &points, [1.0, 0.5, 0.1, 1.0], &SET, [0.0; 3], least);
        assert!(near(got[0], 1.0, 1e-3) && near(got[1], 0.5, 1e-3) && near(got[2], 0.3, 1e-6), "{got:?}");
        // What the resolve left counts against the floored light.
        let left = Pass { most: 4.0, ..SET };
        let dark_points = scene.gather(&dark, &scene.place());
        let got = scene.raise_over([320, 180], &dark_points, [0.002, 0.002, 0.002, 1.0], &left, [9.0; 3], [0.25; 3]);
        assert!(near(got[0], 1.0, 1e-3), "{got:?}");
        // The floor is shaded in a crease as the estimate is.
        let size = [640u32, 360];
        let corner = move |_: u32, y: u32| {
            let down = ((y as f32 + 0.5) / size[1] as f32 * 2.0 - 1.0) * 0.45;
            0.1 / if down > 0.0 { (0.6 / down).min(3.0) } else { 3.0 }
        };
        let scene = Scene { depth: &corner, size, near: 0.1, tan_y: 0.45 };
        let points = scene.gather(&dark, &scene.place());
        let crease = ((0.2 / 0.45 + 1.0) / 2.0 * 360.0) as i32 + 3;
        let (_, open) = scene.surface([320, crease], 1.0, 0.6);
        let got = scene.raise_over([320, crease], &points, [0.002, 0.002, 0.002, 1.0], &SET, [0.0; 3], [0.25; 3]);
        assert!(open < 0.9 && near(got[0], 0.25 * open, 1e-5), "{open} {got:?}");
        // Reflections: held to a part of the floor where the room is dark.
        use super::reference::{bound_air_over, hold, hold_over};
        assert!(hold([9.0; 3], &SET, [0.002; 3], [0.002; 3])[0] < 0.002);
        assert!(near(hold_over([9.0; 3], &SET, [0.002; 3], [0.002; 3], [0.25; 3])[0], 0.125, 1e-4));
        assert!(near(hold_over([9.0; 3], &SET, [2.0; 3], [2.0; 3], [0.25; 3])[0], 1.0, 1e-4));
        // The air: bounded by a multiple of the floor at least.
        assert!(near(bound_air_over([9.0; 3], 3.0, [0.002; 3], [0.002; 3], 0.5, [0.25; 3])[0], 0.75, 1e-4));
    }

    #[test]
    fn what_the_resolve_left_is_replaced_by_default() {
        assert_eq!(left(None), 0.0);
        assert_eq!(left(Some("x")), 0.0);
        assert_eq!(left(Some("-1")), 0.0);
        assert_eq!(left(Some(" 4 ")), 4.0);
    }

    /// The programs are known by the start of their names, and none of the
    /// constant stand-in's programs is taken for one of them.
    #[test]
    fn programs_are_known_by_name() {
        assert_eq!(role_of(b"..tiled_lighting:CS_CacheScreenSpaceRadiance_eed2fc35_Wrapped.."), KEEPS);
        assert_eq!(role_of(b"CS_CacheScreenSpaceRadianc"), 0);
        assert_eq!(role_of(b"scattering_transmittance_system:CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), LIGHTS_AIR);
        assert_eq!(role_of(b"CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped"), 0);
        assert_eq!(role_of(b""), 0);
        use crate::ambient_set as constant;
        assert_eq!(constant::role_of(RADIANCE_PROGRAM) | constant::role_of(AIR_PROGRAM), 0);
        assert_eq!((KEEPS | LIGHTS_AIR) & (constant::RESOLVES | constant::ENDS_REFLECTIONS | constant::DRAWS_SKY), 0);
        assert_eq!(KEEPS & LIGHTS_AIR, 0);
        assert_eq!(role_of(b"..sr_denoising:CS_BilateralBlur_e877ef6e_Wrapped.."), BLURS_REFLECTIONS);
        assert_eq!(role_of(constant::REFLECTED), 0);
        assert_eq!(BLURS_REFLECTIONS & (KEEPS | LIGHTS_AIR | constant::RESOLVES | constant::ENDS_REFLECTIONS | constant::DRAWS_SKY), 0);
        assert_eq!(constant::role_of(BLUR_PROGRAM), 0);
        assert_eq!(role_of(constant::PROGRAM) | role_of(constant::REFLECTED) | role_of(constant::SKY), 0);
    }

    /// The names a shader source's constant buffer declares, in order.
    fn declared(source: &str) -> Vec<String> {
        let buffer = source.split_once("cbuffer").map_or("", |(_, rest)| rest);
        let body = buffer.split_once('{').and_then(|(_, rest)| rest.split_once('}'));
        body.map_or("", |(body, _)| body)
            .split(';')
            .filter_map(|decl| decl.split_whitespace().nth(1).map(str::to_owned))
            .collect()
    }

    /// The constants go out in the order the shader declares them.
    #[test]
    fn constants_follow_the_shader() {
        let p = Pass {
            level: 0.8,
            shade: [1.0, 0.6, 0.1, 0.45],
            follow: 0.04,
            most: 4.0,
            size: [2560, 1440],
            flat: true,
            air: 3.0,
            slices: 64,
            jump: 3.0,
            fresh: 1.5,
            dt: 0.3,
            held: 0.5,
            mix: [0.25, 1.0, 0.5, 2.0],
            sky: [0.108, 0.05, 0.4],
            floor: [0.18, 0.17, 0.16],
            sky_size: [128, 64],
            reflect_least: 0.15,
        };
        let c = constants(&p);
        let floats: Vec<f32> = c[..7].iter().map(|&w| f32::from_bits(w)).collect();
        assert_eq!(floats, [0.8, 1.0, 0.6, 0.1, 0.45, 0.04, 4.0]);
        assert_eq!(c[7..12], [2560, 1440, 1, 3.0f32.to_bits(), 64]);
        assert_eq!(
            c[12..20].iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>(),
            [3.0, 1.5, 0.3, 0.5, 0.25, 1.0, 0.5, 2.0]
        );
        assert_eq!(
            c[20..26].iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>(),
            [0.108, 0.05, 0.4, 0.18, 0.17, 0.16]
        );
        assert_eq!(c[26..28], [128, 64]);
        assert_eq!(f32::from_bits(c[28]), 0.15);
        let declared = declared(SOURCE);
        assert_eq!(
            declared,
            ["level", "strength", "radius", "near_z", "tan_y", "follow", "most", "width", "height", "flat", "air", "slices", "jump", "fresh", "dt", "held", "mean_part", "turn", "least_part", "most_part", "sky_gain", "sky_least", "sky_power", "floor_r", "floor_g", "floor_b", "sky_width", "sky_height", "reflect_least"]
        );
        assert_eq!(declared.len(), CONSTANTS);
    }

    /// What Wine's HLSL compiler (vkd3d-shader) does not take is not in the
    /// source, and every pass has its `main`.
    #[test]
    fn the_source_keeps_to_what_wine_compiles() {
        for refused in [
            "RootSignature",
            "GetDimensions",
            "NonUniformResourceIndex",
            "[]",
            "Interlocked",
            "isnan",
            "isinf",
            ".Load4",
            ".Load(uint",
        ] {
            assert!(!SOURCE.contains(refused), "{refused}");
        }
        // No store through a structured view: `cells[..] =`, `points[..] =`.
        for line in SOURCE.lines() {
            let line = line.trim_start();
            for buffer in ["cells[", "points["] {
                if let Some(rest) = line.strip_prefix(buffer) {
                    let after = rest.split_once(']').map_or("", |(_, after)| after).trim_start();
                    assert!(!after.starts_with('=') || after.starts_with("=="), "{line}");
                }
            }
        }
        assert!(SOURCE.is_ascii());
        assert_eq!(SOURCE.matches("void main(").count(), 7);
        for pass in [KEEP, MEAN, PLACE, GATHER, RAISE, AIR_PASS, HOLD] {
            let source = part(pass);
            assert!(source.starts_with(&format!("#define {pass}\n")), "{pass}");
            assert!(SOURCE.contains(&format!("#ifdef {pass}\n")), "{pass}");
            // The helpers a pass calls are defined for it.
            let body = SOURCE.split_once(&format!("#ifdef {pass}\n")).unwrap().1.split_once("#endif").unwrap().0;
            assert_eq!(body.contains("seen("), source.contains("#define SEES\n"), "{pass}");
            assert_eq!(body.contains("pack("), source.contains("#define PACKS\n"), "{pass}");
        }
        assert_eq!((CELLS, CELL_BYTES, POINT_BYTES, FLOOR_AT), (2304, 73_760, 691_200, 4609));
        assert!(SOURCE.contains("#define FLOOR_AT (2 * CELLS + 1)\n"));
        assert!(SOURCE.contains("#define CELLS 2304\n") && SOURCE.contains("#define PX 160\n"));
    }

    #[test]
    fn the_start_line_says_what_is_on() {
        let line = start_line(1.0, [1.0, 0.6, 0.1, 0.45], 0.0, 0.0, None, 0.5, MIX, 0.3);
        assert!(line.contains("shaded from depth; no texel gets less than the floor, 0.3 of the constant stand-in's light (VICTORY_AMBIENT_FLOOR); what the title's"), "{line}");
        assert!(line.starts_with("VICTORY_AMBIENT=estimate: "), "{line}");
        assert!(line.contains("(1 of it; 64x36 cells, 160x90 points; a texel is 0.2 the slow mean, turned 1.2 by its normal, between 0.2 and 2.5 times the mean around it), shaded from depth"), "{line}");
        assert!(line.contains("what the title's resolve left is replaced; reflections held to 0.5 of the light around; what the air scatters is left alone; "), "{line}");
        assert!(line.ends_with("the mean of the light around goes 0.04 of the way a frame"), "{line}");
        let line = start_line(0.5, [0.0, 0.6, 0.1, 0.45], 4.0, 3.0, settle(Some("250,3")), 0.0, MIX, 0.0);
        assert!(line.contains("not shaded; no floor under it; "), "{line}");
        assert!(line.contains("(0.5 of it;") && line.contains("not shaded"), "{line}");
        assert!(line.contains("counts up to 4 times it; reflections left alone; "), "{line}");
        assert!(line.contains("the air scatters no more than 3 times the light around (VICTORY_AMBIENT_AIR); "), "{line}");
        assert!(line.ends_with("settles by the clock (VICTORY_AMBIENT_SETTLE: 250 ms, a change by more than 3 times at once, 0 s fresh)"), "{line}");
    }

    #[test]
    fn the_frame_line_says_what_the_estimate_is() {
        let seen = Seen { mean: [0.2, 0.1, 0.05], floor: [0.25; 3], points: [0.3, 0.01, 2.0], written: 0.31, under: 0.6 };
        let line = frame_line(0.5, 0.3, Some(seen), [1.0, 2.0, 0.0, 1.0], None);
        assert!(line.starts_with("ambient estimate (VICTORY_AMBIENT=estimate): the light around (slow mean) is 0.2000 0.1000 0.0500, luminance 0.1177;"), "{line}");
        assert!(line.contains("160x90 points hold a luminance of 0.3000 on average (least 0.0100, most 2.0000)"), "{line}");
        assert!(line.contains("the floor (VICTORY_AMBIENT_FLOOR=0.3) is 0.2500, and 60% of the points are under it; texels are written about 0.3100 (the share 0.5 of the estimate, or the floor;"), "{line}");
        assert!(line.ends_with("; 1.0 keepings, 2.0 raisings, 1.0 holds of reflections and 0.0 bounds of the air/frame"), "{line}");
        let line = frame_line(1.0, 0.3, None, [0.0; 4], None);
        assert!(line.contains("nothing kept yet: no dispatch of CS_CacheScreenSpaceRadiance seen"), "{line}");
        // With the probe, the line says what the frames above it lacked.
        let line = frame_line(0.5, 0.3, Some(seen), [1.0, 0.0, 0.0, 1.0], Some(5));
        assert!(line.ends_with("bounds of the air/frame; probe (VICTORY_AMBIENT_PROBE) step 5 of 7: these frames were recorded without the raising with its gather (the title's own light image stays)"), "{line}");
        // The points' luminance: mean, least and most, without what is no
        // number; what is written is the share of each or the floor.
        let points = [[1.0f32; 3], [3.0; 3], [f32::NAN, 0.0, 0.0], [2.0; 3]];
        let ([mean, least, most], written, under) = over_the_picture(points.into_iter(), 0.5, 0.75);
        assert!(near(mean, 2.0, 1e-5) && near(least, 1.0, 1e-5) && near(most, 3.0, 1e-5));
        assert!(near(written, (0.75 + 1.5 + 1.0) / 3.0, 1e-5) && near(under, 1.0 / 3.0, 1e-5), "{written} {under}");
        assert_eq!(over_the_picture(std::iter::empty(), 1.0, 0.0), ([0.0; 3], 0.0, 0.0));
    }

    fn near(a: f32, b: f32, part: f32) -> bool {
        (a - b).abs() <= part * a.abs().max(b.abs()).max(1e-6)
    }

    /// A cell's light is the mean of its texels; a very bright texel counts
    /// for no more than about 20, and a NaN, an infinity or a negative one
    /// for nothing.
    #[test]
    fn cells_hold_the_mean_of_their_texels() {
        let size = [1280, 720];
        let grey = keep(&|_, _| [0.5, 0.25, 0.125], size);
        assert_eq!(grey.len(), CELLS as usize);
        assert!(grey.iter().all(|c| near(c[0], 0.5, 1e-5) && near(c[1], 0.25, 1e-5) && near(c[2], 0.125, 1e-5)));
        // The left half lit: cells keep to their own texels.
        let halves = keep(&|x, _| if x < 640 { [1.0; 3] } else { [0.0; 3] }, size);
        assert!(near(halves[0][0], 1.0, 1e-5) && halves[(CX - 1) as usize][0] == 0.0);
        assert!(near(halves[(CX / 2 - 1) as usize][0], 1.0, 1e-5) && halves[(CX / 2) as usize][0] == 0.0);
        // The sun's disc in one texel of a cell's 25.
        let glint = keep(&|x, y| if (x, y) == (2, 2) { [60_000.0; 3] } else { [0.0; 3] }, size);
        assert!(near(glint[0][0], 20.0 / 25.0, 1e-3), "{}", glint[0][0]);
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let cells = keep(&|_, _| [bad, 1.0, 1.0], size);
            assert!(cells.iter().all(|c| *c == [0.0; 3]), "{bad}");
        }
        // Every texel sampled is inside the image, whatever its size.
        for size in [[1280, 720], [64, 36], [1, 1], [1921, 1081]] {
            keep(&|x, y| { assert!(x < size[0] && y < size[1], "{x} {y}"); [0.0; 3] }, size);
        }
    }

    /// The slow mean starts at the first keeping's mean and then goes a
    /// share of the way at each.
    #[test]
    fn the_slow_mean_follows_by_a_share() {
        let dark = vec![[0.0f32; 3]; CELLS as usize];
        let lit = vec![[2.0f32, 1.0, 0.5]; CELLS as usize];
        let first = mean(&lit, [0.0; 4], FOLLOW, 0.0, 0.1);
        assert!(near(first[0], 2.0, 1e-5) && near(first[2], 0.5, 1e-5) && first[3] > 0.0);
        let next = mean(&dark, first, FOLLOW, 0.0, 0.1);
        assert!(near(next[0], 2.0 * 0.96, 1e-5), "{next:?}");
        // 25 keepings: to 1/e of the way, about.
        let mut m = first;
        for _ in 0..25 {
            m = mean(&dark, m, FOLLOW, 0.0, 0.1);
        }
        assert!(near(m[0], 2.0 * 0.96f32.powi(25), 1e-4) && m[0] < 2.0 * 0.37);
    }

    /// Settling by the clock: the same share of the way a second whatever
    /// the frame rate, a change by more than the jump factor at once, and
    /// the seconds since the last jump counted.
    #[test]
    fn the_slow_mean_settles_by_the_clock() {
        assert_eq!(settle(None), Some(Settle { ms: 250.0, jump: 3.0, fresh: 0.0 }));
        assert_eq!(settle(Some("0")), None);
        assert_eq!(settle(Some("")), None);
        assert_eq!(settle(Some("x,3")), None);
        assert_eq!(settle(Some("250")), Some(Settle { ms: 250.0, jump: 3.0, fresh: 0.0 }));
        assert_eq!(settle(Some(" 500 , 0 , 1.5 ")), Some(Settle { ms: 500.0, jump: 0.0, fresh: 1.5 }));
        assert_eq!(settle(Some("250,-1,x")), Some(Settle { ms: 250.0, jump: 3.0, fresh: 0.0 }));
        // A keeping's time: the first and after a pause as a frame of 33 ms.
        assert_eq!([frame_ms(None), frame_ms(Some(0)), frame_ms(Some(300)), frame_ms(Some(1000)), frame_ms(Some(1001))], [33.0, 1.0, 300.0, 1000.0, 33.0]);
        // Not by the clock: 0.04 whatever the time.
        assert_eq!(follow(None, 300.0), FOLLOW);
        // By the clock, three frames of 100 ms go as far as one of 300 ms.
        let by = settle(Some("250"));
        let (short, long) = (follow(by, 100.0), follow(by, 300.0));
        assert!(near(1.0 - (1.0 - short).powi(3), long, 1e-5) && near(long, 1.0 - (-1.2f32).exp(), 1e-6), "{short} {long}");
        // At this line's 3 frames a second, 0.04 a frame takes 25 frames
        // (8 s) for what 250 ms does in one.
        assert!(long > 0.69 && FOLLOW * 17.0 < long);
        let lit = vec![[1.0f32; 3]; CELLS as usize];
        let dim = vec![[0.5f32; 3]; CELLS as usize];
        let dark = vec![[0.1f32; 3]; CELLS as usize];
        let first = mean(&lit, [0.0; 4], long, 3.0, 0.3);
        assert!(near(first[0], 1.0, 1e-5) && near(first[3], 0.001, 1e-5));
        // Half the light: followed, and the time since the jump runs.
        let next = mean(&dim, first, long, 3.0, 0.3);
        assert!(near(next[0], 1.0 - 0.5 * long, 1e-5) && near(next[3], 0.301, 1e-4), "{next:?}");
        // A tenth of it (more than 3 times less): taken at once.
        let jumped = mean(&dark, next, long, 3.0, 0.3);
        assert!(near(jumped[0], 0.1, 1e-3) && near(jumped[3], 0.001, 1e-5), "{jumped:?}");
        // And as much more.
        assert!(near(mean(&lit, jumped, long, 3.0, 0.3)[0], 1.0, 1e-5));
        // No jump factor: followed.
        assert!(near(mean(&dark, next, long, 0.0, 0.3)[0], next[0] + (0.1 - next[0]) * long, 1e-3));
    }

    /// A wall straight ahead at `metres`.
    fn wall(metres: f32) -> impl Fn(u32, u32) -> f32 {
        move |_, _| 0.1 / metres
    }

    const SET: Pass = Pass {
        level: 1.0,
        shade: [1.0, 0.6, 0.1, 0.45],
        follow: FOLLOW,
        most: 0.0,
        size: [640, 360],
        flat: false,
        air: 0.0,
        slices: 0,
        jump: 0.0,
        fresh: 0.0,
        dt: 0.0,
        held: 0.5,
        mix: MIX,
        sky: [0.0, 0.0, 1.0],
        floor: [0.0; 3],
        sky_size: [0, 0],
        reflect_least: 0.0,
    };

    /// Texels are placed by their depth: the picture's middle straight
    /// ahead, its edges where the field of view puts them.
    #[test]
    fn texels_are_placed_by_depth() {
        let depth = wall(4.0);
        let scene = Scene { depth: &depth, size: [640, 360], near: 0.1, tan_y: 0.45 };
        let top_left = scene.seen([0, 0]);
        assert!(near(top_left[2], 4.0, 1e-5));
        assert!(near(top_left[0], -(1.0 - 1.0 / 640.0) * 0.45 * 640.0 / 360.0 * 4.0, 1e-4), "{top_left:?}");
        assert!(near(top_left[1], (1.0 - 1.0 / 360.0) * 0.45 * 4.0, 1e-4), "{top_left:?}");
        // Outside the picture: the depth of the nearest texel.
        assert!(near(scene.seen([-5, 400])[2], 4.0, 1e-5));
        // The sky (depth 0) is far, not infinitely.
        let sky = |_: u32, _: u32| 0.0;
        assert_eq!(Scene { depth: &sky, size: [640, 360], near: 0.1, tan_y: 0.45 }.seen([320, 180])[2], 1000.0);
        let places = scene.place();
        assert_eq!(places.len(), CELLS as usize);
        assert!(places.iter().all(|c| near(c[2], 4.0, 1e-5) && near(c[3], 16.0, 1e-5)));
        assert!(places[0][0] < 0.0 && places[0][1] > 0.0 && places[(CELLS - 1) as usize][0] > 0.0);
    }

    /// Where every cell sends out the same light, every point is given that
    /// light from no direction, and every texel is written it (the wall is
    /// open, so nothing is shaded).
    #[test]
    fn even_light_stays_even() {
        let depth = wall(4.0);
        let scene = Scene { depth: &depth, size: [640, 360], near: 0.1, tan_y: 0.45 };
        let light = vec![[0.5f32, 0.25, 0.125]; CELLS as usize];
        let points = scene.gather(&light, &scene.place());
        assert_eq!(points.len(), (PX * PY) as usize);
        for p in &points {
            for (colour, want) in [0.5, 0.25, 0.125].into_iter().enumerate() {
                assert!(near(p[colour][3], want, 1e-4), "{p:?}");
                assert!(p[colour][..3].iter().all(|d| d.abs() < 1e-4 * want.max(1.0)), "{p:?}");
            }
        }
        for p in [[0, 0], [320, 180], [639, 359], [17, 301]] {
            let (n, open) = scene.surface(p, 1.0, 0.6);
            assert!(n[2] < -0.999 && open == 1.0, "{n:?} {open}");
            let written = scene.raise(p, &points, [0.5, 0.25, 0.125, 1.0], &SET, [0.0; 3]);
            assert!(near(written[0], 0.5, 1e-3) && near(written[2], 0.125, 1e-3), "{written:?}");
            // A share of it.
            let half = scene.raise(p, &points, [0.5, 0.25, 0.125, 1.0], &Pass { level: 0.5, ..SET }, [0.0; 3]);
            assert!(near(half[0], 0.25, 1e-3), "{half:?}");
        }
    }

    /// A lit half of the wall lights its own side more than the other, and
    /// each point knows which way the light comes from.
    #[test]
    fn light_comes_from_where_it_is() {
        let depth = wall(4.0);
        let scene = Scene { depth: &depth, size: [640, 360], near: 0.1, tan_y: 0.45 };
        let light: Vec<V3> = (0..CELLS).map(|i| if i % CX < CX / 2 { [1.0, 0.0, 0.0] } else { [0.0; 3] }).collect();
        let points = scene.gather(&light, &scene.place());
        let at = |x: u32, y: u32| points[(y * PX + x) as usize];
        let (left, right) = (at(10, 45), at(150, 45));
        assert!(left[0][3] > 0.85 && right[0][3] < 0.15, "{} {}", left[0][3], right[0][3]);
        // Between them: half, and from the left (x right, so less than 0).
        let middle = at(80, 45);
        assert!(middle[0][3] > 0.3 && middle[0][3] < 0.7 && middle[0][0] < -0.01, "{middle:?}");
        // No green or blue anywhere.
        assert!(points.iter().all(|p| p[1] == [0.0; 4] && p[2] == [0.0; 4]));
        // Texels in between are given what lies between their points, and
        // never less than a fifth nor more than 2.5 times the mean there.
        let around = [0.5, 0.0, 0.0, 1.0];
        let dark = scene.raise([600, 180], &points, around, &SET, [0.0; 3])[0];
        let lit = scene.raise([40, 180], &points, around, &SET, [0.0; 3])[0];
        assert!(lit > 0.75 && lit < 1.0 && dark >= 0.1 - 1e-4 && dark < 0.25, "{lit} {dark}");
        // The mix: all of it the slow mean; no turn and no bounds; and the
        // bounds alone.
        let mixed = |mix: [f32; 4], p: [i32; 2]| scene.raise(p, &points, around, &Pass { mix, ..SET }, [0.0; 3])[0];
        assert!(near(mixed([1.0, 1.2, 0.2, 2.5], [600, 180]), 0.5, 1e-4));
        let plain = mixed([0.0, 0.0, 0.0, 9.0], [600, 180]);
        assert!(plain < dark && plain > 0.0 && near(mixed([0.0, 0.0, 2.0, 9.0], [600, 180]), 2.0 * plain, 1e-3), "{plain}");
        // Without a depth image: the slow mean alone.
        let flat = scene.raise([40, 180], &points, around, &Pass { flat: true, ..SET }, [0.0; 3]);
        assert_eq!(flat, [0.5, 0.0, 0.0]);
    }

    /// The crease between a floor and a wall is darker than the open floor.
    #[test]
    fn creases_are_shaded() {
        // A floor 0.6 m under the eye up to a wall 3 m ahead.
        let size = [640u32, 360];
        let depth = move |_: u32, y: u32| {
            let down = ((y as f32 + 0.5) / size[1] as f32 * 2.0 - 1.0) * 0.45;
            let z = if down > 0.0 { (0.6 / down).min(3.0) } else { 3.0 };
            0.1 / z
        };
        let scene = Scene { depth: &depth, size, near: 0.1, tan_y: 0.45 };
        // The row where the floor meets the wall: down * 3 = 0.6.
        let crease = ((0.2 / 0.45 + 1.0) / 2.0 * 360.0) as i32;
        let (_, at_crease) = scene.surface([320, crease + 3], 1.0, 0.6);
        let (_, in_the_open) = scene.surface([320, 350], 1.0, 0.6);
        assert!(at_crease < 0.9 && in_the_open > 0.99, "{at_crease} {in_the_open}");
        // No strength, no shade.
        assert_eq!(scene.surface([320, crease + 3], 0.0, 0.6).1, 1.0);
    }

    /// With a multiple set, what the resolve left counts up to that many
    /// times the estimate's luminance, and never lowers the estimate.
    #[test]
    fn what_the_resolve_left_counts_up_to_a_multiple() {
        let depth = wall(4.0);
        let scene = Scene { depth: &depth, size: [640, 360], near: 0.1, tan_y: 0.45 };
        let light = vec![[0.5f32; 3]; CELLS as usize];
        let points = scene.gather(&light, &scene.place());
        let set = Pass { most: 4.0, ..SET };
        let raise = |was: V3| scene.raise([320, 180], &points, [0.5, 0.5, 0.5, 1.0], &set, was);
        assert!(near(raise([0.0; 3])[0], 0.5, 1e-3));
        assert!(near(raise([1.0; 3])[0], 1.0, 1e-3));
        assert!(near(raise([100.0; 3])[0], 2.0, 1e-3));
        // A coloured leftover keeps its colour, at four times the luminance.
        let magenta = raise([50.0, 0.0, 50.0]);
        assert!(near(luma(magenta), luma([0.5; 3]) * 4.0, 0.3) && magenta[1] >= 0.5 - 1e-3, "{magenta:?}");
        // Replaced when the multiple is 0.
        assert!(near(scene.raise([320, 180], &points, [0.5, 0.5, 0.5, 1.0], &SET, [100.0; 3])[0], 0.5, 1e-3));
        // For the fresh seconds after the light around jumped, what was left
        // counts for no more than the estimate, then rises to the multiple.
        let fresh = Pass { fresh: 2.0, ..set };
        let after = |age: f32| scene.raise([320, 180], &points, [0.5, 0.5, 0.5, age], &fresh, [100.0; 3])[0];
        assert!(near(after(0.001), 0.5, 2e-3) && near(after(1.0), 1.25, 1e-3) && near(after(2.0), 2.0, 1e-3) && near(after(9.0), 2.0, 1e-3));
    }

    /// The light is packed as R9G9B9E5 and comes back within a 512th of its
    /// brightest part.
    #[test]
    fn the_light_is_packed_as_r9g9b9e5() {
        for c in [[0.0f32; 3], [1.0, 0.5, 0.25], [0.01, 0.02, 0.03], [300.0, 2.0, 0.1], [65408.0, 1.0, 0.0], [1e-5, 0.0, 0.0]] {
            let back = unpack(pack(c));
            let top = c[0].max(c[1]).max(c[2]);
            for k in 0..3 {
                assert!((back[k] - c[k]).abs() <= top / 256.0 + 2e-5, "{c:?} {back:?}");
            }
        }
        assert_eq!(pack([0.0; 3]) & 0x07ff_ffff, 0);
        assert_eq!(unpack(pack([1.0; 3])), [1.0; 3]);
    }
}
