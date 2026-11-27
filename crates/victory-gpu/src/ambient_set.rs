//! What the stand-in for ray-traced indirect light (`crate::ambient`) is
//! told by `VICTORY_AMBIENT` and its siblings, which of the title's
//! programs it follows, and its pass's shader and constants. Plain std, so
//! its tests run everywhere.

// Used by `crate::ambient`, which is only built for Windows.
#![cfg_attr(not(windows), allow(dead_code))]

/// Whether `VICTORY_AMBIENT`'s value turns the stand-in on: it is set, and
/// is not empty, `0` or `off`. Off by default on this line. (In Belobug's
/// fork the variable is the light's level and the stand-in is on unless it
/// says `off`; the level is `VICTORY_AMBIENT_LEVEL` here.)
pub(crate) fn is_on(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off"))
}

/// 0.6 looked balanced in the fork's run a2.
pub(crate) const LEVEL: f32 = 0.6;

/// `VICTORY_AMBIENT_LEVEL=<level>`: the light's level
/// while it does not follow the sky ([`LEVEL`] by default; 0 is no indirect
/// diffuse light at all).
pub(crate) fn level(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|l: &f32| l.is_finite() && *l >= 0.0)
        .unwrap_or(LEVEL)
}

/// `VICTORY_AMBIENT_TINT=r,g,b`: the light's colour; warm white otherwise.
pub(crate) fn tint(value: Option<&str>) -> [f32; 3] {
    let set: Vec<f32> = value
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    match set[..] {
        [r, g, b] => [r, g, b],
        _ => [1.0, 0.98, 0.94],
    }
}

/// `VICTORY_AMBIENT_SHADE=s,r,n,t`: how much less of the light reaches
/// creases and corners, worked out from the depth image alone: strength (0
/// for none), the radius looked at in metres, the near plane's distance,
/// and the tangent of half the vertical field of view (the last two are
/// guesses the picture is not sensitive to). Values left out keep their
/// defaults.
pub(crate) fn shade(value: Option<&str>) -> [f32; 4] {
    let mut shade = [1.0, 0.6, 0.1, 0.45];
    overlay(&mut shade, value);
    shade
}

/// `VICTORY_AMBIENT_REFLECT=<part>`: the most a reflection may hold, as a
/// part of the ambient light's level (0.5 by default; 0 leaves the title's
/// reflections alone).
pub(crate) fn keep(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|k: &f32| k.is_finite() && *k >= 0.0)
        .unwrap_or(0.5)
}

/// `VICTORY_AMBIENT_SKY`: gain, least level and power of the light that
/// follows the sky (`gain * mean^power + least`); `off` keeps the constant
/// level. On unless `off`; `gain,least,power` sets them, values left out
/// keep their defaults.
pub(crate) fn follow(value: Option<&str>) -> Option<[f32; 3]> {
    if value.is_some_and(|v| v.trim().eq_ignore_ascii_case("off")) {
        return None;
    }
    let mut follow = [0.36, 0.05, 0.4];
    overlay(&mut follow, value);
    Some(follow)
}

/// Replaces the first values of `to` by those of the comma list `value`
/// that read as numbers, each in its place.
fn overlay(to: &mut [f32], value: Option<&str>) {
    for (to, v) in to.iter_mut().zip(value.unwrap_or_default().split(',')) {
        if let Ok(v) = v.trim().parse() {
            *to = v;
        }
    }
}

/// The program whose output the stand-in replaces, by the start of its name
/// (shader containers hold their program's name), and where it writes the
/// indirect diffuse light (R9G9B9E5 as 32-bit integers) and reads the
/// scene's depth (reversed, D32 with stencil): root parameter and place in
/// its table, in dwords. Fixed for this title's shader build; all six
/// programs of that name in the title's shader database have both places
/// (`notes/ambient.md`).
pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);

/// The program that writes the title's reflections for the last time in a
/// frame, by the start of its name, and where it writes them (R9G9B9E5 as
/// 32-bit integers).
pub(crate) const REFLECTED: &[u8] = b"CS_UpdateHistory";
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

/// The program that draws the sky as the title reflects it, by the start of
/// its name: a small image of the whole sky, in the light's own units.
pub(crate) const SKY: &[u8] = b"ps_sky_paraboloid_reflection";

/// What a program is to the stand-in (bits; 0: nothing).
pub(crate) const RESOLVES: u8 = 1;
pub(crate) const ENDS_REFLECTIONS: u8 = 2;
pub(crate) const DRAWS_SKY: u8 = 4;

/// What the program of shader container `container` is to the stand-in.
pub(crate) fn role_of(container: &[u8]) -> u8 {
    let mut role = 0;
    if holds(container, PROGRAM) {
        role |= RESOLVES;
    }
    if holds(container, REFLECTED) {
        role |= ENDS_REFLECTIONS;
    }
    if holds(container, SKY) {
        role |= DRAWS_SKY;
    }
    role
}

/// What the stand-in calls a role in the log.
pub(crate) fn role_name(role: u8) -> &'static str {
    match role {
        RESOLVES => "resolves the indirect diffuse light",
        ENDS_REFLECTIONS => "writes the reflections last",
        DRAWS_SKY => "draws the sky dome",
        _ => "several",
    }
}

/// Whether `container` holds the bytes `name`.
fn holds(container: &[u8], name: &[u8]) -> bool {
    let Some((first, rest)) = name.split_first() else {
        return false;
    };
    let mut from = 0;
    while let Some(at) = container[from..].iter().position(|b| b == first) {
        from += at + 1;
        if container[from..].starts_with(rest) {
            return true;
        }
    }
    false
}

/// The pass. From the fork's shader, with what Wine's HLSL compiler
/// (vkd3d-shader 1.18 in CrossOver 26.3; `docs/macos.md`) needs changed:
/// - no `[RootSignature]`: Wine ignores it and writes no root signature
///   into the bytecode, so `crate::ambient` builds it in code
///   (`crate::compute::pipeline`): 17 constants in `b0`, then a table of
///   `u0`, `t0`, `t1`;
/// - no `GetDimensions`: Wine has no such method on unordered views
///   (`RWTexture2D`), so the image's size and the sky image's are constants;
/// - the constant buffer is all scalars (four to a row under any packing
///   rule), and conversions are written as casts, as in `crate::twin`'s
///   shader, which this Wine compiles;
/// - the shading is behind an `if`, not a conditional expression (both
///   sides of which are evaluated: 21 depth reads a texel for nothing, and
///   from a null view when there is no depth image); for reflections the
///   light from around is worked out once a texel, not twice.
pub(crate) const SOURCE: &str = r#"
cbuffer Floor : register(b0) {
    float level; float tint_r; float tint_g; float tint_b;
    float strength; float radius; float near_z; float tan_y;
    float gain; float least; float held; float floor_part;
    float power; uint width; uint height; uint sky_width;
    uint sky_height;
};
Texture2DArray<float4> sky : register(t1);
// The light, from the sky's mean when it follows the sky (gain above 0).
float3 from_around() {
    float3 tint = float3(tint_r, tint_g, tint_b);
    if (gain <= 0.0) return tint * level;
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            // The sun's own texels are held down: it lights by itself.
            float3 there = sky.Load(int4((x * 2 + 1) * sky_width / (N * 2), (y * 2 + 1) * sky_height / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / (N * N);
    float m = dot(mean, float3(0.2126, 0.7152, 0.0722));
    // The sky's colour, half washed out (the ground lights too); a dark
    // sky has none worth following.
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    return hue * (gain * pow(max(m, 0.0), power) + least);
}
RWTexture2D<uint> light : register(u0);
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), (int2)size - 1), 0));
    float z = near_z / max(d, 1e-7);
    float2 ndc = ((float2)p + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float jitter = frac(52.9829189 * frac(dot((float2)p, float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + jitter * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + (int2)round(o), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2((float)(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp((int)floor(log2(max(most, 1e-9))) + 16, 0, 31);
    float scale = exp2((float)(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2((float)(24 - e)); }
    uint3 m = min((uint3)(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | ((uint)e << 27);
}
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    if (at.x >= width || at.y >= height) return;
    float3 around = from_around();
    // What the resolve left is not kept: without rays its inputs are never
    // written and it holds whatever their memory did (bright magenta on
    // translucent surfaces, blocks of colours indoors: the fork's run v6,
    // 2026-11-25).
    if (held > 0.0) {
        // Reflections: no brighter than a part of the light around.
        const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
        float3 seen_there = unpack(light[at.xy]);
        float most = dot(around, LUMA) * held;
        float has = dot(seen_there, LUMA);
        float3 floor_light = around * floor_part;
        if (has > most) light[at.xy] = pack(seen_there * (most / has));
        else if (has < dot(floor_light, LUMA)) light[at.xy] = pack(floor_light);
        return;
    }
    float open = 1.0;
    if (strength > 0.0) open = open_to_light((int2)at.xy, float2(width, height));
    light[at.xy] = pack(around * open);
}
"#;

/// What one pass is given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pass {
    pub(crate) level: f32,
    pub(crate) tint: [f32; 3],
    /// Strength (0 without a depth image), radius, near plane, tangent.
    pub(crate) shade: [f32; 4],
    /// Gain, least level and power when the light follows the sky; a gain
    /// of 0 keeps the constant level.
    pub(crate) follow: [f32; 3],
    /// Above 0: the image holds reflections, held to this part of the
    /// light; 0: it is the indirect diffuse light, replaced.
    pub(crate) held: f32,
    /// The least a reflection holds, as a part of the light around.
    pub(crate) floor: f32,
    /// The image's size, and the sky image's.
    pub(crate) size: [u32; 2],
    pub(crate) sky: [u32; 2],
}

/// Dwords of the pass's constant buffer.
pub(crate) const CONSTANTS: usize = 17;

/// The pass's root constants, in the order its constant buffer declares
/// them (all scalars, so four to a row whatever the compiler's packing).
pub(crate) fn constants(p: &Pass) -> [u32; CONSTANTS] {
    let [strength, radius, near, tan_y] = p.shade;
    let [gain, least, power] = p.follow;
    let floats = [
        p.level, p.tint[0], p.tint[1], p.tint[2], strength, radius, near, tan_y, gain, least,
        p.held, p.floor, power,
    ];
    let mut out = [0; CONSTANTS];
    for (to, f) in out.iter_mut().zip(floats) {
        *to = f.to_bits();
    }
    out[13..].copy_from_slice(&[p.size[0], p.size[1], p.sky[0], p.sky[1]]);
    out
}

/// The line logged once at start when the stand-in is on.
pub(crate) fn start_line(
    level: f32,
    shade: [f32; 4],
    keep: f32,
    follow: Option<[f32; 3]>,
) -> String {
    let light = match follow {
        Some([gain, least, power]) => format!(
            "an ambient light that follows the sky dome (gain {gain}, least {least}, power \
             {power}; level {level} until the dome is drawn)"
        ),
        None => format!("a constant ambient light of level {level}"),
    };
    let shaded = if shade[0] > 0.0 { "shaded from depth" } else { "not shaded" };
    let reflections = if keep > 0.0 {
        format!("reflections held to {keep} of it")
    } else {
        "reflections left alone".to_owned()
    };
    format!(
        "VICTORY_AMBIENT: the screen probes' indirect diffuse light is replaced by {light}, \
         {shaded}; {reflections}"
    )
}

/// What `VICTORY_AMBIENT_FILE`'s text sets in a running game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FromFile {
    /// Two, three, six or seven numbers: gain, least, a power if their
    /// count is odd (0.4 otherwise), then the shading's four.
    Follow { follow: [f32; 3], shade: Option<[f32; 4]> },
    /// `level r g b`, then the shading's four, then the part reflections
    /// are held to, then the least they hold (0 if left out).
    Constant { level: f32, tint: [f32; 3], shade: Option<[f32; 4]>, reflections: Option<[f32; 2]> },
}

/// Reads the numbers of `VICTORY_AMBIENT_FILE`'s text; `None` if they are
/// too few.
pub(crate) fn from_file(text: &str) -> Option<FromFile> {
    let read: Vec<f32> = text.split_whitespace().filter_map(|v| v.parse().ok()).collect();
    if let [gain, least, ref rest @ ..] = read[..]
        && matches!(rest.len(), 0 | 1 | 4 | 5)
    {
        let (power, rest) = match rest.len() % 2 {
            1 => (rest[0], &rest[1..]),
            _ => (0.4, rest),
        };
        let shade = match rest[..] {
            [s, r, n, t] => Some([s, r, n, t]),
            _ => None,
        };
        return Some(FromFile::Follow { follow: [gain, least, power], shade });
    }
    let [level, r, g, b, ref rest @ ..] = read[..] else {
        return None;
    };
    let (shade, reflections) = match rest[..] {
        [s, r, n, t, ref last @ ..] => (
            Some([s, r, n, t]),
            last.first().map(|k| [k.max(0.0), last.get(1).map_or(0.0, |f| f.max(0.0))]),
        ),
        _ => (None, None),
    };
    Some(FromFile::Constant { level: level.max(0.0), tint: [r, g, b], shade, reflections })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off unless set to something other than nothing, `0` or `off`.
    #[test]
    fn off_by_default() {
        for off in [None, Some(""), Some("  "), Some("0"), Some(" 0 "), Some("off"), Some("OFF")] {
            assert!(!is_on(off), "{off:?}");
        }
        for on in ["1", "on", "yes", "0.6", "2"] {
            assert!(is_on(Some(on)), "{on:?}");
        }
    }

    #[test]
    fn levels_and_colours_keep_their_defaults_when_unreadable() {
        assert_eq!(level(None), LEVEL);
        assert_eq!(level(Some("x")), LEVEL);
        assert_eq!(level(Some("-1")), LEVEL);
        assert_eq!(level(Some("inf")), LEVEL);
        assert_eq!(level(Some(" 0 ")), 0.0);
        assert_eq!(level(Some("1.4")), 1.4);
        assert_eq!(tint(None), [1.0, 0.98, 0.94]);
        assert_eq!(tint(Some("1,2")), [1.0, 0.98, 0.94]);
        assert_eq!(tint(Some("0.5, 0.25,1")), [0.5, 0.25, 1.0]);
        assert_eq!(shade(None), [1.0, 0.6, 0.1, 0.45]);
        assert_eq!(shade(Some("0")), [0.0, 0.6, 0.1, 0.45]);
        assert_eq!(shade(Some("0.5,x,0.2")), [0.5, 0.6, 0.2, 0.45]);
        assert_eq!(keep(None), 0.5);
        assert_eq!(keep(Some("-2")), 0.5);
        assert_eq!(keep(Some("0")), 0.0);
        assert_eq!(follow(None), Some([0.36, 0.05, 0.4]));
        assert_eq!(follow(Some(" off ")), None);
        assert_eq!(follow(Some("0.5")), Some([0.5, 0.05, 0.4]));
        assert_eq!(follow(Some("0.5,0.1,0.3")), Some([0.5, 0.1, 0.3]));
    }

    /// Programs are known by the start of their name, anywhere in the
    /// container; other programs are nothing to the stand-in.
    #[test]
    fn programs_are_known_by_name() {
        let container = |name: &str| {
            let mut c = vec![0u8; 300];
            c.extend_from_slice(name.as_bytes());
            c.extend_from_slice(b"_Wrapped\0\x01\x02");
            c
        };
        assert_eq!(
            role_of(&container("CS_ScreenProbes_TemporalResolve_ReflectiveCaustics_a911b49c")),
            RESOLVES
        );
        assert_eq!(role_of(&container("CS_ScreenProbes_TemporalResolve_CB_a1c765aa")), RESOLVES);
        assert_eq!(role_of(&container("CS_UpdateHistory_e877ef6e")), ENDS_REFLECTIONS);
        assert_eq!(role_of(&container("ps_sky_paraboloid_reflection_c3dfdbd9")), DRAWS_SKY);
        assert_eq!(role_of(&container("CS_ScreenProbes_ActivateRays_d60186e6")), 0);
        assert_eq!(role_of(&container("CS_ScreenProbes_TemporalResolv")), 0);
        assert_eq!(role_of(b""), 0);
        // A name cut off by the container's end, and one after a false start.
        assert_eq!(role_of(&PROGRAM[..PROGRAM.len() - 1]), 0);
        assert_eq!(role_of(b"CS_CS_ScreenProbes_TemporalResolve"), RESOLVES);
        assert!(holds(b"aab", b"ab") && !holds(b"aab", b"ba") && !holds(b"ab", b""));
    }

    /// The constants go out in the order the shader declares them, the
    /// sizes after the thirteen floats.
    #[test]
    fn constants_follow_the_shader() {
        let p = Pass {
            level: 0.6,
            tint: [1.0, 0.98, 0.94],
            shade: [1.0, 0.6, 0.1, 0.45],
            follow: [0.36, 0.05, 0.4],
            held: 0.5,
            floor: 0.25,
            size: [1920, 1080],
            sky: [128, 64],
        };
        let c = constants(&p);
        let floats: Vec<f32> = c[..13].iter().map(|&w| f32::from_bits(w)).collect();
        assert_eq!(
            floats,
            [0.6, 1.0, 0.98, 0.94, 1.0, 0.6, 0.1, 0.45, 0.36, 0.05, 0.5, 0.25, 0.4]
        );
        assert_eq!(c[13..], [1920, 1080, 128, 64]);
        // What the shader source declares, one name per dword.
        let declared = declared(SOURCE);
        assert_eq!(declared.len(), CONSTANTS);
        assert_eq!(declared[0], "level");
        assert_eq!(declared[10], "held");
        assert_eq!(declared[12], "power");
        assert_eq!(declared[13..], ["width", "height", "sky_width", "sky_height"]);
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

    /// What Wine's HLSL compiler (vkd3d-shader) does not take is not in the
    /// source: a root signature attribute (ignored there, so the signature
    /// is built in code), `GetDimensions` on an unordered view (it has no
    /// such method; sizes are constants), resource arrays.
    #[test]
    fn the_source_keeps_to_what_wine_compiles() {
        for refused in ["RootSignature", "GetDimensions", "NonUniformResourceIndex", "[]"] {
            assert!(!SOURCE.contains(refused), "{refused}");
        }
        for needed in ["register(b0)", "register(u0)", "register(t0)", "register(t1)", "numthreads(8, 8, 1)"] {
            assert!(SOURCE.contains(needed), "{needed}");
        }
        assert!(SOURCE.is_ascii());
    }

    #[test]
    fn the_start_line_says_what_is_on() {
        let line = start_line(0.6, [1.0, 0.6, 0.1, 0.45], 0.5, Some([0.36, 0.05, 0.4]));
        assert!(line.starts_with("VICTORY_AMBIENT: "), "{line}");
        assert!(line.contains("follows the sky dome (gain 0.36, least 0.05, power 0.4"), "{line}");
        assert!(line.ends_with("shaded from depth; reflections held to 0.5 of it"), "{line}");
        let plain = start_line(1.4, [0.0, 0.6, 0.1, 0.45], 0.0, None);
        assert!(
            plain.contains("a constant ambient light of level 1.4, not shaded; reflections left alone"),
            "{plain}"
        );
    }

    /// The file's numbers: two, three, six or seven follow the sky; four or
    /// more of another count set a constant light.
    #[test]
    fn the_file_sets_what_its_count_of_numbers_says() {
        assert_eq!(from_file(""), None);
        assert_eq!(from_file("0.5"), None);
        assert_eq!(
            from_file("0.4 0.02"),
            Some(FromFile::Follow { follow: [0.4, 0.02, 0.4], shade: None })
        );
        assert_eq!(
            from_file("0.4 0.02 0.5\n"),
            Some(FromFile::Follow { follow: [0.4, 0.02, 0.5], shade: None })
        );
        assert_eq!(
            from_file("0.4 0.02 1 0.6 0.1 0.45"),
            Some(FromFile::Follow { follow: [0.4, 0.02, 0.4], shade: Some([1.0, 0.6, 0.1, 0.45]) })
        );
        assert_eq!(
            from_file("0.4 0.02 0.5 1 0.6 0.1 0.45"),
            Some(FromFile::Follow { follow: [0.4, 0.02, 0.5], shade: Some([1.0, 0.6, 0.1, 0.45]) })
        );
        assert_eq!(
            from_file("0.3 1 0.9 0.8"),
            Some(FromFile::Constant { level: 0.3, tint: [1.0, 0.9, 0.8], shade: None, reflections: None })
        );
        assert_eq!(
            from_file("-1 1 1 1 0 0.6 0.1 0.45 0.7 -3"),
            Some(FromFile::Constant {
                level: 0.0,
                tint: [1.0, 1.0, 1.0],
                shade: Some([0.0, 0.6, 0.1, 0.45]),
                reflections: Some([0.7, 0.0]),
            })
        );
        assert_eq!(
            from_file("0.3 1 1 1 1 0.6 0.1 0.45 0.7"),
            Some(FromFile::Constant {
                level: 0.3,
                tint: [1.0, 1.0, 1.0],
                shade: Some([1.0, 0.6, 0.1, 0.45]),
                reflections: Some([0.7, 0.0]),
            })
        );
    }
}
