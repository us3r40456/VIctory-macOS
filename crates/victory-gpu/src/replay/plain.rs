//! What the recorder and the replayer need that is no D3D12: pages told
//! apart, pictures made of a presented texture's rows and compared.

/// A number of a page's bytes, to tell whether the page changed: zero for
/// a page of zeros, of any length. Two pages that differ have the same one
/// once in 2^64 times.
pub(super) fn page_hash(bytes: &[u8]) -> u64 {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x27d4_eb2f_1656_67c5,
    ];
    let mut lanes = [0u64; 4];
    let mut blocks = bytes.chunks_exact(32);
    for block in &mut blocks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let mut word = [0u8; 8];
            word.copy_from_slice(&block[i * 8..i * 8 + 8]);
            *lane = (*lane ^ u64::from_le_bytes(word))
                .wrapping_mul(K[i])
                .rotate_left(29);
        }
    }
    for (i, rest) in blocks.remainder().chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..rest.len()].copy_from_slice(rest);
        lanes[i] = (lanes[i] ^ u64::from_le_bytes(word))
            .wrapping_mul(K[i])
            .rotate_left(29);
    }
    let mut h = lanes[0]
        ^ lanes[1].rotate_left(17)
        ^ lanes[2].rotate_left(31)
        ^ lanes[3].rotate_left(47);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

/// When a frame is captured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum At {
    /// Seconds after the device was made.
    Seconds(f64),
    /// The frame of a number of presents.
    Frame(u64),
    /// The frame a number of presents after the world's first frame (the
    /// first that ran [`WORLD_DRAWS`] draws on the host), as
    /// `VICTORY_CAPTURE_WORLD` counts them.
    World(u64),
    /// When a file named `go` appears in the folder (this works in the
    /// other cases too).
    Trigger,
}

/// The draws of the world's first frame, at the least (`crate::capture`).
pub const WORLD_DRAWS: u64 = 500;

/// Frames before the frame asked for from which a capture is prepared (and
/// lists are recorded, where they are not from the start): a title records
/// lists ahead of the frame that submits them.
pub(super) const LEAD: u64 = 8;

/// Frames waited after the frame asked for, at the most, for one whose
/// predecessor's lists were all recorded.
pub(super) const PATIENCE: u64 = 4;

/// `VICTORY_RECORD_AT`: seconds, `f<frame>`, or `w<frames after the
/// world's first>`; anything else waits for the file.
pub(super) fn parse_at(text: &str) -> At {
    let text = text.trim();
    if let Some(frame) = text.strip_prefix('f').and_then(|n| n.parse().ok()) {
        return At::Frame(frame);
    }
    if let Some(frame) = text.strip_prefix('w').and_then(|n| n.parse().ok()) {
        return At::World(frame);
    }
    match text.parse::<f64>() {
        Ok(s) => At::Seconds(s),
        Err(_) => At::Trigger,
    }
}

/// Whether a capture is prepared from present `presents` on, `seconds`
/// after the device was made, the world's first frame being `world` (0: not
/// seen yet): the frame asked for (0: the first that can be captured).
pub(super) fn due(at: At, presents: u64, world: u64, seconds: f64) -> Option<u64> {
    match at {
        At::Seconds(s) => (seconds >= s).then_some(0),
        At::Frame(f) => (presents + LEAD >= f).then_some(f),
        At::World(k) => (world != 0 && presents + LEAD >= world + k).then_some(world + k),
        At::Trigger => None,
    }
}

/// Whether the frame after present `presents` is the one captured: the
/// capture is prepared since present `since`, `target` is the frame asked
/// for (0: none), and `whole` says whether all lists of the frame just
/// presented were recorded.
pub(super) fn captures_next(presents: u64, since: u64, target: u64, whole: bool) -> bool {
    // Two frames of lists at the least, and not before the frame asked for.
    let first = target.max(since + 3);
    presents + 1 >= first && (whole || presents + 1 >= first + PATIENCE)
}

/// The middle one of `v` in order (the upper of two); 0 of none.
pub(super) fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// A PNG of 8-bit RGB pixels, row after row.
pub(super) fn png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend(kind);
        png.extend(data);
        let mut crc = !0u32;
        for &b in &png[start..] {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
            }
        }
        png.extend((!crc).to_be_bytes());
    }
    let row = width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
    for y in 0..height as usize {
        // Filter 0: the row as it is.
        raw.push(0);
        match rgb.get(y * row..(y + 1) * row) {
            Some(pixels) => raw.extend_from_slice(pixels),
            None => raw.resize(raw.len() + row, 0),
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 1),
    );
    chunk(&mut png, b"IEND", &[]);
    png
}

/// 8-bit RGB of a presented texture's rows (and the greatest channel value
/// of the format, for the signal-to-noise ratio). `None` for formats not
/// shown.
pub(super) fn rgb(
    width: u32,
    height: u32,
    pitch: u32,
    format: i32,
    rows: &[u8],
) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    let half = |b: &[u8]| {
        let h = u16::from_le_bytes([b[0], b[1]]);
        let (s, e, m) = (h >> 15, (h >> 10) & 31, f32::from(h & 1023));
        let v = match e {
            0 => m * 2f32.powi(-24),
            31 => 65536.0,
            e => (1.0 + m / 1024.0) * 2f32.powi(i32::from(e) - 15),
        };
        if s == 1 {
            0
        } else {
            (v.min(1.0) * 255.0) as u8
        }
    };
    for y in 0..height as usize {
        let row = rows.get(y * pitch as usize..)?;
        for x in 0..width as usize {
            match format {
                27..=32 => out.extend_from_slice(row.get(x * 4..x * 4 + 3)?),
                87 | 88 | 90..=93 => {
                    let p = row.get(x * 4..x * 4 + 3)?;
                    out.extend([p[2], p[1], p[0]]);
                }
                23..=25 => {
                    let p = row.get(x * 4..x * 4 + 4)?;
                    let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                    out.extend([(v >> 2) as u8, (v >> 12) as u8, (v >> 22) as u8]);
                }
                9..=14 => {
                    let p = row.get(x * 8..x * 8 + 6)?;
                    out.extend([half(p), half(&p[2..]), half(&p[4..])]);
                }
                _ => return None,
            }
        }
    }
    Some(out)
}

/// Peak signal-to-noise ratio of two pictures of 8-bit channels, in dB
/// (infinite if equal).
pub(super) fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)).pow(2))
        .sum();
    if sum == 0 && a.len() == b.len() {
        return f64::INFINITY;
    }
    let mean = sum as f64 / a.len().min(b.len()).max(1) as f64;
    10.0 * (255.0 * 255.0 / mean).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pictures() {
        assert_eq!(rgb(1, 1, 4, 28, &[1, 2, 3, 4]).unwrap(), [1, 2, 3]);
        assert_eq!(rgb(1, 1, 4, 87, &[1, 2, 3, 4]).unwrap(), [3, 2, 1]);
        let v = (1023u32 | 512 << 10).to_le_bytes();
        assert_eq!(rgb(1, 1, 4, 24, &v).unwrap(), [255, 128, 0]);
        // 1.0, 0.5 and -1.0 as halves.
        assert_eq!(
            rgb(1, 1, 8, 10, &[0, 0x3c, 0, 0x38, 0, 0xbc, 0, 0]).unwrap(),
            [255, 127, 0]
        );
        assert!(rgb(1, 1, 4, 41, &[0; 4]).is_none());
        assert!(rgb(2, 2, 8, 28, &[0; 12]).is_none(), "too few rows");
        assert_eq!(psnr(&[1, 2], &[1, 2]), f64::INFINITY);
        assert!((psnr(&[0; 4], &[255; 4])).abs() < 1e-9);
    }

    #[test]
    fn when_to_record() {
        assert_eq!(parse_at("120"), At::Seconds(120.0));
        assert_eq!(parse_at(" 1.5 "), At::Seconds(1.5));
        assert_eq!(parse_at("f900"), At::Frame(900));
        assert_eq!(parse_at("w70"), At::World(70));
        assert_eq!(parse_at(""), At::Trigger);
        assert_eq!(parse_at("later"), At::Trigger);
        assert_eq!(parse_at("w"), At::Trigger);
    }

    #[test]
    fn the_frame_asked_for_is_the_one_captured() {
        // 70 frames after the world's first frame, 1164: frame 1234.
        let at = At::World(70);
        assert_eq!(due(at, 5000, 0, 9.0), None, "the world's first frame was not seen");
        assert_eq!(due(at, 1164, 1164, 9.0), None);
        assert_eq!(due(at, 1234 - LEAD - 1, 1164, 9.0), None);
        let since = 1234 - LEAD;
        assert_eq!(due(at, since, 1164, 9.0), Some(1234));
        // Decided at the present before it.
        for presents in since..1233 {
            assert!(!captures_next(presents, since, 1234, true), "{presents}");
        }
        assert!(captures_next(1233, since, 1234, true));
        // A frame with lists that were not recorded puts it off, for so
        // many frames.
        for presents in 1233..1233 + PATIENCE {
            assert!(!captures_next(presents, since, 1234, false), "{presents}");
            assert!(captures_next(presents, since, 1234, true), "{presents}");
        }
        assert!(captures_next(1233 + PATIENCE, since, 1234, false));
        // By frame number; one that is past, or nearly, as soon as two
        // frames of lists are there.
        assert_eq!(due(At::Frame(900), 900 - LEAD, 0, 0.0), Some(900));
        assert_eq!(due(At::Frame(3), 1, 0, 0.0), Some(3));
        assert!(!captures_next(2, 1, 3, true));
        assert!(captures_next(3, 1, 3, true));
        // By time and by the file: the first frame that can be.
        assert_eq!(due(At::Seconds(5.0), 7, 0, 4.9), None);
        assert_eq!(due(At::Seconds(5.0), 7, 0, 5.0), Some(0));
        assert_eq!(due(At::Trigger, 7, 3, 99.0), None);
        assert!(!captures_next(8, 7, 0, true));
        assert!(captures_next(9, 7, 0, true));
        assert!(!captures_next(9 + PATIENCE - 1, 7, 0, false));
        assert!(captures_next(9 + PATIENCE, 7, 0, false));
    }

    #[test]
    fn medians() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[3.0]), 3.0);
        assert_eq!(median(&[9.0, 1.0, 5.0]), 5.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 3.0);
    }

    #[test]
    fn pages_are_told_apart() {
        for len in [0usize, 1, 7, 8, 31, 32, 33, 100, 4096] {
            let zeros = vec![0u8; len];
            assert_eq!(page_hash(&zeros), 0, "zeros of {len} bytes");
            // Any one byte, any one bit of it.
            for at in 0..len {
                for bit in [0, 3, 7] {
                    let mut page = zeros.clone();
                    page[at] = 1 << bit;
                    assert_ne!(page_hash(&page), 0, "byte {at} of {len}");
                }
            }
        }
        let page: Vec<u8> = (0..4096u32).map(|i| (i * 31 >> 3) as u8).collect();
        let mut seen = std::collections::HashSet::new();
        seen.insert(page_hash(&page));
        for at in 0..page.len() {
            let mut other = page.clone();
            other[at] ^= 0x10;
            assert!(seen.insert(page_hash(&other)), "byte {at}");
        }
        // Two bytes changed places.
        let mut other = page.clone();
        other.swap(64, 96);
        assert!(seen.insert(page_hash(&other)));
    }

    #[test]
    fn a_png_has_its_parts() {
        let png = png(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9]);
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(png[16..24], [0, 0, 0, 2, 0, 0, 0, 2]);
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
        let data = &png[41..png.len() - 16];
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(data).unwrap();
        assert_eq!(raw, [0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255, 9, 9, 9]);
    }
}
