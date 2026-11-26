//! The visible window caption: the title's text followed by our statistics.

const SEPARATOR: &str = " | VIctory ";

/// Replaces the suffix left by an earlier presenter. A replacement swap
/// chain reads the visible caption, which already contains that suffix.
pub(crate) fn with_stats(caption: &[u16], stats: &str) -> Vec<u16> {
    let separator: Vec<u16> = SEPARATOR.encode_utf16().collect();
    let base_len = caption
        .windows(separator.len())
        .enumerate()
        .rev()
        .find(|(at, part)| {
            **part == separator[..] && looks_like_stats(&caption[*at + separator.len()..])
        })
        .map_or(caption.len(), |(at, _)| at);
    caption[..base_len]
        .iter()
        .copied()
        .chain(separator)
        .chain(stats.encode_utf16())
        .collect()
}

/// Whether `text` has the exact shape Meter appends. The separator alone is
/// not enough: a title's own caption may contain ` | VIctory `.
fn looks_like_stats(text: &[u16]) -> bool {
    let Ok(text) = String::from_utf16(text) else {
        return false;
    };
    let mut words = text.split_whitespace();
    words.next().is_some_and(|v| v.parse::<u64>().is_ok())
        && words.next() == Some("fps")
        && words.next().is_some_and(|v| v.parse::<f64>().is_ok())
        && words.next() == Some("ms/frame")
        && words.next().is_some_and(|v| v.parse::<u64>().is_ok())
        && words.next() == Some("draws/frame")
        && words.next().is_none()
}

#[cfg(test)]
mod tests {
    use super::with_stats;

    fn text(wide: &[u16]) -> String {
        String::from_utf16_lossy(wide)
    }

    #[test]
    fn statistics_replace_the_previous_suffix_and_preserve_the_base_title() {
        let base = "Synthetic base | VIctory base | window";
        let mut caption: Vec<u16> = base.encode_utf16().collect();

        caption = with_stats(&caption, "0 fps 2973.1 ms/frame 4456 draws/frame");
        assert_eq!(
            text(&caption),
            "Synthetic base | VIctory base | window | VIctory 0 fps 2973.1 ms/frame 4456 draws/frame"
        );

        // A replacement presenter reads the caption left by the previous one.
        caption = with_stats(&caption, "1 fps 1425.4 ms/frame 4715 draws/frame");
        assert_eq!(
            text(&caption),
            "Synthetic base | VIctory base | window | VIctory 1 fps 1425.4 ms/frame 4715 draws/frame"
        );
    }
}
