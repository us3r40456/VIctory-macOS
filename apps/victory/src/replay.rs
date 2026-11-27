//! `victory replay`: a captured frame's GPU work run again without the
//! title (docs/replay.md).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use victory_gpu::replay::look;
#[cfg(windows)]
use victory_gpu::replay::play::{self, Detail, Options, Waits};

#[derive(Clone, Copy, ValueEnum)]
enum Timing {
    /// Each submission.
    Submissions,
    /// Each command too: by timestamps where the host's tell commands
    /// apart, else in pieces.
    Detailed,
    /// Each command by a timestamp after it.
    Stamps,
    /// Each command by sending the frame in pieces, each waited for.
    Pieces,
}

#[derive(Args)]
pub(crate) struct Replay {
    /// The folder given as VICTORY_RECORD (its last capture) or a capture.
    folder: PathBuf,
    /// How often the frame is run.
    #[arg(long, default_value_t = 5)]
    loops: u32,
    /// Do not put back, between runs, what a run changed.
    #[arg(long)]
    no_restore: bool,
    #[arg(long, value_enum, default_value_t = Timing::Submissions)]
    timing: Timing,
    /// Pipelines listed with detailed timing.
    #[arg(long, default_value_t = 20)]
    top: usize,
    /// Where the picture goes (default: replay.png in the capture).
    #[arg(long)]
    image: Option<PathBuf>,
    /// Run the frame unchanged too, turn by turn, and print both.
    #[arg(long)]
    compare: bool,
    /// Send all submissions at once and wait once at the end (the runtime,
    /// and a replay without this, wait for each).
    #[arg(long, conflicts_with = "one_submit")]
    no_waits: bool,
    /// Send each list alone and wait for it, and name the heaviest lists.
    #[arg(long, conflicts_with_all = ["one_submit", "no_waits"])]
    wait_lists: bool,
    /// After the report (with --timing detailed): for so many of the
    /// costliest pipelines, run the frame with and without each, turn by
    /// turn, and say what leaving it out saves.
    #[arg(long, default_value_t = 0)]
    leave_out: usize,
    /// Leave out the barriers of buffers (their transitions, and waits for
    /// unordered access to them).
    #[arg(long)]
    no_buffer_barriers: bool,
    /// Leave out all barriers (no valid D3D12; D3DMetal runs it).
    #[arg(long, conflicts_with = "no_buffer_barriers")]
    no_barriers: bool,
    /// Leave out the host layer's argument clamps (the passes, what binds
    /// them and their barriers): the calls they serve read arguments kept
    /// from a first run with them.
    #[arg(long)]
    skip_clamps: bool,
    /// Make every scissor rectangle one texel: draws without their pixels.
    #[arg(long)]
    no_pixels: bool,
    /// Send nothing that works: only what the lists bind, and their barriers.
    #[arg(long)]
    binds_only: bool,
    /// Make every dispatch one of one thread group.
    #[arg(long)]
    one_group: bool,
    /// After this run, another of the same capture (loaded once) with these
    /// switches instead of this run's experiments, as one text: `--also
    /// "--pace 5 --loops 7"`. May be given several times.
    #[arg(long, allow_hyphen_values = true)]
    also: Vec<String>,
    /// One ExecuteCommandLists for the whole frame.
    #[arg(long)]
    one_submit: bool,
    /// Milliseconds the GPU stands idle after each submission, as it does
    /// while a title prepares the next.
    #[arg(long, default_value_t = 0.0)]
    pace: f64,
    /// Leave out what runs a pipeline whose name or DXIL hash has this in it
    /// (may be given several times).
    #[arg(long = "skip-shader")]
    skip_shader: Vec<String>,
    /// Take shaders from this folder where it has them (`<hash>.dxil`, by
    /// the hash the runtime names the title's shader by; see --retranslate).
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// With --compare: the folder the second run's shaders are from.
    #[arg(long)]
    shaders_b: Option<PathBuf>,
    /// Only translate the capture's shaders again, into this folder (for
    /// --shaders), with this build and the VICTORY_* switches of this run.
    /// The title's containers are those the recorder kept beside the capture.
    #[arg(long)]
    retranslate: Option<PathBuf>,
    /// Only write the frame's resources and the commands of these lists
    /// (`3..7`, or `5`). Needs no GPU, and runs on any host.
    #[arg(long)]
    listing: Option<String>,
    /// Only write a summary of the frame: its resources, what each
    /// submission does, which pipelines run how often and into what, the
    /// host layer's own passes. Needs no GPU, and runs on any host.
    #[arg(long)]
    summary: bool,
    /// A file of lines `<hash> <name>`: the names of the title's shaders, by
    /// the hash the runtime logs them by, for the reports.
    #[arg(long)]
    names: Option<PathBuf>,
    /// Use the D3D12 debug layer, where there is one (slow).
    #[arg(long)]
    debug_layer: bool,
}

/// `3..7` (both included) or `5`.
fn range(text: &str) -> Option<(u32, u32)> {
    match text.split_once("..") {
        Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
        None => text.trim().parse().ok().map(|n| (n, n)),
    }
}

pub(crate) fn run(r: Replay) -> Result<()> {
    if let Some(text) = &r.listing {
        let lists = range(text).with_context(|| format!("--listing {text}"))?;
        return look::listing(&r.folder, lists, &mut |line| println!("{line}"));
    }
    if r.summary {
        let names = look::Names::read(r.names.as_deref())?;
        return look::summary(&r.folder, &names, r.top, &mut |line| println!("{line}"));
    }
    #[cfg(not(windows))]
    anyhow::bail!("a frame is run again on a Windows host (on a Mac: tools/macos/replay.sh); --listing and --summary work here");
    #[cfg(windows)]
    replay(r)
}

#[cfg(windows)]
fn replay(r: Replay) -> Result<()> {
    // What a replay writes is made of the title's data, as the capture is.
    for out in r.retranslate.iter().chain(&r.image) {
        victory_core::ensure_outside_repo(out)?;
    }
    if let Some(into) = &r.retranslate {
        let done = play::retranslate(&r.folder, into)?;
        for line in &done.refused {
            println!("not translated: {line}");
        }
        println!(
            "{} of the capture's {} pipelines of the title's had their containers; {} shaders written to {} in {:.1} s, {} of them the capture's own bit for bit",
            done.found,
            done.pipelines,
            done.written,
            into.display(),
            done.seconds,
            done.same
        );
        if done.found < done.pipelines {
            println!(
                "containers are kept in `containers` beside the capture by a run with VICTORY_RECORD set: those of shaders the title passed before that run's recorder was asked for are missing"
            );
        }
        return Ok(());
    }
    // The capture is loaded once; every run after the first is of the same
    // one, with its own switches.
    let mut again = Vec::new();
    for text in &r.also {
        let words = std::iter::once("replay".to_owned())
            .chain(std::iter::once(r.folder.display().to_string()))
            .chain(words(text));
        let mut more = <Again as clap::Parser>::try_parse_from(words)
            .with_context(|| format!("--also {text}"))?
            .replay;
        more.names = r.names.clone();
        again.push((text.clone(), options(more)));
    }
    let folder = r.folder.clone();
    let first = options(r);
    let mut player = play::Player::load(&folder, &first)?;
    let mut run = |what: Option<&str>, o: &Options| -> Result<()> {
        if let Some(what) = what {
            println!("== {what}");
        }
        let reports = play::replay_with(&mut player, o)?;
        for (what, report) in &reports {
            println!("{}", report.text(what));
        }
        if let [(_, before), (_, after)] = &reports[..] {
            println!(
                "{:+.2} ms ({:+.1} %)",
                after.median_ms() - before.median_ms(),
                (after.median_ms() / before.median_ms().max(1e-9) - 1.0) * 100.0
            );
        }
        Ok(())
    };
    run(None, &first)?;
    for (what, o) in &again {
        run(Some(what), o)?;
    }
    Ok(())
}

/// One more run of a capture: the switches of `--also`.
#[cfg(windows)]
#[derive(clap::Parser)]
struct Again {
    #[command(flatten)]
    replay: Replay,
}

/// The words of a text, those in double quotes as one.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let (mut quoted, mut any) = (false, false);
    for c in text.chars() {
        match c {
            '"' => (quoted, any) = (!quoted, true),
            c if c.is_whitespace() && !quoted => {
                if any {
                    out.push(std::mem::take(&mut word));
                }
                any = false;
            }
            c => {
                word.push(c);
                any = true;
            }
        }
    }
    if any {
        out.push(word);
    }
    out
}

/// What a run does, of its switches.
#[cfg(windows)]
fn options(r: Replay) -> Options {
    Options {
        loops: r.loops.max(1),
        restore: !r.no_restore,
        detail: match r.timing {
            // What is left out is chosen by the detailed times.
            Timing::Submissions if r.leave_out > 0 => Detail::Auto,
            Timing::Submissions => Detail::Off,
            Timing::Detailed => Detail::Auto,
            Timing::Stamps => Detail::Stamps,
            Timing::Pieces => Detail::Pieces,
        },
        top: r.top,
        image: r.image,
        waits: match (r.no_waits, r.one_submit, r.wait_lists) {
            (_, true, _) => Waits::OneSubmit,
            (true, _, _) => Waits::None,
            (_, _, true) => Waits::Lists,
            _ => Waits::Each,
        },
        names: r.names,
        leave_out: r.leave_out,
        keep_clamps: r.skip_clamps,
        one_texel: r.no_pixels,
        binds_only: r.binds_only,
        one_group: r.one_group,
        barriers: match (r.no_barriers, r.no_buffer_barriers) {
            (true, _) => play::Barriers::NoneAtAll,
            (_, true) => play::Barriers::NoBufferOnes,
            _ => play::Barriers::AsRecorded,
        },
        pace: r.pace,
        skip_shaders: r.skip_shader,
        shaders: r.shaders,
        shaders_b: r.shaders_b,
        compare: r.compare,
        debug: r.debug_layer,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranges() {
        assert_eq!(super::range("3..7"), Some((3, 7)));
        assert_eq!(super::range("5"), Some((5, 5)));
        assert_eq!(super::range("x"), None);
        assert_eq!(super::range("1..x"), None);
        assert_eq!(
            super::words(r#" --pace 5  --skip-shader "victory: clamp" --compare"#),
            ["--pace", "5", "--skip-shader", "victory: clamp", "--compare"]
        );
        assert!(super::words("  ").is_empty());
    }
}
