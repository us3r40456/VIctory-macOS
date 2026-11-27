use std::path::Path;

use anyhow::{Context, Result};

use crate::{Console, Unimplemented};

#[cfg(any(windows, test))]
const SKIP_CPU_CHECK_ENV: &str = "VICTORY_SKIP_CPU_CHECK";

#[cfg(any(windows, test))]
fn skip_cpu_check(value: Option<&str>) -> bool {
    value == Some("1")
}

#[cfg(any(windows, test))]
fn check_cpu_features_with(has_avx2: bool, skip_env: Option<&str>) -> Result<()> {
    if has_avx2 || skip_cpu_check(skip_env) {
        return Ok(());
    }
    anyhow::bail!(
        "this CPU does not support AVX2, which Xbox Series titles require; \
         set {SKIP_CPU_CHECK_ENV}=1 to bypass this check (for example under CPU emulation such as Intel SDE)"
    );
}

/// Pre-flight check before executing title code: Xbox Series binaries are
/// compiled for Zen 2 and require AVX2.
#[cfg(windows)]
fn check_cpu_features() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = true;
    let skip = std::env::var(SKIP_CPU_CHECK_ENV).ok();
    check_cpu_features_with(has_avx2, skip.as_deref())
}

#[cfg(windows)]
pub fn run(
    exe: &Path,
    unimplemented: Unimplemented,
    strict: bool,
    execute: bool,
    console: Console,
) -> Result<u32> {
    use std::sync::Arc;
    use victory_hle::{HleResolver, HostPolicy};
    use victory_loader::{BindReport, UnimplementedPolicy, WithTitleDlls};

    if execute {
        check_cpu_features()?;
    }
    // A title that relaunched itself: the old process must be gone first.
    victory_hle::modules::lifecycle::wait_for_previous();
    if execute {
        // Before the host's own memory (the image read below, the GPU
        // device, the caches) takes the space the title's 32-bit memory needs.
        // macOS port: off under Rosetta unless VICTORY_LOW_GUARDS=1. With the guards in
        // place, a later address-limited allocation has made Wine search for free space
        // for minutes while it held its virtual-memory lock (three stalled runs on
        // 26 November, watchdog at +0x23cf683); the start-up failure the guards fix on
        // Windows has not been seen here.
        let low_guards = match std::env::var("VICTORY_LOW_GUARDS").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => !victory_gpu::rosetta(),
        };
        if low_guards {
            victory_hle::modules::xmem::low_guards::install();
        }
    }
    let cfg = victory_core::Config::load()?;
    // Before the import registry is first used: it holds the module only
    // when the host's FidelityFX serves.
    victory_hle::modules::fidelityfx::use_host_dir(cfg.fidelityfx_dir.as_deref());
    let original = std::fs::read(exe).with_context(|| format!("reading {}", exe.display()))?;
    let prepared = victory_loader::prepare(&original)?;
    let hash = prepared.original_hash.to_hex();
    let title_dir = exe.parent().unwrap_or(Path::new("."));
    let known = victory_game::identify(&hash);
    match known {
        Some(build) => {
            println!("build          {} ({hash})", build.label);
            if execute {
                check_startup_archive(&title_dir.join(build.startup_archive))?;
            }
        }
        None => println!("build          unknown ({hash})"),
    }
    // Rosetta cannot resume these when they fault partway (#14, #50).
    let masked = victory_loader::masked::masked_stores(&prepared.info, &original);
    println!("masked stores  {}", victory_loader::masked::summary(&masked));
    // Under Rosetta, tracking is off unless asked for, and the stores it
    // cannot resume are patched when it is on (`victory_gpu::fault_tracking`).
    println!(
        "fault tracking {}",
        if victory_gpu::fault_tracking() {
            "on"
        } else if victory_gpu::rosetta() {
            "off (title memory copied whole; VICTORY_FAULT_TRACKING=1 turns it on, which also patches the stores Rosetta cannot resume)"
        } else {
            "off (title memory copied whole)"
        }
    );
    // The file's bytes (the exe's size, 114 MB) are kept only to patch the
    // masked stores after the mapping.
    let needs_patch = victory_gpu::stores_need_patching();
    let mut requested = std::env::var("VICTORY_PATCH_MASKED").ok();
    // Rosetta can also fault midway through plain 256-bit vector stores in
    // a few loops (#144). Their offsets are specific to Series 1.0.2.1, so
    // they are added here only for that build, and an explicit
    // `VICTORY_PATCH_MASKED` always wins.
    if needs_patch
        && requested.is_none()
        && hash.as_str() == "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93"
    {
        requested = Some("all,1e09b31,1e09b37,259509f,25950a5,1e759f0,1e759f7,1e759fe,1e75a05".into());
    }
    let patching = victory_loader::masked::choice(requested, needs_patch);
    let original = patching.is_some().then_some(original);

    let images = cfg.cache_subdir("images")?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let image = victory_loader::load(&prepared, &images)?;
    // Mapped: the patched bytes (the exe's size, 114 MB) are not needed.
    drop(prepared);
    victory_hle::runtime::set_main_image(image.base());
    victory_hle::runtime::set_image_path(image.base(), exe);
    println!(
        "mapped at      {:#x} (preferred {:#x})",
        image.base(),
        image.info().image_base
    );
    // clwb faults under Rosetta, an exception per write-back loop (#135).
    // Only there, the same check fault tracking uses; `VICTORY_NOP_CLWB=0`
    // turns it off and `1` turns it on elsewhere.
    #[cfg(windows)]
    if match std::env::var("VICTORY_NOP_CLWB").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => victory_gpu::rosetta(),
    } {
        println!(
            "clwb loops     {} made NOPs",
            victory_loader::nop_clwb_loops(image.base() as u64, image.info())
        );
    } else {
        println!("clwb loops     left as they are");
    }
    // The title's hang watchdog off, for world loads from slow storage.
    #[cfg(windows)]
    if std::env::var_os("VICTORY_NO_HANG_WATCHDOG").is_some_and(|v| v != "0") {
        match known.map(|b| b.hang_watchdog).filter(|p| !p.is_empty()) {
            Some(patches) => {
                for p in patches {
                    if victory_loader::patch_code(
                        image.base() as u64,
                        p.rva,
                        p.original,
                        p.replacement,
                    ) {
                        println!("hang watchdog  off at +{:#x}", p.rva);
                    } else {
                        println!("hang watchdog  left on: unexpected bytes at +{:#x}", p.rva);
                    }
                }
            }
            None => println!("hang watchdog  left on: unknown build"),
        }
    }
    // The title's output size, render scale and upscaler (`VICTORY_OUTPUT`,
    // `VICTORY_RENDER_SCALE`, `VICTORY_UPSCALER`, `VICTORY_UI_SIZE`; each
    // off without it), ported from avatarmanz/VIctory-7900XTX.
    #[cfg(windows)]
    if execute {
        let size = image.info().size_of_image as usize;
        // SAFETY: the image stays mapped and nothing of it has run yet.
        unsafe { victory_loader::resolution::apply_asked(image.base(), size) };
    }
    // Masked stores redone as plain stores Rosetta can resume, so fault
    // tracking (`VICTORY_FAULT_TRACKING=1`) can run under Rosetta (#50).
    if let (Some(which), Some(original)) = (patching, original) {
        let wanted = victory_loader::masked::wanted_from(Some(&which));
        let n =
            victory_loader::masked::patch(image.base() as u64, image.info(), &original, &wanted);
        println!("masked stores  {n} patched to plain stores");
    }
    // Every other 256-bit store of the title split into 128-bit halves, from
    // a list made offline (`cargo xtask wide-stores`): Rosetta cannot resume
    // one that faults in its second page.
    #[cfg(windows)]
    if let Some(list) = std::env::var_os("VICTORY_PATCH_WIDE").filter(|v| !v.is_empty() && v != "0")
    {
        let (n, refused) =
            victory_loader::wide::patch(image.base() as u64, std::path::Path::new(&list));
        println!("wide stores    {n} split into 128-bit halves ({refused} refused)");
    }
    // macOS port: the title's stores of descriptor records tell the layer
    // the slots they write (victory_loader::descriptor_sites,
    // victory_gpu::heap_reports); off unless asked for.
    #[cfg(windows)]
    if victory_gpu::heap_reports::wanted() {
        use victory_loader::descriptor_sites::{MARKS_LEN, SITES, SWITCH, apply};
        const _: () = assert!(MARKS_LEN == victory_gpu::heap_reports::MARKS_LEN);
        let form = match victory_gpu::heap_reports::mode() {
            victory_gpu::heap_reports::Mode::Check => "check",
            _ => "1",
        };
        let size = image.info().size_of_image as usize;
        // SAFETY: the image stays mapped, nothing of it has run yet, and
        // the cells are a static.
        let done = unsafe { apply(image.base(), size, victory_gpu::heap_reports::marks_address(), &SITES) };
        match done {
            Ok(n) => println!(
                "heap reports   on ({SWITCH}={form}): {n} of {} descriptor record stores of the title tell the layer their slots; its memcpy and memmove are bound to tell too",
                SITES.len()
            ),
            Err(why) => println!(
                "heap reports   on ({SWITCH}={form}), but the title's descriptor record stores are not patched: {why}; heap syncs look at the written pages as before"
            ),
        }
        victory_gpu::heap_reports::hooked(image.base(), size, done.unwrap_or(0), SITES.len());
    }
    // macOS port: a way out of a title lookup that can loop forever
    // (victory_loader::probe_guard); off unless asked for.
    let guard_lookup = std::env::var("VICTORY_GUARD_LOOKUP").as_deref() == Ok("1");
    if guard_lookup {
        match victory_loader::probe_guard::patch(image.base() as u64) {
            Ok(()) => println!("lookup guard   on: the lookup at +0x50eb750 gives up when its key is absent"),
            Err(why) => println!("lookup guard   not applied: {why}"),
        }
    }
    // macOS port: the title's time budget for placing map pieces, made
    // larger (victory_loader::map_budget); off unless asked for.
    if let Some(k) = victory_loader::map_budget::wanted(std::env::var(victory_loader::map_budget::SWITCH).ok().as_deref())
    {
        match victory_loader::map_budget::patch(image.base() as u64, k) {
            Ok(()) => println!(
                "map budget     on: {} to {} us a frame for placing map pieces (the title's limits times {k})",
                250 * k,
                1250 * k
            ),
            Err(why) => println!("map budget     not applied: {why}"),
        }
    }
    // macOS port: the time a frame the title's streaming request list may
    // spend issuing requests, made larger (victory_loader::srl_slice); off
    // unless asked for.
    if let Some(k) = victory_loader::srl_slice::wanted(std::env::var(victory_loader::srl_slice::SWITCH).ok().as_deref())
    {
        match victory_loader::srl_slice::patch(image.base() as u64, k) {
            Ok(us) => println!(
                "request slice  on: {us} us a frame for a time-sliced request list (the title's 500; asked for times {k}, took times {})",
                us / 500
            ),
            Err(why) => println!("request slice  not applied: {why}"),
        }
    }
    // macOS port: the longest step the title's clamped timers take in a
    // frame, made longer (victory_loader::time_step); off unless asked for.
    if let Some(seconds) =
        victory_loader::time_step::wanted(std::env::var(victory_loader::time_step::SWITCH).ok().as_deref())
    {
        match victory_loader::time_step::patch(image.base() as u64, seconds) {
            Ok(step) => println!(
                "time step      on: the title's clamped timers step up to {step} s a frame (the title's 1/15 s; asked for {seconds})"
            ),
            Err(why) => println!("time step      not applied: {why}"),
        }
    }
    // macOS port: map pieces nobody asks for any more are put out when the
    // title's pool of buildings fills up (victory_loader::map_evict); off
    // unless asked for.
    if victory_loader::map_evict::wanted(std::env::var(victory_loader::map_evict::SWITCH).ok().as_deref()) {
        match victory_loader::map_evict::patch(image.base() as u64) {
            Ok(()) => println!(
                "map evict      on: pieces let go of, 1.5 km from the player for 10 s, go while the building pool is half full"
            ),
            Err(why) => println!("map evict      not applied: {why}"),
        }
    }
    // macOS port: on a Series S the title's upscaler choice returns its own
    // upscaler (type 6), not FSR 4, whose passes show nothing here
    // (victory_loader::upscaler_patch); `VICTORY_UPSCALER_PATCH=0` leaves it.
    if let Some(line) = victory_loader::upscaler_patch::at_start(
        matches!(console, Console::SeriesS),
        std::env::var(victory_loader::upscaler_patch::SWITCH).ok().as_deref(),
        &hash,
        // In the registry only when the host's FidelityFX serves.
        victory_hle::registry().iter().any(|m| m.serves("amd_fidelityfx_loader_xbox.dll")),
        image.base() as u64,
        image.info().size_of_image,
    ) {
        println!("upscaler       {line}");
    }
    let resolver = HleResolver::new(if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    });
    let wanted = |dll: &str| title_dll(&resolver.host, dll);
    // A missing manifest leaves id 0 and no storage quota.
    let manifest =
        std::fs::read_to_string(title_dir.join("MicrosoftGame.config")).unwrap_or_default();
    let title = victory_hle::runtime::Title::from_manifest(&manifest);
    let storage = cfg.data_subdir(&format!("{:08X}/storage", title.id))?;
    println!(
        "title          {:08X}, storage {} ({} MB)",
        title.id,
        storage.display(),
        title.storage_bytes >> 20
    );
    // The console's fixed drives: the package as `G:`, temporary files `T:`.
    victory_hle::modules::files::mount_at('G', title_dir);
    let temp = cfg.data_subdir(&format!("{:08X}/temp", title.id))?;
    victory_hle::modules::files::mount_at('T', &temp);
    victory_hle::runtime::set_title(victory_hle::runtime::Title { storage, ..title });
    victory_hle::console::set(match console {
        Console::SeriesX => victory_hle::console::Console::SeriesX,
        Console::SeriesS => victory_hle::console::Console::SeriesS,
    });
    println!("console        {}", victory_hle::console::current().name());
    let dlls = victory_loader::load_title_dlls(image.info(), title_dir, &images, &wanted)?;
    for (name, dll) in &dlls {
        println!("title dll      {name} at {:#x}", dll.base());
        victory_hle::runtime::set_image_path(dll.base(), &title_dir.join(name));
        victory_gpu::pipeline::note_title_dll(name, dll.base(), dll.info().size_of_image as usize);
    }
    let imported: Vec<&str> = image
        .info()
        .imports
        .iter()
        .map(|m| m.dll.as_str())
        .collect();
    let mapped: Vec<&str> = dlls.iter().map(|(name, _)| name.as_str()).collect();
    let absent = absent_title_dlls(&imported, &mapped, &wanted);
    if !absent.is_empty() {
        let message = format!(
            "{} not in {}, and neither the host nor VIctory provides {}: the package is \
             incomplete; copy every file of the package next to the executable",
            absent.join(", "),
            title_dir.display(),
            if absent.len() == 1 { "it" } else { "them" },
        );
        println!("missing dlls   {message}");
        // None of their imports can be bound, so the title stops at its first
        // call into one (Grand Theft Auto VI 1.0.2.1 at +0x2237178 during
        // System Init, #930) with nothing saying why. Stubs only if asked for.
        if execute && matches!(unimplemented, Unimplemented::Trap) {
            anyhow::bail!(
                "the title's own DLLs are missing: {message} (`--unimplemented zero` runs it anyway)"
            );
        }
    }

    let policy = match unimplemented {
        Unimplemented::Trap => UnimplementedPolicy::Trap,
        Unimplemented::Zero => UnimplementedPolicy::ReturnZero,
    };
    let with_dlls = WithTitleDlls {
        inner: &resolver,
        dlls: &dlls,
    };
    let show = |name: &str, report: BindReport| {
        println!(
            "bound          {name}: {} hle, {} host, {} title, {} missing",
            report.hle,
            report.host,
            report.title,
            report.missing.len()
        );
        for m in &report.missing {
            println!("  missing      {}!{}", m.dll, m.symbol);
        }
    };
    for (name, dll) in &dlls {
        show(name, dll.bind_imports(&with_dlls, policy)?);
    }
    let exe_name = exe.file_name().unwrap_or_default().to_string_lossy();
    show(&exe_name, image.bind_imports(&with_dlls, policy)?);
    if !execute {
        println!("not executing (pass --execute to start the title)");
        return Ok(0);
    }
    victory_loader::report_crashes();
    victory_loader::log_crashes_to(temp.with_file_name("crashes.log"));
    victory_hle::runtime::use_console_timers();
    if guard_lookup {
        victory_loader::probe_guard::watch();
    }
    // macOS port: the title's pools of buildings and objects in the log
    // (victory_loader::pool_watch); off unless asked for.
    if victory_loader::pool_watch::watch_wanted(
        std::env::var(victory_loader::pool_watch::WATCH_SWITCH).ok().as_deref(),
    ) {
        match victory_loader::pool_watch::watch(image.base() as u64) {
            Ok(()) => println!("pool watch     on: buildings and objects are logged every 5 s"),
            Err(why) => println!("pool watch     not started: {why}"),
        }
    }
    #[cfg(feature = "diagnostics")]
    diagnostics(&image, &cfg);
    let late = LateDlls {
        host: resolver.host.clone(),
        title_dir: title_dir.to_owned(),
        cache_dir: images,
        policy,
        mapped: std::sync::Mutex::new(dlls.clone()),
        lock: LoaderLock::default(),
    };
    victory_hle::runtime::set_title_dll_loader(Box::new(move |file| {
        late.load(file, show).map_err(|e| format!("{e:#}"))
    }));
    // macOS port: a sampling profile of the process's threads
    // (victory_gpu::profile); off unless VICTORY_PROFILE names a file.
    let titled = std::iter::once((exe_name.as_ref(), image.base(), image.info().size_of_image as usize)).chain(
        dlls.iter().map(|(name, dll)| (name.as_str(), dll.base(), dll.info().size_of_image as usize)),
    );
    if let Some(line) = victory_gpu::profile::start(titled) {
        println!("profile        {line}");
    }
    // SAFETY: imports are bound; from here on we run title code.
    let main = unsafe { Arc::new(image).spawn_main(dlls)? };
    let code = main
        .join()
        .map_err(|_| anyhow::anyhow!("title main thread panicked"))??;
    println!("title exited with {code}");
    Ok(code)
}

#[cfg(windows)]
fn check_startup_archive(path: &Path) -> Result<()> {
    use std::io::Read;

    // Some copies keep each package in a folder of its own name
    // (`appdata0/appdata0.rpf`); the title finds nothing there (#24).
    if !path.exists()
        && let Some(nested) = nested_package(path)
    {
        anyhow::bail!(
            "Cannot find startup archive {}, but {} exists. The archives must sit directly in the game folder, not one per subfolder. Move every package's files up into {}.",
            path.display(),
            nested.display(),
            path.parent().unwrap_or(Path::new(".")).display()
        );
    }
    let mut magic = [0; 4];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .with_context(|| {
            format!(
                "Cannot read startup archive {}. Check that the file from your package is present and readable.",
                path.display()
            )
        })?;
    anyhow::ensure!(
        magic == *b"9FPR",
        "Invalid RPF9 header in {}. Restore the startup archive from your package.",
        path.display()
    );
    Ok(())
}

/// `dir/<stem>/<name>` for `dir/<name>`, where a copy that keeps each
/// package in a folder of its own name has it.
#[cfg(windows)]
fn nested_package(path: &Path) -> Option<std::path::PathBuf> {
    let nested = path
        .parent()?
        .join(path.file_stem()?)
        .join(path.file_name()?);
    nested.is_file().then_some(nested)
}

/// The debugging aids (README): scheduled stack dumps, watched values and
/// memory saved on a crash.
#[cfg(all(windows, feature = "diagnostics"))]
fn diagnostics(image: &victory_loader::LoadedImage, cfg: &victory_core::Config) {
    victory_loader::dump_stacks_on_schedule();
    victory_hle::modules::audio::capture::start();
    watch(image.base(), image.info().size_of_image as usize);
    poke(image.base(), image.info().size_of_image as usize);
    // Debugging aid: `VICTORY_CRASH_SAVE=<rva>:<len>,...` (hex) writes those
    // title ranges to `crash-<rva>.bin` in the cache dir on a crash.
    let size = image.info().size_of_image as usize;
    let ranges = std::env::var("VICTORY_CRASH_SAVE").unwrap_or_default();
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    victory_loader::save_on_crash(
        ranges
            .split(',')
            .filter_map(|r| r.split_once(':'))
            .filter_map(|(rva, len)| Some((hex(rva)?, hex(len)?)))
            .filter(|&(rva, len)| rva.checked_add(len).is_some_and(|end| end <= size))
            .map(|(rva, len)| {
                let file = cfg.cache_dir.join(format!("crash-{rva:x}.bin"));
                (image.base() + rva, len, file)
            })
            .collect(),
    );
}

/// Debugging aid: `VICTORY_WATCH=<rva>,...` (hex) logs the u32 at each
/// title+rva whenever it changes, polled every 5 ms (an engine's state, say).
#[cfg(all(windows, feature = "diagnostics"))]
fn watch(base: usize, size: usize) {
    let Ok(list) = std::env::var("VICTORY_WATCH") else {
        return;
    };
    let rvas: Vec<usize> = list
        .split(',')
        .filter_map(|r| usize::from_str_radix(r.trim().trim_start_matches("0x"), 16).ok())
        .filter(|&rva| rva + 4 <= size)
        .collect();
    std::thread::spawn(move || {
        let mut seen = vec![None; rvas.len()];
        loop {
            for (&rva, seen) in rvas.iter().zip(&mut seen) {
                // SAFETY: four bytes inside the mapped image, checked above.
                let value = unsafe { ((base + rva) as *const u32).read_volatile() };
                if *seen != Some(value) {
                    eprintln!("VIctory: title+{rva:#x} = {value:#x}");
                    *seen = Some(value);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
}

/// Debugging aid: `VICTORY_POKE=<rva>:<byte>,...` (hex) writes the byte at
/// each title+rva twice a second. A title that ends itself when its main
/// loop has not ticked for a while (Grand Theft Auto VI: 30 s) runs on
/// through slower frames with its tick flag poked. The address must be
/// writable title data.
#[cfg(all(windows, feature = "diagnostics"))]
fn poke(base: usize, size: usize) {
    let Ok(list) = std::env::var("VICTORY_POKE") else {
        return;
    };
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    let pokes: Vec<(usize, u8)> = list
        .split(',')
        .filter_map(|p| p.split_once(':'))
        .filter_map(|(rva, byte)| Some((hex(rva)?, u8::try_from(hex(byte)?).ok()?)))
        .filter(|&(rva, _)| rva < size)
        .collect();
    std::thread::spawn(move || {
        loop {
            for &(rva, byte) in &pokes {
                // SAFETY: a byte inside the mapped image, checked above.
                unsafe { ((base + rva) as *mut u8).write_volatile(byte) };
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
}

/// DLLs the host or our runtime provide are never taken from the title.
#[cfg(windows)]
fn title_dll(host: &victory_hle::HostPolicy, dll: &str) -> bool {
    !host.allows(dll) && !victory_hle::registry().iter().any(|m| m.serves(dll))
}

/// The DLLs among `imported` (the main image's static imports) that must
/// come from the title's folder (`wanted`) but were not mapped from it:
/// none of their imports can be bound. Each named once, case ignored.
#[cfg(any(windows, test))]
fn absent_title_dlls(
    imported: &[&str],
    mapped: &[&str],
    wanted: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let mut absent: Vec<String> = Vec::new();
    for &dll in imported {
        let known = |list: &[&str]| list.iter().any(|m| m.eq_ignore_ascii_case(dll));
        if wanted(dll) && !known(mapped) && !absent.iter().any(|a| a.eq_ignore_ascii_case(dll)) {
            absent.push(dll.to_owned());
        }
    }
    absent
}

/// Title DLLs the title loads itself (`LoadLibrary`) that nothing imports,
/// e.g. Grand Theft Auto VI's upscaler behind its FidelityFX loader.
#[cfg(windows)]
struct LateDlls {
    host: victory_hle::HostPolicy,
    title_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    policy: victory_loader::UnimplementedPolicy,
    /// Every title DLL mapped so far, to bind new ones against.
    mapped: std::sync::Mutex<Vec<(String, victory_loader::LoadedImage)>>,
    lock: LoaderLock,
}

#[cfg(windows)]
impl LateDlls {
    /// Every `LoadLibrary` of the title comes here first. Returns a title
    /// image called `file`, mapping it and the title DLLs it needs first,
    /// binding them and running their `DllMain` on the calling thread, as
    /// the Windows loader would; `None` if `file` is no title image.
    fn load(
        &self,
        file: &str,
        show: impl Fn(&str, victory_loader::BindReport),
    ) -> Result<Option<usize>> {
        use victory_hle::runtime::{forget_image, image_named, set_image_path};
        // Waits while another thread initialises a DLL; a `DllMain` that
        // loads another DLL re-enters.
        self.lock.hold(|| {
            if let Some(base) = image_named(file) {
                return Ok(Some(base));
            }
            let new = {
                let mut mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
                let wanted = |dll: &str| title_dll(&self.host, dll) && image_named(dll).is_none();
                let new = victory_loader::load_title_dll(
                    file,
                    &self.title_dir,
                    &self.cache_dir,
                    &wanted,
                )?;
                let hle = victory_hle::HleResolver::new(self.host.clone());
                let old = victory_loader::WithTitleDlls {
                    inner: &hle,
                    dlls: &mapped,
                };
                let resolver = victory_loader::WithTitleDlls {
                    inner: &old,
                    dlls: &new,
                };
                for (name, dll) in &new {
                    println!(
                        "title dll      {name} at {:#x} (loaded by the title)",
                        dll.base()
                    );
                    show(name, dll.bind_imports(&resolver, self.policy)?);
                    set_image_path(dll.base(), &self.title_dir.join(name));
                    victory_gpu::pipeline::note_title_dll(
                        name,
                        dll.base(),
                        dll.info().size_of_image as usize,
                    );
                }
                // Visible before their `DllMain`s run, which may load more.
                mapped.extend(new.iter().cloned());
                new
            };
            let failed = new.iter().enumerate().find_map(|(i, (name, dll))| {
                // SAFETY: bound above, dependencies attached first.
                let attached = unsafe { dll.attach() };
                attached
                    .err()
                    .map(|e| (i, e.context(format!("attaching {name}"))))
            });
            let Some((i, error)) = failed else {
                return Ok(new.last().map(|(_, dll)| dll.base()));
            };
            // As `LoadLibrary` does when a `DllMain` fails: detach what this
            // load attached (the failed one too) and unmap all of it.
            let bases: Vec<_> = new.iter().map(|(_, d)| d.base()).collect();
            self.mapped
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(_, d)| !bases.contains(&d.base()));
            for (_, done) in new[..=i].iter().rev() {
                // SAFETY: attached (or failing to) on this thread just now.
                unsafe { done.detach() };
            }
            for (_, dll) in new {
                forget_image(dll.base());
                // SAFETY: detached, and no title code holds it yet.
                unsafe { dll.unload() };
            }
            Err(error)
        })
    }
}

/// The Windows loader lock's shape: one thread at a time, re-entered by
/// the thread that holds it.
#[cfg(windows)]
#[derive(Default)]
struct LoaderLock {
    owner: std::sync::Mutex<(Option<std::thread::ThreadId>, usize)>,
    free: std::sync::Condvar,
}

#[cfg(windows)]
impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

#[cfg(test)]
mod absent_tests {
    use super::absent_title_dlls;

    #[test]
    fn title_dlls_missing_from_the_folder_are_named_once() {
        // The host provides kernel32, our runtime d3d12_xs; the rest are the
        // title's own (#930: the four DLLs beside the executable missing).
        let wanted = |dll: &str| {
            !["KERNEL32.dll", "d3d12_xs.dll"]
                .iter()
                .any(|h| h.eq_ignore_ascii_case(dll))
        };
        let imported = [
            "KERNEL32.dll",
            "d3d12_xs.dll",
            "XboxBvhBuilder.dll",
            "MSVCP140.dll",
            "Bink2Scarlett.dll",
            "xboxbvhbuilder.dll",
        ];
        let mapped = ["msvcp140.dll"];
        assert_eq!(
            absent_title_dlls(&imported, &mapped, &wanted),
            ["XboxBvhBuilder.dll", "Bink2Scarlett.dll"]
        );
    }

    #[test]
    fn a_complete_package_has_nothing_absent() {
        let wanted = |dll: &str| dll.starts_with("Title");
        assert!(absent_title_dlls(&["Title.dll", "Host.dll"], &["TITLE.DLL"], &wanted).is_empty());
        assert!(absent_title_dlls(&[], &[], &wanted).is_empty());
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::sync::{Arc, Mutex};

    #[test]
    fn nested_packages_are_found() {
        let dir = std::env::temp_dir().join(format!("victory-nested-{}", std::process::id()));
        let flat = dir.join("appdata0.rpf");
        assert_eq!(super::nested_package(&flat), None);
        std::fs::create_dir_all(dir.join("appdata0")).unwrap();
        let nested = dir.join("appdata0").join("appdata0.rpf");
        std::fs::write(&nested, b"9FPR").unwrap();
        assert_eq!(super::nested_package(&flat), Some(nested));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loader_lock_is_reentrant_and_exclusive() {
        let lock = Arc::new(super::LoaderLock::default());
        assert_eq!(lock.hold(|| lock.hold(|| 7)), 7);
        let order = Arc::new(Mutex::new(Vec::new()));
        let other = lock.hold(|| {
            let (theirs, their_order) = (lock.clone(), order.clone());
            let other = std::thread::spawn(move || {
                theirs.hold(|| their_order.lock().unwrap().push(2));
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push(1);
            other
        });
        other.join().unwrap();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn cpu_feature_check_matches_host() {
        let skip = std::env::var(super::SKIP_CPU_CHECK_ENV).ok();
        #[cfg(target_arch = "x86_64")]
        let expected_ok =
            std::arch::is_x86_feature_detected!("avx2") || skip.as_deref() == Some("1");
        #[cfg(not(target_arch = "x86_64"))]
        let expected_ok = true;
        assert_eq!(super::check_cpu_features().is_ok(), expected_ok);
    }
}

#[cfg(test)]
mod cpu_check_tests {
    use super::{SKIP_CPU_CHECK_ENV, check_cpu_features_with};

    #[test]
    fn avx2_present_passes() {
        assert!(check_cpu_features_with(true, None).is_ok());
        assert!(check_cpu_features_with(true, Some("0")).is_ok());
        assert!(check_cpu_features_with(true, Some("1")).is_ok());
    }

    #[test]
    fn avx2_missing_fails_with_clear_error() {
        for skip in [None, Some(""), Some("0"), Some("true")] {
            let err = check_cpu_features_with(false, skip).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("AVX2"), "unexpected error: {msg}");
            assert!(msg.contains("Xbox Series"), "unexpected error: {msg}");
            assert!(msg.contains(SKIP_CPU_CHECK_ENV), "unexpected error: {msg}");
        }
    }

    #[test]
    fn skip_env_var_bypasses_missing_avx2() {
        assert!(check_cpu_features_with(false, Some("1")).is_ok());
    }
}

#[cfg(not(windows))]
pub fn run(_: &Path, _: Unimplemented, _: bool, _: bool, _: Console) -> Result<u32> {
    anyhow::bail!("running titles requires Windows")
}
