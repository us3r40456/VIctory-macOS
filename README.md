<p align="center"><img src="docs/images/banner.svg" alt="VIctory for macOS: a neon tropical sunset with palm trees, a coastal skyline and a MacBook on the shore"></p>

<h1 align="center">VIctory for macOS</h1>

<p align="center">
Grand Theft Auto VI is only out on consoles. This project runs the Xbox Series build on Apple Silicon Macs, from a copy of the game that you own.<br>
It is built on <a href="https://github.com/Marvisak/VIctory">VIctory</a>, which does the same on Windows, and it brings that work to the Mac through Wine and Metal.
</p>

<p align="center">
<img alt="Apple Silicon M1 or newer" src="https://img.shields.io/badge/Apple%20Silicon-M1%2B-555555">
<img alt="macOS 15 or newer" src="https://img.shields.io/badge/macOS-15%2B-blue">
<img alt="701 tests passed" src="https://img.shields.io/badge/tests-701%20passed-2ea44f">
<img alt="GPL-3.0 licence" src="https://img.shields.io/badge/licence-GPL--3.0-lightgrey">
</p>

<p align="center">
If you want to see GTA VI on a Mac, please star the repo and read <a href="CONTRIBUTING.md">CONTRIBUTING.md</a>. Help is welcome at every level.
</p>

> [!WARNING]
> **A note from the maintainer.** I am a student, and I run this project with Claude on a free Claude Pro plan from Anthropic. Reviewing pull requests, answering issues and fixing bugs today used up half of my weekly Claude limit in a single day. My limit resets on Saturday, 28 November, so until then I will be slower to review, reply and merge. Thank you for your patience, and please keep sending reports and pull requests in the meantime.
>
> If you would like to help, a sponsored Claude Pro or Max subscription or some Anthropic API credits would let me keep up with the project at its current pace. I would be very grateful for either. Please reach out on email if you can help: pragyaangaur12@gmail.com

> [!IMPORTANT]
> **The game is not playable yet, on the Mac or anywhere else.** On testers' Macs the game starts, plays its intro and audio, shows its loading screens, and on 25 November the game world appeared on two Macs (#50). Nobody has reached gameplay yet. The loading freezes testers hit on 25 November were two deadlocks inside Wine, found by LakshanSivakumar and fixed in #72, #75 and #78. With `VICTORY_PATCH_MASKED=all VICTORY_FAULT_TRACKING=1` the game now loads into the world without freezing on an M5 Pro and an M3 Max, and stays there for over 20 minutes. It is still far too slow to play, but getting faster: #111 cut world frames from about 4.5 seconds to 2.4 to 2.9 seconds on an M5 Pro (#50). With `VICTORY_PATCH_MASKED=all VICTORY_FAULT_TRACKING=1`, loading now runs at about 26 to 30 fps instead of 8 (#59, #60). VIctory on Windows boots the game, runs its menus, loads the world and now draws it, but only at one or two seconds per frame and with unlit surfaces and missing geometry. This project makes the whole VIctory platform run on a Mac, so the Mac moves forward as VIctory does. No game files are included here or ever will be. You need your own copy of the game.

## What works today

Every part of VIctory now runs on an Apple Silicon Mac, and its full test suite passes on a MacBook with an M4 chip.

| Part | What it does | On the Mac |
| --- | --- | --- |
| Loader | Maps the game's executable and DLLs into one process and runs its code unchanged | Works |
| Console runtime | Stands in for the Xbox: memory, files, saves, input, audio and network rules | Works |
| GPU layer | Turns the console's Direct3D 12.X into desktop Direct3D 12 | Works on Metal through D3DMetal |
| Shader recompiler | Recompiles the console GPU's shaders, checked against the hardware's results | All 69 checks pass on Metal |
| Test suite | 711 tests across every crate | 701 passed, 0 failed, 10 ignored |

<p align="center"><img src="docs/images/tests.svg" width="640" alt="Terminal output of the test suite under CrossOver Wine with D3DMetal on an Apple M4: 701 passed and 0 failed"></p>

This is the real result of a clean run of `./setup.sh` followed by the test suite on a MacBook with an M4 chip.

Getting there took a few Mac-specific fixes, and each one is explained in [docs/macos.md](docs/macos.md). Two of them fixed real bugs that would have broken the picture on screen. Texture uploads were being skipped because Wine cannot see the memory that D3DMetal maps, and Metal's `sin` and `cos` return 0 where the console GPU returns NaN.

## What testers see in the game

Several people who own the game have run it on their Macs and reported back in the issues. The game starts, plays its audio and runs its menus, and on current main the picture shows on Series X. Here is what they have seen so far.

| Mac | What happened | Issue |
| --- | --- | --- |
| M5 MacBook Air | Got past the early Rosetta hang. The intro animation and the menu appeared with a local workaround that showed the game's 3840x2160 image directly, and the first scene's audio played. Without the workaround the window stays black. | #1, #15 |
| M5 Pro | With the Series S profile the game presented 64 frames of real drawing, all of them black, and then stopped on one of the game's own checks during start-up. Packages kept one per subfolder made it stop even earlier, and VIctory now says so at start-up. | #24 |
| M5 | At the Player prompt the game's 3840x2160 image held the picture while the 1920x1080 image sent to the window was black. | #15 |
| M2 Max | Account selection, the first setup screens with keyboard input, and the loading artwork all worked, and then the window went black. | #28 |
| M5 Pro, 24 GB, Series X | On 25 November, with the store patch and fault tracking on, the account picker, the setup screens and the legal notices ran at 25 to 30 fps. About three minutes in, the game reached the "Vice City, two years ago" title card at 2 to 5 seconds per frame and stayed alive there. | #65 |
| M5 Pro, Series X | On 25 November, with no variables set, the game ran its intro and loading at 25 to 30 fps, then its draw count jumped to 3,000 to 3,900 per frame, which suggests it was drawing the world, though the screen stayed grey at 0.1 fps. A later run with `VICTORY_PATCH_MASKED=all` only went on into the world and drew a dark city skyline, still at 0.1 fps. After the freeze fixes, plain main with `VICTORY_PATCH_MASKED=all VICTORY_FAULT_TRACKING=1` loaded into the world with up to 5,400 draws per frame and no freeze, at 0.2 fps. With #111 (fewer memory mappings under Wine's one memory lock) world frames fell from 4.5 s to about 2.4 to 2.9 s while it copied 57 to 83 GB of memory per frame. With the store patch and fault tracking on, copies stayed under a few MB per frame through loading, and it froze about three minutes in. That freeze led to #63. | #50 |
| M3 Max, Series X | On 25 November the picture showed with no workaround. The intro played and the game reached the "Vice City, two years ago" title card, where it froze while Apple's shader compiler kept crashing. A second run with a warm shader cache had no compiler crashes, but the window went black after the loading artwork. In a later run the game world appeared on screen before the game's own watchdog stopped it, with frames taking up to 28 seconds, most of it copying memory. With the store patch from #59 and #60 and fault tracking on, loading ran at 26 fps instead of 8 and copies fell from 158.7 MB to under 1 MB per frame, until a freeze about three minutes in. After the freeze fixes, plain main stayed in the world for over 20 minutes without freezing, at 0.1 to 0.2 fps, with far fewer shader compiler crashes (37 against about 150). With today's speed changes frames there were 14% faster, and a later run stayed in the world for 11 minutes before the game's own timing checks gave up at about 12 seconds per frame. | #50 |

<p align="center"><img loading="lazy" src="https://github.com/user-attachments/assets/f68a0cf3-bebc-4b12-a830-057983deb2cc" width="480" alt="The game world on an M3 Max under VIctory, after the speed changes"><br><sub>The game world on jjlange's M3 Max on current main, after the freeze and speed fixes, 25 November 2026 (#50).</sub></p>

<p align="center">More screenshots and videos from testers are in <a href="docs/gallery.md">docs/gallery.md</a>.</p>

The black screen from #15 is solved: VIctory now shows the larger of the images the game presents, and `VICTORY_PRESENT_COMPOSITE=1` also draws the Series S overlay on top. Loading used to crawl because VIctory copied all of the game's memory on every submission, since tracking which pages changed hangs Rosetta on the game's AVX stores. #59 and #60 rewrite those 1,234 stores at load time so tracking can run. The loading freeze is fixed: Wine's file metadata calls from our shader cache (#72, #75) and write guards that reached the process heap (#78) caused two deadlocks. The open problems now are the game's speed in the world (2.4 to 2.9 seconds per frame after #111, split between preparing copies, the GPU, copying back and memory protection changes) and crashes in Apple's shader compiler while the game builds its world shaders.

## How it works

<p align="center"><img src="docs/images/stack.svg" width="720" alt="The layers between the game and the Mac: the game's code, VIctory, CrossOver Wine and D3DMetal, Rosetta 2 and Metal, and the Apple Silicon Mac at the bottom"></p>

The Xbox Series version of the game is x86-64 code, like a Windows program. VIctory loads that code and supplies everything the console would have supplied. On a Mac, VIctory itself runs as a Windows program:

- **Rosetta 2** runs the x86-64 code, including the AVX2 instructions that the game uses.
- **CrossOver Wine** provides Windows. The setup builds it from the source that CodeWeavers publishes, because Apple's D3DMetal only works with that branch of Wine.
- **D3DMetal**, from Apple's Game Porting Toolkit, turns Direct3D 12 into Metal.

## Set it up in three steps

You need an Apple Silicon Mac (M1 or newer) on macOS 15 Sequoia or newer, [Homebrew](https://brew.sh), and about 15 GB of free space.

**1. Get the code.**

```bash
git clone https://github.com/pragyaangaur/VIctory-macOS.git
cd VIctory-macOS
```

**2. Download Apple's Game Porting Toolkit.** Apple only gives D3DMetal to signed-in developers, so this is the one step nobody can do for you. Open the [Apple developer downloads](https://developer.apple.com/download/all/?q=game%20porting%20toolkit), sign in with any Apple account (a free one works), and download the newest **Game Porting Toolkit** `.dmg`. Leave it in your Downloads folder and do not open it. Use version 4.0 beta 3 or newer: every tester who has reached the game world used it, and the D3DMetal 3.0 that CrossOver bundles cannot keep compiled pipelines (#30). One tester with 4.0 beta 3 still stops at an early start-up check of the game, so the version is not the whole story there (#24).

**3. Run the setup.**

```bash
./setup.sh
```

The setup installs what it needs with Homebrew. It then builds the x86_64 libraries and CrossOver Wine, adds D3DMetal from the toolkit, and builds VIctory. The first run takes about an hour, almost all of it spent building Wine. If anything stops it, fix what it says and run it again, because finished steps are skipped.

When it says **Setup finished**, you can check that VIctory starts:

```bash
tools/macos/victory.sh config
```

## Running the game

Put your own copy of the Xbox Series build in a folder, set `game_dir` in `victory.toml` to that folder, and start it. Every package's files must sit directly in that folder, so `appdata0.rpf` is at `<game_dir>/appdata0.rpf` and not in an `appdata0` subfolder. With one subfolder per package the title asserts early at `+0x2237178`, and `victory` now stops before that and says which file is misplaced (#24).

```bash
tools/macos/victory.sh run --execute <the game's executable>
```

For now this gets as far as VIctory does on Windows, as the note at the top explains.

## Running the tests

These commands are useful for anyone who works on the code. The first runs the tests natively on the Mac, and the second runs them as Windows programs on Wine with D3DMetal, which includes the GPU tests.

```bash
cargo test --workspace
```

```bash
WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
```

## Help test on your Mac

The project needs test results from as many Macs as possible. The [Mac test fleet](docs/fleet.md) runs both test suites on your Mac and reports the counts to this repository. It uses only the project's own tests, never a game. Paste this into an AI coding agent on your Mac, such as Claude Code, Codex or Cursor:

```text
Read https://raw.githubusercontent.com/pragyaangaur/VIctory-macOS/main/docs/fleet-join.md and follow it to help test VIctory on my Mac. Ask me once before installing or posting anything, then keep going until I say stop.
```

Please read [docs/fleet-legal.md](docs/fleet-legal.md) first. You can also run the fleet by hand with `tools/fleet/fleet.sh`.

## Debugging switches

These need a diagnostics build (`VICTORY_DIAGNOSTICS=1 tools/macos/victory.sh ...`) and are for chasing problems, not for playing.

A diagnostics build also logs one bounded line per submission that refreshes title memory: the total range count and bytes, then the eight largest ranges' title start/length and host GPU address. It logs no memory contents. A batch combines requests from several lists, view heaps and page tables, so a range is not attributed to an individual descriptor or view.

| Variable | What it does |
| --- | --- |
| `VICTORY_STACKS=<seconds>,...` | Prints every thread's stack at those times after start. |
| `VICTORY_STACKS_FILE=<path>` | Writes those dumps to a file without taking locks, for freezes. |
| `VICTORY_SKIP_TITLE_INT3=<rva>,...` | Steps over the game's own `int3` asserts at those hex RVAs and logs each one. Every other assert still stops the run. |
| `VICTORY_TRACE_ACTIVATION=1` | On Series 1.0.2.1, logs registration of event `0x9a0892b1`, callback subcodes, and the second registry table before and after each `0x20` call. Look for `VIctory: activation` in stderr. The Mac launcher fetches a pinned Frida Gadget archive on first use. Leave `VICTORY_SKIP_TITLE_INT3` unset to capture the natural assert. |
| `VICTORY_GPU_TIMING=1` | Submits and times each GPU list on its own. |

### Every variable

The on/off switches below (`...=1`) are off when unset or set to `0`.

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_POKE=<rva>:<byte>,...` | writes a title byte twice a second (a hang watchdog's tick, say) |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_BREAKPOINTS=1` | appends first-chance breakpoints to `crashes.log`, including ones a handler may recover from |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_CAPTURE_WORLD=<list>` | with `VICTORY_CAPTURE`, also saves (and with `VICTORY_DUMP_TARGETS` dumps) the frames at these offsets from the first frame that ran 500 draws (the world's first; its number depends on how fast the title loaded) |
| `VICTORY_DUMP_WRITERS=<hash>,...` | with the two above, also the textures a pipeline containing a listed shader hash writes, the first frame it runs, and for its buffers a log line (dwords changed, dwords not zero) |
| `VICTORY_PEEK=<hex address>[:<bytes>],...` | in listed frames, those bytes of title memory logged after each submission. Listed frames also log targets of 16 texels or fewer as values, each table view's view words and whether its host slot is null, the host copy of every buffer of 64 bytes or fewer a draw reads (at submission and after its list), and each 1D or 3D texture a dispatch writes, read back right after it |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | skip host pipeline creation and omit draws and dispatches, including indirect ones, when the pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands; a device removal names the list it happened in and lists all its commands |
| `VICTORY_TRACE_SHADERS=<hash>,...` | with `VICTORY_D3D12_DRED`, every draw and dispatch of a listed shader logged with what it binds (user data, heap slots, root views and their first dwords; the first 500, and all of them in listed capture frames); the records of its indirect executions are read back and logged in every frame (records with work, the first four, the frame) |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_PIPELINE_LOG=<file>` | a line per title pipeline created (key, shaders, state), first used, and made or loaded (library name); `translate --cache --pipelines` writes the last kind too |
| `VICTORY_PREBUILD_KEYS=<file>` | `translate --cache --pipelines` makes nothing and lists each pack record's key, shaders and state (or why it is skipped) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

## Where things are

| Path | What it holds |
| --- | --- |
| `setup.sh` | The one-step setup |
| `tools/macos/` | The Wine build, the library build, the launcher and the test runner |
| `tools/fleet/` | The Mac test fleet: the worker, the issue queue and its tests ([docs/fleet.md](docs/fleet.md)) |
| `docs/macos.md` | The full story of the port, the problems found and how each was fixed |
| `crates/` | VIctory's runtime: loader, console runtime, GPU layer, shader recompiler and formats |
| `docs/architecture.md` | How VIctory itself is put together |

## Acknowledgement

This project is built on top of [VIctory](https://github.com/Marvisak/VIctory) by Marvisak. The loader, the console runtime, the GPU layer and the shader recompiler all come from VIctory, and its full history is kept in this repository. This project adds the macOS side: the Mac build, the Wine and D3DMetal setup, and the fixes that the Mac needed. VIctory is licensed under GPL-3.0, so this project is too.

Thanks also go to CodeWeavers for publishing CrossOver's source, to the Wine project, and to Apple for the Game Porting Toolkit.

## Legal

This project is not affiliated with or endorsed by Rockstar Games, Take-Two Interactive, Microsoft or Apple. Grand Theft Auto is a trademark of Take-Two Interactive. This repository contains no game files, keys or code from the game, and it never will. It only runs a copy that you own. D3DMetal is not included, and each user downloads it from Apple under Apple's terms.

The code is licensed under the [GNU General Public License v3.0](LICENSE).
