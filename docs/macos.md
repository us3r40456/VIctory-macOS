# macOS port status

This file tracks the work to run VIctory on Apple Silicon Macs. No game files are used or needed for any step that is marked as done.

## Done on 24 November 2026

The whole workspace builds natively on macOS (aarch64, Homebrew Rust 1.100). The Windows-only parts are gated with `cfg(windows)`, and a few small stubs stand in for them. The text entry dialog is cancelled on other hosts, a relaunch request stops the process, and the `translate` command reports that it needs Windows. On a Mac M4 the test suite gives 162 passed, 0 failed and 10 ignored. The ignored tests read their own test binary as a PE file or use Windows paths, so they only run on Windows.

A native build proves that the platform code (formats, shader recompiler, PE analysis and most of the HLE layer) is portable. It does not run a title, because the loader maps a Windows PE image into the process and calls Windows APIs.

## Done on 24 November 2026: the Windows build under Wine

VIctory now cross builds on the Mac for `x86_64-pc-windows-gnullvm` and runs under Wine 11.14. The `windows-gnu` target cannot be used, because Rust keeps its thread-locals in OS TLS slots there and the loader only saves and restores static TLS, so std aborted when a thread started. Homebrew has no llvm-mingw, so `tools/macos/gnullvm-cc.sh` links with Homebrew clang and lld against the mingw-w64 runtime.

Run the tests under Wine with this command.

```sh
WINE=/path/to/wine tools/macos/test-wine.sh --workspace --no-fail-fast
```

The loader probe passes, so the stand-in title is mapped, bound, delay loaded and run with its thread callbacks. The HLE tests pass (86 of 86), and `victory.exe --help` runs. Wine lacks three network exports (`WSAConnectByList`, `GetAddrInfoExA` and `WinHttpGetProxySettingsEx`), and VIctory now fails those calls cleanly. Two kinds of test still fail under Wine.

- Seven shader tests need a D3D12 device with shader model 6.6. The Wine build has no D3D12 that works on Metal, so this is the GPU gap that D3DMetal has to fill.
- The xtask policy test starts git from inside Wine. Run `cargo xtask policy` natively instead.

## D3DMetal results, 24 November 2026

With the GPTK cask (`brew install --cask gcenx/wine/game-porting-toolkit`, Wine 7.7 with the D3DMetal of GPTK 1.1), VIctory gets a D3D12 device on Metal, and all 69 shader tests pass on an M4. Metal's sin and cos return 0 for NaN and infinity, where the console GPU returns NaN, so the translator now passes that NaN through. D3DMetal does not check the DXIL container hash.

Wine 7.7 is too old for the rest of VIctory. The console memory tests (xmem and dstorage) and the display test fail on it, and it also lacks `WSAConnectByNameA`. Wine 11.14 passes all of them, but D3DMetal from GPTK 4.0 beta 3 crashes in libsystem_pthread on Wine 11.14, because D3DMetal needs Apple's or CrossOver's patched Wine. A Wine that is new enough for VIctory and also works with D3DMetal is the open problem.

```sh
WINE="/Applications/Game Porting Toolkit.app/Contents/Resources/wine/bin/wine64" WINEDLLOVERRIDES=winedbg.exe=d tools/macos/test-wine.sh --workspace --no-fail-fast
```

## CrossOver Wine with D3DMetal, 24 November 2026

`tools/macos/build-crossover-wine.sh` builds CodeWeavers' CrossOver 26.3 Wine (Wine 11.0) from its published source into `target/wine-cx`, using x86_64 FreeType and GnuTLS from `DEPS`. Copy the `redist/lib` folder of Apple's "Evaluation environment for Windows games" over `target/wine-cx/lib`, and run the tests with this command.

```sh
WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
```

On an M4 with D3DMetal 4.0 beta 3 this gives 260 passed, 4 failed and 4 ignored. Every loader, HLE, console memory and shader test passes on the same Wine. The policy test fails only because it runs git inside Wine. Three GPU tests show real D3DMetal differences that still need work.

- `buffer_data_reaches_the_host_texture`: fixed. The upload was queued as a copy into title memory, and that check uses VirtualQuery, which cannot see D3DMetal's mappings under Wine, so the copy was skipped. It is now queued as a fill of the list's upload memory.
- `argument_clamp_builds`: fixed. Wine's HLSL compiler (vkd3d-shader) cannot compile VIctory's clamp shader. Under Wine, `indirect.rs` now builds a second form of it (`WORDS_SOURCE`) that reads its buffers through structured views and charges the records in one thread, with the same results. This replaced the bytecode that a Windows runner used to compile, because VIctory's asset policy now refuses long runs of embedded hex.
- `a_rejected_blob_gives_an_empty_library`: fixed. D3DMetal accepts any pipeline library bytes, so the cache file now carries a mark and a checksum that VIctory checks itself.

After these fixes the whole suite passed under CrossOver Wine with D3DMetal on an M4: 343 passed, 0 failed and 4 ignored. After the merge of 25 November 2026 and the fixes below it gives 481 passed, 0 failed and 7 ignored.

A Wine built with a macOS 27 SDK used to crash on macOS 26 at its first start, in `server_pipe`. Configure found `pipe2`, which only macOS 27 has (MarshallStunt, #50). Of the 33 functions Wine's configure checks for, it is the only one that macOS 27.0.1 exports and the macOS 26.5 SDK lacks. `build-crossover-wine.sh` now tells configure that `pipe2` is missing, and Wine falls back to `pipe` and `fcntl`. A Wine that crashes this way has to be built again: remove `target/crossover/build` and `target/wine-cx`, then run `./setup.sh`.

## Run-time HLSL and textures under Wine and D3DMetal

Under Wine, `D3DCompile` is Wine's HLSL compiler (vkd3d-shader). In CrossOver 26.3 Wine it differs from Microsoft's compilers in four ways that matter here:

- It ignores `[RootSignature]` (warning W5302) and writes no root signature into the bytecode, so `CreateRootSignature` on that bytecode fails with `E_FAIL`. Build run-time compute shaders with `compute::pipeline` in `victory-gpu`, which serializes the root signature itself.
- `RWByteAddressBuffer` has no `Load`, and `RWStructuredBuffer` stores are not implemented. The depth converter in `twin.rs` reads its scratch buffer through a structured view and writes it through a raw one.
- Shader model 5.1 resource arrays are not implemented, bounded or unbounded (`E_NOTIMPL`), and `NonUniformResourceIndex` is not defined. So `lists_share_a_table_and_copies_keep_their_views` skips under Wine. Embedding FXC's bytes instead is ruled out by VIctory's asset policy, which refuses long runs of embedded hex.
- It still compiles plain `cs_5_1` shaders, and D3DMetal runs their bytecode.

### Building the indirect argument clamp with DXC

If you have DXC, you can build the 64-thread argument clamp from its HLSL source when cross-compiling VIctory. The compiled shader stays in Cargo’s build output. Set `VICTORY_HELPER_DXC` to a native DXC executable, or use the Wine wrapper with an x64 Microsoft DXC distribution:

```sh
export DXC_WINE="$WINE"
export DXC_EXE="/absolute/path/to/dxc.exe"
export VICTORY_HELPER_DXC="$PWD/tools/macos/dxc-wine.sh"
cargo build --locked --target x86_64-pc-windows-gnullvm -p victory
tools/macos/test-wine.sh -p victory-gpu --lib hardware_uses_precompiled_clamp -- --ignored
```

Keep the compiler and its companion DLLs together. DXC signs its output (the digest in the container header and the digest in the `HASH` part, which D3DMetal uses to find compiled shaders) only when `dxil.dll` sits next to `dxc.exe`. Without it the output is unsigned, with both digests zero, and zero digests once made unrelated shaders run each other's code (#4). The runtime therefore checks the compiled clamp when it loads it and sets both digests from the program if they are zero, and it logs when it had to. CI never builds this path, because it never sets `VICTORY_HELPER_DXC`; the ignored test runs the precompiled clamp and the runtime-compiled one on the same records and compares every word they write. On success the runtime logs `argument clamp: using precompiled 64-thread shader`. Without a configured compiler, or if the device rejects the compiled shader, the existing runtime compilation paths remain available.

D3DMetal has no Metal format for `R10G10B10_XR_BIAS_A2_UNORM`. It refuses any castable format list that names it, and creating a texture in that format stops the process in Metal's validation. `castable` leaves it out, so 4-byte colour textures of one memory share one host texture under D3DMetal too.

It also refuses any list that names the packed formats `R8G8_B8G8_UNORM` and `G8R8_G8B8_UNORM` (68 and 69), which every 2-byte format's list did, so no 2-byte texture took castable formats. After the first refusal, VIctory gave each format of one memory its own host texture, and a run on a 16 GB M5 grew to 6.9 GB of host textures and ran out of GPU memory (issue #15). `castable` leaves both out. With all three left out, D3DMetal 4.0 beta 3 takes all 96 lists `castable` makes, checked by creating each format's texture with its list.

D3DMetal rounds float depth down when it stores 24-bit depth (a clear to 0.5 stores 8388607, not 8388608), where Windows GPUs round to nearest.

D3DMetal 4.0 beta 3 reports support for `R32G32B32_FLOAT`, `_UINT` and `_SINT` typed buffer SRVs, but reads them as `R32G32`: later elements shift and z is zero. Under Wine, VIctory backs these views with a cached 16-byte-element copy of the source shadow. The copy keeps the title's 12-byte stride and supplies a format-correct w (`1.0f` bits for FLOAT, integer `1` for UINT and SINT). It refreshes with the source shadow, and views of the same range and w share one copy. This is a separate host copy, so an SRV and UAV aliasing the same title bytes within one submission do not see each other's writes through this workaround.

D3DMetal 3.0, as bundled with CrossOver 26.2 and 26.3, returns zeros when a 24-bit depth texture (`R24G8_TYPELESS` or `D24_UNORM_S8_UINT`) is filled by a copy or copied out, while a clear reads back correctly in a shader (#33, with a standalone repro). 32-bit depth works. GPTK 4.0 beta 3 does not have this problem, and `a_depth_texture_written_through_unordered_views_reads_as_depth` passes there, so VIctory needs D3DMetal 4.0 or newer for depth copies. The test stays as it is on 3.0 so the failure remains visible.

### R32G32B32 typed buffer UAVs, probed on D3DMetal 3.0 (26 November 2026)

A synthetic probe on an Apple M1, macOS 27.0.1, CrossOver 26.3 (Wine 11.0) and its D3DMetal 3.0 characterized `R32G32B32_FLOAT`, `_UINT` and `_SINT` separately. `CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT)` returned success with `Support1=0x3`, including typed UAV support. Direct `CreateUnorderedAccessView` then logged `VK_ERROR_FORMAT_NOT_SUPPORTED` for the corresponding three-component Vulkan format and aborted in Metal's `MTLPixelFormatInvalid` alignment assertion, before returning. No typed store or raw readback ran. `ID3D12Device::CreateUnorderedAccessView` returns `void`, not an `HRESULT`, so a post-call result check cannot prevent this abort.

Under Wine, `buffer_uav` now refuses true typed three-channel 32-bit UAVs before creation. Its caller installs the existing null `R32_UINT` buffer UAV, keeps no host resource and leaves the write mark clear. Raw and structured views are unchanged. On native hosts the code checks typed-UAV format support and device health before and after creation; a removed device is rejected without a kept descriptor or mark. The three serial Wine regressions pass without removing the device. GPTK 4.0 beta 3 was not available for this probe, so the Wine guard is conservative and also rejects these UAVs there. It will be revisited when 4.0 beta 3 is measured; until then do not narrow it.

## Route A: next steps

The first playable route is the upstream Windows build running under Wine, with D3D12 going to Metal through Apple's D3DMetal. DXMT cannot be used because it covers D3D11 only. Rosetta 2 runs the title's x86-64 code and has supported AVX2 since macOS 15.

The steps are as follows.

1. Install rustup and add the `x86_64-pc-windows-gnu` or `x86_64-pc-windows-msvc` target. The MSVC route through `cargo-xwin` downloads the Microsoft CRT and SDK under Microsoft's licence.
2. Build `apps/victory` and the `tests/fixtures/hle-probe` stand-in title for Windows.
3. Run the probe under Wine to test the loader and HLE layer without any game files.
4. Add D3DMetal from Apple's Game Porting Toolkit and check that VIctory creates a D3D12 device with mesh shaders and shader model 6.6.

## Route B: a native macOS host

Later work replaces the Windows host underneath the title. That needs a Rosetta x86-64 process that maps the PE image, a macOS version of the host pass-through in `victory-hle`, a Metal backend for `victory-gpu`, and DXIL to Metal conversion with Apple's Metal shader converter. This only makes sense after Route A shows where things break.

## Running VIctory on a Mac

Run `./setup.sh` from the top of the repository. It checks the Mac, installs the Homebrew tools, builds checksum-verified x86_64 GMP, Nettle, GnuTLS and FreeType, builds CrossOver Wine, adds D3DMetal from the Game Porting Toolkit in Downloads, and builds VIctory. The only manual step is downloading the toolkit from Apple. A clean run on an M4 on 24 November 2026 finished without help, and the full suite then passed (343 passed, 0 failed).

## CrossOver's wineserver is patched

`tools/macos/build-crossover-wine.sh` applies the patches in `tools/macos/patches` to CrossOver's source. The one there now, `wineserver-view-tree.patch`, is UwUCammie's from fxceless/victory-fork#184 (see #155 and WineHQ bug 60868). Wine's server keeps a process's memory views in a list and walks the list on every map and unmap request. The patch also keeps them in a red-black tree by base address, so those requests take O(log n). The protocol is unchanged.

On an M5 Pro with 20,000 live views, the `wine_views` benchmark from #159 took 105 to 124 µs per remap pair on CrossOver's server and 30 to 32 µs on the patched one (six rounds each). The whole Wine suite passed with the patched server: 843 passed, 0 failed, 9 ignored. #155 has the numbers from the game itself.

`./setup.sh` compares a stamp of the patches with `target/wine-cx/.victory-patches`. A Wine built before a patch was added or changed gets only its server rebuilt and installed, which takes a few minutes. That is because `make install` would put Wine's own DLLs back over D3DMetal's. A patch that changes anything outside `server/` needs a new Wine: remove `target/wine-cx` and run `./setup.sh` again.

## Use D3DMetal 4.0 beta 3

On 25 November 2026 two testers ran the same Series X executable (one SHA-256) with the same profile. With CrossOver 26.3's bundled D3DMetal 3.0 on an M5 Pro, the game stopped about seven seconds in at its own assert at `+0x1dac734`, a subsystem checked during start-up (#24). With D3DMetal 4.0 beta 3 on an M3 Max it never hit that assert and reached the game world (#50). D3DMetal 3.0 also names the adapter `VirtualApple @ 2.50GHz` and refuses an empty pipeline library blob (#30). Since #74 asks for an empty library with a null pointer, host pipelines are kept across runs on 3.0 too: mikebuckets171 saw a library written on a first run and loaded back on the second. Use 4.0 beta 3 or newer for the game. It is not the only cause of that assert, though: later the same day us3r40456 hit `+0x1dac734` with D3DMetal 4.0 beta 3 too, on both the Series X and Series S profiles, so something else in their setup also decides it (#24).

## Stacks during a freeze

In the loading freeze from #50 every thread stops logging, and `VICTORY_STACKS` printed nothing, not even its first line. Printing takes the stderr lock and allocates, and the thread snapshot goes through Wine, so a thread stuck while it holds one of those locks stops the dump too. With `VICTORY_STACKS_FILE=<path>` as well (in a build with `VICTORY_DIAGNOSTICS=1`), the dumps go to that file instead. The module table is written at start-up, and each dump writes a marker line before each step and raw addresses for each frame, with no allocation and no stderr lock. Where the file stops shows which step blocked.

## Presented planes

On Series S the game presents two planes. Plane 0 is a 1920x1080 overlay and plane 1 is the 3840x2160 picture (#15). The window shows the plane whose first resource has the most pixels, and `VICTORY_PRESENT_PLANE=<n>` picks another one. With `VICTORY_PRESENT_COMPOSITE=1` the other plane is also blended over the shown one by a compute pass (`victory-gpu/src/composite.rs`), scaled to it and using its alpha as straight alpha. The blend is off by default until a run with the game shows that the overlay is clear where the picture should show.

## D3DMetal's shader cache needs unique shader hashes

D3DMetal keeps a compiled-shader cache for each executable, in `$(getconf DARWIN_USER_CACHE_DIR)d3dm/<exe name>/shaders.cache`. It finds shaders by the DXIL container's shader hash (the `HASH` part), apparently together with the shader's length: two shaders collided only when their lengths matched. VIctory used to write the same all-zero shader hash into every container. So two different shaders of the same length shared a cache entry, and the one compiled second ran the first one's code. The cache is on disk, so the wrong code also came back on every later run of that executable.

This was the "signed zero in minimums" difference listed here before. Each `v_min_f32` and `v_min3_f32` test shader is the same length as the maximum built just before it, so the minimum ran the maximum's code and gave +0 where the console gives -0. Selecting on the integer bits could not help, because the shader that ran was never the one being changed. It also made `v_cmp_class_f16_matches_rdna2` fail whenever the max/min test ran before it, as it does with `--test-threads=1`. The `v_max3_f32` shader and that test's first shader are both 2316 bytes.

`container::shader_hash` now writes the MD5 of the program into the `HASH` part. The minimum cases run under Wine again, and both tests pass in every order on an M5 Pro with D3DMetal 4.0 beta 3. Cache entries written under the old zero hash are never looked up again, so old caches need no cleaning.

## Open differences under D3DMetal

The merge of VIctory's changes up to 70564ae on 25 November 2026 brought new shader tests, and three of them find real differences in D3DMetal 4.0 beta 3 on an M4. The tests skip only those checks under Wine, so the rest of each test still runs.

- 64-bit image atomics. D3DMetal reports `AtomicInt64OnTypedResourceSupported` and `AtomicInt64OnDescriptorHeapResourceSupported`, but its 64-bit atomics in `image_atomics_of_64_bits` leave every texel at zero. The game's alpha prepasses use them. Under Wine the translator now emulates them (`Options::atomic_locks`, the module notes' "Atomic locks", `VICTORY_ATOMIC_LOCKS=0` or `1` to override): the lanes of a wave take turns in a loop over lane indices, and in its turn a lane takes a 32-bit lock word, reads the texel, writes the result and frees the lock. Two simpler forms failed under D3DMetal and are worth knowing about. Lanes of one wave spinning on each other's locks never finish, and `WaveIsFirstLane` and `WaveReadLaneFirst` inside a divergent loop return other lanes' values. The spin is bounded at 65536 tries so a fault can never hang the GPU.
- Sample bias in compute programs. D3DMetal ignores the bias of a sample in a compute shader: a bias of 1 reads level 1 where the console reads level 2. Under Wine the translator now runs those samples as gradient samples with the quad gradients scaled by `2^bias` (`Options::bias_as_gradients`, found by nickorlabs in #30; `VICTORY_BIAS_AS_GRADIENTS=0` or `1` overrides). The bias cases in `image_addresses_of_16_bits_and_clamps` run under Wine again and pass. Pixel shaders keep the plain biased sample.
- 24-bit depth. Apple GPUs have no 24-bit depth format, so D3DMetal keeps it as 32-bit float and rounds down when it stores 24 bits. The depth test clears to 0.75, where both roundings agree (PR #6).
- Depth bounds. D3DMetal reports `DepthBoundsTestSupported` and does not apply the test (Metal has none): in the merged VIctory test a draw whose bounds exclude the stored depth still writes. Under Wine `Host::depth_bounds_supported` is false, so pipelines are made without bounds and one warning says so, which has the same effect and keeps the plain pipeline states. The test follows the flag.
- Marked write shaders. Wine's HLSL compiler cannot build the shader of the five `marked_subpage_uav_*` and `marked_unaligned_root_uav_*` tests (an interlocked write into a byte address buffer), so they skip under Wine.
- Root signature blobs. D3DMetal accepts a truncated root signature blob and crashes on garbage, where Windows returns `E_INVALIDARG`, so `malformed_blobs_are_rejected_by_the_host` skips under Wine.
- Hard links. `a_kept_blob_past_the_link_limit_is_copied` needs NTFS's limit of 1024 links to a file, and APFS has no such limit, so the test skips where the 1025th link succeeds.
- Gather positions. D3DMetal's own gather rounds the position as the console does, but from u × size taken in f32, where the console takes the exact product. On an M5 Pro it starts a texel over on all 12 lanes of `gathers_at_a_level_match_the_hardware` where the two products fall on either side of a texel's threshold, and matches the console on the others. Truncating, as WARP does, would have kept the console's texel on those lanes. VIctory's gathers at a level follow the console, but its level-0 gathers are the host's own, so they can differ there. The test checks that the host's own gather is either the console's or this one.

Run-time compute shaders are built with explicit root signatures (`compute.rs`, PR #6), which Wine's compiler can build. The one exception is the memory test's shader, which needs shader model 5.1 resource arrays, so that test skips under Wine.

## Host view heaps are filled with other threads' submissions held off

Under Wine on D3DMetal, a list that one thread submits while another thread fills a new host view heap can lose its views: its shaders' writes are dropped and its reads give zeros, whatever heap the list binds (issue #31). Each host view heap has 2,359,296 slots, and `HostHeap::fill_null` makes every one null with 36 copies of 65,536 null views. On an M5 Pro with D3DMetal 4.0 beta 3 that takes 30 to 50 ms, and the victory-gpu tests fill several at once when they start.

A texture that a dispatch wrote in that time read back as zeros. A copy of it by the copy engine was zeros too, and so was the same write and read in fresh lists, with fresh pipelines or with the views made again. It took writes again by itself 31 to 49 ms later, once the fills had ended. Skipping the fill, or holding submissions off while a heap is made and filled, stopped it. Holding them off only while its upload buffers were made did not.

So `Host::quiet` holds the submission lock while a view heap is made and filled (`HostHeap::new`, and the first `Heap::host` of a title view heap, which also writes the views queued before it). The fill waits for the submission running then, and the next submission waits for the fill. That happens once per view heap. Elsewhere than Wine nothing is held. In 100 runs of the victory-gpu library tests on that Mac, the four texture tests this hit (1D and 3D, placed view, depth through unordered views, overlay) failed 62 times before, often several in one run, and not once after.

A standalone D3D12 program did not reproduce it: 16 threads writing and reading textures, while 3 or 4 other threads made heaps of the same size and filled them the same way, bound in submissions or not. Something else VIctory does is needed as well, and it is not known what.

Buffers are another matter, still open. `root_views_reach_their_copy_and_read_zeros_past_it` and the two `heap_views_` tests still fail in a few runs in a hundred, with or without the change. In a failed `root_views` run the source's host copy held the title's values while the output stayed zero, when run again at once, 100 ms later and with its view written again, so it does not come back by itself as the textures do.

## Masked stores into protected pages hang Rosetta

On macOS 27.3 (M3 Max) the game stops at start-up with `rosetta error: unexpectedly need to EmulateForward on a synchronous exception` and the process hangs (issue #1). The faulting instruction is `vpmaskmovd [rax], ymm6, ymm4`, an AVX2 masked store in a loop that steps 32 bytes at a time.

The reproducer from issue #1, extended with `vpmaskmovd`, shows the cause without game files. Under CrossOver Wine 26.3.0 with `ROSETTA_ADVERTISE_AVX=1`, SSE and AVX `vmovups` stores survive in every layout, and so does a masked store wholly inside a protected page. A masked store that starts in writable memory and ends in a protected page hangs Rosetta with the same error, both in a block guarded the way `victory-gpu/src/watch.rs` guards it and in `MEM_WRITE_WATCH` memory, which Wine emulates with page protection. Rosetta seems to have written the first lanes before the fault and cannot finish the instruction afterwards.

VIctory uses both kinds of protection to see which title GPU memory changed. Without them, host copies of title memory take it whole every time, as they already do for memory that is not the runtime's, which came to 55 to 115 GB per world frame (#50). Under Rosetta, which `victory_gpu::fault_tracking` detects as Wine on Darwin, tracking is off by default, because Rosetta on macOS 26.6 also stops on a plain 32-byte `vmovups` that crosses into a protected page (#1), and #144 hung on an M1 with both switches on. `VICTORY_FAULT_TRACKING=1` turns tracking on, and the stores Rosetta cannot resume are then patched at load time (below). `0` turns it off anywhere. `victory` prints `fault tracking on` or `off` at start-up, with the variable that turns it on when it is off under Rosetta.

Which stores Rosetta resumes after a write fault was measured on macOS 27.0.1 (M5 Pro) with a native x86-64 test that stores against a no-access page, with the whole store inside it or starting before it, from code translated both ahead of time and at run time. Plain stores of every width (`mov`, `movups`, `vmovups`, `vmovdqu`, `vmovntdq`, `movnti`, `vmovq`, `vpextrq`, `vextractf128`), adjacent pairs of them, `rep stosb`, `rep movsq` and locked atomics all resume. Every AVX masked store (`vpmaskmovd`, `vpmaskmovq`, `vmaskmovps` and `vmaskmovpd`, at 128 and 256 bits) stops Rosetta when it starts before the page, and resumes when it is wholly inside it or its lanes in the page are masked off. `vcvtps2ph` to memory stops Rosetta even wholly inside the page. The store patch below rewrites exactly those.


The game's executable holds 1130 masked stores (789 `vpmaskmovd`, 160 `vpmaskmovq`, 180 `vmaskmovps` and 1 `vmaskmovpd`), which `victory` reports at start-up. `VICTORY_PATCH_MASKED=all` (or a comma list of hex RVAs) rewrites each one at load time as a jump to a small trampoline that does the same stores one lane at a time with plain `mov`s, which Rosetta can resume after a fault (`victory-loader/src/masked.rs`). With that, fault tracking can run under Rosetta, so copies of title memory no longer have to be taken whole (about 129 GB per frame once the world streams in, #50). Whenever tracking runs under Rosetta and `VICTORY_PATCH_MASKED` is unset, `all` is the default, and `none` turns the patch off. Runs with both on in #50, on macOS 27.2.1 with an M2 Pro and an M5 Pro, lasted 6 to 35 minutes and copied 100 to 200 MB per frame in the intro and the world. None ended in Rosetta: each ended when the game itself wrote to address 0, which one tester traced to the game's watchdog. The scan reads bytes, so a rare false match could be patched with `all`, and naming RVAs avoids that. Run 11 in #50 showed that `vcvtps2ph` stores to memory hang Rosetta the same way, so the scan and the patch also cover `vcvtps2ph`, `vextractf128` and `vextracti128` stores. Those are redone into the stack and then copied out with plain `mov`s. With the masked stores patched, copies during loading fell from 158.7 MB to 0.7 MB per frame and frames from 125.7 ms to 38.3 ms.

With every masked store patched and fault tracking on, #144 then stopped on a plain 256-bit store, `vmovups [r8+0x34], ymm0` at RVA 0x1E09B31, with the same Rosetta error. Plain stores like that are far more common than masked ones, so the scan does not look for them. Instead, an RVA in the list that the scan did not find is decoded as an unaligned 256-bit store (`vmovups`, `vmovupd` or `vmovdqu`) and redone the same way, into the stack and out with plain `mov`s. `all` and RVAs can be given together, as in `VICTORY_PATCH_MASKED=all,1e09b31,1e09b37`. A listed RVA that is neither kind is left alone with a warning, and so is a store shorter than the 5-byte jump.

## Host presentation size

Set `VICTORY_PRESENT_MAX=1280x720` to cap the host presentation pass and DXGI swap chain at 720p. Smaller pictures keep their native size. The cap preserves the picture’s aspect ratio and applies after the game’s rendering and upscaling.

## Host textures on Macs with 8 or 16 GB

Host textures fed from title memory are given up when the title has not used them for two minutes, or, past 90 % of the card's memory budget, least recently used first (`victory_gpu::resource::evict_unused`). On a Mac that budget does little, as measured under CrossOver 26.3's Wine with D3DMetal 4.0 beta 3 on an M5 Pro with 48 GB:

- `QueryVideoMemoryInfo` gives a budget of about four fifths of the Mac's memory (40,201 MB of 51,540 MB), the same for local and non-local memory, and `D3D12_FEATURE_ARCHITECTURE` says the adapter is not UMA. Its use follows our textures, heaps and buffers exactly.
- The title's own memory shares that memory but is not in the budget. On an 8 GB Mac the textures alone filled memory into swap long before 90 % of the budget (#135).
- Memory freed through D3DMetal stays with the process. Released heaps, and released committed buffers, leave the process's physical footprint (`footprint -p`) where it was, 20 seconds later too, and `ID3D12Device::Evict` does not lower it. New resources reuse that memory first. So the process holds the most GPU memory it ever used, and textures help only if they are given up before that peak.

`VICTORY_GPU_BUDGET_MB=<MB>` sets the budget the pass keeps host textures under, in place of the device's. A diagnostics build logs `host textures given up: N (M MB); made again after being given up: ...` with the statistics, so a run shows whether the pass gives textures up and whether it gives up ones still in use.

## Metal ends long draws, and can stop the queue

On macOS 27.0.1 (M5 Pro) with D3DMetal 4.0 beta 3, a draw whose every pixel runs long is ended about 50 ms in. D3DMetal logs `Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)`, and the draw writes no pixels. The fence still completes, and `GetDeviceRemovedReason` still returns `S_OK`. The time is per pixel, not per draw. A 512x256 draw of 700 ms ran whole, and so did a 4096x256 draw of 2 s, while a 512x256 draw of 1 s was ended. A compute dispatch was never ended, even one of 29 s.

After its second end, a queue ignores everything submitted to it, presents included. That held in all 7 runs on that Mac, and a first end never did it. D3DMetal logs `Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)`. Its fence still completes, so a run carries on with nothing drawn and the window frozen. A new queue on the same device runs work again, and so does a new swap chain for the same window on the new queue. cocobailey's run in #50 shows the same pair, and VIctory logged neither.

So every submission now ends with a small list that copies the submission's number into read-back memory (`Mark` in `victory-gpu/src/host.rs`). `WriteBufferImmediate` writes nothing under D3DMetal, so it is a copy. When a submission did not run as far as its mark, it was ended, and what it drew is missing. VIctory logs that, then submits the mark alone. If that does not run either, a new queue replaces the old one, and the next present makes its swap chain on it. If the new queue does not run work either, the run ends with a host GPU failure. `work_goes_on_after_macos_stops_the_queue` (ignored, run under Wine on a Mac) shows the whole sequence. Two draws of about 4 s were ended, the second left its queue ignoring work, and a draw on the new queue ran whole.

## The clwb write-back loops

The title has eight loops that write cache lines back with `clwb`, which faults under Rosetta and costs an exception each time (#135). Under Rosetta, the same check that decides fault tracking, `victory` turns the `clwb` of each loop into a NOP when it loads the image. It patches only when it finds exactly eight loops and logs the number it found otherwise. `VICTORY_NOP_CLWB=0` leaves the loops alone and `VICTORY_NOP_CLWB=1` patches them where Rosetta is not detected.
