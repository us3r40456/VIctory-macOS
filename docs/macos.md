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
WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
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

`tools/macos/victory.sh` does this by itself with the DXC that D3DMetal ships (`libdxcompiler.dylib`, x86_64 only, run under Rosetta): it builds a minimal DXC front end (`tools/macos/dxc-d3dmetal.sh`, which holds its C++ source) over that library, and sets `VICTORY_HELPER_DXC` to it unless you already set it. CrossOver's Wine has neither a `d3dcompiler` that takes the 64-thread clamp (E5005 on `RWByteAddressBuffer.Load`) nor `dxcompiler.dll`, so without this VIctory falls back to the one-thread form (`WORDS_SOURCE`), which charges every record in a single thread. D3DMetal's DXC signs its output.

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

D3DMetal 4.0 beta 3 reports support for `R32G32B32_FLOAT`, `_UINT` and `_SINT` typed buffer SRVs, but reads them as `R32G32`: later elements shift and z is zero. Under Wine, VIctory backs these views with a cached 16-byte-element copy of the source shadow. The copy keeps the title's 12-byte stride and supplies a format-correct w (`1.0f` bits for FLOAT, integer `1` for UINT and SINT). It refreshes with the source shadow, and views of the same range and w share one copy.

This is a separate host copy. A buffer or UAV barrier therefore also records a GPU widening copy for the live cached views that overlap UAV writes before it. The copy reads the shadow the UAV wrote, so raw aliases also reach an off-element SRV's exact copy; heap-indexed writes use the ranges their views can reach. A view made after the barrier is brought current before its first read. Widened buffers have explicit host states and return to common when the list closes. Only overlapping views are copied, and the title must still issue the normal UAV barrier between a write and a read.

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

After setup, run `tools/macos/victory.sh gpu-check`. It uses the same D3D12
device and feature checks as the renderer, and needs no game files. It prints
the result for each feature and exits successfully when the device reports
both. A failed feature query appears as `query failed`.

## Route B: a native macOS host

Later work replaces the Windows host underneath the title. That needs a Rosetta x86-64 process that maps the PE image, a macOS version of the host pass-through in `victory-hle`, a Metal backend for `victory-gpu`, and DXIL to Metal conversion with Apple's Metal shader converter. This only makes sense after Route A shows where things break.

## Running VIctory on a Mac

Run `./setup.sh` from the top of the repository. It checks the Mac, installs the Homebrew tools, builds checksum-verified x86_64 GMP, Nettle, GnuTLS and FreeType, builds CrossOver Wine, adds D3DMetal from the Game Porting Toolkit in Downloads, and builds VIctory. The only manual step is downloading the toolkit from Apple. A clean run on an M4 on 24 November 2026 finished without help, and the full suite then passed (343 passed, 0 failed).

## CrossOver's Wine is patched

`tools/macos/build-crossover-wine.sh` applies the patches in `tools/macos/patches` to CrossOver's source. One of them, `wineserver-view-tree.patch`, is UwUCammie's from fxceless/victory-fork#184 (see #155 and WineHQ bug 60868). Wine's server keeps a process's memory views in a list and walks the list on every map and unmap request. The patch also keeps them in a red-black tree by base address, so those requests take O(log n). The protocol is unchanged.

On an M5 Pro with 20,000 live views, the `wine_views` benchmark from #159 took 105 to 124 µs per remap pair on CrossOver's server and 30 to 32 µs on the patched one (six rounds each). The whole Wine suite passed with the patched server: 843 passed, 0 failed, 9 ignored. #155 has the numbers from the game itself.

The other, `ntdll-skip-host-mappings.patch`, is for thread creation. To find room for a new thread's stack, Wine's `try_map_free_area` tries an address and, if something is mapped there already, moves on by one 64 KiB granule. Memory that the host maps outside Wine, such as Metal's, looks free to Wine, so the search stepped through gigabytes of it one granule at a time, with `virtual_mutex` held. Every page fault, `VirtualQuery`, `VirtualProtect` and `GetWriteWatch` of the other threads waits for that lock. On an M3 Max with the Series S profile (`--console series-s`), each thread the title made took about 2.5 s, and the intro ran at 0.1 fps with frames of 7.6 s (#50, runs 65 to 68). Stack dumps showed one thread in `NtCreateThreadEx` → `virtual_alloc_thread_stack` → `map_view` → `try_map_free_area` → `mach_vm_map` in all twelve dumps, and the others waiting for `virtual_mutex`. With the patch, a failed address asks `mach_vm_region` for the mapping in the way and moves past it. Every address it skips overlaps that mapping, so no free area is missed. The same intro then ran at 20 to 30 fps, and the longest wait for the lock fell from 2.5 to 7 s to 27 ms (run 69). World frames with the Series X profile were unchanged (run 70): the title does not make threads there at that rate.

`./setup.sh` compares a stamp of the patches with `target/wine-cx/.victory-patches`. A Wine built before a patch was added or changed gets only its server and `ntdll.so` (ntdll's Unix library) rebuilt and installed, which takes a few minutes. That is because `make install` would put Wine's own DLLs back over D3DMetal's, and D3DMetal replaces neither of those two. A patch that changes anything outside `server/` and `dlls/ntdll/unix/` needs a new Wine: remove `target/wine-cx` and run `./setup.sh` again.

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

- 64-bit image atomics. D3DMetal reports `AtomicInt64OnTypedResourceSupported` and `AtomicInt64OnDescriptorHeapResourceSupported`, but its 64-bit atomics in `image_atomics_of_64_bits` leave every texel at zero. The game's alpha prepasses use them. Under Wine the translator now emulates them (`Options::atomic_locks`, the module notes' "Atomic locks", `VICTORY_ATOMIC_LOCKS=0` or `1` to override): the lanes of a wave take turns in a loop over lane indices, and in its turn a lane takes a 32-bit lock word, reads the texel, writes the result and frees the lock. Two simpler forms failed under D3DMetal and are worth knowing about. Lanes of one wave spinning on each other's locks never finish, and `WaveIsFirstLane` and `WaveReadLaneFirst` inside a divergent loop return other lanes' values. The spin is bounded, so a lock nobody frees can never hang the GPU: 2^22 tries in compute and mesh shaders, which macOS never ends for running long, and 65536 in pixel shaders, whose draws macOS ends after about 50 ms. A lane past the bound goes on without the lock and can lose its update, which 65536 tries allowed in compute under heavy contention (2,048 groups adding into one texel lost adds in 2 of 100 runs on an M5 Pro, #240). A shader the GPU ends while one of its lanes holds a lock never frees it, so after a submission the GPU ended VIctory frees every lock. Each lane that gives up adds one to a count after the lock words, and VIctory reads it after every submission and logs `lanes gave up waiting for a 64-bit atomic's lock` with the total, so a run shows whether the bound is ever reached. On an M2 Pro these locks cost about 6 s of a 7.4 s world frame (#240), almost all of it in the alpha prepasses' 64-bit `umax`. The rounds, on by default wherever the locks are (`VICTORY_ATOMIC_LOCK_ROUNDS=0` goes back to the turns alone), make every lane of a wave first try its lock once in each of four rounds, all at once, so lanes whose locks differ do not wait for each other; only lanes still not done take turns, and a wave whose lanes all finished skips the turns. No lane waits on a lock inside a round, so lanes of one wave still never wait for each other's locks. On an M5 Pro with D3DMetal 4.0 beta 3 a wave's 64-bit `umax` on distinct texels took 29 µs in turns and 6 µs in rounds (`atomic_lock_rounds_timing`, ignored, run under Wine). In the game the rounds took a world frame from 7.9 to 2.5 s on an M2 Max (sys-skye), from 5.29 to 0.71 s on an M4 Pro (evan-kolberg, the same binary with only the switch changed) and from 2.9 to 0.88 s on a 16 GB M4 (Stefa008), and the GPU resets and queue recoveries of the turns stopped (#240). `VICTORY_ATOMIC_UMAX=1` (also opt-in, only with the rounds) exploits what the prepasses' one operation is: a glc-clear `umax` needs no old value and the maximum is associative, so the wave first folds each texel's lanes to their maximum key, in registers, and one representative a group runs the rounds and the turns. A wave whose lanes meet on one texel locks once instead of 32 times; the reduction is one wave-uniform pass per group, its ballots and wave maxima over the lanes the guard let through. A glc-set `umax` reads its old value, so it keeps the general form. The form is exact — the representative applies the group's maximum key, which equals every member's contribution in any order — and #240's negative result stands: the cost was never lanes waiting on locks, it was the serialized sections themselves, which the reduction cuts at every shape, ties included. In pixel programs the rounds leave helper lanes out: their atomics are discarded, so they could never take a lock, and in their turn they would spin to the bound.
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

## CPU-visible tiles in reserved buffers

On an M1 Max with CrossOver Wine 11.17 and D3DMetal 4.0 beta 3, a reserved buffer stayed at one GPU address as its four 64 KiB tiles were mapped and remapped. The probe reads all four CPU-written tiles on the GPU, checks that a GPU write reaches the remapped tile's CPU memory, and verifies that a tile mapped to NULL reads as zeros. The host reported tiled resources tier 2. Its file-mapping heap API returned `E_NOTIMPL` at 64 KiB, 4 MiB and 256 MiB. [Direct3D 12 tile mapping](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-updatetilemappings)

Run the probe under the D3DMetal Wine runner with `tools/macos/test-wine.sh -p victory-gpu --lib inplace_probe_6_remap_cpu_visible_tiles -- --ignored --test-threads=1`.

## Plain 256-bit stores into protected pages stop Rosetta too

The masked-store patch is not enough on its own once fault tracking is on. Measured under CrossOver Wine on an M5 Max (macOS 26): a plain `vmovups`, `vmovdqu` or `vmovaps` of 32 bytes that begins in a writable page and faults in the next one stops Rosetta with the same "unexpectedly need to EmulateForward on a synchronous exception", before any handler runs. The same store wholly inside one page resumes. 128-bit stores, `vextractf128` to memory and smaller stores resume in every layout, and an AVX masked store of 128 bits stops it like one of 256. Series 1.0.2.1 has 119,296 plain 256-bit stores; #144 hit eight of them, which `VICTORY_PATCH_MASKED` now lists by hand.

`cargo xtask wide-stores <image> > wide-sites.txt` lists all of them from the mapped title image in the cache (`<cache_dir>/images/victory-image-<hash>.dll`) with `llvm-objdump`, in about two minutes, as offsets and lengths only. With `VICTORY_PATCH_WIDE=<list>`, `victory-loader/src/wide.rs` turns each into a jump to a trampoline that writes the low half with the same store at 128 bits and the high half with `vextractf128 [m+16], ymm, 1`. A store of four bytes has no room for the jump: its site takes in the instructions next to it that can be moved (copied, RIP-relative ones with their displacement redone, conditional jumps re-encoded), never across a branch target. For 1.0.2.1 that is 117,413 sites, with 35 four-byte stores left that have nothing movable beside them. Every site's bytes are checked before it is patched; one that does not match is left alone and counted.
## Mesh pipelines over 32 KiB hang the queue

On an M5 Max with D3DMetal 4.0 beta 3 the world-entry hang of #172 still happens on current main: a submission that holds the title's indirect mesh draws never completes ("still waiting for the host GPU" without end), right after the "Vice City, two years ago" card. Linking mesh and pixel outputs (#161) was off unless `VICTORY_LINK_MESH_OUTPUTS` was set, and nothing refused a pipeline still over the limit, so the oversized pipelines were made.

Two counts of a mesh shader's outputs disagree. `MeshOutputs::bytes` counts four bytes per declared component; the layout Metal allocates puts clip and cull distances first, then the position, then each other output aligned to its own size, the same for primitives, times the declared maxima (the count #172 used, from the older `metal_mesh_bytes`). For 1.0.2.1 the packed count puts every world mesh shader at or under 32,768 bytes, while the aligned layout puts about 20 of them at 34,816 to 38,400, and those are the pipelines in the submission that hangs.

Under Wine, linking is now on by default, and a mesh pipeline that is still over 32 KiB after it, by either count, is refused: its draws are dropped and the queue keeps running. With that, the same run goes past the title card and into the world; 20 to 25 pipelines are refused per run. `VICTORY_LINK_MESH_OUTPUTS=0` turns linking off and `VICTORY_MESH_OVER_LIMIT=1` makes over-limit pipelines anyway.
The diagnostics switch `VICTORY_PHYSICAL_TILES=1` lets XMem use CPU-visible D3D12 heaps for physical page allocations on Rosetta. Wine supplies its usual placeholder views, and the mapped view and DirectStorage share the heap's bytes. The XMem test checks page aliases, remapping, unmapping, and DirectStorage writes. The GPU probe also checks both directions of a copy through an aliased Windows view, including after that view is unmapped.

A separate 6.10 GiB heap and reserved buffer worked on the same M1 Max. Readback matched data from four sampled tiles, including the last two in the heap. A GPU write reached the final tile's CPU mapping.

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

The same recovery covers GPU page faults. On an M2 Pro with macOS 27.2.1, cocobailey ran the game with #178. All 11 `gpuEvent-wine` reports from that run say `MMU interrupt` with signature 675, not the lockups with signature 579 seen in #173. The queue was replaced 8 times in 1 min 44 s, each new queue ran work, and the picture kept updating. The warning names both causes.

## Deferred UAV-to-SRV transitions

On D3DMetal 4.0 beta 3, a test writing and then reading a 3D texture returned some zero texels when both command lists and the consumer's transition prelude were submitted in one native call ([#242](https://github.com/pragyaangaur/VIctory-macOS/issues/242)). `VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE=1` starts a separate native queue call at a deferred UAV-to-SRV prelude when earlier work is already in the batch. The calls use the same queue and share one final fence wait. Each native call still has a completion mark so an incomplete earlier call reaches the existing queue recovery path. `VICTORY_GPU_TIMING=1` already submits each list separately and takes precedence.

## Indirect dispatches wider than 65,535 groups

The title's tile passes are indirect dispatches of one group per 8x4 tile of the picture: 67,440 and 47,760 groups for its two tile classes in one frame at 2560x1440, as measured on Windows in VIctory-7900XTX (84eae35). The console runs dispatches that wide. The argument clamp (`victory-gpu/src/indirect.rs`) cut every dispatch to D3D12's limit of 65,535 groups in x, so the last tiles of such a pass, the lowest of the picture, never ran. On an M5 Pro with D3DMetal 4.0 beta 3, an indirect dispatch sent to the host as it is ran every one of its groups at 65,536, 67,440, 131,072 and 2^20 groups wide, while through the clamp each ran 65,535. Under Wine the clamp now lets dispatches up to 2^20 groups wide in x run whole (`indirect::widest`, `VICTORY_INDIRECT_WIDE=0` or `1` overrides); 2^20 groups in all still bound a dispatch, and other hosts keep 65,535. `a_wide_tile_pass_runs_every_group_the_host_runs` fails if a later D3DMetal stops running them.

## The clwb write-back loops

The title has eight loops that write cache lines back with `clwb`, which faults under Rosetta and costs an exception each time (#135). Under Rosetta, the same check that decides fault tracking, `victory` turns the `clwb` of each loop into a NOP when it loads the image. It patches only when it finds exactly eight loops and logs the number it found otherwise. `VICTORY_NOP_CLWB=0` leaves the loops alone and `VICTORY_NOP_CLWB=1` patches them where Rosetta is not detected.

## An ambient light in place of ray-traced indirect light

`VICTORY_AMBIENT=1` turns on a stand-in from Belobug's fork (`victory-gpu/src/ambient.rs`). It is off by default, and off it changes nothing. The game lights what the sun does not reach with probes it fills by tracing rays. Right after the game's program that resolves those probes (`CS_ScreenProbes_TemporalResolve_*`), a compute pass of VIctory's own replaces that program's output with an ambient light: the same everywhere in a frame, darker in creases and under objects (worked out from the depth image). By default its level and colour follow the sky dome the game draws for its reflections. After the game's last reflection pass (`CS_UpdateHistory`), a second pass of the same shader holds reflections to half of that light.

VIctory logs one line at start when the switch is on, one line for each program it found by name, and one line the first time each part runs (`ambient light: the first image it writes`, `ambient light follows the sky dome`, `reflections held to the ambient light`). `no ambient light: ...` says why a pass did not run.

| Variable | Effect |
| --- | --- |
| `VICTORY_AMBIENT=1` | Turns the stand-in on. Unset, empty, `0` and `off` leave the game's own output. |
| `VICTORY_AMBIENT_SKY=off` or `gain,least,power` | `off` keeps a constant level. Otherwise the level is `gain * mean^power + least` of the sky dome's mean (0.36, 0.05, 0.4). |
| `VICTORY_AMBIENT_LEVEL=<level>` | The constant level (0.6), used while the light does not follow the sky. |
| `VICTORY_AMBIENT_TINT=r,g,b` | The constant light's colour (1, 0.98, 0.94). |
| `VICTORY_AMBIENT_SHADE=s,r,n,t` | The shading: strength (1; 0 for none), radius in metres (0.6), near plane (0.1), tangent of half the vertical field of view (0.45). |
| `VICTORY_AMBIENT_REFLECT=<part>` | The most a reflection may hold, as a part of the light (0.5). 0 leaves reflections alone. |
| `VICTORY_AMBIENT_FILE=<file>` | Diagnostics builds: the file is read twice a second, to tune a running game. `gain least [power] [s r n t]` follows the sky; `level r g b [s r n t [part [least]]]` sets a constant light. |

In the fork `VICTORY_AMBIENT` is the level and the stand-in is on unless it says `off`. Here it is a switch, and the level has its own variable.

The pass is compiled at run time, so under Wine its shader keeps to what Wine's compiler takes ("Run-time HLSL and textures under Wine and D3DMetal" above): its root signature is built in code, and the image sizes are constants, because Wine's compiler has no `GetDimensions` on unordered views (`RWTexture2D`): the fork's shader stops there with `E5005: Method 'GetDimensions' is not defined on type 'RWTexture2D<uint>'`. It finds the game's programs by the names in their shader containers and the images by fixed places in their tables, so it only fits this game's shader build. It changes no shader translation and no cached pipeline: turning it on or off leaves both caches warm.

### The estimate: `VICTORY_AMBIENT=estimate`

`VICTORY_AMBIENT=estimate` (or `estimate:<share>`) turns on another stand-in in the same place, from avatarmanz's fork (`victory-gpu/src/ambient_estimate.rs`). Every other value of the variable means what it meant before, and the constant stand-in above is unchanged. The constant one gives the whole picture one light. The estimate works the light out from the frame, so it follows the hour, the room and its lamps by itself:

- The game keeps the light its visible surfaces send out, for its own screen traces (`CS_CacheScreenSpaceRadiance`, an R11G11B10 image of half the picture's size). Right after that program, in the same host list, VIctory averages the image in 64x36 cells of a buffer of its own, and moves a slow mean of all cells toward this frame's.
- Right after the program that resolves the screen probes, in the next frame, VIctory places the cells in space by the depth image the resolve reads. At 160x90 points of the picture it sums the cells' light, each cell weighted by the share of the point's surroundings it covers, as a mean and a direction per colour. Every texel of the resolve's light image then gets what lies between the four points around it (those at its own depth first), turned by its normal, a fifth of it the slow mean, darker in creases.
- After the game's last reflection pass, reflections are held to a part of that light, as with the constant stand-in.

It assumes that what is not on screen sends out what is. It has no occlusion but the creases'. Where little light is on screen it has nothing to spread: a room lit only by its own bounce (the windows off screen, no lamp, no patch of sun) settles at black, because each frame's light is the walls' colour times the last frame's, and a night street stays black. So no texel gets less than a floor: a share of the constant stand-in's light above, worked out once a frame from the same sky dome.

| Variable | Effect |
| --- | --- |
| `VICTORY_AMBIENT=estimate` or `estimate:<share>` | Turns the estimate on, with that share of it (1). A share that is not a number above 0 counts as 1. |
| `VICTORY_AMBIENT_SETTLE=<ms>[,<jump>[,<fresh s>]]` | How the slow mean settles: by a time constant in milliseconds, a change by more than `jump` times at once (unset: `250,3`). `0`: 0.04 of the way a frame, which is about 8 s at 3 frames a second. |
| `VICTORY_AMBIENT_FLOOR=<share>` | The least any texel gets, as a share of the constant stand-in's light (0.3). With the sky followed that is `share * gain * mean^power + least`: the sky's part cut to the share, the least level (the night's) whole; `VICTORY_AMBIENT_SKY`, `_LEVEL` and `_TINT` say what that light is. 0: no floor, as in the fork. 1: never darker than the constant stand-in. |
| `VICTORY_AMBIENT_LEFT=<multiple>` | How much of what the game's resolve left counts: up to that many times the estimate's luminance, and the larger of the two is written. Unset or 0: the estimate replaces it. The fork has 4. With it, `fresh` seconds after a jump the leftover counts for no more than the estimate. |
| `VICTORY_AMBIENT_AIR=<multiple>` | Bounds what the air scatters by that multiple of the light around, right after `CS_AccumulateLocalLightInscatteringVolume`. Unset or 0: the game's own. Without rays the air of a room is lit by the sky's light. |
| `VICTORY_AMBIENT_SHADE=s,r,n,t` | The shading in creases, as above. |
| `VICTORY_AMBIENT_MIX=<mean part>,<turn>,<least>,<most>` | How a texel's light is put together: the part of it that is the slow mean of the whole picture (0.2), how strongly its normal turns it toward where the light comes from (1.2), and the least and most it may be as multiples of the mean around it (0.2 and 2.5). These are the fork's fixed numbers. The sky's cells lie far away along the view, so a surface that faces the camera has the sky behind it: a larger mean part or a smaller turn gives it more of the sky. |
| `VICTORY_AMBIENT_REFLECT=<part>` | The most a reflection may hold, as a part of the light around the point seen there (0.5). 0 leaves reflections alone, as the fork does. |

`VICTORY_AMBIENT_SKY`, `_LEVEL`, `_TINT` and `_FILE` belong to the constant stand-in; with the estimate they only say what the floor is a share of. `VICTORY_AMBIENT_LEFT` is no floor: it bounds what the game's resolve left by a multiple of the estimate, so where the estimate is next to nothing it keeps next to nothing.

VIctory logs one line at start (`VICTORY_AMBIENT=estimate: ...`), one line for each program it found by name (the resolve, `keeps the light visible surfaces send out`, and the two others), and one line the first time each part runs (`ambient estimate: the first light it keeps`, `the first image it writes`, `reflections held to the light around`, `the first volume of air it bounds`). With every frame line it logs what the estimate is:

```
ambient estimate (VICTORY_AMBIENT=estimate): the light around (slow mean) is 0.2000 0.1000 0.0500, luminance 0.1177; the picture's 160x90 points hold a luminance of 0.3000 on average (least 0.0100, most 2.0000); the floor (VICTORY_AMBIENT_FLOOR=0.3) is 0.2500, and 60% of the points are under it; texels are written about 0.3100 (the share 1 of the estimate, or the floor; the constant stand-in writes its level); 1.0 keepings, 2.0 raisings, 1.0 holds of reflections and 0.0 bounds of the air/frame
```

The number after `texels are written about` is what to compare with the constant stand-in's level. The part of the points under the floor says how much of the picture is the floor's flat light: nearly all of it in a room with nothing lit on screen, little where the estimate has light to work with. `nothing kept yet` means the game has not run `CS_CacheScreenSpaceRadiance`: until it does, the game's own light stays.

Under Wine the passes' shader keeps to what Wine's compiler takes. Besides what the constant stand-in's shader avoids, it does not store through `RWStructuredBuffer` (each buffer is read through a structured view and written through a raw one, as the depth converter does), and it has one `main` per pass, chosen by a `#define`. It reads the R11G11B10 image through a shader resource view of its own, because Metal has no R11G11B10 texture that one view reads and writes. The air's pass does read and write the R11G11B10 volume through one unordered view, as the game's own program just before it does.

Per frame it adds two small dispatches after `CS_CacheScreenSpaceRadiance` (2,304 and 1 threads), three after each resolve (2,304 threads, 14,400 threads that each walk the 2,304 cells, and one thread per texel of the light image), and one per texel of the reflection image. The constant stand-in adds one per resolve and one for the reflections.

## Title buffers on D3DMetal's pages

Under Rosetta with D3DMetal, memory the title commits read-write with `XMemVirtualAlloc`, in whole 64 KiB pages, is pointed at the pages of a D3DMetal buffer with `mach_vm_remap` (#204). The GPU then reads and writes the title's own memory, so those buffers need no host copy, no copy-back after a submission that wrote them, and no refresh before one. A probe at start-up falls back to the copies if the remap does not work, and `VICTORY_ALIAS_BUFFERS=0` keeps the copies. On this line it is off unless `VICTORY_ALIAS_BUFFERS=1`, and it needs `VICTORY_COPY_ORDER=split` (refused otherwise, with a line that says why): title copies are made on the CPU before the GPU runs the submission, and in aliased memory only the split's cuts keep them behind the commands recorded before them. Here GPU writes into aliased memory mark the other trackers without opening their write guards, memory the title frees or decommits through the host and commits again is pointed back at its buffer, and the end of the frame line counts what took the zero-copy path against the copy path.

## Optional switches for slow storage, low memory and speed

None of these changes the default. Each is read from the environment when the game starts, and all of them are tools for testers, so please post what you see on #131 or #135 when you try one.

| Switch | What it does |
| --- | --- |
| `VICTORY_NO_HANG_WATCHDOG=1` | Turns off the title's hang watchdog. Once a second it crashes the title on purpose when a thread's heartbeat is too old or the main thread has not finished a frame for 30 checks, and a world load from slow storage outlasts that (#135). It applies only to the known Series build 1.0.2.1 and only where the original bytes match. |
| `VICTORY_DSTORAGE_WORKERS=<n>` | Reads with `n` workers (1 to 256) instead of one per core, for storage with a long wait per read such as a network share. |
| `VICTORY_EVICT_HEAP_UAV=1` | Lets textures viewed for unordered access in shader-visible heaps be given up until a shader writes through a heap, then made again. About 1.5 GB on an 8 GB Mac (#135). |
| `VICTORY_SPILL_SHADERS=1` | Keeps shader containers on disk until they are translated. The title passes tens of thousands of containers and draws with a few hundred, and holding them all took about 700 MB of an 8 GB Mac (#135). |
| `VICTORY_RAM_LOW=<share>`, `VICTORY_RAM_FREE=<share>` | Set when host textures are given up for the computer's free memory, each as a share of its memory from 0 to 0.5: below `VICTORY_RAM_LOW` free (0.10 by default), until `VICTORY_RAM_FREE` is (0.15, or 0.05 above a `VICTORY_RAM_LOW` set alone). On a 24 GB Mac macOS was already compressing about 800 MB a second with 14 % free, and 0.20 and 0.26 brought the wired peak from 13.9 to 12.9 GB and a frame in the playable yard from 516 to about 490 ms. More textures are given up and made again (#454). |
| `VICTORY_READ_STATS=1` | Logs the requests and bytes read from each file, every 30 seconds, for the 30 most read. It shows which files a slow share serves most, so you can copy those to fast local storage. |
| `VICTORY_GPU_BUDGET_MB=<MB>` | Sets the budget that host textures are kept under, in place of the device's. |
| `VICTORY_PATCH_WIDE=<list>` | Splits the 256-bit stores listed by `cargo xtask wide-stores` into 128-bit halves, so fault tracking does not stop Rosetta on them (see above). |
| `VICTORY_MESH_OVER_LIMIT=1` | Makes mesh pipelines whose outputs are over Metal's 32 KiB anyway (refused under Wine by default, as they hang the queue; see above). |
| `VICTORY_OUTPUT=<W>x<H>` | Sets the title's output size, one of its own modes: 1280x720, 1664x936, 1920x1080, 2560x1440 or 3840x2160. It patches the title's code at load, only where the bytes match build 1.0.2.1, and logs `resolution: ... Done` for each site. On a 16 GB M4 (Series X), 1280x720 brought the GPU time of a world frame from about 8.4 s to 0.86 s. Ported from avatarmanz/VIctory-7900XTX. |
| `VICTORY_UI_SIZE=<W>x<H>` | One of the title's modes, written into the largest entry of its display mode list (3840x2160), which the interface is sized from. With `VICTORY_OUTPUT=1920x1080` alone the button prompts and subtitles are laid out for 3840x2160 and drawn twice too large; `VICTORY_UI_SIZE=1920x1080` with it is meant to lay them out for the output. Off by default; not yet confirmed in a game run. |
| `VICTORY_RENDER_SCALE=<s>` | 0.5 to 1.0: the title's own upscaler's render scale (2/3 by default, Series X). Not yet tested on a Mac. |
| `VICTORY_UPSCALER=<6\|7>` | Makes the title pick its own upscaler (6) or FidelityFX (7). Not yet tested on a Mac. |
| `VICTORY_STEP_OVER=pool` | Steps over the title's "no free element" traps: an `int3` followed by `xor r, r`, the shape of a full fixed-size pool's allocator, whose caller then gets the NULL it already checks for. Series 1.0.2.1 hits one at `+0x5149aec` (the instance lists' pool, Marvisak/VIctory#698) about 20 minutes into the opening when frames take seconds. A list of hex offsets also works. |
| `VICTORY_WINDOW_SECONDARY=1` | Opens the game window on the only non-primary display. |
| `VICTORY_WINDOW_SECONDARY=x,y` | Opens the game window on the display that holds the desktop point `x,y`, for layouts with more than one extra display (a Sidecar iPad beside two monitors, say). |

To find what the estimate costs on the GPU in a scene, `VICTORY_AMBIENT_PROBE=<n>` leaves its passes out in turn, each step for `n` frame lines: nothing, the gather, the creases' shade, the reflections' hold, the keeping, the raising, every pass. The estimate's line ends with the step the frame line above it was measured with (`probe (VICTORY_AMBIENT_PROBE) step 1 of 7: these frames were recorded without the gather ...`), so one run gives the frame line's GPU time with and without each part. The picture changes with the steps. Each pass is also named among its list's pipelines (`[ambient estimate: gather]` and so on), so a `slow host submission` or `slow host list` line says when a slow list holds one. In a diagnostics build with `VICTORY_GPU_TIMING=1` and `VICTORY_D3D12_DRED=1`, a slow list also logs `slow list commands by kind`, since GPU timestamps read 0 under D3DMetal.
