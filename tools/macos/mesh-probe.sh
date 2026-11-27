#!/bin/sh
# Measures how D3DMetal sizes the outputs of mesh shaders: the probe in
# crates/victory-shader/src/emit/mesh_probe_tests.rs, run under Wine. It
# needs no game and no game data, and opens no window.
# Usage: tools/macos/mesh-probe.sh [file of signature lines]
# Without a file it measures the probe's built-in sweep. The table goes to
# standard output and to target/mesh-probe/table.txt; the probe's output
# with D3DMetal's own messages is kept in target/mesh-probe/output.log.
# WINE names the wine binary (one with D3DMetal, as for victory.sh) and
# WINEPREFIX the prefix (default target/wineprefix-probe).
# VICTORY_MESH_PROBE_DRAW=1 also draws with each line's pipeline.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
# Homebrew's rustup is keg-only (see test-wine.sh).
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
WINE="${WINE:-$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine}"
[ -x "$WINE" ] || { echo "set WINE to a wine binary with D3DMetal" >&2; exit 1; }
TARGET=x86_64-pc-windows-gnullvm
OUT="$ROOT/target/mesh-probe"
mkdir -p "$OUT"
# The test binary, from the line cargo prints for it.
EXE="$(cargo test --no-run -p victory-shader --lib --target "$TARGET" 2>&1 |
    sed -n 's/^ *Executable unittests src\/lib.rs (\(.*\))$/\1/p')"
[ -f "$ROOT/$EXE" ] || { echo "the test binary did not build: cargo test --no-run -p victory-shader --lib --target $TARGET" >&2; exit 1; }
# A game running on the same GPU needs most of the memory: wait for it.
while pgrep -f 'release/victory\.exe' > /dev/null; do
    echo "mesh-probe.sh: victory.exe is running; waiting for it to end" >&2
    sleep 60
done
windows() { printf 'Z:%s' "$1" | tr / '\\'; }
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-probe}"
export WINEDEBUG="${WINEDEBUG:--all}"
# No Mono or Gecko install prompts and no crash dialog (as test-wine.sh),
# and D3DMetal's D3D12 (as victory.sh).
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
: > "$OUT/output.log"
rm -f "$OUT/table.txt"
# D3DMetal prints its messages on the Unix standard output: the probe
# reads them back from this file.
VICTORY_MESH_PROBE_LOG="$(windows "$OUT/output.log")"
VICTORY_MESH_PROBE_OUT="$(windows "$OUT/table.txt")"
export VICTORY_MESH_PROBE_LOG VICTORY_MESH_PROBE_OUT
if [ -n "$1" ]; then
    VICTORY_MESH_PROBE_SIGS="$(windows "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")")"
    export VICTORY_MESH_PROBE_SIGS
fi
"$WINE" "$ROOT/$EXE" --ignored --exact \
    emit::mesh_probe_tests::gpu::mesh_output_sizes_as_d3dmetal_counts_them \
    --test-threads=1 --nocapture >> "$OUT/output.log" 2>&1 || {
    # D3DMetal does not survive every signature: the rows before it, and
    # the line it was at.
    [ -f "$OUT/table.txt" ] && cat "$OUT/table.txt"
    echo "mesh-probe.sh: the probe stopped at \"$(grep '^measuring: ' "$OUT/output.log" | tail -1 | cut -c12-)\"; see $OUT/output.log" >&2
    exit 1
}
[ -f "$OUT/table.txt" ] || { echo "mesh-probe.sh: no table; see $OUT/output.log" >&2; exit 1; }
cat "$OUT/table.txt"
