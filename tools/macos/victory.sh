#!/bin/sh
# Builds VIctory for Windows and runs it on macOS under CrossOver Wine with
# Apple's D3DMetal. Arguments go to victory.exe, for example:
#   tools/macos/victory.sh config
#   tools/macos/victory.sh run --execute <path to the game's executable>
# WINE defaults to the Wine that tools/macos/build-crossover-wine.sh builds.
# VICTORY_DIAGNOSTICS=1 builds with the diagnostics feature (debug logs,
# black-frame reports, frame capture and the other debugging switches).
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
WINE="${WINE:-$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine}"
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ -d "$(dirname "$WINE")/../lib/external/D3DMetal.framework" ] || {
    echo "D3DMetal is missing: copy the redist/lib folder of Apple's evaluation environment over $(dirname "$WINE")/../lib" >&2
    exit 1
}
build_victory() {
    cargo build --quiet --release -p victory --bin victory --target x86_64-pc-windows-gnullvm \
        --message-format=json-render-diagnostics "$@"
}
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ] || [ "${VICTORY_TRACE_ACTIVATION:-0}" = 1 ] || [ "${VICTORY_TRACE_INIT:-0}" = 1 ]; then
    artifacts=$(build_victory --features diagnostics)
else
    artifacts=$(build_victory)
fi
# Cargo may place the executable outside the checkout (e.g. build.target-dir).
# Read its actual compiler artifact, including when the build was already fresh.
victory=$(printf '%s\n' "$artifacts" | python3 -c '
import json, sys
for message in map(json.loads, sys.stdin):
    if (message["reason"] == "compiler-artifact" and message["target"]["name"] == "victory"
            and "bin" in message["target"]["kind"] and message.get("executable")):
        print(message["executable"])
        break
else:
    sys.exit("Cargo did not report the victory executable")
')
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-play}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
# Rosetta tells the game about AVX and AVX2, which it may check for.
export ROSETTA_ADVERTISE_AVX=1
if [ "${VICTORY_TRACE_ACTIVATION:-0}" = 1 ] || [ "${VICTORY_TRACE_INIT:-0}" = 1 ]; then
    exec sh "$ROOT/tools/macos/trace-activation.sh" "$WINE" "$victory" "$@"
fi
exec "$WINE" "$victory" "$@"
