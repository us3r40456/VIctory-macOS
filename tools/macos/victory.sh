#!/bin/sh
# Builds VIctory for Windows and runs it on macOS under CrossOver Wine with
# Apple's D3DMetal. Arguments go to victory.exe, for example:
#   tools/macos/victory.sh config
#   tools/macos/victory.sh run --execute <path to the game's executable>
# WINE defaults to the Wine that tools/macos/build-crossover-wine.sh builds:
# this checkout's own (target/wine-cx), else the one behind the space-free
# link in ~/Library/Caches/victory-macos. Every checkout's setup points that
# link at its own target/ (a fleet task's too), so it names whichever ran
# setup last; only the build needs it.
# VICTORY_DIAGNOSTICS=1 builds with the diagnostics feature (debug logs,
# black-frame reports, frame capture and the other debugging switches).
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
TARGET=x86_64-pc-windows-gnullvm
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if [ -z "${WINE:-}" ]; then
    WINE="$ROOT/target/wine-cx/bin/wine"
    [ -x "$WINE" ] || WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"
fi
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ -d "$(dirname "$WINE")/../lib/external/D3DMetal.framework" ] || {
    echo "D3DMetal is missing: copy the redist/lib folder of Apple's evaluation environment over $(dirname "$WINE")/../lib" >&2
    exit 1
}
# The 64-thread argument clamp is compiled at build time with the DXC that
# D3DMetal ships: CrossOver's own compiler refuses it, and without it the
# one-thread form runs instead. A VICTORY_HELPER_DXC already set wins.
DXC_LIB="$(dirname "$WINE")/../lib/external/D3DMetal.framework/Versions/A/Resources/libdxcompiler.dylib"
if [ -z "${VICTORY_HELPER_DXC:-}" ] && [ -f "$DXC_LIB" ]; then
    # A build product of this checkout, as its target/ holds them.
    helper="$ROOT/target/dxc-d3dmetal"
    mkdir -p "$ROOT/target"
    if [ ! -x "$helper" ] || [ tools/macos/dxc-d3dmetal.sh -nt "$helper" ]; then
        sh tools/macos/dxc-d3dmetal.sh "$helper" || helper=
    fi
    if [ -n "$helper" ]; then
        export DXC_LIB VICTORY_HELPER_DXC="$helper"
    fi
fi
# Build the Route A.5 Metal unixlib bridge when Wine headers are present so
# WINEDLLPATH exposes victory_metal.dll + victory_metal.so with no Wine rebuild.
wine_root="$(cd "$(dirname "$WINE")/.." && pwd)"
metal_dir="$ROOT/target/victory-metal-unixlib"
if [ -f "$wine_root/include/wine/unixlib.h" ] && [ -x "$wine_root/bin/winegcc" ]; then
    if [ ! -f "$metal_dir/x86_64-windows/victory_metal.dll" ] \
        || [ ! -f "$metal_dir/x86_64-unix/victory_metal.so" ] \
        || [ tools/macos/victory-metal-unixlib.sh -nt "$metal_dir/x86_64-unix/victory_metal.so" ]; then
        sh tools/macos/victory-metal-unixlib.sh "$metal_dir" "$wine_root" >/dev/null || true
    fi
    if [ -f "$metal_dir/x86_64-windows/victory_metal.dll" ]; then
        export WINEDLLPATH="$metal_dir${WINEDLLPATH:+:$WINEDLLPATH}"
        VICTORY_METAL_DLL="Z:$(printf '%s' "$metal_dir" | tr / '\\')\\x86_64-windows\\victory_metal.dll"
        export VICTORY_METAL_DLL
    fi
fi
metal_ir="$wine_root/lib/external/D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib"
if [ -z "${VICTORY_METAL_IR_LIB:-}" ] && [ -f "$metal_ir" ]; then
    export VICTORY_METAL_IR_LIB="$metal_ir"
fi
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ]; then
    cargo build --quiet --release -p victory --features diagnostics --target "$TARGET"
else
    cargo build --quiet --release -p victory --target "$TARGET"
fi
# Ask Cargo where it built the executable, including any configured target directory.
metadata=$(cargo metadata --format-version 1 --no-deps)
target_dir=$(printf '%s\n' "$metadata" | /usr/bin/plutil -extract target_directory raw -o - -)
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-play}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
# Rosetta tells the game about AVX and AVX2, which it may check for.
export ROSETTA_ADVERTISE_AVX=1
exec "$WINE" "$target_dir/$TARGET/release/victory.exe" "$@"
