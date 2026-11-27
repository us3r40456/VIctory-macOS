#!/bin/sh
# Builds VIctory for Windows (gnullvm) on macOS and runs its tests under Wine.
# Usage: tools/macos/test-wine.sh [cargo test arguments]
# WINE names the wine binary. Cargo splits the runner on spaces, so the
# runner is a wrapper script in a temporary folder without spaces.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
# Homebrew's rustup is keg-only, so another cargo, such as Homebrew's rust,
# can come first on PATH. That one has no gnullvm standard library, and the
# first build below fails. victory.sh puts rustup first in the same way.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
# Unset, WINE is this checkout's own Wine (setup.sh builds it into
# target/wine-cx), else the one behind the space-free link in the cache
# folder (a checkout of its own that ran setup, such as a worktree's main
# checkout), else a wine on PATH. The link names whichever checkout ran
# setup last, a fleet task's too, so this checkout's own comes first.
if [ -z "${WINE:-}" ]; then
    for wine in "$ROOT/target/wine-cx/bin/wine" "$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"; do
        [ -x "$wine" ] && { WINE="$wine"; break; }
    done
fi
WINE="${WINE:-$(command -v wine || true)}"
[ -x "$WINE" ] || { echo "set WINE to a wine binary" >&2; exit 1; }
TARGET=x86_64-pc-windows-gnullvm
mkdir -p target/macos
# A wrapper rather than a symlink: some Wine builds find their loader next
# to the path they were started from.
RUNNER_DIR="$(mktemp -d /tmp/victory-wine.XXXXXX)"
LINK="$RUNNER_DIR/wine"
trap 'rm -f "$LINK" && rmdir "$RUNNER_DIR" || true' EXIT
printf '#!/bin/sh\nexec "%s" "$@"\n' "$WINE" > "$LINK"
chmod +x "$LINK"
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix}"
export WINEDEBUG="${WINEDEBUG:--all}"
# A new prefix would wait on hidden Mono and Gecko install prompts, and the
# tests need neither. Wine's crash dialog is turned off for the same reason.
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
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
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_RUNNER="$LINK"
cargo build --quiet --manifest-path tests/fixtures/hle-probe/Cargo.toml \
    --target-dir target/fixtures --target "$TARGET"
VICTORY_HLE_PROBE="Z:$(printf '%s' "$ROOT" | tr / '\\')\\target\\fixtures\\$TARGET\\debug\\hle_probe.dll"
export VICTORY_HLE_PROBE
cargo test --target "$TARGET" "$@"
