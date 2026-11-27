#!/bin/sh
# Runs a captured frame again (docs/replay.md) on macOS: builds VIctory for
# Windows and runs `victory replay` under CrossOver Wine with Apple's
# D3DMetal, in a Wine prefix of its own, never the one a title runs in.
#   tools/macos/replay.sh <capture folder> [replay arguments]
# Paths among the arguments (the capture, --shaders, --retranslate, --image)
# are macOS paths; they are handed to Wine as Z: paths.
# WINE defaults to the Wine that tools/macos/build-crossover-wine.sh builds,
# WINEPREFIX to target/wineprefix-replay. VICTORY_DIAGNOSTICS=1 runs the
# build with the diagnostics feature (the one a recording run uses). The
# VICTORY_* switches of the translator are passed on (for --retranslate).
# The GPU is shared: do not run this while a title runs.
#
# In a checkout that has ../shared/wine-cx beside it (several checkouts with
# one Wine and one title run at a time, started by ../shared/run-game.sh):
# WINE defaults to that Wine, WINEPREFIX to ../state/<checkout>/wineprefix-replay
# (outside target/, which `cargo clean` empties), and the build and the
# replay run under the lock a title's run holds (../shared/game.lock): a
# replay of a title's frame takes most of the memory a title's run does. It
# waits for a run that is going on, and a run started meanwhile waits for it
# (not for --summary or --listing, which need no GPU).
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SHARED="$(dirname "$ROOT")/shared"
if [ -x "$SHARED/wine-cx/bin/wine" ]; then
    WINE="${WINE:-$SHARED/wine-cx/bin/wine}"
    WINEPREFIX="${WINEPREFIX:-$(dirname "$ROOT")/state/$(basename "$ROOT")/wineprefix-replay}"
    mkdir -p "$(dirname "$WINEPREFIX")"
    export WINE WINEPREFIX
    # A summary or a listing reads the capture without a GPU: no lock.
    case " $* " in
        *" --summary "* | *" --listing "*) VICTORY_REPLAY_LOCKED=1 ;;
    esac
    if [ -z "$VICTORY_REPLAY_LOCKED" ] && [ -f "$SHARED/game.lock" ] && command -v zsh > /dev/null; then
        # The lock is the starting shell's (zsh closes it in what it starts)
        # and goes when that shell ends.
        export VICTORY_REPLAY_LOCKED=1
        # shellcheck disable=SC2016
        exec zsh -c '
            zmodload zsh/system || exit 2
            if ! zsystem flock -t 0 -f fd "$1" 2> /dev/null; then
                print -u2 "replay.sh: the lock of a title run is held ($(cat "$2" 2> /dev/null || print "owner unknown")). Waiting for it..."
                zsystem flock -f fd "$1" || exit 2
            fi
            print -r -- "replay (pid $$) since $(date "+%Y-%m-%d %H:%M:%S")" > "$2"
            shift 2
            "$@"' replay "$SHARED/game.lock" "$SHARED/game.owner" "$ROOT/tools/macos/replay.sh" "$@"
    fi
fi
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
WINE="${WINE:-$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine}"
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ $# -ge 1 ] || { echo "usage: tools/macos/replay.sh <capture folder> [replay arguments]" >&2; exit 1; }
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ]; then
    cargo build --quiet --release -p victory --features diagnostics --target x86_64-pc-windows-gnullvm
else
    cargo build --quiet --release -p victory --target x86_64-pc-windows-gnullvm
fi
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-replay}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export ROSETTA_ADVERTISE_AVX=1
# An argument that is an absolute macOS path becomes a Z: path.
EXE="$ROOT/target/x86_64-pc-windows-gnullvm/release/victory.exe"
set -- replay "$@"
n=$#
while [ "$n" -gt 0 ]; do
    a=$1
    shift
    case "$a" in
        /*) a="Z:$(printf '%s' "$a" | tr / '\\')" ;;
    esac
    set -- "$@" "$a"
    n=$((n - 1))
done
exec "$WINE" "$EXE" "$@"
