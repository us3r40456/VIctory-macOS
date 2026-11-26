#!/bin/sh
# Stage the fixed activation tracer in a private directory for this run.
set -eu
saved_umask=$(umask)
umask 077

if [ "$#" -lt 2 ]; then
    echo "usage: trace-activation.sh WINE victory.exe [arguments...]" >&2
    exit 2
fi
wine=$1
shift

for tool in curl shasum xz; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "activation tracing requires $tool on PATH" >&2
        exit 1
    fi
done

asset=frida-gadget-17.5.1-windows-x86_64.dll.xz
asset_hash=533cce66a09f603cca0e25306ba3fd41550711f4434d3363e947d308a22a7f73
cache="${XDG_CACHE_HOME:-$HOME/Library/Caches}/victory-macos/frida"
cached_file="$cache/$asset"
download=
session=
guest=

cleanup() {
    if [ -n "$download" ]; then rm -f -- "$download"; fi
    if [ -n "$session" ]; then rm -rf -- "$session"; fi
}

interrupted() {
    trap '' HUP INT TERM
    # Use $! as well if the signal arrived just before storing the guest PID.
    running=${guest:-${!:-}}
    if [ -n "$running" ]; then
        kill -TERM "$running" 2>/dev/null || :
        wait "$running" || :
    fi
    exit "$1"
}

trap cleanup EXIT
trap 'interrupted 129' HUP
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

matches_hash() {
    result=$(shasum -a 256 < "$1") || return 1
    [ "${result%% *}" = "$2" ]
}

mkdir -p "$cache"
if [ -L "$cached_file" ] || { [ -e "$cached_file" ] && [ ! -f "$cached_file" ]; }; then
    echo "activation trace cache entry is not a regular file: $cached_file" >&2
    exit 1
fi
if [ ! -f "$cached_file" ] || ! matches_hash "$cached_file" "$asset_hash"; then
    download=$(mktemp "$cache/.download.XXXXXX")
    curl --fail --silent --show-error --location --retry 2 \
        --proto '=https' --proto-redir '=https' --output "$download" \
        "https://github.com/frida/frida/releases/download/17.5.1/$asset"
    if ! matches_hash "$download" "$asset_hash"; then
        echo "activation trace download failed the pinned SHA-256 check" >&2
        exit 1
    fi
    # Every writer validates the same asset before publishing it atomically.
    mv -f -- "$download" "$cached_file"
    download=
fi

session=$(mktemp -d "${TMPDIR:-/tmp}/victory-activation.XXXXXX")
xz -dc -- "$cached_file" > "$session/frida-gadget.dll"

session_path=$(CDPATH='' cd "$session" && pwd -P)
guest_path=$(printf '%s' "$session_path" | tr / '\\')
export VICTORY_ACTIVATION_SESSION="Z:$guest_path"

# Keep the guest's standard streams attached while waiting so we can clean up.
umask "$saved_umask"
exec 3<&0
"$wine" "$@" <&3 3<&- &
guest=$!
exec 3<&-
status=0
wait "$guest" || status=$?
exit "$status"
