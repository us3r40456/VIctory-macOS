#!/bin/sh
# Starts the game with the settings main-mac is measured with (see README.md).
# Usage: tools/macos/play-main-mac.sh <the game's executable> [more victory arguments]
# Any switch you set yourself before running this wins over the defaults below.
[ -n "$1" ] || { echo "usage: $0 <the game's executable>" >&2; exit 2; }
exe="$1"; shift
d() { eval "[ -n \"\${$1+x}\" ]" || export "$1=$2"; }
d VICTORY_PATCH_MASKED all;        d VICTORY_FAULT_TRACKING 1
d VICTORY_HALVES_LOCAL 1;          d VICTORY_HALVES_SHARED_LDS 1
d VICTORY_HALVES_EXCHANGE 1;       d VICTORY_HALVES_WALK 1
d VICTORY_GUARD_LOOKUP 1;          d VICTORY_AMBIENT estimate
d VICTORY_MAPDATA_BUDGET 40;       d VICTORY_COPY_ORDER split
d VICTORY_BATCH_CLAMPS 1;          d VICTORY_UAV_SIZE_LEVEL0 07f451a14b13fecb
d VICTORY_TEXTURE_BACK 256x256f41,256x256f16,64x64f16,128x128f36
d VICTORY_HEAP_LOCKS 1;            d VICTORY_HEAP_SYNC_THREADS 4
d VICTORY_ASYNC_SUBMIT 1;          d VICTORY_ASYNC_PREWARM 1
d VICTORY_ASYNC_HEAP_SYNC 1;       d VICTORY_ASYNC_SIGNALS 1
d VICTORY_ASYNC_SETTLE_ALL 1;      d VICTORY_ALIAS_BUFFERS 1
# Presses Enter on the first screens (20, 30 and 40 s in). Set VICTORY_KEYS= to turn it off.
d VICTORY_KEYS 20:Enter,30:Enter,40:Enter
# Faster, lower resolution: VICTORY_OUTPUT=1920x1080
exec "$(dirname "$0")/victory.sh" run --execute "$exe" "$@"
