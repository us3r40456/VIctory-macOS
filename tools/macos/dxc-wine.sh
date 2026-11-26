#!/bin/sh
# Cargo invokes this wrapper from the graphics crate directory.
set -eu
: "${DXC_WINE:?Set DXC_WINE to the Wine executable}"
: "${DXC_EXE:?Set DXC_EXE to the Microsoft x64 dxc.exe}"
if [ "$#" -ne 7 ] || [ "$1" != "-T" ] || [ "$3" != "-E" ] || [ "$5" != "-Fo" ]; then
    echo "Expected -T PROFILE -E ENTRY -Fo OUTPUT INPUT" >&2
    exit 2
fi
case "$6" in /*) output="Z:$6" ;; *) output=$6 ;; esac
case "$7" in /*) source="Z:$7" ;; *) source=$7 ;; esac
exec "$DXC_WINE" "$DXC_EXE" -T "$2" -E "$4" -Fo "$output" "$source"
