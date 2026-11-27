#!/bin/zsh
# Names the victory.exe addresses in a VICTORY_PROFILE report (crates/victory-gpu/src/profile.rs).
#
#   tools/macos/symbolize-profile.sh <report> [<victory.exe>] > <report>.named.txt
#
# The release build keeps line tables (Cargo.toml: debug = "line-tables-only"), so LLVM's
# symbolizer gives each victory.exe+0x... its function, the function it is inlined into and the
# source line; the script appends them to every line that holds such an address. Give it the
# victory.exe of the same build as the run (the run log's commit line names it); the default is
# this checkout's release build. Title and system addresses are left as they are: the title's
# stay RVAs. LLVM_BIN names LLVM's tools (Homebrew's llvm by default); PROFILE_MODULE another
# image of ours by its name in the report (a test executable, say).
set -eu
(( $# >= 1 )) || { print -u2 "usage: tools/macos/symbolize-profile.sh <report> [<victory.exe>]"; exit 2; }
report=$1
exe=${2:-${0:A:h:h:h}/target/x86_64-pc-windows-gnullvm/release/victory.exe}
module=${PROFILE_MODULE:-victory.exe}
llvm=${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}
[[ -x $llvm/llvm-symbolizer && -x $llvm/llvm-readobj ]] || { print -u2 "symbolize-profile.sh: no llvm-symbolizer in $llvm (brew install llvm, or set LLVM_BIN)"; exit 2; }
[[ -f $exe ]] || { print -u2 "symbolize-profile.sh: no $exe"; exit 2; }
[[ -f $report ]] || { print -u2 "symbolize-profile.sh: no $report"; exit 2; }

base=$("$llvm/llvm-readobj" --file-headers "$exe" | awk '/ImageBase:/ {print $2; exit}')
pattern="${module//./\\.}\\+0x[0-9a-fA-F]+"
rvas=(${(u)${(f)"$(grep -aoE "$pattern" "$report" | sed "s/^.*+//")"}})
if (( ${#rvas} == 0 )); then
  cat "$report"
  exit 0
fi
table=$(mktemp -t symbolize-profile)
trap 'rm -f "$table"' EXIT
# One block per address from the symbolizer: function and file:line:column, once per inlined
# frame from the innermost, then an empty line.
for rva in $rvas; do printf '0x%x\n' $(( base + rva )); done |
  "$llvm/llvm-symbolizer" --obj="$exe" --inlining --demangle |
  awk -v list="${(j: :)rvas}" '
    BEGIN { split(list, rva, " "); i = 1; frames = 0 }
    function short(s) { sub(/::h[0-9a-f]+$/, "", s); return s }
    function where(s) {
      if (s ~ /^\?\?/) return ""
      sub(/:[0-9]+$/, "", s)
      gsub(/\\/, "/", s)
      if (match(s, /(crates|apps|library|crt)\/.*/)) s = substr(s, RSTART)
      return " (" s ")"
    }
    /^$/ {
      if (frames && inner !~ /^\?\?/) print rva[i] "\t" inner (frames > 1 ? " in " outer : "")
      i++; frames = 0; next
    }
    { fn = $0; getline loc; frames++; text = short(fn) where(loc); if (frames == 1) inner = text; outer = text }
  ' > "$table"
PATTERN=$pattern awk -F'\t' -v skip=$(( ${#module} + 1 )) '
  BEGIN { pattern = ENVIRON["PATTERN"] }
  NR == FNR { name[$1] = $2; next }
  # Page starts are no code of their own: the 4 KiB table is left as it is.
  /^== / { pages = ($0 ~ /4 KiB pages/) }
  pages { print; next }
  {
    extra = ""; s = $0
    while (match(s, pattern)) {
      a = substr(s, RSTART + skip, RLENGTH - skip)
      if ((a in name) && !(a in shown)) { extra = extra "  [" a " = " name[a] "]"; shown[a] = 1 }
      s = substr(s, RSTART + RLENGTH)
    }
    delete shown
    print $0 extra
  }' "$table" "$report"
