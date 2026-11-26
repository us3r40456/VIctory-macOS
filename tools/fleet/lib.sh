#!/bin/bash
# Shared, side-effect-free functions for the Mac test fleet (see docs/fleet.md).
# Sourced by fleet.sh (the worker), queue.sh (the GitHub Actions side),
# maintainer.sh and test.sh. Works with macOS's bash 3.2 and GNU bash.

# shellcheck disable=SC2016 # jq programs in single quotes
FLEET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_CONF="$FLEET_DIR/fleet.json"

conf() { jq -r "$1" "$FLEET_CONF"; }
label() { jq -r --arg k "$1" '.labels[$k]' "$FLEET_CONF"; }
chip_label() { printf 'fleet:%s\n' "$1"; }

is_maintainer() {
    jq -e --arg a "$1" '.maintainers | index($a) != null' "$FLEET_CONF" > /dev/null
}

# Tasks come from maintainers or from this repository's own workflows
# (automatic rounds); findings only from the workflows.
FLEET_BOT_LOGINS="github-actions[bot] app/github-actions github-actions"
is_bot_login() { case " $FLEET_BOT_LOGINS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
is_trusted_author() { is_maintainer "$1" || is_bot_login "$1"; }

is_chip() {
    [ -n "$1" ] && jq -e --arg c "$1" '.chips | index($c) != null' "$FLEET_CONF" > /dev/null
}

is_release_reason() {
    [ -n "$1" ] && jq -e --arg r "$1" '.release_reasons | index($r) != null' "$FLEET_CONF" > /dev/null
}

valid_login() { [[ $1 =~ ^[A-Za-z0-9]([A-Za-z0-9-]{0,38})$ ]]; }

# "Apple M3 Pro" -> m3. Prints nothing for anything else.
chip_family_of() {
    sed -nE 's/^Apple (M[0-9]{1,2})( Pro| Max| Ultra)?$/\1/p' <<< "$1" | tr 'M' 'm'
}

# Normalize Apple's DMG filename to the version format accepted by results.
# For example, Game_Porting_Toolkit_4.0_beta_3.dmg becomes 4.0-beta3.
gptk_version_of() { # DMG PATH -> VERSION or unknown
    local v
    v=$(sed -nE 's/^Game_Porting_Toolkit[_ -]?v?(.+)\.dmg$/\1/p' <<< "${1##*/}" |
        sed -E 's/[_ -]beta[_ -]?/-beta/' |
        sed -nE '/^[0-9]{1,2}(\.[0-9]{1,2}){0,2}(-beta[0-9]?)?$/p')
    printf '%s\n' "${v:-unknown}"
}

# The body of a task issue. The HTML comment is what machines read. The
# optional fields: MACOS (a macOS major version the Mac must run), PRIORITY
# (0-9, higher first), REPLICA (which independent run of this commit and
# chip this is) and VERIFIES (a finding whose fix this run checks).
task_body() { # COMMIT CHIP REPO [MACOS] [PRIORITY] [REPLICA] [VERIFIES]
    local chip_upper why="" fields=""
    chip_upper=$(tr 'm' 'M' <<< "$2")
    [ -n "${4:-}" ] && why="$why This run needs macOS $4, to fill a gap in coverage." && fields="${fields}macos: $4"$'\n'
    [ -n "${5:-}" ] && fields="${fields}priority: $5"$'\n'
    [ -n "${6:-}" ] && why="$why It is run $6 of several independent runs, which lets the fleet spot flaky tests: take it only if you haven't already run this commit on this chip." && fields="${fields}replica: $6"$'\n'
    [ -n "${7:-}" ] && why="$why It checks whether the fix for #$7 holds." && fields="${fields}verifies: $7"$'\n'
    cat << EOF
<!-- fleet-task
kind: test-matrix
commit: $1
chip: $2
${fields}-->
**Test matrix: commit \`${1:0:12}\` on an Apple $chip_upper Mac** (any $chip_upper, $chip_upper Pro, $chip_upper Max or $chip_upper Ultra).$why

Run the native and the Wine test suites at this commit and report the counts. Only a short, strictly checked result is posted. Logs stay on your Mac.

To take this task, point your AI coding agent at [docs/fleet-join.md](https://github.com/$3/blob/main/docs/fleet-join.md), or read [docs/fleet.md](https://github.com/$3/blob/main/docs/fleet.md). Follow [docs/fleet-legal.md](https://github.com/$3/blob/main/docs/fleet-legal.md): nothing from any game is ever posted here.
EOF
}


# The optional fields of a task, as JSON: {macos, priority, replica, verifies}.
task_meta() { # BODY
    local block
    block=$(tr -d '\r' <<< "$1" | awk '/^<!-- fleet-task[ \t]*$/ { f = 1; next } f && /^-->/ { exit } f')
    jq -cn --arg m "$(sed -nE 's/^macos: *([0-9]{2}) *$/\1/p' <<< "$block" | head -n 1)" \
        --arg p "$(sed -nE 's/^priority: *([0-9]) *$/\1/p' <<< "$block" | head -n 1)" \
        --arg r "$(sed -nE 's/^replica: *([1-9]) *$/\1/p' <<< "$block" | head -n 1)" \
        --arg v "$(sed -nE 's/^verifies: *([0-9]{1,7}) *$/\1/p' <<< "$block" | head -n 1)" \
        '{macos: $m, priority: (if $p == "" then 1 else ($p | tonumber) end), replica: $r,
          verifies: (if $v == "" then null else ($v | tonumber) end)}'
}

# Reads a task issue body. Prints "COMMIT CHIP" or fails.
parse_task() {
    local block kind commit chip
    block=$(tr -d '\r' <<< "$1" | awk '/^<!-- fleet-task[ \t]*$/ { f = 1; next } f && /^-->/ { exit } f')
    kind=$(sed -nE 's/^kind: *([a-z-]+) *$/\1/p' <<< "$block" | head -n 1)
    commit=$(sed -nE 's/^commit: *([0-9a-f]{40}) *$/\1/p' <<< "$block" | head -n 1)
    chip=$(sed -nE 's/^chip: *(m[0-9]{1,2}) *$/\1/p' <<< "$block" | head -n 1)
    [ "$kind" = test-matrix ] && [ -n "$commit" ] && is_chip "$chip" || return 1
    printf '%s %s\n' "$commit" "$chip"
}

# Reads a comment. Prints "claim", "beat", "release REASON" or "result",
# or nothing when the comment is not a fleet command. Commands other than
# /result must be the whole comment.
parse_command() {
    local text first rest
    text=$(tr -d '\r' <<< "$1")
    first=$(sed -n '1p' <<< "$text" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
    rest=$(sed '1d' <<< "$text" | tr -d '[:space:]')
    case "$first" in
        /claim | /beat)
            [ -z "$rest" ] && echo "${first#/}"
            ;;
        "/release "*)
            local reason=${first#/release }
            [ -z "$rest" ] && is_release_reason "$reason" && echo "release $reason"
            ;;
        /result)
            echo result
            ;;
    esac
    return 0
}

# Prints the JSON of a /result comment. Fails unless the comment is exactly
# "/result" followed by one ```json block and nothing else.
result_json() {
    tr -d '\r' <<< "$1" | awk '
        NR == 1 { if ($0 !~ /^\/result[ \t]*$/) bad = 1; next }
        state == 0 && /^```json[ \t]*$/ { state = 1; next }
        state == 1 && /^```[ \t]*$/ { state = 2; next }
        state == 1 { print; next }
        /^[ \t]*$/ { next }
        { bad = 1 }
        END { if (bad || state != 2) exit 1 }'
}

# The result schema. Every field is checked against a narrow pattern, so a
# valid result cannot carry free text, links or file contents.
FLEET_TOPICS="shader gpu d3dmetal hle loader formats pipeline game-support core wine memory timeout numerics build performance other"
FLEET_VALIDATE_JQ='
def fields: ["binaries","chip","failures","gptk","hw_model","kind","macos","native","schema","setup","timings","upstream_commit","wine"];
def str($re): type == "string" and test($re);
def count: type == "number" and . == floor and . >= 0 and . <= 100000;
# At most two decimals. Not "(. * 100 | floor) == . * 100": in binary
# floating point 0.28 * 100 is 28.000000000000004, which that rejected.
# Rounding back gives the same double as the literal for two decimals.
def secs: type == "number" and . >= 0 and . <= 100000 and (. * 100 | round) / 100 == .;
def counts: . == null or (type == "object" and (keys == ["failed","ignored","passed"])
    and (.passed | count) and (.failed | count) and (.ignored | count));
def topics: $topics | split(" ");
if type != "object" then "result must be a JSON object"
else
  (if (keys - fields) != [] then "result has fields that are not in the schema" else empty end),
  (if (fields - keys) != [] then "result is missing fields: " + ((fields - keys) | join(", ")) else empty end),
  (if .schema != 1 then "schema must be 1" else empty end),
  (if .kind != "test-matrix" then "kind must be test-matrix" else empty end),
  (if .upstream_commit != $commit then "upstream_commit is not the task commit" else empty end),
  (if (.hw_model | str("^[A-Za-z]{2,20}[0-9]{1,3},[0-9]{1,3}$")) then empty
   else "hw_model is not a Mac model identifier" end),
  (if (.chip | str("^Apple M[0-9]{1,2}( Pro| Max| Ultra)?$")) | not then "chip is not an Apple M-series name"
   elif (.chip | ascii_downcase | capture("(?<f>m[0-9]+)").f) != $chip then "chip is not the task chip family"
   else empty end),
  (if (.macos | str("^[0-9]{2}(\\.[0-9]{1,2}){0,2}$")) | not then "macos is not a version number"
   elif $want_macos != "" and (.macos | split(".")[0]) != $want_macos then "this task needs macOS \($want_macos)"
   else empty end),
  (if (.gptk | str("^([0-9]{1,2}(\\.[0-9]{1,2}){0,2}(-beta[0-9]?)?|unknown)$")) then empty
   else "gptk is not a version number or unknown" end),
  (if .setup == "ok" or .setup == "failed" then empty else "setup must be ok or failed" end),
  (if (.native | counts) then empty else "native must be null or {passed, failed, ignored}" end),
  (if (.wine | counts) then empty else "wine must be null or {passed, failed, ignored}" end),
  (if (.failures | type) == "array" and (.failures | length) <= 100
      and all(.failures[]; type == "object" and keys == ["sig","test","topic"]
          and (.test | str("^(native|wine)/[a-z0-9_]{1,64}::[A-Za-z0-9_:]{1,200}$"))
          and (.sig | str("^[0-9a-f]{12}$"))
          and (.topic as $t | topics | index($t) != null)) then empty
   else "failures must be up to 100 {test, sig, topic} entries" end),
  (if (.timings | type) == "object" and (.timings | keys) == ["native","setup","wine"]
      and all(.timings[]; . == null or secs) then empty
   else "timings must be {setup, native, wine} in seconds or null" end),
  (if (.binaries | type) == "array" and (.binaries | length) <= 80
      and all(.binaries[]; type == "object" and keys == ["name","seconds","suite"]
          and (.suite == "native" or .suite == "wine") and (.name | str("^[a-z0-9_]{1,64}$")) and (.seconds | secs))
   then empty else "binaries must be up to 80 {suite, name, seconds} entries" end)
end'

# Prints one error per line and fails if the result is not valid for the task.
validate_result() { # JSON COMMIT CHIP [MACOS_MAJOR]
    local errors
    if [ "${#1}" -gt 8000 ]; then
        echo "result is larger than 8000 bytes"
        return 1
    fi
    if ! jq -e . > /dev/null 2>&1 <<< "$1"; then
        echo "result is not valid JSON"
        return 1
    fi
    errors=$(jq -r --arg commit "$2" --arg chip "$3" --arg want_macos "${4:-}" --arg topics "$FLEET_TOPICS" "$FLEET_VALIDATE_JQ" <<< "$1")
    if [ -n "$errors" ]; then
        printf '%s\n' "$errors"
        return 1
    fi
}

# A result counts as failing unless setup worked and both suites ran clean.
result_failing() {
    jq -e '.setup != "ok" or .native == null or .wine == null
        or .native.failed > 0 or .wine.failed > 0' > /dev/null <<< "$1"
}

# Screens free text in fleet issues. Prints "delete REASON", "hide REASON"
# or nothing. Deleting is for links that distribute files; hiding sends
# anything that might be game material to a maintainer for review.
moderate() {
    local lower
    lower=$(tr '[:upper:]' '[:lower:]' <<< "$1")
    if grep -Eq 'magnet:\?|\.torrent([^a-z]|$)|(mega\.(nz|io)|mediafire\.com|gofile\.io|pixeldrain\.com|1fichier\.com|rapidgator\.net|anonfiles|krakenfiles\.com|workupload\.com|uploadhaven\.com|send\.cm|terabox\.com|qiwi\.gg|buzzheavier\.com)' <<< "$lower"; then
        echo "delete file-sharing or torrent link"
    elif grep -Eq '[A-Za-z0-9+/=_-]{200,}' <<< "$1"; then
        echo "hide long encoded data"
    elif grep -Eq 'decrypt|(^|[^a-z])(dump(ed|ing|s)?|rip(ped)?) (of )?(the |my |a )?(game|disc|console|title)|(game|disc|title) (dump|rip)|(^|[^a-z])(xvd|xvc|msixvc|cik|odk|rpf)([^a-z]|$)|title ?key|(^|[^a-z])keys? (file|dump)|pirat|warez|cracked|user-attachments|user-images\.githubusercontent|private-user-images|screenshot' <<< "$lower"; then
        echo "hide possible game material or circumvention"
    fi
    return 0
}

# Sums the "test result:" lines of cargo test output. Prints
# "PASSED FAILED IGNORED", or nothing if no test binary ran.
parse_counts() { # LOG
    awk '/^test result: / {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^passed;?$/) p += $(i - 1)
                if ($i ~ /^failed;?$/) f += $(i - 1)
                if ($i ~ /^ignored;?$/) g += $(i - 1)
            }
            n++
        }
        END { if (n) print p + 0, f + 0, g + 0 }' "$1"
}

counts_json() { # LOG [EXIT STATUS] -> {passed, failed, ignored} or null
    local c status=${2:-0}
    c=$(parse_counts "$1")
    if [ -z "$c" ]; then
        echo null
        return
    fi
    # shellcheck disable=SC2086 # three numbers
    set -- $c
    # Passing binaries do not prove the suite finished: a later build or runner
    # can fail. Keep counted test failures, but mark other unsuccessful runs null.
    if [ "$status" -ne 0 ] && [ "$2" -eq 0 ]; then
        echo null
        return
    fi
    jq -cn --argjson p "$1" --argjson f "$2" --argjson i "$3" '{passed: $p, failed: $f, ignored: $i}'
}

# Reads cargo test output. Prints one line per test binary,
#   B<TAB>binary<TAB>seconds
# and one per failed test, with its failure text normalised so the same
# failure gives the same text on every Mac:
#   F<TAB>binary::test<TAB>normalised text
# Binary names drop cargo's hash and .exe; doc tests become doc_<crate>.
parse_cargo() { # LOG
    tr -d '\r' < "$1" | awk '
        function norm(t,   d, n, i) {
            n = split("crates apps xtask tests", d, " ")
            for (i = 1; i <= n; i++) gsub("/[^ :]*/" d[i] "/", d[i] "/", t)
            gsub(/0x[0-9a-fA-F]+/, "@HEX@", t)
            gsub(/[0-9]+(\.[0-9]+)?/, "#", t)
            gsub(/@HEX@/, "0x#", t)
            # Windows OS error messages are localized; code and kind carry
            # the stable meaning (for example PermissionDenied on every Mac).
            gsub(/, message: "[^"]*"/, "", t)
            gsub(/[ \t]+/, " ", t)
            sub(/^ /, "", t)
            return substr(t, 1, 400)
        }
        function flush() {
            if (fname != "") { printf "F\t%s::%s\t%s\n", bin, fname, norm(msg); fname = "" }
        }
        /^ *Running .*\(/ {
            flush()
            b = $0; sub(/.*\(/, "", b); sub(/\).*/, "", b); n = split(b, parts, "/"); b = parts[n]
            sub(/\.exe$/, "", b); sub(/-[0-9a-f]+$/, "", b); gsub(/-/, "_", b); bin = tolower(b); next
        }
        /^ *Doc-tests / { flush(); bin = "doc_" tolower($2); gsub(/-/, "_", bin); next }
        /^---- .* stdout ----$/ { flush(); fname = $0; sub(/^---- /, "", fname); sub(/ stdout ----$/, "", fname); msg = ""; next }
        /^failures:$/ || /^test result: / {
            flush()
            if ($0 ~ /^test result: / && bin != "") {
                s = $0; sub(/.*finished in /, "", s); sub(/s$/, "", s)
                printf "B\t%s\t%.2f\n", bin, s + 0
            }
            next
        }
        fname != "" && !/^note: run with/ && !/^stack backtrace:/ && NF { msg = msg " " $0 }
        END { flush() }'
}

failure_topic() { # BINARY::TEST NORMALISED_TEXT
    local lower
    lower=$(tr '[:upper:]' '[:lower:]' <<< "$1 $2")
    case "$lower" in
        *"timed out"* | *timeout*) echo timeout ;;
        *"out of memory"* | *"memory allocation"*) echo memory ;;
        *nan* | *"not finite"*) echo numerics ;;
        *d3dmetal* | *metal*) echo d3dmetal ;;
        *wine*) echo wine ;;
        victory_shader* | *shader*) echo shader ;;
        victory_gpu* | *d3d12*) echo gpu ;;
        victory_hle*) echo hle ;;
        victory_loader* | victory_pe* | load_probe*) echo loader ;;
        victory_formats*) echo formats ;;
        victory_pipeline*) echo pipeline ;;
        victory_game*) echo game-support ;;
        victory_core*) echo core ;;
        xtask*) echo build ;;
        *) echo other ;;
    esac
}

signature() { printf '%s' "$1" | shasum -a 256 2> /dev/null | cut -c 1-12 || printf '%s' "$1" | sha256sum | cut -c 1-12; }

# PARSED SUITE -> JSON list of {test, sig, topic}. The signature hashes the
# test and its normalised failure text, so only the hash is shared.
failures_json() {
    local name text test
    { grep '^F' <<< "$1" || true; } | head -n 100 | while IFS="$(printf '\t')" read -r _ name text; do
        test="$2/$name"
        [[ $test =~ ^(native|wine)/[a-z0-9_]{1,64}::[A-Za-z0-9_:]{1,200}$ ]] || continue
        jq -cn --arg t "$test" --arg s "$(signature "$test"$'\n'"$text")" --arg k "$(failure_topic "$name" "$text")" \
            '{test: $t, sig: $s, topic: $k}'
    done | jq -sc .
}

binaries_json() { # PARSED SUITE -> JSON list of {suite, name, seconds}
    { grep '^B' <<< "$1" || true; } | head -n 40 |
        jq -Rsc --arg s "$2" 'split("\n") | map(select(length > 0) | split("\t")
            | {suite: $s, name: .[1], seconds: (.[2] | tonumber + 0)}) | map(select(.name | test("^[a-z0-9_]{1,64}$")))'
}

# Prints the local, unshared text behind each signature: sig, test, text.
signature_lines() { # PARSED SUITE
    { grep '^F' <<< "$1" || true; } | while IFS="$(printf '\t')" read -r _ name text; do
        printf '%s\t%s/%s\t%s\n' "$(signature "$2/$name"$'\n'"$text")" "$2" "$name" "$text"
    done
}

# --- findings ---------------------------------------------------------------

finding_body() { # SIG TEST TOPIC REPO
    cat << EOF
<!-- fleet-finding
sig: $1
test: $2
topic: $3
-->
**$3 finding:** \`$2\`

Opened automatically by the Mac test fleet the first time a worker reported this failure signature. Each later sighting is added as a comment. The signature is a hash of the test and its normalised failure text; the text itself stays on the worker's Mac.

**Working on it?** Comment \`/claim\` first so that nobody else repeats the work (\`/release other\` to hand it back). Reproduce it locally with \`tools/fleet/fleet.sh repro <this issue number>\`, and reference this issue from your fix's pull request. See [docs/fleet.md](https://github.com/$4/blob/main/docs/fleet.md#findings).
EOF
}

parse_finding() { # BODY -> "SIG TEST TOPIC"
    local block sig test topic
    block=$(tr -d '\r' <<< "$1" | awk '/^<!-- fleet-finding[ \t]*$/ { f = 1; next } f && /^-->/ { exit } f')
    sig=$(sed -nE 's/^sig: *([0-9a-f]{12}) *$/\1/p' <<< "$block" | head -n 1)
    test=$(sed -nE 's/^test: *((native|wine)\/[a-z0-9_]{1,64}(::[A-Za-z0-9_:]{1,200})?) *$/\1/p' <<< "$block" | head -n 1)
    topic=$(sed -nE 's/^topic: *([a-z-]+) *$/\1/p' <<< "$block" | head -n 1)
    [ -n "$sig" ] && [ -n "$test" ] && [[ " $FLEET_TOPICS " == *" $topic "* ]] || return 1
    printf '%s %s %s\n' "$sig" "$test" "$topic"
}

# Seconds a result took end to end (setup and both suites).
result_seconds() { jq -r '[.timings[] // 0] | add | floor' <<< "$1"; }

# Time limit for a task lease, in hours: three times the 90th percentile of
# recent end-to-end durations, clamped to 2-12 hours, or DEFAULT with fewer
# than three samples. Reads one duration in seconds per line.
time_limit_hours() { # DEFAULT
    jq -rs --argjson d "$1" '
        sort as $s | ($s | length) as $n
        | if $n < 3 then $d
          else ($s[[($n * 0.9 | floor), $n - 1] | min] * 3 / 3600 | ceil)
               | if . < 2 then 2 elif . > 12 then 12 else . end end'
}

# Signatures that one of two results of the same commit and chip has and the
# other lacks, when both ran both suites: the likely flaky tests.
flaky_sigs() { # RESULT_A RESULT_B
    jq -rn --argjson a "$1" --argjson b "$2" '
        if [$a, $b] | any(.native == null or .wine == null) then empty
        else (([$a.failures[].sig] - [$b.failures[].sig]) + ([$b.failures[].sig] - [$a.failures[].sig])) | unique[] end'
}

# PREVIOUS CURRENT (results) -> lines "suite/binary<TAB>old<TAB>new" for
# test binaries that got at least 50% and 5 seconds slower.
slowdowns() {
    jq -rn --argjson a "$1" --argjson b "$2" '
        ($a.binaries | map({key: "\(.suite)/\(.name)", value: .seconds}) | from_entries) as $old
        | $b.binaries[] | "\(.suite)/\(.name)" as $k | select($old[$k] != null)
        | select(.seconds >= $old[$k] * 1.5 and .seconds - $old[$k] >= 5)
        | "\($k)\t\($old[$k])\t\(.seconds)"'
}
# Turns accepted results (one JSON object per line, each with "n" for the
# issue number and "closed" for when it closed) into a Markdown report:
# coverage per commit, chip-specific failures, the slowest test binaries,
# slowdowns between commits on the same chip family, and every result.
report_markdown() {
    jq -rs --argjson chips "$(jq -c .chips "$FLEET_CONF")" '
        def fam: .chip | ascii_downcase | capture("(?<f>m[0-9]+)").f;
        def pf($c): if $c == null then "-" else "\($c.passed) / \($c.failed)" end;
        def failing: .setup != "ok" or .native == null or .wine == null or .native.failed > 0 or .wine.failed > 0;
        def none($x): if ($x | length) == 0 then "None found." else ($x | join("\n")) end;
        sort_by(.closed) as $rs
        | ($rs | group_by(.upstream_commit) | sort_by(map(.closed) | min)) as $commits
        | "# Mac test fleet report\n\n\($rs | length) accepted results across \($commits | length) commits.\n\n"
        + "## Coverage by commit\n\n| Commit | First result | Chip families | Results failing |\n|---|---|---|---|\n"
        + ($commits | map(
            (map(fam) | unique) as $f
            | "| `\(.[0].upstream_commit[0:12])` | \(map(.closed) | min | .[0:10]) | \($f | length) of \($chips | length): \($f | join(", ")) | \(map(select(failing)) | length) of \(length) |")
          | join("\n"))
        + "\n\n## Tests that fail on some chips but not others\n\n"
        + none([$commits[] | . as $g
            | ($g | map(select(.native != null and .wine != null)) | map(fam) | unique) as $ran
            | ($g | map(fam as $c | .failures[] | {t: .test, s: .sig, k: .topic, c: $c}) | group_by(.t)[]
               | {t: .[0].t, s: .[0].s, k: .[0].k, bad: (map(.c) | unique)})
            | select((.bad | length) < ($ran | length))
            | "- `\($g[0].upstream_commit[0:12])` `\(.t)` (\(.k), sig `\(.s)`): fails on \(.bad | join(", ")); passes on \($ran - .bad | join(", "))"])
        + "\n\n## Slowest test binaries (latest result per chip family)\n\n"
        + none([$rs | group_by(fam)[] | last | . as $r
            | (.binaries | sort_by(-.seconds) | .[0:3][])
            | "- \($r | fam) (`\($r.upstream_commit[0:12])`): \(.suite)/\(.name) \(.seconds) s"])
        + "\n\n## Slowdowns between commits (same chip family, at least 50% and 5 s slower)\n\n"
        + none([$rs | group_by(fam)[] | . as $g | range(1; length) as $i
            | ($g[$i - 1]) as $a | ($g[$i]) as $b | select($a.upstream_commit != $b.upstream_commit)
            | ($a.binaries | map({key: "\(.suite)/\(.name)", value: .seconds}) | from_entries) as $old
            | $b.binaries[] | "\(.suite)/\(.name)" as $k | select($old[$k] != null)
            | select(.seconds >= $old[$k] * 1.5 and .seconds - $old[$k] >= 5)
            | "- \($b | fam): \($k) \($old[$k]) s at `\($a.upstream_commit[0:12])` -> \(.seconds) s at `\($b.upstream_commit[0:12])`"])
        + "\n\n## Workers\n\n| Worker | Results | Chips | Compute hours | Last result |\n|---|---|---|---|---|\n"
        + ($rs | map(select(.worker != null)) | group_by(.worker) | sort_by(-length)
           | map("| @\(.[0].worker) | \(length) | \(map(fam) | unique | join(", ")) | \(map([.timings[] // 0] | add) | add / 3600 * 10 | round / 10) | \(map(.closed) | max | .[0:10]) |")
           | if length == 0 then "None recorded." else join("\n") end)
        + "\n\n## All results\n\n| Task | Closed | Commit | Chip | Model | macOS | Toolkit | Native passed / failed | Wine passed / failed | Wall time native / wine |\n|---|---|---|---|---|---|---|---|---|---|\n"
        + ($rs | map("| #\(.n) | \(.closed[0:10]) | `\(.upstream_commit[0:12])` | \(.chip) | \(.hw_model) | \(.macos) | \(.gptk) | \(pf(.native)) | \(pf(.wine)) | \(.timings.native // "-") s / \(.timings.wine // "-") s |") | join("\n"))'
}
