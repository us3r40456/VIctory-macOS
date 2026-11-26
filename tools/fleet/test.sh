#!/bin/bash
# Tests for the fleet's parsing, validation and moderation (no network).
# Run: tools/fleet/test.sh
# shellcheck disable=SC2016,SC2001 # literal backticks, jq and GraphQL in single quotes
set -euo pipefail
# shellcheck source=tools/fleet/lib.sh
. "$(dirname "$0")/lib.sh"

pass=0
fail=0
ok() { # NAME COMMAND... : the command must succeed
    local name=$1
    shift
    if "$@" > /dev/null 2>&1; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        echo "FAIL: $name"
    fi
}
no() { # NAME COMMAND... : the command must fail
    local name=$1
    shift
    if "$@" > /dev/null 2>&1; then
        fail=$((fail + 1))
        echo "FAIL: $name"
    else pass=$((pass + 1)); fi
}
eq() { # NAME EXPECTED ACTUAL
    if [ "$2" = "$3" ]; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        printf 'FAIL: %s\n  expected: %s\n  actual:   %s\n' "$1" "$2" "$3"
    fi
}

SHA=0123456789abcdef0123456789abcdef01234567
REPO=example/VIctory-macOS
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# --- tasks
body=$(task_body "$SHA" m3 "$REPO")
eq "task body round-trips" "$SHA m3" "$(parse_task "$body")"
eq "task body passes moderation" "" "$(moderate "$body")"
eq "task body with CRLF" "$SHA m3" "$(parse_task "$(sed 's/$/\r/' <<< "$body")")"
no "task with unknown chip" parse_task "$(task_body "$SHA" m9 "$REPO")"
no "task with short commit" parse_task "$(task_body 0123abc m3 "$REPO")"
no "task without a block" parse_task "commit: $SHA"
no "task with another kind" parse_task "$(sed 's/^kind: test-matrix/kind: shader/' <<< "$body")"

# --- toolkit filenames
eq "toolkit 4 beta 3 filename" "4.0-beta3" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_3.dmg')"
eq "toolkit 2 beta 3 filename" "2.0-beta3" "$(gptk_version_of 'Game_Porting_Toolkit_2.0_beta_3.dmg')"
eq "toolkit stable filename" "3.0" "$(gptk_version_of 'Game_Porting_Toolkit_3.0.dmg')"
eq "toolkit legacy beta filename" "3.0-beta4" "$(gptk_version_of 'Game_Porting_Toolkit_3.0-beta4.dmg')"
eq "toolkit beta without number" "4.0-beta" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta.dmg')"
eq "toolkit version prefix" "3.0" "$(gptk_version_of 'Game_Porting_Toolkit_v3.0.dmg')"
eq "toolkit path with spaces" "4.0-beta3" "$(gptk_version_of '/tmp/path with spaces/Game_Porting_Toolkit_4.0 beta 3.dmg')"
eq "missing toolkit filename" "unknown" "$(gptk_version_of '')"
eq "unrecognized toolkit filename" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_latest.dmg')"
eq "unrelated disk image" "unknown" "$(gptk_version_of '4.0.dmg')"
eq "unfinished toolkit download" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_3.dmg.part')"
eq "unsupported toolkit beta number" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_10.dmg')"

# --- commands
eq "claim" claim "$(parse_command "/claim")"
eq "claim with spaces" claim "$(parse_command "  /claim  ")"
eq "beat" beat "$(parse_command "/beat")"
eq "release" "release no_time" "$(parse_command "/release no_time")"
eq "release with a bad reason" "" "$(parse_command "/release because")"
eq "claim with extra text" "" "$(parse_command "/claim
please")"
eq "command not on the first line" "" "$(parse_command "hi
/claim")"
eq "plain comment" "" "$(parse_command "Works on my M2 Air")"
eq "result" result "$(parse_command "/result")"

# --- results
good=$(jq -cn --arg c "$SHA" '{schema: 1, kind: "test-matrix", upstream_commit: $c, hw_model: "Mac15,12",
    chip: "Apple M3 Pro", macos: "15.4.1", gptk: "3.0", setup: "ok",
    native: {passed: 300, failed: 0, ignored: 2}, wine: {passed: 340, failed: 3, ignored: 0},
    failures: [{test: "wine/victory_gpu::tests::depth16", sig: "0123456789ab", topic: "gpu"}],
    timings: {setup: 61, native: 42, wine: 130},
    binaries: [{suite: "native", name: "victory_gpu", seconds: 1.25}, {suite: "wine", name: "victory_gpu", seconds: 20.5}]}')
comment=$(printf '/result\n```json\n%s\n```\n' "$good")
eq "result block extracts" "$good" "$(result_json "$comment")"
no "result with text before the block" result_json "$(printf '/result\nhello\n```json\n{}\n```')"
no "result with text after the block" result_json "$(printf '/result\n```json\n{}\n```\nmore')"
no "result without a block" result_json "/result"
ok "valid result" validate_result "$good" "$SHA" m3
no "wrong commit" validate_result "$good" "${SHA%7}8" m3
no "wrong chip family" validate_result "$good" "$SHA" m2
bad() { validate_result "$(jq -c "$1" <<< "$good")" "$SHA" m3; }
no "extra field" bad '.notes = "see https://example.com"'
no "missing field" bad 'del(.gptk)'
no "free text in hw_model" bad '.hw_model = "Mac15,12 see my link"'
no "free text in chip" bad '.chip = "Apple M3 Pro https://x"'
no "free text in macos" bad '.macos = "15.4 beta (lol)"'
no "free text in a test name" bad '.failures[0].test = "wine/victory_gpu::a b"'
no "test name without a suite" bad '.failures[0].test = "victory_gpu::x"'
no "test name without a binary" bad '.failures[0].test = "wine/x"'
no "too many failures" bad '.failures = [range(101) | {test: "wine/b::t\(.)", sig: "0123456789ab", topic: "gpu"}]'
no "signature with text" bad '.failures[0].sig = "see my link"'
no "unknown topic" bad '.failures[0].topic = "rumours"'
no "extra failure field" bad '.failures[0].message = "boom"'
no "timings with text" bad '.timings.native = "slow"'
no "timings with an extra field" bad '.timings.note = 1'
no "seconds with too many decimals" bad '.binaries[0].seconds = 1.234'
ok "seconds as cargo prints them" bad '.binaries = [{suite: "native", name: "census", seconds: 0.28},
    {suite: "native", name: "victory_formats", seconds: 4.89}, {suite: "wine", name: "victory", seconds: 0.07}]'
ok "every time to two decimals" bad '.binaries = [range(80) | {suite: "native", name: "b\(.)", seconds: ("0.\(. + 10)" | tonumber)}]'
# Includes task #98's rejected values and valid hundredths near the bounds.
for seconds in 0.55 0.07 0.28 0 0.01 0.29 1.13 1.14 9.49 14.23 25.15 99999.99 100000; do
    ok "binary seconds $seconds" bad ".binaries[0].seconds = $seconds"
    ok "suite seconds $seconds" bad ".timings.native = $seconds"
done
for seconds in -0.01 100000.01 0.551 0.5500000000000001 99999.999; do
    no "invalid binary seconds $seconds" bad ".binaries[0].seconds = $seconds"
    no "invalid suite seconds $seconds" bad ".timings.native = $seconds"
done
no "numeric text is not binary seconds" bad '.binaries[0].seconds = "0.55"'
no "numeric text is not suite seconds" bad '.timings.native = "0.55"'
no "binary name with text" bad '.binaries[0].name = "see https://x"'
no "unknown suite" bad '.binaries[0].suite = "game"'
ok "null timings" bad '.timings = {setup: 5, native: null, wine: null}'
no "negative count" bad '.native.failed = -1'
no "fractional count" bad '.native.failed = 1.5'
no "extra count field" bad '.native.note = 1'
no "bad setup value" bad '.setup = "maybe"'
no "array instead of object" validate_result '[1]' "$SHA" m3
no "not JSON" validate_result 'nope' "$SHA" m3
no "too large" validate_result "$(printf '%9000s' x)" "$SHA" m3
ok "null suites when setup failed" bad '.setup = "failed" | .native = null | .wine = null | .failures = [] | .binaries = []'
ok "unknown toolkit version" bad '.gptk = "unknown"'
ok "normalized toolkit beta version passes schema" bad '.gptk = "4.0-beta3"'
ok "MacBook model id" bad '.hw_model = "MacBookAir10,1"'
ok "failing result" result_failing "$good"
no "passing result" result_failing "$(jq -c '.wine.failed = 0' <<< "$good")"
ok "setup failure is failing" result_failing "$(jq -c '.setup = "failed" | .native = null | .wine = null' <<< "$good")"
eq "validation error text" "upstream_commit is not the task commit" "$(validate_result "$good" "${SHA%7}8" m3 || true)"

# --- moderation
verdict() { moderate "$1" | cut -d ' ' -f 1; }
eq "clean comment" "" "$(verdict "Setup failed at the Wine build on my M1, log attached locally")"
eq "torrent link" delete "$(verdict "grab it here magnet:?xt=urn:btih:abc")"
eq "file host link" delete "$(verdict "https://MEGA.nz/file/abc")"
eq "game dump talk" hide "$(verdict "where do I get a game dump")"
eq "dumped my game" hide "$(verdict "I dumped my game yesterday")"
eq "decryption" hide "$(verdict "how do I decrypt the package")"
eq "container names" hide "$(verdict "it is an XVD file")"
eq "image attachment" hide "$(verdict "![x](https://github.com/user-attachments/assets/1)")"
eq "long encoded run" hide "$(verdict "$(printf 'A%.0s' $(seq 1 220))")"
eq "words containing short tokens" "" "$(verdict "the dumpster fire of the rpfoo tickets")"

# --- cargo output
cat > "$tmp/native.log" << 'EOF'
     Running unittests src/lib.rs (target/debug/deps/victory_gpu-0f1e2d3c4b5a6978)

running 3 tests
test tests::one ... ok
test tests::two ... FAILED
test tests::three ... ignored

failures:

---- tests::two stdout ----

thread 'tests::two' panicked at /Users/someone/src/VIctory-macOS/crates/victory-gpu/src/x.rs:12:5:
assertion `left == right` failed: texture at 0x7ff3a0 not uploaded
  left: 3
 right: 4
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::two

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.52s

     Running tests/load_probe.rs (target/x86_64-pc-windows-gnullvm/debug/deps/load_probe-99aa.exe)

running 2 tests
test probe_loads ... ok
test probe timeout ... FAILED

failures:

---- probe timeout stdout ----
timed out

failures:
    probe timeout

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.00s

   Doc-tests victory_core

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
EOF
eq "counts are summed" "2 2 1" "$(parse_counts "$tmp/native.log")"
eq "counts json" '{"passed":2,"failed":2,"ignored":1}' "$(counts_json "$tmp/native.log")"
parsed=$(parse_cargo "$tmp/native.log")
eq "binaries and times" "victory_gpu 0.52|load_probe 12.00|doc_victory_core 0.00" "$(grep '^B' <<< "$parsed" | cut -f 2,3 | tr '\t' ' ' | paste -sd '|' -)"
eq "failure text is normalised" 'victory_gpu::tests::two' "$(grep '^F' <<< "$parsed" | head -n 1 | cut -f 2)"
ok "paths, numbers and addresses are removed" grep -q "panicked at crates/victory-gpu/src/x.rs:#:#: assertion \`left == right\` failed: texture at 0x# not uploaded left: # right: #" <<< "$parsed"
no "notes are dropped" grep -q 'RUST_BACKTRACE' <<< "$parsed"
failures=$(failures_json "$parsed" native)
eq "test names with spaces are skipped" 1 "$(jq length <<< "$failures")"
eq "failure entry" "native/victory_gpu::tests::two gpu" "$(jq -r '.[0] | "\(.test) \(.topic)"' <<< "$failures")"
ok "signature is 12 hex" grep -Eq '^[0-9a-f]{12}$' <<< "$(jq -r '.[0].sig' <<< "$failures")"
sed 's/0x7ff3a0/0x1234/; s/left: 3/left: 9/; s|/Users/someone/src|/home/runner/work|' "$tmp/native.log" > "$tmp/other-mac.log"
eq "same failure on another Mac gives the same signature" "$(jq -r '.[0].sig' <<< "$failures")" \
    "$(failures_json "$(parse_cargo "$tmp/other-mac.log")" native | jq -r '.[0].sig')"
cat > "$tmp/english-os.log" << 'EOF'
      Running unittests src/lib.rs (target/debug/deps/victory_loader-0f1e2d3c4b5a6978)
running 1 test
test publish::tests::concurrent_publishers_do_not_share_a_staging_file ... FAILED
failures:
---- publish::tests::concurrent_publishers_do_not_share_a_staging_file stdout ----
called `Result::unwrap()` on an `Err` value: Os { code: 5, kind: PermissionDenied, message: "Access is denied." }
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.22s
EOF
sed 's/Access is denied\./Accès refusé./' "$tmp/english-os.log" > "$tmp/french-os.log"
english_sig=$(failures_json "$(parse_cargo "$tmp/english-os.log")" wine | jq -r '.[0].sig')
french_sig=$(failures_json "$(parse_cargo "$tmp/french-os.log")" wine | jq -r '.[0].sig')
eq "localized OS error text gives the same signature" "$english_sig" "$french_sig"
eq "binaries json" '[{"suite":"wine","name":"victory_gpu","seconds":0.52},{"suite":"wine","name":"load_probe","seconds":12},{"suite":"wine","name":"doc_victory_core","seconds":0}]' "$(binaries_json "$parsed" wine)"
eq "signature lines keep the text locally" "native/victory_gpu::tests::two" "$(signature_lines "$parsed" native | head -n 1 | cut -f 2)"
eq "topic: timeout wins" timeout "$(failure_topic "victory_gpu::x" "timed out after #s")"
eq "topic: crate" shader "$(failure_topic "victory_shader::t" "assertion failed")"
eq "topic: metal" d3dmetal "$(failure_topic "victory_gpu::t" "Metal returned #")"
eq "topic: unknown" other "$(failure_topic "fidelityfx::t" "assertion failed")"
full=$(jq -c --argjson f "$failures" --argjson b "$(binaries_json "$parsed" native)" '.failures = $f | .binaries = $b' <<< "$good")
ok "a parsed run makes a valid result" validate_result "$full" "$SHA" m3

# --- findings
fbody=$(finding_body 0123456789ab native/victory_gpu::tests::two gpu "$REPO")
eq "finding round-trips" "0123456789ab native/victory_gpu::tests::two gpu" "$(parse_finding "$fbody")"
eq "performance finding round-trips" "0123456789ab wine/victory_gpu performance" "$(parse_finding "$(finding_body 0123456789ab wine/victory_gpu performance "$REPO")")"
eq "finding body passes moderation" "" "$(moderate "$fbody")"
no "finding with unknown topic" parse_finding "$(finding_body 0123456789ab native/a::b rumours "$REPO")"
no "a task is not a finding" parse_finding "$body"
ok "bots are trusted authors" is_trusted_author "github-actions[bot]"
no "strangers are not trusted" is_trusted_author "mallory"
slower=$(jq -c '.binaries[1].seconds = 40' <<< "$good")
eq "slowdown found" "wine/victory_gpu	20.5	40" "$(slowdowns "$good" "$slower")"
eq "small slowdowns ignored" "" "$(slowdowns "$good" "$(jq -c '.binaries[0].seconds = 3' <<< "$good")")"

printf 'error: could not compile\n' > "$tmp/broken.log"
eq "no test run gives null" null "$(counts_json "$tmp/broken.log")"
printf 'test result: ok. 3 passed; 0 failed; 1 ignored;\nerror: a later test binary failed to build\n' > "$tmp/partial.log"
eq "successful command keeps its counts" '{"passed":3,"failed":0,"ignored":1}' "$(counts_json "$tmp/partial.log" 0)"
eq "failed command cannot pass on partial counts" null "$(counts_json "$tmp/partial.log" 101)"
eq "aborted command cannot pass on partial counts" null "$(counts_json "$tmp/partial.log" 137)"
eq "counted failures retain their counts" '{"passed":2,"failed":2,"ignored":1}' "$(counts_json "$tmp/native.log" 101)"
eq "failed command with no tests gives null" null "$(counts_json "$tmp/broken.log" 101)"
eq "chip family" m3 "$(chip_family_of "Apple M3 Pro")"
eq "chip family of an Intel Mac" "" "$(chip_family_of "Intel(R) Core(TM) i7")"

# --- scheduling
fb=$(task_body "$SHA" m3 "$REPO" 26 5 2 57)
eq "task with fields round-trips" "$SHA m3" "$(parse_task "$fb")"
eq "task fields" '{"macos":"26","priority":5,"replica":"2","verifies":57}' "$(task_meta "$fb")"
eq "default task fields" '{"macos":"","priority":1,"replica":"","verifies":null}' "$(task_meta "$body")"
eq "task with fields passes moderation" "" "$(moderate "$fb")"
ok "macOS constraint met" validate_result "$good" "$SHA" m3 15
eq "macOS constraint missed" "this task needs macOS 26" "$(validate_result "$good" "$SHA" m3 26 || true)"
eq "result seconds" 233 "$(result_seconds "$good")"
eq "time limit default with few samples" 6 "$(printf '3600\n3600\n' | time_limit_hours 6)"
eq "time limit from the 90th percentile" 6 "$(printf '3600\n3600\n7200\n' | time_limit_hours 6)"
eq "time limit at least 2 hours" 2 "$(printf '60\n60\n60\n' | time_limit_hours 6)"
eq "time limit at most 12 hours" 12 "$(printf '90000\n90000\n90000\n' | time_limit_hours 6)"
clean=$(jq -c '.failures = []' <<< "$good")
eq "flaky: one run saw it, the other did not" 0123456789ab "$(flaky_sigs "$good" "$clean")"
eq "flaky: same failures" "" "$(flaky_sigs "$good" "$good")"
eq "flaky: a suite did not run" "" "$(flaky_sigs "$good" "$(jq -c '.wine = null' <<< "$clean")")"

# --- report
r1=$(jq -c '. + {n: 1, closed: "2026-11-20T10:00:00Z", worker: "alice"}' <<< "$good")
r2=$(jq -c '. + {n: 2, closed: "2026-11-21T10:00:00Z", chip: "Apple M1", hw_model: "MacBookAir10,1", failures: [], wine: {passed: 343, failed: 0, ignored: 0}}' <<< "$good")
r3=$(jq -c '. + {n: 3, closed: "2026-11-22T10:00:00Z", chip: "Apple M2", setup: "failed", native: null, wine: null, failures: [], binaries: []}' <<< "$good")
report=$(printf '%s\n%s\n%s\n' "$r1" "$r2" "$r3" | report_markdown)
ok "report counts results" grep -q '^3 accepted results across 1 commits' <<< "$report"
ok "report coverage" grep -q '3 of 5: m1, m2, m3 | 2 of 3' <<< "$report"
ok "report chip-specific failure" grep -q 'victory_gpu::tests::depth16` (gpu, sig `0123456789ab`): fails on m3; passes on m1' <<< "$report"
ok "report lists every result" grep -q '^| #3 |' <<< "$report"
ok "report workers" grep -q '^| @alice | 1 | m3 | 0.1 | 2026-11-20 |' <<< "$report"
ok "report slowest binaries" grep -q 'm3 (`0123456789ab`): wine/victory_gpu 20.5 s' <<< "$report"
r4=$(jq -c '. + {n: 4, closed: "2026-11-23T10:00:00Z", upstream_commit: "fedcba9876543210fedcba9876543210fedcba98", binaries: [{suite: "wine", name: "victory_gpu", seconds: 40}]}' <<< "$good")
ok "report slowdowns" grep -q 'm3: wine/victory_gpu 20.5 s at `0123456789ab` -> 40 s at `fedcba987654`' <<< "$(printf '%s\n%s\n' "$r1" "$r4" | report_markdown)"
eq "empty report" "None found." "$(: | report_markdown | sed -n '/^## Tests/{n;n;p;}')"

# --- worker EXIT cleanup (stub all network, checkout and build operations)
# A separate shell makes EXIT run after cmd_run's locals have gone out of scope.
cat > "$tmp/worker-cleanup.sh" <<'EOF'
case_dir=$1
setup_status=$2
validation_status=$3
stub_native_status=$4
stub_wine_status=$5
set -- help
. "$0" > /dev/null
CACHE="$case_dir/cache"
WORK="$case_dir/work"
mkdir -p "$WORK/tools/macos"
printf '#!/bin/sh\nexit %s\n' "$setup_status" > "$WORK/setup.sh"
chmod +x "$WORK/setup.sh"
printf '#!/bin/sh\necho "test result: ok. 1 passed; 0 failed; 0 ignored;"\nexit %s\n' "$stub_wine_status" > "$WORK/tools/macos/test-wine.sh"
chmod +x "$WORK/tools/macos/test-wine.sh"
accepted() { return 0; }
load_task() { T_COMMIT=0123456789abcdef0123456789abcdef01234567; T_CHIP=m3; T_MACOS=""; }
require_lease() { return 0; }
git() { if [ "${3:-}" = rev-parse ]; then echo "$T_COMMIT"; fi; return 0; }
gh() { echo 'unexpected GitHub call' >&2; exit 99; }
# Replace the heartbeat subshell itself with sleep, so tests leave no sleep child.
sleep() { exec /bin/sleep 30; }
kill() { printf '%s\n' "$1" > "$case_dir/killed-pid"; builtin kill "$@"; }
cargo() { echo 'test result: ok. 1 passed; 0 failed; 0 ignored;'; return "$stub_native_status"; }
sysctl() { echo Mac15,11; }
sw_vers() { echo 15.4; }
brand() { echo 'Apple M3 Max'; }
gptk_version() { echo 4.0-beta3; }
say() {
    case "$*" in "running setup.sh"*) jobs -p > "$case_dir/heartbeat-pid" ;; esac
}
if [ "$validation_status" != 0 ]; then validate_result() { return 1; }; fi
cmd_run 1
EOF
worker_case() {
    local dir="$tmp/worker-$1" status=0
    mkdir -p "$dir"
    bash -c '. "$1" "$2" "$3" "$4" "$5" "$6"' "$FLEET_DIR/fleet.sh" "$tmp/worker-cleanup.sh" "$dir" "$2" "$3" "${6:-0}" "${7:-0}" > "$dir/output" 2>&1 || status=$?
    eq "worker $1 exit status" "$4" "$status"
    no "worker $1 has no unbound variable" grep -q 'unbound variable' "$dir/output"
    ok "worker $1 ran heartbeat cleanup" test -s "$dir/killed-pid"
    if [ -s "$dir/heartbeat-pid" ]; then
        local pid
        pid=$(cat "$dir/heartbeat-pid")
        no "worker $1 heartbeat stopped" kill -0 "$pid"
        # Always stop this test's job, including when testing the original bug.
        kill "$pid" 2> /dev/null || true
        if [ -s "$dir/killed-pid" ]; then
            eq "worker $1 cleaned up its own heartbeat" "$pid" "$(cat "$dir/killed-pid")"
        fi
    else
        ok "worker $1 started heartbeat" test -s "$dir/heartbeat-pid"
    fi
    if [ -f "$dir/cache/logs/1/result.json" ]; then
        eq "worker $1 setup result" "$5" "$(jq -r .setup "$dir/cache/logs/1/result.json")"
    fi
}
worker_case success 0 0 0 ok
worker_case setup-failure 1 0 0 failed
worker_case invalid-result 0 1 1 unused

# A test command can print passing binaries and still fail or crash later.
suite_exit_case() {
    worker_case "$1" 0 0 0 ok "$2" "$3"
    local result
    result=$(cat "$tmp/worker-$1/cache/logs/1/result.json")
    eq "$1 native counts" "$4" "$(jq -c .native <<< "$result")"
    eq "$1 Wine counts" "$5" "$(jq -c .wine <<< "$result")"
    ok "$1 is a failing result" result_failing "$result"
    ok "$1 still passes schema" validate_result "$result" "$SHA" m3
}
passed_counts='{"passed":1,"failed":0,"ignored":0}'
suite_exit_case native-error 101 0 null "$passed_counts"
suite_exit_case wine-error 0 101 "$passed_counts" null
suite_exit_case both-error 101 101 null null
suite_exit_case wine-aborted 0 137 "$passed_counts" null

# --- claim polling (GitHub updates the assignee and labels separately)
cat > "$tmp/worker-claim.sh" <<'EOF'
case_dir=$1
scenario=$2
set -- help
. "$0" > /dev/null
REPO=example/VIctory-macOS
accepted() { return 0; }
me() { echo worker; }
sleep() { :; }
gh() { printf '%s\n' "$*" >> "$case_dir/posts"; }
load_count=0
load_task() {
    load_count=$((load_count + 1))
    printf '%s\n' "$load_count" > "$case_dir/loads"
    T_STATE=OPEN
    T_ASSIGNEE=""
    T_LABELS="fleet:open"
    [ "$load_count" -gt 1 ] || return 0
    case "$scenario" in
        delayed-label)
            T_ASSIGNEE=worker
            if [ "$load_count" = 3 ]; then T_LABELS=""; fi
            if [ "$load_count" -ge 4 ]; then T_LABELS="fleet:leased"; fi
            ;;
        delayed-assignee)
            if [ "$load_count" -ge 3 ]; then T_ASSIGNEE=worker; T_LABELS="fleet:leased"; fi
            ;;
        immediate) T_ASSIGNEE=worker; T_LABELS="fleet:leased" ;;
        other-worker) T_ASSIGNEE=another-worker; T_LABELS="fleet:leased" ;;
        missing-label) T_ASSIGNEE=worker ;;
    esac
}
# The claim reads the issue (load_issue) whatever its kind, then the
# task's fields once: the scenario counts issue reads only.
eval "fake_load() $(declare -f load_task | tail -n +2)"
load_task() { :; }
load_issue() { fake_load "$@"; T_KIND=task; }
cmd_claim 1
EOF
claim_case() {
    local dir="$tmp/claim-$1" status=0
    mkdir -p "$dir"
    bash -c '. "$1" "$2" "$3"' "$FLEET_DIR/fleet.sh" "$tmp/worker-claim.sh" "$dir" "$1" > "$dir/output" 2>&1 || status=$?
    eq "claim $1 exit status" "$2" "$status"
    ok "claim $1 outcome" grep -q "$3" "$dir/output"
    eq "claim $1 polling count" "$4" "$(cat "$dir/loads")"
    eq "claim $1 posts once" 'issue comment 1 -R example/VIctory-macOS --body /claim' "$(cat "$dir/posts")"
}
claim_case delayed-label 0 '#1 is yours' 4
claim_case delayed-assignee 0 '#1 is yours' 3
claim_case immediate 0 '#1 is yours' 2
claim_case other-worker 1 'went to someone else' 2
claim_case missing-label 1 'no lease after 3 minutes' 37

echo "fleet tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
