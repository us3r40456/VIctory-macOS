#!/bin/zsh
# One-step setup of VIctory for macOS on an Apple Silicon Mac.
#
#   ./setup.sh
#
# It installs the tools it needs with Homebrew, builds an x86_64 Wine from
# CodeWeavers' CrossOver source, adds Apple's D3DMetal from the Game Porting
# Toolkit in your Downloads folder, and builds VIctory. Run it again at any
# time: finished steps are skipped. The first run takes about an hour.
set -eu
ROOT="${0:A:h}"
cd "$ROOT"
say() { print -P "%F{cyan}==>%f $*"; }
die() { print -P "%F{red}error:%f $*" >&2; exit 1; }

# 1. The Mac itself.
[[ $(uname -m) == arm64 ]] || die "this needs an Apple Silicon Mac (M1 or newer)"
(( ${$(sw_vers -productVersion)%%.*} >= 15 )) || die "this needs macOS 15 Sequoia or newer"
arch -x86_64 /usr/bin/true 2> /dev/null || die "Rosetta 2 is missing. Install it with: softwareupdate --install-rosetta"
xcode-select -p > /dev/null 2>&1 || die "the Xcode command line tools are missing. Install them with: xcode-select --install"
command -v brew > /dev/null || die "Homebrew is missing. Install it from https://brew.sh and run ./setup.sh again"

# 2. Tools from Homebrew.
BREW="$(brew --prefix)"
wanted=(rustup llvm lld mingw-w64 bison flex pkgconf)
missing=()
for f in $wanted; do brew list --formula "$f" > /dev/null 2>&1 || missing+=("$f"); done
if (( ${#missing} )); then
  say "installing ${missing[*]} with Homebrew"
  # brew exits with an error when a post-install step fails, even if the
  # formula itself installed (rustup's can fail on a completion link it may
  # not write). Ask Homebrew what is installed instead of trusting the exit.
  brew install "${missing[@]}" || true
  failed=()
  for f in $missing; do brew list --formula "$f" > /dev/null 2>&1 || failed+=("$f"); done
  (( ${#failed} == 0 )) || die "Homebrew could not install ${failed[*]}. See its output above, fix that and run ./setup.sh again"
fi
export PATH="$BREW/opt/rustup/bin:$PATH"

# 3. Rust with the Windows target VIctory is built for on a Mac.
say "setting up Rust"
rustup show active-toolchain > /dev/null 2>&1 || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)" --profile minimal
rustup target add x86_64-pc-windows-gnullvm > /dev/null

# Wine's build breaks on paths with spaces, so the large build folders live
# under a space-free link in the cache folder.
ALIAS="$HOME/Library/Caches/victory-macos"
mkdir -p "$ALIAS" "$ROOT/target"
ln -sfn "$ROOT/target" "$ALIAS/target"
WINE_DIR="$ALIAS/target/wine-cx"

# 4. x86_64 libraries for Wine.
say "building the x86_64 libraries Wine needs (about 10 minutes the first time)"
tools/macos/build-deps.sh "$ALIAS/target/deps-x86"

# 5. CrossOver Wine, with the patches in tools/macos/patches. A Wine built
# before a patch was added or changed gets only its server rebuilt.
if [[ -x $WINE_DIR/bin/wine && "$(cat "$WINE_DIR/.victory-patches" 2> /dev/null)" == "$(tools/macos/build-crossover-wine.sh --stamp)" ]]; then
  say "Wine already built"
elif [[ -x $WINE_DIR/bin/wine ]]; then
  say "rebuilding Wine's server with the patches in tools/macos/patches (a few minutes)"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh
else
  say "building CrossOver Wine (about 30 minutes the first time)"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh
fi

# 6. D3DMetal from Apple's Game Porting Toolkit.
if [[ -d $WINE_DIR/lib/external/D3DMetal.framework ]]; then
  say "D3DMetal already added"
else
  dmg="${GPTK_DMG:-$(ls -t "$HOME"/Downloads/Game_Porting_Toolkit*.dmg 2> /dev/null | head -n 1)}"
  if [[ -z $dmg || ! -f $dmg ]]; then
    print
    print -P "%F{yellow}One manual step is left.%f Apple only gives D3DMetal to people who sign in, so you have to download it yourself:"
    print "  1. Open https://developer.apple.com/download/all/?q=game%20porting%20toolkit"
    print "  2. Sign in with any Apple account (a free one works) and download the newest Game Porting Toolkit .dmg"
    print "  3. Leave it in your Downloads folder and run ./setup.sh again"
    exit 1
  fi
  say "adding D3DMetal from ${dmg:t}"
  outer="$(mktemp -d)"; inner="$(mktemp -d)"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$outer" "$dmg"
  eval_dmg="$(ls "$outer"/Evaluation*.dmg | head -n 1)"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$inner" "$eval_dmg"
  ditto "$inner/redist/lib/" "$WINE_DIR/lib/"
  hdiutil detach -quiet "$inner"; hdiutil detach -quiet "$outer"
fi

# 7. VIctory itself.
say "building VIctory"
cargo build --quiet --release -p victory --target x86_64-pc-windows-gnullvm
[[ -f victory.toml ]] || cp victory.example.toml victory.toml
tools/macos/victory.sh config > /dev/null 2>&1 || die "VIctory did not start under Wine. Run tools/macos/victory.sh config to see why"

print
print -P "%F{green}Setup finished.%f"
print "Put your own copy of the game in a folder, set game_dir in victory.toml, and start it with:"
print "  tools/macos/victory.sh run --execute <the game's executable>"
