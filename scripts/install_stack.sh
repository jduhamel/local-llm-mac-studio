#!/usr/bin/env bash
#
# install_stack.sh — build the source-installed runtimes that switch_top_model.py's
# LAUNCH_RECIPES expect but that ship no binary: mainline llama.cpp, the TheTom llama.cpp
# TurboQuant fork, and SGLang's Apple-Silicon/MLX path.
#
# All land under a single stack root (default ~/Stack) rather than scattered directly
# in $HOME. Point switch_top_model.py at the same root with --stack-dir / $MACSTUDIO_STACK_DIR.
#
# Run with bash (NOT the interactive fish shell):  bash scripts/install_stack.sh
# Idempotent — an existing checkout is fetched + reset instead of re-cloned, and cmake
# reuses its build dir, so re-running is safe and cheap.
#
# Env overrides:
#   STACK_DIR   install root                       (default: ~/Stack)
#   JOBS        cmake parallel build jobs          (default: 8)
#   ONLY        mainline | thetom | sglang         (default: all three)
set -euo pipefail

STACK_DIR="${STACK_DIR:-$HOME/Stack}"
JOBS="${JOBS:-8}"
ONLY="${ONLY:-}"

MAINLINE_REPO="https://github.com/ggml-org/llama.cpp.git"
MAINLINE_DIR="$STACK_DIR/llama-cpp-mainline"

THETOM_REPO="https://github.com/TheTom/llama-cpp-turboquant.git"
THETOM_BRANCH="feature/turboquant-kv-cache"
THETOM_DIR="$STACK_DIR/llama-cpp-thetom"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_DIR="$STACK_DIR/sglang"
SGLANG_VENV="sglang-mps"          # relative to $SGLANG_DIR — matches the runbook

CMAKE="${CMAKE:-$(command -v cmake || echo /opt/homebrew/bin/cmake)}"
UV="${UV:-$(command -v uv || echo /opt/homebrew/bin/uv)}"
PY311="${PY311:-$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)}"

mkdir -p "$STACK_DIR"
echo "stack root: $STACK_DIR"
echo

# clone_or_update <dir> <repo> [branch]
# Fresh clone, or fetch + hard-reset an existing checkout to the branch tip.
clone_or_update() {
  local dir="$1" repo="$2" branch="${3:-}"
  if [[ -d "$dir/.git" ]]; then
    echo "==> updating $dir"
    git -C "$dir" fetch --depth 1 origin ${branch:+"$branch"}
    git -C "$dir" reset --hard FETCH_HEAD
  else
    echo "==> cloning $repo -> $dir"
    git clone --depth 1 ${branch:+-b "$branch"} "$repo" "$dir"
  fi
}

# build_llama_cpp <dir>
# Metal + server build, shared by mainline and every fork. ~2 min on an M3 Ultra
# (the bundled Svelte web UI dominates; the C++ itself is under a minute).
build_llama_cpp() {
  local dir="$1"
  if [[ ! -x "$CMAKE" ]]; then
    echo "error: cmake not found ('$CMAKE'). brew install cmake" >&2; exit 1
  fi
  echo "==> configuring + building $(basename "$dir") (Metal)"
  "$CMAKE" -S "$dir" -B "$dir/build" \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON
  "$CMAKE" --build "$dir/build" --config Release -j "$JOBS"
  echo "built: $dir/build/bin/llama-server"
  echo
}

# --- 1. mainline llama.cpp (port 8100 recipes: MTP, runtime LoRA, stock GGUF) ---
# The preferred llama.cpp binary — the legacy am17an@mtp-clean fork is only kept for
# reproducing old MTP results. Gemma 4 external-assistant support needs a build at or
# past ggml-org/llama.cpp#24277, which any current master satisfies.
if [[ -z "$ONLY" || "$ONLY" == "mainline" ]]; then
  clone_or_update "$MAINLINE_DIR" "$MAINLINE_REPO"
  build_llama_cpp "$MAINLINE_DIR"
fi

# --- 2. TheTom llama.cpp TurboQuant fork (port 8099 recipes) ----------------
if [[ -z "$ONLY" || "$ONLY" == "thetom" ]]; then
  clone_or_update "$THETOM_DIR" "$THETOM_REPO" "$THETOM_BRANCH"
  build_llama_cpp "$THETOM_DIR"
fi

# --- 3. SGLang, Apple-Silicon / MLX path (port 30000 recipes) ---------------
# Source install only — there is no wheel for the MPS extras. Python 3.11 is required;
# the repo ships the Apple variant as python/pyproject_other.toml and it must replace
# python/pyproject.toml before `uv pip install -e 'python[all_mps]'` resolves.
if [[ -z "$ONLY" || "$ONLY" == "sglang" ]]; then
  if [[ ! -x "$UV" ]]; then
    echo "error: uv not found ('$UV'). brew install uv" >&2; exit 1
  fi
  if [[ ! -x "$PY311" ]]; then
    echo "error: python3.11 not found ('$PY311'). brew install python@3.11" >&2; exit 1
  fi
  clone_or_update "$SGLANG_DIR" "$SGLANG_REPO"
  if [[ ! -d "$SGLANG_DIR/$SGLANG_VENV" ]]; then
    echo "==> creating venv $SGLANG_DIR/$SGLANG_VENV (python3.11)"
    (cd "$SGLANG_DIR" && "$UV" venv -p "$PY311" "$SGLANG_VENV")
  fi
  echo "==> installing sglang[all_mps] (editable; several minutes — torch + mlx + transformers)"
  (
    cd "$SGLANG_DIR"
    # shellcheck disable=SC1091
    . "$SGLANG_VENV/bin/activate"
    if [[ -f python/pyproject_other.toml ]]; then
      rm -f python/pyproject.toml
      mv python/pyproject_other.toml python/pyproject.toml
    fi
    "$UV" pip install --upgrade pip
    "$UV" pip install -e 'python[all_mps]'
  )
  # Mandatory: without this, the MLX backend SIGQUITs on its first inference request.
  # Re-apply after any `git pull` in the checkout — see the patch docstring.
  echo "==> patching MLX scheduler (launch_ts crash)"
  "$SGLANG_DIR/$SGLANG_VENV/bin/python" \
    "$(dirname "${BASH_SOURCE[0]}")/patches/patch_sglang_mlx_launch_ts.py"
  echo "installed: $SGLANG_DIR/$SGLANG_VENV"
  echo
fi

echo "Done."
echo "Point the switcher at this root:  export MACSTUDIO_STACK_DIR=$STACK_DIR"
echo "  (or pass --stack-dir $STACK_DIR to scripts/switch_top_model.py)"
