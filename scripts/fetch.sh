#!/usr/bin/env bash
#
# fetch.sh — download the models that switch_top_model.py's LAUNCH_RECIPES expect
# but that are NOT present on scotland (/Users/joe/Models).
#
# Run with bash (NOT the interactive fish shell):  bash scripts/fetch.sh
# It is idempotent — hf skips files already fully downloaded, so re-running is safe.
#
# Repo IDs + exact GGUF filenames are taken from the repo docs (per-model summaries and
# benchmark logs), not guessed. Two recipes need more than a model download — this script
# fetches their weights, `scripts/install_stack.sh` builds their runtimes:
#   * qwen3.6-35b-a3b-turboquant-turbo3 — needs the TheTom llama.cpp fork built
#     (the GGUF base it quantizes at runtime IS fetched below).
#   * openbmb/MiniCPM5-1B — needs an sglang source install; the HF checkpoint IS fetched
#     (into the HF cache, where sglang --model-path resolves it).
#
# Env overrides:
#   MODELS_DIR   LM Studio downloadsFolder (default: /Users/joe/Models)
#   HF           path to the hf CLI        (default: hf on PATH, else ~/.local/bin/hf)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/Users/joe/Models}"
HF="${HF:-$(command -v hf || echo "$HOME/.local/bin/hf")}"

if [[ ! -x "$HF" ]]; then
  echo "error: hf CLI not found (looked for '$HF'). Install with: pip install -U huggingface_hub" >&2
  exit 1
fi
mkdir -p "$MODELS_DIR"

echo "hf:         $HF"
echo "models dir: $MODELS_DIR"
avail_gib=$(df -g "$MODELS_DIR" | awk 'NR==2 {print $4}')
echo "free space: ${avail_gib} GiB  (the 7 models total ~166 GB)"
if [[ "${avail_gib:-0}" -lt 180 ]]; then
  echo "warning: <180 GiB free — a full fetch may not fit. Ctrl-C to abort, or fetch selectively." >&2
fi
echo

# --- helpers ---------------------------------------------------------------

# dl_gguf <publisher> <repo> <file.gguf>
# Single-file GGUF → <MODELS_DIR>/<publisher>/<repo>/<file>  (LM Studio auto-indexes it).
dl_gguf() {
  local publisher="$1" repo="$2" file="$3"
  local dest="$MODELS_DIR/$publisher/$repo"
  echo "==> $publisher/$repo :: $file"
  "$HF" download "$publisher/$repo" "$file" --local-dir "$dest"
}

# dl_repo <publisher> <repo>
# Full multi-file repo (MLX safetensors / HF checkpoint) → <MODELS_DIR>/<publisher>/<repo>.
dl_repo() {
  local publisher="$1" repo="$2"
  local dest="$MODELS_DIR/$publisher/$repo"
  echo "==> $publisher/$repo :: (full repo)"
  "$HF" download "$publisher/$repo" --local-dir "$dest"
}

# --- the 7 missing models --------------------------------------------------

# 1. unsloth Qwen3.6-35B-A3B UD-Q6_K — base blob the TheTom turbo3 recipe quantizes.
dl_gguf unsloth Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q6_K.gguf

# 2. TrevorJS Gemma 4 26B-A4B Uncensored Q8_0 (lm-studio).
dl_gguf TrevorJS gemma-4-26B-A4B-it-uncensored-GGUF gemma-4-26B-A4B-it-uncensored-Q8_0.gguf

# 3. prithivMLmods Qwen3.6-35B-A3B Aggressive Q6_K — mradermacher plain-GGUF quant (lm-studio).
dl_gguf mradermacher Qwen3.6-35B-A3B-Uncensored-Aggressive-GGUF Qwen3.6-35B-A3B-Uncensored-Aggressive.Q6_K.gguf

# 4. HauhauCS Qwen3.6-35B-A3B Aggressive Q6_K_P (lm-studio).
#    NB: custom K_P quant — always hf-download directly; `lms get` mis-resolves K_P labels.
dl_gguf HauhauCS Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf

# 5. Dolphin Mistral 24B Venice Edition, MLX 8-bit (lm-studio).
dl_repo mlx-community Dolphin-Mistral-24B-Venice-Edition-mlx-8Bit

# 6. Dolphin 3.0 R1 Mistral 24B, MLX 8-bit (lm-studio).
dl_repo moot20 Dolphin3.0-R1-Mistral-24B-MLX-8bits

# 7. openbmb MiniCPM5-1B HF checkpoint (sglang). Into the HF cache, where sglang
#    --model-path openbmb/MiniCPM5-1B resolves it — NOT the LM Studio models dir.
echo "==> openbmb/MiniCPM5-1B :: (full repo → HF cache for sglang)"
"$HF" download openbmb/MiniCPM5-1B

echo
echo "Done. Verify LM Studio indexed the GGUF/MLX models:  ~/.cache/lm-studio/bin/lms ls"
echo "Next: bash scripts/install_stack.sh  — builds the TheTom llama.cpp fork (turbo3 recipe)"
echo "      and the sglang MPS venv (MiniCPM5 recipe) under \$STACK_DIR (default ~/Stack)."
