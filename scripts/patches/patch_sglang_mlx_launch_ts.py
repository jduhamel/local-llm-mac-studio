"""Patch an SGLang source checkout for a crash on the Apple-Silicon / MLX path.

Why:
  `Scheduler._record_step_counters()` (step-timing metrics, added upstream after this
  repo's 2026-05-30 MiniCPM5 runbook) computes `time.monotonic() - batch.launch_ts`.
  `launch_ts` is stamped in `Scheduler.run_batch()` — but the MLX overlap loop
  (`hardware_backend/mlx/scheduler_mixin.py::event_loop_overlap_mlx`) deliberately
  bypasses `run_batch()`, so it stays None. Every MLX-backend server dies on its
  *first* inference request with:

      TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
      SIGQUIT received ... It usually means one child failed.

  The server starts and answers GET /v1/models fine, so the breakage only shows up
  under real traffic. Reproduced on sglang main @ 2026-07-28 with SGLANG_USE_MLX=1.

Two edits, both metrics-only — no effect on generation:
  1. scheduler_mixin.py — stamp `batch.launch_ts` where the MLX loop launches a batch,
     mirroring what run_batch() does, so prefill spans are actually measured.
  2. scheduler.py — return early when `launch_ts` is still None, so any other
     run_batch()-bypassing path degrades to "no metric" instead of a crash.

Run with the sglang venv's python:
    ~/Stack/sglang/sglang-mps/bin/python scripts/patches/patch_sglang_mlx_launch_ts.py

Re-apply after `git pull` in the checkout (the whole point of a source install).
Env override: SGLANG_DIR — checkout root, if `import sglang` can't resolve it.
"""

import os
import sys
from pathlib import Path

# Edit 1a: the mixin does not import time.
OLD_IMPORT = "import logging\nfrom dataclasses import dataclass"
NEW_IMPORT = "import logging\nimport time\nfrom dataclasses import dataclass"

# Edit 1b: stamp launch_ts at the MLX launch site, as run_batch() would have.
OLD_LAUNCH = "            resolve_forward_inputs(batch, self.future_map)\n"
NEW_LAUNCH = (
    "            # Scheduler.run_batch() stamps launch_ts; this loop bypasses run_batch(),\n"
    "            # so stamp it here or _record_step_counters() divides by None.\n"
    "            batch.launch_ts = time.monotonic()\n"
    "            resolve_forward_inputs(batch, self.future_map)\n"
)

# Edit 2: defensive guard in the metrics accumulator itself.
OLD_GUARD = (
    "        if all(is_health_check_generate_req(req) for req in batch.reqs):\n"
    "            return\n"
)
NEW_GUARD = (
    "        if all(is_health_check_generate_req(req) for req in batch.reqs):\n"
    "            return\n"
    "        if batch.launch_ts is None:\n"
    "            # Reached run_batch()-free (MLX overlap loop): the span is unmeasurable,\n"
    "            # not zero. Skip the counters rather than crashing the scheduler.\n"
    "            return\n"
)


def find_srt_dir() -> Path:
    """Locate `sglang/srt/` in the checkout — env override first, then the import."""
    override = os.environ.get("SGLANG_DIR")
    if override:
        srt = Path(override).expanduser() / "python" / "sglang" / "srt"
        if srt.is_dir():
            return srt
        print(f"Error: $SGLANG_DIR set but {srt} not found.", file=sys.stderr)
        sys.exit(1)
    try:
        import sglang
    except ImportError:
        print("Error: sglang not importable. Run this with the sglang venv's python "
              "(e.g. ~/Stack/sglang/sglang-mps/bin/python), or set $SGLANG_DIR.",
              file=sys.stderr)
        sys.exit(1)
    srt = Path(sglang.__file__).parent / "srt"
    if not srt.is_dir():
        print(f"Error: {srt} not found.", file=sys.stderr)
        sys.exit(1)
    return srt


def apply(path: Path, edits) -> bool:
    """Apply (label, old, new) edits to `path`. Returns True if the file changed."""
    text = path.read_text()
    changed = False
    for label, old, new in edits:
        if new in text and old not in text:
            print(f"{path.name} — {label}: already patched.")
        elif old in text:
            if text.count(old) != 1:
                print(f"Warning: {path.name} — {label}: anchor found "
                      f"{text.count(old)}× (expected 1); skipping.", file=sys.stderr)
                continue
            text = text.replace(old, new)
            print(f"{path.name} — {label}: patched.")
            changed = True
        else:
            print(f"Warning: {path.name} — {label}: anchor not found. SGLang may have "
                  f"moved past this fix — verify upstream before re-patching.", file=sys.stderr)
    if changed:
        path.write_text(text)
        print(f"Wrote {path}")
    return changed


def main() -> int:
    srt = find_srt_dir()
    mixin = srt / "hardware_backend" / "mlx" / "scheduler_mixin.py"
    scheduler = srt / "managers" / "scheduler.py"
    for p in (mixin, scheduler):
        if not p.exists():
            print(f"Error: {p} not found — SGLang layout changed.", file=sys.stderr)
            return 1

    changed = apply(mixin, [
        ("import time", OLD_IMPORT, NEW_IMPORT),
        ("stamp launch_ts at MLX launch", OLD_LAUNCH, NEW_LAUNCH),
    ])
    changed |= apply(scheduler, [
        ("guard None launch_ts", OLD_GUARD, NEW_GUARD),
    ])

    print("Patched — restart the server." if changed else "No changes needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
