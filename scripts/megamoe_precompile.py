"""Precompile (cache-populate) pass for the megamoe brute-force sweep.

FINDINGS that shaped this script (2026-08-13):
  * The megamoe shim compiles via a DIRECT cute.compile call
    (shim/nvfp4.py _ensure_mega_compiled); it never goes through
    flashinfer's JitSpecCuteDsl disk cache (~/.cache/flashinfer cached_ops
    has no megamoe entries).  FLASHINFER_CUTE_DSL_DISABLE_CACHE is
    irrelevant to this kernel either way (left unset per campaign rules).
  * cute.compile HARDWIRES kwargs["no_cache"] = True
    (cutlass.base_dsl.compiler.CompileCallable._compile), so the CuTeDSL
    IR file cache never engages by default -- confirmed empty cache dirs
    after the whole prior campaign.
  * Compilation is INSEPARABLE from workspace creation: cute.compile traces
    with live CUDA tensors incl. the NVSHMEM symmetric-heap workspace
    (ws8), and get_workspace_sizes/get_device_properties need a device.
    A no-GPU, no-dist compile probe is therefore impossible; per-config
    kernels also key on world_size=8, so 1-GPU probes would compile the
    wrong kernel.

Consequence: "precompile" = an 8-rank compile-populate session that runs
apply_knobs + one compile per candidate WITHOUT benching, with the worker's
--dsl-file-cache monkeypatch re-enabling the CuTeDSL IR file cache (keyed on
the traced-IR bytecode SHA -- knob constants are baked in the IR, so no
collisions).  Cache dir: out/bf/dslcache (mounted, survives restarts).
Compile is CPU-bound; the session's GPU footprint is memory + a handful of
tiny staging launches, so it can tolerate a busier box than a bench session
-- but it still spawns 8 ranks, so the torchrun-coexistence gate applies.

Usage:
    python3 scripts/megamoe_precompile.py --candidates out/bf/stage2_candidates.json
"""

import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "/home/jessie/iket-lab/scripts/megamoe_sweep.py",
        "--compile-only",
        "--dsl-file-cache",
        "--batch-size",
        "40",
    ] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))
