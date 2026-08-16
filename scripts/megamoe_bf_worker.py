"""Brute-force sweep batch worker: bench N megamoe knob configs in ONE torchrun session.

Launched by scripts/megamoe_sweep.py (host orchestrator) as:
    torchrun --nproc-per-node 8 scripts/megamoe_bf_worker.py \
        --batch-json out/bf/batch_X.json --tokens 128 --y-ref out/yref/m128 \
        --results-jsonl out/bf/results_X.jsonl --heartbeat out/bf/hb_X.json \
        [--dsl-file-cache] [--compile-only]

Design mirrors shim/autotune.autotune_knobs: allocate the symm buffer ONCE,
then per candidate frontend.apply_knobs(full_knob_dict) -> recompile ->
bitexact (copy + view) -> CUPTI bench (view closure) -> rank0 appends one
JSONL row.  Collective: every rank walks the same candidate list in lockstep
(barriers around compile and timing); per-candidate ctor/compile failures are
deterministic across ranks (same static problem) so score-and-continue keeps
the collective aligned.  CUDA launch/bench errors abort the session (context
unreliable); the orchestrator marks the in-flight config and reruns the rest.

--dsl-file-cache: surgically re-enables the CuTeDSL IR file cache that
cute.compile hardwires off (kwargs["no_cache"] = True in
cutlass.base_dsl.compiler.CompileCallable._compile).  The cache key is a
SHA over the traced module IR *bytecode* + envars + compile options
(BaseDSL.get_module_hash), so every knob value baked into the IR is in the
key -- distinct configs cannot collide.  Cache dir is CUTE_DSL_CACHE_DIR
(the orchestrator points it into the mounted lab tree so it survives
container restarts and is host-inspectable).  In-process patch only: no
kernel source files are modified.
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

_TESTS = os.path.join(os.path.dirname(__file__), "..", "flashinfer", "tests", "moe_ep")
sys.path.insert(0, _TESTS)

HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 384
TOPK = 6
GATE_UP_CLAMP = 10.0

# Per-phase watchdog budgets (seconds); rank0 writes them into the heartbeat
# file so the host orchestrator kills the session iff a phase overstays.
PHASE_BUDGETS = {
    "bootstrap": 600.0,
    "compile": 420.0,   # mission per-run cap; compile dominates a run
    "bitexact": 240.0,
    "bench": 420.0,
}

_TUPLE_KNOBS = ("mma_tiler_mnk", "cluster_shape_mnk", "epi_flag_batch")


def _normalize_knobs(knobs: dict) -> dict:
    """JSON lists -> tuples for tuple-valued knobs; strip _meta keys (_rep)."""
    out = {k: v for k, v in knobs.items() if not k.startswith("_")}
    for k in _TUPLE_KNOBS:
        if k in out and isinstance(out[k], list):
            out[k] = tuple(out[k])
    return out


def enable_dsl_file_cache() -> None:
    """Re-exec CompileCallable._compile with the no_cache pin flipped off.

    Asserts the exact line exists so a DSL upgrade that moves it fails loudly
    instead of silently benching without the cache.
    """
    import inspect
    import textwrap

    from cutlass.base_dsl import compiler as _comp

    src = inspect.getsource(_comp.CompileCallable._compile)
    needle = 'kwargs["no_cache"] = True'
    assert needle in src, "CuTeDSL _compile changed; no_cache pin not found"
    src = textwrap.dedent(src.replace(needle, 'kwargs["no_cache"] = False'))
    ns = dict(_comp.__dict__)
    exec(compile(src, _comp.__file__, "exec"), ns)
    _comp.CompileCallable._compile = ns["_compile"]


class Heartbeat:
    """Rank0-only phase heartbeat for the host watchdog."""

    def __init__(self, path, rank):
        self.path = path
        self.rank = rank

    def phase(self, config_id, phase):
        if self.rank != 0 or not self.path:
            return
        payload = {
            "config_id": config_id,
            "phase": phase,
            "phase_start": time.time(),
            "budget_s": PHASE_BUDGETS.get(phase, 420.0),
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)


def build_problem(rank, world_size, num_tokens, max_tokens):
    from test_moe_ep_nvfp4_cutedsl_mega_multirank import (
        _identity_epilogue_params,
        _make_bf16_weights,
        _make_inputs,
    )

    num_local = NUM_EXPERTS // world_size
    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank, num_tokens=num_tokens, hidden=HIDDEN, num_experts=NUM_EXPERTS, topk=TOPK
    )
    forced = (
        torch.arange(min(TOPK, world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    topk_ids[0, : forced.numel()] = forced
    w13, w2 = _make_bf16_weights(
        rank, num_local_experts=num_local, hidden=HIDDEN, intermediate=INTERMEDIATE
    )
    fc1_alpha, fc2_alpha, fc1_norm_const = _identity_epilogue_params(num_local)
    return dict(
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=NUM_EXPERTS,
        topk=TOPK,
        gate_up_clamp=GATE_UP_CLAMP,
        fast_math=True,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-json", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--y-ref", default=None)
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--dsl-file-cache", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--repeat-iters", type=int, default=30)
    args = ap.parse_args()

    if args.dsl_file_cache:
        enable_dsl_file_cache()

    with open(args.batch_json) as f:
        batch = json.load(f)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    hb = Heartbeat(args.heartbeat, rank)
    hb.phase("(bootstrap)", "bootstrap")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_launch_thunk,
        nvfp4_mega_moe,
    )
    from test_moe_ep_nvfp4_cutedsl_mega_multirank import _megakernel_config

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    n = args.tokens
    problem = build_problem(rank, world_size, n, n)
    kernel = create_mega_kernel(
        _megakernel_config(
            problem,
            epilogue_via_config=True,
            in_kernel_fc2_reduce=False,
            combine_dtype="bf16",
        )
    )
    runtime = bootstrap_moe_ep_runtime(bootstrap, kernel.runtime_requirements(bootstrap))

    def emit(row: dict) -> None:
        if rank != 0:
            return
        row = dict(row)
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(args.results_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    exit_code = 0
    try:
        # Buffer allocated once with the first candidate's knobs; every
        # candidate then goes through apply_knobs with a FULL knob dict, so
        # no state leaks between candidates.
        first_knobs = _normalize_knobs(batch[0]["knobs"])
        symm_buffer = get_symm_buffer_for_mega_moe(
            problem["num_experts"],
            problem["max_tokens"],
            problem["topk"],
            problem["hidden"],
            2 * problem["intermediate"],
            rank,
            world_size,
            gate_up_clamp=problem["gate_up_clamp"],
            in_kernel_fc2_reduce=False,
            combine_dtype="bf16",
            fc1_alpha=problem["fc1_alpha"],
            fc2_alpha=problem["fc2_alpha"],
            fc1_norm_const=problem["fc1_norm_const"],
            knobs=first_knobs,
        )
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x[:n],
            symm_buffer.x_sf[:n],
            symm_buffer.topk_idx[:n],
            symm_buffer.topk_weights[:n],
        )
        transformed_l1, transformed_l2 = preprocess_mega_weights(
            MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            intermediate_size=problem["intermediate"],
            hidden_size=problem["hidden"],
            gate_up_clamp=problem["gate_up_clamp"],
        )
        y = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device="cuda")

        y_ref = None
        if args.y_ref:
            ref_path = f"{args.y_ref}.rank{rank}.pt"
            if os.path.exists(ref_path):
                y_ref = torch.load(ref_path, map_location="cuda")
            elif rank == 0:
                print(f"WARN no y-ref anchor at {ref_path}", flush=True)

        frontend = symm_buffer._frontend
        from flashinfer.testing import bench_gpu_time

        for cand in batch:
            cid = cand["config_id"]
            knobs = _normalize_knobs(cand["knobs"])
            paced = bool(cand.get("paced", False))
            row_base = dict(
                config_id=cid,
                stage=cand.get("stage"),
                tokens=n,
                knobs=cand["knobs"],
                closure="view",
                paced=paced,
                world_size=world_size,
            )
            t_cfg0 = time.time()
            # ---- compile phase (deterministic failures: score and continue)
            hb.phase(cid, "compile")
            try:
                frontend.apply_knobs(knobs)
                dist.barrier()
                t0 = time.time()
                nvfp4_mega_moe(  # first call compiles, then launches + copies
                    y,
                    transformed_l1,
                    transformed_l2,
                    symm_buffer,
                    num_tokens=n,
                    gate_up_clamp=problem["gate_up_clamp"],
                    fast_math=problem["fast_math"],
                    sync=True,
                )
                compile_s = time.time() - t0
                torch.cuda.synchronize()
                dist.barrier()
            except Exception as exc:  # noqa: BLE001 -- deterministic across ranks
                torch.cuda.synchronize()
                emit(
                    dict(
                        row_base,
                        status="FAIL",
                        error=f"{type(exc).__name__}: {exc}"[:500],
                        compile_s=round(time.time() - t_cfg0, 1),
                    )
                )
                if rank == 0:
                    print(f"CONFIG {cid} FAIL(compile): {exc}", flush=True)
                dist.barrier()
                continue

            if args.compile_only:
                emit(dict(row_base, status="COMPILED", compile_s=round(compile_s, 1)))
                dist.barrier()
                continue

            # ---- bitexact + bench: CUDA errors here poison the context ->
            # record and abort the session; orchestrator reruns the rest.
            try:
                hb.phase(cid, "bitexact")
                bitexact_copy = None
                maxd_copy = None
                if y_ref is not None:
                    exact = bool(torch.equal(y_ref, y))
                    maxd = (y_ref.float() - y.float()).abs().max().item()
                    flags = torch.tensor(
                        [0.0 if exact else 1.0, maxd], dtype=torch.float64, device="cuda"
                    )
                    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                    bitexact_copy = bool(flags[0].item() == 0.0)
                    maxd_copy = flags[1].item()

                thunk = nvfp4_mega_launch_thunk(
                    transformed_l1, transformed_l2, symm_buffer
                )
                thunk()
                torch.cuda.synchronize()
                dist.barrier()
                bitexact_view = None
                maxd_view = None
                if y_ref is not None:
                    view = symm_buffer.output_activation[:n]
                    exact = bool(torch.equal(y_ref, view))
                    maxd = (y_ref.float() - view.float()).abs().max().item()
                    flags = torch.tensor(
                        [0.0 if exact else 1.0, maxd], dtype=torch.float64, device="cuda"
                    )
                    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                    bitexact_view = bool(flags[0].item() == 0.0)
                    maxd_view = flags[1].item()

                hb.phase(cid, "bench")
                t0 = time.time()
                if paced:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    times = []
                    for _ in range(args.repeat_iters):
                        dist.barrier()
                        start.record()
                        thunk()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                else:
                    times = bench_gpu_time(
                        thunk,
                        dry_run_iters=5,
                        repeat_iters=args.repeat_iters,
                        enable_cupti=True,
                        use_cuda_graph=False,
                    )
                bench_s = time.time() - t0
                times = (
                    list(times) if isinstance(times, (list, tuple)) else [float(times)]
                )
                med = float(sorted(times)[len(times) // 2])
                med_t = torch.zeros(world_size, dtype=torch.float64, device="cuda")
                med_t[rank] = med
                dist.all_reduce(med_t, op=dist.ReduceOp.SUM)
                medians = [round(v, 4) for v in med_t.tolist()]
                dist.barrier()
                emit(
                    dict(
                        row_base,
                        status="OK",
                        median_ms=medians[0],
                        median_max_ms=max(medians),
                        medians=medians,
                        bitexact_copy=bitexact_copy,
                        bitexact_view=bitexact_view,
                        maxd_copy=maxd_copy,
                        maxd_view=maxd_view,
                        compile_s=round(compile_s, 1),
                        bench_s=round(bench_s, 1),
                    )
                )
                if rank == 0:
                    print(
                        f"CONFIG {cid} OK median_ms={medians[0]:.4f} "
                        f"max={max(medians):.4f} bitexact={bitexact_copy}/"
                        f"{bitexact_view} compile_s={compile_s:.0f}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 -- CUDA state unreliable
                emit(
                    dict(
                        row_base,
                        status="FAIL_CUDA",
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    )
                )
                if rank == 0:
                    print(
                        f"CONFIG {cid} FAIL_CUDA, aborting session:\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
                exit_code = 3
                break

        hb.phase("(end)", "bootstrap")
        torch.cuda.synchronize()
        dist.barrier()
        symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
