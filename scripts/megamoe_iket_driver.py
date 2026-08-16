"""CuTeDSL NVFP4 MegaMoE at DeepSeek-V4-Pro shapes: IKET profiling + A/B driver.

Real shapes, from DeepSeek-V4-Pro's config.json (HF cache, deepseek-ai/DeepSeek-V4-Pro):
    hidden_size = 7168, moe_intermediate_size = 3072,
    n_routed_experts = 384, num_experts_per_tok (top-k) = 6, expert_dtype = fp4.
EP4 (4 ranks) is the upstream-validated geometry (DeepSeek-V4-Flash e2e ran 4x GB200 EP4).
384 % 4 == 0 -> 96 local experts per rank.

The megamoe kernel drop ships with IKET spans already in the device code
(Dispatch_Prep/Barrier/Pull, Pull.TMA_NVLink_Roundtrip, tma_*_fc1/fc2 waits,
mma_fc1/fc2, fc1/fc2_epi, token_back, Kernel_Tail, ...) behind src/src/iket_compat.py;
with nvidia-cutlass-dsl 4.7.0 the real dialect loads and they light up under run-iket.

Modes:
  oracle   - one launch + pure-torch global-expert oracle check (use small --tokens)
  profile  - --iters launches for IKET tracing (run under run-iket)
  bench    - CUPTI-timed benchmark via flashinfer.testing.bench_gpu_time (NEVER under run-iket)

A/B knobs: --combine {bf16,nvfp4,mxfp8}, --ikr (in_kernel_fc2_reduce).

Launch:  torchrun --nproc-per-node 4 scripts/megamoe_iket_driver.py --mode profile --tokens 128
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

_TESTS = os.path.join(os.path.dirname(__file__), "..", "flashinfer", "tests", "moe_ep")
sys.path.insert(0, _TESTS)

from test_moe_ep_nvfp4_cutedsl_mega_multirank import (  # noqa: E402
    _identity_epilogue_params,
    _make_bf16_weights,
    _make_inputs,
    _megakernel_config,
)

HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 384
TOPK = 6
GATE_UP_CLAMP = 10.0


def build_problem(rank, world_size, num_tokens, max_tokens):
    assert HIDDEN % 128 == 0 and INTERMEDIATE % 128 == 0
    assert NUM_EXPERTS % world_size == 0
    num_local = NUM_EXPERTS // world_size
    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank, num_tokens=num_tokens, hidden=HIDDEN, num_experts=NUM_EXPERTS, topk=TOPK
    )
    # Guarantee cross-rank traffic: token 0 routes one expert per EP rank.
    forced = (
        torch.arange(min(TOPK, world_size), device="cuda", dtype=torch.int64) * num_local
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
    ap.add_argument("--mode", choices=["oracle", "profile", "bench"], default="profile")
    ap.add_argument("--tokens", type=int, default=128, help="tokens per rank")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--iters", type=int, default=3, help="profile-mode launches")
    ap.add_argument("--combine", choices=["bf16", "nvfp4", "mxfp8"], default="bf16")
    ap.add_argument("--ikr", action="store_true", help="in_kernel_fc2_reduce")
    ap.add_argument("--knobs", choices=["none", "auto"], default="none",
                    help="'auto' = collective online autotune at first launch")
    ap.add_argument("--knob", action="append", default=[],
                    help="explicit knob override key=value (python literal), repeatable")
    ap.add_argument("--y-ref", default=None,
                    help="bitwise-exactness anchor: save y here on first use, "
                         "torch.equal-compare on later runs (per-rank files)")
    ap.add_argument("--paced", action="store_true",
                    help="bench with a dist.barrier between iterations (single-launch "
                         "CUDA-event timing). Needed for configs whose cross-rank "
                         "flow control (ikr/quantized combine) assumes paced launches.")
    ap.add_argument("--closure", choices=["copy", "view"], default="copy",
                    help="bench closure: 'copy' = nvfp4_mega_moe with owned-y copy; "
                         "'view' = nvfp4_mega_launch_thunk (bare kernel launch, no "
                         "output copy — matches deepgemm's direct-write closure and "
                         "flashinfer PR #4341 workspace-view semantics)")
    args = ap.parse_args()
    max_tokens = args.max_tokens or args.tokens

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
        nvfp4_mega_moe,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = build_problem(rank, world_size, args.tokens, max_tokens)
    config_extra = dict(in_kernel_fc2_reduce=args.ikr, combine_dtype=args.combine)
    # Explicit knob overrides ride on the session workspace
    # (get_symm_buffer_for_mega_moe(knobs=...)), on top of the heuristic
    # defaults for this token capacity.
    knob_overrides = None
    if args.knob:
        import ast

        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.tuner import (
            default_knobs,
        )

        knob_overrides = dict(default_knobs(max_tokens, dtype="nvfp4"))
        for kv in args.knob:
            key, _, val = kv.partition("=")
            try:
                knob_overrides[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                knob_overrides[key] = val
        if os.environ.get("RANK") == "0":
            print(f"KNOBS {knob_overrides}", flush=True)
    kernel = create_mega_kernel(
        _megakernel_config(problem, epilogue_via_config=True, **config_extra)
    )
    runtime = bootstrap_moe_ep_runtime(bootstrap, kernel.runtime_requirements(bootstrap))
    try:
        n = problem["num_tokens"]
        symm_buffer = get_symm_buffer_for_mega_moe(
            problem["num_experts"],
            problem["max_tokens"],
            problem["topk"],
            problem["hidden"],
            2 * problem["intermediate"],
            rank,
            world_size,
            gate_up_clamp=problem["gate_up_clamp"],
            in_kernel_fc2_reduce=args.ikr,
            combine_dtype=args.combine,
            fc1_alpha=problem["fc1_alpha"],
            fc2_alpha=problem["fc2_alpha"],
            fc1_norm_const=problem["fc1_norm_const"],
            knobs=knob_overrides,
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
        x_local = symm_buffer.x[:n].clone()
        x_sf_local = symm_buffer.x_sf[:n].clone()
        idx_local = symm_buffer.topk_idx[:n].clone()
        w_local = symm_buffer.topk_weights[:n].clone()

        transformed_l1, transformed_l2 = preprocess_mega_weights(
            MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            intermediate_size=problem["intermediate"],
            hidden_size=problem["hidden"],
            gate_up_clamp=problem["gate_up_clamp"],
        )
        y = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device="cuda")

        def run_once():
            nvfp4_mega_moe(
                y,
                transformed_l1,
                transformed_l2,
                symm_buffer,
                num_tokens=n,
                gate_up_clamp=problem["gate_up_clamp"],
                fast_math=problem["fast_math"],
            )

        # compile + first launch
        run_once()
        torch.cuda.synchronize()
        dist.barrier()
        if args.y_ref:
            ref_path = f"{args.y_ref}.rank{rank}.pt"
            if os.path.exists(ref_path):
                y_ref = torch.load(ref_path, map_location="cuda")
                exact = bool(torch.equal(y_ref, y))
                max_d = (y_ref.float() - y.float()).abs().max().item()
                print(f"BITEXACT rank={rank} exact={exact} max|d|={max_d:.6g}", flush=True)
            else:
                os.makedirs(os.path.dirname(ref_path) or ".", exist_ok=True)
                torch.save(y, ref_path)
                if rank == 0:
                    print(f"YREF saved to {args.y_ref}.rank*.pt", flush=True)
        if rank == 0:
            print(
                f"MANIFEST warmup mode={args.mode} tokens={n} combine={args.combine} "
                f"ikr={args.ikr} experts={NUM_EXPERTS} topk={TOPK} h={HIDDEN} i={INTERMEDIATE}",
                flush=True,
            )

        if args.mode == "profile":
            for i in range(args.iters):
                run_once()
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    print(f"MANIFEST exec={i} tokens={n} combine={args.combine} ikr={args.ikr}", flush=True)
            assert torch.isfinite(y).all()

        elif args.mode == "bench":
            from flashinfer.testing import bench_gpu_time

            if args.closure == "view":
                from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
                    nvfp4_mega_launch_thunk,
                )

                run_once = nvfp4_mega_launch_thunk(
                    transformed_l1, transformed_l2, symm_buffer
                )
                run_once()
                torch.cuda.synchronize()
                if args.y_ref:
                    ref_path = f"{args.y_ref}.rank{rank}.pt"
                    if os.path.exists(ref_path):
                        y_ref = torch.load(ref_path, map_location="cuda")
                        view = symm_buffer.output_activation[:n]
                        exact = bool(torch.equal(y_ref, view))
                        print(
                            f"BITEXACT-VIEW rank={rank} exact={exact} "
                            f"max|d|={(y_ref.float() - view.float()).abs().max().item():.6g}",
                            flush=True,
                        )

            dist.barrier()
            if args.paced:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(30):
                    dist.barrier()
                    start.record()
                    run_once()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
            else:
                # Explicit equal iteration counts on every rank: the kernel has
                # cross-rank barriers, so auto (time-based) counts would deadlock.
                times = bench_gpu_time(
                    run_once, dry_run_iters=5, repeat_iters=30, enable_cupti=True,
                    use_cuda_graph=False,
                )
            times = list(times) if isinstance(times, (list, tuple)) else [float(times)]
            med = sorted(times)[len(times) // 2]
            print(
                f"BENCH rank={rank} tokens={n} combine={args.combine} ikr={args.ikr} "
                f"knobs={args.knobs} closure={args.closure} median_ms={float(med):.4f} "
                f"n={len(times)}",
                flush=True,
            )
            dist.barrier()

        elif args.mode == "oracle":
            from test_moe_ep_nvfp4_cutedsl_mega_multirank import _all_gather_stack
            from test_nvfp4_cutedsl_kernel_vs_reference import (
                _plain_nvfp4_from_bf16,
                _torch_nvfp4_mega_reference,
            )

            fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)
            y_ref = _torch_nvfp4_mega_reference(
                act_packed=x_local,
                act_sf=x_sf_local,
                topk_idx=idx_local,
                topk_weights=w_local,
                fc1_weight=_all_gather_stack(fc1_plain).flatten(0, 1),
                fc1_sf=_all_gather_stack(fc1_sf).flatten(0, 1),
                fc2_weight=_all_gather_stack(fc2_plain).flatten(0, 1),
                fc2_sf=_all_gather_stack(fc2_sf).flatten(0, 1),
                hidden=problem["hidden"],
                intermediate=problem["intermediate"],
                gate_up_clamp=problem["gate_up_clamp"],
                term_transform=None,
            )
            yk, yr = y.float(), y_ref.float()
            rel_l2 = ((yk - yr).norm() / yr.norm().clamp_min(1e-6)).item()
            print(f"ORACLE rank={rank} rel_l2={rel_l2:.4g} max|d|={(yk - yr).abs().max().item():.4g}", flush=True)
            assert rel_l2 < 0.03, f"oracle mismatch rel_l2={rel_l2}"

        torch.cuda.synchronize()
        dist.barrier()
        symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


if __name__ == "__main__":
    main()
