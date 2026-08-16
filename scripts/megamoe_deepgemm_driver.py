"""DeepGEMM fp8xfp4 MegaMoE benchmark at DeepSeek-V4-Pro shapes (the rival baseline).

Same geometry as scripts/megamoe_iket_driver.py: hidden 7168, inter 3072,
384 experts, top-k 6, torchrun world_size ranks. DeepGEMM's mega kernel is CUDA C++
(not CuTe DSL), so IKET cannot see inside it — it is benchmarked here purely as the
wall-time rival for the CuTeDSL megamoe hillclimb, with the same CUPTI timing.

Launch:  torchrun --nproc-per-node 8 scripts/megamoe_deepgemm_driver.py --tokens 4096
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

_TESTS = os.path.join(os.path.dirname(__file__), "..", "flashinfer", "tests", "moe_ep")
sys.path.insert(0, _TESTS)

from test_moe_ep_deep_gemm_mega_multirank import (  # noqa: E402
    _make_inputs,
    _make_moe_weight_pack,
)

HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 384
TOPK = 6
ACTIVATION_CLAMP = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--paced", action="store_true",
                    help="per-iteration dist.barrier + single-launch event timing")
    args = ap.parse_args()
    max_tokens = args.max_tokens or args.tokens

    import deep_gemm
    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        ensure_moe_ep_cuda_device,
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.testing import bench_gpu_time

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", device_id=torch.device("cuda", local_rank)
    )
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    group = dist.group.WORLD

    num_local = NUM_EXPERTS // world_size
    n = args.tokens
    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank, num_tokens=n, hidden=HIDDEN, num_experts=NUM_EXPERTS, topk=TOPK
    )
    forced = (
        torch.arange(min(TOPK, world_size), device="cuda", dtype=torch.int64) * num_local
    )
    topk_ids[0, : forced.numel()] = forced
    weights = _make_moe_weight_pack(
        rank, num_local_experts=num_local, hidden=HIDDEN, intermediate=INTERMEDIATE
    )

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, NUM_EXPERTS, max_tokens, TOPK, HIDDEN, INTERMEDIATE
    )
    stage_mega_moe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        symm_buffer.x[:n],
        symm_buffer.x_sf[:n],
        symm_buffer.topk_idx[:n],
        symm_buffer.topk_weights[:n],
    )
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        weights, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    y = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    kernel = DeepGemmMegaKernelBackend(
        DeepGemmMegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOPK,
            activation_clamp=ACTIVATION_CLAMP,
            fast_math=True,
        )
    )

    def run_once():
        kernel.compute(symm_buffer, (transformed_l1, transformed_l2), output=y)

    run_once()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    dist.barrier()

    if args.paced:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(30):
            dist.barrier()
            start_ev.record()
            run_once()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    else:
        times = bench_gpu_time(
            run_once, dry_run_iters=5, repeat_iters=30, enable_cupti=True,
            use_cuda_graph=False,
        )
    times = list(times) if isinstance(times, (list, tuple)) else [float(times)]
    med = sorted(times)[len(times) // 2]
    print(f"BENCH-DG rank={rank} tokens={n} paced={args.paced} median_ms={float(med):.4f} n={len(times)}", flush=True)
    dist.barrier()
    symm_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
