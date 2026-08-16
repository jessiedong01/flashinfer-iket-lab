"""LL vs BT vs HT protocol comparison for flashinfer's MNNVL CuTe-DSL AllReduce fusion,
profiled with IKET, using a 3x3 Latin-square execution order per shape.

Shapes: the only upstream-tuned configuration is hidden_size=8192, tp=8, bf16
(the GB300 TP8 H8192 presets), so the real-shape axis is the token count m.
The m values cover the three serving regimes the DEFAULT_CONFIG router uses:
decode (m <= ~52 -> LL), mixed batch (m <= 1024 -> BT), prefill chunks (m > 1024 -> HT).

Latin square: for each m, the three protocols run in 3 rotated orders
(LL,BT,HT), (BT,HT,LL), (HT,LL,BT), so every protocol appears once in every
position. This cancels order effects (clock ramp, cache state, compile warmup).
One extra warmup execution per (m, protocol) runs before the square; the
analysis skips it.

Launch (8 ranks, one B300 each):

    FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/mnnvl --clobber \
        profile --postprocess all -- \
        torchrun --nproc-per-node 8 scripts/mnnvl_protocols_iket_driver.py

The driver prints one MANIFEST line per kernel-executing call, in execution
order, so trace launches can be mapped back to (m, phase, position, protocol).
"""

import argparse
import os

import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import BT_ONLY_CONFIG, HT_ONLY_CONFIG, LL_ONLY_CONFIG
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

HIDDEN = 8192
TOP_K = 10
RMS_EPS = 1e-6
WEIGHT_BIAS = 1.0
CONFIGS = {"ll": LL_ONLY_CONFIG, "bt": BT_ONLY_CONFIG, "ht": HT_ONLY_CONFIG}
BT_MAX_M = 1024  # BT_ONLY_CONFIG routes stop at m=1024 upstream


def latin_orders(protocols):
    n = len(protocols)
    rows = [tuple(protocols[(r + i) % n] for i in range(n)) for r in range(n)]
    if n == 2:
        rows = rows * 2  # two 2x2 squares -> 4 reps, positions balanced
    return rows


def reference(local_stack, residual, gamma):
    prenorm = local_stack.float().sum(dim=0) + residual.float()
    inv_rms = torch.rsqrt(prenorm.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    return prenorm, prenorm * inv_rms * (gamma.float() + WEIGHT_BIAS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-list", default="8,32,256,1024,4096,8192")
    args = ap.parse_args()
    m_list = [int(x) for x in args.m_list.split(",")]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", device_id=torch.device("cuda", local_rank)
    )
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    cap = max(m_list)

    workspaces = {}
    for name, cfg in CONFIGS.items():
        ws_cap = min(cap, BT_MAX_M) if name == "bt" else cap
        workspaces[name] = MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=world,
            tp_rank=rank,
            max_token_num=ws_cap,
            hidden_dim=HIDDEN,
            dtype=torch.bfloat16,
            group=group,
            top_k=TOP_K,
            rms_eps=RMS_EPS,
            weight_bias=WEIGHT_BIAS,
            config=cfg,
        )
        torch.cuda.synchronize()
        dist.barrier(group)

    exec_idx = 0
    for m in m_list:
        gen = torch.Generator(device="cuda").manual_seed(1234 + m)
        residual = torch.randn(m, HIDDEN, generator=gen, dtype=torch.bfloat16, device="cuda")
        gamma = torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device="cuda")
        rank_gen = torch.Generator(device="cuda").manual_seed(1000 * (rank + 1) + m)
        local = 0.125 * torch.randn(
            m, HIDDEN, generator=rank_gen, dtype=torch.bfloat16, device="cuda"
        )

        # torch reference: gather every rank's input, sum, add residual, rmsnorm
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local, group=group)
        ref_prenorm, ref_norm = reference(torch.stack(gathered), residual, gamma)

        def run_once(protocol, m=m, local=local, residual=residual, gamma=gamma):
            residual_out = torch.empty_like(local)
            norm_out = torch.empty_like(local)
            allreduce_fusion(
                input=local,
                workspace=workspaces[protocol],
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                weight_bias=WEIGHT_BIAS,
            )
            torch.cuda.synchronize()
            return residual_out, norm_out

        protocols = ("ll", "bt", "ht") if m <= BT_MAX_M else ("ll", "ht")
        if rank == 0 and len(protocols) == 2:
            print(f"NOTE m={m}: bt skipped (BT_ONLY_CONFIG caps at m={BT_MAX_M})", flush=True)

        # warmup + correctness check, one per protocol
        for protocol in protocols:
            residual_out, norm_out = run_once(protocol)
            cos = torch.nn.functional.cosine_similarity(
                ref_norm.reshape(-1), norm_out.float().reshape(-1), dim=0
            ).item()
            max_diff = (ref_prenorm - residual_out.float()).abs().max().item()
            if rank == 0:
                print(
                    f"MANIFEST exec={exec_idx} m={m} phase=warmup pos=- proto={protocol} "
                    f"norm_cos={cos:.6f} prenorm_max_diff={max_diff:.4f}",
                    flush=True,
                )
            assert cos > 0.999, f"{protocol} m={m}: cos {cos}"
            exec_idx += 1
            dist.barrier(group)

        # Latin square (3x3, or two balanced 2x2 rounds when bt is out of range)
        for order_idx, order in enumerate(latin_orders(protocols)):
            for pos, protocol in enumerate(order):
                run_once(protocol)
                if rank == 0:
                    print(
                        f"MANIFEST exec={exec_idx} m={m} phase=latin order={order_idx} "
                        f"pos={pos} proto={protocol}",
                        flush=True,
                    )
                exec_idx += 1
                dist.barrier(group)

    if rank == 0:
        print(f"done: {exec_idx} executions across m={m_list}", flush=True)
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
