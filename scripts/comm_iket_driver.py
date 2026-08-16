"""Driver for IKET profiling of flashinfer's CuTe-DSL fused GEMM + two-shot AllReduce.

Launch with torchrun under run-iket (8 ranks, one B300 each; the kernel requires
world_size == 8 when all_reduce != "none"):

    FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber \
        profile --postprocess all -- \
        torchrun --nproc-per-node 8 scripts/comm_iket_driver.py

Reuses the upstream test's run() (tensor creation incl. multicast symm-mem tensors,
kernel build, cute.compile, launch, reference check). One trace file per rank pid.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flashinfer", "tests", "gemm"))

import cutlass
import torch
import torch.distributed as dist

from test_cute_dsl_gemm_allreduce_two_shot import run  # noqa: E402


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        device_id=torch.device("cuda", local_rank),
    )
    try:
        run(
            mnkl=(2048, 2048, 4096, 1),
            ab_dtype=cutlass.TFloat32,
            c_dtype=cutlass.Float32,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            use_tma_store=False,
            tolerance=1e-01,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=False,
            use_cold_l2=False,
            all_reduce="two_shot",
        )
        print(f"rank {dist.get_rank()}: gemm+two_shot allreduce OK")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
