# Survey: every communication-related kernel in flashinfer main, and which ones IKET can profile

Date of survey: 2026-08-13, flashinfer commit `b8c21928`.

IKET only works on kernels written in the CuTe DSL. A kernel written in CUDA C++, Triton, or
cuTile cannot be profiled with IKET, no matter how interesting it is. So the survey is split
into two lists.

## List 1: communication kernels written in CuTe DSL (IKET can profile these)

| # | path | what it is | last commit touching it |
|---|---|---|---|
| 1 | `flashinfer/comm/mnnvl_cutedsl/kernel_ll/` | **Low-latency (LL) protocol** allreduce. Lamport-style flags. The device kernels fuse MoE finalize, TP reduction, residual add, and RMSNorm into the allreduce (`_LamportResidualRMSNormDeviceKernel`, `_ScalarFinalizePublishDeviceKernel`, `_QuadFinalizePublishDeviceKernel`, `_SharedOnlyPublishDeviceKernel`). | 2026-08-11 |
| 2 | `flashinfer/comm/mnnvl_cutedsl/kernel_bt/` | **Balanced (BT) protocol** allreduce, same fusion surface. | 2026-08-11 |
| 3 | `flashinfer/comm/mnnvl_cutedsl/kernel_ht/` | **High-throughput (HT) protocol** allreduce, same fusion surface. | 2026-08-11 |
| 4 | `flashinfer/cute_dsl/gemm_allreduce_two_shot.py` | GEMM with a two-shot multimem allreduce fused into the epilogue. This is the kernel profiled in this repo. | 2026-07-30 |
| 5 | `flashinfer/moe_ep/` (the `cutedsl_megamoe` sm100 kernels, mxfp8 and nvfp4) | Expert-parallel MoE megakernels. Peer-to-peer dispatch and combine over NVLink are fused with the grouped GEMM inside one kernel. | 2026-08-13 (touched daily) |

Notes on list 1:

- Kernels 1-3 are selected per token count by the public backend
  `flashinfer/comm/mnnvl_cutedsl_ar.py` (`MNNVLCuteDSLAllReduceFusionWorkspace`). The supported
  fusion patterns are `kARResidualRMSNorm` and `kMoEFinalizeARResidualRMSNorm`.
- The test `tests/comm/test_mnnvl_cutedsl_numerical_contract.py` runs kernels 1-3 with
  WORLD_SIZE 8 or 16 and requires NVLS symmetric-memory mappings. A single 8-GPU NVLink node
  satisfies this (the same requirement the two-shot kernel had, and that worked here).
- Despite the "MNNVL" name, the backend allocates buffers with
  `torch.distributed._symmetric_memory`, so it does not need a multi-node NVLink fabric.

## List 2: communication kernels NOT written in CuTe DSL (IKET cannot profile these)

| path | implementation | what it is |
|---|---|---|
| `flashinfer/comm/trtllm_ar.py` | CUDA C++ | one-shot / two-shot allreduce with residual+RMSNorm(+quant) fusion patterns. The incumbent production path. |
| `flashinfer/comm/trtllm_mnnvl_ar.py` | CUDA C++ | MNNVL allreduce ported from TensorRT-LLM. |
| `flashinfer/comm/vllm_ar.py` | CUDA C++ | vLLM-style custom allreduce. |
| `flashinfer/comm/quantized_allreduce.py` | CUDA C++ | FP8/FP4-quantized allreduce (last touched 2026-07-24). |
| `flashinfer/comm/nvshmem_allreduce.py` | C++ / NVSHMEM | NVSHMEM-based allreduce. |
| `flashinfer/comm/trtllm_alltoall.py`, `trtllm_moe_alltoall.py` | CUDA C++ | MoE all-to-all. |
| `flashinfer/comm/dcp_alltoall.py` | host code + C++ kernels | decode context-parallel all-to-all. |
| `flashinfer/comm/all_gather_matmul/` | cuTile (`import cuda.tile`) and Triton variants | fused all-gather + matmul. cuTile is a different DSL from CuTe DSL, so IKET does not apply. |
| `flashinfer/comm/allreduce.py`, `mixed_comm.py`, `ulysses*.py` | host-side Python | dispatchers and orchestration, no device code. |

## Recommendation for the next profiling target

The `mnnvl_cutedsl` LL kernel is the best next target, for three reasons:

1. It is a true low-latency communication kernel (Lamport flags), which is the regime where
   in-kernel timing is most interesting: the interesting quantities are spin-wait times, and
   only IKET can attribute them per warp from inside the kernel.
2. It is the newest CuTe-DSL communication code in the repo (last touched 2 days before this
   survey), newer than `gemm_allreduce_two_shot`.
3. It comes with two sibling protocols (BT, HT) behind the same interface, so one harness gives
   a three-way protocol comparison (LL vs BT vs HT) across token counts, with per-phase and
   per-rank breakdowns.

The heavyweight follow-up after that is the `moe_ep` megamoe kernel family: it is the most
recent fused-communication code in flashinfer overall, but its multirank test harness is much
larger, so it is a bigger project.
