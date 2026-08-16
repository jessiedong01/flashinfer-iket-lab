"""Driver for IKET profiling of flashinfer's CuTe-DSL NVFP4 block-scaled GEMM (SM103).

Run under run-iket so the kernel JIT-compiles inside the profiled process:

    FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/gemm --clobber \
        profile --postprocess all -- python scripts/gemm_iket_driver.py --mnk 4096,4096,7168
"""

import argparse

import torch
import torch.nn.functional as F

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnk", default="4096,4096,7168", help="comma-separated M,N,K")
    ap.add_argument("--iters", type=int, default=2, help="profiled launches after the compile launch")
    args = ap.parse_args()
    m, n, k = map(int, args.mnk.split(","))

    torch.manual_seed(0)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    gs_a = (448 * 6) / a.float().abs().nan_to_num().max()
    gs_b = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_inv_s = nvfp4_quantize(a, gs_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b_fp4, b_inv_s = nvfp4_quantize(b, gs_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    alpha = 1.0 / (gs_a * gs_b)

    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    for _ in range(1 + args.iters):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_inv_s,
            b_inv_s.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
            use_nvfp4=True,
        )
    torch.cuda.synchronize()

    ref = torch.mm(a, b.T)
    cos = F.cosine_similarity(ref.float().reshape(-1), out.float().reshape(-1), dim=0)
    print(f"ran mnk=({m},{n},{k}) iters={args.iters} cos_sim={cos.item():.5f}")
    assert cos > 0.97, "reference check failed"


if __name__ == "__main__":
    main()
