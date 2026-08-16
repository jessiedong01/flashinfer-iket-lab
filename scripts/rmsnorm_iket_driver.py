"""Driver for IKET profiling of flashinfer's CuTe-DSL fused RMSNorm + FP4 quant kernel.

Run under run-iket so the kernel JIT-compiles inside the profiled process:

    FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/rmsnorm --clobber \
        profile --postprocess all -- python scripts/rmsnorm_iket_driver.py --rows 2048 --hidden 7168

Keep --iters small: every launch is instrumented and collected per-warp.
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048, help="number of tokens (rows)")
    ap.add_argument("--hidden", type=int, default=7168, help="hidden size")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--block-size", type=int, default=16, choices=[16, 32])
    ap.add_argument("--swizzled", action="store_true", help="swizzled scale-factor layout")
    ap.add_argument("--iters", type=int, default=2, help="profiled launches after the compile launch")
    args = ap.parse_args()

    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=dtype)
    w = torch.randn(args.hidden, device="cuda", dtype=dtype)
    global_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # First call compiles the kernel (inside the run-iket process) and launches once.
    y_fp4, block_scale = rmsnorm_fp4quant(
        x,
        w,
        global_scale=global_scale,
        block_size=args.block_size,
        is_sf_swizzled_layout=args.swizzled,
        enable_pdl=False,
    )
    torch.cuda.synchronize()

    for _ in range(args.iters):
        rmsnorm_fp4quant(
            x,
            w,
            y_fp4=y_fp4,
            block_scale=block_scale,
            global_scale=global_scale,
            block_size=args.block_size,
            is_sf_swizzled_layout=args.swizzled,
            enable_pdl=False,
        )
    torch.cuda.synchronize()

    # Cheap sanity check against a fp32 reference on a few rows.
    x32 = x[:4].float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    print(f"ran m={args.rows} h={args.hidden} dtype={args.dtype} iters={args.iters}")
    print(f"y_fp4 {tuple(y_fp4.shape)} {y_fp4.dtype}, scales {tuple(block_scale.shape)} {block_scale.dtype}")
    print(f"ref row-norm max |y|: {ref.abs().max().item():.3f} (sanity only)")


if __name__ == "__main__":
    main()
