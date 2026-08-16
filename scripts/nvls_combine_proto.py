"""NVLS switch-reduced MoE combine: standalone proof kernel vs push+reduce baseline.

Context (iket-lab kernel-innovations campaign): the CuTeDSL megamoe decode combine
is fc2 -> NVLink unicast push into home-rank (token, topk, hidden) staging ->
separate TopkReduce kernel.  DeepGEMM fuses the same push + local reduce into its
persistent kernel.  Both are unicast-push architectures.  This proto measures a
third architecture on B300 NVL8:

  NVLS combine: every rank writes its fc2 contributions LOCALLY (no NVLink on the
  writer side) into a symmetric multicast-mapped partials buffer P[global_token,
  hidden] (local red.add for topk collisions), then after one multimem barrier the
  home rank issues multimem.ld_reduce (add.acc::f32, v4.bf16x2) on the MC address:
  the NVSwitch sums the 8 ranks' partial rows in-fabric, delivering the finished
  combine at 1/8 the ingress bytes of a pull and 0 sys-scope atomics.

Both paths use identical simulated fc2 traffic at DeepSeek-V4-Pro decode geometry
(hidden 7168, top-6 of 384 experts, m tokens/rank, EP8; topk weights pre-applied
upstream, as in the real kernel where PostSwiglu folds them before fc2).

Fabric math per rank (m=128): baseline pushes 6144/8*14336B ~= 11 MB egress and
receives ~11 MB; NVLS writes ~11 MB locally, then the reader's ld_reduce pulls
128*7168*2 = 1.8 MB of switch-reduced rows (egress ~12.9 MB serving other ranks'
reductions).  NVLS additionally pays a P re-zero (14.3 MB local memset) per launch.

Launch:
  torchrun --nproc-per-node 8 scripts/nvls_combine_proto.py --tokens 128 --iters 30
"""

import argparse
import os

import torch
import torch.distributed as dist

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flashinfer.comm.mnnvl_cutedsl.symmetric_buffer import SymmetricBuffer
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    bf16x8_to_packed_u32x4,
    ldmc_bf16x8,
    load_global_u32x4_address,
    packed_u32x4_to_bf16x8,
    store_global_u32x4,
)
from flashinfer.cute_dsl.gemm_allreduce_two_shot import (
    sm_wise_inter_gpu_multimem_barrier,
)

HIDDEN = 7168
TOPK = 6
NUM_EXPERTS = 384
THREADS = 256
CHUNKS = HIDDEN * 2 // 16  # 16B chunks per bf16 row = 896
PASSES = (CHUNKS + THREADS - 1) // THREADS  # 4 (last pass predicated)


# ---------------------------------------------------------------------------
# extra asm helper: local-scope vector bf16 reduction (writer-side collisions)
# ---------------------------------------------------------------------------


@dsl_user_op
def _red_add_relaxed_gpu_v4_bf16x2(address: Int64, packed, *, loc=None, ip=None):
    """``red.relaxed.gpu.global.add.noftz.v4.bf16x2 [addr], {v0..v3};``

    16B local-scope reduction: the NVLS writer's collision-safe accumulate into
    the rank's own P copy.  gpu scope: P is only ever written by its owner rank
    (readers come through the MC address after a sys barrier).
    """
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "red.relaxed.gpu.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@cute.kernel
def _k_push(
    contrib_base: Int64,
    tok: cute.Tensor,
    slot: cute.Tensor,
    peer: cute.Tensor,
    m_tokens: Int32,
):
    """Baseline writer: unicast NVLink push of each contribution row into the
    home rank's (token, topk, hidden) staging (the megamoe STG fc2-return shape).
    grid.x = n_pairs, one CTA per contribution row."""
    pair = cute.arch.block_idx()[0]
    tidx = cute.arch.thread_idx()[0]
    t = tok[pair]
    s = slot[pair]
    dst_rank = t // m_tokens
    local_t = t % m_tokens
    dst_base = peer[dst_rank] + Int64(
        (local_t * Int32(TOPK) + s) * Int32(HIDDEN) * Int32(2)
    )
    src_base = contrib_base + Int64(pair) * Int64(HIDDEN * 2)
    for i in cutlass.range_constexpr(PASSES):
        c = Int32(i * THREADS) + tidx
        if c < Int32(CHUNKS):
            v = load_global_u32x4_address(src_base + Int64(c) * Int64(16))
            store_global_u32x4(dst_base + Int64(c) * Int64(16), v)


@cute.kernel
def _k_reduce(staging_base: Int64, y_base: Int64):
    """Baseline reader: local TopkReduce equivalent -- fp32-accumulate the topk
    staged rows of one token, store bf16 y row.  grid.x = m."""
    t = cute.arch.block_idx()[0]
    tidx = cute.arch.thread_idx()[0]
    for i in cutlass.range_constexpr(PASSES):
        c = Int32(i * THREADS) + tidx
        if c < Int32(CHUNKS):
            acc = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
            acc.fill(Float32(0.0))
            for k in cutlass.range_constexpr(TOPK):
                addr = staging_base + Int64(
                    (t * Int32(TOPK) + Int32(k)) * Int32(HIDDEN) * Int32(2)
                    + c * Int32(16)
                )
                v = load_global_u32x4_address(addr)
                acc.store(acc.load() + packed_u32x4_to_bf16x8(v).to(Float32))
            out = bf16x8_to_packed_u32x4(acc.load().to(BFloat16))
            store_global_u32x4(
                y_base + Int64(t * Int32(HIDDEN) * Int32(2) + c * Int32(16)), out
            )


@cute.kernel
def _k_nvls_write(contrib_base: Int64, tok: cute.Tensor, p_base: Int64):
    """NVLS writer: LOCAL red.add of each contribution row into this rank's own
    copy of P[global_token, hidden].  Zero NVLink traffic.  grid.x = n_pairs."""
    pair = cute.arch.block_idx()[0]
    tidx = cute.arch.thread_idx()[0]
    t = tok[pair]
    dst_base = p_base + Int64(t) * Int64(HIDDEN * 2)
    src_base = contrib_base + Int64(pair) * Int64(HIDDEN * 2)
    for i in cutlass.range_constexpr(PASSES):
        c = Int32(i * THREADS) + tidx
        if c < Int32(CHUNKS):
            v = load_global_u32x4_address(src_base + Int64(c) * Int64(16))
            _red_add_relaxed_gpu_v4_bf16x2(dst_base + Int64(c) * Int64(16), v)


@cute.kernel
def _k_nvls_read(p_mc_base: Int64, y_base: Int64, row0: Int32):
    """NVLS reader: one multimem.ld_reduce (add.acc::f32) sweep over this rank's
    home token rows -- the NVSwitch returns the 8-rank sum.  grid.x = m.

    All PASSES ld_reduce ops are issued back-to-back into registers before any
    store: ldmc is a side-effecting asm block the compiler will not reorder, so
    an interleaved load/store loop serializes PASSES switch round-trips per
    thread (measured 26us for 1.8 MB at EP4); batching keeps them in flight."""
    t = cute.arch.block_idx()[0]
    tidx = cute.arch.thread_idx()[0]
    src_row = p_mc_base + Int64(row0 + t) * Int64(HIDDEN * 2)
    dst_row = y_base + Int64(t) * Int64(HIDDEN * 2)
    vals = cute.make_rmem_tensor(cute.make_layout((PASSES, 4)), Uint32)
    for i in cutlass.range_constexpr(PASSES):
        c = Int32(i * THREADS) + tidx
        if c < Int32(CHUNKS):
            v = ldmc_bf16x8(src_row + Int64(c) * Int64(16))
            for j in cutlass.range_constexpr(4):
                vals[i, j] = v[j]
    for i in cutlass.range_constexpr(PASSES):
        c = Int32(i * THREADS) + tidx
        if c < Int32(CHUNKS):
            store_global_u32x4(
                dst_row + Int64(c) * Int64(16), vals[i, None].load()
            )


@cute.kernel
def _k_barrier(flag_uc: cute.Tensor, flag_mc: cute.Tensor, slot: Int32, ranks: Int32):
    """One-CTA inter-GPU barrier: multimem arrive + CAS-acquire spin.  A fresh
    flag slot per call (monotonic) sidesteps reuse/wraparound races entirely."""
    tidx = cute.arch.thread_idx()[0]
    if tidx == 0:
        sm_wise_inter_gpu_multimem_barrier(
            flag_uc.iterator + slot, flag_mc.iterator + slot, ranks
        )


# ---------------------------------------------------------------------------
# jit launchers
# ---------------------------------------------------------------------------


@cute.jit
def launch_push(
    contrib_base: Int64,
    tok: cute.Tensor,
    slot: cute.Tensor,
    peer: cute.Tensor,
    m_tokens: Int32,
    n_pairs: Int32,
    stream: cuda.CUstream,
):
    _k_push(contrib_base, tok, slot, peer, m_tokens).launch(
        grid=[n_pairs, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_reduce(
    staging_base: Int64, y_base: Int64, m_tokens: Int32, stream: cuda.CUstream
):
    _k_reduce(staging_base, y_base).launch(
        grid=[m_tokens, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_nvls_write(
    contrib_base: Int64,
    tok: cute.Tensor,
    p_base: Int64,
    n_pairs: Int32,
    stream: cuda.CUstream,
):
    _k_nvls_write(contrib_base, tok, p_base).launch(
        grid=[n_pairs, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_nvls_read(
    p_mc_base: Int64, y_base: Int64, row0: Int32, m_tokens: Int32, stream: cuda.CUstream
):
    _k_nvls_read(p_mc_base, y_base, row0).launch(
        grid=[m_tokens, 1, 1], block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_barrier(
    flag_uc: cute.Tensor,
    flag_mc: cute.Tensor,
    slot: Int32,
    ranks: Int32,
    stream: cuda.CUstream,
):
    _k_barrier(flag_uc, flag_mc, slot, ranks).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream
    )


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------


def make_i32(t: torch.Tensor):
    return from_dlpack(t).mark_layout_dynamic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128, help="tokens per rank")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=device
    )
    group = dist.group.WORLD

    m = args.tokens
    T_tokens = world * m
    experts_per_rank = NUM_EXPERTS // world

    # -- identical problem on every rank (same seed => no comms needed) -------
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    topk_ids = torch.rand(T_tokens, NUM_EXPERTS, generator=gen).argsort(-1)[:, :TOPK]
    contrib_all = (
        torch.randn(T_tokens, TOPK, HIDDEN, generator=gen, dtype=torch.float32)
        .bfloat16()
        .to(device)
    )
    expert_rank = topk_ids // experts_per_rank  # (T, K)
    pairs = (expert_rank == rank).nonzero()  # (n_pairs, 2) [t, k]
    n_pairs = pairs.shape[0]
    tok_ids = pairs[:, 0].int().contiguous().to(device)
    slot_ids = pairs[:, 1].int().contiguous().to(device)
    contrib = contrib_all[pairs[:, 0], pairs[:, 1]].contiguous()  # (n_pairs, H) bf16
    y_ref = (
        contrib_all[rank * m : (rank + 1) * m].float().sum(1)
    )  # home tokens, fp32 ref
    del contrib_all
    torch.cuda.empty_cache()

    # -- symmetric buffers -----------------------------------------------------
    staging = SymmetricBuffer.allocate(
        (m, TOPK, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        group=group,
        require_multicast=False,
        materialize_peer_addresses=True,
    )
    P = SymmetricBuffer.allocate(
        (T_tokens, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        group=group,
        require_multicast=True,
        materialize_peer_addresses=False,
    )
    n_flags = 4 * args.iters + 64
    flags = SymmetricBuffer.allocate(
        (n_flags,),
        dtype=torch.int32,
        device=device,
        group=group,
        require_multicast=True,
        materialize_peer_addresses=False,
    )
    flags.tensor.zero_()
    P.tensor.zero_()
    flags_mc_t = torch.empty(0)  # placeholder to keep names obvious below

    y_base_t = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=device)
    y_nvls_t = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=device)

    import cutlass.torch as cutlass_torch

    flags_mc_view = cutlass_torch.as_tensor(
        flags.multicast_address, flags.tensor.shape, flags.tensor.dtype
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    tok_c = make_i32(tok_ids)
    slot_c = make_i32(slot_ids)
    peer_c = make_i32(staging.peer_addresses)
    flag_uc_c = make_i32(flags.tensor)
    flag_mc_c = make_i32(flags_mc_view)

    contrib_base = Int64(contrib.data_ptr())
    staging_base = Int64(staging.tensor.data_ptr())
    p_base = Int64(P.tensor.data_ptr())
    p_mc_base = Int64(P.multicast_address)
    y_base = Int64(y_base_t.data_ptr())
    y_nvls = Int64(y_nvls_t.data_ptr())

    dist.barrier()

    # -- compile ---------------------------------------------------------------
    c_push = cute.compile(
        launch_push, contrib_base, tok_c, slot_c, peer_c, Int32(m), Int32(n_pairs),
        stream,
    )
    c_reduce = cute.compile(launch_reduce, staging_base, y_base, Int32(m), stream)
    c_write = cute.compile(
        launch_nvls_write, contrib_base, tok_c, p_base, Int32(n_pairs), stream
    )
    c_read = cute.compile(
        launch_nvls_read, p_mc_base, y_nvls, Int32(rank * m), Int32(m), stream
    )
    c_bar = cute.compile(
        launch_barrier, flag_uc_c, flag_mc_c, Int32(0), Int32(world), stream
    )

    bar_slot = [0]

    def barrier_kernel():
        c_bar(flag_uc_c, flag_mc_c, Int32(bar_slot[0]), Int32(world), stream)
        bar_slot[0] += 1
        assert bar_slot[0] < n_flags

    def run_baseline():
        c_push(contrib_base, tok_c, slot_c, peer_c, Int32(m), Int32(n_pairs), stream)
        barrier_kernel()
        c_reduce(staging_base, y_base, Int32(m), stream)

    def run_nvls():
        # Zero at iteration HEAD: the paced loop's host barrier guarantees all
        # ranks' previous-iteration readers are drained, so zeroing our local P
        # copy cannot race a peer's in-flight multimem.ld_reduce.  (Zeroing at
        # the tail did race: a slow peer reader saw partially-zeroed rows.)
        P.tensor.zero_()
        c_write(contrib_base, tok_c, p_base, Int32(n_pairs), stream)
        barrier_kernel()
        c_read(p_mc_base, y_nvls, Int32(rank * m), Int32(m), stream)

    def run_nvls_hot():
        # Timing-only variant without the re-zero (results are garbage after
        # the first call; atomic-add/read cost is layout-identical).  In an
        # integrated kernel the zero overlaps the ~300us weight-bound mainloop.
        c_write(contrib_base, tok_c, p_base, Int32(n_pairs), stream)
        barrier_kernel()
        c_read(p_mc_base, y_nvls, Int32(rank * m), Int32(m), stream)

    # -- correctness -----------------------------------------------------------
    dist.barrier()
    run_baseline()
    torch.cuda.synchronize()
    dist.barrier()
    rel_base = (
        (y_base_t.float() - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
    ).item()
    run_nvls()
    torch.cuda.synchronize()
    dist.barrier()
    rel_nvls = (
        (y_nvls_t.float() - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
    ).item()
    xrank = int((tok_ids // m != rank).sum().item())
    print(
        f"CHECK rank={rank} n_pairs={n_pairs} cross_rank_pairs={xrank} "
        f"rel_l2_baseline={rel_base:.4g} rel_l2_nvls={rel_nvls:.4g}",
        flush=True,
    )
    assert rel_base < 0.03, f"baseline mismatch {rel_base}"
    assert rel_nvls < 0.03, f"nvls mismatch {rel_nvls}"

    # -- paced timing (serving-like) --------------------------------------------
    def bench(fn, iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        out = []
        for _ in range(iters):
            dist.barrier()
            ev0.record()
            fn()
            ev1.record()
            torch.cuda.synchronize()
            out.append(ev0.elapsed_time(ev1))
        out.sort()
        return out[len(out) // 2], out[int(len(out) * 0.9)]

    for _ in range(3):  # warm
        run_baseline()
        run_nvls()
    torch.cuda.synchronize()

    med_b, p90_b = bench(run_baseline, args.iters)
    med_n, p90_n = bench(run_nvls, args.iters)
    med_nh, p90_nh = bench(run_nvls_hot, args.iters)

    # phase timing: writer / barrier / reader in isolation (still paced)
    med_w, _ = bench(
        lambda: c_write(contrib_base, tok_c, p_base, Int32(n_pairs), stream),
        args.iters,
    )
    med_r, _ = bench(
        lambda: c_read(p_mc_base, y_nvls, Int32(rank * m), Int32(m), stream),
        args.iters,
    )
    med_z, _ = bench(lambda: P.tensor.zero_(), args.iters)
    med_pu, _ = bench(
        lambda: c_push(
            contrib_base, tok_c, slot_c, peer_c, Int32(m), Int32(n_pairs), stream
        ),
        args.iters,
    )
    med_re, _ = bench(lambda: c_reduce(staging_base, y_base, Int32(m), stream), args.iters)
    med_ba, _ = bench(barrier_kernel, args.iters)

    print(
        f"BENCH-COMBINE rank={rank} m={m} pairs={n_pairs} "
        f"baseline_ms={med_b:.4f} (p90 {p90_b:.4f}) nvls_ms={med_n:.4f} (p90 {p90_n:.4f}) "
        f"nvls_hot_ms={med_nh:.4f} (p90 {p90_nh:.4f}) "
        f"| phases: push={med_pu*1e3:.1f}us reduce={med_re*1e3:.1f}us "
        f"nvls_write={med_w*1e3:.1f}us nvls_read={med_r*1e3:.1f}us zero={med_z*1e3:.1f}us "
        f"barrier={med_ba*1e3:.1f}us",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
