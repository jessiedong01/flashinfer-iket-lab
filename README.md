# flashinfer-iket-lab

In-kernel event tracing (IKET) applied to FlashInfer's CuTe-DSL kernels on 8x NVIDIA B300.
Every step below gives the exact command and the raw output it printed. Uncut logs are in
`logs/`, traces in `traces/`, analyzer output in `results/`, and every kernel edit is the single
patch in `patches/`.

**What IKET is.** An experimental profiler shipping with `nvidia-cutlass-dsl` 4.7.0. Named ranges
go *inside* a CuTe-DSL kernel (`cute.experimental.iket.range_push("name")` / `range_pop()`), the
program runs under the `run-iket` tool, and the result is a per-warp timeline at 32 ns resolution.
It shows what CUPTI-based tools cannot: how long each warp sat in a spin-wait, a pipeline stall or
a barrier, and — for multi-GPU kernels — how cross-rank skew gets absorbed inside the kernel. It
only works on CuTe-DSL kernels; CUDA C++, Triton and cuTile are out of reach.

## Findings

| # | Kernel | What the trace showed |
|---|---|---|
| 1 | RMSNorm + NVFP4 quant | Sum-of-squares and quantize-store cost the same, 2.11 vs 2.16 µs — the row is read twice |
| 2 | NVFP4 block-scaled GEMM | Epilogue warps wait 7.9 of 9.5 µs per tile on the accumulator; the memory pipeline is idle |
| 3 | GEMM + two-shot AllReduce, 8 GPUs | Communication work is balanced at 58–65 µs; time in the kernel ranges 188–786 µs |
| 4 | MNNVL AllReduce, LL vs BT vs HT | HT wins at every token count on a single node, against the shipped router's defaults |
| 5 | MegaMoE at DeepSeek-V4-Pro shapes | Decode is weight-bandwidth bound; a tile change gives −1.7% bit-exact, an NVFP4 combine wire −11.6% at prefill |

Two results came out of the megakernel work that a launch-level profiler could not have produced:

- **A bistable degraded mode.** Three unrelated configurations all benched at ~8.9 ms, 26x above
  baseline. Their traces showed clean 361 µs launches. Re-benching with `dist.barrier` pacing
  returned all three to ~0.34 ms and moved the 9 ms mode onto the *default* configuration. Launch
  pacing triggers it on any config, with `Dispatch_Barrier` absorbing ~4.7 ms.
- **Nsight Compute cannot profile this kernel class at all.** Kernel replay deadlocks against
  cross-rank NVLink barriers. See [Step 10](#step-10-what-ncu-can-and-cannot-see).

## Hardware and software

| item | value |
|---|---|
| GPUs | 8x NVIDIA B300 SXM6 AC (SM103), driver 610.43.02, NVLink + NVLS multicast |
| container | flashinfer dev image: CUDA 13.2, torch 2.13.0.dev cu132, python 3.12 |
| DSL | `nvidia-cutlass-dsl[cu13]==4.7.0` — the first version shipping `run-iket` |
| flashinfer | `main` at `b8c21928`, plus `patches/iket-instrumentation.diff` |

A survey of every communication kernel in flashinfer, classified by whether IKET can reach it, is
in [COMM_KERNEL_SURVEY.md](COMM_KERNEL_SURVEY.md).

---

## Step 1: container and GPU visibility

The image predates driver 610 on the host, so two box-specific fixes are needed: copy the host's
NVML library into the container, and match the container user's uid to the host user so the
bind-mounted home directory stays readable.

```bash
docker run -d --name flashinfer-iket-dev --gpus all --network=host --shm-size=64g \
    --cap-add=SYS_PTRACE --ipc=host -v /home/jessie:/home/jessie \
    --workdir /home/jessie/iket-lab flashinfer-cu132-dev:local sleep infinity

docker cp /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.610.43.02 flashinfer-iket-dev:/usr/lib/x86_64-linux-gnu/
docker exec -u root flashinfer-iket-dev bash -c 'cd /usr/lib/x86_64-linux-gnu && \
    ln -sf libnvidia-ml.so.610.43.02 libnvidia-ml.so.1 && ln -sf libnvidia-ml.so.1 libnvidia-ml.so && ldconfig'
```

```
GPU 0: NVIDIA B300 SXM6 AC (UUID: GPU-daaa8d02-04fc-d666-a488-8eb3721df133)
GPU 1: NVIDIA B300 SXM6 AC (UUID: GPU-d1d60fa6-6734-0390-5487-e6f8b4b572a9)
```

## Step 2: upgrade nvidia-cutlass-dsl to 4.7.0

The image ships 4.5.1, which has no IKET.

```bash
pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.7.0" && run-iket --help
```

```
    Found existing installation: nvidia-cutlass-dsl 4.5.1
    Uninstalling nvidia-cutlass-dsl-4.5.1:
      Successfully uninstalled nvidia-cutlass-dsl-4.5.1

Successfully installed nvidia-cutlass-dsl-4.7.0 nvidia-cutlass-dsl-libs-base-4.7.0 ...
usage: run-iket [-h] [--use-injection-lib USE_INJECTION_LIB] [--skip-run]
                [--context-buffer-size CONTEXT_BUFFER_SIZE]
                [--log-level {error,warn,info,debug,trace}]
                [--working-dir WORKING_DIR] [--output-dir OUTPUT_DIR]
                [--clobber]
                {profile,postprocess} ...

positional arguments:
  {profile,postprocess}
    profile             run In-Kernel-Event-Tracing
    postprocess         post-process an existing IKET run directory to
                        generate traces
```

## Step 3: build flashinfer with the instrumentation patch

```bash
git clone https://github.com/flashinfer-ai/flashinfer && cd flashinfer
git checkout b8c21928 && git submodule update --init --recursive
git apply ../patches/iket-instrumentation.diff
pip install --no-build-isolation --no-deps -e .
python -c "import flashinfer; print(flashinfer.__version__)"
```

```
Successfully built flashinfer-python
Successfully installed flashinfer-python-0.6.18
0.6.18
```

## Step 4: smoke test on NVIDIA's pre-instrumented tutorial

Before touching flashinfer kernels, the CUTLASS tutorial GEMM that NVIDIA ships already
instrumented validates the whole pipeline on SM103.

```bash
CUDA_VISIBLE_DEVICES=7 run-iket --output-dir out/smoke --clobber profile --postprocess all -- \
    python /path/to/cutlass/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py --mnk 512,1024,64
```

```
Verifying reference result...
PASS
[run-iket] Dumped perfetto trace to /home/jessie/iket-lab/out/smoke/iket_pid_0xa61.pftrace
[run-iket] Dumped json trace to /home/jessie/iket-lab/out/smoke/iket_pid_0xa61.trace.json
[run-iket] Dumped compressed trace to /home/jessie/iket-lab/out/smoke/iket_pid_0xa61.pftrace.gz
[run-iket] Dumped perfetto HTML viewer to /home/jessie/iket-lab/out/smoke/iket_pid_0xa61.html
```

`results/smoke_summary.txt` shows 28 named ranges across the tutorial's warp roles, confirming
markers, ranges, JSON and Perfetto output all work on this machine.

---

## Step 5: memory-bound op — fused RMSNorm + FP4 quant

Six ranges go into `flashinfer/cute_dsl/rmsnorm_fp4quant.py`: `setup`, `g2s_load` (the cp.async
copy of the input row into shared memory), `sumsq_reduce` (sum of squares plus the cross-warp
reduction), `post_sync` (the barrier after it), `quant_store` (the fused multiply, quantize, pack
and store loop), and a whole-kernel range `kernel_e2e`.

Shape: 2048 tokens by hidden size 7168 in bf16. 7168 is the DeepSeek-V3/R1 hidden size; 2048
tokens is a realistic chunked-prefill batch.

```bash
CUDA_VISIBLE_DEVICES=7 FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/rmsnorm \
    --clobber profile --postprocess all -- python scripts/rmsnorm_iket_driver.py --rows 2048 --hidden 7168
```

```
[run-iket] ============= Dry run application first to collect some info... =============
ran m=2048 h=7168 dtype=bfloat16 iters=2
y_fp4 (2048, 3584) torch.float4_e2m1fn_x2, scales (2048, 448) torch.float8_e4m3fn
ref row-norm max |y|: 9.707 (sanity only)
[run-iket] Auto-computed context buffer size: 0.00 GB (2,260,992 bytes)
[run-iket] ============= Run application for real profiling... =============
[run-iket] Dumped perfetto trace to /home/jessie/iket-lab/out/rmsnorm/iket_pid_0xbdf.pftrace
```

```bash
python scripts/analyze_iket_trace.py --launch -1 out/rmsnorm/iket_pid_0xbdf.trace.json
```

```
[eager] kernel=kernel_cutlass_kernel_flashinfercute_dslrmsnorm_fp4quantRMSNormFP4Quan
  grid=(2048,1,1) block=(128,1,1) warps=8192 markers=0 wall=9.2 us
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  kernel_e2e           8192    51862.9     6330.9       6272       8320       8608
  quant_store          8192    17679.0     2158.1       2112       3008       3328
  sumsq_reduce         8192    17292.2     2110.9       1824       3584       3936
  g2s_load             8192     5994.2      731.7        640       1920       3168
  setup                8192     2428.3      296.4        192       1600       2368
  post_sync            8192      833.7      101.8         96        384        736
```

Each warp spends about 6.3 µs in the kernel, and the two big phases cost the same: 2.11 µs for the
sum of squares, 2.16 µs for quantize-and-store. The kernel reads the input row twice — once through
shared memory for the sum of squares, then again from global memory inside the quantization loop.
The second read mostly hits cache, but the fused multiply, max-search, FP4 conversion and packing
make it as expensive as the first.

## Step 6: compute-bound GEMM — NVFP4 block-scaled

The kernel is warp-specialized: 4 epilogue warps, 1 MMA warp and 1 TMA warp per CTA. Thirteen
ranges go into `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py` — per-role "main" and
per-tile ranges, `tma_acquire` (waiting for a free buffer) and `tma_issue` in the load loop,
`mma_ab_wait` and `mma_acc_acquire` in the MMA warp, and `epi_acc_wait` in the epilogue.

Shape: M=N=4096, K=7168, NVFP4 with block size 16.

```bash
CUDA_VISIBLE_DEVICES=7 FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/gemm \
    --clobber profile --postprocess all -- python scripts/gemm_iket_driver.py --mnk 4096,4096,7168
```

```
ran mnk=(4096,4096,7168) iters=2 cos_sim=0.99097
[run-iket] Auto-computed context buffer size: 0.01 GB (6,911,481 bytes)
```

`cos_sim=0.99097` is cosine similarity against a bf16 `torch.mm` reference; the driver asserts it
stays above 0.97.

```
[eager] kernel=kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm10
  grid=(2,1,74) block=(192,1,1) warps=888 markers=0 wall=39.9 us
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  kernel_e2e            888    27640.3    31126.4      29920      39040      39072
  epi_main              592    19592.3    33095.0      29216      38208      38272
  epi_tile             2048    19523.1     9532.8       9280      10880      11072
  epi_acc_wait         2048    16254.0     7936.5       7648       9216       9312
  tma_main              148     4719.9    31891.5      27872      37024      37024
  tma_tile              512     4510.3     8809.2       9088       9472       9568
  tma_acquire         14336     2887.1      201.4        192        736        800
  mma_main              148     2670.6    18044.5      27424      37088      37120
  mma_tile              512     2594.5     5067.4       8416       9536       9568
  tma_issue           14336      614.8       42.9         32        128        384
  prologue              888      293.2      330.2        320        384        384
  mma_ab_wait          7168      270.3       37.4         32        256        576
  mma_acc_acquire       256      117.0      457.1        608        704        768
```

512 output tiles run on 148 persistent CTAs with 28 load steps per tile. The epilogue warps spend
7.9 of their 9.5 µs per tile waiting on the accumulator. Meanwhile `mma_ab_wait` averages 37 ns and
`tma_acquire` 201 ns, so neither the MMA warp nor the TMA warp is stalling. At this shape the
kernel is limited purely by tensor-core throughput and the memory pipeline has headroom.

## Step 7: fused communication kernel — GEMM + two-shot AllReduce, 8 GPUs

`flashinfer/cute_dsl/gemm_allreduce_two_shot.py` runs GEMM warps plus 4 dedicated AllReduce warps
per CTA. Added ranges: `epi_ar_arrive` (the epilogue's per-tile multicast "data ready" signal),
`ar_flag_wait` (spinning until all 8 ranks signal a tile), `ar_bar_sync` (the barrier between AR
warps), `ar_ldst` (the multimem ld_reduce + multicast store sweep), and `ar_final_bar`.

Shape: M=N=2048, K=4096, 8 ranks, bf16 A/B and output with f32 accumulation — the real
tensor-parallel serving dtype. The upstream test uses tf32/f32; only the dtypes differ here, and
the bf16 output exercises the `multimem_ld_reduce_8xbf16` path.

```bash
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/comm_iket_driver.py
```

```
mnkl: (2048, 2048, 4096, 1)
AB dtype: BFloat16, C dtype: BFloat16, Acc dtype: Float32
Mma Tiler (M, N): (128, 128), Cluster Shape (M, N): (1, 1)
Fused AllReduce Op: two_shot
rank N: gemm+two_shot allreduce OK    (16 lines: 8 ranks x 2 passes, reference check on each)
```

`run-iket` writes one trace per rank process.

```bash
python scripts/comm_skew_table.py out/comm/iket_pid_*.trace.json
```

```
pid        wall_us epi_arrive  bar_sync     ldst flag_wait final_bar  (mean us, last launch)
0x1394       552.2       88.9      49.6     60.0      1.13       3.5
0x1395       384.6        5.5      30.3     60.2      1.11       3.5
0x1396       187.8       77.2       7.8     58.2      1.10       3.8
0x1397       351.4        5.3      26.4     59.9      1.15       3.7
0x1398       240.6      103.3      12.6     65.3      1.20       3.8
0x1399       490.8       53.6      42.5     60.4      1.09       3.7
0x139a       438.5       23.9      36.5     60.6      1.15       3.3
0x139b       785.6       85.6      76.7     59.8      1.09       3.6
```

Two things stand out. The actual communication work (`ldst`) takes 58–65 µs on every rank, so the
work is balanced. Total time inside the kernel is not: it runs from 188 to 786 µs, and almost all
of that variation sits in `epi_arrive` and `bar_sync`. Ranks that start early idle in those waits
until the slowest rank's tiles arrive. From outside, every rank's kernel simply looks long. One
caveat: each rank's trace has its own timebase, so durations compare across ranks but absolute
start times do not.

## Step 8: LL vs BT vs HT AllReduce protocols

`flashinfer/comm/mnnvl_cutedsl/` implements fused AllReduce + residual-add + RMSNorm three times,
with three protocols: LL (low latency, Lamport-flag based), BT (balanced) and HT (high
throughput). The public router picks one from the token count. Each was forced with its
`*_ONLY_CONFIG` and compared head-to-head.

The LL Lamport kernel got full phase ranges (`ll_preload`, `ll_pdl_wait`, `ll_spin_reduce`,
`ll_sentinel_clear`, `ll_rms_reduce`, `ll_norm_store`, `ll_lamport_e2e`). BT got `bt_spin_reduce`
around its spin loop; every other kernel in the three pipelines got a marker so IKET records its
warp lifetimes.

The upstream presets are tuned for one static configuration — hidden size 8192, TP 8, bf16, on
GB300-class hardware, which is exactly this box. So the realistic axis is token count:
m = 8, 32 (decode), 256, 1024 (mixed), 4096, 8192 (prefill chunks). These bracket the router's own
crossovers, which switch LL to BT at about m=52 and BT to HT at m=1024.

**The Latin square.** For each m the protocols run in rotated orders — (LL,BT,HT), (BT,HT,LL),
(HT,LL,BT) — so every protocol appears once in every position, cancelling order effects such as
clock ramp-up and cache state. Above m=1024, two balanced 2x2 rounds of (LL,HT)/(HT,LL) are used.
One warmup per (m, protocol) runs before each square and is excluded from statistics, and every
warmup is checked numerically against a torch reference.

```bash
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/mnnvl --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/mnnvl_protocols_iket_driver.py
```

```
MANIFEST exec=0 m=8 phase=warmup pos=- proto=ll norm_cos=0.999997 prenorm_max_diff=0.0120
MANIFEST exec=1 m=8 phase=warmup pos=- proto=bt norm_cos=0.999997 prenorm_max_diff=0.0120
MANIFEST exec=2 m=8 phase=warmup pos=- proto=ht norm_cos=0.999997 prenorm_max_diff=0.0149
MANIFEST exec=3 m=8 phase=latin order=0 pos=0 proto=ll
done: 68 executions across m=[8, 32, 256, 1024, 4096, 8192]
```

```
== span per (m, protocol), latin executions pooled across ranks ==
     m  proto  reps   mean_us   min_us   max_us
     8     bt     3     144.4    120.4    163.8
     8     ht     3      20.2     18.2     21.3
     8     ll     3      96.8     70.0    113.8
    32     bt     3     170.8    154.0    186.0
    32     ht     3      20.2     16.3     27.6
    32     ll     3      76.1     71.9     78.6
   256     bt     3     255.0    146.7    457.1
   256     ht     3      61.9     33.5     81.6
   256     ll     3     103.1     95.3    109.0
  1024     bt     3     628.5    189.5   1258.2
  1024     ht     3     144.3     61.3    302.6
  1024     ll     3     300.7    291.2    315.5
  4096     ht     4     187.6    174.5    218.1
  4096     ll     4     985.8    837.4   1041.0
  8192     ht     4     344.9    318.6    374.7
  8192     ll     4    1740.9   1247.3   2030.8

== LL lamport kernel phase means (ns per warp) ==
     m    lamport_e2e     norm_store       pdl_wait        preload     rms_reduce sentinel_clear    spin_reduce
     8           2729             38              8            266            801            160           1365
    32           3012             40             10            283           1078            160           1325
   256           6126             56             16            581           2608            193           2403
  1024           6834             53             21            999           2660            147           2743
  4096           9228             53             17           1079           2815            136           4932
```

On this machine — one node, 8x B300, NVLS multicast — HT wins at every token count tested,
including the smallest. At m=8 HT finishes in about 20 µs against LL's 97 and BT's 144. At m=8192
HT is about 5x faster than LL, 345 against 1741. That contradicts the shipped router, which sends
token counts up to about 50 to LL.

Two honest caveats. These presets were tuned for multi-node GB300 NVL72 systems, where LL's
Lamport design fights real fabric latency; a single node is not that environment. And the LL kernel
carries about 14 IKET events per warp against HT's 1, so LL pays more instrumentation overhead —
sub-microsecond per warp, which cannot explain a 4–9x gap, but it is not nothing.

The phase table gives the mechanism: LL launches one CTA row per token, so its grid scales with m
and its `spin_reduce` and `rms_reduce` phases grow with it, while HT keeps a fixed persistent grid.
The Latin-square position table shows no consistent first-position penalty, so the ranking is not
an ordering artifact. BT's spread — `bt_spin_reduce` p-max hits 247 µs at m=1024 — is real variance
in its spin phase, not measurement noise.

## Step 9: MegaMoE at DeepSeek-V4-Pro shapes

A megakernel is the case where every classical profiler is blind by construction: expert-parallel
dispatch over NVLink, the grouped FC1 GEMM, SwiGLU, FC2 and the cross-rank combine all execute
inside one persistent kernel launch, so a CUPTI timeline shows one long rectangle per layer.

FlashInfer's `moe_ep` "mega" path ships two CuTe-DSL megakernels, NVFP4 and MXFP8, from the NVIDIA
kernel team (PR #3852/#3980/#4079). And here is the detail that makes the case better than any
argument: **the kernel team ships these megakernels with IKET spans already in the device code** —
`Dispatch_Prep`, `Dispatch_Barrier`, `Pull.TMA_NVLink_Roundtrip`, `Pull.Arrival_Atomic`,
`tma_weight_fc1`, `tma_token_fc1_wait`, `mma_fc1`, `fc1_epi`, `mma_fc2`, `token_back`,
`Kernel_Tail` and more — behind a no-op compatibility shim (`src/src/iket_compat.py`). With
`nvidia-cutlass-dsl` 4.7.0 installed, the real IKET dialect loads and every span lights up under
`run-iket`. In-kernel tracing is not a research toy. It is how the people who write megakernels
debug them.

**Shapes.** From DeepSeek-V4-Pro's `config.json`: hidden_size 7168, moe_intermediate_size 3072,
384 routed experts, top-k 6, `expert_dtype = fp4` — which makes the NVFP4 CuTeDSL megamoe this
model's literal serving path. EP4 is the upstream-validated deployment geometry; the hillclimb
below runs at world_size 8 (EP8, 48 local experts per rank).

The CuTe-DSL 4.7.0 staging fix from open flashinfer PR #4449 is applied first — a verbatim re-sync
of `src/src/inputs_process.py`, which reworks the fused bf16→quant staging that could misalign
under 4.7.0.

**Correctness first.** `scripts/megamoe_iket_driver.py` reuses the upstream multirank test helpers
and checks the kernel against a pure-torch oracle over the global 384-expert set:

```
ORACLE rank=0 rel_l2=0.002621 max|d|=512
ORACLE rank=1 rel_l2=0.002627 max|d|=512
...
ORACLE rank=7 rel_l2=0.002621 max|d|=512
```

`rel_l2` of 0.0026 is NVFP4 round-to-nearest noise; with random unscaled weights `max|d|=512` is
about 0.3% of the output range, matching upstream's own oracle bands.

### What the trace shows at decode

m=128 tokens/rank, EP4, one rank's steady-state launch. Full output in
`results/megamoe_m128_summary.txt`:

```
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  tma_token_fc1        4608    54728.9    11876.9      11456      24832      25728
  tma_weight_fc1       4608    54711.4    11873.1      11424      24672      25568
  mma_fc1              4608    27641.3     5998.5       7488      26848      29344
  tma_token_fc2        5376    25328.5     4711.4       4672       7552      15872
  tma_weight_fc2       5376    25289.5     4704.2       4672       7488      15840
  fc1_epi             18432    22183.7     1203.5       1120       2528       3488
  produce_tile_id     10132    21821.8     2153.7       1728      14368      14784
  mma_fc2              5376    12740.1     2369.8       1984       6688      14880
  fc2_epi             21504     9533.7      443.3        384       1408       3456
  tma_token_fc1_wait      592     2471.8     4175.3        672      15520      15648
  Dispatch_Barrier        1        8.8     8832.0       8832       8832       8832
```

At decode batch — 128 tokens x top-6 across 384 experts, roughly 16 tokens per expert once all
ranks are counted — the FC1 weight loads run at twice the MMA time, 11.9 µs against 6.0 µs per
task. The kernel is weight-bandwidth bound, exactly as MoE decode theory predicts. The cross-rank
dispatch machinery is nearly free at this size: `Dispatch_Barrier` is microseconds and the `Pull.*`
spans are tiny.

That diagnosis directed the whole hillclimb. At decode, no combine wire or scheduling knob can
create bandwidth, but tile shape can trim wasted work. At prefill, the growing term is combine
traffic, which the combine wire dtype controls directly.

### The hillclimb, under a bitwise-exactness rule

Every candidate claiming a speedup had to reproduce the baseline's output bit for bit
(`torch.equal`, per rank, same seeds — the `--y-ref` mechanism in the driver). Numerics-changing
configurations are reported separately and labeled. Full table in
`results/megamoe_ws8_hillclimb.txt`; summary at world_size 8:

| tokens/rank | best bit-exact config | median | vs baseline | numerics-trading best | median | vs baseline |
|---|---|---|---|---|---|---|
| 128 (decode) | tile (256,**64**,256) + epi_flag (1,2) | 0.3369 ms | **−1.8%** | — (nvfp4 combine is slower here) | — | — |
| 2048 (prefill) | baseline knobs already optimal | 0.5992 ms | — | combine=nvfp4 | 0.5295 ms | **−11.6%** |
| 4096 (prefill) | token_back=standalone | 0.9298 ms | −0.8% | combine=nvfp4 | 0.8161 ms | **−12.9%** |

The decode win came straight from the IKET diagnosis. With ~16 tokens per expert the default
(256,**128**,256) tile pads the token dimension almost 8x; narrowing the token tile to 64 trims
epilogue and combine work without touching weight traffic, and the output stays bit-identical
(`max|d|=0` on every rank). Twelve other knob axes came back flat or negative, which is itself
worth knowing — the GB200-derived heuristic is already close to optimal on B300 at decode.

### Head to head against DeepGEMM megamoe

DeepSeek's own kernel, `deep_gemm` 2.6.1, fp8-activation x fp4-weight wire. A different
quantization wire, so **not** bit-comparable with the CuTeDSL kernel; both are real DSV4 serving
configurations, both run their own upstream heuristics, same CUPTI benchmark, same machine, same
geometry.

```
tokens/rank    cutedsl best         deepgemm     verdict
   128         0.3369 ms (bitexact) 0.3217 ms    deepgemm +4.5% faster
  2048         0.5295 ms (nvfp4)    0.7506 ms    cutedsl 29.5% faster
  4096         0.8161 ms (nvfp4)    1.3620 ms    cutedsl 40.1% faster
```

**A fairness note.** The first CuTeDSL closure included a device-to-device copy of the output into
a caller-owned tensor, while DeepGEMM writes its output tensor directly. The shim already ships the
fix — `nvfp4_mega_launch_thunk`, a prebuilt bare-kernel launcher with no per-call Python and no
output copy, the same workspace-view semantics as flashinfer PR #4341. Final numbers use that
copy-free closure, verified bit-exact against the same reference:

```
tokens/rank  cutedsl (copy-free closure, view)      deepgemm (same session)   verdict
   128       0.3340 ms  (winner knobs, BITEXACT)    0.3216 ms                 deepgemm +3.9%
  2048       0.5231 ms  (combine=nvfp4)             0.7506 ms                 cutedsl 30.3% faster
  4096       0.7987 ms  (combine=nvfp4)             1.3709 ms                 cutedsl 41.7% faster
```

One timing disclosure applies to every benchmark number in this repository: `cupti-python` is not
installed in this container, so `bench_gpu_time` fell back to CUDA-event timing. Every cell used
the identical method, 30 measured iterations, medians — so A/B and cross-kernel ratios stand, but
absolute microseconds carry CUDA-event granularity.

### The bistable degraded mode

Chasing three "pathological" readings produced the most instructive finding of the campaign.
`in_kernel_fc2_reduce`, the mxfp8 wire and `num_sched_stages=4` all benched at a suspiciously
identical ~8.9 ms — 26x slower than baseline. An IKET trace of the `ikr` config told a different
story: a clean 361 µs launch, with the cross-rank REDG combine costing only ~+2 µs per task inside
`fc2_epi`.

Re-benching with a `dist.barrier` between iterations — serving-like pacing — returned all three
configs to ~0.34–0.36 ms, and the 9 ms mode instead captured the *default* config on that occasion.
The megamoe kernel has a bistable degraded mode at ~9 ms/launch, with traces showing
`Dispatch_Barrier` absorbing ~4.7 ms per launch, and launch pacing can trigger it on any config.
That is a kernel-robustness issue worth reporting upstream, with trace evidence in this repository.

Without IKET, "ikr is 26x slower on B300" would have shipped as a fact.

Final paced decode table, median of per-rank medians: cutedsl winner-knobs ~0.340 (bit-exact),
ikr+winner ~0.342, mxfp8 ~0.363, DeepGEMM ~0.324 at m=128; at m=64, ~0.335 against ~0.321.
DeepGEMM keeps ~4±1% at decode under every harness style, with every tuning, integration and
config axis exhausted. The residual is combine/dispatch architecture. The remaining kernel-work
items are a ring-coupled combine that avoids sys-scope NVLink signaling, and a fix for the bistable
mode — both IKET-instrumented projects now, not guesses.

### Before and after of the winning prefill config

The bench numbers say the NVFP4 combine wire is faster at prefill. The traces say why. Same rank,
same steady-state launch, m=2048:

```
span                      bf16 combine        nvfp4 combine       change
token_back  (mean ns)     6134                2025                -67%  <- 4x smaller wire
fc2_epi     (mean ns)     2095                2578                +23%  <- encoder moved here
Dispatch_Pull (total us)  180.3               172.5               ~flat
kernel wall (us)          610.5               577.2               -5.5% (instrumented launch)
```

The cross-rank push-back span collapses by 3x when the wire shrinks from bf16 to NVFP4, and part of
the saving is paid back inside `fc2_epi`, which now runs the NVFP4 encoder. That is the causal
chain behind the −11.6% bench delta, read directly off the warps.

One gotcha captured on the way: the nvfp4 IKET run segfaulted on first attempt, because the sizing
pass underestimates this config's per-warp event count. `--max-ts-cnt-per-warp 8192` fixes it,
matching the buffer-sizing caveat in the IKET guide.

## Step 10: what NCU can and cannot see

This machine restricts GPU performance counters to admin (`RmProfilingAdminOnly: 1`), so ncu
(2026.1.0) ran as root with extended privileges inside the container. The result is the strongest
argument for in-kernel tracing in this repository.

1. **Multi-pass collection deadlocks on this kernel class.** NCU's kernel-replay model re-executes
   a kernel to collect more counters than fit in one pass. The megamoe kernel contains cross-rank
   NVLink barriers, so a replayed kernel waits for peer ranks that are not re-running, and the
   profile hangs forever. Raw log (`logs/megamoe_ncu_m128.log`): eight `==PROF== Connected` lines,
   GPUs allocated, zero progress until it was killed.

2. **Even a single-metric, single-pass collection stalled.** Retrying with nothing but
   `--metrics gpu__time_duration.sum` and `--launch-count 1` under a 600-second guard: NCU
   serializes the profiled launch while it arms collection, the peer ranks run ahead and then spin
   inside their own kernels waiting for the profiled rank at the NVLink barrier, and no rank ever
   completes. The run ended with `==PROF== Trying to shutdown target application` when the guard
   fired (`logs/megamoe_ncu_minimal.log`). With sudo, root and a privileged container, the best NCU
   delivered was nothing.

Even in the best case, NCU's unit of observation is the whole kernel: one row of aggregate counters
for a launch that internally contains dispatch, two grouped GEMMs, an activation and a cross-rank
combine. IKET's unit of observation is a named phase on a warp. For megakernels that difference is
not a convenience — it is the difference between a profiler that works and one that structurally
cannot.

## Step 11: reading the traces

Every directory in `traces/` contains one `.pftrace.gz` per process. Open
https://ui.perfetto.dev/ and load the file directly; Perfetto reads `.gz`. Tracks are grouped by
GPU location (SM, CTA, warp), and the named ranges from the patch appear per warp. W/A/S/D zoom and
pan. The `.trace.json.gz` files hold the same events for scripted analysis, and
`scripts/analyze_iket_trace.py` shows how to read them.

---

## Pitfalls

1. **A driver flag named `--h` silently kills the run.** CuTe DSL runs its own
   `argparse.parse_known_args()` when it compiles a kernel, so an application flag such as
   `--h 4096` abbreviation-matches argparse's built-in `--help`. The DSL's parser prints help and
   exits the process before any kernel runs. The trace comes out empty with no error.

2. **FlashInfer's CuTe-DSL disk cache silently defeats IKET.** FlashInfer caches compiled kernels
   as `.o` files and reloads them on later runs. A reloaded kernel never JIT-compiles inside the
   profiled process, so `run-iket` cannot instrument it and the trace comes out empty. Always set
   `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` when profiling flashinfer.

3. **`mm_fp4(backend="cute-dsl")` on SM103 does not run the SM103 kernel file.** It runs
   `Sm100BlockScaledPersistentDenseGemmKernel` from `dense_blockscaled_gemm_sm100.py`, even on
   SM103 hardware. Instrumenting `dense_blockscaled_gemm_sm103.py` first produced an empty trace.
   Check the kernel name inside the trace before trusting instrumentation placement.

4. **`BT_ONLY_CONFIG` cannot serve more than 1024 tokens.** Building its workspace with a larger
   capacity raises `ValueError: Finalize routes do not cover the requested workspace capacity`.
   This is an upstream routing bound, not a bug in the experiment.

5. **IKET cannot run together with CUPTI tools.** Do not combine `run-iket` with Nsight Systems,
   Nsight Compute, or `cupti-python` benchmarking in the same run.

6. **Multi-process notes.** `torchrun` works out of the box (the injection reaches child
   processes), each rank gets its own trace file, the whole job runs twice — a buffer-sizing pass
   then a collection pass — and rank traces do not share a timeline.

7. **The 4.7.0 wheel does not ship `--enabled-cluster`.** The documentation describes it, but
   `run-iket profile --help` in 4.7.0 has no such flag. Full-grid dumps are the only option, and
   postprocessing time scales with total warp-event count — the m=8192 AllReduce cells took hours.

8. **Megamoe knobs plumb through the workspace, not the kernel config.** Passing `knobs=` to
   `Nvfp4CutedslMegaMoeConfig` and launching through the shim silently does nothing; the shim reads
   knobs from `get_symm_buffer_for_mega_moe(knobs=...)`. The first sweep produced bit-identical
   medians for every "candidate" before this surfaced.

9. **NCU cannot kernel-replay a cross-rank-coupled kernel.** A replayed kernel spins on peer
   barriers forever. Single-pass metric sets or application replay are the only options, and
   application replay does not compose with torchrun rendezvous.

## Layout

| path | contents |
|---|---|
| `patches/iket-instrumentation.diff` | every kernel edit, applies onto flashinfer `b8c21928` |
| `scripts/` | four drivers plus two analysis scripts |
| `logs/` | full raw stdout/stderr of every profiled run |
| `traces/` | Perfetto (`.pftrace.gz`) and JSON traces per experiment |
| `results/` | raw analyzer output quoted above |
| `COMM_KERNEL_SURVEY.md` | flashinfer's communication kernels, classified by IKET-ability |

IKET is experimental; its API and output format may change. See the CUTLASS documentation page
[IKET Profiling (In-Kernel Event Tracing)](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html).
