"""Candidate-list generator for the megamoe brute-force sweep (EP8, decode).

Stages (see repo/results/brute_force_sweep.csv for the ledger):
  stage1: correctness-combo factorial at m=128, default perf knobs.
          5 tiles x 3 token_back x 2 load_balance x 2 non_ubulk x 2 tail_fused
          = 120 cells; invalid cells (mirroring tuner.is_valid for the fixed
          bf16/no-ikr/cluster-(2,1,1) subspace) are marked invalid and
          ledgered without GPU time.
  stage2: perf grid on stage-1 winners read from the CSV.
          #1: group_hint{64,128,256,512} x flag_batch{1,2,4,8,16} x all 13
          epi_flag_batch = 260; #2-#5: 2-level screening subset = 16 each.
  stage3: top-8 overall -> {m64,m128} x {unpaced,paced} x 3 reps.
          The _rep knob key is a ledger-key disambiguator; the worker strips
          "_"-prefixed keys before apply_knobs.

Fixed axes (measured dead/required, do not retest):
  cluster_shape_mnk=(2,1,1)   (1,1,1) fails compile; (4,1,1) HANGS
  in_kernel_fc2_reduce=False  bitexact track
  combine=bf16                quantized wires are a numerics tradeoff
  group_hint=None excluded    (-18% measured)

Note: (128,64,256) is in the mission's stage-1 tile list but NOT in
tuner.CORRECTNESS_KNOBS["mma_tiler_mnk"]; kept runnable on purpose -- a
ctor reject lands as a FAIL row, which is informative coverage.
"""

import argparse
import csv
import itertools
import json
import os

LAB = "/home/jessie/iket-lab"
CSV_PATH = f"{LAB}/repo/results/brute_force_sweep.csv"
OUT_DIR = f"{LAB}/out/bf"

TILES = [
    (256, 64, 256),
    (256, 128, 256),
    (128, 64, 256),
    (128, 128, 256),
    (256, 256, 256),
]
TOKEN_BACK = ["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
LOAD_BALANCE = ["atomic_counter", "static"]
NON_UBULK = [True, False]
TAIL_FUSED = [True, False]

# m=128 heuristic defaults (tuner._SMALL_TOKEN_KNOBS) = "default perf knobs"
DEFAULT_PERF = dict(group_hint=512, flag_batch=4, epi_flag_batch=(1, 2))
FIXED = dict(cluster_shape_mnk=(2, 1, 1), in_kernel_fc2_reduce=False)

GROUP_HINTS = [64, 128, 256, 512]
FLAG_BATCHES = [1, 2, 4, 8, 16]
EPI_FLAG_BATCHES = [
    (1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 2),
    (4, 4), (8, 2), (8, 4), (2, 8), (4, 8), (16, 16),
]
SCREEN_GH = [128, 512]
SCREEN_FB = [2, 8]
SCREEN_EFB = [(1, 2), (2, 4), (4, 4), (8, 4)]


def check_invalid(tile, token_back, non_ubulk, tail_fused):
    """Mirror of tuner.is_valid for the fixed subspace; returns reason or None."""
    if tail_fused and token_back != "epi_warps":
        return "tail_fused_reduce requires epi_warps token-back"
    if token_back != "epi_warps" and not non_ubulk:
        return "dispatch-warp token-back requires non-UBLK fc2 store"
    # tile M==256 needs even cluster M -- cluster fixed (2,1,1): satisfied.
    return None


def jsonable(knobs):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}


def gen_stage1():
    cands = []
    i = 0
    for tile, tb, lb, nub, tfr in itertools.product(
        TILES, TOKEN_BACK, LOAD_BALANCE, NON_UBULK, TAIL_FUSED
    ):
        knobs = dict(
            FIXED,
            **DEFAULT_PERF,
            mma_tiler_mnk=tile,
            token_back_mode=tb,
            load_balance_mode=lb,
            non_ubulk_fc2_store=nub,
            tail_fused_reduce=tfr,
        )
        reason = check_invalid(tile, tb, nub, tfr)
        c = dict(
            config_id=f"s1-{i:03d}",
            stage=1,
            tokens=128,
            closure="view",
            paced=False,
            knobs=jsonable(knobs),
        )
        if reason:
            c["invalid"] = True
            c["invalid_reason"] = reason
        cands.append(c)
        i += 1
    return cands


def load_ok_rows(stages, tokens=128):
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if (
                row["status"] == "OK"
                and row["bitexact"] == "true"
                and row["stage"] in {str(s) for s in stages}
                and row["tokens"] == str(tokens)
                and row["paced"] == "false"
                and row["median_ms"]
            ):
                rows.append(row)
    # dedupe by knobs_json keeping best median
    best = {}
    for r in rows:
        k = r["knobs_json"]
        if k not in best or float(r["median_ms"]) < float(best[k]["median_ms"]):
            best[k] = r
    return sorted(best.values(), key=lambda r: float(r["median_ms"]))


CORRECTNESS_KEYS = [
    "mma_tiler_mnk",
    "cluster_shape_mnk",
    "token_back_mode",
    "load_balance_mode",
    "non_ubulk_fc2_store",
    "tail_fused_reduce",
    "in_kernel_fc2_reduce",
]


def gen_stage2():
    winners = load_ok_rows([1])[:5]
    if not winners:
        raise SystemExit("no bitexact OK stage-1 rows in the CSV yet")
    cands = []
    for w_idx, w in enumerate(winners):
        wk = json.loads(w["knobs_json"])
        base = {k: wk[k] for k in CORRECTNESS_KEYS if k in wk}
        if w_idx == 0:
            grid = itertools.product(GROUP_HINTS, FLAG_BATCHES, EPI_FLAG_BATCHES)
        else:
            grid = itertools.product(SCREEN_GH, SCREEN_FB, SCREEN_EFB)
        for j, (gh, fb, efb) in enumerate(grid):
            knobs = dict(base, group_hint=gh, flag_batch=fb, epi_flag_batch=efb)
            cands.append(
                dict(
                    config_id=f"s2-w{w_idx}-{j:03d}",
                    stage=2,
                    tokens=128,
                    closure="view",
                    paced=False,
                    knobs=jsonable(knobs),
                )
            )
    print(f"stage2: winners from stage1 (best first):")
    for w in winners:
        print(f"  {w['median_ms']} ms  {w['knobs_json']}")
    return cands


def gen_stage3():
    finalists = load_ok_rows([1, 2])[:8]
    if not finalists:
        raise SystemExit("no bitexact OK stage-1/2 rows in the CSV yet")
    cands = []
    for f_idx, w in enumerate(finalists):
        wk = json.loads(w["knobs_json"])
        wk.pop("_rep", None)
        for tokens in (64, 128):
            for paced in (False, True):
                for rep in (1, 2, 3):
                    knobs = dict(wk, _rep=rep)
                    cands.append(
                        dict(
                            config_id=f"s3-f{f_idx}-m{tokens}-"
                            f"{'p' if paced else 'u'}-r{rep}",
                            stage=3,
                            tokens=tokens,
                            closure="view",
                            paced=paced,
                            knobs=knobs,
                        )
                    )
    print("stage3 finalists (best first):")
    for w in finalists:
        print(f"  {w['median_ms']} ms  stage={w['stage']}  {w['knobs_json']}")
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "stage3"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cands = {"stage1": gen_stage1, "stage2": gen_stage2, "stage3": gen_stage3}[
        args.stage
    ]()
    out = args.out or f"{OUT_DIR}/{args.stage}_candidates.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(cands, f, indent=1)
    n_invalid = sum(1 for c in cands if c.get("invalid"))
    print(f"{args.stage}: {len(cands)} candidates ({n_invalid} invalid) -> {out}")


if __name__ == "__main__":
    main()
