"""Host orchestrator for the megamoe brute-force knob sweep (EP8 only).

Reads a candidate JSON list, drops candidates already ledgered in the CSV,
groups the rest into batches, and for each batch:
  1. GPU-coexistence gate: refuse to launch while ANY other torchrun is
     alive or any foreign pid holds >10GB on a GPU; re-check every 10 min.
     CPU-side work (candidate generation, CSV upkeep) is always allowed.
  2. Launches scripts/megamoe_bf_worker.py under torchrun --nproc-per-node 8
     inside the flashinfer-iket-dev container (docker exec).
  3. Watchdog: the worker heartbeats {config_id, phase, phase_start,
     budget_s}; a phase overstaying its budget (compile/bench: 420s) means a
     hang -> kill the worker tree in the container, ledger the in-flight
     config as HANG, and continue with the remaining candidates in a fresh
     session.
  4. Appends worker JSONL rows to repo/results/brute_force_sweep.csv with
     the kernel commit hash captured at batch launch.

RESUMABLE: candidates whose (stage, tokens, knobs_json, closure, paced) key
already has a CSV row with status OK/INVALID/COMPILED-terminal are skipped;
FAIL rows are skipped too unless --retry-failed; HANG unless --retry-hangs.

INVALID candidates (marked by the generator, mirroring tuner.is_valid) are
ledgered directly without GPU time.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

LAB = "/home/jessie/iket-lab"
CONTAINER = "flashinfer-iket-dev"
FLASHINFER_DIR = f"{LAB}/flashinfer"
CSV_PATH = f"{LAB}/repo/results/brute_force_sweep.csv"
BF_DIR = f"{LAB}/out/bf"
DSL_CACHE_DIR = f"{LAB}/out/bf/dslcache"

CSV_COLUMNS = [
    "stage",
    "tokens",
    "knobs_json",
    "closure",
    "paced",
    "median_ms",
    "bitexact",
    "status",
    "kernel_commit",
    "timestamp",
    "config_id",
    "median_max_ms",
    "bitexact_view",
    "maxd",
    "compile_s",
    "error",
]

SKIP_STATUSES_ALWAYS = {"OK", "INVALID"}


def knobs_json(knobs: dict) -> str:
    return json.dumps(knobs, sort_keys=True, separators=(",", ":"))


def row_key(stage, tokens, kjson, closure, paced):
    return (str(stage), str(tokens), kjson, closure, str(bool(paced)).lower())


def load_done(csv_path, retry_failed, retry_hangs, compile_only=False):
    done = {}
    if not os.path.exists(csv_path):
        return done
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            status = row["status"]
            skip = status in SKIP_STATUSES_ALWAYS
            if status == "COMPILED":
                # compile-only rows satisfy only another compile-only pass
                skip = compile_only
            if status == "FAIL" or status == "FAIL_CUDA":
                skip = not retry_failed
            if status == "HANG":
                skip = not retry_hangs
            if skip:
                done[
                    row_key(
                        row["stage"],
                        row["tokens"],
                        row["knobs_json"],
                        row["closure"],
                        row["paced"],
                    )
                ] = status
    return done


def append_rows(csv_path, rows):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def kernel_commit() -> str:
    return subprocess.run(
        ["git", "-C", FLASHINFER_DIR, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def gpu_busy() -> str:
    """Non-empty reason string when the box is not ours to bench on."""
    out = subprocess.run(
        ["pgrep", "-af", "torchru[n]"], capture_output=True, text=True
    ).stdout.strip()
    if out:
        return f"other torchrun alive: {out.splitlines()[0][:120]}"
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    for line in smi.splitlines():
        if not line.strip():
            continue
        pid_s, mem_s = [x.strip() for x in line.split(",")[:2]]
        try:
            mem = int(mem_s)
        except ValueError:
            continue
        if mem > 10240:
            return f"pid {pid_s} holds {mem} MiB"
    return ""


def wait_for_gpus(poll_s=600):
    while True:
        reason = gpu_busy()
        if not reason:
            return
        print(f"[gate] GPUs busy ({reason}); recheck in {poll_s}s", flush=True)
        time.sleep(poll_s)


def jsonl_to_csv_row(jr, commit):
    knobs = jr.get("knobs") or {}
    bit_copy = jr.get("bitexact_copy")
    bit_view = jr.get("bitexact_view")
    bitexact = ""
    if bit_copy is not None or bit_view is not None:
        bitexact = str(bool(bit_copy) and bool(bit_view)).lower()
    maxd_vals = [v for v in (jr.get("maxd_copy"), jr.get("maxd_view")) if v is not None]
    return {
        "stage": jr.get("stage"),
        "tokens": jr.get("tokens"),
        "knobs_json": knobs_json(knobs),
        "closure": jr.get("closure", "view"),
        "paced": str(bool(jr.get("paced", False))).lower(),
        "median_ms": jr.get("median_ms", ""),
        "bitexact": bitexact,
        "status": jr.get("status"),
        "kernel_commit": commit,
        "timestamp": jr.get("ts", time.strftime("%Y-%m-%dT%H:%M:%S")),
        "config_id": jr.get("config_id"),
        "median_max_ms": jr.get("median_max_ms", ""),
        "bitexact_view": "" if bit_view is None else str(bool(bit_view)).lower(),
        "maxd": max(maxd_vals) if maxd_vals else "",
        "compile_s": jr.get("compile_s", ""),
        "error": jr.get("error", ""),
    }


def kill_worker():
    """Kill the worker tree inside the container (bracketed pattern)."""
    subprocess.run(
        ["docker", "exec", CONTAINER, "pkill", "-TERM", "-f", "megamoe_bf_worke[r]"],
        capture_output=True,
    )
    time.sleep(10)
    subprocess.run(
        ["docker", "exec", CONTAINER, "pkill", "-KILL", "-f", "megamoe_bf_worke[r]"],
        capture_output=True,
    )


def run_batch(cands, tokens, yref, tag, dsl_cache, repeat_iters, compile_only):
    """Run one torchrun session; returns (jsonl rows, hang_config_id|None)."""
    os.makedirs(BF_DIR, exist_ok=True)
    batch_path = f"{BF_DIR}/batch_{tag}.json"
    results_path = f"{BF_DIR}/results_{tag}.jsonl"
    hb_path = f"{BF_DIR}/hb_{tag}.json"
    log_path = f"{BF_DIR}/log_{tag}.txt"
    for p in (results_path, hb_path):
        if os.path.exists(p):
            os.remove(p)
    with open(batch_path, "w") as f:
        json.dump(cands, f, indent=1)

    inner = (
        "export TORCH_NATIVE_SKIP_VERSION_CHECK=1 && "
        f"export CUTE_DSL_CACHE_DIR={DSL_CACHE_DIR} && "
        "torchrun --nproc-per-node 8 scripts/megamoe_bf_worker.py "
        f"--batch-json {batch_path} --tokens {tokens} "
        f"--results-jsonl {results_path} --heartbeat {hb_path} "
        f"--repeat-iters {repeat_iters}"
    )
    if yref:
        inner += f" --y-ref {yref}"
    if dsl_cache:
        inner += " --dsl-file-cache"
    if compile_only:
        inner += " --compile-only"

    logf = open(log_path, "w")
    proc = subprocess.Popen(
        ["docker", "exec", "-w", LAB, CONTAINER, "bash", "-c", inner],
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    hang_cid = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        hb = None
        try:
            with open(hb_path) as f:
                hb = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        if hb is None:
            # no heartbeat yet: bootstrap/import window, give it 600s from launch
            if time.time() - os.path.getmtime(batch_path) > 600:
                hang_cid = "(bootstrap)"
        else:
            if time.time() - hb["phase_start"] > hb["budget_s"]:
                hang_cid = hb["config_id"]
        if hang_cid is not None:
            print(
                f"[watchdog] {tag}: phase overstay (config {hang_cid}); killing",
                flush=True,
            )
            kill_worker()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(15)
    logf.close()

    rows = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows, hang_cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidate JSON list")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--dsl-file-cache", action="store_true")
    ap.add_argument("--repeat-iters", type=int, default=30)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--retry-hangs", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    with open(args.candidates) as f:
        cands = json.load(f)

    done = load_done(
        args.csv, args.retry_failed, args.retry_hangs, args.compile_only
    )
    commit = kernel_commit()

    # Ledger INVALID candidates directly (no GPU).
    invalid_rows = []
    runnable = []
    for c in cands:
        key = row_key(
            c["stage"],
            c["tokens"],
            knobs_json(c["knobs"]),
            c.get("closure", "view"),
            c.get("paced", False),
        )
        if key in done:
            continue
        if c.get("invalid"):
            invalid_rows.append(
                {
                    "stage": c["stage"],
                    "tokens": c["tokens"],
                    "knobs_json": knobs_json(c["knobs"]),
                    "closure": c.get("closure", "view"),
                    "paced": str(bool(c.get("paced", False))).lower(),
                    "median_ms": "",
                    "bitexact": "",
                    "status": "INVALID",
                    "kernel_commit": commit,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "config_id": c["config_id"],
                    "median_max_ms": "",
                    "bitexact_view": "",
                    "maxd": "",
                    "compile_s": "",
                    "error": c.get("invalid_reason", "tuner.is_valid reject"),
                }
            )
        else:
            runnable.append(c)
    if invalid_rows:
        append_rows(args.csv, invalid_rows)
        print(f"[ledger] wrote {len(invalid_rows)} INVALID rows", flush=True)

    # Group runnable candidates by (tokens, paced, yref) into batches.
    groups = {}
    for c in runnable:
        groups.setdefault((c["tokens"], bool(c.get("paced", False))), []).append(c)

    n_batches = 0
    for (tokens, paced), group in sorted(groups.items()):
        yref = f"out/yref/m{tokens}"
        if not os.path.exists(f"{LAB}/{yref}.rank0.pt"):
            yref = None
        pending = list(group)
        attempts = {}
        while pending:
            if args.max_batches is not None and n_batches >= args.max_batches:
                print("[done] max-batches reached", flush=True)
                return
            chunk = pending[: args.batch_size]
            for c in chunk:
                attempts[c["config_id"]] = attempts.get(c["config_id"], 0) + 1
            wait_for_gpus()
            commit = kernel_commit()  # re-read: kernel agent may have committed
            tag = f"{int(time.time())}_{chunk[0]['config_id']}"
            print(
                f"[batch] {tag}: {len(chunk)} configs, tokens={tokens}, "
                f"paced={paced}, commit={commit}",
                flush=True,
            )
            rows, hang_cid = run_batch(
                chunk,
                tokens,
                yref,
                tag,
                args.dsl_file_cache,
                args.repeat_iters,
                args.compile_only,
            )
            csv_rows = [jsonl_to_csv_row(r, commit) for r in rows]
            got = {r["config_id"] for r in rows}
            if hang_cid is not None and hang_cid not in ("(bootstrap)", "(end)"):
                hung = [c for c in chunk if c["config_id"] == hang_cid]
                for c in hung:
                    csv_rows.append(
                        jsonl_to_csv_row(
                            dict(
                                config_id=c["config_id"],
                                stage=c["stage"],
                                tokens=tokens,
                                knobs=c["knobs"],
                                closure=c.get("closure", "view"),
                                paced=paced,
                                status="HANG",
                                error="watchdog: phase budget exceeded",
                            ),
                            commit,
                        )
                    )
                    got.add(c["config_id"])
            # Requeue unprocessed configs (session died mid-batch), with a
            # 2-attempt cap so a session-killer config can't loop forever.
            missing = [c for c in chunk if c["config_id"] not in got]
            aborted = [c for c in missing if attempts[c["config_id"]] >= 2]
            requeue = [c for c in missing if attempts[c["config_id"]] < 2]
            for c in aborted:
                csv_rows.append(
                    jsonl_to_csv_row(
                        dict(
                            config_id=c["config_id"],
                            stage=c["stage"],
                            tokens=tokens,
                            knobs=c["knobs"],
                            closure=c.get("closure", "view"),
                            paced=paced,
                            status="ABORTED",
                            error="2 sessions died before this config ran",
                        ),
                        commit,
                    )
                )
            if csv_rows:
                append_rows(args.csv, csv_rows)
            print(
                f"[batch] {tag}: ledgered {len(csv_rows)} rows "
                f"(hang={hang_cid}, requeued={len(requeue)})",
                flush=True,
            )
            n_batches += 1
            processed = got | {c["config_id"] for c in aborted}
            pending = [c for c in pending if c["config_id"] not in processed]
    print("[done] sweep pass complete", flush=True)


if __name__ == "__main__":
    main()
