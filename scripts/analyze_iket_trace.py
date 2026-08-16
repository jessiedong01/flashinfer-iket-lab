"""Summarize run-iket JSON traces: per-launch phase breakdown per range name.

Usage:
    python scripts/analyze_iket_trace.py out/rmsnorm/iket_pid_*.trace.json
    python scripts/analyze_iket_trace.py --launch -1 out/comm/iket_pid_*.trace.json

With several trace files (one per rank in multi-GPU runs), each file is reported
separately, so per-rank skew is visible side by side. Timestamps are trace-local;
durations within one trace are comparable, absolute values across traces are not.
"""

import argparse
import json
import statistics
from pathlib import Path


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def all_launches(doc):
    for launch in doc.get("launches", []):
        yield "eager", launch
    for graph_key, entries in doc.get("graphLaunches", {}).items():
        for launch in entries:
            yield graph_key, launch


def summarize_launch(doc, launch, source):
    st = doc["stringTable"]
    per_name = {}
    for r in launch.get("ranges", []):
        per_name.setdefault(st[r["rangeNameIdx"]], []).append(r["endTs"] - r["startTs"])

    wl = launch.get("warpLifetimes", [])
    wall = 0
    if wl:
        wall = max(w["endTs"] for w in wl) - min(w["startTs"] for w in wl)

    print(
        f"\n[{source}] kernel={launch['kernelName'][:70]}\n"
        f"  grid=({launch['gridDimX']},{launch['gridDimY']},{launch['gridDimZ']})"
        f" block=({launch['blockDimX']},{launch['blockDimY']},{launch['blockDimZ']})"
        f" warps={len(wl)} markers={len(launch.get('markers', []))}"
        f" wall(first-warp-start -> last-warp-end)={wall / 1e3:.1f} us"
    )
    header = f"  {'range':<16} {'count':>8} {'total_us':>10} {'mean_ns':>10} {'p50_ns':>10} {'p99_ns':>10} {'max_ns':>10}"
    print(header)
    for name, durs in sorted(per_name.items(), key=lambda kv: -sum(kv[1])):
        durs.sort()
        print(
            f"  {name:<16} {len(durs):>8} {sum(durs) / 1e3:>10.1f} "
            f"{statistics.fmean(durs):>10.1f} {pct(durs, 0.5):>10} {pct(durs, 0.99):>10} {durs[-1]:>10}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="iket_pid_*.trace.json files")
    ap.add_argument(
        "--launch",
        type=int,
        default=None,
        help="only this launch index per file (e.g. -1 for the last, steady-state one)",
    )
    args = ap.parse_args()

    for path in args.traces:
        doc = json.loads(Path(path).read_text())
        launches = list(all_launches(doc))
        print(f"\n=== {path} ({len(launches)} instrumented launches) ===")
        if not launches:
            continue
        selected = [launches[args.launch]] if args.launch is not None else launches
        for source, launch in selected:
            summarize_launch(doc, launch, source)


if __name__ == "__main__":
    main()
