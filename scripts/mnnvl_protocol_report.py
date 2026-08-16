"""Aggregate the LL/BT/HT Latin-square IKET traces into a protocol comparison.

Usage:
    python scripts/mnnvl_protocol_report.py <run_log> <trace.json> [<trace.json> ...]

The run log provides the MANIFEST lines (one per allreduce_fusion execution, in
execution order, identical on every rank because the driver barriers between
executions). Each rank's trace lists instrumented launches in execution order:
LL executions produce 2 launches (publish + lamport), BT 3 (publish + owner
reduce + rmsnorm materialize), HT 1. Launch-to-execution mapping is validated
against the marker/range names in each launch.

Reported per rank and per (m, protocol):
  span_us   = last warp end - first warp start across the execution's launches
  reps      = Latin-square executions only (warmups excluded)
Plus a position table to verify the Latin square removed order effects, and
LL/BT spin-phase means.
"""

import json
import re
import statistics
import sys

LAUNCHES_PER_PROTO = {"ll": 2, "bt": 3, "ht": 1}
PROTO_TAGS = {"ll": ("ll_", ), "bt": ("bt_", ), "ht": ("ht_", )}


def parse_manifest(log_path):
    execs = []
    pat = re.compile(
        r"MANIFEST exec=(\d+) m=(\d+) phase=(\w+)(?: order=(\d+))? pos=([-\d]+) proto=(\w+)"
    )
    for line in open(log_path):
        match = pat.search(line)
        if match:
            execs.append(
                dict(
                    exec=int(match[1]),
                    m=int(match[2]),
                    phase=match[3],
                    order=None if match[4] is None else int(match[4]),
                    pos=None if match[5] == "-" else int(match[5]),
                    proto=match[6],
                )
            )
    # the log contains the dry-run pass and the real pass; keep the second half
    if len(execs) % 2 == 0 and len(execs) > 1:
        half = len(execs) // 2
        if [e["exec"] for e in execs[:half]] == [e["exec"] for e in execs[half:]]:
            execs = execs[half:]
    return execs


def launch_names(doc, launch):
    st = doc["stringTable"]
    names = {st[r["rangeNameIdx"]] for r in launch.get("ranges", [])}
    names |= {st[m["markerNameIdx"]] for m in launch.get("markers", [])}
    return names


def launch_proto(names):
    for proto, tags in PROTO_TAGS.items():
        if any(n.startswith(tags) for n in names):
            return proto
    return None


def wall(launch):
    wl = launch["warpLifetimes"]
    return min(w["startTs"] for w in wl), max(w["endTs"] for w in wl)


def main() -> None:
    log_path, trace_paths = sys.argv[1], sys.argv[2:]
    manifest = parse_manifest(log_path)

    per_key = {}       # (m, proto) -> [span_ns per rank-execution]
    per_pos = {}       # (m, proto, pos) -> [span_ns]
    ll_phases = {}     # (m, phase_name) -> [dur_ns]
    bt_spin = {}       # m -> [dur_ns]

    for path in trace_paths:
        doc = json.loads(open(path).read())
        launches = doc.get("launches", [])
        if not launches:
            print(f"{path}: no instrumented launches (torchrun parent), skipping")
            continue
        st = doc["stringTable"]
        cursor = 0
        for entry in manifest:
            n = LAUNCHES_PER_PROTO[entry["proto"]]
            group = launches[cursor:cursor + n]
            cursor += n
            protos = {launch_proto(launch_names(doc, launch)) for launch in group}
            assert protos == {entry["proto"]}, (
                f"{path}: launch/manifest mismatch at exec={entry['exec']}: "
                f"{protos} != {entry['proto']}"
            )
            starts_ends = [wall(launch) for launch in group]
            span = max(e for _, e in starts_ends) - min(s for s, _ in starts_ends)
            if entry["phase"] == "latin":
                per_key.setdefault((entry["m"], entry["proto"]), []).append(span)
                per_pos.setdefault((entry["m"], entry["proto"], entry["pos"]), []).append(span)
                for launch in group:
                    for r in launch.get("ranges", []):
                        nm = st[r["rangeNameIdx"]]
                        dur = r["endTs"] - r["startTs"]
                        if nm.startswith("ll_"):
                            ll_phases.setdefault((entry["m"], nm), []).append(dur)
                        elif nm == "bt_spin_reduce":
                            bt_spin.setdefault(entry["m"], []).append(dur)
        assert cursor == len(launches), f"{path}: {len(launches) - cursor} unmatched launches"

    print("== span per (m, protocol), latin executions pooled across ranks ==")
    print(f"{'m':>6} {'proto':>6} {'reps':>5} {'mean_us':>9} {'min_us':>8} {'max_us':>8}")
    for (m, proto), vals in sorted(per_key.items()):
        print(f"{m:>6} {proto:>6} {len(vals):>5} {statistics.fmean(vals)/1e3:>9.1f} "
              f"{min(vals)/1e3:>8.1f} {max(vals)/1e3:>8.1f}")

    print("\n== span by Latin-square position (order-effect check) ==")
    print(f"{'m':>6} {'proto':>6} " + " ".join(f"pos{p}_us" for p in range(3)))
    seen = sorted({(m, proto) for (m, proto, _) in per_pos})
    for m, proto in seen:
        cells = []
        for p in range(3):
            vals = per_pos.get((m, proto, p))
            cells.append(f"{statistics.fmean(vals)/1e3:7.1f}" if vals else "      -")
        print(f"{m:>6} {proto:>6} " + " ".join(cells))

    print("\n== LL lamport kernel phase means (ns per warp) ==")
    names = sorted({nm for (_, nm) in ll_phases})
    ms = sorted({m for (m, _) in ll_phases})
    print(f"{'m':>6} " + " ".join(f"{nm[3:]:>14}" for nm in names))
    for m in ms:
        row = [f"{statistics.fmean(ll_phases[(m, nm)]):>14.0f}" if (m, nm) in ll_phases else f"{'-':>14}" for nm in names]
        print(f"{m:>6} " + " ".join(row))

    if bt_spin:
        print("\n== BT owner-reduce spin (bt_spin_reduce) mean ns per warp ==")
        for m in sorted(bt_spin):
            print(f"  m={m:<6} mean={statistics.fmean(bt_spin[m]):>10.0f} "
                  f"p_max={max(bt_spin[m]):>10}")


if __name__ == "__main__":
    main()
