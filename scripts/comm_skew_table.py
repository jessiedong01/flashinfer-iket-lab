"""Print a per-rank table of kernel wall time and mean wait/work phases for the
GEMM + two-shot AllReduce IKET traces (one trace file per rank).

Usage:
    python scripts/comm_skew_table.py out/comm/iket_pid_*.trace.json
"""

import json
import sys


def main() -> None:
    rows = []
    for path in sorted(sys.argv[1:]):
        doc = json.load(open(path))
        st = doc["stringTable"]
        launches = doc.get("launches", [])
        pid = path.split("_")[-1].split(".")[0]
        if not launches:
            rows.append((pid, None))
            continue
        launch = launches[-1]  # steady-state launch
        wl = launch["warpLifetimes"]
        wall = (max(w["endTs"] for w in wl) - min(w["startTs"] for w in wl)) / 1e3
        agg = {}
        for r in launch["ranges"]:
            agg.setdefault(st[r["rangeNameIdx"]], []).append(r["endTs"] - r["startTs"])

        def mean_us(name):
            vals = agg.get(name, [0])
            return sum(vals) / len(vals) / 1e3

        rows.append(
            (pid, (wall, mean_us("epi_ar_arrive"), mean_us("ar_bar_sync"),
                   mean_us("ar_ldst"), mean_us("ar_flag_wait"), mean_us("ar_final_bar")))
        )

    print(f"{'pid':8s} {'wall_us':>9} {'epi_arrive':>10} {'bar_sync':>9} {'ldst':>8} "
          f"{'flag_wait':>9} {'final_bar':>9}  (mean us, last launch)")
    for pid, vals in rows:
        if vals is None:
            print(f"{pid:8s} (no instrumented launches - torchrun parent process)")
            continue
        print(f"{pid:8s} {vals[0]:9.1f} {vals[1]:10.1f} {vals[2]:9.1f} {vals[3]:8.1f} "
              f"{vals[4]:9.2f} {vals[5]:9.1f}")


if __name__ == "__main__":
    main()
